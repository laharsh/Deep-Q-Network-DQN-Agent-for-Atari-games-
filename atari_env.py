"""
Atari Environment Wrapper with Preprocessing
Handles frame preprocessing, action space, and episode management for RL training.
"""

import gymnasium as gym
import numpy as np
import cv2
from collections import deque
import torch

# Register ALE environments
try:
    import ale_py
    gym.register_envs(ale_py)
except ImportError:
    print("Warning: ale_py not installed. Install with: pip install ale_py")
except Exception as e:
    print(f"Warning: Could not register ALE environments: {e}")


class VectorizedAtariEnv:
    """Runs multiple Atari environments in parallel"""
    def __init__(self, game_name: str, num_envs: int = 8, frame_size=(84, 84)):
        self.num_envs = num_envs
        self.frame_size = frame_size
        # Create multiple environments
        self.envs = [gym.make(game_name, render_mode="rgb_array") 
                     for _ in range(num_envs)]
        
        self.action_space = self.envs[0].action_space
        # Preallocate arrays for efficiency
        self.current_frames = np.zeros((num_envs, 4, *frame_size), dtype=np.float32)
        self.frame_buffers = [deque(maxlen=4) for _ in range(num_envs)]
    
    def preprocess_frame(self, frame: np.ndarray) -> np.ndarray:
        """Vectorized frame preprocessing using optimized OpenCV."""
        # Grayscale conversion (optimized)
        if len(frame.shape) == 3:
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        
        # Resize with INTER_AREA (best for downsampling)
        frame = cv2.resize(frame, self.frame_size, interpolation=cv2.INTER_AREA)
        
        # Normalize to [0, 1] - vectorized operation
        return frame.astype(np.float32) / 255.0

    def reset(self) -> np.ndarray:
        """Reset all environments and return initial states."""
        states = []
        for i, env in enumerate(self.envs):
            obs, _ = env.reset()
            frame = self.preprocess_frame(obs)
            
            # Initialize frame stack
            for _ in range(4):
                self.frame_buffers[i].append(frame)
            
            state = np.stack(list(self.frame_buffers[i]), axis=0)
            states.append(state)
        
        return np.array(states, dtype=np.float32)  # Shape: [num_envs, 4, 84, 84]
    
    def step(self, actions: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, list]:

        """ Execute actions in all environments simultaneously. """

        states, rewards, dones, infos = [], [], [], []
        
        for i, (env, action) in enumerate(zip(self.envs, actions)):
            obs, reward, terminated, truncated, info = env.step(int(action))
            done = terminated or truncated
            
            # Preprocess and stack
            frame = self.preprocess_frame(obs)
            self.frame_buffers[i].append(frame)
            state = np.stack(list(self.frame_buffers[i]), axis=0)
            
            states.append(state)
            rewards.append(reward)
            dones.append(done)
            infos.append(info)
            
            # Auto-reset if done
            if done:
                obs, _ = env.reset()
                frame = self.preprocess_frame(obs)
                for _ in range(4):
                    self.frame_buffers[i].append(frame)
        
        return (np.array(states, dtype=np.float32),
                np.array(rewards, dtype=np.float32),
                np.array(dones, dtype=bool),
                infos)
    
    def close(self):
        for env in self.envs:
            env.close()


class AtariEnvironment:
    """
    Complete Atari environment wrapper with preprocessing and utilities.
    """
    
    def __init__(self, 
                 game_name="ALE/Breakout-v5", 
                 frame_size=(84, 84),
                 num_frames=4,
                 skip_frames=4,
                 no_op_max=30):
        """
        Initialize Atari environment.
        
        Args:
            game_name: Name of the Atari game
            frame_size: Size to resize frames to
            num_frames: Number of frames to stack
            skip_frames: Number of frames to skip between actions
            no_op_max: Maximum number of no-op actions at episode start
        """
        self.game_name = game_name
        self.skip_frames = skip_frames
        self.no_op_max = no_op_max
        
        # Create environment
        self.env = gym.make(game_name, render_mode="rgb_array")
        self.action_space = self.env.action_space
        
        # Initialize preprocessor and frame stacker
        self.preprocessor = AtariPreprocessor(frame_size)
        self.frame_stack = FrameStack(num_frames)
        
        # Episode statistics
        self.episode_reward = 0
        self.episode_length = 0
        self.lives = 0
        
    def reset(self):
        """Reset environment and return initial state."""
        obs, info = self.env.reset()
        self.episode_reward = 0
        self.episode_length = 0
        
        # Get initial lives count
        self.lives = info.get('lives', 0)
        
        # Apply random no-op actions
        for _ in range(np.random.randint(0, self.no_op_max)):
            obs, _, terminated, truncated, info = self.env.step(0)  # No-op action
            if terminated or truncated:
                obs, info = self.env.reset()
                self.lives = info.get('lives', 0)
        
        # Preprocess and stack initial frame
        processed_frame = self.preprocessor.preprocess_frame(obs)
        state = self.frame_stack.reset(processed_frame)
        
        return state, info
    
    def step(self, action):  
        """
        Execute action with frame skipping and return processed state.
        """
        total_reward = 0
        
        # Execute action for skip_frames steps
        for _ in range(self.skip_frames):
            obs, reward, terminated, truncated, info = self.env.step(action)
            total_reward += reward
            
            # Check for life loss (important for some games)
            current_lives = info.get('lives', 0)
            life_lost = (current_lives < self.lives) and (current_lives > 0)
            self.lives = current_lives
            
            if terminated or truncated or life_lost:
                break
        
        # Update episode statistics
        self.episode_reward += total_reward
        self.episode_length += 1
        
        # Preprocess frame and update stack
        processed_frame = self.preprocessor.preprocess_frame(obs)
        state = self.frame_stack.update(processed_frame)
        
        # Add episode statistics to info
        info['episode_reward'] = self.episode_reward
        info['episode_length'] = self.episode_length
        info['life_lost'] = life_lost if 'life_lost' in locals() else False
        
        return state, total_reward, terminated, truncated, info
    
    def render(self):
        """Render the environment."""
        return self.env.render()
    
    def close(self):
        """Close the environment."""
        self.env.close()
    
    def get_action_space_size(self):
        """Get the size of the action space."""
        return self.action_space.n
    
    def get_state_shape(self):
        """Get the shape of the preprocessed state."""
        # If frame stack is empty, create a dummy state to get the shape
        if len(self.frame_stack.frames) == 0:
            # Create a dummy frame with the expected preprocessed size
            dummy_frame = np.zeros(self.preprocessor.frame_size, dtype=np.float32)
            dummy_state = np.stack([dummy_frame] * self.frame_stack.num_frames, axis=0)
            return dummy_state.shape
        else:
            return self.frame_stack.get_state().shape


class AtariEnvironmentManager:
    """
    Manager class for handling multiple Atari environments and common operations.
    """
    
    @staticmethod
    def get_available_games():
        """Get list of available Atari games."""
        atari_games = [
            "ALE/Breakout-v5", "ALE/Pong-v5", "ALE/SpaceInvaders-v5", "ALE/Asteroids-v5",
            "ALE/BeamRider-v5", "ALE/Bowling-v5", "ALE/Boxing-v5", "ALE/Centipede-v5",
            "ALE/ChopperCommand-v5", "ALE/CrazyClimber-v5", "ALE/Defender-v5",
            "ALE/DemonAttack-v5", "ALE/DoubleDunk-v5", "ALE/Enduro-v5", "ALE/Fishing-v5",
            "ALE/Freeway-v5", "ALE/Frostbite-v5", "ALE/Gopher-v5", "ALE/Gravitar-v5",
            "ALE/Hero-v5", "ALE/IceHockey-v5", "ALE/Jamesbond-v5", "ALE/Kangaroo-v5",
            "ALE/Krull-v5", "ALE/KungFuMaster-v5", "ALE/MontezumaRevenge-v5",
            "ALE/MsPacman-v5", "ALE/NameThisGame-v5", "ALE/Phoenix-v5", "ALE/Pitfall-v5",
            "ALE/PrivateEye-v5", "ALE/Qbert-v5", "ALE/Riverraid-v5", "ALE/RoadRunner-v5",
            "ALE/Robotank-v5", "ALE/Seaquest-v5", "ALE/Skiing-v5", "ALE/Solaris-v5",
            "ALE/StarGunner-v5", "ALE/Tennis-v5", "ALE/TimePilot-v5", "ALE/Tutankham-v5",
            "ALE/UpNDown-v5", "ALE/Venture-v5", "ALE/VideOlympics-v5", "ALE/WizardOfWor-v5",
            "ALE/YarsRevenge-v5", "ALE/Zaxxon-v5"
        ]
        return atari_games
    
    @staticmethod
    def create_environment(game_name, **kwargs):
        """Create a new Atari environment with given parameters."""
        return AtariEnvironment(game_name, **kwargs)
    
    @staticmethod
    def test_environment(game_name="ALE/Breakout-v5", num_steps=100):
        """
        Test an environment to ensure it works properly.
        Returns basic statistics about the environment.
        """
        env = AtariEnvironment(game_name)
        
        try:
            state, info = env.reset()
            total_reward = 0
            
            for step in range(num_steps):
                action = env.action_space.sample()  # Random action
                state, reward, terminated, truncated, info = env.step(action)
                total_reward += reward
                
                if terminated or truncated:
                    state, info = env.reset()
            
            stats = {
                'game_name': game_name,
                'state_shape': state.shape,
                'action_space_size': env.get_action_space_size(),
                'total_reward': total_reward,
                'test_steps': num_steps,
                'status': 'success'
            }
            
            env.close()
            return stats
            
        except Exception as e:
            env.close()
            return {
                'game_name': game_name,
                'status': 'error',
                'error': str(e)
            }


if __name__ == "__main__":
    # Test the environment
    print("Testing Atari Environment...")
    
    # Test default Breakout environment
    stats = AtariEnvironmentManager.test_environment("ALE/Breakout-v5")
    print(f"Test Results: {stats}")
    
    # List available games
    games = AtariEnvironmentManager.get_available_games()
    print(f"\nAvailable games ({len(games)}): {games[:10]}...")  # Show first 10
