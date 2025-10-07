"""
Evaluation and Visualization Script for DQN Atari Agent
Allows testing trained models and creating gameplay videos.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
import argparse
import json
from datetime import datetime
import time

from atari_env import AtariEnvironment
from dqn_agent import DQNAgent


class GameplayRecorder:
    """Records gameplay videos and statistics."""
    
    def __init__(self, output_dir="recordings"):
        """Initialize recorder."""
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
    def record_gameplay(self, env, agent, num_episodes=5, max_steps=10000):
        """
        Record gameplay videos and collect statistics.
        
        Args:
            env: Atari environment
            agent: Trained DQN agent
            num_episodes: Number of episodes to record
            max_steps: Maximum steps per episode
            
        Returns:
            Dictionary with gameplay statistics
        """
        print(f"Recording {num_episodes} episodes...")
        
        episode_rewards = []
        episode_lengths = []
        all_actions = []
        
        for episode in range(num_episodes):
            print(f"Recording episode {episode + 1}/{num_episodes}")
            
            # Setup video writer
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            video_path = os.path.join(self.output_dir, f"gameplay_ep{episode+1}_{timestamp}.mp4")
            
            # Get first frame to determine video size
            state, _ = env.reset()
            frame = env.render()
            height, width = frame.shape[:2]
            
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            video_writer = cv2.VideoWriter(video_path, fourcc, 30.0, (width, height))
            
            episode_reward = 0
            episode_length = 0
            episode_actions = []
            
            for step in range(max_steps):
                # Select action
                action = agent.select_action(state, training=False)
                episode_actions.append(action)
                
                # Execute action
                next_state, reward, terminated, truncated, info = env.step(action)
                
                # Record frame
                frame = env.render()
                frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                video_writer.write(frame_bgr)
                
                # Update statistics
                episode_reward += reward
                episode_length += 1
                state = next_state
                
                if terminated or truncated:
                    break
            
            # Close video writer
            video_writer.release()
            
            # Store episode statistics
            episode_rewards.append(episode_reward)
            episode_lengths.append(episode_length)
            all_actions.extend(episode_actions)
            
            print(f"Episode {episode + 1}: Reward = {episode_reward:.1f}, Length = {episode_length}")
        
        # Compile statistics
        stats = {
            'num_episodes': num_episodes,
            'episode_rewards': episode_rewards,
            'episode_lengths': episode_lengths,
            'mean_reward': np.mean(episode_rewards),
            'std_reward': np.std(episode_rewards),
            'mean_length': np.mean(episode_lengths),
            'std_length': np.std(episode_lengths),
            'total_steps': sum(episode_lengths),
            'action_distribution': self._analyze_actions(all_actions, env.get_action_space_size())
        }
        
        return stats
    
    def _analyze_actions(self, actions, num_actions):
        """Analyze action distribution."""
        action_counts = np.bincount(actions, minlength=num_actions)
        action_probs = action_counts / len(actions)
        
        return {
            'counts': action_counts.tolist(),
            'probabilities': action_probs.tolist(),
            'entropy': -np.sum(action_probs * np.log(action_probs + 1e-8))
        }


class ModelEvaluator:
    """Comprehensive model evaluation and analysis."""
    
    def __init__(self, model_path, game_name="ALE/Breakout-v5"):
        """Initialize evaluator with model and environment."""
        self.model_path = model_path
        self.game_name = game_name
        
        # Create environment
        self.env = AtariEnvironment(game_name)
        
        # Load agent
        state_shape = self.env.get_state_shape()
        num_actions = self.env.get_action_space_size()
        
        self.agent = DQNAgent(state_shape, num_actions)
        
        if not self.agent.load_model(model_path):
            raise ValueError(f"Failed to load model from {model_path}")
        
        print(f"Loaded model: {model_path}")
        print(f"Environment: {game_name}")
        print(f"State shape: {state_shape}")
        print(f"Action space: {num_actions}")
    
    def evaluate_performance(self, num_episodes=100):
        """
        Evaluate agent performance over multiple episodes.
        
        Args:
            num_episodes: Number of episodes for evaluation
            
        Returns:
            Performance statistics
        """
        print(f"Evaluating performance over {num_episodes} episodes...")
        
        rewards = []
        lengths = []
        lives_data = []
        
        for episode in range(num_episodes):
            if (episode + 1) % 10 == 0:
                print(f"Episode {episode + 1}/{num_episodes}")
            
            state, _ = self.env.reset()
            episode_reward = 0
            episode_length = 0
            episode_lives = []
            
            while True:
                action = self.agent.select_action(state, training=False)
                state, reward, terminated, truncated, info = self.env.step(action)
                
                episode_reward += reward
                episode_length += 1
                
                if 'lives' in info:
                    episode_lives.append(info['lives'])
                
                if terminated or truncated:
                    break
            
            rewards.append(episode_reward)
            lengths.append(episode_length)
            if episode_lives:
                lives_data.append(episode_lives)
        
        # Calculate statistics
        stats = {
            'num_episodes': num_episodes,
            'mean_reward': np.mean(rewards),
            'std_reward': np.std(rewards),
            'min_reward': np.min(rewards),
            'max_reward': np.max(rewards),
            'mean_length': np.mean(lengths),
            'std_length': np.std(lengths),
            'rewards': rewards,
            'lengths': lengths
        }
        
        return stats
    
    def analyze_q_values(self, num_states=1000):
        """
        Analyze Q-value distributions for random states.
        
        Args:
            num_states: Number of states to analyze
            
        Returns:
            Q-value analysis results
        """
        print(f"Analyzing Q-values for {num_states} states...")
        
        q_values_all = []
        max_q_values = []
        action_preferences = []
        
        for _ in range(num_states):
            # Get random state
            state, _ = self.env.reset()
            for _ in range(np.random.randint(1, 100)):
                action = self.env.action_space.sample()
                state, _, terminated, truncated, _ = self.env.step(action)
                if terminated or truncated:
                    state, _ = self.env.reset()
                    break
            
            # Get Q-values
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.agent.device)
                q_values = self.agent.q_network(state_tensor).cpu().numpy()[0]
            
            q_values_all.append(q_values)
            max_q_values.append(np.max(q_values))
            action_preferences.append(np.argmax(q_values))
        
        q_values_array = np.array(q_values_all)
        
        analysis = {
            'mean_q_values': np.mean(q_values_array, axis=0).tolist(),
            'std_q_values': np.std(q_values_array, axis=0).tolist(),
            'mean_max_q': np.mean(max_q_values),
            'std_max_q': np.std(max_q_values),
            'action_distribution': np.bincount(action_preferences, 
                                             minlength=self.env.get_action_space_size()).tolist()
        }
        
        return analysis
    
    def create_evaluation_report(self, output_dir="evaluation_results"):
        """Create comprehensive evaluation report."""
        os.makedirs(output_dir, exist_ok=True)
        
        print("Creating comprehensive evaluation report...")
        
        # Performance evaluation
        perf_stats = self.evaluate_performance(num_episodes=100)
        
        # Q-value analysis
        q_analysis = self.analyze_q_values(num_states=500)
        
        # Create visualizations
        self._plot_performance_analysis(perf_stats, output_dir)
        self._plot_q_value_analysis(q_analysis, output_dir)
        
        # Save detailed results
        results = {
            'model_path': self.model_path,
            'game_name': self.game_name,
            'evaluation_date': datetime.now().isoformat(),
            'performance_stats': perf_stats,
            'q_value_analysis': q_analysis,
            'agent_stats': self.agent.get_stats()
        }
        
        results_path = os.path.join(output_dir, "evaluation_results.json")
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"Evaluation report saved to {output_dir}")
        return results
    
    def _plot_performance_analysis(self, stats, output_dir):
        """Create performance analysis plots."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(f'Performance Analysis - {self.game_name}', fontsize=16)
        
        # Reward distribution
        axes[0, 0].hist(stats['rewards'], bins=30, alpha=0.7, edgecolor='black')
        axes[0, 0].axvline(stats['mean_reward'], color='red', linestyle='--', 
                          label=f'Mean: {stats["mean_reward"]:.1f}')
        axes[0, 0].set_xlabel('Episode Reward')
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].set_title('Reward Distribution')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Reward over episodes
        axes[0, 1].plot(stats['rewards'])
        axes[0, 1].axhline(stats['mean_reward'], color='red', linestyle='--', alpha=0.7)
        axes[0, 1].set_xlabel('Episode')
        axes[0, 1].set_ylabel('Reward')
        axes[0, 1].set_title('Rewards Over Episodes')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Episode length distribution
        axes[1, 0].hist(stats['lengths'], bins=30, alpha=0.7, edgecolor='black')
        axes[1, 0].axvline(stats['mean_length'], color='red', linestyle='--', 
                          label=f'Mean: {stats["mean_length"]:.1f}')
        axes[1, 0].set_xlabel('Episode Length')
        axes[1, 0].set_ylabel('Frequency')
        axes[1, 0].set_title('Episode Length Distribution')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # Scatter plot: reward vs length
        axes[1, 1].scatter(stats['lengths'], stats['rewards'], alpha=0.6)
        axes[1, 1].set_xlabel('Episode Length')
        axes[1, 1].set_ylabel('Episode Reward')
        axes[1, 1].set_title('Reward vs Episode Length')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'performance_analysis.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_q_value_analysis(self, analysis, output_dir):
        """Create Q-value analysis plots."""
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        fig.suptitle('Q-Value Analysis', fontsize=16)
        
        # Q-value distribution by action
        num_actions = len(analysis['mean_q_values'])
        actions = list(range(num_actions))
        
        axes[0].bar(actions, analysis['mean_q_values'], 
                   yerr=analysis['std_q_values'], capsize=5)
        axes[0].set_xlabel('Action')
        axes[0].set_ylabel('Mean Q-Value')
        axes[0].set_title('Mean Q-Values by Action')
        axes[0].grid(True, alpha=0.3)
        
        # Action preference distribution
        axes[1].bar(actions, analysis['action_distribution'])
        axes[1].set_xlabel('Action')
        axes[1].set_ylabel('Selection Frequency')
        axes[1].set_title('Action Selection Frequency')
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'q_value_analysis.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()


def main():
    """Main evaluation function."""
    parser = argparse.ArgumentParser(description='Evaluate trained DQN agent')
    
    parser.add_argument('model_path', type=str, help='Path to trained model')
    parser.add_argument('--game', type=str, default='ALE/Breakout-v5', help='Atari game name')
    parser.add_argument('--episodes', type=int, default=100, help='Number of evaluation episodes')
    parser.add_argument('--record', action='store_true', help='Record gameplay videos')
    parser.add_argument('--output-dir', type=str, default='evaluation_results', help='Output directory')
    
    args = parser.parse_args()
    
    # Create evaluator
    evaluator = ModelEvaluator(args.model_path, args.game)
    
    # Run evaluation
    results = evaluator.create_evaluation_report(args.output_dir)
    
    # Record gameplay if requested
    if args.record:
        recorder = GameplayRecorder(os.path.join(args.output_dir, "recordings"))
        gameplay_stats = recorder.record_gameplay(evaluator.env, evaluator.agent, num_episodes=5)
        
        # Save gameplay statistics
        with open(os.path.join(args.output_dir, "gameplay_stats.json"), 'w') as f:
            json.dump(gameplay_stats, f, indent=2)
    
    # Print summary
    print("\n" + "="*60)
    print("EVALUATION SUMMARY")
    print("="*60)
    print(f"Model: {args.model_path}")
    print(f"Game: {args.game}")
    print(f"Episodes: {args.episodes}")
    print(f"Mean Reward: {results['performance_stats']['mean_reward']:.2f} ± {results['performance_stats']['std_reward']:.2f}")
    print(f"Mean Episode Length: {results['performance_stats']['mean_length']:.1f}")
    print(f"Best Episode Reward: {results['performance_stats']['max_reward']:.1f}")
    print(f"Results saved to: {args.output_dir}")
    
    # Close environment
    evaluator.env.close()


if __name__ == "__main__":
    main()
