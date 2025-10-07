"""
Resume Training from Checkpoint

This allows you to continue training from where you left off,
preserving all learned knowledge and training state.
"""

import torch
import os
import json

# ============================================================================
# Method 1: Resume Training Function (RECOMMENDED)
# ============================================================================

def resume_training(checkpoint_path: str,
                   additional_episodes: int = 400,
                   game_name: str = "ALE/Pong-v5",
                   num_envs: int = 8,
                   save_dir: str = "./results"):
    """
    Resume training from a saved checkpoint.
    
    Args:
        checkpoint_path: Path to the checkpoint file (e.g., "results/.../models/best_model.pth")
        additional_episodes: How many MORE episodes to train
        game_name: Atari game name
        num_envs: Number of parallel environments
        save_dir: Where to save new results
    
    Example:
        # You trained for 100 episodes, now train 400 more (total 500)
        resume_training(
            checkpoint_path="results/training_20250106/models/best_model.pth",
            additional_episodes=400
        )
    """
    import time
    from datetime import datetime
    from tqdm import tqdm
    import numpy as np
    
    print("=" * 60)
    print("RESUMING TRAINING FROM CHECKPOINT")
    print("=" * 60)
    
    # Load checkpoint
    print(f"📂 Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, weights_only=False)

    
    starting_episode = checkpoint.get('episode', 0)
    loaded_epsilon = checkpoint.get('epsilon', 1.0)
    best_reward_so_far = checkpoint.get('best_reward', float('-inf'))
    
    print(f"✅ Checkpoint loaded successfully!")
    print(f"   Starting from episode: {starting_episode}")
    print(f"   Current epsilon: {loaded_epsilon:.3f}")
    print(f"   Best reward so far: {best_reward_so_far:.2f}")
    print(f"   Training for {additional_episodes} more episodes")
    print(f"   Target total: {starting_episode + additional_episodes} episodes")
    print("=" * 60)
    
    # Create new save directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = os.path.join(save_dir, f"resumed_{timestamp}")
    model_dir = os.path.join(log_dir, "models")
    plot_dir = os.path.join(log_dir, "plots")
    
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(plot_dir, exist_ok=True)
    
    # Create environment
    from atari_env import VectorizedAtariEnv
    from dqn_agent import OptimizedDQNAgent
    
    env = VectorizedAtariEnv(game_name, num_envs=num_envs)
    state_shape = (4, 84, 84)
    num_actions = env.action_space.n
    
    # Create agent
    agent = OptimizedDQNAgent(state_shape, num_actions, use_mixed_precision=True)
    
    # Restore model state
    agent.q_network.load_state_dict(checkpoint['model_state_dict'])
    agent.target_network.load_state_dict(checkpoint['target_state_dict'])
    agent.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    agent.epsilon = loaded_epsilon  # CRITICAL: Continue from same exploration rate
    
    print(f"🧠 Model state restored successfully!")
    
    # Load previous metrics if available
    original_log_dir = os.path.dirname(os.path.dirname(checkpoint_path))
    previous_metrics_path = os.path.join(original_log_dir, "training_metrics.json")
    
    previous_rewards = []
    previous_losses = []
    previous_epsilons = []
    
    if os.path.exists(previous_metrics_path):
        with open(previous_metrics_path, 'r') as f:
            previous_metrics = json.load(f)
            previous_rewards = previous_metrics.get('episode_rewards', [])
            previous_losses = previous_metrics.get('losses', [])
            previous_epsilons = previous_metrics.get('epsilon_values', [])
        print(f"📊 Loaded {len(previous_rewards)} previous episode metrics")
    
    # Training metrics (combining old + new)
    episode_rewards = previous_rewards.copy()
    episode_lengths = []
    average_rewards = []
    losses = previous_losses.copy()
    epsilon_values = previous_epsilons.copy()
    steps_per_second = []
    
    # Best model tracking (start from previous best)
    best_reward = best_reward_so_far
    best_episode = starting_episode
    
    # Training loop
    states = env.reset()
    episode_reward_buffers = [0.0] * num_envs
    episode_length_buffers = [0] * num_envs
    episode_loss_buffers = [[] for _ in range(num_envs)]
    
    start_time = time.time()
    total_steps = 0
    completed_episodes = 0
    
    try:
        with tqdm(total=additional_episodes, desc="Resumed Training") as pbar:
            while completed_episodes < additional_episodes:
                episode_start = time.time()
                
                for step in range(1000):
                    # Select actions
                    actions = agent.select_actions(states, training=True)
                    
                    # Step environments
                    next_states, rewards, dones, infos = env.step(actions)
                    
                    # Store experiences
                    agent.memory.push_batch(states, actions, rewards, next_states, dones)
                    
                    # Train
                    loss = None
                    if len(agent.memory) >= 32:
                        loss = agent.train_step(batch_size=32)
                    
                    # Update buffers
                    for i in range(num_envs):
                        episode_reward_buffers[i] += rewards[i]
                        episode_length_buffers[i] += 1
                        if loss is not None:
                            episode_loss_buffers[i].append(loss)
                        
                        if dones[i]:
                            # Episode completed
                            episode_rewards.append(episode_reward_buffers[i])
                            episode_lengths.append(episode_length_buffers[i])
                            
                            avg_loss = np.mean(episode_loss_buffers[i]) if episode_loss_buffers[i] else 0.0
                            losses.append(avg_loss)
                            epsilon_values.append(agent.epsilon)
                            
                            # Calculate moving average
                            if len(episode_rewards) >= 100:
                                recent_avg = np.mean(episode_rewards[-100:])
                            else:
                                recent_avg = np.mean(episode_rewards)
                            average_rewards.append(recent_avg)
                            
                            # Check if best model
                            if recent_avg > best_reward:
                                best_reward = recent_avg
                                best_episode = starting_episode + completed_episodes + 1
                                
                                # Save new best model
                                best_model_path = os.path.join(model_dir, "best_model.pth")
                                torch.save({
                                    'episode': best_episode,
                                    'model_state_dict': agent.q_network.state_dict(),
                                    'target_state_dict': agent.target_network.state_dict(),
                                    'optimizer_state_dict': agent.optimizer.state_dict(),
                                    'epsilon': agent.epsilon,
                                    'best_reward': best_reward,
                                }, best_model_path)
                            
                            # Reset buffers
                            episode_reward_buffers[i] = 0.0
                            episode_length_buffers[i] = 0
                            episode_loss_buffers[i] = []
                            
                            completed_episodes += 1
                            pbar.update(1)
                            
                            if completed_episodes >= additional_episodes:
                                break
                    
                    states = next_states
                    total_steps += num_envs
                    
                    if completed_episodes >= additional_episodes:
                        break
                
                # Calculate steps per second
                episode_time = time.time() - episode_start
                sps = (1000 * num_envs) / episode_time if episode_time > 0 else 0
                steps_per_second.append(sps)
                
                # Progress reporting
                if completed_episodes % 50 == 0 and completed_episodes > 0:
                    elapsed_time = time.time() - start_time
                    episodes_per_hour = completed_episodes / (elapsed_time / 3600)
                    current_episode = starting_episode + completed_episodes
                    avg_reward = average_rewards[-1] if average_rewards else 0
                    avg_sps = np.mean(steps_per_second[-10:]) if steps_per_second else 0
                    
                    print(f"\nEpisode {current_episode:4d} | "
                          f"Reward: {episode_rewards[-1]:6.1f} | "
                          f"Avg(100): {avg_reward:6.1f} | "
                          f"Best: {best_reward:6.1f}@{best_episode:4d} | "
                          f"Epsilon: {agent.epsilon:.3f} | "
                          f"SPS: {avg_sps:6.0f}")
                
                # Save checkpoint
                if completed_episodes % 200 == 0 and completed_episodes > 0:
                    current_episode = starting_episode + completed_episodes
                    checkpoint_path_new = os.path.join(model_dir, f"checkpoint_ep{current_episode}.pth")
                    torch.save({
                        'episode': current_episode,
                        'model_state_dict': agent.q_network.state_dict(),
                        'target_state_dict': agent.target_network.state_dict(),
                        'optimizer_state_dict': agent.optimizer.state_dict(),
                        'epsilon': agent.epsilon,
                    }, checkpoint_path_new)
                    print(f"💾 Checkpoint saved: {checkpoint_path_new}")
    
    except KeyboardInterrupt:
        print("\n⏹️  Training interrupted by user")
    
    finally:
        env.close()
        
        # Save final model
        final_episode = starting_episode + completed_episodes
        final_model_path = os.path.join(model_dir, "final_model.pth")
        torch.save({
            'episode': final_episode,
            'model_state_dict': agent.q_network.state_dict(),
            'target_state_dict': agent.target_network.state_dict(),
            'optimizer_state_dict': agent.optimizer.state_dict(),
            'epsilon': agent.epsilon,
        }, final_model_path)
        
        # Calculate statistics
        total_time = time.time() - start_time
        final_avg_reward = average_rewards[-1] if average_rewards else 0
        initial_reward = previous_rewards[0] if previous_rewards else episode_rewards[0]
        improvement = final_avg_reward - initial_reward
        avg_sps = np.mean(steps_per_second) if steps_per_second else 0
        
        # Final results
        final_results = {
            'game': game_name,
            'resumed_from_episode': starting_episode,
            'additional_episodes': completed_episodes,
            'total_episodes': final_episode,
            'training_time_seconds': total_time,
            'training_time_minutes': total_time / 60,
            'best_avg_reward': best_reward,
            'best_episode': best_episode,
            'final_avg_reward': final_avg_reward,
            'initial_reward': initial_reward,
            'total_improvement': improvement,
            'avg_steps_per_second': avg_sps,
            'final_epsilon': agent.epsilon,
        }
        
        # Save combined metrics
        metrics_path = os.path.join(log_dir, "combined_training_metrics.json")
        with open(metrics_path, 'w') as f:
            json.dump({
                'episode_rewards': episode_rewards,
                'episode_lengths': episode_lengths,
                'average_rewards': average_rewards,
                'losses': losses,
                'epsilon_values': epsilon_values,
                'steps_per_second': steps_per_second
            }, f, indent=2)
        
        # Save results
        results_path = os.path.join(log_dir, "final_results.json")
        with open(results_path, 'w') as f:
            json.dump(final_results, f, indent=2)
        
        # Create plots
        from train_improved import create_training_plots
        create_training_plots(plot_dir, episode_rewards, average_rewards,
                            losses, epsilon_values, steps_per_second)
        
        # Print summary
        print("\n" + "=" * 60)
        print("🎉 RESUMED TRAINING COMPLETED!")
        print("=" * 60)
        print("📊 Combined Results:")
        print(f"   Started from episode: {starting_episode}")
        print(f"   Additional episodes: {completed_episodes}")
        print(f"   Total episodes: {final_episode}")
        print(f"   Training time: {final_results['training_time_minutes']:.1f} minutes")
        print(f"   Best average reward: {final_results['best_avg_reward']:.2f} (Episode {final_results['best_episode']})")
        print(f"   Final average reward: {final_results['final_avg_reward']:.2f}")
        print(f"   Total improvement: {final_results['total_improvement']:.2f}")
        print(f"   Average SPS: {final_results['avg_steps_per_second']:.0f}")
        print(f"   Final epsilon: {final_results['final_epsilon']:.3f}")
        
        # Learning assessment
        if final_results['total_improvement'] > 20:
            print("\n🎯 EXCELLENT! Massive improvement achieved!")
        elif final_results['total_improvement'] > 10:
            print("\n✅ GREAT! Strong improvement achieved!")
        elif final_results['total_improvement'] > 5:
            print("\n✅ GOOD! Solid improvement achieved!")
        else:
            print("\n⚠️  Modest improvement. Consider training longer.")
        
        print(f"\n📁 Results saved to: {log_dir}")
        print(f"📁 Best model: {os.path.join(model_dir, 'best_model.pth')}")
        
        return final_results


# ============================================================================
# Method 2: Quick Resume Script
# ============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Resume DQN training from checkpoint")
    parser.add_argument("--checkpoint", type=str, required=True,
                       help="Path to checkpoint file (e.g., results/.../models/best_model.pth)")
    parser.add_argument("--episodes", type=int, default=400,
                       help="Additional episodes to train (default: 400)")
    parser.add_argument("--game", type=str, default="ALE/Pong-v5",
                       help="Game name")
    parser.add_argument("--num_envs", type=int, default=8,
                       help="Number of parallel environments")
    parser.add_argument("--save_dir", type=str, default="./results",
                       help="Directory to save results")
    
    args = parser.parse_args()
    
    # Resume training
    results = resume_training(
        checkpoint_path=args.checkpoint,
        additional_episodes=args.episodes,
        game_name=args.game,
        num_envs=args.num_envs,
        save_dir=args.save_dir
    )
    
    print("\n✅ Training resumed successfully!")


# ============================================================================
# Usage Examples
# ============================================================================

"""
EXAMPLE 1: Resume from command line
------------------------------------
python resume_training.py --checkpoint results/training_20250106/models/best_model.pth --episodes 400

EXAMPLE 2: Resume from Python script
-------------------------------------
from resume_training import resume_training

results = resume_training(
    checkpoint_path="results/training_20250106_143022/models/best_model.pth",
    additional_episodes=400  # Train 400 more (100 + 400 = 500 total)
)

EXAMPLE 3: Multiple resume sessions
------------------------------------
# Session 1: Train 100 episodes (testing)
python train_optimized.py --episodes 100

# Session 2: Resume and train 200 more (total 300)
python resume_training.py --checkpoint results/.../best_model.pth --episodes 200

# Session 3: Resume again and train 200 more (total 500)
python resume_training.py --checkpoint results/.../best_model.pth --episodes 200


BENEFITS OF RESUMING:
----------------------
✅ Don't lose learned knowledge
✅ Continue from same exploration rate (epsilon)
✅ Save time (no need to relearn basics)
✅ Can train in multiple sessions
✅ Easy to extend if results aren't good enough
✅ Optimizer momentum preserved
"""