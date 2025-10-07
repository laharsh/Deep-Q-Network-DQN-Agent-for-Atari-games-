#!/usr/bin/env python3
"""
Improved Training Script with Better Hyperparameters
Fixed the issues that were preventing learning in the original quick training.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import os
import time
import json
from datetime import datetime
from tqdm import tqdm

from atari_env import VectorizedAtariEnv
from dqn_agent import OptimizedDQNAgent
import gymnasium as gym



def optimized_train(game_name: str = "ALE/Pong-v5", 
                   episodes: int = 500,
                   num_envs: int = 8,
                   save_dir: str = "./results"):
    """
    Training loop with all optimizations + model saving & visualization:
    1. Vectorized environments (8x parallel collection)
    2. Optimized replay buffer (NumPy arrays)
    3. Pinned memory (faster GPU transfers)
    4. Mixed precision (2x faster training)
    5. Batch inference (process all envs at once)
    6. Best model saving
    7. Training visualization & metrics logging
    """
    
    import os
    import json
    import matplotlib.pyplot as plt
    from datetime import datetime
    
    # Create directories
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = os.path.join(save_dir, f"training_{timestamp}")
    model_dir = os.path.join(log_dir, "models")
    plot_dir = os.path.join(log_dir, "plots")
    
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(plot_dir, exist_ok=True)
    
    print("=" * 60)
    print("OPTIMIZED DQN TRAINING")
    print("=" * 60)
    print(f"Game: {game_name}")
    print(f"Episodes: {episodes}")
    print(f"Parallel Environments: {num_envs}")
    print(f"Save Directory: {log_dir}")
    print("=" * 60)
    
    # Create vectorized environment
    env = VectorizedAtariEnv(game_name, num_envs=num_envs)
    
    # Create agent
    state_shape = (4, 84, 84)
    num_actions = env.action_space.n
    agent = OptimizedDQNAgent(state_shape, num_actions, use_mixed_precision=True)
    
    # Training metrics
    episode_rewards = []
    episode_lengths = []
    average_rewards = []
    losses = []
    epsilon_values = []
    steps_per_second = []
    
    # Best model tracking
    best_reward = float('-inf')
    best_episode = 0
    
    import time
    from tqdm import tqdm
    
    # Training loop
    states = env.reset()
    episode_reward_buffers = [0.0] * num_envs
    episode_length_buffers = [0] * num_envs
    episode_loss_buffers = [[] for _ in range(num_envs)]
    
    start_time = time.time()
    total_steps = 0
    completed_episodes = 0
    
    try:
        with tqdm(total=episodes, desc="Training Progress") as pbar:
            while completed_episodes < episodes:
                episode_start = time.time()
                
                for step in range(1000):  # Max steps per episode
                    # Select actions (batch inference!)
                    actions = agent.select_actions(states, training=True)
                    
                    # Step all environments simultaneously
                    next_states, rewards, dones, infos = env.step(actions)
                    
                    # Store experiences in batch
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
                            
                            # Average loss for this episode
                            avg_loss = np.mean(episode_loss_buffers[i]) if episode_loss_buffers[i] else 0.0
                            losses.append(avg_loss)
                            
                            # Track epsilon
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
                                best_episode = completed_episodes + 1
                                
                                # Save best model
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
                            
                            if completed_episodes >= episodes:
                                break
                    
                    states = next_states
                    total_steps += num_envs
                    
                    if completed_episodes >= episodes:
                        break
                
                # Calculate steps per second
                episode_time = time.time() - episode_start
                sps = (1000 * num_envs) / episode_time if episode_time > 0 else 0
                steps_per_second.append(sps)
                
                # Progress reporting
                if completed_episodes % 50 == 0 and completed_episodes > 0:
                    elapsed_time = time.time() - start_time
                    episodes_per_hour = completed_episodes / (elapsed_time / 3600)
                    avg_reward = average_rewards[-1] if average_rewards else 0
                    avg_sps = np.mean(steps_per_second[-10:]) if steps_per_second else 0
                    
                    print(f"\nEpisode {completed_episodes:4d} | "
                          f"Reward: {episode_rewards[-1]:6.1f} | "
                          f"Avg(100): {avg_reward:6.1f} | "
                          f"Best: {best_reward:6.1f}@{best_episode:4d} | "
                          f"Epsilon: {agent.epsilon:.3f} | "
                          f"SPS: {avg_sps:6.0f} | "
                          f"Speed: {episodes_per_hour:.1f} ep/hr")
                
                # Save checkpoint periodically
                if completed_episodes % 200 == 0 and completed_episodes > 0:
                    checkpoint_path = os.path.join(model_dir, f"checkpoint_ep{completed_episodes}.pth")
                    torch.save({
                        'episode': completed_episodes,
                        'model_state_dict': agent.q_network.state_dict(),
                        'target_state_dict': agent.target_network.state_dict(),
                        'optimizer_state_dict': agent.optimizer.state_dict(),
                        'epsilon': agent.epsilon,
                    }, checkpoint_path)
                    print(f"💾 Checkpoint saved: {checkpoint_path}")
    
    except KeyboardInterrupt:
        print("\n⏹️  Training interrupted by user")
    
    finally:
        env.close()
        
        # Save final model
        final_model_path = os.path.join(model_dir, "final_model.pth")
        torch.save({
            'episode': completed_episodes,
            'model_state_dict': agent.q_network.state_dict(),
            'target_state_dict': agent.target_network.state_dict(),
            'optimizer_state_dict': agent.optimizer.state_dict(),
            'epsilon': agent.epsilon,
        }, final_model_path)
        
        # Calculate final statistics
        total_time = time.time() - start_time
        final_avg_reward = average_rewards[-1] if average_rewards else 0
        improvement = final_avg_reward - episode_rewards[0] if episode_rewards else 0
        avg_sps = np.mean(steps_per_second) if steps_per_second else 0
        
        # Final results
        final_results = {
            'game': game_name,
            'total_episodes': completed_episodes,
            'total_time_seconds': total_time,
            'total_time_minutes': total_time / 60,
            'best_avg_reward': best_reward,
            'best_episode': best_episode,
            'final_avg_reward': final_avg_reward,
            'initial_reward': episode_rewards[0] if episode_rewards else 0,
            'improvement': improvement,
            'avg_steps_per_second': avg_sps,
            'final_epsilon': agent.epsilon,
            'num_parallel_envs': num_envs,
            'timestamp': timestamp
        }
        
        # Save metrics
        metrics_path = os.path.join(log_dir, "training_metrics.json")
        with open(metrics_path, 'w') as f:
            json.dump({
                'episode_rewards': episode_rewards,
                'episode_lengths': episode_lengths,
                'average_rewards': average_rewards,
                'losses': losses,
                'epsilon_values': epsilon_values,
                'steps_per_second': steps_per_second
            }, f, indent=2)
        
        # Save final results
        results_path = os.path.join(log_dir, "final_results.json")
        with open(results_path, 'w') as f:
            json.dump(final_results, f, indent=2)
        
        # Create comprehensive training plot
        create_training_plots(plot_dir, episode_rewards, average_rewards, 
                            losses, epsilon_values, steps_per_second)
        
        # Print final summary
        print("\n" + "=" * 60)
        print("🎉 OPTIMIZED TRAINING COMPLETED!")
        print("=" * 60)
        print("📊 Final Results:")
        print(f"   Total Episodes: {final_results['total_episodes']}")
        print(f"   Total Time: {final_results['total_time_minutes']:.1f} minutes")
        print(f"   Best Average Reward: {final_results['best_avg_reward']:.2f} (Episode {final_results['best_episode']})")
        print(f"   Final Average Reward: {final_results['final_avg_reward']:.2f}")
        print(f"   Improvement: {final_results['improvement']:.2f}")
        print(f"   Average SPS: {final_results['avg_steps_per_second']:.0f}")
        print(f"   Final Epsilon: {final_results['final_epsilon']:.3f}")
        
        # Learning assessment
        if final_results['improvement'] > 5:
            print("\n🎯 EXCELLENT LEARNING! Significant improvement achieved!")
        elif final_results['improvement'] > 1:
            print("\n✅ GOOD LEARNING! Some improvement achieved!")
        elif final_results['improvement'] > 0:
            print("\n⚠️  MINIMAL LEARNING! Very small improvement.")
        else:
            print("\n❌ NO LEARNING! Consider adjusting hyperparameters.")
        
        print(f"\n📁 Results saved to: {log_dir}")
        print(f"📁 Best model: {os.path.join(model_dir, 'best_model.pth')}")
        print(f"📁 Plots: {plot_dir}")
        
        return final_results


def create_training_plots(plot_dir, episode_rewards, average_rewards, 
                         losses, epsilon_values, steps_per_second):
    """Create comprehensive training visualization."""
    import matplotlib.pyplot as plt
    
    # Create figure with subplots
    fig = plt.figure(figsize=(16, 12))
    gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)
    
    # 1. Episode Rewards with Moving Average
    ax1 = fig.add_subplot(gs[0, :])
    ax1.plot(episode_rewards, alpha=0.4, label='Episode Reward', color='lightblue', linewidth=0.5)
    ax1.plot(average_rewards, label='100-Episode Moving Average', 
             linewidth=2.5, color='darkblue')
    ax1.set_xlabel('Episode', fontsize=12)
    ax1.set_ylabel('Reward', fontsize=12)
    ax1.set_title('Training Progress: Episode Rewards', fontsize=14, fontweight='bold')
    ax1.legend(loc='lower right')
    ax1.grid(True, alpha=0.3)
    
    # Add horizontal line at 0
    ax1.axhline(y=0, color='red', linestyle='--', alpha=0.5, linewidth=1)
    
    # 2. Training Loss
    ax2 = fig.add_subplot(gs[1, 0])
    ax2.plot(losses, color='orange', alpha=0.7, linewidth=1)
    ax2.set_xlabel('Episode', fontsize=12)
    ax2.set_ylabel('Loss', fontsize=12)
    ax2.set_title('Training Loss Over Time', fontsize=13, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    # 3. Epsilon Decay
    ax3 = fig.add_subplot(gs[1, 1])
    ax3.plot(epsilon_values, color='purple', alpha=0.7, linewidth=1.5)
    ax3.set_xlabel('Episode', fontsize=12)
    ax3.set_ylabel('Epsilon (Exploration Rate)', fontsize=12)
    ax3.set_title('Exploration Rate Decay', fontsize=13, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    
    # 4. Reward Distribution
    ax4 = fig.add_subplot(gs[2, 0])
    ax4.hist(episode_rewards, bins=50, alpha=0.7, color='green', edgecolor='darkgreen')
    ax4.set_xlabel('Reward', fontsize=12)
    ax4.set_ylabel('Frequency', fontsize=12)
    ax4.set_title('Reward Distribution', fontsize=13, fontweight='bold')
    ax4.grid(True, alpha=0.3, axis='y')
    
    # Add mean line
    mean_reward = np.mean(episode_rewards)
    ax4.axvline(x=mean_reward, color='red', linestyle='--', 
                linewidth=2, label=f'Mean: {mean_reward:.1f}')
    ax4.legend()
    
    # 5. Steps Per Second (Performance)
    ax5 = fig.add_subplot(gs[2, 1])
    ax5.plot(steps_per_second, color='teal', alpha=0.7, linewidth=1)
    ax5.set_xlabel('Episode', fontsize=12)
    ax5.set_ylabel('Steps/Second', fontsize=12)
    ax5.set_title('Training Performance (SPS)', fontsize=13, fontweight='bold')
    ax5.grid(True, alpha=0.3)
    
    # Add mean line
    mean_sps = np.mean(steps_per_second)
    ax5.axhline(y=mean_sps, color='red', linestyle='--', 
                linewidth=2, label=f'Avg: {mean_sps:.0f} SPS')
    ax5.legend()
    
    # Overall title
    fig.suptitle('Optimized DQN Training Results', 
                 fontsize=16, fontweight='bold', y=0.995)
    
    # Save plot
    plot_path = os.path.join(plot_dir, "training_results.png")
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"📊 Training plots saved to: {plot_path}")
    plt.close()
    
    # Create additional detailed reward plot
    fig2, ax = plt.subplots(figsize=(12, 6))
    ax.plot(episode_rewards, alpha=0.3, label='Episode Reward', color='lightblue')
    ax.plot(average_rewards, label='100-Episode MA', linewidth=2, color='darkblue')
    
    # Add trend line
    if len(episode_rewards) > 10:
        z = np.polyfit(range(len(average_rewards)), average_rewards, 1)
        p = np.poly1d(z)
        ax.plot(range(len(average_rewards)), p(range(len(average_rewards))), 
                "r--", alpha=0.8, linewidth=2, label=f'Trend: {z[0]:.2f}x')
    
    ax.set_xlabel('Episode', fontsize=12)
    ax.set_ylabel('Reward', fontsize=12)
    ax.set_title('Detailed Reward Progression', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    reward_detail_path = os.path.join(plot_dir, "reward_detail.png")
    plt.savefig(reward_detail_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"📊 Detailed reward plot saved to: {reward_detail_path}")


# Example usage
if __name__ == "__main__":
    rewards = optimized_train("ALE/Pong-v5", episodes=500, num_envs=8)
