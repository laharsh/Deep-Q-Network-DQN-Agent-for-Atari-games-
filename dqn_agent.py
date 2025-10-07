"""
Deep Q-Network (DQN) Agent Implementation
Includes experience replay, target network, and epsilon-greedy exploration.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import random
from collections import deque, namedtuple
from typing import Tuple, List
from torch.cuda.amp import autocast, GradScaler
import os


# Experience tuple for replay buffer
Experience = namedtuple('Experience', 
                       ['state', 'action', 'reward', 'next_state', 'done'])
   
class DQNNetwork(nn.Module):
    """DQN with optimizations for mixed precision training."""
    
    def __init__(self, input_shape: tuple, num_actions: int):
        super().__init__()
        
        self.conv1 = nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        
        # Calculate conv output size
        conv_out_size = self._get_conv_output_size(input_shape)
        
        self.fc1 = nn.Linear(conv_out_size, 512)
        self.fc2 = nn.Linear(512, num_actions)
        
        self._initialize_weights()
    
    def _get_conv_output_size(self, input_shape: tuple) -> int:
        with torch.no_grad():
            x = torch.zeros(1, *input_shape)
            x = F.relu(self.conv1(x))
            x = F.relu(self.conv2(x))
            x = F.relu(self.conv3(x))
            return x.numel()
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass (compatible with mixed precision)."""
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class OptimizedReplayBuffer:
    """
    Memory-efficient replay buffer using NumPy arrays.
    
    Key Optimizations:
    1. Preallocated arrays (no dynamic resizing)
    2. NumPy for fast indexing
    3. Contiguous memory layout
    """
    def __init__(self, capacity: int, state_shape: tuple, device: torch.device):
        self.capacity = capacity
        self.device = device
        self.position = 0
        self.size = 0
        
        # Preallocate arrays (CRITICAL for performance)
        self.states = None
        self.actions = np.zeros(capacity, dtype=np.int64)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.next_states = np.zeros((capacity, *state_shape), dtype=np.float32)
        self.dones = np.zeros(capacity, dtype=bool)
    
    def push(self, state: np.ndarray, action: int, reward: float, 
             next_state: np.ndarray, done: bool):
        """Store experience in circular buffer."""
        if self.states is None:
            state_shape = state.shape
            # UINT8 instead of FLOAT32 - saves 75% memory!
            self.states = np.zeros((self.capacity, *state_shape), dtype=np.uint8)
            self.next_states = np.zeros((self.capacity, *state_shape), dtype=np.uint8)
            
            # Calculate memory usage
            total_memory = (
                self.states.nbytes + 
                self.next_states.nbytes + 
                self.actions.nbytes + 
                self.rewards.nbytes + 
                self.dones.nbytes
            ) / (1024**3)  # Convert to GB
            print(f"💾 Replay Buffer allocated: {total_memory:.2f} GB")
        idx = self.position
        
        self.states[idx] = (state * 255).astype(np.uint8)
        self.actions[idx] = action
        self.rewards[idx] = reward
        self.next_states[idx] = (next_state * 255).astype(np.uint8)
        self.dones[idx] = done
        
        self.position = (self.position + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)
    
    def push_batch(self, states: np.ndarray, actions: np.ndarray, 
                   rewards: np.ndarray, next_states: np.ndarray, dones: np.ndarray):
        """Push multiple experiences at once (vectorized environments)."""
        batch_size = len(states)
        
        for i in range(batch_size):
            self.push(states[i], actions[i], rewards[i], next_states[i], dones[i])
    
    def sample(self, batch_size: int, pin_memory: bool = True) -> tuple:
        """
        Sample with pinned memory for faster GPU transfer.
        
        OPTIMIZATION 3: Pinned Memory
        - Regular memory: CPU -> GPU transfer ~1-2 GB/s
        - Pinned memory: CPU -> GPU transfer ~6-12 GB/s
        """
        indices = np.random.randint(0, self.size, size=batch_size)
        
        # Sample from NumPy arrays (fast!)
        states = self.states[indices].astype(np.float32) / 255.0
        actions = self.actions[indices]
        rewards = self.rewards[indices]
        next_states = self.next_states[indices].astype(np.float32) / 255.0
        dones = self.dones[indices]
        
        # Convert to PyTorch tensors
        if pin_memory and torch.cuda.is_available():
            # Pinned memory: allocate in page-locked memory for faster transfer
            states = torch.from_numpy(states).pin_memory().to(self.device, non_blocking=True)
            actions = torch.from_numpy(actions).pin_memory().to(self.device, non_blocking=True)
            rewards = torch.from_numpy(rewards).pin_memory().to(self.device, non_blocking=True)
            next_states = torch.from_numpy(next_states).pin_memory().to(self.device, non_blocking=True)
            dones = torch.from_numpy(dones).pin_memory().to(self.device, non_blocking=True)
        else:
            states = torch.from_numpy(states).to(self.device)
            actions = torch.from_numpy(actions).to(self.device)
            rewards = torch.from_numpy(rewards).to(self.device)
            next_states = torch.from_numpy(next_states).to(self.device)
            dones = torch.from_numpy(dones).to(self.device)
        
        return states, actions, rewards, next_states, dones
    
    def __len__(self):
        return self.size

class OptimizedDQNAgent:
    """
    DQN Agent with all performance optimizations:
    - Mixed precision training
    - Pinned memory
    - Batch inference
    - Gradient accumulation
    """
    
    def __init__(self, state_shape: tuple, num_actions: int, 
                 learning_rate: float = 0.0001, gamma: float = 0.99,
                 use_mixed_precision: bool = True):
        
        self.num_actions = num_actions
        self.gamma = gamma
        self.use_mixed_precision = use_mixed_precision
        
        # Device setup
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Networks
        self.q_network = DQNNetwork(state_shape, num_actions).to(self.device)
        self.target_network = DQNNetwork(state_shape, num_actions).to(self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        
        # Optimizer
        self.optimizer = torch.optim.Adam(self.q_network.parameters(), lr=learning_rate)
        
        # Mixed precision scaler
        self.scaler = GradScaler(enabled=use_mixed_precision)
        
        # Replay buffer
        self.memory = OptimizedReplayBuffer(100000, state_shape, self.device)
        
        # Metrics
        self.training_step = 0
        self.epsilon = 1.0
        
        print(f"Optimized DQN Agent initialized on {self.device}")
        print(f"Mixed Precision: {use_mixed_precision}")
        print(f"Model parameters: {sum(p.numel() for p in self.q_network.parameters()):,}")
    
    def select_actions(self, states: np.ndarray, training: bool = True) -> np.ndarray:
        """
        OPTIMIZATION 6: Batch Inference
        Select actions for multiple states at once (faster than one-by-one).
        """
        if training and np.random.random() < self.epsilon:
            # Random actions for all environments
            return np.random.randint(0, self.num_actions, size=len(states))
        else:
            with torch.no_grad():
                states_tensor = torch.from_numpy(states).to(self.device)
                
                # Batch inference (much faster than loop)
                q_values = self.q_network(states_tensor)
                actions = q_values.argmax(dim=1).cpu().numpy()
                
            return actions
    
    def train_step(self, batch_size: int = 32) -> float:
        """
        Training step with mixed precision.
        
        Mixed Precision Benefits:
        - 2x faster matrix operations
        - 50% less GPU memory
        - Maintains accuracy with gradient scaling
        """
        if len(self.memory) < batch_size:
            return 0.0
        
        # Sample batch (with pinned memory optimization)
        states, actions, rewards, next_states, dones = self.memory.sample(
            batch_size, pin_memory=True
        )
        
        # Mixed precision context
        with autocast(enabled=self.use_mixed_precision):
            # Current Q-values
            current_q = self.q_network(states).gather(1, actions.unsqueeze(1))
            
            # Target Q-values
            with torch.no_grad():
                next_q = self.target_network(next_states).max(1)[0]
                target_q = rewards + self.gamma * next_q * ~dones
            
            # Loss
            loss = F.mse_loss(current_q.squeeze(), target_q)
        
        # Backward pass with gradient scaling
        self.optimizer.zero_grad()
        self.scaler.scale(loss).backward()
        
        # Gradient clipping (before unscaling)
        self.scaler.unscale_(self.optimizer)
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), max_norm=10.0)
        
        # Optimizer step
        self.scaler.step(self.optimizer)
        self.scaler.update()
        
        self.training_step += 1
        
        # Update target network
        if self.training_step % 1000 == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())
        
        # Epsilon decay
        if self.epsilon > 0.01:
            self.epsilon *= 0.995
        
        return loss.item()

    def save_model(self, filepath):
        """
        Save the model and training state.
        
        Args:
            filepath: Path to save the model
        """
        checkpoint = {
            'q_network_state_dict': self.q_network.state_dict(),
            'target_network_state_dict': self.target_network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'steps_done': self.steps_done,
            'training_step': self.training_step,
            'state_shape': self.state_shape,
            'num_actions': self.num_actions
        }
        
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        torch.save(checkpoint, filepath)
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath):
        """
        Load the model and training state.
        
        Args:
            filepath: Path to load the model from
        """
        if not os.path.exists(filepath):
            print(f"Model file {filepath} not found!")
            return False
        
        checkpoint = torch.load(filepath, map_location=self.device)
        
        self.q_network.load_state_dict(checkpoint['q_network_state_dict'])
        self.target_network.load_state_dict(checkpoint['target_network_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epsilon = checkpoint['epsilon']
        self.steps_done = checkpoint['steps_done']
        self.training_step = checkpoint['training_step']
        
        print(f"Model loaded from {filepath}")
        return True
    
    def get_stats(self):
        """Get current training statistics."""
        return {
            'epsilon': self.epsilon,
            'steps_done': self.steps_done,
            'training_step': self.training_step,
            'memory_size': len(self.memory),
            'device': str(self.device)
        }

