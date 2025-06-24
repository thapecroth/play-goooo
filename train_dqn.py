import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import random
import os
from collections import deque, namedtuple
import pickle
from server import GoDQN, GoGame
import time
import argparse
from typing import List, Tuple, Dict, Optional
import threading
from concurrent.futures import ThreadPoolExecutor
import multiprocessing

# Named tuple for storing experience
Experience = namedtuple('Experience', ['state', 'action', 'reward', 'next_state', 'done'])

class ReplayBuffer:
    def __init__(self, capacity: int = 50000):
        self.buffer = deque(maxlen=capacity)
        self.capacity = capacity
    
    def push(self, experience: Experience):
        assert isinstance(experience, Experience), f"Expected Experience tuple, got {type(experience)}"
        assert isinstance(experience.state, torch.Tensor), f"State must be tensor, got {type(experience.state)}"
        assert isinstance(experience.action, int), f"Action must be int, got {type(experience.action)}"
        assert isinstance(experience.reward, (int, float)), f"Reward must be numeric, got {type(experience.reward)}"
        assert experience.next_state is None or isinstance(experience.next_state, torch.Tensor), \
            f"Next state must be tensor or None, got {type(experience.next_state)}"
        assert isinstance(experience.done, bool), f"Done must be bool, got {type(experience.done)}"
        
        self.buffer.append(experience)
    
    def sample(self, batch_size: int) -> List[Experience]:
        assert len(self.buffer) >= batch_size, \
            f"Not enough samples in buffer. Have {len(self.buffer)}, need {batch_size}"
        return random.sample(self.buffer, batch_size)
    
    def __len__(self):
        return len(self.buffer)

class GoDQNTrainer:
    def __init__(self, 
                 board_size: int = 9,
                 depth: int = 6,
                 learning_rate: float = 1e-4,
                 gamma: float = 0.99,
                 epsilon_start: float = 1.0,
                 epsilon_end: float = 0.01,
                 epsilon_decay: int = 100000,
                 batch_size: int = 64,  # Increased batch size for better GPU utilization
                 target_update_freq: int = 1000,
                 buffer_capacity: int = 50000,
                 num_workers: int = None,  # For parallel episode generation
                 use_mixed_precision: bool = True):
        
        # Assertions for parameters
        assert board_size > 0, f"Board size must be positive, got {board_size}"
        assert 0 < learning_rate < 1, f"Learning rate must be in (0,1), got {learning_rate}"
        assert 0 <= gamma <= 1, f"Gamma must be in [0,1], got {gamma}"
        assert 0 <= epsilon_start <= 1, f"Epsilon start must be in [0,1], got {epsilon_start}"
        assert 0 <= epsilon_end <= 1, f"Epsilon end must be in [0,1], got {epsilon_end}"
        assert epsilon_decay > 0, f"Epsilon decay must be positive, got {epsilon_decay}"
        assert batch_size > 0, f"Batch size must be positive, got {batch_size}"
        assert target_update_freq > 0, f"Target update frequency must be positive, got {target_update_freq}"
        assert buffer_capacity > batch_size, f"Buffer capacity must be > batch size, got {buffer_capacity}"
        
        self.board_size = board_size
        self.depth = depth
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        self.num_workers = num_workers or min(4, multiprocessing.cpu_count())
        self.use_mixed_precision = use_mixed_precision
        
        # Device setup with MPS support
        if torch.backends.mps.is_available():
            self.device = torch.device('mps')
            print(f"Using MPS device for training")
        elif torch.cuda.is_available():
            self.device = torch.device('cuda')
            print(f"Using CUDA device for training")
            # Enable cuDNN benchmarking for faster training
            torch.backends.cudnn.benchmark = True
        else:
            self.device = torch.device('cpu')
            print(f"Using CPU device for training")
        
        # Initialize networks
        self.q_network = GoDQN(board_size, depth).to(self.device)
        self.target_network = GoDQN(board_size, depth).to(self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()
        
        # Compile networks for faster execution (PyTorch 2.0+)
        if hasattr(torch, 'compile'):
            try:
                self.q_network = torch.compile(self.q_network)
                self.target_network = torch.compile(self.target_network)
                print("✓ Networks compiled for faster execution")
            except Exception as e:
                print(f"Warning: Could not compile networks: {e}")
        
        # Optimizer with optimized settings
        self.optimizer = optim.AdamW(  # AdamW often performs better than Adam
            self.q_network.parameters(), 
            lr=learning_rate,
            weight_decay=1e-4,  # L2 regularization
            eps=1e-8,
            betas=(0.9, 0.999)
        )
        
        # Mixed precision training
        self.scaler = torch.cuda.amp.GradScaler() if use_mixed_precision and torch.cuda.is_available() else None
        
        # Replay buffer
        self.replay_buffer = ReplayBuffer(buffer_capacity)
        
        # Training state
        self.steps_done = 0
        self.episode_rewards = []
        self.episode_lengths = []
        self.losses = []
        
        # Validation
        self._validate_networks()
        
    def _validate_networks(self):
        """Validate network architectures and device placement"""
        # Test forward pass
        dummy_input = torch.randn(1, 3, self.board_size, self.board_size).to(self.device)
        
        with torch.no_grad():
            q_out = self.q_network(dummy_input)
            target_out = self.target_network(dummy_input)
        
        expected_actions = self.board_size * self.board_size + 1
        assert q_out.shape == (1, expected_actions), \
            f"Q-network output shape mismatch: expected (1, {expected_actions}), got {q_out.shape}"
        assert target_out.shape == (1, expected_actions), \
            f"Target network output shape mismatch: expected (1, {expected_actions}), got {target_out.shape}"
        
        # Check device placement
        q_device = next(self.q_network.parameters()).device
        target_device = next(self.target_network.parameters()).device
        assert q_device.type == self.device.type, \
            f"Q-network not on correct device: expected {self.device}, got {q_device}"
        assert target_device.type == self.device.type, \
            f"Target network not on correct device: expected {self.device}, got {target_device}"
        
        print(f"✓ Networks validated: output shape {q_out.shape}, device {self.device}")
    
    def get_epsilon(self) -> float:
        """Get current epsilon value for epsilon-greedy exploration"""
        epsilon = self.epsilon_end + (self.epsilon_start - self.epsilon_end) * \
                 np.exp(-self.steps_done / self.epsilon_decay)
        assert 0 <= epsilon <= 1, f"Epsilon out of bounds: {epsilon}"
        return epsilon
    
    def select_action(self, state: torch.Tensor, valid_actions: List[int], training: bool = True) -> int:
        """Select action using epsilon-greedy policy"""
        assert isinstance(state, torch.Tensor), f"State must be tensor, got {type(state)}"
        assert state.dim() == 4, f"State must be 4D (batch, channels, h, w), got {state.dim()}D"
        assert len(valid_actions) > 0, "Must have at least one valid action"
        assert all(0 <= a < self.board_size * self.board_size + 1 for a in valid_actions), \
            f"Invalid actions: {[a for a in valid_actions if not (0 <= a < self.board_size * self.board_size + 1)]}"
        
        if training and random.random() < self.get_epsilon():
            # Random action
            action = random.choice(valid_actions)
        else:
            # Greedy action
            with torch.no_grad():
                q_values = self.q_network(state.to(self.device))
                assert q_values.dim() == 2, f"Q-values must be 2D, got {q_values.dim()}D"
                assert q_values.size(0) == 1, f"Batch size must be 1, got {q_values.size(0)}"
                
                # Mask invalid actions
                masked_q_values = q_values.clone()
                mask = torch.full((self.board_size * self.board_size + 1,), float('-inf'))
                mask[valid_actions] = 0
                masked_q_values += mask.to(self.device)
                
                action = masked_q_values.argmax().item()
                assert action in valid_actions, f"Selected invalid action {action}, valid: {valid_actions}"
        
        self.steps_done += 1
        return action
    
    def game_to_tensor(self, game: GoGame, player: str) -> torch.Tensor:
        """Convert game state to tensor representation (optimized)"""
        # Pre-allocate tensor on device for better performance
        tensor = torch.zeros(3, game.size, game.size, device=self.device, dtype=torch.float32)
        opponent = 'white' if player == 'black' else 'black'
        
        # Vectorized tensor operations for better performance
        board_array = np.array(game.board)
        
        # Create masks for each channel
        player_mask = (board_array == player)
        opponent_mask = (board_array == opponent)
        empty_mask = (board_array == None)
        
        # Set tensor values using vectorized operations
        tensor[0] = torch.from_numpy(player_mask.astype(np.float32)).to(self.device)
        tensor[1] = torch.from_numpy(opponent_mask.astype(np.float32)).to(self.device)
        tensor[2] = torch.from_numpy(empty_mask.astype(np.float32)).to(self.device)
        
        # Add batch dimension
        return tensor.unsqueeze(0)
    
    def action_to_move(self, action: int) -> Optional[Tuple[int, int]]:
        """Convert action index to board coordinates"""
        assert 0 <= action <= self.board_size * self.board_size, f"Invalid action: {action}"
        
        if action == self.board_size * self.board_size:
            return None  # Pass move
        
        x = action % self.board_size
        y = action // self.board_size
        assert 0 <= x < self.board_size and 0 <= y < self.board_size, \
            f"Invalid coordinates: ({x}, {y})"
        return x, y
    
    def move_to_action(self, move: Optional[Tuple[int, int]]) -> int:
        """Convert board coordinates to action index"""
        if move is None:
            return self.board_size * self.board_size  # Pass action
        
        x, y = move
        assert 0 <= x < self.board_size and 0 <= y < self.board_size, \
            f"Invalid coordinates: ({x}, {y})"
        
        action = y * self.board_size + x
        assert 0 <= action < self.board_size * self.board_size, f"Invalid action: {action}"
        return action
    
    def get_valid_actions(self, game: GoGame, player: str) -> List[int]:
        """Get list of valid action indices"""
        valid_moves = game.get_valid_moves(player)
        actions = [self.move_to_action((move['x'], move['y'])) for move in valid_moves]
        actions.append(self.board_size * self.board_size)  # Always allow pass
        
        assert len(actions) > 0, "Must have at least one valid action"
        assert all(0 <= a <= self.board_size * self.board_size for a in actions), \
            f"Invalid actions: {actions}"
        
        return actions
    
    def calculate_reward(self, game: GoGame, player: str, action_taken: int) -> float:
        """Calculate reward for the given state and action"""
        if game.game_over:
            if game.winner == player:
                return 10.0  # Win reward
            elif game.winner is None:
                return 0.0   # Draw
            else:
                return -10.0 # Loss
        
        # Small negative reward for each move to encourage efficiency
        reward = -0.01
        
        # Bonus for capturing opponent stones
        if hasattr(game, 'last_captures') and game.last_captures:
            reward += len(game.last_captures) * 0.5
        
        return reward
    
    def train_step(self):
        """Perform one training step (optimized)"""
        if len(self.replay_buffer) < self.batch_size:
            return None
        
        # Sample batch
        experiences = self.replay_buffer.sample(self.batch_size)
        
        # Pre-allocate tensors for better memory efficiency
        states_list = []
        next_states_list = []
        actions_list = []
        rewards_list = []
        dones_list = []
        
        for e in experiences:
            states_list.append(e.state)
            actions_list.append(e.action)
            rewards_list.append(e.reward)
            dones_list.append(e.done)
            
            if e.next_state is not None:
                next_states_list.append(e.next_state)
            else:
                next_states_list.append(torch.zeros_like(e.state))
        
        # Create batched tensors efficiently
        states = torch.cat(states_list, dim=0)
        next_states = torch.cat(next_states_list, dim=0)
        actions = torch.tensor(actions_list, dtype=torch.long, device=self.device)
        rewards = torch.tensor(rewards_list, dtype=torch.float32, device=self.device)
        dones = torch.tensor(dones_list, dtype=torch.bool, device=self.device)
        
        # Training with mixed precision if available
        if self.scaler is not None:
            # Mixed precision training
            with torch.cuda.amp.autocast():
                # Current Q values
                current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1))
                
                # Next Q values from target network
                with torch.no_grad():
                    next_q_values = self.target_network(next_states).max(1)[0].detach()
                    target_q_values = rewards + (self.gamma * next_q_values * ~dones)
                
                # Compute loss
                loss = F.mse_loss(current_q_values.squeeze(), target_q_values)
            
            # Optimize with gradient scaling
            self.optimizer.zero_grad()
            self.scaler.scale(loss).backward()
            
            # Gradient clipping (unscaled)
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), max_norm=1.0)
            
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            # Standard precision training
            # Current Q values
            current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1))
            
            # Next Q values from target network
            with torch.no_grad():
                next_q_values = self.target_network(next_states).max(1)[0].detach()
                target_q_values = rewards + (self.gamma * next_q_values * ~dones)
            
            # Compute loss
            loss = F.mse_loss(current_q_values.squeeze(), target_q_values)
            
            # Optimize
            self.optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), max_norm=1.0)
            
            self.optimizer.step()
        
        # Update target network
        if self.steps_done % self.target_update_freq == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())
            print(f"✓ Target network updated at step {self.steps_done}")
        
        self.losses.append(loss.item())
        return loss.item()
    
    def play_episode(self, opponent_strength: str = 'random') -> Tuple[float, int]:
        """Play one episode for training"""
        game = GoGame(self.board_size)
        player = 'black'
        opponent = 'white'
        
        episode_reward = 0
        episode_length = 0
        
        while not game.game_over and episode_length < 400:  # Max moves per episode
            current_player = game.current_player
            
            if current_player == player:
                # AI player's turn
                state = self.game_to_tensor(game, player)
                valid_actions = self.get_valid_actions(game, player)
                action = self.select_action(state, valid_actions, training=True)
                
                # Make move
                move = self.action_to_move(action)
                if move is None:
                    success = game.pass_turn(player)
                else:
                    x, y = move
                    success = game.make_move(x, y, player)
                
                if success:
                    # Calculate reward
                    reward = self.calculate_reward(game, player, action)
                    episode_reward += reward
                    
                    # Get next state
                    if not game.game_over:
                        next_state = self.game_to_tensor(game, player)
                    else:
                        next_state = None
                    
                    # Store experience
                    experience = Experience(
                        state=state,
                        action=action,
                        reward=reward,
                        next_state=next_state,
                        done=game.game_over
                    )
                    self.replay_buffer.push(experience)
                
            else:
                # Opponent's turn
                if opponent_strength == 'random':
                    valid_moves = game.get_valid_moves(opponent)
                    if valid_moves and random.random() > 0.1:  # 10% chance to pass
                        move = random.choice(valid_moves)
                        game.make_move(move['x'], move['y'], opponent)
                    else:
                        game.pass_turn(opponent)
                else:
                    # Could implement other opponent types here
                    game.pass_turn(opponent)
            
            episode_length += 1
        
        return episode_reward, episode_length
    
    def play_episodes_parallel(self, num_episodes: int) -> List[Tuple[float, int]]:
        """Play multiple episodes in parallel for faster data collection"""
        def play_single_episode(seed):
            # Set random seed for reproducibility
            random.seed(seed)
            np.random.seed(seed)
            return self.play_episode()
        
        with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            seeds = [random.randint(0, 2**32-1) for _ in range(num_episodes)]
            results = list(executor.map(play_single_episode, seeds))
        
        return results
    
    def train(self, num_episodes: int = 10000, save_freq: int = 1000, log_freq: int = 100, 
              episodes_per_batch: int = 4):
        """Main training loop"""
        assert num_episodes > 0, f"Number of episodes must be positive, got {num_episodes}"
        assert save_freq > 0, f"Save frequency must be positive, got {save_freq}"
        assert log_freq > 0, f"Log frequency must be positive, got {log_freq}"
        
        print(f"Starting DQN training for {num_episodes} episodes")
        print(f"Device: {self.device}")
        print(f"Board size: {self.board_size}")
        print(f"Network parameters: {sum(p.numel() for p in self.q_network.parameters())}")
        print(f"Parallel workers: {self.num_workers}")
        print(f"Mixed precision: {self.use_mixed_precision}")
        print(f"Episodes per batch: {episodes_per_batch}")
        
        start_time = time.time()
        episode = 0
        
        while episode < num_episodes:
            # Determine batch size for this iteration
            batch_size = min(episodes_per_batch, num_episodes - episode)
            
            # Play episodes in parallel for faster data collection
            if batch_size > 1:
                episode_results = self.play_episodes_parallel(batch_size)
            else:
                episode_results = [self.play_episode()]
            
            # Process all episodes in this batch
            for episode_reward, episode_length in episode_results:
                self.episode_rewards.append(episode_reward)
                self.episode_lengths.append(episode_length)
                episode += 1
                
                # Training step (multiple training steps for better sample efficiency)
                for _ in range(2):  # Train 2 times per episode for better learning
                    loss = self.train_step()
                    if loss is None:
                        break
                
                # Logging
                if episode % log_freq == 0:
                    avg_reward = np.mean(self.episode_rewards[-100:])
                    avg_length = np.mean(self.episode_lengths[-100:])
                    avg_loss = np.mean(self.losses[-100:]) if self.losses else 0
                    epsilon = self.get_epsilon()
                    
                    elapsed_time = time.time() - start_time
                    episodes_per_sec = episode / elapsed_time if elapsed_time > 0 else 0
                    
                    print(f"Episode {episode:6d} | "
                          f"Reward: {episode_reward:7.2f} | "
                          f"Avg Reward: {avg_reward:7.2f} | "
                          f"Length: {episode_length:3d} | "
                          f"Avg Length: {avg_length:5.1f} | "
                          f"Loss: {avg_loss:8.4f} | "
                          f"Epsilon: {epsilon:.3f} | "
                          f"Steps: {self.steps_done:7d} | "
                          f"Eps/sec: {episodes_per_sec:.2f} | "
                          f"Time: {elapsed_time:6.1f}s")
                
                # Save model
                if episode % save_freq == 0 and episode > 0:
                    model_name = f'dqn_go_board{self.board_size}_depth{self.depth}_episode_{episode}.pth'
                    self.save_model(model_name)
                    print(f"✓ Model saved at episode {episode}")
        
        print(f"Training completed in {time.time() - start_time:.1f} seconds")
        
        # Final save
        final_model_name = f'dqn_go_board{self.board_size}_depth{self.depth}_final.pth'
        stats_name = f'training_stats_board{self.board_size}_depth{self.depth}.pkl'
        self.save_model(final_model_name)
        self.save_training_stats(stats_name)
    
    def save_model(self, filepath: str):
        """Save the trained model"""
        # Create models directory if it doesn't exist
        os.makedirs('models', exist_ok=True)
        full_path = os.path.join('models', filepath)
        
        torch.save({
            'q_network_state_dict': self.q_network.state_dict(),
            'target_network_state_dict': self.target_network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'steps_done': self.steps_done,
            'board_size': self.board_size,
            'depth': self.depth,
            'hyperparameters': {
                'learning_rate': self.learning_rate,
                'gamma': self.gamma,
                'epsilon_start': self.epsilon_start,
                'epsilon_end': self.epsilon_end,
                'epsilon_decay': self.epsilon_decay,
                'batch_size': self.batch_size,
                'target_update_freq': self.target_update_freq,
            }
        }, full_path)
    
    def load_model(self, filepath: str):
        """Load a trained model"""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.q_network.load_state_dict(checkpoint['q_network_state_dict'])
        self.target_network.load_state_dict(checkpoint['target_network_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.steps_done = checkpoint['steps_done']
        
        print(f"✓ Model loaded from {filepath}")
    
    def save_training_stats(self, filepath: str):
        """Save training statistics"""
        # Create models directory if it doesn't exist
        os.makedirs('models', exist_ok=True)
        full_path = os.path.join('models', filepath)
        
        stats = {
            'episode_rewards': self.episode_rewards,
            'episode_lengths': self.episode_lengths,
            'losses': self.losses,
            'steps_done': self.steps_done
        }
        with open(full_path, 'wb') as f:
            pickle.dump(stats, f)

def main():
    parser = argparse.ArgumentParser(description='Train DQN for Go game (Optimized)')
    parser.add_argument('--board-size', type=int, default=9, help='Board size (default: 9)')
    parser.add_argument('--depth', type=int, default=6, help='Number of residual blocks (default: 6)')
    parser.add_argument('--episodes', type=int, default=20000, help='Number of training episodes (default: 20000)')
    parser.add_argument('--learning-rate', type=float, default=1e-4, help='Learning rate (default: 1e-4)')
    parser.add_argument('--batch-size', type=int, default=64, help='Batch size (default: 64)')
    parser.add_argument('--save-freq', type=int, default=2000, help='Model save frequency (default: 2000)')
    parser.add_argument('--log-freq', type=int, default=100, help='Logging frequency (default: 100)')
    parser.add_argument('--num-workers', type=int, default=None, help='Number of parallel workers (default: auto)')
    parser.add_argument('--episodes-per-batch', type=int, default=4, help='Episodes per parallel batch (default: 4)')
    parser.add_argument('--no-mixed-precision', action='store_true', help='Disable mixed precision training')
    parser.add_argument('--buffer-capacity', type=int, default=50000, help='Replay buffer capacity (default: 50000)')
    
    args = parser.parse_args()
    
    # Training configuration
    config = {
        'board_size': args.board_size,
        'depth': args.depth,
        'learning_rate': args.learning_rate,
        'gamma': 0.99,
        'epsilon_start': 1.0,
        'epsilon_end': 0.01,
        'epsilon_decay': 50000,
        'batch_size': args.batch_size,
        'target_update_freq': 1000,
        'buffer_capacity': args.buffer_capacity,
        'num_workers': args.num_workers,
        'use_mixed_precision': not args.no_mixed_precision,
        'num_episodes': args.episodes,
        'save_freq': args.save_freq,
        'log_freq': args.log_freq,
        'episodes_per_batch': args.episodes_per_batch
    }
    
    print(f"Initializing DQN trainer with board size {args.board_size} and depth {args.depth}...")
    trainer = GoDQNTrainer(**{k: v for k, v in config.items() if k not in ['num_episodes', 'save_freq', 'log_freq']})
    
    print("Starting optimized training...")
    trainer.train(
        num_episodes=config['num_episodes'],
        save_freq=config['save_freq'],
        log_freq=config['log_freq'],
        episodes_per_batch=config['episodes_per_batch']
    )

if __name__ == "__main__":
    main()