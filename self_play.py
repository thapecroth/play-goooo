
import torch
import torch.optim as optim
import numpy as np
from collections import deque
import random
import os
import pickle
import argparse
import time
from datetime import datetime
from tqdm import tqdm
from optimized_go import OptimizedGoGame, BLACK, WHITE
from alpha_go import PolicyValueNet, AlphaGoPlayer

class SelfPlayTrainer:
    def __init__(self, board_size=9, num_blocks=5, learning_rate=1e-3, buffer_size=10000):
        self.board_size = board_size
        self.policy_value_net = PolicyValueNet(board_size, num_blocks=num_blocks)
        self.optimizer = optim.Adam(self.policy_value_net.parameters(), lr=learning_rate)
        self.replay_buffer = deque(maxlen=buffer_size)
        self.best_model = PolicyValueNet(board_size, num_blocks=num_blocks)
        self.best_model.load_state_dict(self.policy_value_net.state_dict())
        
        # Add device management - prefer MPS on Mac
        if torch.backends.mps.is_available():
            self.device = torch.device('mps')
        elif torch.cuda.is_available():
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')
        self.policy_value_net.to(self.device)
        self.best_model.to(self.device)
        print(f"Using device: {self.device}")

    def collect_game_data(self, num_games=1):
        games_data = []
        
        # Create progress bar with better tracking
        games_pbar = tqdm(range(num_games), 
                         desc="Self-play games", 
                         position=0,
                         leave=True,
                         bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]')
        
        for game_idx in games_pbar:
            game = OptimizedGoGame(self.board_size)
            # Use fewer simulations for faster self-play
            player = AlphaGoPlayer(self.policy_value_net, 
                                 simulations=50,  # Reduced from 100
                                 is_self_play=True,
                                 device=self.device)
            game_data = []
            move_count = 0
            max_moves = self.board_size * self.board_size * 2
            
            while not game.game_over and move_count < max_moves:
                # Update progress description
                games_pbar.set_description(f"Game {game_idx+1}/{num_games}, Move {move_count+1}")
                
                # Store state before move
                state_tensor = player._game_to_tensor(game)
                
                # Get move with error handling
                try:
                    move = player.get_move(game, 'black' if game.current_player == BLACK else 'white')
                except Exception as e:
                    print(f"\nError in move generation: {e}")
                    move = None
                
                # Create policy target from MCTS statistics
                policy_target = torch.zeros(self.board_size * self.board_size + 1)
                
                if hasattr(player, 'last_root') and player.last_root and player.last_root.children:
                    # Collect visit counts from root's children
                    total_visits = sum(child.visits for child in player.last_root.children)
                    
                    if total_visits > 0:
                        for child in player.last_root.children:
                            if child.move is None:
                                action = self.board_size * self.board_size
                            else:
                                action = child.move[1] * self.board_size + child.move[0]
                            policy_target[action] = child.visits / total_visits

                game_data.append([state_tensor, policy_target, 0]) # Placeholder for value

                if move is None:
                    game.pass_turn('black' if game.current_player == BLACK else 'white')
                else:
                    success = game.make_move(move[0], move[1], 'black' if game.current_player == BLACK else 'white')
                    if not success:
                        game.pass_turn('black' if game.current_player == BLACK else 'white')
                
                move_count += 1

            # Check if game ended naturally or was terminated
            if move_count >= max_moves:
                print(f"\nGame {game_idx+1} terminated after {max_moves} moves")
                game.game_over = True
                game._calculate_winner()
            
            # Assign final game result to all states
            if game.winner == 'black':
                winner_val = 1.0
            elif game.winner == 'white':
                winner_val = -1.0
            else:
                winner_val = 0.0

            for i in range(len(game_data)):
                game_data[i][2] = winner_val if i % 2 == 0 else -winner_val
            
            games_data.extend(game_data)
            games_pbar.set_description(f"Game {game_idx+1} finished ({move_count} moves)")
        
        # Add to replay buffer
        self.replay_buffer.extend(games_data)
        print(f"\nCollected {len(games_data)} positions from {num_games} games")
        return len(games_data)

    def train_step(self, batch_size=64):
        if len(self.replay_buffer) < batch_size:
            return None

        batch = random.sample(self.replay_buffer, batch_size)
        states, policy_targets, value_targets = zip(*batch)

        states = torch.cat(states)
        policy_targets = torch.stack(policy_targets).to(self.device)
        value_targets = torch.tensor(value_targets).float().unsqueeze(1).to(self.device)

        self.optimizer.zero_grad()
        
        log_policies, values = self.policy_value_net(states)
        
        # Loss calculation
        policy_loss = -torch.mean(torch.sum(policy_targets * log_policies, 1))
        value_loss = torch.nn.functional.mse_loss(values, value_targets)
        loss = policy_loss + value_loss
        
        loss.backward()
        self.optimizer.step()
        
        return loss.item()

    def evaluate_model(self, num_games=10):
        current_player = AlphaGoPlayer(self.policy_value_net, simulations=50, device=self.device)
        best_player = AlphaGoPlayer(self.best_model, simulations=50, device=self.device)
        wins = 0

        for i in tqdm(range(num_games), desc="Evaluating model", leave=False):
            game = OptimizedGoGame(self.board_size)
            p1 = current_player if i % 2 == 0 else best_player
            p2 = best_player if i % 2 == 0 else current_player
            
            while not game.game_over:
                player_to_move = p1 if game.current_player == BLACK else p2
                move = player_to_move.get_move(game, 'black' if game.current_player == BLACK else 'white')
                
                if move is None:
                    game.pass_turn('black' if game.current_player == BLACK else 'white')
                else:
                    success = game.make_move(move[0], move[1], 'black' if game.current_player == BLACK else 'white')
                    if not success:
                        game.pass_turn('black' if game.current_player == BLACK else 'white')

            if (game.winner == 'black' and i % 2 == 0) or (game.winner == 'white' and i % 2 != 0):
                wins += 1
        
        return wins / num_games

    def run(self, num_iterations=100, num_games_per_iter=25, batch_size=64, eval_games=10, win_ratio_to_update=0.55, model_name='best_alpha_go_model.pth'):
        import time
        from datetime import datetime
        
        print(f"\n{'='*60}")
        print(f"Starting Self-Play Training")
        print(f"Board size: {self.board_size}x{self.board_size}")
        print(f"Device: {self.device}")
        print(f"{'='*60}\n")
        
        for i in range(num_iterations):
            iter_start_time = time.time()
            print(f"\n{'='*50}")
            print(f"Iteration {i+1}/{num_iterations}")
            print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"{'='*50}")
            
            # Collect self-play data
            print("\n1. Collecting self-play data...")
            positions_collected = self.collect_game_data(num_games_per_iter)
            print(f"   Buffer size: {len(self.replay_buffer)}")
            
            # Training
            print("\n2. Training network...")
            losses = []
            train_pbar = tqdm(range(num_games_per_iter * 10), desc="Training steps", leave=False)
            
            for step in train_pbar:
                loss = self.train_step(batch_size)
                if loss:
                    losses.append(loss)
                    if len(losses) % 10 == 0:
                        avg_loss = sum(losses[-10:]) / 10
                        train_pbar.set_description(f"Training (loss: {avg_loss:.4f})")
            
            if losses:
                avg_loss = sum(losses) / len(losses)
                print(f"   Average loss: {avg_loss:.4f}")

            # Evaluation
            print("\n3. Evaluating model...")
            win_ratio = self.evaluate_model(eval_games)
            print(f"   Win ratio against best model: {win_ratio:.2%}")

            if win_ratio > win_ratio_to_update:
                print(f"\n✓ New best model found! (win ratio: {win_ratio:.2%})")
                self.best_model.load_state_dict(self.policy_value_net.state_dict())
                self.save_model(model_name)
            else:
                print(f"\n✗ Model not improved (threshold: {win_ratio_to_update:.2%})")
            
            # Iteration summary
            iter_time = time.time() - iter_start_time
            print(f"\nIteration completed in {iter_time:.1f}s")

    def save_model(self, filename):
        os.makedirs('models', exist_ok=True)
        filepath = os.path.join('models', filename)
        torch.save(self.best_model.state_dict(), filepath)

    def load_model(self, filename):
        filepath = os.path.join('models', filename)
        if os.path.exists(filepath):
            try:
                state_dict = torch.load(filepath, map_location=self.device)
                self.policy_value_net.load_state_dict(state_dict)
                self.best_model.load_state_dict(state_dict)
                print(f"Model loaded from {filepath}")
                return True
            except Exception as e:
                print(f"Error loading model: {e}")
                return False
        else:
            print(f"Model file not found: {filepath}")
            return False

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train an AlphaGo model through self-play.')
    parser.add_argument('--board-size', type=int, default=9, help='Size of the Go board (e.g., 9, 13, 19).')
    parser.add_argument('--num-blocks', type=int, default=5, help='Number of residual blocks in the policy-value network.')
    parser.add_argument('--learning-rate', type=float, default=1e-3, help='Learning rate for the optimizer.')
    parser.add_argument('--buffer-size', type=int, default=10000, help='Size of the replay buffer.')
    parser.add_argument('--num-iterations', type=int, default=100, help='Number of training iterations.')
    parser.add_argument('--num-games-per-iter', type=int, default=25, help='Number of self-play games to generate per iteration.')
    parser.add_argument('--batch-size', type=int, default=64, help='Batch size for training the neural network.')
    parser.add_argument('--eval-games', type=int, default=10, help='Number of games to play for model evaluation.')
    parser.add_argument('--win-ratio-to-update', type=float, default=0.55, help='Win ratio needed to update the best model.')
    parser.add_argument('--model-name', type=str, default='best_alpha_go_model.pth', help='Filename for saving the best model.')
    
    args = parser.parse_args()

    trainer = SelfPlayTrainer(
        board_size=args.board_size,
        num_blocks=args.num_blocks,
        learning_rate=args.learning_rate,
        buffer_size=args.buffer_size
    )
    
    trainer.load_model(args.model_name)
    trainer.run(
        num_iterations=args.num_iterations,
        num_games_per_iter=args.num_games_per_iter,
        batch_size=args.batch_size,
        eval_games=args.eval_games,
        win_ratio_to_update=args.win_ratio_to_update,
        model_name=args.model_name
    )
