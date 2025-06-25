
import torch
import torch.optim as optim
import numpy as np
from collections import deque
import random
import os
import pickle
import argparse
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

    def collect_game_data(self, num_games=1):
        for _ in range(num_games):
            game = OptimizedGoGame(self.board_size)
            player = AlphaGoPlayer(self.policy_value_net, simulations=100, is_self_play=True)
            game_data = []
            
            while not game.game_over:
                move = player.get_move(game, 'black' if game.current_player == BLACK else 'white')
                
                # Store state, policy, and eventual value
                state_tensor = player._game_to_tensor(game)
                
                # Create policy target
                policy_target = torch.zeros(self.board_size * self.board_size + 1)
                children_visits = {child.move: child.visits for child in player.stats.visits.keys()}
                total_visits = sum(children_visits.values())
                
                for move, visits in children_visits.items():
                    if move is None:
                        action = self.board_size * self.board_size
                    else:
                        action = move[1] * self.board_size + move[0]
                    policy_target[action] = visits / total_visits

                game_data.append([state_tensor, policy_target, 0]) # Placeholder for value

                if move is None:
                    game.pass_turn('black' if game.current_player == BLACK else 'white')
                else:
                    game.make_move(move[0], move[1], 'black' if game.current_player == BLACK else 'white')

            # Assign final game result to all states
            if game.winner == 'black':
                winner_val = 1.0
            elif game.winner == 'white':
                winner_val = -1.0
            else:
                winner_val = 0.0

            for i in range(len(game_data)):
                game_data[i][2] = winner_val if i % 2 == 0 else -winner_val
            
            self.replay_buffer.extend(game_data)

    def train_step(self, batch_size=64):
        if len(self.replay_buffer) < batch_size:
            return None

        batch = random.sample(self.replay_buffer, batch_size)
        states, policy_targets, value_targets = zip(*batch)

        states = torch.cat(states)
        policy_targets = torch.stack(policy_targets)
        value_targets = torch.tensor(value_targets).float().unsqueeze(1)

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
        current_player = AlphaGoPlayer(self.policy_value_net, simulations=100)
        best_player = AlphaGoPlayer(self.best_model, simulations=100)
        wins = 0

        for i in range(num_games):
            game = OptimizedGoGame(self.board_size)
            p1 = current_player if i % 2 == 0 else best_player
            p2 = best_player if i % 2 == 0 else current_player
            
            while not game.game_over:
                player_to_move = p1 if game.current_player == BLACK else p2
                move = player_to_move.get_move(game, 'black' if game.current_player == BLACK else 'white')
                
                if move is None:
                    game.pass_turn('black' if game.current_player == BLACK else 'white')
                else:
                    game.make_move(move[0], move[1], 'black' if game.current_player == BLACK else 'white')

            if (game.winner == 'black' and i % 2 == 0) or (game.winner == 'white' and i % 2 != 0):
                wins += 1
        
        return wins / num_games

    def run(self, num_iterations=100, num_games_per_iter=25, batch_size=64, eval_games=10, win_ratio_to_update=0.55, model_name='best_alpha_go_model.pth'):
        for i in range(num_iterations):
            print(f"Iteration {i+1}/{num_iterations}")
            self.collect_game_data(num_games_per_iter)
            
            for _ in range(num_games_per_iter):
                loss = self.train_step(batch_size)
                if loss:
                    print(f"Loss: {loss:.4f}")

            win_ratio = self.evaluate_model(eval_games)
            print(f"Win ratio against best model: {win_ratio:.2f}")

            if win_ratio > win_ratio_to_update:
                print("New best model found!")
                self.best_model.load_state_dict(self.policy_value_net.state_dict())
                self.save_model(model_name)

    def save_model(self, filename):
        os.makedirs('models', exist_ok=True)
        filepath = os.path.join('models', filename)
        torch.save(self.best_model.state_dict(), filepath)

    def load_model(self, filename):
        filepath = os.path.join('models', filename)
        if os.path.exists(filepath):
            self.policy_value_net.load_state_dict(torch.load(filepath))
            self.best_model.load_state_dict(torch.load(filepath))
            print(f"Model loaded from {filepath}")

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
