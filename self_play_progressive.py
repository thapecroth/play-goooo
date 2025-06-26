import torch
import torch.optim as optim
import torch.multiprocessing as mp
import numpy as np
from collections import deque
import random
import os
import pickle
import argparse
import time
from datetime import datetime
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing
from queue import Queue

# Try to import the fast compiled version first, fall back to OptimizedGoGame
try:
    # Import the Python version of FastGoGame for now
    # Once compiled, we can use subprocess to call the binary
    from go_game_codon_fast import FastGoGame, BLACK, WHITE, EMPTY
    
    # Wrapper to make FastGoGame compatible with OptimizedGoGame interface
    class OptimizedGoGame:
        def __init__(self, board_size: int = 9):
            self.game = FastGoGame(board_size)
            self.board_size = board_size
            self.size = board_size  # Add size attribute for compatibility
            # Create numpy view of the board
            self._update_board_view()
        
        def _update_board_view(self):
            """Update numpy board view from FastGoGame"""
            self.board = np.array([[self.game.board[y][x] for x in range(self.board_size)] 
                                  for y in range(self.board_size)], dtype=np.int8)
        
        @property
        def current_player(self):
            return self.game.current_player
        
        @current_player.setter
        def current_player(self, value):
            self.game.current_player = value
        
        @property
        def game_over(self):
            return self.game.game_over
        
        @game_over.setter
        def game_over(self, value):
            self.game.game_over = value
        
        @property
        def winner(self):
            if self.game.winner == BLACK:
                return 'black'
            elif self.game.winner == WHITE:
                return 'white'
            else:
                return 'draw'
        
        @winner.setter
        def winner(self, value):
            if value == 'black':
                self.game.winner = BLACK
            elif value == 'white':
                self.game.winner = WHITE
            else:
                self.game.winner = 0
        
        def make_move(self, x: int, y: int, color: str) -> bool:
            # Save current player
            saved_player = self.game.current_player
            # Set correct player
            self.game.current_player = BLACK if color == 'black' else WHITE
            success = self.game.make_move(x, y)
            if not success:
                self.game.current_player = saved_player
            self._update_board_view()
            return success
        
        def pass_turn(self, color: str):
            self.game.pass_turn()
            self._update_board_view()
        
        def _calculate_winner(self):
            self.game._calculate_winner()
        
        def copy(self):
            """Create a copy of the game state"""
            new_game = OptimizedGoGame(self.board_size)
            # Copy the internal game state
            new_game.game.board = [row[:] for row in self.game.board]
            new_game.game.current_player = self.game.current_player
            new_game.game.game_over = self.game.game_over
            new_game.game.winner = self.game.winner
            new_game.game.pass_count = self.game.pass_count
            new_game.game.captured_black = self.game.captured_black
            new_game.game.captured_white = self.game.captured_white
            new_game.game.ko_x = self.game.ko_x
            new_game.game.ko_y = self.game.ko_y
            new_game._update_board_view()
            return new_game
        
        def get_valid_moves(self, color: str):
            """Get valid moves for the current player"""
            moves = self.game.get_valid_moves()
            # Return moves as list of tuples, with None for pass
            return moves if moves else []
    
    print("Using FastGoGame implementation for improved performance")
    
except ImportError:
    # Fall back to original implementation
    from optimized_go import OptimizedGoGame, BLACK, WHITE
    EMPTY = 0
    print("Using standard OptimizedGoGame implementation")

from alpha_go import PolicyValueNet, AlphaGoPlayer
from classic_go_ai import ClassicGoAI

# Set multiprocessing start method for compatibility
if __name__ == '__main__':
    mp.set_start_method('spawn', force=True)

def get_device():
    """Get the best available device (MPS > CUDA > CPU)"""
    if torch.backends.mps.is_available():
        return torch.device('mps')
    elif torch.cuda.is_available():
        return torch.device('cuda')
    else:
        return torch.device('cpu')

def play_against_classic_ai(args):
    """Play a single game against classic AI for warmup"""
    game_idx, board_size, model_state_dict, device_str, simulations, ai_depth, num_blocks = args
    
    device = torch.device(device_str)
    
    # Create model and load state
    policy_value_net = PolicyValueNet(board_size, num_blocks=num_blocks)
    policy_value_net.load_state_dict(model_state_dict)
    policy_value_net.to(device)
    policy_value_net.eval()
    
    game = OptimizedGoGame(board_size)
    alpha_player = AlphaGoPlayer(policy_value_net, simulations=simulations, device=device)
    classic_ai = ClassicGoAI(board_size)
    
    game_data = []
    move_count = 0
    max_moves = board_size * board_size * 2
    
    # Randomly assign colors
    alpha_is_black = random.choice([True, False])
    
    start_time = time.time()
    
    while not game.game_over and move_count < max_moves:
        current_is_alpha = (game.current_player == BLACK) == alpha_is_black
        
        if current_is_alpha:
            # AlphaGo move - collect training data
            state_tensor = alpha_player._game_to_tensor(game)
            
            try:
                move = alpha_player.get_move(game, 'black' if game.current_player == BLACK else 'white')
            except Exception as e:
                print(f"\nError in AlphaGo move: {e}")
                move = None
            
            # Create policy target from MCTS statistics
            policy_target = torch.zeros(board_size * board_size + 1)
            
            if hasattr(alpha_player, 'last_root') and alpha_player.last_root and alpha_player.last_root.children:
                total_visits = sum(child.visits for child in alpha_player.last_root.children)
                
                if total_visits > 0:
                    for child in alpha_player.last_root.children:
                        if child.move is None:
                            action = board_size * board_size
                        else:
                            action = child.move[1] * board_size + child.move[0]
                        policy_target[action] = child.visits / total_visits
            
            game_data.append([state_tensor.cpu(), policy_target, 0])
        else:
            # Classic AI move with variable depth
            legal_moves = classic_ai.get_legal_moves(game.board, game.current_player)
            if legal_moves:
                # Limit moves evaluation based on depth
                # Higher depth = more thorough evaluation
                max_moves_to_eval = min(len(legal_moves), 5 + ai_depth * 3)
                
                # Sort moves by heuristic for better pruning
                move_scores = []
                for mv in legal_moves:
                    # mv is in (row, col) format from ClassicGoAI
                    quick_score = classic_ai.evaluate_move(game.board, mv[0], mv[1], game.current_player)
                    move_scores.append((quick_score, mv))
                
                move_scores.sort(reverse=True)
                best_score = -float('inf')
                best_move = None
                
                # Evaluate top moves more thoroughly based on depth
                for score, mv in move_scores[:max_moves_to_eval]:
                    # Simulate deeper if ai_depth is higher
                    if ai_depth > 2:
                        # Simple minimax-style lookahead
                        sim_game = OptimizedGoGame(board_size)
                        sim_game.board = game.board.copy()  # Use numpy copy for numpy array
                        sim_game.current_player = game.current_player
                        
                        # Convert (row, col) to (x, y) for OptimizedGoGame
                        x, y = mv[1], mv[0]
                        if sim_game.make_move(x, y, 'black' if game.current_player == BLACK else 'white'):
                            # Evaluate resulting position
                            future_score = classic_ai.evaluate_position(sim_game.board, sim_game.current_player)
                            score = score * 0.7 + future_score * 0.3 * (1 + ai_depth * 0.1)
                    
                    if score > best_score:
                        best_score = score
                        best_move = mv
                
                # Convert best_move from (row, col) to (x, y) format
                if best_move:
                    move = (best_move[1], best_move[0])  # Convert to (x, y)
                else:
                    move = None
            else:
                move = None
        
        # Execute move
        if move is None:
            game.pass_turn('black' if game.current_player == BLACK else 'white')
        else:
            success = game.make_move(move[0], move[1], 'black' if game.current_player == BLACK else 'white')
            if not success:
                game.pass_turn('black' if game.current_player == BLACK else 'white')
        
        move_count += 1
    
    # Determine winner and assign values
    if move_count >= max_moves:
        game.game_over = True
        game._calculate_winner()
    
    # Only include AlphaGo's moves in training data
    if game.winner == 'black':
        winner_val = 1.0 if alpha_is_black else -1.0
    elif game.winner == 'white':
        winner_val = -1.0 if alpha_is_black else 1.0
    else:
        winner_val = 0.0
    
    for i in range(len(game_data)):
        game_data[i][2] = winner_val
    
    game_time = time.time() - start_time
    alpha_won = (game.winner == 'black' and alpha_is_black) or (game.winner == 'white' and not alpha_is_black)
    
    return game_data, move_count, game_time, alpha_won

def play_self_play_game(args):
    """Play a single self-play game"""
    game_idx, board_size, model_state_dict, device_str, simulations, num_blocks = args
    
    device = torch.device(device_str)
    
    # Create model and load state
    policy_value_net = PolicyValueNet(board_size, num_blocks=num_blocks)
    policy_value_net.load_state_dict(model_state_dict)
    policy_value_net.to(device)
    policy_value_net.eval()
    
    game = OptimizedGoGame(board_size)
    player = AlphaGoPlayer(policy_value_net, simulations=simulations, is_self_play=True, device=device)
    
    game_data = []
    move_count = 0
    max_moves = board_size * board_size * 2
    
    start_time = time.time()
    
    while not game.game_over and move_count < max_moves:
        # Store state before move
        state_tensor = player._game_to_tensor(game)
        
        try:
            move = player.get_move(game, 'black' if game.current_player == BLACK else 'white')
        except Exception as e:
            print(f"\nError in self-play move: {e}")
            move = None
        
        # Create policy target from MCTS statistics
        policy_target = torch.zeros(board_size * board_size + 1)
        
        if hasattr(player, 'last_root') and player.last_root and player.last_root.children:
            total_visits = sum(child.visits for child in player.last_root.children)
            
            if total_visits > 0:
                for child in player.last_root.children:
                    if child.move is None:
                        action = board_size * board_size
                    else:
                        action = child.move[1] * board_size + child.move[0]
                    policy_target[action] = child.visits / total_visits

        game_data.append([state_tensor.cpu(), policy_target, 0])

        if move is None:
            game.pass_turn('black' if game.current_player == BLACK else 'white')
        else:
            success = game.make_move(move[0], move[1], 'black' if game.current_player == BLACK else 'white')
            if not success:
                game.pass_turn('black' if game.current_player == BLACK else 'white')
        
        move_count += 1
    
    # Check if game ended naturally or was terminated
    if move_count >= max_moves:
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
    
    game_time = time.time() - start_time
    
    return game_data, move_count, game_time

class ProgressiveTrainer:
    def __init__(self, board_size=9, num_blocks=5, learning_rate=1e-3, buffer_size=50000, num_workers=None):
        self.board_size = board_size
        self.num_blocks = num_blocks
        self.policy_value_net = PolicyValueNet(board_size, num_blocks=num_blocks)
        self.optimizer = optim.Adam(self.policy_value_net.parameters(), lr=learning_rate)
        self.replay_buffer = deque(maxlen=buffer_size)
        self.best_model = PolicyValueNet(board_size, num_blocks=num_blocks)
        self.best_model.load_state_dict(self.policy_value_net.state_dict())
        
        # Device management with MPS priority
        self.device = get_device()
        self.policy_value_net.to(self.device)
        self.best_model.to(self.device)
        print(f"Using device: {self.device}")
        
        # Set number of workers
        if num_workers is None:
            self.num_workers = min(multiprocessing.cpu_count(), 16)
        else:
            self.num_workers = num_workers
        print(f"Using {self.num_workers} parallel workers")
        
        # Statistics tracking
        self.classic_ai_progress = []  # Track win rates at each depth
        self.current_classic_depth = 1  # Start with easiest AI
        self.max_classic_depth = 6      # Maximum depth to reach
        self.consistency_threshold = 0.75  # Win rate needed to advance
        self.consistency_games = 20     # Games needed to check consistency

    def test_against_classic(self, ai_depth, num_games=20, simulations=30):
        """Test current model against classic AI at specific depth"""
        print(f"\nTesting against Classic AI (depth {ai_depth})...")
        
        # For multiprocessing, we need to use CPU even if MPS is available
        model_state_dict = {k: v.cpu() for k, v in self.policy_value_net.state_dict().items()}
        device_str = 'cpu'  # Always use CPU for parallel workers
        
        game_args = [
            (i, self.board_size, model_state_dict, device_str, simulations, ai_depth, self.num_blocks)
            for i in range(num_games)
        ]
        
        wins = 0
        total_time = 0
        
        with ProcessPoolExecutor(max_workers=self.num_workers) as executor:
            futures = {executor.submit(play_against_classic_ai, args): i 
                      for i, args in enumerate(game_args)}
            
            with tqdm(total=num_games, desc=f"Testing vs AI depth {ai_depth}", 
                     bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} Win rate: {postfix}') as pbar:
                for future in as_completed(futures):
                    try:
                        _, _, game_time, alpha_won = future.result()
                        total_time += game_time
                        if alpha_won:
                            wins += 1
                        
                        win_rate = wins / (pbar.n + 1) * 100
                        pbar.set_postfix_str(f'{win_rate:.1f}%')
                        pbar.update(1)
                        
                    except Exception as e:
                        print(f"\nError in test game: {e}")
                        pbar.update(1)
        
        win_rate = wins / num_games
        print(f"Win rate: {wins}/{num_games} ({win_rate*100:.1f}%)")
        
        return win_rate

    def progressive_warmup(self, games_per_depth=50, simulations=30):
        """Progressive warmup against increasingly difficult classic AI"""
        print(f"\n{'='*60}")
        print(f"PROGRESSIVE WARMUP PHASE")
        print(f"Target: Beat classic AI from depth 1 to {self.max_classic_depth}")
        print(f"Consistency threshold: {self.consistency_threshold*100:.0f}% win rate")
        print(f"{'='*60}\n")
        
        total_games = 0
        total_positions = 0
        
        while self.current_classic_depth <= self.max_classic_depth:
            print(f"\n{'='*50}")
            print(f"Training against Classic AI depth {self.current_classic_depth}")
            print(f"{'='*50}")
            
            # Train against current depth
            positions = self.train_against_classic(
                ai_depth=self.current_classic_depth,
                num_games=games_per_depth,
                simulations=simulations
            )
            total_games += games_per_depth
            total_positions += positions
            
            # Train the network on collected data
            if len(self.replay_buffer) >= 64:
                self.train_network(num_epochs=200, batch_size=64)
            
            # Test consistency
            win_rate = self.test_against_classic(
                ai_depth=self.current_classic_depth,
                num_games=self.consistency_games,
                simulations=simulations
            )
            
            self.classic_ai_progress.append({
                'depth': self.current_classic_depth,
                'win_rate': win_rate,
                'games_trained': games_per_depth
            })
            
            if win_rate >= self.consistency_threshold:
                print(f"✓ Consistently beating depth {self.current_classic_depth} AI!")
                print(f"  Win rate: {win_rate*100:.1f}%")
                self.current_classic_depth += 1
            else:
                print(f"✗ Not yet consistent at depth {self.current_classic_depth}")
                print(f"  Win rate: {win_rate*100:.1f}% (need {self.consistency_threshold*100:.0f}%)")
                print(f"  Training more games at this level...")
        
        print(f"\n{'='*60}")
        print(f"PROGRESSIVE WARMUP COMPLETE!")
        print(f"Successfully reached classic AI depth {self.max_classic_depth}")
        print(f"Total games: {total_games}, Total positions: {total_positions}")
        print(f"\nProgression summary:")
        for progress in self.classic_ai_progress:
            print(f"  Depth {progress['depth']}: {progress['win_rate']*100:.1f}% win rate")
        print(f"{'='*60}\n")
        
        return total_positions

    def train_against_classic(self, ai_depth, num_games, simulations):
        """Train against classic AI at specific depth"""
        # For multiprocessing, we need to use CPU even if MPS is available
        model_state_dict = {k: v.cpu() for k, v in self.policy_value_net.state_dict().items()}
        device_str = 'cpu'  # Always use CPU for parallel workers
        
        game_args = [
            (i, self.board_size, model_state_dict, device_str, simulations, ai_depth, self.num_blocks)
            for i in range(num_games)
        ]
        
        all_game_data = []
        total_moves = 0
        total_time = 0
        wins = 0
        
        with ProcessPoolExecutor(max_workers=self.num_workers) as executor:
            futures = {executor.submit(play_against_classic_ai, args): i 
                      for i, args in enumerate(game_args)}
            
            with tqdm(total=num_games, desc=f"Training vs AI depth {ai_depth}", 
                     bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} Win rate: {postfix}') as pbar:
                for future in as_completed(futures):
                    game_idx = futures[future]
                    try:
                        game_data, move_count, game_time, alpha_won = future.result()
                        all_game_data.extend(game_data)
                        total_moves += move_count
                        total_time += game_time
                        if alpha_won:
                            wins += 1
                        
                        win_rate = wins / (pbar.n + 1) * 100
                        pbar.set_postfix_str(f'{win_rate:.1f}%')
                        pbar.update(1)
                        
                    except Exception as e:
                        print(f"\nError in training game {game_idx}: {e}")
                        pbar.update(1)
        
        self.replay_buffer.extend(all_game_data)
        
        print(f"\nTraining complete for depth {ai_depth}")
        print(f"Win rate: {wins}/{num_games} ({wins/num_games*100:.1f}%)")
        print(f"Collected {len(all_game_data)} positions")
        
        return len(all_game_data)

    def collect_self_play_data(self, num_games=25, simulations=50):
        """Stage 1: Parallel data collection"""
        print(f"\n{'='*60}")
        print(f"SELF-PLAY DATA COLLECTION")
        print(f"Games: {num_games}, Workers: {self.num_workers}")
        print(f"{'='*60}\n")
        
        # For multiprocessing, we need to use CPU even if MPS is available
        model_state_dict = {k: v.cpu() for k, v in self.policy_value_net.state_dict().items()}
        device_str = 'cpu'  # Always use CPU for parallel workers
        
        game_args = [
            (i, self.board_size, model_state_dict, device_str, simulations, self.num_blocks)
            for i in range(num_games)
        ]
        
        all_game_data = []
        total_moves = 0
        total_time = 0
        
        with ProcessPoolExecutor(max_workers=self.num_workers) as executor:
            futures = {executor.submit(play_self_play_game, args): i 
                      for i, args in enumerate(game_args)}
            
            with tqdm(total=num_games, desc="Self-Play", 
                     bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}] {postfix}') as pbar:
                for future in as_completed(futures):
                    game_idx = futures[future]
                    try:
                        game_data, move_count, game_time = future.result()
                        all_game_data.extend(game_data)
                        total_moves += move_count
                        total_time += game_time
                        
                        avg_moves = total_moves / (pbar.n + 1)
                        pbar.set_postfix_str(f'Avg {avg_moves:.1f} moves/game')
                        pbar.update(1)
                        
                    except Exception as e:
                        print(f"\nError in self-play game {game_idx}: {e}")
                        pbar.update(1)
        
        self.replay_buffer.extend(all_game_data)
        
        print(f"\nSelf-play complete!")
        print(f"Collected {len(all_game_data)} positions from {num_games} games")
        print(f"Average game: {total_moves/num_games:.1f} moves, {total_time/num_games:.1f}s")
        print(f"Total buffer size: {len(self.replay_buffer)} positions")
        
        return len(all_game_data)

    def train_network(self, num_epochs=500, batch_size=64):
        """Network training"""
        if len(self.replay_buffer) < batch_size:
            print("Not enough data for training!")
            return []
        
        losses = []
        best_loss = float('inf')
        
        with tqdm(total=num_epochs, desc="Training Network", 
                 bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} Loss: {postfix}') as pbar:
            
            for epoch in range(num_epochs):
                epoch_losses = []
                
                # Multiple batches per epoch
                batches_per_epoch = min(len(self.replay_buffer) // batch_size, 10)
                
                for _ in range(batches_per_epoch):
                    try:
                        batch = random.sample(self.replay_buffer, batch_size)
                        states, policy_targets, value_targets = zip(*batch)
                        
                        # Move tensors to device
                        states = torch.cat([s.to(self.device) for s in states])
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
                        
                        epoch_losses.append({
                            'total': loss.item(),
                            'policy': policy_loss.item(),
                            'value': value_loss.item()
                        })
                        
                    except Exception as e:
                        print(f"\nError in training batch: {e}")
                
                if epoch_losses:
                    avg_loss = sum(l['total'] for l in epoch_losses) / len(epoch_losses)
                    losses.extend(epoch_losses)
                    
                    if avg_loss < best_loss:
                        best_loss = avg_loss
                    
                    pbar.set_postfix_str(f'{avg_loss:.4f} (best: {best_loss:.4f})')
                
                pbar.update(1)
        
        return losses

    def evaluate_model(self, num_games=10):
        """Evaluate current model against best model"""
        print(f"\nEvaluating model ({num_games} games)...")
        
        current_player = AlphaGoPlayer(self.policy_value_net, simulations=50, device=self.device)
        best_player = AlphaGoPlayer(self.best_model, simulations=50, device=self.device)
        wins = 0
        
        with tqdm(total=num_games, desc="Evaluation", 
                 bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} Win rate: {postfix}') as pbar:
            
            for i in range(num_games):
                game = OptimizedGoGame(self.board_size)
                p1 = current_player if i % 2 == 0 else best_player
                p2 = best_player if i % 2 == 0 else current_player
                
                move_count = 0
                max_moves = self.board_size * self.board_size * 2
                
                while not game.game_over and move_count < max_moves:
                    player_to_move = p1 if game.current_player == BLACK else p2
                    
                    try:
                        move = player_to_move.get_move(game, 'black' if game.current_player == BLACK else 'white')
                    except Exception as e:
                        move = None
                    
                    if move is None:
                        game.pass_turn('black' if game.current_player == BLACK else 'white')
                    else:
                        success = game.make_move(move[0], move[1], 'black' if game.current_player == BLACK else 'white')
                        if not success:
                            game.pass_turn('black' if game.current_player == BLACK else 'white')
                    
                    move_count += 1
                
                if move_count >= max_moves:
                    game.game_over = True
                    game._calculate_winner()
                
                if (game.winner == 'black' and i % 2 == 0) or (game.winner == 'white' and i % 2 != 0):
                    wins += 1
                
                win_rate = wins / (i + 1) * 100
                pbar.set_postfix_str(f'{win_rate:.1f}%')
                pbar.update(1)
        
        return wins / num_games

    def run_self_play_iteration(self, iteration, num_games=25, num_epochs=500, batch_size=64, 
                               eval_games=10, win_ratio_to_update=0.55):
        """Run one complete iteration of self-play training"""
        print(f"\n{'='*70}")
        print(f"SELF-PLAY ITERATION {iteration}")
        print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"{'='*70}")
        
        iter_start = time.time()
        
        # Data Collection
        positions = self.collect_self_play_data(num_games)
        
        # Training
        losses = self.train_network(num_epochs, batch_size)
        
        # Evaluation
        win_ratio = self.evaluate_model(eval_games)
        print(f"Win ratio against best model: {win_ratio:.2%}")
        
        # Update best model if improved
        improved = False
        if win_ratio > win_ratio_to_update:
            print(f"✓ New best model found! (win ratio: {win_ratio:.2%})")
            self.best_model.load_state_dict(self.policy_value_net.state_dict())
            improved = True
        else:
            print(f"✗ Model not improved (threshold: {win_ratio_to_update:.2%})")
        
        iter_time = time.time() - iter_start
        print(f"\nIteration completed in {iter_time:.1f}s")
        
        return improved, win_ratio

    def save_model(self, filename):
        """Save model with metadata"""
        try:
            os.makedirs('models', exist_ok=True)
            filepath = os.path.join('models', filename)
            torch.save({
                'model_state_dict': self.best_model.state_dict(),
                'board_size': self.board_size,
                'timestamp': datetime.now().isoformat(),
                'device': str(self.device),
                'progression': {
                    'classic_ai_progress': self.classic_ai_progress,
                    'current_classic_depth': self.current_classic_depth,
                    'max_depth_reached': self.current_classic_depth - 1
                }
            }, filepath)
            print(f"Model saved to {filepath}")
        except Exception as e:
            print(f"Error saving model: {e}")

    def load_model(self, filename):
        """Load model with error handling"""
        filepath = os.path.join('models', filename)
        if os.path.exists(filepath):
            try:
                checkpoint = torch.load(filepath, map_location=self.device)
                if 'model_state_dict' in checkpoint:
                    self.policy_value_net.load_state_dict(checkpoint['model_state_dict'])
                    self.best_model.load_state_dict(checkpoint['model_state_dict'])
                else:
                    self.policy_value_net.load_state_dict(checkpoint)
                    self.best_model.load_state_dict(checkpoint)
                print(f"Model loaded from {filepath}")
                
                if 'progression' in checkpoint:
                    prog = checkpoint['progression']
                    print(f"Previous progression:")
                    if 'classic_ai_progress' in prog:
                        for p in prog['classic_ai_progress']:
                            print(f"  Depth {p['depth']}: {p['win_rate']*100:.1f}% win rate")
                    if 'max_depth_reached' in prog:
                        print(f"  Max depth reached: {prog['max_depth_reached']}")
                        self.current_classic_depth = prog.get('current_classic_depth', 1)
                
                return True
            except Exception as e:
                print(f"Error loading model: {e}")
                return False
        else:
            print(f"Model file not found: {filepath}")
            return False

def main():
    parser = argparse.ArgumentParser(description='Progressive AlphaGo training')
    parser.add_argument('--board-size', type=int, default=9, help='Size of the Go board')
    parser.add_argument('--num-blocks', type=int, default=5, help='Number of residual blocks')
    parser.add_argument('--learning-rate', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--buffer-size', type=int, default=50000, help='Replay buffer size')
    
    # Progressive warmup parameters
    parser.add_argument('--max-classic-depth', type=int, default=6, help='Maximum classic AI depth to reach')
    parser.add_argument('--games-per-depth', type=int, default=50, help='Games to play at each depth level')
    parser.add_argument('--consistency-threshold', type=float, default=0.75, help='Win rate needed to advance')
    parser.add_argument('--consistency-games', type=int, default=20, help='Games to test consistency')
    parser.add_argument('--skip-warmup', action='store_true', help='Skip progressive warmup')
    
    # Self-play parameters
    parser.add_argument('--num-iterations', type=int, default=100, help='Number of self-play iterations')
    parser.add_argument('--games-per-iter', type=int, default=25, help='Self-play games per iteration')
    parser.add_argument('--epochs-per-iter', type=int, default=500, help='Training epochs per iteration')
    parser.add_argument('--batch-size', type=int, default=64, help='Training batch size')
    parser.add_argument('--eval-games', type=int, default=10, help='Evaluation games')
    parser.add_argument('--win-ratio-to-update', type=float, default=0.55, help='Win ratio to update best model')
    
    # System parameters
    parser.add_argument('--num-workers', type=int, help='Number of parallel workers')
    parser.add_argument('--model-name', type=str, default='progressive_model.pth', help='Model filename')
    parser.add_argument('--resume', action='store_true', help='Resume from existing model')
    
    args = parser.parse_args()
    
    # Create trainer
    trainer = ProgressiveTrainer(
        board_size=args.board_size,
        num_blocks=args.num_blocks,
        learning_rate=args.learning_rate,
        buffer_size=args.buffer_size,
        num_workers=args.num_workers
    )
    
    # Configure progressive parameters
    trainer.max_classic_depth = args.max_classic_depth
    trainer.consistency_threshold = args.consistency_threshold
    trainer.consistency_games = args.consistency_games
    
    # Load existing model if resuming
    if args.resume:
        trainer.load_model(args.model_name)
    
    # Progressive warmup phase
    if not args.skip_warmup:
        trainer.progressive_warmup(
            games_per_depth=args.games_per_depth,
            simulations=30
        )
        # Save after warmup
        trainer.save_model(f'warmup_{args.model_name}')
    
    # Main self-play training loop
    print(f"\n{'='*70}")
    print(f"MAIN SELF-PLAY TRAINING")
    print(f"Iterations: {args.num_iterations}")
    print(f"Games per iteration: {args.games_per_iter}")
    print(f"Epochs per iteration: {args.epochs_per_iter}")
    print(f"{'='*70}")
    
    for i in range(args.num_iterations):
        improved, win_ratio = trainer.run_self_play_iteration(
            iteration=i+1,
            num_games=args.games_per_iter,
            num_epochs=args.epochs_per_iter,
            batch_size=args.batch_size,
            eval_games=args.eval_games,
            win_ratio_to_update=args.win_ratio_to_update
        )
        
        # Save model if improved or checkpoint
        if improved:
            trainer.save_model(args.model_name)
        
        if (i + 1) % 10 == 0:
            checkpoint_name = f'checkpoint_progressive_iter_{i+1}.pth'
            trainer.save_model(checkpoint_name)
            print(f"Checkpoint saved: {checkpoint_name}")
        
        # Periodically test against classic AI to track progress
        if (i + 1) % 5 == 0:
            print(f"\nTesting against classic AI levels...")
            for depth in range(1, min(trainer.current_classic_depth + 1, trainer.max_classic_depth + 1)):
                win_rate = trainer.test_against_classic(depth, num_games=10, simulations=30)
                print(f"  Depth {depth}: {win_rate*100:.1f}% win rate")
    
    print(f"\n{'='*70}")
    print("Training completed!")
    print(f"{'='*70}")

if __name__ == '__main__':
    main()