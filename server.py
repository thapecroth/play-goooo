import asyncio
import json
import os
import glob
import re
from typing import Dict, Optional, List, Tuple
import torch
import torch.nn as nn
import numpy as np
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import socketio
from optimized_go import OptimizedGoGame, OptimizedGoAI
from mcts_go import MCTSPlayer
from alpha_go import PolicyValueNet, AlphaGoPlayer

# Deep Q-Network for Go
class GoDQN(nn.Module):
    def __init__(self, board_size=9, depth=6):
        super(GoDQN, self).__init__()
        self.board_size = board_size
        self.depth = depth
        self.num_actions = board_size * board_size + 1  # +1 for pass action
        
        # Convolutional layers for board pattern recognition
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        
        # Residual blocks for deeper feature extraction
        self.res_blocks = nn.ModuleList([
            self._make_res_block(256) for _ in range(depth)
        ])
        
        # Q-value head
        self.q_head = nn.Sequential(
            nn.Conv2d(256, 64, kernel_size=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64 * board_size * board_size, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, self.num_actions)
        )
        
    def _make_res_block(self, channels):
        return nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(channels),
            nn.ReLU(),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(channels)
        )
    
    def forward(self, x):
        assert x.dim() == 4, f"Expected 4D input (batch, channels, height, width), got {x.dim()}D"
        assert x.size(1) == 3, f"Expected 3 input channels, got {x.size(1)}"
        assert x.size(2) == self.board_size and x.size(3) == self.board_size, \
            f"Expected board size {self.board_size}x{self.board_size}, got {x.size(2)}x{x.size(3)}"
        
        # Initial convolutions
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = torch.relu(self.conv3(x))
        
        # Residual blocks
        for res_block in self.res_blocks:
            residual = x
            x = res_block(x)
            x = torch.relu(x + residual)
        
        # Q-values for all actions
        q_values = self.q_head(x)
        
        assert q_values.size(1) == self.num_actions, \
            f"Expected {self.num_actions} Q-values, got {q_values.size(1)}"
        
        return q_values

class GoGame:
    def __init__(self, size=9):
        self.size = size
        self.board = [[None for _ in range(size)] for _ in range(size)]
        self.current_player = 'black'
        self.captures = {'black': 0, 'white': 0}
        self.history = []
        self.passes = 0
        self.game_over = False
        self.winner = None
        self.last_move = None
        self.ko = None
    
    def make_move(self, x: int, y: int, color: str) -> bool:
        if (self.game_over or x < 0 or x >= self.size or 
            y < 0 or y >= self.size or self.board[y][x] is not None or
            color != self.current_player):
            return False
        
        board_copy = self.copy_board()
        self.board[y][x] = color
        
        opposite_color = 'white' if color == 'black' else 'black'
        captured_stones = self.capture_stones(opposite_color)
        
        if not self.has_liberties(x, y, color):
            self.board = board_copy
            return False
        
        if self.is_ko(board_copy):
            self.board = board_copy
            return False
        
        self.captures[color] += len(captured_stones)
        self.history.append({
            'board': board_copy,
            'move': {'x': x, 'y': y, 'color': color},
            'captures': self.captures.copy()
        })
        
        self.last_move = {'x': x, 'y': y}
        self.passes = 0
        self.current_player = opposite_color
        self.ko = captured_stones[0] if len(captured_stones) == 1 else None
        
        return True
    
    def capture_stones(self, color: str) -> List[Dict]:
        captured = []
        processed = set()
        
        for y in range(self.size):
            for x in range(self.size):
                if (self.board[y][x] == color and 
                    f"{x},{y}" not in processed):
                    group = self.get_group(x, y)
                    if not self.group_has_liberties(group):
                        for stone in group:
                            self.board[stone['y']][stone['x']] = None
                            captured.append(stone)
                            processed.add(f"{stone['x']},{stone['y']}")
                    else:
                        # Mark group as processed even if not captured
                        for stone in group:
                            processed.add(f"{stone['x']},{stone['y']}")
        
        return captured
    
    def get_group(self, x: int, y: int) -> List[Dict]:
        color = self.board[y][x]
        if not color:
            return []
        
        group = []
        visited = set()
        stack = [{'x': x, 'y': y}]
        
        while stack:
            pos = stack.pop()
            key = f"{pos['x']},{pos['y']}"
            
            if key in visited:
                continue
            visited.add(key)
            
            if self.board[pos['y']][pos['x']] == color:
                group.append(pos)
                
                for neighbor in self.get_neighbors(pos['x'], pos['y']):
                    neighbor_key = f"{neighbor['x']},{neighbor['y']}"
                    if neighbor_key not in visited:
                        stack.append(neighbor)
        
        return group
    
    def get_neighbors(self, x: int, y: int) -> List[Dict]:
        neighbors = []
        if x > 0:
            neighbors.append({'x': x - 1, 'y': y})
        if x < self.size - 1:
            neighbors.append({'x': x + 1, 'y': y})
        if y > 0:
            neighbors.append({'x': x, 'y': y - 1})
        if y < self.size - 1:
            neighbors.append({'x': x, 'y': y + 1})
        return neighbors
    
    def group_has_liberties(self, group: List[Dict]) -> bool:
        for stone in group:
            for neighbor in self.get_neighbors(stone['x'], stone['y']):
                if self.board[neighbor['y']][neighbor['x']] is None:
                    return True
        return False
    
    def has_liberties(self, x: int, y: int, color: str) -> bool:
        temp_color = self.board[y][x]
        self.board[y][x] = color
        group = self.get_group(x, y)
        has_lib = self.group_has_liberties(group)
        self.board[y][x] = temp_color
        return has_lib
    
    def is_ko(self, previous_board) -> bool:
        if len(self.history) < 1:
            return False
        
        for y in range(self.size):
            for x in range(self.size):
                if self.board[y][x] != previous_board[y][x]:
                    return False
        return True
    
    def copy_board(self):
        return [row[:] for row in self.board]
    
    def pass_turn(self, color: str) -> bool:
        if color != self.current_player:
            return False
        
        self.passes += 1
        self.current_player = 'white' if color == 'black' else 'black'
        
        if self.passes >= 2:
            self.end_game()
        
        return True
    
    def resign(self, color: str):
        self.game_over = True
        self.winner = 'white' if color == 'black' else 'black'
    
    def end_game(self):
        self.game_over = True
        score = self.calculate_score()
        self.winner = 'black' if score['black'] > score['white'] else 'white'
    
    def calculate_score(self) -> Dict[str, float]:
        territory = {'black': 0, 'white': 0}
        visited = set()
        
        for y in range(self.size):
            for x in range(self.size):
                key = f"{x},{y}"
                if self.board[y][x] is None and key not in visited:
                    region = self.get_empty_region(x, y, visited)
                    owner = self.get_region_owner(region)
                    if owner:
                        territory[owner] += len(region)
        
        stones = {'black': 0, 'white': 0}
        for y in range(self.size):
            for x in range(self.size):
                if self.board[y][x]:
                    stones[self.board[y][x]] += 1
        
        return {
            'black': stones['black'] + territory['black'] + self.captures['black'],
            'white': stones['white'] + territory['white'] + self.captures['white'] + 6.5
        }
    
    def get_empty_region(self, start_x: int, start_y: int, visited: set) -> List[Dict]:
        region = []
        stack = [{'x': start_x, 'y': start_y}]
        
        while stack:
            pos = stack.pop()
            key = f"{pos['x']},{pos['y']}"
            
            if key in visited:
                continue
            visited.add(key)
            
            if self.board[pos['y']][pos['x']] is None:
                region.append(pos)
                for neighbor in self.get_neighbors(pos['x'], pos['y']):
                    stack.append(neighbor)
        
        return region
    
    def get_region_owner(self, region: List[Dict]) -> Optional[str]:
        border_colors = set()
        
        for pos in region:
            for neighbor in self.get_neighbors(pos['x'], pos['y']):
                color = self.board[neighbor['y']][neighbor['x']]
                if color:
                    border_colors.add(color)
        
        if len(border_colors) == 1:
            return list(border_colors)[0]
        return None
    
    def get_valid_moves(self, color: str) -> List[Dict]:
        moves = []
        for y in range(self.size):
            for x in range(self.size):
                if self.board[y][x] is None:
                    board_copy = self.copy_board()
                    current_player_copy = self.current_player
                    self.current_player = color
                    
                    if self.make_move(x, y, color):
                        moves.append({'x': x, 'y': y})
                        self.board = board_copy
                        self.current_player = current_player_copy
                    else:
                        self.board = board_copy
                        self.current_player = current_player_copy
        
        return moves
    
    def get_state(self):
        # Ensure all values are JSON serializable
        captures = {k: int(v) if hasattr(v, 'item') else v for k, v in self.captures.items()}
        last_move = None
        if self.last_move:
            last_move = {k: int(v) if hasattr(v, 'item') else v for k, v in self.last_move.items()}
        
        return {
            'board': self.board,
            'currentPlayer': self.current_player,
            'captures': captures,
            'gameOver': self.game_over,
            'winner': self.winner,
            'lastMove': last_move,
            'score': self.calculate_score() if self.game_over else None
        }
    
    def to_tensor(self) -> torch.Tensor:
        """Convert board state to PyTorch tensor for neural network input"""
        # 3 channels: black stones, white stones, empty positions
        tensor = torch.zeros(3, self.size, self.size)
        
        for y in range(self.size):
            for x in range(self.size):
                if self.board[y][x] == 'black':
                    tensor[0, y, x] = 1.0
                elif self.board[y][x] == 'white':
                    tensor[1, y, x] = 1.0
                else:
                    tensor[2, y, x] = 1.0
        
        return tensor.unsqueeze(0)  # Add batch dimension

class GoAI:
    def __init__(self, game: GoGame, ai_type: str = 'classic', model_path: Optional[str] = None):
        self.game = game
        self.max_depth = 3
        self.ai_type = ai_type
        self.device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
        
        # Initialize DQN (will be loaded on demand)
        self.dqn = None
        self.current_model_path = None
        self.model_info = None

        # Initialize AlphaGo
        self.alpha_go_net = None
        self.alpha_go_player = None
        
        # Initialize optimized AI for classic moves
        self.optimized_ai = OptimizedGoAI(max_depth=self.max_depth)
        self.mcts_ai = MCTSPlayer(simulations=1000, exploration_constant=1.414, time_limit=5.0)
        self.optimized_game = None  # Will be synced when needed
        self.classic_algorithm = 'minimax'  # 'minimax' or 'mcts'
        
        # Load trained model if path provided and AI type is DQN
        if ai_type == 'dqn' and model_path:
            self.load_model(model_path)
        
        # Fallback evaluation weights for traditional minimax
        self.evaluation_weights = {
            'territory': 1.0,
            'captures': 10.0,
            'liberties': 0.5,
            'influence': 0.3
        }
        
        # Training mode flag
        self.training_mode = False
    
    def set_ai_type(self, ai_type: str):
        """Switch between classic and DQN AI types"""
        self.ai_type = ai_type
        print(f"AI type switched to: {ai_type}")
    
    def set_classic_algorithm(self, algorithm: str):
        """Switch between minimax and MCTS for classic AI"""
        if algorithm in ['minimax', 'mcts']:
            self.classic_algorithm = algorithm
            print(f"Classic algorithm switched to: {algorithm}")
    
    def set_mcts_params(self, simulations: int = None, exploration: float = None, time_limit: int = None):
        """Update MCTS parameters"""
        if simulations is not None:
            self.mcts_ai.simulations = simulations
        if exploration is not None:
            self.mcts_ai.exploration_constant = exploration
        if time_limit is not None:
            self.mcts_ai.time_limit = float(time_limit)
    
    def load_model(self, model_path: str) -> bool:
        """Load a DQN model from file"""
        full_path = os.path.join('models', model_path) if not model_path.startswith('models/') else model_path
        
        if not os.path.exists(full_path):
            print(f"Model file not found: {full_path}")
            return False
        
        try:
            # Load checkpoint
            checkpoint = torch.load(full_path, map_location=self.device)
            
            # Extract model parameters
            board_size = checkpoint.get('board_size', 9)
            depth = checkpoint.get('depth', 6)
            
            # Initialize DQN with correct parameters
            self.dqn = GoDQN(board_size, depth).to(self.device)
            self.dqn.load_state_dict(checkpoint['q_network_state_dict'])
            self.dqn.eval()
            
            self.current_model_path = model_path
            self.model_info = {
                'name': model_path,
                'board_size': board_size,
                'depth': depth,
                'steps_done': checkpoint.get('steps_done', 0)
            }
            
            print(f"Successfully loaded DQN model: {model_path} (board: {board_size}x{board_size}, depth: {depth})")
            return True
            
        except Exception as e:
            print(f"Error loading model {model_path}: {e}")
            return False

    def load_alpha_go_model(self, model_path: str) -> bool:
        full_path = os.path.join('models', model_path) if not model_path.startswith('models/') else model_path
        
        if not os.path.exists(full_path):
            print(f"Model file not found: {full_path}")
            return False
        
        try:
            self.alpha_go_net = PolicyValueNet(self.game.size)
            self.alpha_go_net.load_state_dict(torch.load(full_path, map_location=self.device))
            self.alpha_go_net.eval()
            self.alpha_go_player = AlphaGoPlayer(self.alpha_go_net)
            print(f"Successfully loaded AlphaGo model: {model_path}")
            return True
        except Exception as e:
            print(f"Error loading AlphaGo model {model_path}: {e}")
            return False
    
    def get_best_move(self, color: str) -> Optional[Dict]:
        valid_moves = self.game.get_valid_moves(color)
        if not valid_moves:
            return None
        
        if self.ai_type == 'dqn' and self.dqn is not None:
            return self.get_dqn_move(color)
        elif self.ai_type == 'alpha_go' and self.alpha_go_player is not None:
            return self.get_alpha_go_move(color)
        else:
            return self.get_classic_move(color)
    
    def get_dqn_move(self, color: str) -> Optional[Dict]:
        """Get move using DQN"""
        valid_moves = self.game.get_valid_moves(color)
        if not valid_moves:
            return None
        
        # Use DQN for move selection during inference
        with torch.no_grad():
            # Convert game state to tensor
            state_tensor = self.game_to_tensor(color).to(self.device)
            
            # Get Q-values from DQN
            q_values = self.dqn(state_tensor)
            q_values = q_values.cpu().numpy().flatten()
            
            # Create list of valid actions
            valid_actions = []
            for move in valid_moves:
                action_idx = move['y'] * self.game.size + move['x']
                valid_actions.append((action_idx, move))
            
            # Find best valid action
            best_move = None
            best_q_value = -float('inf')
            
            for action_idx, move in valid_actions:
                if q_values[action_idx] > best_q_value:
                    best_q_value = q_values[action_idx]
                    best_move = move
            
            # Fallback to traditional evaluation if Q-values are poor
            if best_q_value < -10.0 and len(valid_moves) > 0:
                print("DQN confidence low, using traditional evaluation as fallback")
                return self.get_classic_move(color)
            
            return best_move

    def get_alpha_go_move(self, color: str) -> Optional[Dict]:
        self._sync_to_optimized_game()
        best_move_tuple = self.alpha_go_player.get_move(self.optimized_game, color)
        if best_move_tuple is None:
            return None
        return {'x': int(best_move_tuple[0]), 'y': int(best_move_tuple[1])}

    
    def get_q_values_for_visualization(self, color: str) -> Optional[List[List[float]]]:
        """Get Q-values for all board positions for visualization"""
        if self.ai_type != 'dqn' or self.dqn is None:
            return None
        
        with torch.no_grad():
            # Convert game state to tensor
            state_tensor = self.game_to_tensor(color).to(self.device)
            
            # Get Q-values from DQN
            q_values = self.dqn(state_tensor)
            q_values = q_values.cpu().numpy().flatten()
            
            # Convert to 2D board format (exclude pass action)
            board_q_values = []
            for y in range(self.game.size):
                row = []
                for x in range(self.game.size):
                    action_idx = y * self.game.size + x
                    if action_idx < len(q_values) - 1:  # Exclude pass action
                        # Set to None for occupied positions
                        if self.game.board[y][x] is not None:
                            row.append(None)
                        else:
                            row.append(float(q_values[action_idx]))
                    else:
                        row.append(None)
                board_q_values.append(row)
            
            return board_q_values
    
    def get_classic_move(self, color: str) -> Optional[Dict]:
        """Get move using selected classic algorithm (minimax or MCTS)"""
        # Sync game state to optimized representation
        self._sync_to_optimized_game()
        
        if self.classic_algorithm == 'mcts':
            # Use MCTS
            best_move_tuple = self.mcts_ai.get_move(self.optimized_game, color)
        else:
            # Use minimax (default)
            best_move_tuple = self.optimized_ai.get_best_move(self.optimized_game, color)
        
        if best_move_tuple is None:
            return None
        
        # Convert back to dict format
        return {'x': int(best_move_tuple[0]), 'y': int(best_move_tuple[1])}
    
    def _sync_to_optimized_game(self):
        """Sync current game state to optimized game representation"""
        self.optimized_game = OptimizedGoGame(self.game.size)
        
        # Copy board state
        for y in range(self.game.size):
            for x in range(self.game.size):
                if self.game.board[y][x] == 'black':
                    self.optimized_game.board[y, x] = 1
                elif self.game.board[y][x] == 'white':
                    self.optimized_game.board[y, x] = 2
        
        # Copy game state
        self.optimized_game.current_player = 1 if self.game.current_player == 'black' else 2
        self.optimized_game.captures = {1: self.game.captures['black'], 2: self.game.captures['white']}
        self.optimized_game.passes = self.game.passes
        self.optimized_game.game_over = self.game.game_over
        self.optimized_game.winner = self.game.winner
        if self.game.last_move:
            self.optimized_game.last_move = (self.game.last_move['x'], self.game.last_move['y'])
        
        # Update max depth if needed
        self.optimized_ai.max_depth = self.max_depth
    
    def game_to_tensor(self, player: str) -> torch.Tensor:
        """Convert game state to tensor for DQN input"""
        # 3 channels: current player stones, opponent stones, empty positions
        tensor = torch.zeros(3, self.game.size, self.game.size)
        opponent = 'white' if player == 'black' else 'black'
        
        for y in range(self.game.size):
            for x in range(self.game.size):
                if self.game.board[y][x] == player:
                    tensor[0, y, x] = 1.0
                elif self.game.board[y][x] == opponent:
                    tensor[1, y, x] = 1.0
                else:
                    tensor[2, y, x] = 1.0
        
        return tensor.unsqueeze(0)  # Add batch dimension
    
    def get_fallback_move(self, color: str) -> Optional[Dict]:
        """Traditional minimax-based move selection as fallback"""
        valid_moves = self.game.get_valid_moves(color)
        if not valid_moves:
            return None
        
        best_move = None
        best_score = -float('inf')
        
        for move in valid_moves:
            score = self.evaluate_move(move['x'], move['y'], color)
            if score > best_score:
                best_score = score
                best_move = move
        
        return best_move
    
    def load_model(self, model_path: str):
        """Load a trained DQN model"""
        try:
            checkpoint = torch.load(model_path, map_location=self.device)
            self.dqn.load_state_dict(checkpoint['q_network_state_dict'])
            self.dqn.eval()
            print(f"✓ Successfully loaded DQN model from {model_path}")
        except Exception as e:
            print(f"Error loading model from {model_path}: {e}")
            print("Using untrained DQN model")
    
    def set_training_mode(self, training: bool):
        """Toggle between training and inference modes"""
        self.training_mode = training
        if training:
            self.dqn.train()
        else:
            self.dqn.eval()
    
    def evaluate_move(self, x: int, y: int, color: str) -> float:
        # Traditional evaluation as fallback
        original_board = self.game.copy_board()
        original_captures = self.game.captures.copy()
        original_player = self.game.current_player
        original_last_move = self.game.last_move
        original_ko = self.game.ko
        
        self.game.current_player = color
        move_success = self.game.make_move(x, y, color)
        
        if not move_success:
            self.game.board = original_board
            self.game.captures = original_captures
            self.game.current_player = original_player
            self.game.last_move = original_last_move
            self.game.ko = original_ko
            return -float('inf')
        
        score = self.minimax(self.max_depth - 1, -float('inf'), float('inf'), False, color)
        
        # Restore game state
        self.game.board = original_board
        self.game.captures = original_captures
        self.game.current_player = original_player
        self.game.last_move = original_last_move
        self.game.ko = original_ko
        
        return score
    
    def minimax(self, depth: int, alpha: float, beta: float, is_maximizing: bool, ai_color: str) -> float:
        if depth == 0 or self.game.game_over:
            return self.evaluate_position(ai_color)
        
        current_color = self.game.current_player
        valid_moves = self.game.get_valid_moves(current_color)
        
        if not valid_moves:
            original_passes = self.game.passes
            self.game.pass_turn(current_color)
            score = self.minimax(depth - 1, alpha, beta, not is_maximizing, ai_color)
            self.game.passes = original_passes
            self.game.current_player = current_color
            return score
        
        if is_maximizing:
            max_score = -float('inf')
            
            for move in valid_moves:
                board_copy = self.game.copy_board()
                captures_copy = self.game.captures.copy()
                player_copy = self.game.current_player
                
                if self.game.make_move(move['x'], move['y'], current_color):
                    score = self.minimax(depth - 1, alpha, beta, False, ai_color)
                    max_score = max(max_score, score)
                    alpha = max(alpha, score)
                    
                    self.game.board = board_copy
                    self.game.captures = captures_copy
                    self.game.current_player = player_copy
                    
                    if beta <= alpha:
                        break
            
            return max_score
        else:
            min_score = float('inf')
            
            for move in valid_moves:
                board_copy = self.game.copy_board()
                captures_copy = self.game.captures.copy()
                player_copy = self.game.current_player
                
                if self.game.make_move(move['x'], move['y'], current_color):
                    score = self.minimax(depth - 1, alpha, beta, True, ai_color)
                    min_score = min(min_score, score)
                    beta = min(beta, score)
                    
                    self.game.board = board_copy
                    self.game.captures = captures_copy
                    self.game.current_player = player_copy
                    
                    if beta <= alpha:
                        break
            
            return min_score
    
    def evaluate_position(self, ai_color: str) -> float:
        score = 0
        opposite_color = 'white' if ai_color == 'black' else 'black'
        
        # Capture advantage
        score += (self.game.captures[ai_color] - self.game.captures[opposite_color]) * self.evaluation_weights['captures']
        
        # Territory estimation
        territory_score = self.estimate_territory(ai_color)
        score += territory_score * self.evaluation_weights['territory']
        
        # Liberty advantage
        liberty_score = self.count_liberties(ai_color) - self.count_liberties(opposite_color)
        score += liberty_score * self.evaluation_weights['liberties']
        
        # Influence
        influence_score = self.calculate_influence(ai_color)
        score += influence_score * self.evaluation_weights['influence']
        
        return score
    
    def estimate_territory(self, color: str) -> float:
        territory = 0
        opposite_color = 'white' if color == 'black' else 'black'
        
        for y in range(self.game.size):
            for x in range(self.game.size):
                if self.game.board[y][x] is None:
                    influence = self.get_point_influence(x, y)
                    if influence[color] > influence[opposite_color] * 1.5:
                        territory += 1
                    elif influence[opposite_color] > influence[color] * 1.5:
                        territory -= 1
        
        return territory
    
    def get_point_influence(self, x: int, y: int) -> Dict[str, float]:
        influence = {'black': 0, 'white': 0}
        max_distance = 4
        
        for dy in range(-max_distance, max_distance + 1):
            for dx in range(-max_distance, max_distance + 1):
                nx, ny = x + dx, y + dy
                
                if 0 <= nx < self.game.size and 0 <= ny < self.game.size:
                    stone = self.game.board[ny][nx]
                    if stone:
                        distance = abs(dx) + abs(dy)
                        influence[stone] += 1 / (distance + 1)
        
        return influence
    
    def count_liberties(self, color: str) -> int:
        total_liberties = 0
        counted = set()
        
        for y in range(self.game.size):
            for x in range(self.game.size):
                if self.game.board[y][x] == color and f"{x},{y}" not in counted:
                    group = self.game.get_group(x, y)
                    liberties = set()
                    
                    for stone in group:
                        counted.add(f"{stone['x']},{stone['y']}")
                        for neighbor in self.game.get_neighbors(stone['x'], stone['y']):
                            if self.game.board[neighbor['y']][neighbor['x']] is None:
                                liberties.add(f"{neighbor['x']},{neighbor['y']}")
                    
                    total_liberties += len(liberties)
        
        return total_liberties
    
    def calculate_influence(self, color: str) -> float:
        influence = 0
        
        for y in range(self.game.size):
            for x in range(self.game.size):
                if self.game.board[y][x] == color:
                    influence += self.get_stone_value(x, y)
        
        return influence
    
    def get_stone_value(self, x: int, y: int) -> float:
        value = 1.0
        
        distance_to_edge = min(x, y, self.game.size - 1 - x, self.game.size - 1 - y)
        if distance_to_edge == 0:
            value *= 0.7
        elif distance_to_edge == 1:
            value *= 0.85
        
        # Star points bonus
        if ((x == 3 or x == self.game.size - 4) and 
            (y == 3 or y == self.game.size - 4)):
            value *= 1.2
        
        return value

# FastAPI and Socket.IO setup
app = FastAPI()
sio = socketio.AsyncServer(cors_allowed_origins="*", async_mode='asgi')
socket_app = socketio.ASGIApp(sio, app)

# Serve static files
app.mount("/static", StaticFiles(directory="public"), name="static")

@app.get("/")
async def read_index():
    return FileResponse('public/index.html')

# Game storage
games: Dict[str, Dict] = {}

def get_available_models(model_type: str) -> List[Dict]:
    """Get list of available trained models"""
    models_dir = "models"
    if not os.path.exists(models_dir):
        return []
    
    models = []
    file_pattern = "*.pth"
    if model_type == 'dqn':
        file_pattern = "dqn_*.pth"
    elif model_type == 'alpha_go':
        file_pattern = "alpha_go_*.pth"

    for file in glob.glob(os.path.join(models_dir, file_pattern)):
        filename = os.path.basename(file)
        
        # Parse model info from filename
        # Format: dqn_go_board{size}_depth{depth}_episode_{episode}.pth
        match = re.match(r'dqn_go_board(\d+)_depth(\d+)_(?:episode_(\d+)|final)\.pth', filename)
        if match:
            board_size = int(match.group(1))
            depth = int(match.group(2))
            episodes = match.group(3) if match.group(3) else 'final'
            
            models.append({
                'name': filename,
                'boardSize': board_size,
                'depth': depth,
                'episodes': episodes
            })
        else:
            # Fallback for other filename formats
            models.append({
                'name': filename,
                'boardSize': 9,
                'depth': 6,
                'episodes': 'unknown'
            })
    
    # Sort by board size, then depth, then episodes
    models.sort(key=lambda x: (x['boardSize'], x['depth'], 
                              float('inf') if x['episodes'] == 'final' else 
                              int(x['episodes']) if x['episodes'].isdigit() else 0))
    
    return models

@sio.event
async def connect(sid, environ):
    print(f'New client connected: {sid}')

@sio.event
async def disconnect(sid):
    print(f'Client disconnected: {sid}')
    if sid in games:
        del games[sid]

@sio.event
async def newGame(sid, board_size=9):
    game = GoGame(board_size)
    ai = GoAI(game, ai_type='classic')
    games[sid] = {
        'game': game, 
        'ai': ai, 
        'ai_type': 'classic',
        'current_model': None
    }
    await sio.emit('gameState', game.get_state(), room=sid)

@sio.event
async def makeMove(sid, data):
    if sid not in games:
        await sio.emit('error', 'No game found', room=sid)
        return
    
    game_data = games[sid]
    game, ai = game_data['game'], game_data['ai']
    x, y = data['x'], data['y']
    
    if game.make_move(x, y, 'black'):
        await sio.emit('gameState', game.get_state(), room=sid)
        
        if not game.game_over:
            # Add delay for AI move
            await asyncio.sleep(0.8)
            ai_move = ai.get_best_move('white')
            if ai_move:
                game.make_move(ai_move['x'], ai_move['y'], 'white')
            else:
                game.pass_turn('white')
            await sio.emit('gameState', game.get_state(), room=sid)
    else:
        await sio.emit('invalidMove', {'x': x, 'y': y}, room=sid)

@sio.event
async def pass_move(sid):
    if sid not in games:
        return
    
    game_data = games[sid]
    game, ai = game_data['game'], game_data['ai']
    game.pass_turn('black')
    await sio.emit('gameState', game.get_state(), room=sid)
    
    if not game.game_over:
        await asyncio.sleep(0.8)
        ai_move = ai.get_best_move('white')
        if ai_move:
            game.make_move(ai_move['x'], ai_move['y'], 'white')
        else:
            game.pass_turn('white')
        await sio.emit('gameState', game.get_state(), room=sid)

@sio.event
async def setAiDepth(sid, depth):
    if sid not in games:
        return
    
    ai = games[sid]['ai']
    ai.max_depth = max(1, min(5, depth))

@sio.event
async def setAiType(sid, ai_type):
    if sid not in games:
        return
    
    game_data = games[sid]
    game_data['ai'].set_ai_type(ai_type)
    game_data['ai_type'] = ai_type
    print(f"AI type set to {ai_type} for client {sid}")

@sio.event
async def getAvailableModels(sid, data):
    model_type = data.get('modelType', 'all')
    models = get_available_models(model_type)
    await sio.emit('availableModels', {'modelType': model_type, 'models': models}, room=sid)

@sio.event
async def setAlphaGoModel(sid, model_name):
    if sid not in games:
        return
    
    game_data = games[sid]
    ai = game_data['ai']
    
    if ai.load_alpha_go_model(model_name):
        game_data['current_model'] = model_name
        await sio.emit('modelLoaded', {'name': model_name}, room=sid)
        print(f"AlphaGo model {model_name} loaded for client {sid}")
    else:
        await sio.emit('modelError', f"Failed to load model: {model_name}", room=sid)

@sio.event
async def setDqnModel(sid, model_name):
    if sid not in games:
        return
    
    game_data = games[sid]
    ai = game_data['ai']
    
    if ai.load_model(model_name):
        game_data['current_model'] = model_name
        await sio.emit('modelLoaded', ai.model_info, room=sid)
        print(f"DQN model {model_name} loaded for client {sid}")
    else:
        await sio.emit('modelError', f"Failed to load model: {model_name}", room=sid)

@sio.event
async def setClassicAlgorithm(sid, algorithm):
    if sid not in games:
        return
    
    game_data = games[sid]
    ai = game_data['ai']
    ai.set_classic_algorithm(algorithm)
    print(f"Classic algorithm set to {algorithm} for client {sid}")

@sio.event
async def setMctsParams(sid, params):
    if sid not in games:
        return
    
    game_data = games[sid]
    ai = game_data['ai']
    
    simulations = params.get('simulations')
    exploration = params.get('exploration')
    time_limit = params.get('timeLimit')
    
    ai.set_mcts_params(simulations=simulations, exploration=exploration, time_limit=time_limit)
    print(f"MCTS parameters updated for client {sid}: {params}")

@sio.event
async def getQValues(sid, data):
    if sid not in games:
        return
    
    game_data = games[sid]
    ai = game_data['ai']
    color = data.get('color', 'black')
    
    q_values = ai.get_q_values_for_visualization(color)
    if q_values is not None:
        await sio.emit('qValues', {
            'qValues': q_values,
            'color': color
        }, room=sid)
    else:
        await sio.emit('qValuesError', 'Q-values not available for current AI type', room=sid)

@sio.event
async def resign(sid):
    if sid not in games:
        return
    
    game = games[sid]['game']
    game.resign('black')
    await sio.emit('gameState', game.get_state(), room=sid)

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 3000))
    uvicorn.run(socket_app, host="127.0.0.1", port=port)