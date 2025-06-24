"""
Optimized Go game implementation for faster AI performance
"""
import numpy as np
from typing import Set, Tuple, Optional, List, Dict
from dataclasses import dataclass
import numba
from numba import njit, types
from numba.typed import List as TypedList

# Constants for board representation
EMPTY = 0
BLACK = 1
WHITE = 2

@dataclass
class Move:
    x: int
    y: int
    color: int

class OptimizedGoGame:
    """Highly optimized Go game implementation using NumPy and Numba"""
    
    def __init__(self, size: int = 9):
        self.size = size
        self.board = np.zeros((size, size), dtype=np.int8)
        self.current_player = BLACK
        self.captures = {BLACK: 0, WHITE: 0}
        self.history = []
        self.passes = 0
        self.game_over = False
        self.winner = None
        self.last_move = None
        self.ko_point = None
        
        # Pre-compute neighbor offsets for better performance
        self._neighbor_offsets = np.array([[-1, 0], [1, 0], [0, -1], [0, 1]], dtype=np.int8)
        
        # Cache for liberty calculations
        self._liberty_cache = {}
        self._group_cache = {}
        
    def copy(self):
        """Fast copy of game state"""
        new_game = OptimizedGoGame(self.size)
        new_game.board = self.board.copy()
        new_game.current_player = self.current_player
        new_game.captures = self.captures.copy()
        new_game.passes = self.passes
        new_game.game_over = self.game_over
        new_game.winner = self.winner
        new_game.last_move = self.last_move
        new_game.ko_point = self.ko_point
        return new_game
    
    def get_color_int(self, color: str) -> int:
        """Convert string color to integer"""
        return BLACK if color == 'black' else WHITE
    
    def get_opponent(self, color: int) -> int:
        """Get opponent color"""
        return WHITE if color == BLACK else BLACK
    
    @staticmethod
    @njit
    def _get_neighbors_numba(x: int, y: int, size: int) -> List[Tuple[int, int]]:
        """Get valid neighbors (Numba optimized)"""
        neighbors = TypedList()
        if x > 0:
            neighbors.append((x - 1, y))
        if x < size - 1:
            neighbors.append((x + 1, y))
        if y > 0:
            neighbors.append((x, y - 1))
        if y < size - 1:
            neighbors.append((x, y + 1))
        return neighbors
    
    def get_group(self, x: int, y: int) -> Set[Tuple[int, int]]:
        """Get all stones in the same group (optimized with caching)"""
        color = self.board[y, x]
        if color == EMPTY:
            return set()
        
        # Check cache
        cache_key = (x, y, tuple(self.board.flatten()))
        if cache_key in self._group_cache:
            return self._group_cache[cache_key]
        
        group = set()
        stack = [(x, y)]
        
        while stack:
            cx, cy = stack.pop()
            if (cx, cy) in group:
                continue
            
            if self.board[cy, cx] == color:
                group.add((cx, cy))
                
                # Add neighbors
                for nx, ny in self._get_neighbors_numba(cx, cy, self.size):
                    if (nx, ny) not in group:
                        stack.append((nx, ny))
        
        # Cache result
        self._group_cache[cache_key] = group
        return group
    
    def count_liberties(self, group: Set[Tuple[int, int]]) -> int:
        """Count liberties for a group (optimized)"""
        liberties = set()
        
        for x, y in group:
            for nx, ny in self._get_neighbors_numba(x, y, self.size):
                if self.board[ny, nx] == EMPTY:
                    liberties.add((nx, ny))
        
        return len(liberties)
    
    def remove_captured_stones(self, color: int) -> List[Tuple[int, int]]:
        """Remove captured stones (optimized)"""
        captured = []
        checked = set()
        
        # Use NumPy to find all stones of the given color
        color_positions = np.argwhere(self.board == color)
        
        for y, x in color_positions:
            if (x, y) in checked:
                continue
            
            group = self.get_group(x, y)
            checked.update(group)
            
            if self.count_liberties(group) == 0:
                for gx, gy in group:
                    self.board[gy, gx] = EMPTY
                    captured.append((gx, gy))
        
        return captured
    
    def is_valid_move(self, x: int, y: int, color: int) -> bool:
        """Check if move is valid (optimized)"""
        if self.board[y, x] != EMPTY:
            return False
        
        # Temporarily place stone
        self.board[y, x] = color
        
        # Check for immediate capture of opponent stones
        opponent = self.get_opponent(color)
        opponent_captured = False
        
        for nx, ny in self._get_neighbors_numba(x, y, self.size):
            if self.board[ny, nx] == opponent:
                opponent_group = self.get_group(nx, ny)
                if self.count_liberties(opponent_group) == 0:
                    opponent_captured = True
                    break
        
        # Check for self-capture
        if not opponent_captured:
            own_group = self.get_group(x, y)
            if self.count_liberties(own_group) == 0:
                self.board[y, x] = EMPTY
                return False
        
        # Check ko rule
        if self.ko_point == (x, y):
            self.board[y, x] = EMPTY
            return False
        
        self.board[y, x] = EMPTY
        return True
    
    def make_move(self, x: int, y: int, color: str) -> bool:
        """Make a move (optimized)"""
        color_int = self.get_color_int(color)
        
        if self.game_over or not (0 <= x < self.size and 0 <= y < self.size):
            return False
        
        if not self.is_valid_move(x, y, color_int):
            return False
        
        # Clear caches
        self._liberty_cache.clear()
        self._group_cache.clear()
        
        # Place stone
        self.board[y, x] = color_int
        
        # Capture opponent stones
        opponent = self.get_opponent(color_int)
        captured = self.remove_captured_stones(opponent)
        self.captures[color_int] += len(captured)
        
        # Update ko point
        if len(captured) == 1 and self.count_liberties(self.get_group(x, y)) == 1:
            self.ko_point = captured[0]
        else:
            self.ko_point = None
        
        # Update game state
        self.last_move = (x, y)
        self.passes = 0
        self.current_player = opponent
        
        # Store in history (lightweight)
        self.history.append({
            'board_hash': hash(self.board.tobytes()),
            'captures': self.captures.copy()
        })
        
        return True
    
    def pass_turn(self, color: str) -> bool:
        """Pass turn"""
        color_int = self.get_color_int(color)
        if self.game_over or self.current_player != color_int:
            return False
        
        self.passes += 1
        self.current_player = self.get_opponent(color_int)
        
        if self.passes >= 2:
            self.game_over = True
            self._calculate_winner()
        
        return True
    
    def get_valid_moves(self, color: str) -> List[Tuple[int, int]]:
        """Get all valid moves (optimized)"""
        color_int = self.get_color_int(color)
        valid_moves = []
        
        # Use NumPy to find empty positions
        empty_positions = np.argwhere(self.board == EMPTY)
        
        for y, x in empty_positions:
            if self.is_valid_move(x, y, color_int):
                valid_moves.append((x, y))
        
        return valid_moves
    
    def _calculate_winner(self):
        """Calculate final winner"""
        # Simple territory counting
        black_score = self.captures[BLACK]
        white_score = self.captures[WHITE] + 6.5  # Komi
        
        # Count territory (simplified)
        for y in range(self.size):
            for x in range(self.size):
                if self.board[y, x] == BLACK:
                    black_score += 1
                elif self.board[y, x] == WHITE:
                    white_score += 1
        
        if black_score > white_score:
            self.winner = 'black'
        else:
            self.winner = 'white'
    
    def get_state_for_ai(self):
        """Get state in format compatible with existing AI"""
        board_list = []
        for y in range(self.size):
            row = []
            for x in range(self.size):
                if self.board[y, x] == EMPTY:
                    row.append(None)
                elif self.board[y, x] == BLACK:
                    row.append('black')
                else:
                    row.append('white')
            board_list.append(row)
        
        return {
            'board': board_list,
            'currentPlayer': 'black' if self.current_player == BLACK else 'white',
            'captures': {'black': int(self.captures[BLACK]), 'white': int(self.captures[WHITE])},
            'gameOver': self.game_over,
            'winner': self.winner,
            'lastMove': {'x': int(self.last_move[0]), 'y': int(self.last_move[1])} if self.last_move else None
        }


class OptimizedGoAI:
    """Optimized AI for Go using efficient algorithms"""
    
    def __init__(self, max_depth: int = 3):
        self.max_depth = max_depth
        self.transposition_table = {}  # Cache for board positions
        self.move_ordering_cache = {}  # Cache for move ordering
        
    def get_best_move(self, game: OptimizedGoGame, color: str) -> Optional[Tuple[int, int]]:
        """Get best move using optimized minimax with alpha-beta pruning"""
        color_int = game.get_color_int(color)
        valid_moves = game.get_valid_moves(color)
        
        if not valid_moves:
            return None
        
        # Quick evaluation for opening moves
        if len(game.history) < 4:
            return self._get_opening_move(game, valid_moves)
        
        best_move = None
        best_score = -float('inf')
        alpha = -float('inf')
        beta = float('inf')
        
        # Order moves for better pruning
        ordered_moves = self._order_moves(game, valid_moves, color_int)
        
        for move in ordered_moves:
            # Make move on a copy
            game_copy = game.copy()
            game_copy.make_move(move[0], move[1], color)
            
            # Evaluate position
            score = self._minimax(game_copy, self.max_depth - 1, alpha, beta, False, color_int)
            
            if score > best_score:
                best_score = score
                best_move = move
                alpha = max(alpha, score)
        
        return best_move
    
    def _get_opening_move(self, game: OptimizedGoGame, valid_moves: List[Tuple[int, int]]) -> Tuple[int, int]:
        """Get good opening move"""
        # Prefer corners and star points in opening
        size = game.size
        star_points = []
        
        if size == 9:
            star_points = [(2, 2), (2, 6), (6, 2), (6, 6), (4, 4)]
        elif size == 13:
            star_points = [(3, 3), (3, 9), (9, 3), (9, 9), (6, 6)]
        elif size == 19:
            star_points = [(3, 3), (3, 9), (3, 15), (9, 3), (9, 9), (9, 15), (15, 3), (15, 9), (15, 15)]
        
        # Check if any star point is available
        for point in star_points:
            if point in valid_moves:
                return point
        
        # Otherwise return a random valid move
        return valid_moves[0]
    
    def _order_moves(self, game: OptimizedGoGame, moves: List[Tuple[int, int]], color: int) -> List[Tuple[int, int]]:
        """Order moves for better alpha-beta pruning"""
        # Simple heuristic: prefer moves near existing stones
        scored_moves = []
        
        for move in moves:
            score = 0
            x, y = move
            
            # Check neighbors
            for nx, ny in game._get_neighbors_numba(x, y, game.size):
                if game.board[ny, nx] != EMPTY:
                    score += 1
            
            # Prefer center area
            center = game.size // 2
            distance_to_center = abs(x - center) + abs(y - center)
            score -= distance_to_center * 0.1
            
            scored_moves.append((score, move))
        
        # Sort by score (descending)
        scored_moves.sort(reverse=True)
        return [move for _, move in scored_moves]
    
    def _minimax(self, game: OptimizedGoGame, depth: int, alpha: float, beta: float, 
                 is_maximizing: bool, ai_color: int) -> float:
        """Minimax with alpha-beta pruning and transposition table"""
        # Check transposition table
        board_hash = hash(game.board.tobytes())
        tt_key = (board_hash, depth, is_maximizing)
        if tt_key in self.transposition_table:
            return self.transposition_table[tt_key]
        
        # Terminal node
        if depth == 0 or game.game_over:
            score = self._evaluate_position(game, ai_color)
            self.transposition_table[tt_key] = score
            return score
        
        current_color = game.current_player
        valid_moves = game.get_valid_moves('black' if current_color == BLACK else 'white')
        
        if not valid_moves:
            # Pass
            game_copy = game.copy()
            game_copy.pass_turn('black' if current_color == BLACK else 'white')
            score = self._minimax(game_copy, depth - 1, alpha, beta, not is_maximizing, ai_color)
            self.transposition_table[tt_key] = score
            return score
        
        if is_maximizing:
            max_score = -float('inf')
            
            for move in self._order_moves(game, valid_moves, current_color):
                game_copy = game.copy()
                game_copy.make_move(move[0], move[1], 'black' if current_color == BLACK else 'white')
                
                score = self._minimax(game_copy, depth - 1, alpha, beta, False, ai_color)
                max_score = max(max_score, score)
                alpha = max(alpha, score)
                
                if beta <= alpha:
                    break  # Beta cutoff
            
            self.transposition_table[tt_key] = max_score
            return max_score
        else:
            min_score = float('inf')
            
            for move in self._order_moves(game, valid_moves, current_color):
                game_copy = game.copy()
                game_copy.make_move(move[0], move[1], 'black' if current_color == BLACK else 'white')
                
                score = self._minimax(game_copy, depth - 1, alpha, beta, True, ai_color)
                min_score = min(min_score, score)
                beta = min(beta, score)
                
                if beta <= alpha:
                    break  # Alpha cutoff
            
            self.transposition_table[tt_key] = min_score
            return min_score
    
    def _evaluate_position(self, game: OptimizedGoGame, ai_color: int) -> float:
        """Fast position evaluation"""
        score = 0.0
        opponent_color = game.get_opponent(ai_color)
        
        # Material (captures)
        score += (game.captures[ai_color] - game.captures[opponent_color]) * 10
        
        # Territory estimation (simplified but fast)
        ai_stones = np.sum(game.board == ai_color)
        opponent_stones = np.sum(game.board == opponent_color)
        score += (ai_stones - opponent_stones) * 1.0
        
        # Liberty advantage (simplified)
        ai_liberties = 0
        opponent_liberties = 0
        
        # Sample a few positions for liberty counting (faster than checking all)
        sample_positions = np.random.choice(game.size * game.size, 
                                          min(20, game.size * game.size), 
                                          replace=False)
        
        for pos in sample_positions:
            y, x = pos // game.size, pos % game.size
            if game.board[y, x] == ai_color:
                group = game.get_group(x, y)
                ai_liberties += game.count_liberties(group) / len(group)
            elif game.board[y, x] == opponent_color:
                group = game.get_group(x, y)
                opponent_liberties += game.count_liberties(group) / len(group)
        
        score += (ai_liberties - opponent_liberties) * 0.5
        
        return score