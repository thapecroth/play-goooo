"""
Monte Carlo Tree Search (MCTS) implementation for Go
"""
import numpy as np
import math
import random
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import time
from optimized_go import OptimizedGoGame, BLACK, WHITE, EMPTY

@dataclass
class MCTSStats:
    """Statistics for MCTS visualization"""
    visits: Dict[Tuple[int, int], int]
    win_rates: Dict[Tuple[int, int], float]
    best_move: Optional[Tuple[int, int]]
    total_simulations: int
    thinking_time: float


class MCTSNode:
    """Node in the MCTS tree"""
    
    def __init__(self, game_state: OptimizedGoGame, parent=None, move=None, color=None):
        self.game_state = game_state
        self.parent = parent
        self.move = move  # The move that led to this state
        self.color = color  # The color that made the move
        
        self.visits = 0
        self.wins = 0.0
        self.children = []
        self.untried_moves = None
        self._untried_moves_initialized = False
        
    def get_untried_moves(self) -> List[Tuple[int, int]]:
        """Get list of untried moves from this position"""
        if not self._untried_moves_initialized:
            color_str = 'black' if self.game_state.current_player == BLACK else 'white'
            self.untried_moves = self.game_state.get_valid_moves(color_str)
            self._untried_moves_initialized = True
        return self.untried_moves
    
    def is_fully_expanded(self) -> bool:
        """Check if all moves have been tried"""
        return len(self.get_untried_moves()) == 0
    
    def is_terminal(self) -> bool:
        """Check if this is a terminal node"""
        return self.game_state.game_over
    
    def get_best_child(self, exploration_constant: float) -> 'MCTSNode':
        """Select best child using UCB1 formula"""
        best_value = -float('inf')
        best_child = None
        
        for child in self.children:
            # UCB1 formula: exploitation + exploration
            if child.visits == 0:
                value = float('inf')
            else:
                exploitation = child.wins / child.visits
                exploration = exploration_constant * math.sqrt(math.log(self.visits) / child.visits)
                value = exploitation + exploration
            
            if value > best_value:
                best_value = value
                best_child = child
        
        return best_child
    
    def add_child(self, move: Tuple[int, int], game_state: OptimizedGoGame) -> 'MCTSNode':
        """Add a child node"""
        child = MCTSNode(
            game_state=game_state,
            parent=self,
            move=move,
            color=self.game_state.current_player
        )
        self.children.append(child)
        self.untried_moves.remove(move)
        return child
    
    def update(self, result: float):
        """Update node statistics"""
        self.visits += 1
        self.wins += result
        
    def get_win_rate(self) -> float:
        """Get win rate for this node"""
        if self.visits == 0:
            return 0.0
        return self.wins / self.visits


class MCTSPlayer:
    """MCTS-based Go player"""
    
    def __init__(self, simulations: int = 1000, exploration_constant: float = 1.414, 
                 time_limit: Optional[float] = None, use_rave: bool = False):
        """
        Initialize MCTS player
        
        Args:
            simulations: Number of simulations per move
            exploration_constant: UCB1 exploration parameter (higher = more exploration)
            time_limit: Optional time limit per move in seconds
            use_rave: Whether to use RAVE (Rapid Action Value Estimation)
        """
        self.simulations = simulations
        self.exploration_constant = exploration_constant
        self.time_limit = time_limit
        self.use_rave = use_rave
        self.stats = None  # Last move statistics
        
    def get_move(self, game: OptimizedGoGame, color: str) -> Optional[Tuple[int, int]]:
        """Get best move using MCTS"""
        start_time = time.time()
        color_int = game.get_color_int(color)
        
        # Create root node
        root = MCTSNode(game.copy())
        
        # Statistics tracking
        move_visits = {}
        move_wins = {}
        
        # Run simulations
        simulations_run = 0
        while simulations_run < self.simulations:
            # Check time limit
            if self.time_limit and (time.time() - start_time) > self.time_limit:
                break
                
            # Run one simulation
            self._run_simulation(root)
            simulations_run += 1
        
        # Collect statistics from children
        for child in root.children:
            if child.move:
                move_visits[child.move] = child.visits
                move_wins[child.move] = child.wins
        
        # Choose best move (most visited)
        if not root.children:
            return None
            
        best_child = max(root.children, key=lambda c: c.visits)
        best_move = best_child.move
        
        # Store statistics
        self.stats = MCTSStats(
            visits=move_visits,
            win_rates={move: wins/visits if visits > 0 else 0 
                      for move, (wins, visits) in 
                      zip(move_wins.keys(), zip(move_wins.values(), move_visits.values()))},
            best_move=best_move,
            total_simulations=simulations_run,
            thinking_time=time.time() - start_time
        )
        
        return best_move
    
    def _run_simulation(self, root: MCTSNode) -> float:
        """Run a single MCTS simulation"""
        node = root
        
        # Selection phase - traverse tree using UCB1
        while not node.is_terminal() and node.is_fully_expanded():
            node = node.get_best_child(self.exploration_constant)
        
        # Expansion phase - add new child if not terminal
        if not node.is_terminal() and not node.is_fully_expanded():
            # Choose random untried move
            untried_moves = node.get_untried_moves()
            if untried_moves:
                move = random.choice(untried_moves)
                
                # Make move on game copy
                game_copy = node.game_state.copy()
                color_str = 'black' if game_copy.current_player == BLACK else 'white'
                
                if move is None:  # Pass move
                    game_copy.pass_turn(color_str)
                else:
                    game_copy.make_move(move[0], move[1], color_str)
                
                # Add child node
                node = node.add_child(move, game_copy)
        
        # Simulation phase - play out random game
        result = self._simulate_random_game(node.game_state.copy(), node.color)
        
        # Backpropagation phase - update statistics
        while node is not None:
            # Result is from perspective of node.color
            node_result = result if node.color == root.game_state.current_player else 1 - result
            node.update(node_result)
            node = node.parent
        
        return result
    
    def _simulate_random_game(self, game: OptimizedGoGame, perspective_color: int) -> float:
        """Simulate a random game to completion"""
        consecutive_passes = 0
        max_moves = 300  # Prevent infinite games
        moves_played = 0
        
        while not game.game_over and moves_played < max_moves:
            color_str = 'black' if game.current_player == BLACK else 'white'
            valid_moves = game.get_valid_moves(color_str)
            
            if not valid_moves or (len(valid_moves) == 1 and valid_moves[0] is None):
                # Must pass
                game.pass_turn(color_str)
                consecutive_passes += 1
                if consecutive_passes >= 2:
                    break
            else:
                # Use simple heuristics for playouts
                move = self._select_playout_move(game, valid_moves, game.current_player)
                
                if move is None:
                    game.pass_turn(color_str)
                    consecutive_passes += 1
                else:
                    game.make_move(move[0], move[1], color_str)
                    consecutive_passes = 0
            
            moves_played += 1
        
        # Evaluate final position
        return self._evaluate_final_position(game, perspective_color)
    
    def _select_playout_move(self, game: OptimizedGoGame, valid_moves: List[Tuple[int, int]], 
                            color: int) -> Optional[Tuple[int, int]]:
        """Select move during playout (with simple heuristics)"""
        if random.random() < 0.1:  # 10% chance to pass
            return None
            
        # Filter out obviously bad moves (self-atari of large groups)
        good_moves = []
        for move in valid_moves:
            if move is not None:
                # Quick check: would this move put our stones in atari?
                game_copy = game.copy()
                color_str = 'black' if color == BLACK else 'white'
                game_copy.make_move(move[0], move[1], color_str)
                
                # Check if we created a group in atari
                group = game_copy.get_group(move[0], move[1])
                if len(group) > 3 and game_copy.count_liberties(group) == 1:
                    continue  # Skip this move
                    
                good_moves.append(move)
        
        if not good_moves:
            good_moves = valid_moves
        
        # Prefer moves near existing stones
        if random.random() < 0.7 and len(good_moves) > 1:
            scored_moves = []
            for move in good_moves:
                if move is not None:
                    score = 0
                    # Count nearby stones
                    for dx in [-1, 0, 1]:
                        for dy in [-1, 0, 1]:
                            nx, ny = move[0] + dx, move[1] + dy
                            if 0 <= nx < game.size and 0 <= ny < game.size:
                                if game.board[ny, nx] != EMPTY:
                                    score += 1
                    scored_moves.append((score, move))
            
            # Weight selection by score
            scored_moves.sort(reverse=True)
            # Take top 50% of moves
            good_moves = [move for _, move in scored_moves[:len(scored_moves)//2 + 1]]
        
        return random.choice(good_moves)
    
    def _evaluate_final_position(self, game: OptimizedGoGame, perspective_color: int) -> float:
        """Evaluate final game position"""
        # Simple evaluation: count territory and captures
        black_score = game.captures[BLACK]
        white_score = game.captures[WHITE]
        
        # Count stones on board
        black_stones = np.sum(game.board == BLACK)
        white_stones = np.sum(game.board == WHITE)
        
        black_score += black_stones
        white_score += white_stones
        
        # Simple territory estimation using flood fill
        territory = self._estimate_territory_fast(game)
        black_score += territory[BLACK]
        white_score += territory[WHITE]
        
        # Add komi
        white_score += 6.5
        
        # Return win probability from perspective color
        if perspective_color == BLACK:
            return 1.0 if black_score > white_score else 0.0
        else:
            return 1.0 if white_score > black_score else 0.0
    
    def _estimate_territory_fast(self, game: OptimizedGoGame) -> Dict[int, int]:
        """Fast territory estimation"""
        territory = {BLACK: 0, WHITE: 0}
        visited = set()
        
        for y in range(game.size):
            for x in range(game.size):
                if game.board[y, x] == EMPTY and (x, y) not in visited:
                    # Flood fill to find territory
                    region, border_colors = self._flood_fill_territory(game, x, y, visited)
                    
                    # Assign territory if surrounded by one color
                    if len(border_colors) == 1:
                        color = border_colors.pop()
                        territory[color] += len(region)
        
        return territory
    
    def _flood_fill_territory(self, game: OptimizedGoGame, start_x: int, start_y: int, 
                             visited: set) -> Tuple[set, set]:
        """Flood fill to find empty region and bordering colors"""
        region = set()
        border_colors = set()
        stack = [(start_x, start_y)]
        
        while stack:
            x, y = stack.pop()
            
            if (x, y) in visited:
                continue
                
            if game.board[y, x] == EMPTY:
                visited.add((x, y))
                region.add((x, y))
                
                # Check neighbors
                for nx, ny in game._get_neighbors_numba(x, y, game.size):
                    if (nx, ny) not in visited:
                        stack.append((nx, ny))
            else:
                # Found a border stone
                border_colors.add(game.board[y, x])
        
        return region, border_colors
    
    def get_stats(self) -> Optional[MCTSStats]:
        """Get statistics from last move"""
        return self.stats
    
    def set_parameters(self, simulations: int = None, exploration_constant: float = None,
                      time_limit: float = None):
        """Update MCTS parameters"""
        if simulations is not None:
            self.simulations = simulations
        if exploration_constant is not None:
            self.exploration_constant = exploration_constant
        if time_limit is not None:
            self.time_limit = time_limit