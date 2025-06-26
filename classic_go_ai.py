"""
Classic Go AI Engine using traditional algorithms
Optimized for speed and designed to be compatible with Codon compilation
"""

import numpy as np
from typing import List, Tuple, Optional, Set
import random
from collections import deque

class ClassicGoAI:
    """Fast classical Go AI using pattern matching and strategic heuristics"""
    
    def __init__(self, board_size: int = 19):
        self.board_size = board_size
        self.ko_point = None
        
        # Pre-computed neighbor offsets for faster access
        self.neighbors = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        
        # Pattern weights for evaluation
        self.pattern_weights = {
            'corner': 10,
            'edge': 5,
            'center': 3,
            'eye': 20,
            'capture': 15,
            'atari_save': 12,
            'atari_attack': 10,
            'connection': 8,
            'cut': 7,
            'territory': 4,
            'influence': 2
        }
        
    def get_legal_moves(self, board: np.ndarray, player: int) -> List[Tuple[int, int]]:
        """Get all legal moves for the current player"""
        legal_moves = []
        
        for i in range(self.board_size):
            for j in range(self.board_size):
                if board[i, j] == 0 and self.is_legal_move(board, i, j, player):
                    legal_moves.append((i, j))
        
        return legal_moves
    
    def is_legal_move(self, board: np.ndarray, row: int, col: int, player: int) -> bool:
        """Check if a move is legal (not suicide, not ko)"""
        if board[row, col] != 0:
            return False
            
        # Check ko rule
        if self.ko_point and (row, col) == self.ko_point:
            return False
        
        # Create temporary board with the move
        temp_board = board.copy()
        temp_board[row, col] = player
        
        # Check if the move would capture enemy stones
        opponent = 3 - player
        captured_any = False
        
        for dr, dc in self.neighbors:
            nr, nc = row + dr, col + dc
            if 0 <= nr < self.board_size and 0 <= nc < self.board_size:
                if temp_board[nr, nc] == opponent:
                    if not self.has_liberties(temp_board, nr, nc):
                        captured_any = True
                        self.remove_group(temp_board, nr, nc)
        
        # Check if the move would be suicide
        if not captured_any and not self.has_liberties(temp_board, row, col):
            return False
            
        return True
    
    def has_liberties(self, board: np.ndarray, row: int, col: int) -> bool:
        """Check if a stone or group has liberties"""
        color = board[row, col]
        if color == 0:
            return True
            
        visited = set()
        queue = deque([(row, col)])
        
        while queue:
            r, c = queue.popleft()
            if (r, c) in visited:
                continue
                
            visited.add((r, c))
            
            for dr, dc in self.neighbors:
                nr, nc = r + dr, c + dc
                if 0 <= nr < self.board_size and 0 <= nc < self.board_size:
                    if board[nr, nc] == 0:
                        return True
                    elif board[nr, nc] == color and (nr, nc) not in visited:
                        queue.append((nr, nc))
        
        return False
    
    def remove_group(self, board: np.ndarray, row: int, col: int) -> int:
        """Remove a captured group and return the number of stones removed"""
        color = board[row, col]
        if color == 0:
            return 0
            
        removed = 0
        queue = deque([(row, col)])
        
        while queue:
            r, c = queue.popleft()
            if board[r, c] == color:
                board[r, c] = 0
                removed += 1
                
                for dr, dc in self.neighbors:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < self.board_size and 0 <= nc < self.board_size:
                        if board[nr, nc] == color:
                            queue.append((nr, nc))
        
        return removed
    
    def count_liberties(self, board: np.ndarray, row: int, col: int) -> int:
        """Count the number of liberties for a group"""
        color = board[row, col]
        if color == 0:
            return 0
            
        visited = set()
        liberties = set()
        queue = deque([(row, col)])
        
        while queue:
            r, c = queue.popleft()
            if (r, c) in visited:
                continue
                
            visited.add((r, c))
            
            for dr, dc in self.neighbors:
                nr, nc = r + dr, c + dc
                if 0 <= nr < self.board_size and 0 <= nc < self.board_size:
                    if board[nr, nc] == 0:
                        liberties.add((nr, nc))
                    elif board[nr, nc] == color and (nr, nc) not in visited:
                        queue.append((nr, nc))
        
        return len(liberties)
    
    def is_eye(self, board: np.ndarray, row: int, col: int, player: int) -> bool:
        """Check if a position is an eye for the player"""
        if board[row, col] != 0:
            return False
            
        # Check all adjacent points
        for dr, dc in self.neighbors:
            nr, nc = row + dr, col + dc
            if 0 <= nr < self.board_size and 0 <= nc < self.board_size:
                if board[nr, nc] != player:
                    return False
            # Edge/corner cases - boundary counts as friendly
        
        # Check diagonal points (at least 3 out of 4 should be friendly)
        diagonals = [(-1, -1), (-1, 1), (1, -1), (1, 1)]
        friendly_diagonals = 0
        total_diagonals = 0
        
        for dr, dc in diagonals:
            nr, nc = row + dr, col + dc
            if 0 <= nr < self.board_size and 0 <= nc < self.board_size:
                total_diagonals += 1
                if board[nr, nc] == player:
                    friendly_diagonals += 1
        
        # For corner/edge, require all diagonals to be friendly
        # For center, require at least 3 out of 4
        if total_diagonals < 4:
            return friendly_diagonals == total_diagonals
        else:
            return friendly_diagonals >= 3
    
    def evaluate_move(self, board: np.ndarray, row: int, col: int, player: int) -> float:
        """Evaluate a potential move using various heuristics"""
        score = 0.0
        
        # Create temporary board with the move
        temp_board = board.copy()
        temp_board[row, col] = player
        opponent = 3 - player
        
        # Position value (corners > edges > center)
        if (row < 3 or row >= self.board_size - 3) and (col < 3 or col >= self.board_size - 3):
            score += self.pattern_weights['corner']
        elif row < 3 or row >= self.board_size - 3 or col < 3 or col >= self.board_size - 3:
            score += self.pattern_weights['edge']
        else:
            score += self.pattern_weights['center']
        
        # Check for captures
        captures = 0
        for dr, dc in self.neighbors:
            nr, nc = row + dr, col + dc
            if 0 <= nr < self.board_size and 0 <= nc < self.board_size:
                if temp_board[nr, nc] == opponent:
                    if self.count_liberties(board, nr, nc) == 1:
                        captures += self.count_group_size(board, nr, nc)
        
        score += captures * self.pattern_weights['capture']
        
        # Check if move saves own groups in atari
        saves = 0
        for dr, dc in self.neighbors:
            nr, nc = row + dr, col + dc
            if 0 <= nr < self.board_size and 0 <= nc < self.board_size:
                if board[nr, nc] == player:
                    if self.count_liberties(board, nr, nc) == 1:
                        saves += self.count_group_size(board, nr, nc)
        
        score += saves * self.pattern_weights['atari_save']
        
        # Check if move puts opponent groups in atari
        ataris = 0
        for dr, dc in self.neighbors:
            nr, nc = row + dr, col + dc
            if 0 <= nr < self.board_size and 0 <= nc < self.board_size:
                if temp_board[nr, nc] == opponent:
                    lib_count = self.count_liberties(temp_board, nr, nc)
                    if lib_count == 1:
                        ataris += self.count_group_size(temp_board, nr, nc)
        
        score += ataris * self.pattern_weights['atari_attack']
        
        # Connection value (connecting friendly stones)
        connections = 0
        for dr, dc in self.neighbors:
            nr, nc = row + dr, col + dc
            if 0 <= nr < self.board_size and 0 <= nc < self.board_size:
                if board[nr, nc] == player:
                    connections += 1
        
        score += connections * self.pattern_weights['connection']
        
        # Influence (empty points near the move)
        influence = 0
        for dr in range(-2, 3):
            for dc in range(-2, 3):
                nr, nc = row + dr, col + dc
                if 0 <= nr < self.board_size and 0 <= nc < self.board_size:
                    if board[nr, nc] == 0:
                        influence += 1.0 / (abs(dr) + abs(dc) + 1)
        
        score += influence * self.pattern_weights['influence']
        
        # Penalty for playing in own eyes
        if self.is_eye(board, row, col, player):
            score -= self.pattern_weights['eye'] * 2
        
        return score
    
    def get_group(self, board: np.ndarray, row: int, col: int) -> Set[Tuple[int, int]]:
        """Get all stones in the same group"""
        color = board[row, col]
        if color == 0:
            return set()
            
        visited = set()
        queue = deque([(row, col)])
        group = set()
        
        while queue:
            r, c = queue.popleft()
            if (r, c) in visited:
                continue
                
            visited.add((r, c))
            group.add((r, c))
            
            for dr, dc in self.neighbors:
                nr, nc = r + dr, c + dc
                if 0 <= nr < self.board_size and 0 <= nc < self.board_size:
                    if board[nr, nc] == color and (nr, nc) not in visited:
                        queue.append((nr, nc))
        
        return group
    
    def count_group_size(self, board: np.ndarray, row: int, col: int) -> int:
        """Count the size of a group"""
        color = board[row, col]
        if color == 0:
            return 0
            
        visited = set()
        queue = deque([(row, col)])
        size = 0
        
        while queue:
            r, c = queue.popleft()
            if (r, c) in visited:
                continue
                
            visited.add((r, c))
            size += 1
            
            for dr, dc in self.neighbors:
                nr, nc = r + dr, c + dc
                if 0 <= nr < self.board_size and 0 <= nc < self.board_size:
                    if board[nr, nc] == color and (nr, nc) not in visited:
                        queue.append((nr, nc))
        
        return size
    
    def get_best_move(self, board: np.ndarray, player: int, depth: int = 1) -> Optional[Tuple[int, int]]:
        """Get the best move using evaluation function"""
        legal_moves = self.get_legal_moves(board, player)
        
        if not legal_moves:
            return None
        
        # Evaluate all legal moves
        move_scores = []
        for move in legal_moves:
            score = self.evaluate_move(board, move[0], move[1], player)
            move_scores.append((score, move))
        
        # Sort by score and add some randomness to top moves
        move_scores.sort(reverse=True)
        
        # Consider top N moves with some randomness
        top_n = min(5, len(move_scores))
        if top_n > 1:
            # Add small random factor to scores
            top_moves = [(score + random.uniform(-1, 1), move) for score, move in move_scores[:top_n]]
            top_moves.sort(reverse=True)
            return top_moves[0][1]
        else:
            return move_scores[0][1]
    
    def evaluate_position(self, board: np.ndarray, player: int) -> float:
        """Evaluate the current board position for a player"""
        opponent = 3 - player
        score = 0.0
        
        # Territory estimation
        black_territory, white_territory = self.estimate_territory(board)
        if player == 1:  # Black
            score += (black_territory - white_territory) * self.pattern_weights['territory']
        else:  # White
            score += (white_territory - black_territory) * self.pattern_weights['territory']
        
        # Count stones on board
        player_stones = np.sum(board == player)
        opponent_stones = np.sum(board == opponent)
        score += (player_stones - opponent_stones) * 2
        
        # Evaluate each stone's position value
        position_score = 0
        for i in range(self.board_size):
            for j in range(self.board_size):
                if board[i, j] == player:
                    # Corner stones are valuable
                    if (i < 3 or i >= self.board_size - 3) and (j < 3 or j >= self.board_size - 3):
                        position_score += 3
                    # Edge stones are moderately valuable
                    elif i < 3 or i >= self.board_size - 3 or j < 3 or j >= self.board_size - 3:
                        position_score += 2
                    else:
                        position_score += 1
                elif board[i, j] == opponent:
                    if (i < 3 or i >= self.board_size - 3) and (j < 3 or j >= self.board_size - 3):
                        position_score -= 3
                    elif i < 3 or i >= self.board_size - 3 or j < 3 or j >= self.board_size - 3:
                        position_score -= 2
                    else:
                        position_score -= 1
        
        score += position_score
        
        # Groups in atari (endangered groups)
        groups_in_atari = 0
        opponent_groups_in_atari = 0
        checked = set()
        
        for i in range(self.board_size):
            for j in range(self.board_size):
                if (i, j) not in checked and board[i, j] != 0:
                    group = self.get_group(board, i, j)
                    checked.update(group)
                    liberties = self.count_liberties(board, i, j)
                    
                    if liberties == 1:
                        if board[i, j] == player:
                            groups_in_atari += len(group)
                        else:
                            opponent_groups_in_atari += len(group)
        
        score -= groups_in_atari * self.pattern_weights['atari_save']
        score += opponent_groups_in_atari * self.pattern_weights['atari_attack']
        
        return score
    
    def estimate_territory(self, board: np.ndarray) -> Tuple[int, int]:
        """Estimate territory for both players"""
        black_territory = 0
        white_territory = 0
        
        # Simple flood fill to find territories
        visited = set()
        
        for i in range(self.board_size):
            for j in range(self.board_size):
                if board[i, j] == 0 and (i, j) not in visited:
                    territory, owner = self.flood_fill_territory(board, i, j, visited)
                    if owner == 1:
                        black_territory += territory
                    elif owner == 2:
                        white_territory += territory
        
        return black_territory, white_territory
    
    def flood_fill_territory(self, board: np.ndarray, start_row: int, start_col: int, 
                           visited: Set[Tuple[int, int]]) -> Tuple[int, int]:
        """Flood fill to find territory and its owner"""
        queue = deque([(start_row, start_col)])
        territory_points = []
        bordering_colors = set()
        
        while queue:
            row, col = queue.popleft()
            if (row, col) in visited:
                continue
                
            visited.add((row, col))
            territory_points.append((row, col))
            
            for dr, dc in self.neighbors:
                nr, nc = row + dr, col + dc
                if 0 <= nr < self.board_size and 0 <= nc < self.board_size:
                    if board[nr, nc] == 0 and (nr, nc) not in visited:
                        queue.append((nr, nc))
                    elif board[nr, nc] != 0:
                        bordering_colors.add(board[nr, nc])
        
        # Determine owner
        if len(bordering_colors) == 1:
            owner = bordering_colors.pop()
        else:
            owner = 0  # Neutral territory
        
        return len(territory_points), owner


class MCTSNode:
    """Node for Monte Carlo Tree Search"""
    
    def __init__(self, board: np.ndarray, player: int, move: Optional[Tuple[int, int]] = None, 
                 parent: Optional['MCTSNode'] = None):
        self.board = board.copy()
        self.player = player
        self.move = move
        self.parent = parent
        self.children = []
        self.visits = 0
        self.wins = 0.0
        self.untried_moves = None
        
    def is_fully_expanded(self) -> bool:
        return self.untried_moves is not None and len(self.untried_moves) == 0
    
    def best_child(self, exploration_weight: float = 1.4) -> 'MCTSNode':
        """Select best child using UCB1 formula"""
        import math
        
        best_score = -float('inf')
        best_child = None
        
        for child in self.children:
            if child.visits == 0:
                return child
            
            exploitation = child.wins / child.visits
            exploration = exploration_weight * math.sqrt(math.log(self.visits) / child.visits)
            score = exploitation + exploration
            
            if score > best_score:
                best_score = score
                best_child = child
        
        return best_child


class MCTSGoAI(ClassicGoAI):
    """Monte Carlo Tree Search based Go AI"""
    
    def __init__(self, board_size: int = 19, simulations: int = 1000):
        super().__init__(board_size)
        self.simulations = simulations
        
    def get_best_move_mcts(self, board: np.ndarray, player: int) -> Optional[Tuple[int, int]]:
        """Get best move using MCTS"""
        root = MCTSNode(board, player)
        
        for _ in range(self.simulations):
            node = self.select(root)
            if node.visits > 0 and not self.is_terminal(node.board):
                node = self.expand(node)
            
            result = self.simulate(node)
            self.backpropagate(node, result)
        
        # Select most visited child
        if not root.children:
            return None
            
        best_child = max(root.children, key=lambda c: c.visits)
        return best_child.move
    
    def select(self, node: MCTSNode) -> MCTSNode:
        """Select node to expand using tree policy"""
        while not self.is_terminal(node.board):
            if not node.is_fully_expanded():
                return node
            else:
                node = node.best_child()
        return node
    
    def expand(self, node: MCTSNode) -> MCTSNode:
        """Expand node by adding a new child"""
        if node.untried_moves is None:
            node.untried_moves = self.get_legal_moves(node.board, node.player)
            random.shuffle(node.untried_moves)
        
        if node.untried_moves:
            move = node.untried_moves.pop()
            next_board = node.board.copy()
            next_board[move[0], move[1]] = node.player
            
            # Handle captures
            opponent = 3 - node.player
            for dr, dc in self.neighbors:
                nr, nc = move[0] + dr, move[1] + dc
                if 0 <= nr < self.board_size and 0 <= nc < self.board_size:
                    if next_board[nr, nc] == opponent:
                        if not self.has_liberties(next_board, nr, nc):
                            self.remove_group(next_board, nr, nc)
            
            child = MCTSNode(next_board, 3 - node.player, move, node)
            node.children.append(child)
            return child
        
        return node
    
    def simulate(self, node: MCTSNode) -> float:
        """Run a random playout from the node"""
        board = node.board.copy()
        player = node.player
        passes = 0
        max_moves = self.board_size * self.board_size
        
        for _ in range(max_moves):
            legal_moves = self.get_legal_moves(board, player)
            
            if not legal_moves:
                passes += 1
                if passes >= 2:
                    break
            else:
                passes = 0
                move = random.choice(legal_moves)
                board[move[0], move[1]] = player
                
                # Handle captures
                opponent = 3 - player
                for dr, dc in self.neighbors:
                    nr, nc = move[0] + dr, move[1] + dc
                    if 0 <= nr < self.board_size and 0 <= nc < self.board_size:
                        if board[nr, nc] == opponent:
                            if not self.has_liberties(board, nr, nc):
                                self.remove_group(board, nr, nc)
            
            player = 3 - player
        
        # Evaluate final position
        black_territory, white_territory = self.estimate_territory(board)
        black_stones = np.sum(board == 1)
        white_stones = np.sum(board == 2)
        
        black_score = black_territory + black_stones
        white_score = white_territory + white_stones + 7.5  # Komi
        
        # Return result from the perspective of the original player
        if node.player == 1:  # Black
            return 1.0 if black_score > white_score else 0.0
        else:  # White
            return 1.0 if white_score > black_score else 0.0
    
    def backpropagate(self, node: MCTSNode, result: float) -> None:
        """Backpropagate the result up the tree"""
        while node is not None:
            node.visits += 1
            node.wins += result
            result = 1 - result  # Flip for opponent
            node = node.parent
    
    def is_terminal(self, board: np.ndarray) -> bool:
        """Check if the game is over (simplified)"""
        # Game is over if no legal moves for either player
        black_moves = self.get_legal_moves(board, 1)
        white_moves = self.get_legal_moves(board, 2)
        return len(black_moves) == 0 and len(white_moves) == 0