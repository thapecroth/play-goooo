#!/usr/bin/env python3
"""
Optimized Go AI implementation specifically for Codon compilation.
Removes type annotations for compatibility and fixes initialization order.
"""

import math

class Move:
    x: int
    y: int
    score: float
    
    def __init__(self, x: int, y: int, score: float = 0.0):
        self.x = x
        self.y = y
        self.score = score

class Position:
    x: int
    y: int
    
    def __init__(self, x: int, y: int):
        self.x = x
        self.y = y

class GoAIOptimized:
    size: int
    max_depth: int
    weight_territory: float
    weight_captures: float
    weight_liberties: float
    weight_influence: float
    board: list[list[int]]
    visited: list[list[bool]]
    EMPTY: int
    BLACK: int
    WHITE: int
    neighbor_offsets: list[tuple[int, int]]
    transposition_table: dict[int, float]
    
    def __init__(self, board_size: int = 9):
        self.size = board_size
        self.max_depth = 3
        
        # Evaluation weights
        self.weight_territory = 1.0
        self.weight_captures = 10.0
        self.weight_liberties = 0.5
        self.weight_influence = 0.3
        
        # Pre-allocate arrays for better performance
        self.board = [[0 for _ in range(board_size)] for _ in range(board_size)]
        self.visited = [[False for _ in range(board_size)] for _ in range(board_size)]
        
        # Stone values: 0 = empty, 1 = black, 2 = white
        self.EMPTY = 0
        self.BLACK = 1
        self.WHITE = 2
        
        # Pre-computed neighbor offsets
        self.neighbor_offsets = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        
        # Transposition table for memoization
        self.transposition_table = {}
        
    def get_best_move(self, board: list[list[int]], color: int, captures_black: int, captures_white: int) -> tuple[int, int] | None:
        """Find the best move for the given color."""
        self.board = board
        valid_moves = self.get_valid_moves(color)
        
        if not valid_moves:
            return None
            
        best_move = None
        best_score = -1000000.0
        
        # Sort moves by heuristic for better alpha-beta pruning
        moves_with_scores = []
        for move in valid_moves:
            heuristic_score = self.evaluate_move_heuristic(move.x, move.y, color)
            moves_with_scores.append((move, heuristic_score))
        
        # Sort by heuristic score (descending)
        moves_with_scores.sort(key=lambda x: x[1], reverse=True)
        
        alpha = -1000000.0
        beta = 1000000.0
        
        for move, _ in moves_with_scores:
            score = self.evaluate_move(move.x, move.y, color, captures_black, captures_white, alpha, beta)
            if score > best_score:
                best_score = score
                best_move = move
            alpha = max(alpha, best_score)
            
        if best_score < -50.0:
            return None
            
        return (best_move.x, best_move.y) if best_move else None
    
    def evaluate_move_heuristic(self, x: int, y: int, color: int) -> float:
        """Quick heuristic evaluation for move ordering."""
        score = 0.0
        
        # Prefer center positions
        center = float(self.size - 1) / 2.0
        distance_to_center = abs(x - center) + abs(y - center)
        score += 10.0 / (distance_to_center + 1.0)
        
        # Prefer star points on larger boards
        if self.size >= 9:
            star_points = self.get_star_points()
            for sp in star_points:
                if x == sp[0] and y == sp[1]:
                    score += 5.0
        
        # Check for potential captures
        opposite_color = 3 - color
        for dx, dy in self.neighbor_offsets:
            nx = x + dx
            ny = y + dy
            if 0 <= nx < self.size and 0 <= ny < self.size:
                if self.board[ny][nx] == opposite_color:
                    # Check if this group has few liberties
                    liberties = self.count_group_liberties_fast(nx, ny)
                    if liberties == 1:
                        score += 20.0
                    elif liberties == 2:
                        score += 10.0
        
        return score
    
    def get_star_points(self) -> list[tuple[int, int]]:
        """Get star points for the current board size."""
        if self.size == 9:
            return [(2, 2), (2, 6), (6, 2), (6, 6), (4, 4)]
        elif self.size == 13:
            return [(3, 3), (3, 9), (9, 3), (9, 9), (6, 6)]
        elif self.size == 19:
            return [(3, 3), (3, 9), (3, 15), (9, 3), (9, 9), (9, 15), (15, 3), (15, 9), (15, 15)]
        return []
    
    def evaluate_move(self, x: int, y: int, color: int, captures_black: int, captures_white: int, alpha: float, beta: float) -> float:
        """Evaluate a move using minimax with alpha-beta pruning."""
        # Make the move
        original_stone = self.board[y][x]
        self.board[y][x] = color
        
        # Simulate captures
        captured = self.simulate_captures(3 - color)
        new_captures_black = captures_black
        new_captures_white = captures_white
        
        if color == self.BLACK:
            new_captures_black += len(captured)
        else:
            new_captures_white += len(captured)
        
        # Check if the move is valid (has liberties after captures)
        if not self.has_liberties_fast(x, y):
            self.board[y][x] = original_stone
            # Restore captured stones
            for pos in captured:
                self.board[pos.y][pos.x] = 3 - color
            return -1000000.0
        
        # Calculate board hash for transposition table
        board_hash = self.calculate_board_hash()
        
        # Check transposition table
        if board_hash in self.transposition_table:
            score = self.transposition_table[board_hash]
        else:
            # Evaluate using minimax
            score = self.minimax(self.max_depth - 1, False, color, new_captures_black, 
                               new_captures_white, alpha, beta)
            self.transposition_table[board_hash] = score
        
        # Restore the board
        self.board[y][x] = original_stone
        for pos in captured:
            self.board[pos.y][pos.x] = 3 - color
            
        return score
    
    def minimax(self, depth: int, is_maximizing: bool, ai_color: int, captures_black: int, captures_white: int, alpha: float, beta: float) -> float:
        """Minimax algorithm with alpha-beta pruning."""
        if depth == 0:
            return self.evaluate_position(ai_color, captures_black, captures_white)
        
        current_color = ai_color if is_maximizing else 3 - ai_color
        valid_moves = self.get_valid_moves(current_color)
        
        if not valid_moves:
            # Pass
            return self.minimax(depth - 1, not is_maximizing, ai_color, 
                              captures_black, captures_white, alpha, beta)
        
        if is_maximizing:
            max_score = -1000000.0
            for move in valid_moves:
                score = self.evaluate_move_minimax(move.x, move.y, current_color, depth, 
                                                 ai_color, captures_black, captures_white, 
                                                 alpha, beta, False)
                max_score = max(max_score, score)
                alpha = max(alpha, score)
                if beta <= alpha:
                    break
            return max_score
        else:
            min_score = 1000000.0
            for move in valid_moves:
                score = self.evaluate_move_minimax(move.x, move.y, current_color, depth, 
                                                 ai_color, captures_black, captures_white, 
                                                 alpha, beta, True)
                min_score = min(min_score, score)
                beta = min(beta, score)
                if beta <= alpha:
                    break
            return min_score
    
    def evaluate_move_minimax(self, x: int, y: int, color: int, depth: int, ai_color: int, captures_black: int, captures_white: int, alpha: float, beta: float, next_is_max: bool) -> float:
        """Evaluate a move within minimax."""
        original_stone = self.board[y][x]
        self.board[y][x] = color
        
        captured = self.simulate_captures(3 - color)
        new_captures_black = captures_black
        new_captures_white = captures_white
        
        if color == self.BLACK:
            new_captures_black += len(captured)
        else:
            new_captures_white += len(captured)
        
        score = self.minimax(depth - 1, next_is_max, ai_color, new_captures_black, 
                           new_captures_white, alpha, beta)
        
        self.board[y][x] = original_stone
        for pos in captured:
            self.board[pos.y][pos.x] = 3 - color
            
        return score
    
    def evaluate_position(self, ai_color: int, captures_black: int, captures_white: int) -> float:
        """Evaluate the current board position."""
        score = 0.0
        
        # Capture difference
        if ai_color == self.BLACK:
            score += float(captures_black - captures_white) * self.weight_captures
        else:
            score += float(captures_white - captures_black) * self.weight_captures
        
        # Territory estimation
        territory_score = self.estimate_territory_fast(ai_color)
        score += float(territory_score) * self.weight_territory
        
        # Liberty difference
        liberty_score = self.count_all_liberties(ai_color) - self.count_all_liberties(3 - ai_color)
        score += float(liberty_score) * self.weight_liberties
        
        # Influence
        influence_score = self.calculate_influence_fast(ai_color)
        score += influence_score * self.weight_influence
        
        return score
    
    def get_valid_moves(self, color: int) -> list[Move]:
        """Get all valid moves for the given color."""
        moves = []
        
        for y in range(self.size):
            for x in range(self.size):
                if self.board[y][x] == self.EMPTY:
                    # Quick check: if the point has friendly neighbors, it's likely valid
                    has_friendly_neighbor = False
                    has_liberty = False
                    
                    for dx, dy in self.neighbor_offsets:
                        nx = x + dx
                        ny = y + dy
                        if 0 <= nx < self.size and 0 <= ny < self.size:
                            if self.board[ny][nx] == self.EMPTY:
                                has_liberty = True
                            elif self.board[ny][nx] == color:
                                has_friendly_neighbor = True
                    
                    # If it has a liberty or friendly neighbor, it might be valid
                    if has_liberty or has_friendly_neighbor:
                        moves.append(Move(x, y))
                    else:
                        # Check if it captures enemy stones
                        self.board[y][x] = color
                        captured = self.simulate_captures(3 - color)
                        if captured and self.has_liberties_fast(x, y):
                            moves.append(Move(x, y))
                        self.board[y][x] = self.EMPTY
                        for pos in captured:
                            self.board[pos.y][pos.x] = 3 - color
        
        return moves
    
    def simulate_captures(self, color: int) -> list[Position]:
        """Simulate capturing stones of the given color."""
        captured = []
        
        for y in range(self.size):
            for x in range(self.size):
                if self.board[y][x] == color and not self.has_liberties_fast(x, y):
                    group = self.get_group_fast(x, y)
                    for pos in group:
                        self.board[pos.y][pos.x] = self.EMPTY
                        captured.append(pos)
        
        return captured
    
    def has_liberties_fast(self, x: int, y: int) -> bool:
        """Fast check if a group has liberties."""
        color = self.board[y][x]
        if color == self.EMPTY:
            return True
            
        # Reset visited array
        for row in self.visited:
            for i in range(self.size):
                row[i] = False
                
        return self._has_liberties_dfs(x, y, color)
    
    def _has_liberties_dfs(self, x: int, y: int, color: int) -> bool:
        """DFS helper for liberty checking."""
        if self.visited[y][x]:
            return False
        self.visited[y][x] = True
        
        for dx, dy in self.neighbor_offsets:
            nx = x + dx
            ny = y + dy
            if 0 <= nx < self.size and 0 <= ny < self.size:
                if self.board[ny][nx] == self.EMPTY:
                    return True
                elif self.board[ny][nx] == color and not self.visited[ny][nx]:
                    if self._has_liberties_dfs(nx, ny, color):
                        return True
        
        return False
    
    def get_group_fast(self, x: int, y: int) -> list[Position]:
        """Get all stones in a group using iterative approach."""
        color = self.board[y][x]
        if color == self.EMPTY:
            return []
            
        group = []
        stack = [Position(x, y)]
        
        # Reset visited array
        for row in self.visited:
            for i in range(self.size):
                row[i] = False
        
        while stack:
            pos = stack.pop()
            if self.visited[pos.y][pos.x]:
                continue
                
            self.visited[pos.y][pos.x] = True
            group.append(pos)
            
            for dx, dy in self.neighbor_offsets:
                nx = pos.x + dx
                ny = pos.y + dy
                if 0 <= nx < self.size and 0 <= ny < self.size:
                    if self.board[ny][nx] == color and not self.visited[ny][nx]:
                        stack.append(Position(nx, ny))
        
        return group
    
    def count_group_liberties_fast(self, x: int, y: int) -> int:
        """Count liberties for a group."""
        color = self.board[y][x]
        if color == self.EMPTY:
            return 0
            
        group = self.get_group_fast(x, y)
        liberty_set = set()
        
        for pos in group:
            for dx, dy in self.neighbor_offsets:
                nx = pos.x + dx
                ny = pos.y + dy
                if 0 <= nx < self.size and 0 <= ny < self.size:
                    if self.board[ny][nx] == self.EMPTY:
                        liberty_set.add((nx, ny))
        
        return len(liberty_set)
    
    def count_all_liberties(self, color: int) -> int:
        """Count total liberties for all groups of a color."""
        total_liberties = 0
        counted = set()
        
        for y in range(self.size):
            for x in range(self.size):
                if self.board[y][x] == color and (x, y) not in counted:
                    group = self.get_group_fast(x, y)
                    liberties = self.count_group_liberties_fast(x, y)
                    total_liberties += liberties
                    
                    for pos in group:
                        counted.add((pos.x, pos.y))
        
        return total_liberties
    
    def estimate_territory_fast(self, color: int) -> int:
        """Fast territory estimation."""
        territory = 0
        
        for y in range(self.size):
            for x in range(self.size):
                if self.board[y][x] == self.EMPTY:
                    influence = self.get_point_influence_fast(x, y)
                    if influence[color - 1] > influence[2 - color] * 1.5:
                        territory += 1
                    elif influence[2 - color] > influence[color - 1] * 1.5:
                        territory -= 1
        
        return territory
    
    def get_point_influence_fast(self, x: int, y: int) -> tuple[float, float]:
        """Calculate influence at a point (black_influence, white_influence)."""
        black_influence = 0.0
        white_influence = 0.0
        max_distance = 4
        
        for dy in range(-max_distance, max_distance + 1):
            for dx in range(-max_distance, max_distance + 1):
                nx = x + dx
                ny = y + dy
                if 0 <= nx < self.size and 0 <= ny < self.size:
                    stone = self.board[ny][nx]
                    if stone != self.EMPTY:
                        distance = abs(dx) + abs(dy)
                        influence = 1.0 / float(distance + 1)
                        if stone == self.BLACK:
                            black_influence += influence
                        else:
                            white_influence += influence
        
        return (black_influence, white_influence)
    
    def calculate_influence_fast(self, color: int) -> float:
        """Calculate total influence for a color."""
        influence = 0.0
        
        for y in range(self.size):
            for x in range(self.size):
                if self.board[y][x] == color:
                    influence += self.get_stone_value(x, y)
        
        return influence
    
    def get_stone_value(self, x: int, y: int) -> float:
        """Get positional value of a stone."""
        value = 1.0
        
        # Distance to edge
        distance_to_edge = min(x, y, self.size - 1 - x, self.size - 1 - y)
        if distance_to_edge == 0:
            value *= 0.7
        elif distance_to_edge == 1:
            value *= 0.85
        
        # Star points
        if self.size >= 9:
            star_points = self.get_star_points()
            for sp in star_points:
                if x == sp[0] and y == sp[1]:
                    value *= 1.2
                    break
        
        return value
    
    def calculate_board_hash(self) -> int:
        """Calculate a hash for the current board state."""
        hash_value = 0
        for y in range(self.size):
            for x in range(self.size):
                hash_value = hash_value * 3 + self.board[y][x]
        return hash_value


def main():
    """Test the AI implementation."""
    ai = GoAIOptimized(9)
    
    # Initialize a test board
    test_board = [[0 for _ in range(9)] for _ in range(9)]
    test_board[3][3] = 1  # Black stone
    test_board[3][4] = 2  # White stone
    test_board[4][3] = 2  # White stone
    test_board[4][4] = 1  # Black stone
    
    # Find best move for black
    move = ai.get_best_move(test_board, ai.BLACK, 0, 0)
    if move:
        print(f"Best move for black: ({move[0]}, {move[1]})")
    else:
        print("No valid moves found")


if __name__ == "__main__":
    main()