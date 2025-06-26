#!/usr/bin/env python3
"""
Optimized Go game implementation for Codon compilation.
Uses simple text-based communication instead of JSON.
"""

# Constants
EMPTY = 0
BLACK = 1
WHITE = 2

class FastGoGame:
    """Fast Go game implementation for Codon"""
    board_size: int
    board: list[list[int]]
    current_player: int
    game_over: bool
    winner: int
    pass_count: int
    captured_black: int
    captured_white: int
    ko_x: int
    ko_y: int
    
    def __init__(self, board_size: int = 9):
        self.board_size = board_size
        self.board = [[EMPTY for _ in range(board_size)] for _ in range(board_size)]
        self.current_player = BLACK
        self.game_over = False
        self.winner = 0
        self.pass_count = 0
        self.captured_black = 0
        self.captured_white = 0
        self.ko_x = -1
        self.ko_y = -1
    
    def make_move(self, x: int, y: int) -> bool:
        """Make a move on the board"""
        if x < 0 or x >= self.board_size or y < 0 or y >= self.board_size:
            return False
        
        if self.board[y][x] != EMPTY:
            return False
        
        if self.ko_x == x and self.ko_y == y:
            return False
        
        # Place stone
        self.board[y][x] = self.current_player
        
        # Capture enemy stones
        captured = self._capture_stones(3 - self.current_player)
        
        # Check suicide
        if not self._has_liberties(x, y) and captured == 0:
            self.board[y][x] = EMPTY
            return False
        
        # Update captures
        if self.current_player == BLACK:
            self.captured_black += captured
        else:
            self.captured_white += captured
        
        # Simple ko detection
        if captured == 1:
            self.ko_x = -1  # Simplified ko handling
            self.ko_y = -1
        else:
            self.ko_x = -1
            self.ko_y = -1
        
        # Switch player
        self.current_player = 3 - self.current_player
        self.pass_count = 0
        
        return True
    
    def pass_turn(self):
        """Pass the turn"""
        self.pass_count += 1
        self.current_player = 3 - self.current_player
        
        if self.pass_count >= 2:
            self.game_over = True
            self._calculate_winner()
    
    def _capture_stones(self, color: int) -> int:
        """Capture stones without liberties"""
        captured = 0
        
        for y in range(self.board_size):
            for x in range(self.board_size):
                if self.board[y][x] == color:
                    if not self._has_liberties(x, y):
                        captured += self._remove_group(x, y)
        
        return captured
    
    def _has_liberties(self, x: int, y: int) -> bool:
        """Check if group has liberties"""
        color = self.board[y][x]
        if color == EMPTY:
            return True
        
        visited = [[False for _ in range(self.board_size)] for _ in range(self.board_size)]
        stack_x = [x]
        stack_y = [y]
        
        while len(stack_x) > 0:
            cx = stack_x.pop()
            cy = stack_y.pop()
            
            if visited[cy][cx]:
                continue
            
            visited[cy][cx] = True
            
            # Check neighbors
            for dx in [-1, 0, 1]:
                for dy in [-1, 0, 1]:
                    if abs(dx) + abs(dy) != 1:  # Only orthogonal
                        continue
                    
                    nx = cx + dx
                    ny = cy + dy
                    
                    if 0 <= nx < self.board_size and 0 <= ny < self.board_size:
                        if self.board[ny][nx] == EMPTY:
                            return True
                        elif self.board[ny][nx] == color and not visited[ny][nx]:
                            stack_x.append(nx)
                            stack_y.append(ny)
        
        return False
    
    def _remove_group(self, x: int, y: int) -> int:
        """Remove a group and return count"""
        color = self.board[y][x]
        if color == EMPTY:
            return 0
        
        count = 0
        stack_x = [x]
        stack_y = [y]
        
        while len(stack_x) > 0:
            cx = stack_x.pop()
            cy = stack_y.pop()
            
            if self.board[cy][cx] != color:
                continue
            
            self.board[cy][cx] = EMPTY
            count += 1
            
            # Add neighbors
            for dx in [-1, 0, 1]:
                for dy in [-1, 0, 1]:
                    if abs(dx) + abs(dy) != 1:
                        continue
                    
                    nx = cx + dx
                    ny = cy + dy
                    
                    if 0 <= nx < self.board_size and 0 <= ny < self.board_size:
                        if self.board[ny][nx] == color:
                            stack_x.append(nx)
                            stack_y.append(ny)
        
        return count
    
    def _calculate_winner(self):
        """Calculate winner"""
        black_score = float(self.captured_black)
        white_score = float(self.captured_white) + 6.5
        
        # Simple territory count
        for y in range(self.board_size):
            for x in range(self.board_size):
                if self.board[y][x] == BLACK:
                    black_score += 1.0
                elif self.board[y][x] == WHITE:
                    white_score += 1.0
        
        if black_score > white_score:
            self.winner = BLACK
        else:
            self.winner = WHITE
    
    def get_valid_moves(self) -> list[tuple[int, int]]:
        """Get all valid moves"""
        moves = []
        
        for y in range(self.board_size):
            for x in range(self.board_size):
                if self.board[y][x] == EMPTY:
                    # Quick liberty check
                    has_liberty = False
                    for dx in [-1, 0, 1]:
                        for dy in [-1, 0, 1]:
                            if abs(dx) + abs(dy) != 1:
                                continue
                            nx = x + dx
                            ny = y + dy
                            if 0 <= nx < self.board_size and 0 <= ny < self.board_size:
                                if self.board[ny][nx] == EMPTY:
                                    has_liberty = True
                                    break
                    
                    if has_liberty:
                        moves.append((x, y))
                    else:
                        # Check if captures
                        self.board[y][x] = self.current_player
                        would_capture = False
                        
                        for dx in [-1, 0, 1]:
                            for dy in [-1, 0, 1]:
                                if abs(dx) + abs(dy) != 1:
                                    continue
                                nx = x + dx
                                ny = y + dy
                                if 0 <= nx < self.board_size and 0 <= ny < self.board_size:
                                    if self.board[ny][nx] == 3 - self.current_player:
                                        if not self._has_liberties(nx, ny):
                                            would_capture = True
                                            break
                        
                        if would_capture or self._has_liberties(x, y):
                            if self.ko_x != x or self.ko_y != y:
                                moves.append((x, y))
                        
                        self.board[y][x] = EMPTY
        
        return moves
    
    def print_board(self):
        """Print the board state"""
        for y in range(self.board_size):
            for x in range(self.board_size):
                if self.board[y][x] == EMPTY:
                    print(".", end=" ")
                elif self.board[y][x] == BLACK:
                    print("X", end=" ")
                else:
                    print("O", end=" ")
            print()


def benchmark_moves():
    """Benchmark move generation"""
    game = FastGoGame(9)
    
    # Add some stones
    game.board[3][3] = BLACK
    game.board[3][4] = WHITE
    game.board[4][3] = WHITE
    game.board[4][4] = BLACK
    
    total_moves = 0
    for i in range(1000):
        moves = game.get_valid_moves()
        total_moves += len(moves)
    
    print(f"Total moves found: {total_moves}")


def benchmark_game():
    """Benchmark full game simulation"""
    game = FastGoGame(9)
    move_count = 0
    
    while move_count < 100 and not game.game_over:
        moves = game.get_valid_moves()
        if len(moves) > 0:
            # Pick first valid move
            x, y = moves[0]
            if game.make_move(x, y):
                move_count += 1
        else:
            game.pass_turn()
    
    print(f"Game completed with {move_count} moves")
    print(f"Winner: {game.winner}")


def main():
    """Main benchmark"""
    print("FastGoGame Benchmark")
    print("===================")
    
    print("\nMove generation benchmark:")
    benchmark_moves()
    
    print("\nFull game benchmark:")
    benchmark_game()


if __name__ == "__main__":
    main() 