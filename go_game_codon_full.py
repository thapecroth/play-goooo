#!/usr/bin/env python3
"""
Full Go game implementation optimized for Codon compilation.
This provides all game functionality needed for AlphaGo training.
"""

import sys
import json

# Constants
EMPTY = 0
BLACK = 1
WHITE = 2

class CodonGoGame:
    """Complete Go game implementation for Codon compilation"""
    
    def __init__(self, board_size: int = 9):
        self.board_size = board_size
        self.board = [[EMPTY for _ in range(board_size)] for _ in range(board_size)]
        self.current_player = BLACK
        self.game_over = False
        self.winner = 0  # 0: none, 1: black, 2: white
        self.pass_count = 0
        self.captured_black = 0
        self.captured_white = 0
        self.ko_point_x = -1
        self.ko_point_y = -1
        self.move_count = 0
    
    def reset(self):
        """Reset the game state"""
        for y in range(self.board_size):
            for x in range(self.board_size):
                self.board[y][x] = EMPTY
        self.current_player = BLACK
        self.game_over = False
        self.winner = 0
        self.pass_count = 0
        self.captured_black = 0
        self.captured_white = 0
        self.ko_point_x = -1
        self.ko_point_y = -1
        self.move_count = 0
    
    def make_move(self, x: int, y: int) -> bool:
        """Make a move on the board"""
        # Basic validation
        if x < 0 or x >= self.board_size or y < 0 or y >= self.board_size:
            return False
        
        if self.board[y][x] != EMPTY:
            return False
        
        # Check ko rule
        if self.ko_point_x == x and self.ko_point_y == y:
            return False
        
        # Place stone
        self.board[y][x] = self.current_player
        
        # Capture enemy stones
        captured_count = self._capture_stones(3 - self.current_player)
        
        # Check suicide rule
        if not self._has_liberties(x, y) and captured_count == 0:
            self.board[y][x] = EMPTY
            return False
        
        # Update captures
        if self.current_player == BLACK:
            self.captured_black += captured_count
        else:
            self.captured_white += captured_count
        
        # Update ko point
        if captured_count == 1 and self._is_single_stone_group(x, y):
            # Find the captured stone position for ko
            for dy in range(self.board_size):
                for dx in range(self.board_size):
                    if self.board[dy][dx] == EMPTY and self._was_just_captured(dx, dy):
                        self.ko_point_x = dx
                        self.ko_point_y = dy
                        break
        else:
            self.ko_point_x = -1
            self.ko_point_y = -1
        
        # Update game state
        self.current_player = 3 - self.current_player
        self.pass_count = 0
        self.move_count += 1
        
        return True
    
    def pass_turn(self):
        """Pass the turn"""
        self.pass_count += 1
        self.current_player = 3 - self.current_player
        
        if self.pass_count >= 2:
            self.game_over = True
            self._calculate_winner()
    
    def _capture_stones(self, color: int) -> int:
        """Capture stones of the given color that have no liberties"""
        captured_count = 0
        
        for y in range(self.board_size):
            for x in range(self.board_size):
                if self.board[y][x] == color:
                    if not self._has_liberties(x, y):
                        captured_count += self._remove_group(x, y)
        
        return captured_count
    
    def _has_liberties(self, x: int, y: int) -> bool:
        """Check if a stone/group has liberties using iterative DFS"""
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
            
            # Check all four directions
            for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                nx = cx + dx
                ny = cy + dy
                
                if nx >= 0 and nx < self.board_size and ny >= 0 and ny < self.board_size:
                    if self.board[ny][nx] == EMPTY:
                        return True
                    elif self.board[ny][nx] == color and not visited[ny][nx]:
                        stack_x.append(nx)
                        stack_y.append(ny)
        
        return False
    
    def _remove_group(self, x: int, y: int) -> int:
        """Remove a group of stones and return the count"""
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
            
            # Add neighbors to stack
            for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                nx = cx + dx
                ny = cy + dy
                
                if nx >= 0 and nx < self.board_size and ny >= 0 and ny < self.board_size:
                    if self.board[ny][nx] == color:
                        stack_x.append(nx)
                        stack_y.append(ny)
        
        return count
    
    def _is_single_stone_group(self, x: int, y: int) -> bool:
        """Check if a position contains a single stone group"""
        color = self.board[y][x]
        if color == EMPTY:
            return False
        
        count = 0
        visited = [[False for _ in range(self.board_size)] for _ in range(self.board_size)]
        stack_x = [x]
        stack_y = [y]
        
        while len(stack_x) > 0 and count <= 1:
            cx = stack_x.pop()
            cy = stack_y.pop()
            
            if visited[cy][cx]:
                continue
            
            visited[cy][cx] = True
            count += 1
            
            if count > 1:
                return False
            
            for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                nx = cx + dx
                ny = cy + dy
                
                if nx >= 0 and nx < self.board_size and ny >= 0 and ny < self.board_size:
                    if self.board[ny][nx] == color and not visited[ny][nx]:
                        stack_x.append(nx)
                        stack_y.append(ny)
        
        return count == 1
    
    def _was_just_captured(self, x: int, y: int) -> bool:
        """Check if a position was just captured (heuristic)"""
        if self.board[y][x] != EMPTY:
            return False
        
        # Check if surrounded by enemy stones
        enemy = 3 - self.current_player  # Previous player
        for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            nx = x + dx
            ny = y + dy
            
            if nx >= 0 and nx < self.board_size and ny >= 0 and ny < self.board_size:
                if self.board[ny][nx] != enemy:
                    return False
        
        return True
    
    def _calculate_winner(self):
        """Calculate the winner using area scoring"""
        black_score = float(self.captured_black)
        white_score = float(self.captured_white) + 6.5  # Komi
        
        # Count territory
        territory = self._count_territory()
        black_score += float(territory[0])
        white_score += float(territory[1])
        
        if black_score > white_score:
            self.winner = BLACK
        else:
            self.winner = WHITE
    
    def _count_territory(self) -> tuple[int, int]:
        """Count territory for each player"""
        black_territory = 0
        white_territory = 0
        visited = [[False for _ in range(self.board_size)] for _ in range(self.board_size)]
        
        for y in range(self.board_size):
            for x in range(self.board_size):
                if self.board[y][x] == EMPTY and not visited[y][x]:
                    region_size, owner = self._get_territory_owner(x, y, visited)
                    
                    if owner == BLACK:
                        black_territory += region_size
                    elif owner == WHITE:
                        white_territory += region_size
        
        return (black_territory, white_territory)
    
    def _get_territory_owner(self, x: int, y: int, visited: list[list[bool]]) -> tuple[int, int]:
        """Get the owner of an empty region"""
        region_size = 0
        borders_black = False
        borders_white = False
        
        stack_x = [x]
        stack_y = [y]
        
        while len(stack_x) > 0:
            cx = stack_x.pop()
            cy = stack_y.pop()
            
            if visited[cy][cx]:
                continue
            
            visited[cy][cx] = True
            region_size += 1
            
            for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                nx = cx + dx
                ny = cy + dy
                
                if nx >= 0 and nx < self.board_size and ny >= 0 and ny < self.board_size:
                    if self.board[ny][nx] == EMPTY and not visited[ny][nx]:
                        stack_x.append(nx)
                        stack_y.append(ny)
                    elif self.board[ny][nx] == BLACK:
                        borders_black = True
                    elif self.board[ny][nx] == WHITE:
                        borders_white = True
        
        # Determine owner
        if borders_black and not borders_white:
            return (region_size, BLACK)
        elif borders_white and not borders_black:
            return (region_size, WHITE)
        else:
            return (region_size, 0)  # Neutral
    
    def get_valid_moves(self) -> list[tuple[int, int]]:
        """Get all valid moves for the current player"""
        moves = []
        
        for y in range(self.board_size):
            for x in range(self.board_size):
                if self.board[y][x] == EMPTY:
                    # Quick check for obviously valid moves
                    if self._has_adjacent_liberty(x, y):
                        moves.append((x, y))
                    else:
                        # Check if move would be valid (captures or has liberties)
                        self.board[y][x] = self.current_player
                        captured = self._would_capture(3 - self.current_player)
                        has_libs = self._has_liberties(x, y)
                        self.board[y][x] = EMPTY
                        
                        if captured > 0 or has_libs:
                            # Check ko rule
                            if self.ko_point_x != x or self.ko_point_y != y:
                                moves.append((x, y))
        
        return moves
    
    def _has_adjacent_liberty(self, x: int, y: int) -> bool:
        """Check if position has an adjacent empty point"""
        for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            nx = x + dx
            ny = y + dy
            
            if nx >= 0 and nx < self.board_size and ny >= 0 and ny < self.board_size:
                if self.board[ny][nx] == EMPTY:
                    return True
        
        return False
    
    def _would_capture(self, color: int) -> int:
        """Check how many stones would be captured without modifying board"""
        count = 0
        
        for y in range(self.board_size):
            for x in range(self.board_size):
                if self.board[y][x] == color:
                    if not self._has_liberties(x, y):
                        count += self._count_group_size(x, y)
        
        return count
    
    def _count_group_size(self, x: int, y: int) -> int:
        """Count the size of a group without modifying board"""
        color = self.board[y][x]
        if color == EMPTY:
            return 0
        
        count = 0
        visited = [[False for _ in range(self.board_size)] for _ in range(self.board_size)]
        stack_x = [x]
        stack_y = [y]
        
        while len(stack_x) > 0:
            cx = stack_x.pop()
            cy = stack_y.pop()
            
            if visited[cy][cx]:
                continue
            
            visited[cy][cx] = True
            count += 1
            
            for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                nx = cx + dx
                ny = cy + dy
                
                if nx >= 0 and nx < self.board_size and ny >= 0 and ny < self.board_size:
                    if self.board[ny][nx] == color and not visited[ny][nx]:
                        stack_x.append(nx)
                        stack_y.append(ny)
        
        return count
    
    def get_board_state(self) -> dict:
        """Get the current board state as a dictionary"""
        return {
            'board': [[self.board[y][x] for x in range(self.board_size)] for y in range(self.board_size)],
            'current_player': self.current_player,
            'game_over': self.game_over,
            'winner': self.winner,
            'captured_black': self.captured_black,
            'captured_white': self.captured_white,
            'move_count': self.move_count
        }
    
    def set_board_state(self, state: dict):
        """Set the board state from a dictionary"""
        board_data = state['board']
        for y in range(self.board_size):
            for x in range(self.board_size):
                self.board[y][x] = board_data[y][x]
        
        self.current_player = state['current_player']
        self.game_over = state['game_over']
        self.winner = state['winner']
        self.captured_black = state['captured_black']
        self.captured_white = state['captured_white']
        self.move_count = state.get('move_count', 0)


def process_command(game: CodonGoGame, command: str, data: dict) -> dict:
    """Process a command and return result"""
    if command == "make_move":
        x = data['x']
        y = data['y']
        success = game.make_move(x, y)
        return {
            'success': success,
            'state': game.get_board_state()
        }
    
    elif command == "pass_turn":
        game.pass_turn()
        return {
            'state': game.get_board_state()
        }
    
    elif command == "get_valid_moves":
        moves = game.get_valid_moves()
        return {
            'moves': [[x, y] for x, y in moves]
        }
    
    elif command == "get_state":
        return {
            'state': game.get_board_state()
        }
    
    elif command == "set_state":
        game.set_board_state(data['state'])
        return {'success': True}
    
    elif command == "reset":
        game.reset()
        return {
            'state': game.get_board_state()
        }
    
    else:
        return {'error': f'Unknown command: {command}'}


def main():
    """Main entry point for compiled binary"""
    if len(sys.argv) < 2:
        print("Usage: go_game_codon_full <command>")
        sys.exit(1)
    
    command = sys.argv[1]
    
    # Read input data from stdin
    input_data = sys.stdin.read()
    data = json.loads(input_data) if input_data else {}
    
    # Create or restore game
    board_size = data.get('board_size', 9)
    game = CodonGoGame(board_size)
    
    # Process command
    result = process_command(game, command, data)
    
    # Output result
    print(json.dumps(result))


if __name__ == "__main__":
    main() 