import subprocess
import json
import numpy as np
from typing import Optional, Tuple, List

# Constants matching the compiled binary
EMPTY = 0
BLACK = 1
WHITE = 2

class CodonGoGame:
    """Wrapper for the Codon-compiled Go game engine.
    
    This class maintains game state and uses the compiled binary for 
    computationally intensive operations like move evaluation.
    """
    
    def __init__(self, board_size: int = 9):
        self.board_size = board_size
        self.board = np.zeros((board_size, board_size), dtype=np.int8)
        self._current_player = BLACK
        self.game_over = False
        self.winner = None
        self.pass_count = 0
        self.captured_black = 0
        self.captured_white = 0
        self.move_history = []
        self.ko_point = None
        
        # Check if compiled binary exists
        self.use_binary = self._check_binary_exists()
        if not self.use_binary:
            print("Warning: go_ai_codon_compiled not found. Falling back to Python implementation.")
    
    def _check_binary_exists(self) -> bool:
        """Check if the compiled binary exists"""
        try:
            result = subprocess.run(['./go_ai_codon_compiled'], 
                                  capture_output=True, text=True)
            return True
        except FileNotFoundError:
            return False
    
    def _call_binary(self, command: str, data: dict) -> dict:
        """Call the compiled binary with a command and data"""
        if not self.use_binary:
            return self._fallback_implementation(command, data)
        
        # Convert data to JSON for communication
        input_json = json.dumps(data)
        
        try:
            result = subprocess.run(
                ['./go_ai_codon_compiled', command],
                input=input_json,
                capture_output=True,
                text=True
            )
            
            if result.returncode != 0:
                print(f"Binary error: {result.stderr}")
                return self._fallback_implementation(command, data)
            
            return json.loads(result.stdout)
        except Exception as e:
            print(f"Error calling binary: {e}")
            return self._fallback_implementation(command, data)
    
    def _fallback_implementation(self, command: str, data: dict) -> dict:
        """Fallback Python implementation for when binary is not available"""
        if command == "get_best_move":
            # Simple fallback - find first valid move
            board_list = data['board']
            color = data['color']
            
            for y in range(self.board_size):
                for x in range(self.board_size):
                    if board_list[y][x] == EMPTY:
                        if self._is_valid_move_simple(x, y, color, board_list):
                            return {'move': [x, y]}
            
            return {'move': None}
        
        elif command == "evaluate_position":
            # Simple material count
            board_list = data['board']
            color = data['color']
            score = 0
            for row in board_list:
                for cell in row:
                    if cell == color:
                        score += 1
                    elif cell == 3 - color:
                        score -= 1
            return {'score': score}
        
        return {}
    
    def _is_valid_move_simple(self, x: int, y: int, color: int, board_list: list) -> bool:
        """Simple validity check for fallback"""
        # Check if has at least one liberty
        for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            nx, ny = x + dx, y + dy
            if 0 <= nx < self.board_size and 0 <= ny < self.board_size:
                if board_list[ny][nx] == EMPTY:
                    return True
        return False
    
    def _board_to_list(self) -> List[List[int]]:
        """Convert numpy board to list for binary communication"""
        return self.board.tolist()
    
    def make_move(self, x: int, y: int, color: str) -> bool:
        """Make a move on the board"""
        player_num = BLACK if color == 'black' else WHITE
        
        # Basic validation
        if not (0 <= x < self.board_size and 0 <= y < self.board_size):
            return False
        
        if self.board[y, x] != EMPTY:
            return False
        
        # Check ko rule
        if self.ko_point and self.ko_point == (x, y):
            return False
        
        # Place stone
        self.board[y, x] = player_num
        
        # Capture enemy stones
        captured = self._capture_stones(3 - player_num)
        
        # Check suicide rule
        if not self._has_liberties(x, y) and not captured:
            self.board[y, x] = EMPTY
            return False
        
        # Update captures
        if player_num == BLACK:
            self.captured_black += len(captured)
        else:
            self.captured_white += len(captured)
        
        # Update ko point
        if len(captured) == 1 and self._is_single_stone_capture(x, y):
            self.ko_point = captured[0]
        else:
            self.ko_point = None
        
        # Update game state
        self.move_history.append((x, y, player_num))
        self._current_player = 3 - self._current_player
        self.pass_count = 0
        
        return True
    
    def _capture_stones(self, color: int) -> List[Tuple[int, int]]:
        """Capture stones of the given color that have no liberties"""
        captured = []
        
        for y in range(self.board_size):
            for x in range(self.board_size):
                if self.board[y, x] == color:
                    if not self._has_liberties(x, y):
                        group = self._get_group(x, y)
                        for gx, gy in group:
                            self.board[gy, gx] = EMPTY
                            captured.append((gx, gy))
        
        return captured
    
    def _has_liberties(self, x: int, y: int) -> bool:
        """Check if a stone/group has liberties"""
        color = self.board[y, x]
        if color == EMPTY:
            return True
        
        visited = set()
        stack = [(x, y)]
        
        while stack:
            cx, cy = stack.pop()
            if (cx, cy) in visited:
                continue
            
            visited.add((cx, cy))
            
            for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                nx, ny = cx + dx, cy + dy
                if 0 <= nx < self.board_size and 0 <= ny < self.board_size:
                    if self.board[ny, nx] == EMPTY:
                        return True
                    elif self.board[ny, nx] == color and (nx, ny) not in visited:
                        stack.append((nx, ny))
        
        return False
    
    def _get_group(self, x: int, y: int) -> List[Tuple[int, int]]:
        """Get all stones in a group"""
        color = self.board[y, x]
        if color == EMPTY:
            return []
        
        group = []
        visited = set()
        stack = [(x, y)]
        
        while stack:
            cx, cy = stack.pop()
            if (cx, cy) in visited:
                continue
            
            visited.add((cx, cy))
            group.append((cx, cy))
            
            for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                nx, ny = cx + dx, cy + dy
                if 0 <= nx < self.board_size and 0 <= ny < self.board_size:
                    if self.board[ny, nx] == color and (nx, ny) not in visited:
                        stack.append((nx, ny))
        
        return group
    
    def _is_single_stone_capture(self, x: int, y: int) -> bool:
        """Check if this move captured exactly one stone in a ko-like pattern"""
        # Simple heuristic for ko detection
        group = self._get_group(x, y)
        return len(group) == 1
    
    def pass_turn(self, color: str) -> None:
        """Pass the turn"""
        self.pass_count += 1
        self._current_player = 3 - self._current_player
        
        if self.pass_count >= 2:
            self.game_over = True
            self._calculate_winner()
    
    def _calculate_winner(self) -> None:
        """Calculate the winner using area scoring"""
        black_score = self.captured_black
        white_score = self.captured_white + 6.5  # Komi
        
        # Count territory
        territory = self._count_territory()
        black_score += territory['black']
        white_score += territory['white']
        
        if black_score > white_score:
            self.winner = 'black'
        else:
            self.winner = 'white'
    
    def _count_territory(self) -> dict:
        """Count territory for each player"""
        territory = {'black': 0, 'white': 0}
        visited = set()
        
        for y in range(self.board_size):
            for x in range(self.board_size):
                if self.board[y, x] == EMPTY and (x, y) not in visited:
                    region, borders = self._get_empty_region(x, y, visited)
                    
                    # Check who controls this region
                    if len(borders) == 1:
                        if BLACK in borders:
                            territory['black'] += len(region)
                        else:
                            territory['white'] += len(region)
        
        return territory
    
    def _get_empty_region(self, x: int, y: int, visited: set) -> Tuple[List[Tuple[int, int]], set]:
        """Get an empty region and its bordering colors"""
        region = []
        borders = set()
        stack = [(x, y)]
        
        while stack:
            cx, cy = stack.pop()
            if (cx, cy) in visited:
                continue
            
            visited.add((cx, cy))
            region.append((cx, cy))
            
            for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                nx, ny = cx + dx, cy + dy
                if 0 <= nx < self.board_size and 0 <= ny < self.board_size:
                    if self.board[ny, nx] == EMPTY and (nx, ny) not in visited:
                        stack.append((nx, ny))
                    elif self.board[ny, nx] != EMPTY:
                        borders.add(self.board[ny, nx])
        
        return region, borders
    
    @property
    def current_player(self) -> str:
        """Get current player as string for compatibility"""
        return 'black' if self._current_player == BLACK else 'white'
    
    @current_player.setter
    def current_player(self, value):
        """Set current player from string or int"""
        if isinstance(value, str):
            self._current_player = BLACK if value == 'black' else WHITE
        else:
            self._current_player = value
    
    @property
    def captures(self) -> dict:
        """Get captures as dict for compatibility"""
        return {'black': self.captured_black, 'white': self.captured_white}
    
    @property
    def size(self) -> int:
        """Get board size for compatibility"""
        return self.board_size
    
    def get_best_move_from_binary(self, color: int) -> Optional[Tuple[int, int]]:
        """Get best move using the compiled binary"""
        data = {
            'board': self._board_to_list(),
            'color': color,
            'board_size': self.board_size
        }
        
        result = self._call_binary('get_best_move', data)
        move = result.get('move')
        
        if move and isinstance(move, list) and len(move) == 2:
            return tuple(move)
        return None
    
    def evaluate_position_from_binary(self, color: int) -> float:
        """Evaluate position using the compiled binary"""
        data = {
            'board': self._board_to_list(),
            'color': color,
            'board_size': self.board_size
        }
        
        result = self._call_binary('evaluate_position', data)
        return result.get('score', 0.0) 