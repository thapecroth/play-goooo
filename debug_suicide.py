from server import GoGame

# Create a debug version that shows what's happening
class DebugGoGame(GoGame):
    def make_move(self, x, y, color):
        print(f"\n=== Making move at ({x},{y}) with {color} ===")
        
        # Basic checks
        if self.game_over:
            print("FAIL: Game is over")
            return False
        if x < 0 or x >= self.size or y < 0 or y >= self.size:
            print("FAIL: Out of bounds")
            return False
        if self.board[y][x] is not None:
            print("FAIL: Position occupied")
            return False
        if color != self.current_player:
            print(f"FAIL: Wrong turn (expected {self.current_player})")
            return False
        
        board_copy = self.copy_board()
        self.board[y][x] = color
        print("Stone placed")
        
        opposite_color = 'white' if color == 'black' else 'black'
        captured_stones = self.capture_stones(opposite_color)
        print(f"Captured {len(captured_stones)} {opposite_color} stones")
        
        # Check liberties
        liberties_result = self.has_liberties(x, y, color)
        print(f"Liberties check: {liberties_result}")
        
        if not liberties_result:
            print("FAIL: No liberties (suicide)")
            self.board = board_copy
            return False
            
        print("SUCCESS: Move allowed")
        return True

game = DebugGoGame(5)
game.board[1][1] = 'white'   # White stone at (1,1)
game.board[1][0] = 'black'   # Black left of white at (0,1) 
game.board[1][2] = 'black'   # Black right of white at (2,1)
game.board[0][1] = 'black'   # Black above white at (1,0)
game.board[2][0] = 'white'   # White below-left at (0,2)
game.board[2][2] = 'white'   # White below-right at (2,2)
# Position (2,1) should be empty for the move
game.current_player = 'black'

print("Detailed board state:")
for y in range(5):
    for x in range(5):
        print(f"({x},{y}):{game.board[y][x] or 'None'}", end="  ")
    print()

print("Before move:")
for y in range(5):
    for x in range(5):
        cell = game.board[y][x]
        if cell == 'black':
            print('B', end=' ')
        elif cell == 'white':
            print('W', end=' ')
        else:
            print('.', end=' ')
    print()

result = game.make_move(1, 2, 'black')  # Move to (1,2)
print(f"\nFinal result: {result}")