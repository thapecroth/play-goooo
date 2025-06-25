from server import GoGame

class DebugGoGame(GoGame):
    def make_move(self, x, y, color):
        if (self.game_over or x < 0 or x >= self.size or 
            y < 0 or y >= self.size or self.board[y][x] is not None or
            color != self.current_player):
            return False
        
        board_copy = self.copy_board()
        self.board[y][x] = color
        print(f'Stone placed at ({x},{y})')
        
        opposite_color = 'white' if color == 'black' else 'black'
        captured_stones = self.capture_stones(opposite_color)
        print(f'Captured {len(captured_stones)} stones')
        
        liberties_check = self.has_liberties(x, y, color)
        print(f'Liberties check: {liberties_check}')
        if not liberties_check:
            print('REJECTED: No liberties')
            self.board = board_copy
            return False
        
        ko_check = self.is_ko(board_copy)
        print(f'Ko check: {ko_check}')
        if ko_check:
            print('REJECTED: Ko rule')
            self.board = board_copy
            return False
        
        print('ACCEPTED')
        return True

game = DebugGoGame(3)
game.make_move(1, 0, 'black')
game.make_move(1, 1, 'white')  
game.make_move(0, 1, 'black')
result = game.make_move(0, 0, 'white')
print('Final result:', result)