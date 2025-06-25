
class SimpleGoBoard:
    def __init__(self, size):
        self.size = size
        self.board = [[0] * size for _ in range(size)]

    def make_move(self, x, y, color):
        if self.board[y][x] != 0:
            return False
        self.board[y][x] = color
        return True

    def get_board(self):
        return self.board
