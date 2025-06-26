#!/usr/bin/env python3
"""
Simplified Go AI for Codon compilation - minimal version
"""

def get_best_move(board, color, board_size):
    """Find a valid move for the given color"""
    EMPTY = 0
    valid_moves = []
    
    for y in range(board_size):
        for x in range(board_size):
            if board[y][x] == EMPTY:
                # Check if has neighbor liberty
                has_liberty = False
                for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    nx = x + dx
                    ny = y + dy
                    if 0 <= nx < board_size and 0 <= ny < board_size:
                        if board[ny][nx] == EMPTY:
                            has_liberty = True
                            break
                
                if has_liberty:
                    valid_moves.append((x, y))
    
    if valid_moves:
        return valid_moves[0]
    return None

def evaluate_position(board, color, board_size):
    """Simple position evaluation"""
    score = 0
    
    for y in range(board_size):
        for x in range(board_size):
            if board[y][x] == color:
                score += 1
            elif board[y][x] == 3 - color:
                score -= 1
    
    return score

def main():
    """Test the functions"""
    board_size = 9
    board = [[0 for _ in range(board_size)] for _ in range(board_size)]
    
    # Add some test stones
    board[3][3] = 1
    board[3][4] = 2
    board[4][3] = 2
    board[4][4] = 1
    
    move = get_best_move(board, 1, board_size)
    if move is not None:
        print(f"Best move: {move[0]}, {move[1]}")
    else:
        print("No valid moves")
    
    score = evaluate_position(board, 1, board_size)
    print(f"Position score: {score}")

if __name__ == "__main__":
    main()