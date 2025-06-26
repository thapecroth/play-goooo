#!/usr/bin/env python3
"""
Visual test of capture scenarios.
"""

from server import GoGame
from optimized_go import OptimizedGoGame


def print_board(game, name):
    """Print board state."""
    print(f"\n{name} board:")
    size = game.size
    
    # Column numbers
    print("  ", end="")
    for x in range(size):
        print(f"{x} ", end="")
    print()
    
    for y in range(size):
        print(f"{y} ", end="")
        for x in range(size):
            if isinstance(game, GoGame):
                stone = game.board[y][x]
                if stone == 'black':
                    print("● ", end="")
                elif stone == 'white':
                    print("○ ", end="")
                else:
                    print(". ", end="")
            else:
                stone = game.board[y, x]
                if stone == 1:
                    print("● ", end="")
                elif stone == 2:
                    print("○ ", end="")
                else:
                    print(". ", end="")
        print()


def test_corner_capture():
    """Test corner capture visually."""
    print("Testing corner capture scenario...")
    
    games = {
        'Basic': GoGame(5),
        'Optimized': OptimizedGoGame(5)
    }
    
    # Corrected moves for actual corner capture
    moves = [
        (0, 1, 'black'),   # B at (0,1)
        (0, 0, 'white'),   # W at corner (0,0)
        (1, 0, 'black'),   # B at (1,0) - captures W!
    ]
    
    for i, (x, y, color) in enumerate(moves):
        print(f"\n=== Move {i+1}: {color} at ({x},{y}) ===")
        
        for name, game in games.items():
            success = game.make_move(x, y, color)
            print(f"{name}: {'OK' if success else 'FAILED'}")
            
            if success:
                print_board(game, name)
                
                # Show captures
                if hasattr(game, 'captures'):
                    print(f"Captures: {game.captures}")


def test_center_capture():
    """Test capturing a stone in center."""
    print("\n\nTesting center capture scenario...")
    
    games = {
        'Basic': GoGame(5),
        'Optimized': OptimizedGoGame(5)
    }
    
    # Surround white stone at (2,2)
    moves = [
        (2, 2, 'black'),   # B center
        (2, 1, 'white'),   # W above B
        (1, 1, 'black'),   # B to left of W
        (0, 0, 'white'),   # W corner
        (3, 1, 'black'),   # B to right of W
        (0, 1, 'white'),   # W elsewhere
        (2, 0, 'black'),   # B above W - captures!
    ]
    
    for i, (x, y, color) in enumerate(moves):
        print(f"\n=== Move {i+1}: {color} at ({x},{y}) ===")
        
        for name, game in games.items():
            success = game.make_move(x, y, color)
            print(f"{name}: {'OK' if success else 'FAILED'}")
            
            if success and (i == len(moves) - 1 or i == 1):  # Show after placing white and after capture
                print_board(game, name)
                
                # Show captures
                if hasattr(game, 'captures'):
                    print(f"Captures: {game.captures}")


if __name__ == "__main__":
    test_corner_capture()
    test_center_capture()