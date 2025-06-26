#!/usr/bin/env python3
"""
Test capture functionality carefully.
"""

from server import GoGame
from optimized_go import OptimizedGoGame
from codon_game_wrapper import CodonGoGame


def test_capture_with_proper_turns():
    """Test capture with proper turn alternation."""
    print("Testing capture with proper turn alternation...")
    
    games = {
        'Basic': GoGame(9),
        'Optimized': OptimizedGoGame(9),
        'Codon': CodonGoGame(9)
    }
    
    # Sequence that creates a capture situation
    # Black will surround and capture a white stone
    moves = [
        (4, 4, 'black'),   # B plays center
        (4, 5, 'white'),   # W plays below
        (3, 4, 'black'),   # B left of center
        (8, 8, 'white'),   # W elsewhere
        (5, 4, 'black'),   # B right of center
        (8, 7, 'white'),   # W elsewhere
        (4, 3, 'black'),   # B above center
        (3, 5, 'white'),   # W below B's left stone
        (3, 6, 'black'),   # B below W
        (5, 5, 'white'),   # W below B's right stone
        (5, 6, 'black'),   # B below W
        (4, 6, 'white'),   # W at bottom center
        (4, 7, 'black'),   # B below W
        (2, 5, 'white'),   # W left of its stone
        (6, 5, 'black'),   # B right of W stone
        (0, 0, 'white'),   # W corner
        # Now black can capture the white stone at (4,5)
        (4, 5, 'black'),   # B captures W!
    ]
    
    print("\nPlaying moves...")
    for i, (x, y, color) in enumerate(moves[:-1]):  # All but last move
        print(f"Move {i+1}: {color} at ({x},{y})")
        for name, game in games.items():
            success = game.make_move(x, y, color)
            if not success:
                print(f"  {name}: FAILED!")
                # Debug current player
                if hasattr(game, 'current_player'):
                    print(f"    Expected {color}, current player is {game.current_player}")
    
    print("\nBefore capture:")
    # Check white stone is there
    for name, game in games.items():
        if isinstance(game, GoGame):
            value = game.board[5][4]  # Note: y,x order
            print(f"  {name} at (4,5): {value}")
        else:
            value = game.board[5, 4]
            print(f"  {name} at (4,5): {value}")
    
    # Make capturing move
    print(f"\nCapturing move: black at (4,5)")
    for name, game in games.items():
        success = game.make_move(4, 5, 'black')
        print(f"  {name}: {'OK' if success else 'FAILED'}")
    
    print("\nAfter capture:")
    # Check captures
    for name, game in games.items():
        if hasattr(game, 'captures'):
            if isinstance(game.captures, dict):
                black_cap = game.captures.get('black', 0)
                white_cap = game.captures.get('white', 0)
            else:
                black_cap = game.captures.get(1, 0)
                white_cap = game.captures.get(2, 0)
        else:
            black_cap = getattr(game, 'captured_black', 0)
            white_cap = getattr(game, 'captured_white', 0)
        
        print(f"  {name}: Black captures={black_cap}, White captures={white_cap}")


def test_simpler_capture():
    """Test a simpler capture scenario."""
    print("\n\nTesting simpler capture...")
    
    games = {
        'Basic': GoGame(5),
        'Optimized': OptimizedGoGame(5),
        'Codon': CodonGoGame(5)
    }
    
    # Simple corner capture
    moves = [
        (0, 0, 'black'),   # B corner
        (1, 0, 'white'),   # W next to B
        (1, 1, 'black'),   # B diagonal
        (2, 0, 'white'),   # W extends
        (0, 1, 'black'),   # B captures W at (1,0)!
    ]
    
    for i, (x, y, color) in enumerate(moves):
        print(f"\nMove {i+1}: {color} at ({x},{y})")
        for name, game in games.items():
            # Show current player before move
            if hasattr(game, 'current_player'):
                current = game.current_player
                print(f"  {name}: current={current}, trying={color}")
            
            success = game.make_move(x, y, color)
            print(f"  {name}: {'OK' if success else 'FAILED'}")
            
            if i == len(moves) - 1:  # Last move
                # Check captures
                if hasattr(game, 'captures'):
                    print(f"    Captures: {game.captures}")


if __name__ == "__main__":
    test_capture_with_proper_turns()
    test_simpler_capture()