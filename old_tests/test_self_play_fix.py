#!/usr/bin/env python3
import torch
from alpha_go import PolicyValueNet, AlphaGoPlayer
from optimized_go import OptimizedGoGame, BLACK, WHITE

def test_self_play_move_generation():
    """Test that self-play move generation works without NaN errors"""
    print("Testing self-play move generation...")
    
    board_size = 9
    net = PolicyValueNet(board_size=board_size)
    net.eval()
    
    # Test with different temperature values
    temperatures = [0.0, 0.5, 1.0, 2.0]
    
    for temp in temperatures:
        print(f"\nTesting with temperature={temp}")
        
        game = OptimizedGoGame(board_size)
        player = AlphaGoPlayer(net, simulations=50, is_self_play=True)
        
        # Test first few moves
        for i in range(5):
            if game.game_over:
                break
                
            color = 'black' if game.current_player == BLACK else 'white'
            
            # This should not raise ValueError about NaN
            try:
                move = player.get_move(game, color, temperature=temp)
                print(f"  Move {i+1}: {move}")
                
                # Verify we can access MCTS statistics
                if hasattr(player, 'last_root') and player.last_root.children:
                    total_visits = sum(child.visits for child in player.last_root.children)
                    print(f"  Total MCTS visits: {total_visits}")
                
                # Make the move
                if move is None:
                    game.pass_turn(color)
                else:
                    game.make_move(move[0], move[1], color)
                    
            except ValueError as e:
                print(f"  ERROR: {e}")
                return False
    
    print("\nAll tests passed!")
    return True

def test_edge_cases():
    """Test edge cases that might cause issues"""
    print("\n\nTesting edge cases...")
    
    board_size = 9
    net = PolicyValueNet(board_size=board_size)
    net.eval()
    
    # Test with very few simulations (might lead to zero visits)
    print("Testing with minimal simulations (1)...")
    game = OptimizedGoGame(board_size)
    player = AlphaGoPlayer(net, simulations=1, is_self_play=True)
    
    try:
        move = player.get_move(game, 'black', temperature=1.0)
        print(f"  Move with 1 simulation: {move}")
    except ValueError as e:
        print(f"  ERROR: {e}")
        return False
    
    # Test late game position
    print("\nTesting late game position...")
    game = OptimizedGoGame(board_size)
    
    # Fill board partially
    moves = [(i, j) for i in range(0, board_size-1, 2) for j in range(0, board_size-1, 2)]
    for idx, (i, j) in enumerate(moves[:20]):
        color = 'black' if idx % 2 == 0 else 'white'
        game.make_move(i, j, color)
    
    player = AlphaGoPlayer(net, simulations=50, is_self_play=True)
    
    try:
        move = player.get_move(game, 'black' if game.current_player == BLACK else 'white', temperature=0.5)
        print(f"  Late game move: {move}")
    except ValueError as e:
        print(f"  ERROR: {e}")
        return False
    
    print("\nEdge case tests passed!")
    return True

if __name__ == "__main__":
    success = test_self_play_move_generation() and test_edge_cases()
    
    if success:
        print("\n✓ All tests passed! The NaN error has been fixed.")
    else:
        print("\n✗ Some tests failed.")