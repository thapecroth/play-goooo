#!/usr/bin/env python3
"""Test the AlphaGo fix for empty move list"""

import torch
from optimized_go import OptimizedGoGame, BLACK, WHITE
from alpha_go import PolicyValueNet, AlphaGoPlayer

def test_empty_moves_scenario():
    """Test AlphaGo behavior when no valid moves are available"""
    print("Testing AlphaGo with no valid moves scenario...")
    
    # Create a small board for easier testing
    board_size = 9
    game = OptimizedGoGame(board_size)
    
    # Create the model
    device = torch.device('cpu')
    policy_value_net = PolicyValueNet(board_size, num_blocks=3)
    
    # Create players
    alpha_player = AlphaGoPlayer(policy_value_net, simulations=10, device=device)
    alpha_player_self_play = AlphaGoPlayer(policy_value_net, simulations=10, is_self_play=True, device=device)
    
    # Test 1: Normal game state
    print("\n1. Testing normal game state...")
    try:
        move = alpha_player.get_move(game, 'black')
        print(f"   Normal mode - Move: {move}")
        move = alpha_player_self_play.get_move(game, 'black')
        print(f"   Self-play mode - Move: {move}")
    except Exception as e:
        print(f"   ERROR: {e}")
    
    # Test 2: Game over state
    print("\n2. Testing game over state...")
    game.game_over = True
    try:
        move = alpha_player.get_move(game, 'black')
        print(f"   Normal mode - Move: {move}")
        move = alpha_player_self_play.get_move(game, 'black')
        print(f"   Self-play mode - Move: {move}")
    except Exception as e:
        print(f"   ERROR: {e}")
    
    # Test 3: Create a board state with very few valid moves
    print("\n3. Testing board with limited moves...")
    game = OptimizedGoGame(3)  # Very small board
    
    # Fill most of the board
    positions = [(0,0), (0,1), (1,0), (1,1), (2,0), (2,1), (0,2), (1,2)]
    for i, (x, y) in enumerate(positions):
        color = 'black' if i % 2 == 0 else 'white'
        game.make_move(x, y, color)
    
    print(f"   Board state:")
    for row in game.board:
        print(f"   {row}")
    
    alpha_player_small = AlphaGoPlayer(PolicyValueNet(3, num_blocks=2), simulations=5, device=device)
    
    try:
        move = alpha_player_small.get_move(game, 'black')
        print(f"   Move found: {move}")
    except Exception as e:
        print(f"   ERROR: {e}")
    
    print("\nAll tests completed!")

def test_with_classic_ai():
    """Test the two-stage training scenario"""
    print("\n\nTesting two-stage training scenario...")
    
    from classic_go_ai import ClassicGoAI
    
    board_size = 9
    game = OptimizedGoGame(board_size)
    
    # Create AI players
    device = torch.device('cpu')
    policy_value_net = PolicyValueNet(board_size, num_blocks=3)
    alpha_player = AlphaGoPlayer(policy_value_net, simulations=10, device=device)
    classic_ai = ClassicGoAI(board_size)
    
    print("\nPlaying a few moves...")
    for i in range(5):
        current_color = 'black' if game.current_player == BLACK else 'white'
        
        if i % 2 == 0:
            # AlphaGo move
            print(f"\nMove {i+1}: AlphaGo ({current_color})")
            try:
                move = alpha_player.get_move(game, current_color)
                print(f"   AlphaGo selected: {move}")
            except Exception as e:
                print(f"   ERROR in AlphaGo: {e}")
                move = None
        else:
            # Classic AI move
            print(f"\nMove {i+1}: Classic AI ({current_color})")
            legal_moves = classic_ai.get_legal_moves(game.board, game.current_player)
            if legal_moves:
                move = legal_moves[0]  # Just pick first legal move
                print(f"   Classic AI selected: {move}")
            else:
                move = None
                print(f"   No legal moves")
        
        if move is None:
            game.pass_turn(current_color)
            print(f"   Passed")
        else:
            success = game.make_move(move[0], move[1], current_color)
            print(f"   Move success: {success}")
    
    print("\nTest completed successfully!")

if __name__ == "__main__":
    test_empty_moves_scenario()
    test_with_classic_ai()