#!/usr/bin/env python3
"""
Minimal test to debug self-play hanging issue
"""

import torch
import time
from optimized_go import OptimizedGoGame, BLACK, WHITE
from alpha_go import PolicyValueNet, AlphaGoPlayer

def test_single_move():
    """Test a single move generation to see where it hangs"""
    print("Testing single move generation...")
    
    board_size = 9
    game = OptimizedGoGame(board_size)
    
    # Create a simple network
    policy_value_net = PolicyValueNet(board_size, num_blocks=2)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    policy_value_net.to(device)
    
    # Create player with minimal simulations
    player = AlphaGoPlayer(policy_value_net, simulations=10, device=device, is_self_play=True)
    
    print("Game initialized, attempting to get first move...")
    print(f"Current player: {'black' if game.current_player == BLACK else 'white'}")
    
    start_time = time.time()
    try:
        move = player.get_move(game, 'black')
        elapsed = time.time() - start_time
        print(f"Move generated in {elapsed:.2f}s: {move}")
        return True
    except Exception as e:
        print(f"Error generating move: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_full_game():
    """Test a full game with timeout"""
    print("\nTesting full game...")
    
    board_size = 9
    game = OptimizedGoGame(board_size)
    
    policy_value_net = PolicyValueNet(board_size, num_blocks=2)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    policy_value_net.to(device)
    
    player = AlphaGoPlayer(policy_value_net, simulations=10, device=device, is_self_play=True)
    
    move_count = 0
    max_moves = 50  # Limit for testing
    
    start_time = time.time()
    
    while not game.game_over and move_count < max_moves:
        move_start = time.time()
        
        current_color = 'black' if game.current_player == BLACK else 'white'
        print(f"\nMove {move_count + 1}, Player: {current_color}")
        
        try:
            move = player.get_move(game, current_color)
            move_time = time.time() - move_start
            
            if move is None:
                print(f"Pass (took {move_time:.2f}s)")
                game.pass_turn(current_color)
            else:
                print(f"Move: {move} (took {move_time:.2f}s)")
                success = game.make_move(move[0], move[1], current_color)
                if not success:
                    print("Invalid move, passing instead")
                    game.pass_turn(current_color)
            
            move_count += 1
            
        except Exception as e:
            print(f"Error during move: {e}")
            import traceback
            traceback.print_exc()
            break
    
    total_time = time.time() - start_time
    print(f"\nGame finished after {move_count} moves in {total_time:.2f}s")
    print(f"Average time per move: {total_time/move_count:.2f}s")
    
    if game.game_over:
        print(f"Winner: {game.winner}")

def test_mcts_expansion():
    """Test MCTS expansion specifically"""
    print("\nTesting MCTS expansion...")
    
    board_size = 9
    game = OptimizedGoGame(board_size)
    
    policy_value_net = PolicyValueNet(board_size, num_blocks=2)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    policy_value_net.to(device)
    
    player = AlphaGoPlayer(policy_value_net, simulations=1, device=device)
    
    # Test single simulation
    from alpha_go import AlphaGoMCTSNode
    root = AlphaGoMCTSNode(game.copy())
    
    print("Running single simulation...")
    start_time = time.time()
    try:
        player._run_simulation(root)
        elapsed = time.time() - start_time
        print(f"Simulation completed in {elapsed:.4f}s")
        print(f"Root visits: {root.visits}")
        print(f"Root children: {len(root.children)}")
    except Exception as e:
        print(f"Error in simulation: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    print("="*60)
    print("Self-Play Debug Test")
    print("="*60)
    
    # Test 1: Single move
    if test_single_move():
        print("\n✓ Single move test passed")
    else:
        print("\n✗ Single move test failed")
        exit(1)
    
    # Test 2: MCTS expansion
    test_mcts_expansion()
    
    # Test 3: Full game (if previous tests pass)
    test_full_game()