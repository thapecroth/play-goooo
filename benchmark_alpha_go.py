#!/usr/bin/env python3
import time
import torch
import numpy as np
from alpha_go import PolicyValueNet, AlphaGoPlayer
from alpha_go_optimized import BatchAlphaGoPlayer
from optimized_go import OptimizedGoGame, BLACK, WHITE

def benchmark_single_game(player_class, net, board_size=9, num_moves=20, simulations=100):
    """Benchmark a single game with specified number of moves"""
    game = OptimizedGoGame(board_size)
    player = player_class(net, simulations=simulations, is_self_play=True)
    
    start_time = time.time()
    moves_made = 0
    
    for i in range(num_moves):
        if game.game_over:
            break
            
        color = 'black' if game.current_player == BLACK else 'white'
        move = player.get_move(game, color, temperature=1.0)
        
        if move is None:
            game.pass_turn(color)
        else:
            game.make_move(move[0], move[1], color)
        moves_made += 1
    
    elapsed = time.time() - start_time
    return elapsed, moves_made

def benchmark_batch_games(batch_player, games, num_moves=20):
    """Benchmark batch processing of multiple games"""
    start_time = time.time()
    total_moves = 0
    
    for _ in range(num_moves):
        # Check if any games are over
        active_games = []
        active_colors = []
        
        for game in games:
            if not game.game_over:
                active_games.append(game)
                active_colors.append('black' if game.current_player == BLACK else 'white')
        
        if not active_games:
            break
        
        # Get moves for all active games
        moves = batch_player.get_batch_moves(active_games, active_colors, temperature=1.0)
        
        # Apply moves
        for game, move, color in zip(active_games, moves, active_colors):
            if move is None:
                game.pass_turn(color)
            else:
                game.make_move(move[0], move[1], color)
            total_moves += 1
    
    elapsed = time.time() - start_time
    return elapsed, total_moves

def main():
    print("AlphaGo Performance Benchmark")
    print("=" * 50)
    
    board_size = 9
    # Default to MPS on Mac, then CUDA, then CPU
    if torch.backends.mps.is_available():
        device = torch.device('mps')
    elif torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    print(f"Device: {device}")
    
    # Create network
    net = PolicyValueNet(board_size=board_size).to(device)
    net.eval()
    
    # Warm up
    print("\nWarming up...")
    game = OptimizedGoGame(board_size)
    player = AlphaGoPlayer(net, simulations=10, device=device)
    player.get_move(game, 'black')
    
    # Benchmark different configurations
    configs = [
        (50, 10),   # 50 simulations, 10 moves
        (100, 10),  # 100 simulations, 10 moves
        (200, 10),  # 200 simulations, 10 moves
        (100, 20),  # 100 simulations, 20 moves
    ]
    
    print("\nSingle Game Performance:")
    print("-" * 50)
    print(f"{'Simulations':<12} {'Moves':<8} {'Total Time':<12} {'Time/Move':<12}")
    print("-" * 50)
    
    for simulations, num_moves in configs:
        elapsed, moves_made = benchmark_single_game(
            AlphaGoPlayer, net, board_size, num_moves, simulations
        )
        time_per_move = elapsed / moves_made if moves_made > 0 else 0
        print(f"{simulations:<12} {moves_made:<8} {elapsed:<12.3f} {time_per_move:<12.3f}")
    
    # Benchmark batch processing
    print("\nBatch Processing Performance:")
    print("-" * 50)
    
    batch_sizes = [1, 2, 4, 8]
    simulations = 100
    num_moves = 10
    
    print(f"{'Batch Size':<12} {'Total Moves':<12} {'Total Time':<12} {'Time/Move':<12} {'Speedup':<12}")
    print("-" * 50)
    
    # Get baseline (batch size 1)
    baseline_player = BatchAlphaGoPlayer(net, simulations=simulations, device=device, batch_size=1)
    games = [OptimizedGoGame(board_size) for _ in range(1)]
    baseline_time, baseline_moves = benchmark_batch_games(baseline_player, games, num_moves)
    baseline_time_per_move = baseline_time / baseline_moves if baseline_moves > 0 else 0
    
    for batch_size in batch_sizes:
        batch_player = BatchAlphaGoPlayer(net, simulations=simulations, device=device, batch_size=batch_size)
        games = [OptimizedGoGame(board_size) for _ in range(batch_size)]
        
        elapsed, total_moves = benchmark_batch_games(batch_player, games, num_moves)
        time_per_move = elapsed / total_moves if total_moves > 0 else 0
        speedup = baseline_time_per_move / time_per_move if time_per_move > 0 else 0
        
        print(f"{batch_size:<12} {total_moves:<12} {elapsed:<12.3f} {time_per_move:<12.3f} {speedup:<12.2f}x")
    
    # Memory usage
    print("\nMemory Usage:")
    print("-" * 50)
    if torch.cuda.is_available():
        print(f"GPU Memory Allocated: {torch.cuda.memory_allocated(device) / 1024**2:.2f} MB")
        print(f"GPU Memory Cached: {torch.cuda.memory_reserved(device) / 1024**2:.2f} MB")
    
    # Compare with numpy version
    print("\nNumPy vs PyTorch Comparison:")
    print("-" * 50)
    
    # Create a simple numpy-based operation vs torch
    board = np.random.randint(0, 3, (board_size, board_size))
    
    # NumPy version
    start = time.time()
    for _ in range(1000):
        np_result = (board == 1).astype(np.float32)
    numpy_time = time.time() - start
    
    # PyTorch version
    board_tensor = torch.from_numpy(board).to(device)
    start = time.time()
    for _ in range(1000):
        torch_result = (board_tensor == 1).float()
        torch.cuda.synchronize() if torch.cuda.is_available() else None
    torch_time = time.time() - start
    
    print(f"NumPy time (1000 iterations): {numpy_time:.4f}s")
    print(f"PyTorch time (1000 iterations): {torch_time:.4f}s")
    print(f"PyTorch speedup: {numpy_time/torch_time:.2f}x")

if __name__ == "__main__":
    main()