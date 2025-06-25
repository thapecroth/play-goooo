#!/usr/bin/env python3
import time
import torch
import numpy as np
from alpha_go import PolicyValueNet, AlphaGoPlayer
from optimized_go import OptimizedGoGame, BLACK

def quick_benchmark():
    print("Quick AlphaGo Performance Test")
    print("=" * 40)
    
    board_size = 9
    # Default to MPS on Mac, then CUDA, then CPU
    if torch.backends.mps.is_available():
        device = torch.device('mps')
    elif torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    print(f"Device: {device}\n")
    
    # Create network
    net = PolicyValueNet(board_size=board_size).to(device)
    net.eval()
    
    # Test 1: Move generation speed
    print("Test 1: Move Generation Speed")
    print("-" * 40)
    
    game = OptimizedGoGame(board_size)
    player = AlphaGoPlayer(net, simulations=50, device=device)
    
    # Warm up
    player.get_move(game, 'black')
    
    # Time 5 moves
    start = time.time()
    for i in range(5):
        move = player.get_move(game, 'black' if i % 2 == 0 else 'white')
        if move:
            game.make_move(move[0], move[1], 'black' if i % 2 == 0 else 'white')
    
    elapsed = time.time() - start
    print(f"5 moves with 50 simulations: {elapsed:.3f}s")
    print(f"Average time per move: {elapsed/5:.3f}s\n")
    
    # Test 2: Tensor conversion speed
    print("Test 2: Tensor Conversion Speed")
    print("-" * 40)
    
    # NumPy vs PyTorch comparison
    board = np.random.randint(0, 3, (board_size, board_size))
    
    # NumPy operations
    start = time.time()
    for _ in range(10000):
        current = (board == 1).astype(np.float32)
        opponent = (board == 2).astype(np.float32)
    numpy_time = time.time() - start
    
    # PyTorch operations
    board_tensor = torch.from_numpy(board).to(device)
    start = time.time()
    for _ in range(10000):
        current = (board_tensor == 1).float()
        opponent = (board_tensor == 2).float()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        elif torch.backends.mps.is_available():
            torch.mps.synchronize()
    torch_time = time.time() - start
    
    print(f"NumPy (10k iterations): {numpy_time:.4f}s")
    print(f"PyTorch (10k iterations): {torch_time:.4f}s")
    print(f"Speedup: {numpy_time/torch_time:.2f}x\n")
    
    # Test 3: Network inference speed
    print("Test 3: Neural Network Inference")
    print("-" * 40)
    
    batch_sizes = [1, 4, 8]
    
    for batch_size in batch_sizes:
        input_tensor = torch.randn(batch_size, 3, board_size, board_size).to(device)
        
        # Time inference
        start = time.time()
        with torch.no_grad():
            for _ in range(100):
                policy, value = net(input_tensor)
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                elif torch.backends.mps.is_available():
                    torch.mps.synchronize()
        
        elapsed = time.time() - start
        print(f"Batch size {batch_size}: {elapsed:.3f}s for 100 inferences")
        print(f"  {elapsed/100*1000:.2f}ms per batch, {elapsed/100/batch_size*1000:.2f}ms per sample")
    
    print("\n✓ Benchmark completed successfully!")

if __name__ == "__main__":
    quick_benchmark()