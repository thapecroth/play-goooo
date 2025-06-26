#!/usr/bin/env python3
"""
Test parallel self-play with MPS support
"""

import torch
import time
import multiprocessing as mp

def test_device_selection():
    """Test device selection priority"""
    print("Testing device selection...")
    
    if torch.backends.mps.is_available():
        print("✓ MPS (Metal Performance Shaders) is available")
        device = torch.device('mps')
    elif torch.cuda.is_available():
        print("✓ CUDA is available")
        device = torch.device('cuda')
    else:
        print("✓ Using CPU")
        device = torch.device('cpu')
    
    print(f"Selected device: {device}")
    
    # Test tensor operations
    try:
        x = torch.randn(10, 10).to(device)
        y = torch.randn(10, 10).to(device)
        z = torch.matmul(x, y)
        print(f"✓ Tensor operations work on {device}")
    except Exception as e:
        print(f"✗ Error with tensor operations: {e}")
    
    return device

def test_parallel_capability():
    """Test multiprocessing capability"""
    print("\nTesting parallel processing capability...")
    
    cpu_count = mp.cpu_count()
    print(f"CPU cores available: {cpu_count}")
    print(f"Recommended workers: {min(cpu_count, 16)}")
    
    return cpu_count

def test_quick_parallel_game():
    """Quick test of parallel game generation"""
    print("\nTesting parallel game generation...")
    
    from self_play_parallel import SelfPlayTrainer
    
    # Create trainer with small board for quick test
    trainer = SelfPlayTrainer(board_size=5, num_blocks=2)
    
    # Test sequential
    print("\nTesting sequential collection (2 games)...")
    start_time = time.time()
    trainer.collect_game_data(2)
    seq_time = time.time() - start_time
    print(f"Sequential time: {seq_time:.2f}s")
    
    # Clear buffer
    trainer.replay_buffer.clear()
    
    # Test parallel (only if not MPS, as MPS doesn't work well with multiprocessing)
    if trainer.device.type != 'mps':
        print("\nTesting parallel collection (2 games)...")
        start_time = time.time()
        trainer.collect_game_data_parallel(2, simulations=10)
        par_time = time.time() - start_time
        print(f"Parallel time: {par_time:.2f}s")
        print(f"Speedup: {seq_time/par_time:.2f}x")
    else:
        print("\nSkipping parallel test on MPS device (not compatible with multiprocessing)")

def test_model_operations():
    """Test model operations on selected device"""
    print("\nTesting model operations...")
    
    from alpha_go import PolicyValueNet
    from optimized_go import OptimizedGoGame
    
    device = test_device_selection()
    
    # Create small model
    model = PolicyValueNet(board_size=5, num_blocks=2)
    model.to(device)
    model.eval()
    
    # Test forward pass
    try:
        with torch.no_grad():
            dummy_input = torch.randn(1, 3, 5, 5).to(device)
            policy, value = model(dummy_input)
            print(f"✓ Model forward pass works on {device}")
            print(f"  Policy shape: {policy.shape}")
            print(f"  Value shape: {value.shape}")
    except Exception as e:
        print(f"✗ Error with model forward pass: {e}")

if __name__ == "__main__":
    print("="*60)
    print("Parallel Self-Play Test Suite")
    print("="*60)
    
    # Test 1: Device selection
    device = test_device_selection()
    
    # Test 2: Parallel capability
    cpu_count = test_parallel_capability()
    
    # Test 3: Model operations
    test_model_operations()
    
    # Test 4: Quick parallel game (if enough time)
    test_quick_parallel_game()
    
    print("\n" + "="*60)
    print("Test completed!")
    print("="*60)