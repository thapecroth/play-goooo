#!/usr/bin/env python3
"""
Test the two-stage training system
"""

import torch
import time
from self_play_two_stage import TwoStageTrainer, get_device

def test_device():
    """Test device selection"""
    print("Testing device selection...")
    device = get_device()
    print(f"Selected device: {device}")
    return device

def test_warmup_phase():
    """Test warmup against classic AI"""
    print("\n" + "="*60)
    print("Testing Warmup Phase")
    print("="*60)
    
    trainer = TwoStageTrainer(board_size=5, num_blocks=2, num_workers=2)
    
    # Test with just 2 games for speed
    start_time = time.time()
    positions = trainer.warmup_against_classic(num_games=2, simulations=10, ai_depth=1)
    elapsed = time.time() - start_time
    
    print(f"\nWarmup test completed in {elapsed:.1f}s")
    print(f"Positions collected: {positions}")
    print(f"Buffer size: {len(trainer.replay_buffer)}")

def test_two_stage_iteration():
    """Test a complete two-stage iteration"""
    print("\n" + "="*60)
    print("Testing Two-Stage Iteration")
    print("="*60)
    
    trainer = TwoStageTrainer(board_size=5, num_blocks=2, num_workers=2)
    
    # Skip warmup for this test
    print("\nRunning single iteration...")
    start_time = time.time()
    
    improved, win_ratio = trainer.run_iteration(
        iteration=1,
        num_games=2,      # Just 2 games
        num_epochs=10,    # Just 10 epochs
        batch_size=16,
        eval_games=2,
        win_ratio_to_update=0.4  # Lower threshold for testing
    )
    
    elapsed = time.time() - start_time
    print(f"\nIteration completed in {elapsed:.1f}s")
    print(f"Model improved: {improved}")
    print(f"Win ratio: {win_ratio:.2%}")

def test_data_collection_only():
    """Test just the data collection stage"""
    print("\n" + "="*60)
    print("Testing Data Collection Stage")
    print("="*60)
    
    trainer = TwoStageTrainer(board_size=5, num_blocks=2, num_workers=4)
    
    start_time = time.time()
    positions = trainer.collect_self_play_data(num_games=4, simulations=10)
    elapsed = time.time() - start_time
    
    print(f"\nData collection completed in {elapsed:.1f}s")
    print(f"Positions collected: {positions}")
    print(f"Average time per game: {elapsed/4:.1f}s")

def test_training_only():
    """Test just the training stage"""
    print("\n" + "="*60)
    print("Testing Training Stage")
    print("="*60)
    
    trainer = TwoStageTrainer(board_size=5, num_blocks=2)
    
    # First collect some data
    print("Collecting initial data...")
    trainer.collect_self_play_data(num_games=2, simulations=10)
    
    # Then test training
    print("\nTesting training...")
    start_time = time.time()
    losses = trainer.train_network(num_epochs=20, batch_size=16)
    elapsed = time.time() - start_time
    
    print(f"\nTraining completed in {elapsed:.1f}s")
    print(f"Total loss updates: {len(losses)}")

if __name__ == "__main__":
    print("="*60)
    print("Two-Stage Training System Test")
    print("="*60)
    
    # Test 1: Device selection
    device = test_device()
    
    # Test 2: Warmup phase
    test_warmup_phase()
    
    # Test 3: Data collection
    test_data_collection_only()
    
    # Test 4: Training stage
    test_training_only()
    
    # Test 5: Full iteration
    test_two_stage_iteration()
    
    print("\n" + "="*60)
    print("All tests completed!")
    print("="*60)