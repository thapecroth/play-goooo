#!/usr/bin/env python3
"""
Demo of the two-stage training system
"""

import sys
import os

# Add the project directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from self_play_two_stage import TwoStageTrainer

def main():
    print("="*70)
    print("Two-Stage AlphaGo Training Demo")
    print("="*70)
    print("\nThis demo shows:")
    print("1. Warmup phase: Learn from classic Go AI")
    print("2. Data collection: Parallel self-play games")
    print("3. Training phase: Update neural network")
    print("4. Evaluation: Test against previous best model")
    print("\n" + "="*70)
    
    # Create trainer with small settings for demo
    trainer = TwoStageTrainer(
        board_size=9,      # 9x9 board for faster demo
        num_blocks=3,      # Smaller network
        learning_rate=1e-3,
        buffer_size=10000,
        num_workers=4      # Use 4 workers
    )
    
    # Phase 1: Warmup against classic AI
    print("\n" + "="*70)
    input("Press Enter to start WARMUP phase...")
    trainer.warmup_against_classic(
        num_games=10,      # Play 10 games
        simulations=20,    # Quick MCTS
        ai_depth=1        # Fast classic AI
    )
    
    # Run one training iteration
    print("\n" + "="*70)
    input("Press Enter to start TRAINING iteration...")
    
    improved, win_ratio = trainer.run_iteration(
        iteration=1,
        num_games=8,       # 8 self-play games
        num_epochs=100,    # 100 training epochs
        batch_size=32,
        eval_games=4,      # 4 evaluation games
        win_ratio_to_update=0.55
    )
    
    # Summary
    print("\n" + "="*70)
    print("DEMO COMPLETE!")
    print("="*70)
    print(f"\nResults:")
    print(f"- Warmup win rate: {trainer.warmup_wins}/{trainer.warmup_games} ({trainer.warmup_wins/trainer.warmup_games*100:.1f}%)")
    print(f"- Buffer size: {len(trainer.replay_buffer)} positions")
    print(f"- Final evaluation: {win_ratio:.2%} win rate")
    print(f"- Model improved: {'Yes' if improved else 'No'}")
    
    if improved:
        trainer.save_model('demo_model.pth')
        print("\nModel saved as 'demo_model.pth'")

if __name__ == '__main__':
    main()