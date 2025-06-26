#!/usr/bin/env python3
"""
Test script for progressive training system
"""

import subprocess
import sys
import time

def run_progressive_training():
    """Run a quick test of the progressive training system"""
    print("="*70)
    print("TESTING PROGRESSIVE TRAINING SYSTEM")
    print("="*70)
    print("\nThis test will:")
    print("1. Start with classic AI depth 1 (easiest)")
    print("2. Train until 75% win rate is achieved")
    print("3. Progress to harder depths automatically")
    print("4. Then switch to self-play training")
    print("\n" + "="*70)
    
    # Test with small parameters for quick demonstration
    cmd = [
        sys.executable,
        'self_play_progressive.py',
        '--board-size', '9',
        '--num-blocks', '3',  # Smaller network for faster training
        '--max-classic-depth', '3',  # Only go up to depth 3 for demo
        '--games-per-depth', '20',  # Fewer games for quick test
        '--consistency-threshold', '0.6',  # Lower threshold for demo
        '--consistency-games', '10',  # Fewer consistency check games
        '--num-iterations', '5',  # Just 5 self-play iterations
        '--games-per-iter', '10',  # Fewer games per iteration
        '--epochs-per-iter', '100',  # Fewer epochs
        '--num-workers', '4'  # Limit workers
    ]
    
    print("\nRunning command:")
    print(" ".join(cmd))
    print("\n" + "="*70 + "\n")
    
    try:
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, 
                                 universal_newlines=True, bufsize=1)
        
        # Stream output in real-time
        for line in iter(process.stdout.readline, ''):
            print(line, end='')
        
        process.wait()
        
        if process.returncode == 0:
            print("\n" + "="*70)
            print("TEST COMPLETED SUCCESSFULLY!")
            print("="*70)
        else:
            print(f"\nTest failed with return code: {process.returncode}")
            
    except KeyboardInterrupt:
        print("\n\nTest interrupted by user")
        process.terminate()
        process.wait()
    except Exception as e:
        print(f"\nError running test: {e}")

def check_dependencies():
    """Check if required files exist"""
    required_files = [
        'self_play_progressive.py',
        'optimized_go.py',
        'alpha_go.py',
        'classic_go_ai.py'
    ]
    
    missing = []
    for file in required_files:
        try:
            with open(file, 'r'):
                pass
        except FileNotFoundError:
            missing.append(file)
    
    if missing:
        print("Error: Missing required files:")
        for file in missing:
            print(f"  - {file}")
        return False
    
    return True

if __name__ == '__main__':
    print("Progressive Training Test Script")
    print("================================\n")
    
    if not check_dependencies():
        sys.exit(1)
    
    print("All dependencies found. Starting test...\n")
    time.sleep(1)
    
    run_progressive_training()