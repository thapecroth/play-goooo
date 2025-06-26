#!/usr/bin/env python3
"""
Quick benchmark comparison between Python and Codon Go implementations
"""

import time
import subprocess
from go_game_codon_fast import FastGoGame

def benchmark_python_fast(iterations=100):
    """Quick benchmark of Python FastGoGame"""
    game = FastGoGame(9)
    
    # Add some stones for a more realistic scenario
    game.board[3][3] = 1
    game.board[3][4] = 2
    game.board[4][3] = 2
    game.board[4][4] = 1
    
    start = time.time()
    total_moves = 0
    
    for _ in range(iterations):
        moves = game.get_valid_moves()
        total_moves += len(moves)
    
    elapsed = time.time() - start
    return elapsed, total_moves

def benchmark_compiled_binary():
    """Benchmark the compiled binary"""
    try:
        start = time.time()
        result = subprocess.run(['./go_game_fast_compiled'], 
                              capture_output=True, text=True)
        elapsed = time.time() - start
        
        if result.returncode == 0:
            return elapsed, result.stdout
        else:
            return None, "Binary execution failed"
    except FileNotFoundError:
        return None, "Compiled binary not found"

def main():
    print("=" * 60)
    print("Quick Go Implementation Benchmark")
    print("=" * 60)
    
    # Python benchmark
    print("\n1. Python FastGoGame (100 move generations):")
    py_time, py_moves = benchmark_python_fast(100)
    print(f"   Time: {py_time:.4f}s")
    print(f"   Total moves found: {py_moves}")
    print(f"   Moves/second: {py_moves/py_time:.0f}")
    
    # Compiled binary benchmark (runs 1000 iterations internally)
    print("\n2. Codon Compiled Binary (1000 move generations):")
    bin_time, bin_output = benchmark_compiled_binary()
    
    if bin_time:
        print(f"   Total execution time: {bin_time:.4f}s")
        # Extract moves from output
        for line in bin_output.split('\n'):
            if "Total moves found:" in line:
                moves = int(line.split(": ")[1])
                # The binary does 1000 iterations
                print(f"   Moves/second: {moves/bin_time:.0f}")
                
                # Estimate speedup (accounting for 10x more iterations)
                python_1000_estimate = py_time * 10
                speedup = python_1000_estimate / bin_time
                print(f"\n   Estimated speedup: {speedup:.1f}x faster than Python")
    else:
        print(f"   Error: {bin_output}")
    
    # Simple move making benchmark
    print("\n3. Move Making Speed Test:")
    game = FastGoGame(9)
    
    start = time.time()
    moves_made = 0
    for i in range(50):
        x, y = i % 9, (i * 2) % 9
        if game.make_move(x, y):
            moves_made += 1
    py_move_time = time.time() - start
    
    print(f"   Python: {moves_made} moves in {py_move_time:.4f}s")
    print(f"   Speed: {moves_made/py_move_time:.0f} moves/second")
    
    print("\n" + "=" * 60)
    print("Summary: The Codon-compiled version should be significantly")
    print("faster for compute-intensive operations like move generation")
    print("and game simulation.")
    print("=" * 60)

if __name__ == "__main__":
    main() 