#!/usr/bin/env python3
"""
Benchmark comparison between Python and Codon-compiled Go game implementations
"""

import time
import numpy as np
from optimized_go import OptimizedGoGame as PythonGoGame
from go_game_codon_fast import FastGoGame as PythonFastGoGame

def benchmark_python_optimized(board_size=9, num_games=10):
    """Benchmark the original Python OptimizedGoGame"""
    start_time = time.time()
    total_moves = 0
    
    for _ in range(num_games):
        game = PythonGoGame(board_size)
        moves = 0
        max_moves = board_size * board_size
        
        while moves < max_moves and not game.game_over:
            # Simple random play
            valid_positions = []
            for y in range(board_size):
                for x in range(board_size):
                    if game.board[y, x] == 0:
                        valid_positions.append((x, y))
            
            if valid_positions:
                x, y = valid_positions[moves % len(valid_positions)]
                color = 'black' if moves % 2 == 0 else 'white'
                if game.make_move(x, y, color):
                    moves += 1
                else:
                    game.pass_turn(color)
            else:
                color = 'black' if moves % 2 == 0 else 'white'
                game.pass_turn(color)
            
            total_moves += 1
    
    elapsed = time.time() - start_time
    return elapsed, total_moves

def benchmark_python_fast(board_size=9, num_games=10):
    """Benchmark the Python FastGoGame"""
    start_time = time.time()
    total_moves = 0
    
    for _ in range(num_games):
        game = PythonFastGoGame(board_size)
        moves = 0
        max_moves = board_size * board_size
        
        while moves < max_moves and not game.game_over:
            valid_moves = game.get_valid_moves()
            
            if valid_moves:
                x, y = valid_moves[moves % len(valid_moves)]
                if game.make_move(x, y):
                    moves += 1
            else:
                game.pass_turn()
            
            total_moves += 1
    
    elapsed = time.time() - start_time
    return elapsed, total_moves

def benchmark_move_generation(board_size=9, num_positions=100):
    """Benchmark move generation speed"""
    print(f"\nMove Generation Benchmark (board size: {board_size}, positions: {num_positions})")
    print("-" * 60)
    
    # Python OptimizedGoGame
    game1 = PythonGoGame(board_size)
    # Add some random stones
    for i in range(20):
        x, y = i % board_size, (i * 3) % board_size
        if game1.board[y, x] == 0:
            game1.board[y, x] = 1 if i % 2 == 0 else 2
    
    start = time.time()
    for _ in range(num_positions):
        # Count valid moves manually
        count = 0
        for y in range(board_size):
            for x in range(board_size):
                if game1.board[y, x] == 0:
                    count += 1
    python_opt_time = time.time() - start
    
    # Python FastGoGame
    game2 = PythonFastGoGame(board_size)
    # Copy board state
    for y in range(board_size):
        for x in range(board_size):
            game2.board[y][x] = int(game1.board[y, x])
    
    start = time.time()
    for _ in range(num_positions):
        moves = game2.get_valid_moves()
    python_fast_time = time.time() - start
    
    print(f"Python OptimizedGoGame: {python_opt_time:.4f}s")
    print(f"Python FastGoGame: {python_fast_time:.4f}s")
    print(f"Speedup: {python_opt_time/python_fast_time:.2f}x")

def main():
    print("=" * 60)
    print("Go Game Implementation Benchmark")
    print("=" * 60)
    
    # Test different board sizes
    board_sizes = [9, 13]
    num_games = 5
    
    for board_size in board_sizes:
        print(f"\nFull Game Simulation (board size: {board_size}, games: {num_games})")
        print("-" * 60)
        
        # Benchmark Python OptimizedGoGame
        print("Running Python OptimizedGoGame...")
        py_opt_time, py_opt_moves = benchmark_python_optimized(board_size, num_games)
        print(f"Time: {py_opt_time:.3f}s, Total moves: {py_opt_moves}")
        
        # Benchmark Python FastGoGame
        print("\nRunning Python FastGoGame...")
        py_fast_time, py_fast_moves = benchmark_python_fast(board_size, num_games)
        print(f"Time: {py_fast_time:.3f}s, Total moves: {py_fast_moves}")
        
        # Calculate speedup
        print(f"\nSpeedup (FastGoGame vs OptimizedGoGame): {py_opt_time/py_fast_time:.2f}x")
    
    # Move generation benchmark
    benchmark_move_generation(9, 1000)
    benchmark_move_generation(13, 500)
    
    print("\n" + "=" * 60)
    print("Note: To see Codon speedup, run: ./go_game_fast_compiled")
    print("The compiled version should be 10-100x faster than Python")
    print("=" * 60)

if __name__ == "__main__":
    main() 