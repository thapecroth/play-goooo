#!/usr/bin/env python3
"""Test Codon compilation vs Python performance"""

import time
import subprocess
import sys

def test_python():
    """Test Python version"""
    start = time.time()
    result = subprocess.run([sys.executable, "-c", """
import go_ai_codon_simple
board = [[0]*9 for _ in range(9)]
for i in range(100):
    move = go_ai_codon_simple.get_best_move(board, 1, 9)
    score = go_ai_codon_simple.evaluate_position(board, 1, 9)
"""], capture_output=True, text=True)
    
    if result.returncode != 0:
        print(f"Python error: {result.stderr}")
        return None
    
    return time.time() - start

def test_codon():
    """Test Codon compiled version"""
    # First compile if needed
    subprocess.run(["codon", "build", "-release", "-o", "go_ai_codon_test", "go_ai_codon_simple.py"], 
                   capture_output=True)
    
    start = time.time()
    result = subprocess.run(["./go_ai_codon_test"], capture_output=True, text=True)
    
    if result.returncode != 0:
        print(f"Codon error: {result.stderr}")
        return None
        
    return time.time() - start

def main():
    print("Testing Go AI performance: Python vs Codon\n")
    
    # Test Python
    print("Running Python version...")
    python_time = test_python()
    if python_time:
        print(f"Python time: {python_time:.4f}s")
    
    # Test Codon
    print("\nRunning Codon version...")
    codon_time = test_codon()
    if codon_time:
        print(f"Codon time: {codon_time:.4f}s")
    
    # Compare
    if python_time and codon_time:
        speedup = python_time / codon_time
        print(f"\nCodon speedup: {speedup:.1f}x faster than Python")

if __name__ == "__main__":
    main()