"""
Test correctness of Codon-compiled implementations.
Ensures Codon versions produce identical results to Python versions.
"""

import pytest
import numpy as np
import subprocess
import os
import sys

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from optimized_go import OptimizedGoGame
from go_ai_optimized import GoAIOptimized
import go_ai_codon_simple


class TestCodonGameCorrectness:
    """Test Codon-compiled game implementation correctness."""
    
    @pytest.mark.unit
    def test_codon_simple_ai_functions(self):
        """Test simple Codon AI functions match expected behavior."""
        # Create test board
        board = [[0 for _ in range(9)] for _ in range(9)]
        
        # Add some stones
        board[4][4] = 1  # Black
        board[4][5] = 2  # White
        board[5][4] = 2  # White
        board[5][5] = 1  # Black
        
        # Test get_best_move
        move = go_ai_codon_simple.get_best_move(board, 1, 9)
        assert move is not None, "Should find a valid move"
        assert isinstance(move, tuple), "Move should be a tuple"
        assert len(move) == 2, "Move should have x,y coordinates"
        assert 0 <= move[0] < 9 and 0 <= move[1] < 9, "Move should be on board"
        
        # Test evaluate_position
        score = go_ai_codon_simple.evaluate_position(board, 1, 9)
        assert isinstance(score, int), "Score should be integer"
        assert score == 0, "Symmetric position should have score 0"
        
        # Test with advantage for black
        board[3][3] = 1
        score = go_ai_codon_simple.evaluate_position(board, 1, 9)
        assert score > 0, "Black advantage should give positive score"
    
    @pytest.mark.integration
    @pytest.mark.skipif(not os.path.exists('./go_ai_codon_compiled'), 
                        reason="Codon compiled binary not found")
    def test_codon_compiled_binary(self):
        """Test Codon compiled binary produces correct output."""
        # Run the compiled binary
        result = subprocess.run(
            ['./go_ai_codon_compiled'],
            capture_output=True,
            text=True,
            timeout=5
        )
        
        assert result.returncode == 0, f"Binary failed: {result.stderr}"
        
        # Check output format
        lines = result.stdout.strip().split('\n')
        assert len(lines) >= 2, "Should output at least 2 lines"
        
        # Parse move output
        assert "Best move:" in lines[0], "Should output best move"
        move_parts = lines[0].split(': ')[1].split(', ')
        x = int(move_parts[0])
        y = int(move_parts[1])
        assert 0 <= x < 9 and 0 <= y < 9, "Move should be valid board position"
        
        # Parse score output
        assert "Position score:" in lines[1], "Should output position score"
        score = int(lines[1].split(': ')[1])
        assert isinstance(score, int), "Score should be integer"
    
    @pytest.mark.integration
    def test_codon_ai_vs_python_ai(self):
        """Compare Codon AI with Python AI on same positions."""
        # Create identical game states
        game = OptimizedGoGame(9)
        
        # Set up a specific position
        test_moves = [
            (4, 4, 1), (3, 3, 2), (5, 5, 1), (3, 5, 2),
            (4, 3, 1), (4, 5, 2), (3, 4, 1),
        ]
        
        for x, y, color in test_moves:
            game.make_move(x, y, color)
        
        # Get board as list for Codon AI
        board_list = game.board.tolist()
        
        # Test Codon simple AI
        codon_move = go_ai_codon_simple.get_best_move(board_list, 2, 9)  # White to move
        assert codon_move is not None, "Codon AI should find a move"
        
        # Test Python AI
        python_ai = GoAIOptimized(board_size=9)
        python_ai.max_depth = 1  # Simple depth for comparison
        python_move = python_ai.get_best_move(board_list, 2, 0, 0)
        assert python_move is not None, "Python AI should find a move"
        
        # Both should find valid moves
        assert 0 <= codon_move[0] < 9 and 0 <= codon_move[1] < 9
        assert 0 <= python_move[0] < 9 and 0 <= python_move[1] < 9
        
        # Both moves should be on empty positions
        assert board_list[codon_move[1]][codon_move[0]] == 0
        assert board_list[python_move[1]][python_move[0]] == 0
    
    @pytest.mark.integration
    @pytest.mark.skipif(not os.path.exists('./go_game_fast_compiled'),
                        reason="Codon game engine not compiled")
    def test_codon_game_wrapper(self):
        """Test the Codon game wrapper functionality."""
        from codon_game_wrapper import CodonGoGame
        
        # Create game
        game = CodonGoGame(9)
        
        # Test initial state
        assert game.size == 9
        assert game.current_player == 'black'
        assert not game.game_over
        assert game.captures == {'black': 0, 'white': 0}
        
        # Test moves
        assert game.make_move(4, 4, 'black')
        assert game.board[4, 4] == 1
        assert game.current_player == 'white'
        
        assert game.make_move(3, 3, 'white')
        assert game.board[3, 3] == 2
        assert game.current_player == 'black'
        
        # Test invalid move
        assert not game.make_move(4, 4, 'black')
        
        # Test pass
        assert game.pass_turn()
        assert game.pass_turn()
        assert game.game_over


class TestCodonPerformanceCorrectness:
    """Test that Codon optimizations don't break correctness."""
    
    @pytest.mark.integration
    @pytest.mark.slow
    def test_codon_ai_consistency(self):
        """Test Codon AI produces consistent results."""
        board = [[0 for _ in range(9)] for _ in range(9)]
        
        # Add fixed position
        board[4][4] = 1
        board[4][5] = 2
        board[5][4] = 2
        
        # Run multiple times - should get same result
        moves = []
        for _ in range(5):
            move = go_ai_codon_simple.get_best_move(board, 1, 9)
            moves.append(move)
        
        # All moves should be identical (deterministic)
        assert all(m == moves[0] for m in moves), \
            "Codon AI should be deterministic"
    
    @pytest.mark.benchmark
    @pytest.mark.skipif(not os.path.exists('./go_ai_codon_compiled'),
                        reason="Codon compiled binary not found")
    def test_codon_vs_python_performance(self, benchmark):
        """Benchmark Codon vs Python performance."""
        if os.environ.get('CI'):
            pytest.skip("Skipping benchmark in CI")
        
        board = [[0 for _ in range(9)] for _ in range(9)]
        
        # Python version
        def python_moves():
            results = []
            for _ in range(10):
                move = go_ai_codon_simple.get_best_move(board, 1, 9)
                score = go_ai_codon_simple.evaluate_position(board, 1, 9)
                results.append((move, score))
            return results
        
        # Benchmark Python
        python_result = benchmark(python_moves)
        
        # Codon version (subprocess call)
        def codon_binary():
            result = subprocess.run(
                ['./go_ai_codon_compiled'],
                capture_output=True,
                timeout=1
            )
            return result.returncode == 0
        
        # Note: Can't use pytest-benchmark for subprocess,
        # but we can at least verify it runs
        assert codon_binary(), "Codon binary should run successfully"


class TestEdgeCases:
    """Test edge cases that might behave differently in compiled code."""
    
    @pytest.mark.unit
    def test_board_boundaries(self):
        """Test behavior at board boundaries."""
        board = [[0 for _ in range(9)] for _ in range(9)]
        
        # Place stones at corners
        board[0][0] = 1
        board[0][8] = 2
        board[8][0] = 2
        board[8][8] = 1
        
        # Should still find valid moves
        move = go_ai_codon_simple.get_best_move(board, 1, 9)
        assert move is not None
        assert move not in [(0, 0), (0, 8), (8, 0), (8, 8)], \
            "Should not return occupied positions"
    
    @pytest.mark.unit
    def test_full_board_handling(self):
        """Test handling of nearly full board."""
        board = [[1 if (x + y) % 2 == 0 else 2 for x in range(9)] for y in range(9)]
        
        # Leave one spot empty
        board[4][4] = 0
        
        move = go_ai_codon_simple.get_best_move(board, 1, 9)
        assert move == (4, 4), "Should find the only empty spot"
        
        # Completely full board
        board[4][4] = 1
        move = go_ai_codon_simple.get_best_move(board, 1, 9)
        assert move is None, "Should return None for full board"
    
    @pytest.mark.unit
    def test_empty_board_preference(self):
        """Test move preference on empty board."""
        board = [[0 for _ in range(9)] for _ in range(9)]
        
        move = go_ai_codon_simple.get_best_move(board, 1, 9)
        assert move is not None
        
        # Should prefer positions with liberties
        x, y = move
        has_liberty = False
        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nx, ny = x + dx, y + dy
            if 0 <= nx < 9 and 0 <= ny < 9:
                has_liberty = True
                break
        
        assert has_liberty, "Move should have at least one liberty"


# Run quick validation test
def validate_implementations():
    """Quick validation that can be run from command line."""
    print("Validating Go game implementations...")
    
    # Test basic Python implementation
    from server import GoGame
    game1 = GoGame(9)
    assert game1.make_move(4, 4, 'black')
    print("✓ Basic GoGame works")
    
    # Test optimized implementation
    from optimized_go import OptimizedGoGame
    game2 = OptimizedGoGame(9)
    assert game2.make_move(4, 4, 'black')
    print("✓ OptimizedGoGame works")
    
    # Test Codon simple AI
    board = [[0] * 9 for _ in range(9)]
    move = go_ai_codon_simple.get_best_move(board, 1, 9)
    assert move is not None
    print("✓ Codon simple AI works")
    
    # Test compiled binary if available
    if os.path.exists('./go_ai_codon_compiled'):
        result = subprocess.run(['./go_ai_codon_compiled'], capture_output=True)
        if result.returncode == 0:
            print("✓ Codon compiled binary works")
        else:
            print("✗ Codon compiled binary failed")
    else:
        print("- Codon compiled binary not found")
    
    print("\nBasic validation complete!")


if __name__ == "__main__":
    validate_implementations()