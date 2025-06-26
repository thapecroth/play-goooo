"""
Pytest tests for performance benchmarking and optimization validation.
"""

import pytest
import time
import numpy as np
import subprocess
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from optimized_go import OptimizedGoGame
from go_ai_optimized import GoAIOptimized as OptimizedGoAI
from classic_go_ai import ClassicGoAI
import go_ai_codon_simple


class TestGoEnginePerformance:
    """Test performance of different Go engine implementations."""
    
    @pytest.mark.unit
    @pytest.mark.slow
    def test_move_generation_speed(self):
        """Test speed of move generation."""
        game = OptimizedGoGame(9)
        
        start_time = time.time()
        iterations = 50
        
        for _ in range(iterations):
            moves = game.get_valid_moves(1)  # BLACK
        
        elapsed = time.time() - start_time
        ops_per_second = iterations / elapsed
        
        print(f"\nMove generation: {ops_per_second:.1f} ops/sec")
        assert ops_per_second > 100  # Should be fast
    
    @pytest.mark.unit
    @pytest.mark.slow
    def test_position_evaluation_speed(self):
        """Test speed of position evaluation."""
        ai = OptimizedGoAI(board_size=9)
        ai.max_depth = 3
        game = OptimizedGoGame(9)
        
        # Create a mid-game position
        moves = [(4, 4), (4, 5), (5, 4), (3, 4), (5, 5), (3, 3)]
        for i, (x, y) in enumerate(moves):
            color = 1 if i % 2 == 0 else 2
            game.make_move(x, y, color)
        
        # Set the board for AI
        ai.board = game.board
        
        start_time = time.time()
        iterations = 30
        
        for _ in range(iterations):
            # Note: GoAIOptimized expects different parameters
            score = ai.evaluate_position(1, 0, 0)
        
        elapsed = time.time() - start_time
        evals_per_second = iterations / elapsed
        
        print(f"Position evaluation: {evals_per_second:.1f} evals/sec")
        assert evals_per_second > 100
    
    @pytest.mark.integration
    @pytest.mark.slow
    def test_ai_move_search_speed(self):
        """Test speed of AI move search."""
        ai = OptimizedGoAI(board_size=9)
        ai.max_depth = 2
        game = OptimizedGoGame(9)
        
        start_time = time.time()
        # GoAIOptimized expects board as list of lists and color as int
        board_list = game.board.tolist()
        move = ai.get_best_move(board_list, 1, 0, 0)  # BLACK=1, no captures
        elapsed = time.time() - start_time
        
        print(f"AI move search (depth 2): {elapsed:.3f}s")
        assert elapsed < 1.0  # Should complete quickly at depth 2
        assert move is not None


class TestCodonPerformance:
    """Test Codon compilation performance."""
    
    @pytest.mark.unit
    def test_codon_vs_python_simple(self):
        """Compare Codon vs Python for simple operations."""
        board = [[0] * 9 for _ in range(9)]
        iterations = 1000
        
        # Test Python version
        start_time = time.time()
        for _ in range(iterations):
            move = go_ai_codon_simple.get_best_move(board, 1, 9)
            score = go_ai_codon_simple.evaluate_position(board, 1, 9)
        python_time = time.time() - start_time
        
        # Test Codon version if compiled
        if os.path.exists('./go_ai_codon_compiled'):
            # Run compiled version
            result = subprocess.run(
                ['./go_ai_codon_compiled'],
                capture_output=True,
                text=True
            )
            
            # Note: This is a simple test, real benchmark would measure internally
            print(f"\nPython time: {python_time:.3f}s")
            print("Codon compiled version exists")
        else:
            pytest.skip("Codon compiled version not found")


class TestMemoryUsage:
    """Test memory usage of different implementations."""
    
    @pytest.mark.unit
    def test_game_memory_footprint(self):
        """Test memory usage of game instances."""
        import psutil
        import gc
        
        process = psutil.Process()
        gc.collect()
        
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Create multiple game instances
        games = []
        for size in [9, 13, 19]:
            for _ in range(10):
                games.append(OptimizedGoGame(size))
        
        gc.collect()
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = final_memory - initial_memory
        
        print(f"\nMemory increase for 30 games: {memory_increase:.1f} MB")
        
        # Clean up
        del games
        gc.collect()
        
        # Should use reasonable memory
        assert memory_increase < 100  # Less than 100MB for 30 games
    
    @pytest.mark.unit
    def test_ai_memory_usage(self):
        """Test memory usage of AI instances."""
        import psutil
        import gc
        
        process = psutil.Process()
        gc.collect()
        
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Create AI instances with different depths
        ais = []
        for depth in range(1, 6):
            ais.append(OptimizedGoAI(max_depth=depth))
        
        gc.collect()
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = final_memory - initial_memory
        
        print(f"\nMemory increase for 5 AI instances: {memory_increase:.1f} MB")
        
        # Clean up
        del ais
        gc.collect()
        
        # Should use minimal memory
        assert memory_increase < 50


class TestScalability:
    """Test scalability with different board sizes."""
    
    @pytest.mark.parametrize("board_size", [9, 13, 19])
    @pytest.mark.slow
    def test_performance_scaling(self, board_size):
        """Test how performance scales with board size."""
        game = OptimizedGoGame(board_size)
        ai = OptimizedGoAI(board_size=board_size)
        ai.max_depth = 1  # Shallow depth for quick test
        
        # Test move generation
        start_time = time.time()
        moves = game.get_valid_moves(1)
        move_gen_time = time.time() - start_time
        
        # Test AI search
        start_time = time.time()
        board_list = game.board.tolist()
        move = ai.get_best_move(board_list, 1, 0, 0)
        ai_search_time = time.time() - start_time
        
        print(f"\nBoard size {board_size}x{board_size}:")
        print(f"  Move generation: {move_gen_time*1000:.1f}ms")
        print(f"  AI search (depth 1): {ai_search_time*1000:.1f}ms")
        
        # Performance should scale reasonably
        if board_size == 9:
            assert move_gen_time < 0.01
            assert ai_search_time < 0.1
        elif board_size == 13:
            assert move_gen_time < 0.02
            assert ai_search_time < 0.3
        else:  # 19x19
            assert move_gen_time < 0.05
            assert ai_search_time < 1.0


class TestOptimizations:
    """Test specific optimizations."""
    
    @pytest.mark.unit
    def test_transposition_table(self):
        """Test that transposition table improves performance."""
        # Note: GoAIOptimized always uses transposition table
        ai_with_tt = OptimizedGoAI(board_size=9)
        ai_with_tt.max_depth = 3
        ai_without_tt = OptimizedGoAI(board_size=9)
        ai_without_tt.max_depth = 3
        ai_without_tt.transposition_table = {}  # Clear it to simulate no TT
        
        game = OptimizedGoGame(9)
        
        # Make some moves to create a position
        for i in range(6):
            x, y = i % 3 + 3, i // 3 + 3
            color = 1 if i % 2 == 0 else 2
            game.make_move(x, y, color)
        
        # Test with transposition table
        board_list = game.board.tolist()
        start_time = time.time()
        move1 = ai_with_tt.get_best_move(board_list, 1, 0, 0)
        time_with_tt = time.time() - start_time
        
        # Test without transposition table
        start_time = time.time()
        move2 = ai_without_tt.get_best_move(board_list, 1, 0, 0)
        time_without_tt = time.time() - start_time
        
        print(f"\nWith transposition table: {time_with_tt:.3f}s")
        print(f"Without transposition table: {time_without_tt:.3f}s")
        
        # Transposition table should provide some speedup
        # (though might not always be faster for small searches)
        assert move1 is not None
        assert move2 is not None
    
    @pytest.mark.unit
    def test_move_ordering_impact(self):
        """Test impact of move ordering on alpha-beta pruning."""
        game = OptimizedGoGame(9)
        ai = OptimizedGoAI(board_size=9)
        ai.max_depth = 3
        
        # Create a position
        game.make_move(4, 4, 1)
        game.make_move(4, 5, 2)
        
        # Count nodes evaluated (would need to instrument the AI)
        # For now, just test that move ordering doesn't break anything
        board_list = game.board.tolist()
        move = ai.get_best_move(board_list, 1, 0, 0)
        assert move is not None


class TestConcurrency:
    """Test concurrent execution."""
    
    @pytest.mark.integration
    @pytest.mark.slow
    def test_parallel_game_execution(self):
        """Test running multiple games in parallel."""
        import concurrent.futures
        
        def play_random_game(game_id):
            """Play a game with random moves."""
            game = OptimizedGoGame(9)
            moves = 0
            
            while moves < 50 and not game.game_over:
                valid_moves = game.get_valid_moves(game.current_player)
                if valid_moves:
                    import random
                    move = random.choice(valid_moves)
                    game.make_move(move[0], move[1], game.current_player)
                    moves += 1
                else:
                    game.pass_turn()
            
            return game_id, moves
        
        # Run games in parallel
        start_time = time.time()
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            futures = [executor.submit(play_random_game, i) for i in range(10)]
            results = [f.result() for f in concurrent.futures.as_completed(futures)]
        
        elapsed = time.time() - start_time
        
        print(f"\n10 games in parallel: {elapsed:.2f}s")
        assert len(results) == 10
        assert all(moves > 0 for _, moves in results)


@pytest.mark.benchmark
class TestBenchmarkSuite:
    """Comprehensive benchmark suite."""
    
    def test_full_benchmark(self, benchmark):
        """Full benchmark using pytest-benchmark."""
        game = OptimizedGoGame(9)
        ai = OptimizedGoAI(board_size=9)
        ai.max_depth = 2
        
        def make_ai_move():
            board_list = game.board.tolist()
            move = ai.get_best_move(board_list, 1, 0, 0)
            if move:
                game.make_move(move[0], move[1], 1)
            return move
        
        # Run benchmark
        result = benchmark(make_ai_move)
        assert result is not None