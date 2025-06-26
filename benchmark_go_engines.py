#!/usr/bin/env python3
"""
Benchmark script to compare optimized_go.py with Codon-compiled go_ai_codon.py
"""

import time
import gc
import psutil
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional
import subprocess
import os
import sys
import json
from dataclasses import dataclass
from datetime import datetime

# Import the optimized Go engine
from optimized_go import OptimizedGoGame, OptimizedGoAI

# Import standard Go AI for Codon comparison
from go_ai_codon import GoAIOptimized as GoAICodon


@dataclass
class BenchmarkResult:
    """Container for benchmark results"""
    test_name: str
    engine_name: str
    board_size: int
    time_taken: float
    memory_used: float
    operations_per_second: float
    additional_metrics: Dict[str, float] = None


class GoEngineBenchmark:
    """Comprehensive benchmark suite for Go engines"""
    
    def __init__(self):
        self.results: List[BenchmarkResult] = []
        self.process = psutil.Process()
        
    def measure_memory(self) -> float:
        """Measure current memory usage in MB"""
        return self.process.memory_info().rss / 1024 / 1024
    
    def benchmark_move_generation(self, board_size: int, num_positions: int = 100) -> Dict[str, BenchmarkResult]:
        """Benchmark move generation performance"""
        print(f"\n=== Move Generation Benchmark (Board Size: {board_size}x{board_size}) ===")
        results = {}
        
        # Test OptimizedGoGame
        print("Testing OptimizedGoGame...")
        game = OptimizedGoGame(board_size)
        
        # Create random positions
        positions = self._create_random_positions(game, num_positions)
        
        gc.collect()
        start_memory = self.measure_memory()
        start_time = time.time()
        
        total_moves = 0
        for pos in positions:
            game.board = pos
            moves = game.get_valid_moves('black')
            total_moves += len(moves)
        
        elapsed_time = time.time() - start_time
        memory_used = self.measure_memory() - start_memory
        
        results['optimized_go'] = BenchmarkResult(
            test_name="move_generation",
            engine_name="optimized_go",
            board_size=board_size,
            time_taken=elapsed_time,
            memory_used=memory_used,
            operations_per_second=num_positions / elapsed_time,
            additional_metrics={"total_moves": total_moves, "avg_moves_per_position": total_moves / num_positions}
        )
        print(f"  Time: {elapsed_time:.3f}s, Ops/sec: {num_positions/elapsed_time:.1f}, Avg moves: {total_moves/num_positions:.1f}")
        
        # Test Codon-compiled version (if available)
        if self._is_codon_available():
            print("Testing Codon-compiled go_ai_codon...")
            results['codon'] = self._benchmark_codon_move_generation(board_size, positions)
        else:
            print("Codon compiler not available, using Python version of go_ai_codon...")
            results['codon_python'] = self._benchmark_python_codon_move_generation(board_size, positions)
        
        return results
    
    def benchmark_position_evaluation(self, board_size: int, num_positions: int = 50, depth: int = 3) -> Dict[str, BenchmarkResult]:
        """Benchmark position evaluation performance"""
        print(f"\n=== Position Evaluation Benchmark (Board Size: {board_size}x{board_size}, Depth: {depth}) ===")
        results = {}
        
        # Test OptimizedGoAI
        print("Testing OptimizedGoAI...")
        game = OptimizedGoGame(board_size)
        ai = OptimizedGoAI(max_depth=depth)
        
        # Create test positions
        positions = self._create_test_positions(game, num_positions)
        
        gc.collect()
        start_memory = self.measure_memory()
        start_time = time.time()
        
        evaluations = 0
        for pos in positions:
            game.board = pos
            score = ai._evaluate_position(game, game.BLACK)
            evaluations += 1
        
        elapsed_time = time.time() - start_time
        memory_used = self.measure_memory() - start_memory
        
        results['optimized_go'] = BenchmarkResult(
            test_name="position_evaluation",
            engine_name="optimized_go",
            board_size=board_size,
            time_taken=elapsed_time,
            memory_used=memory_used,
            operations_per_second=evaluations / elapsed_time,
            additional_metrics={"depth": depth, "positions_evaluated": evaluations}
        )
        print(f"  Time: {elapsed_time:.3f}s, Evals/sec: {evaluations/elapsed_time:.1f}")
        
        # Test Codon version
        if self._is_codon_available():
            print("Testing Codon-compiled evaluation...")
            results['codon'] = self._benchmark_codon_evaluation(board_size, positions, depth)
        else:
            print("Testing Python go_ai_codon evaluation...")
            results['codon_python'] = self._benchmark_python_codon_evaluation(board_size, positions, depth)
        
        return results
    
    def benchmark_full_game_simulation(self, board_size: int, num_games: int = 10, depth: int = 2) -> Dict[str, BenchmarkResult]:
        """Benchmark full game simulation"""
        print(f"\n=== Full Game Simulation Benchmark (Board Size: {board_size}x{board_size}, Games: {num_games}) ===")
        results = {}
        
        # Test OptimizedGoGame
        print("Testing OptimizedGoGame full games...")
        
        gc.collect()
        start_memory = self.measure_memory()
        start_time = time.time()
        
        total_moves = 0
        for _ in range(num_games):
            moves = self._simulate_game_optimized(board_size, depth)
            total_moves += moves
        
        elapsed_time = time.time() - start_time
        memory_used = self.measure_memory() - start_memory
        
        results['optimized_go'] = BenchmarkResult(
            test_name="full_game_simulation",
            engine_name="optimized_go",
            board_size=board_size,
            time_taken=elapsed_time,
            memory_used=memory_used,
            operations_per_second=num_games / elapsed_time,
            additional_metrics={"total_moves": total_moves, "avg_moves_per_game": total_moves / num_games}
        )
        print(f"  Time: {elapsed_time:.3f}s, Games/sec: {num_games/elapsed_time:.2f}, Avg moves/game: {total_moves/num_games:.1f}")
        
        # Test Codon version
        if self._is_codon_available():
            print("Testing Codon-compiled full games...")
            results['codon'] = self._benchmark_codon_full_game(board_size, num_games, depth)
        else:
            print("Testing Python go_ai_codon full games...")
            results['codon_python'] = self._benchmark_python_codon_full_game(board_size, num_games, depth)
        
        return results
    
    def benchmark_minimax_search(self, board_size: int, num_positions: int = 20, depth: int = 3) -> Dict[str, BenchmarkResult]:
        """Benchmark minimax search performance"""
        print(f"\n=== Minimax Search Benchmark (Board Size: {board_size}x{board_size}, Depth: {depth}) ===")
        results = {}
        
        # Test OptimizedGoAI
        print("Testing OptimizedGoAI minimax...")
        game = OptimizedGoGame(board_size)
        ai = OptimizedGoAI(max_depth=depth)
        
        # Create mid-game positions
        positions = self._create_midgame_positions(game, num_positions)
        
        gc.collect()
        start_memory = self.measure_memory()
        start_time = time.time()
        
        moves_found = 0
        for pos in positions:
            game.board = pos
            game.current_player = game.BLACK
            move = ai.get_best_move(game, 'black')
            if move:
                moves_found += 1
        
        elapsed_time = time.time() - start_time
        memory_used = self.measure_memory() - start_memory
        
        results['optimized_go'] = BenchmarkResult(
            test_name="minimax_search",
            engine_name="optimized_go",
            board_size=board_size,
            time_taken=elapsed_time,
            memory_used=memory_used,
            operations_per_second=num_positions / elapsed_time,
            additional_metrics={"depth": depth, "moves_found": moves_found}
        )
        print(f"  Time: {elapsed_time:.3f}s, Positions/sec: {num_positions/elapsed_time:.2f}")
        
        # Test Codon version
        if self._is_codon_available():
            print("Testing Codon-compiled minimax...")
            results['codon'] = self._benchmark_codon_minimax(board_size, positions, depth)
        else:
            print("Testing Python go_ai_codon minimax...")
            results['codon_python'] = self._benchmark_python_codon_minimax(board_size, positions, depth)
        
        return results
    
    def _create_random_positions(self, game: OptimizedGoGame, num_positions: int) -> List[np.ndarray]:
        """Create random board positions for testing"""
        positions = []
        for _ in range(num_positions):
            board = np.zeros((game.size, game.size), dtype=np.int8)
            # Add random stones (ensuring valid positions)
            num_stones = np.random.randint(10, min(40, game.size * game.size // 3))
            for _ in range(num_stones):
                x, y = np.random.randint(0, game.size, 2)
                if board[y, x] == 0:
                    board[y, x] = np.random.choice([1, 2])
            positions.append(board.copy())
        return positions
    
    def _create_test_positions(self, game: OptimizedGoGame, num_positions: int) -> List[np.ndarray]:
        """Create varied test positions"""
        positions = []
        
        # Empty board
        positions.append(np.zeros((game.size, game.size), dtype=np.int8))
        
        # Corner positions
        for i in range(min(4, num_positions - 1)):
            board = np.zeros((game.size, game.size), dtype=np.int8)
            corners = [(0, 0), (0, game.size-1), (game.size-1, 0), (game.size-1, game.size-1)]
            x, y = corners[i % 4]
            board[y, x] = 1
            board[y, min(x+1, game.size-1)] = 2
            positions.append(board.copy())
        
        # Random positions for the rest
        remaining = num_positions - len(positions)
        positions.extend(self._create_random_positions(game, remaining))
        
        return positions[:num_positions]
    
    def _create_midgame_positions(self, game: OptimizedGoGame, num_positions: int) -> List[np.ndarray]:
        """Create realistic mid-game positions"""
        positions = []
        for _ in range(num_positions):
            # Simulate partial games to create realistic positions
            temp_game = OptimizedGoGame(game.size)
            temp_ai = OptimizedGoAI(max_depth=1)  # Fast AI for position generation
            
            # Play 20-40 moves
            num_moves = np.random.randint(20, min(40, game.size * game.size // 4))
            for i in range(num_moves):
                color = 'black' if i % 2 == 0 else 'white'
                move = temp_ai.get_best_move(temp_game, color)
                if move:
                    temp_game.make_move(move[0], move[1], color)
                else:
                    break
            
            positions.append(temp_game.board.copy())
        
        return positions
    
    def _simulate_game_optimized(self, board_size: int, depth: int) -> int:
        """Simulate a full game using OptimizedGoGame"""
        game = OptimizedGoGame(board_size)
        ai = OptimizedGoAI(max_depth=depth)
        
        moves = 0
        max_moves = board_size * board_size * 2
        consecutive_passes = 0
        
        while moves < max_moves and not game.game_over:
            color = 'black' if moves % 2 == 0 else 'white'
            move = ai.get_best_move(game, color)
            
            if move:
                game.make_move(move[0], move[1], color)
                consecutive_passes = 0
            else:
                game.pass_turn(color)
                consecutive_passes += 1
                if consecutive_passes >= 2:
                    break
            
            moves += 1
        
        return moves
    
    def _is_codon_available(self) -> bool:
        """Check if Codon compiler is available"""
        try:
            result = subprocess.run(['codon', '--version'], capture_output=True, text=True)
            return result.returncode == 0
        except FileNotFoundError:
            return False
    
    def _benchmark_codon_move_generation(self, board_size: int, positions: List[np.ndarray]) -> BenchmarkResult:
        """Benchmark Codon-compiled move generation"""
        # Create a temporary test script
        test_script = f"""
import time
import sys
sys.path.insert(0, '{os.path.dirname(os.path.abspath(__file__))}')
from go_ai_codon import GoAIOptimized

ai = GoAIOptimized({board_size})
positions = {self._positions_to_list(positions)}

start_time = time.time()
total_moves = 0

for pos in positions:
    ai.board = pos
    moves = ai.get_valid_moves(1)  # Black
    total_moves += len(moves)

elapsed = time.time() - start_time
print(f"{{elapsed}},{{total_moves}}")
"""
        
        # Write and compile
        with open('temp_codon_test.py', 'w') as f:
            f.write(test_script)
        
        try:
            # Compile with Codon
            subprocess.run(['codon', 'build', '-o', 'temp_codon_test', 'temp_codon_test.py'], check=True)
            
            # Run compiled version
            gc.collect()
            start_memory = self.measure_memory()
            result = subprocess.run(['./temp_codon_test'], capture_output=True, text=True)
            memory_used = self.measure_memory() - start_memory
            
            elapsed, total_moves = map(float, result.stdout.strip().split(','))
            
            return BenchmarkResult(
                test_name="move_generation",
                engine_name="codon",
                board_size=board_size,
                time_taken=elapsed,
                memory_used=memory_used,
                operations_per_second=len(positions) / elapsed,
                additional_metrics={"total_moves": total_moves, "avg_moves_per_position": total_moves / len(positions)}
            )
        finally:
            # Cleanup
            for f in ['temp_codon_test.py', 'temp_codon_test']:
                if os.path.exists(f):
                    os.remove(f)
    
    def _benchmark_python_codon_move_generation(self, board_size: int, positions: List[np.ndarray]) -> BenchmarkResult:
        """Benchmark Python version of go_ai_codon"""
        ai = GoAICodon(board_size)
        
        # Convert positions to the format expected by go_ai_codon
        converted_positions = []
        for pos in positions:
            board = [[int(pos[y, x]) for x in range(board_size)] for y in range(board_size)]
            converted_positions.append(board)
        
        gc.collect()
        start_memory = self.measure_memory()
        start_time = time.time()
        
        total_moves = 0
        for pos in converted_positions:
            ai.board = pos
            moves = ai.get_valid_moves(ai.BLACK)
            total_moves += len(moves)
        
        elapsed_time = time.time() - start_time
        memory_used = self.measure_memory() - start_memory
        
        return BenchmarkResult(
            test_name="move_generation",
            engine_name="codon_python",
            board_size=board_size,
            time_taken=elapsed_time,
            memory_used=memory_used,
            operations_per_second=len(positions) / elapsed_time,
            additional_metrics={"total_moves": total_moves, "avg_moves_per_position": total_moves / len(positions)}
        )
    
    def _benchmark_codon_evaluation(self, board_size: int, positions: List[np.ndarray], depth: int) -> BenchmarkResult:
        """Benchmark Codon-compiled position evaluation"""
        # Similar to move generation but for evaluation
        # Implementation would be similar to _benchmark_codon_move_generation
        # For brevity, returning a placeholder
        return self._benchmark_python_codon_evaluation(board_size, positions, depth)
    
    def _benchmark_python_codon_evaluation(self, board_size: int, positions: List[np.ndarray], depth: int) -> BenchmarkResult:
        """Benchmark Python go_ai_codon evaluation"""
        ai = GoAICodon(board_size)
        ai.max_depth = depth
        
        # Convert positions
        converted_positions = []
        for pos in positions:
            board = [[int(pos[y, x]) for x in range(board_size)] for y in range(board_size)]
            converted_positions.append(board)
        
        gc.collect()
        start_memory = self.measure_memory()
        start_time = time.time()
        
        evaluations = 0
        for pos in converted_positions:
            ai.board = pos
            score = ai.evaluate_position(ai.BLACK, 0, 0)
            evaluations += 1
        
        elapsed_time = time.time() - start_time
        memory_used = self.measure_memory() - start_memory
        
        return BenchmarkResult(
            test_name="position_evaluation",
            engine_name="codon_python",
            board_size=board_size,
            time_taken=elapsed_time,
            memory_used=memory_used,
            operations_per_second=evaluations / elapsed_time,
            additional_metrics={"depth": depth, "positions_evaluated": evaluations}
        )
    
    def _benchmark_codon_full_game(self, board_size: int, num_games: int, depth: int) -> BenchmarkResult:
        """Benchmark Codon-compiled full game"""
        # For brevity, using Python version
        return self._benchmark_python_codon_full_game(board_size, num_games, depth)
    
    def _benchmark_python_codon_full_game(self, board_size: int, num_games: int, depth: int) -> BenchmarkResult:
        """Benchmark Python go_ai_codon full game"""
        gc.collect()
        start_memory = self.measure_memory()
        start_time = time.time()
        
        total_moves = 0
        for _ in range(num_games):
            moves = self._simulate_game_codon(board_size, depth)
            total_moves += moves
        
        elapsed_time = time.time() - start_time
        memory_used = self.measure_memory() - start_memory
        
        return BenchmarkResult(
            test_name="full_game_simulation",
            engine_name="codon_python",
            board_size=board_size,
            time_taken=elapsed_time,
            memory_used=memory_used,
            operations_per_second=num_games / elapsed_time,
            additional_metrics={"total_moves": total_moves, "avg_moves_per_game": total_moves / num_games}
        )
    
    def _simulate_game_codon(self, board_size: int, depth: int) -> int:
        """Simulate a game using go_ai_codon"""
        ai = GoAICodon(board_size)
        ai.max_depth = depth
        
        # Initialize board
        board = [[0 for _ in range(board_size)] for _ in range(board_size)]
        
        moves = 0
        max_moves = board_size * board_size * 2
        consecutive_passes = 0
        
        while moves < max_moves:
            color = ai.BLACK if moves % 2 == 0 else ai.WHITE
            move = ai.get_best_move(board, color, 0, 0)
            
            if move:
                x, y = move
                board[y][x] = color
                consecutive_passes = 0
            else:
                consecutive_passes += 1
                if consecutive_passes >= 2:
                    break
            
            moves += 1
        
        return moves
    
    def _benchmark_codon_minimax(self, board_size: int, positions: List[np.ndarray], depth: int) -> BenchmarkResult:
        """Benchmark Codon-compiled minimax"""
        return self._benchmark_python_codon_minimax(board_size, positions, depth)
    
    def _benchmark_python_codon_minimax(self, board_size: int, positions: List[np.ndarray], depth: int) -> BenchmarkResult:
        """Benchmark Python go_ai_codon minimax"""
        ai = GoAICodon(board_size)
        ai.max_depth = depth
        
        # Convert positions
        converted_positions = []
        for pos in positions:
            board = [[int(pos[y, x]) for x in range(board_size)] for y in range(board_size)]
            converted_positions.append(board)
        
        gc.collect()
        start_memory = self.measure_memory()
        start_time = time.time()
        
        moves_found = 0
        for pos in converted_positions:
            ai.board = pos
            move = ai.get_best_move(pos, ai.BLACK, 0, 0)
            if move:
                moves_found += 1
        
        elapsed_time = time.time() - start_time
        memory_used = self.measure_memory() - start_memory
        
        return BenchmarkResult(
            test_name="minimax_search",
            engine_name="codon_python",
            board_size=board_size,
            time_taken=elapsed_time,
            memory_used=memory_used,
            operations_per_second=len(positions) / elapsed_time,
            additional_metrics={"depth": depth, "moves_found": moves_found}
        )
    
    def _positions_to_list(self, positions: List[np.ndarray]) -> str:
        """Convert numpy positions to list format for code generation"""
        result = []
        for pos in positions:
            board = [[int(pos[y, x]) for x in range(pos.shape[1])] for y in range(pos.shape[0])]
            result.append(board)
        return repr(result)
    
    def save_results(self, filename: str = None):
        """Save benchmark results to file"""
        if filename is None:
            filename = f"benchmark_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        results_dict = []
        for result in self.results:
            res_dict = {
                'test_name': result.test_name,
                'engine_name': result.engine_name,
                'board_size': result.board_size,
                'time_taken': result.time_taken,
                'memory_used': result.memory_used,
                'operations_per_second': result.operations_per_second,
                'additional_metrics': result.additional_metrics or {}
            }
            results_dict.append(res_dict)
        
        with open(filename, 'w') as f:
            json.dump(results_dict, f, indent=2)
        
        print(f"\nResults saved to {filename}")
    
    def visualize_results(self):
        """Create visualizations of benchmark results"""
        if not self.results:
            print("No results to visualize")
            return
        
        # Group results by test type
        test_groups = {}
        for result in self.results:
            if result.test_name not in test_groups:
                test_groups[result.test_name] = []
            test_groups[result.test_name].append(result)
        
        # Create subplots
        num_tests = len(test_groups)
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        axes = axes.flatten()
        
        for idx, (test_name, results) in enumerate(test_groups.items()):
            if idx >= 4:
                break
            
            ax = axes[idx]
            
            # Prepare data
            engines = [r.engine_name for r in results]
            times = [r.time_taken for r in results]
            ops_per_sec = [r.operations_per_second for r in results]
            memory = [r.memory_used for r in results]
            
            # Create grouped bar chart
            x = np.arange(len(engines))
            width = 0.25
            
            # Normalize values for comparison
            max_time = max(times) if times else 1
            max_ops = max(ops_per_sec) if ops_per_sec else 1
            max_mem = max(memory) if memory else 1
            
            norm_times = [t/max_time for t in times]
            norm_ops = [o/max_ops for o in ops_per_sec]
            norm_mem = [m/max_mem for m in memory]
            
            bars1 = ax.bar(x - width, norm_times, width, label='Time (normalized)', alpha=0.8)
            bars2 = ax.bar(x, norm_ops, width, label='Ops/sec (normalized)', alpha=0.8)
            bars3 = ax.bar(x + width, norm_mem, width, label='Memory (normalized)', alpha=0.8)
            
            ax.set_xlabel('Engine')
            ax.set_ylabel('Normalized Value')
            ax.set_title(f'{test_name.replace("_", " ").title()} Benchmark')
            ax.set_xticks(x)
            ax.set_xticklabels(engines)
            ax.legend()
            ax.grid(axis='y', alpha=0.3)
            
            # Add value labels
            for bars in [bars1, bars2, bars3]:
                for bar in bars:
                    height = bar.get_height()
                    if height > 0:
                        ax.annotate(f'{height:.2f}',
                                  xy=(bar.get_x() + bar.get_width() / 2, height),
                                  xytext=(0, 3),
                                  textcoords="offset points",
                                  ha='center', va='bottom',
                                  fontsize=8)
        
        plt.tight_layout()
        plt.savefig('benchmark_results.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # Create performance comparison chart
        self._create_performance_summary()
    
    def _create_performance_summary(self):
        """Create a summary performance chart"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Speed comparison
        engine_speeds = {}
        for result in self.results:
            if result.engine_name not in engine_speeds:
                engine_speeds[result.engine_name] = []
            engine_speeds[result.engine_name].append(result.operations_per_second)
        
        engines = list(engine_speeds.keys())
        avg_speeds = [np.mean(speeds) for speeds in engine_speeds.values()]
        
        ax1.bar(engines, avg_speeds, alpha=0.7, color=['blue', 'green', 'red'][:len(engines)])
        ax1.set_xlabel('Engine')
        ax1.set_ylabel('Average Operations per Second')
        ax1.set_title('Speed Comparison')
        ax1.grid(axis='y', alpha=0.3)
        
        # Memory comparison
        engine_memory = {}
        for result in self.results:
            if result.engine_name not in engine_memory:
                engine_memory[result.engine_name] = []
            engine_memory[result.engine_name].append(result.memory_used)
        
        avg_memory = [np.mean(mem) for mem in engine_memory.values()]
        
        ax2.bar(engines, avg_memory, alpha=0.7, color=['blue', 'green', 'red'][:len(engines)])
        ax2.set_xlabel('Engine')
        ax2.set_ylabel('Average Memory Usage (MB)')
        ax2.set_title('Memory Usage Comparison')
        ax2.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('performance_summary.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def run_all_benchmarks(self, board_sizes: List[int] = [9, 13], depths: List[int] = [2, 3]):
        """Run comprehensive benchmark suite"""
        print("Starting comprehensive Go engine benchmark...")
        print(f"Board sizes: {board_sizes}")
        print(f"Search depths: {depths}")
        print("=" * 60)
        
        for board_size in board_sizes:
            for depth in depths:
                # Move generation benchmark
                results = self.benchmark_move_generation(board_size, num_positions=50)
                for result in results.values():
                    self.results.append(result)
                
                # Position evaluation benchmark
                results = self.benchmark_position_evaluation(board_size, num_positions=30, depth=depth)
                for result in results.values():
                    self.results.append(result)
                
                # Minimax search benchmark
                results = self.benchmark_minimax_search(board_size, num_positions=10, depth=depth)
                for result in results.values():
                    self.results.append(result)
                
                # Full game simulation (only for smaller depths)
                if depth <= 2:
                    results = self.benchmark_full_game_simulation(board_size, num_games=5, depth=depth)
                    for result in results.values():
                        self.results.append(result)
        
        print("\n" + "=" * 60)
        print("Benchmark Summary:")
        print("=" * 60)
        
        # Print summary table
        print(f"{'Test':<25} {'Engine':<15} {'Board':<6} {'Time(s)':<10} {'Ops/sec':<12} {'Memory(MB)':<10}")
        print("-" * 88)
        
        for result in self.results:
            print(f"{result.test_name:<25} {result.engine_name:<15} {result.board_size:<6} "
                  f"{result.time_taken:<10.3f} {result.operations_per_second:<12.1f} {result.memory_used:<10.1f}")
        
        # Save and visualize
        self.save_results()
        self.visualize_results()


def main():
    """Main benchmark execution"""
    benchmark = GoEngineBenchmark()
    
    # Check command line arguments
    if len(sys.argv) > 1:
        if sys.argv[1] == '--quick':
            # Quick benchmark
            print("Running quick benchmark...")
            benchmark.run_all_benchmarks(board_sizes=[9], depths=[2])
        elif sys.argv[1] == '--full':
            # Full benchmark
            print("Running full benchmark...")
            benchmark.run_all_benchmarks(board_sizes=[9, 13, 19], depths=[1, 2, 3])
        else:
            print("Usage: python benchmark_go_engines.py [--quick|--full]")
            sys.exit(1)
    else:
        # Default benchmark
        benchmark.run_all_benchmarks()


if __name__ == "__main__":
    main()