# Go Engine Benchmark Suite

This benchmark suite compares the performance of different Go game engine implementations:
- **optimized_go.py**: NumPy/Numba optimized Python implementation
- **go_ai_codon.py**: Python implementation designed for Codon compilation
- **Codon-compiled version**: Native compiled version of go_ai_codon.py

## Prerequisites

### Required
- Python 3.8+
- NumPy
- Matplotlib
- psutil
- Numba

### Optional (for Codon benchmarks)
- Codon compiler: https://github.com/exaloop/codon

## Installation

1. Install Python dependencies:
```bash
pip install numpy matplotlib psutil numba
```

2. (Optional) Install Codon:
```bash
/bin/bash -c "$(curl -fsSL https://exaloop.io/install.sh)"
```

## Running Benchmarks

### Quick Benchmark (9x9 board only)
```bash
python benchmark_go_engines.py --quick
```

### Default Benchmark (9x9 and 13x13 boards)
```bash
python benchmark_go_engines.py
```

### Full Benchmark (9x9, 13x13, and 19x19 boards)
```bash
python benchmark_go_engines.py --full
```

## Compiling with Codon

If you have Codon installed, compile the Go AI engine:

```bash
./compile_codon.sh
```

This creates an optimized native executable that the benchmark will automatically use.

## Benchmark Tests

The suite includes four main benchmark categories:

### 1. Move Generation
- Tests how quickly each engine can generate valid moves
- Measures moves per second across random board positions

### 2. Position Evaluation
- Tests the speed of static position evaluation
- Measures evaluations per second

### 3. Minimax Search
- Tests full minimax search with alpha-beta pruning
- Measures search performance at different depths

### 4. Full Game Simulation
- Simulates complete games between AI players
- Measures overall engine performance

## Output

The benchmark generates:
- **Console output**: Real-time progress and summary statistics
- **JSON file**: Detailed results saved as `benchmark_results_YYYYMMDD_HHMMSS.json`
- **Visualizations**: 
  - `benchmark_results.png`: Detailed comparison charts
  - `performance_summary.png`: Overall performance summary

## Interpreting Results

### Performance Metrics
- **Time**: Lower is better (faster execution)
- **Ops/sec**: Higher is better (more operations per second)
- **Memory**: Lower is better (less memory usage)

### Expected Performance

Typical speedup factors with Codon compilation:
- Move generation: 5-10x faster
- Position evaluation: 10-20x faster
- Minimax search: 5-15x faster
- Full game simulation: 3-8x faster

## Customization

Edit `benchmark_go_engines.py` to adjust:
- Number of test positions
- Search depths
- Board sizes
- Test iterations

## Troubleshooting

### Codon not found
If Codon is not installed, the benchmark will automatically fall back to comparing:
- optimized_go.py (NumPy/Numba)
- go_ai_codon.py (Pure Python)

### Memory issues
For large board sizes (19x19) with deep search, you may need to:
- Reduce the number of test positions
- Decrease search depth
- Run with `--quick` flag

### Performance variations
Results may vary based on:
- CPU architecture
- Python version
- NumPy/Numba optimizations
- System load

## Example Output

```
=== Move Generation Benchmark (Board Size: 9x9) ===
Testing OptimizedGoGame...
  Time: 0.123s, Ops/sec: 406.5, Avg moves: 45.2
Testing Codon-compiled go_ai_codon...
  Time: 0.018s, Ops/sec: 2777.8, Avg moves: 45.2

=== Position Evaluation Benchmark (Board Size: 9x9, Depth: 3) ===
Testing OptimizedGoAI...
  Time: 0.567s, Evals/sec: 88.2
Testing Codon-compiled evaluation...
  Time: 0.045s, Evals/sec: 1111.1
```

## Contributing

To add new engines or tests:
1. Implement the engine interface (see optimized_go.py for reference)
2. Add benchmark methods in GoEngineBenchmark class
3. Update visualization code if needed