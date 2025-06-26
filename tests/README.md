# Go Game Test Suite

This directory contains the unified pytest test suite for the Go game project.

## Test Structure

### Core Test Files

1. **test_go_rules.py** - Comprehensive tests for Go game rules
   - Basic game functionality
   - Capture mechanics
   - Ko rule implementation
   - Self-capture (suicide) rules
   - Game ending conditions
   - Valid move generation
   - Multiple board sizes (9x9, 13x13, 19x19)

2. **test_alpha_go_unified.py** - AlphaGo AI implementation tests
   - PolicyValueNet neural network
   - MCTS (Monte Carlo Tree Search) functionality
   - Complete AlphaGo player behavior
   - Training and data generation
   - Different board sizes

3. **test_self_play_training.py** - Self-play and training system tests
   - Basic self-play functionality
   - Two-stage training system
   - Progressive difficulty training
   - Data augmentation
   - Memory management
   - Temperature effects on move selection

4. **test_performance.py** - Performance and optimization tests
   - Go engine performance benchmarks
   - Codon compilation comparison
   - Memory usage profiling
   - Scalability with different board sizes
   - Optimization validation (transposition tables, etc.)
   - Concurrent execution tests

### Supporting Files

- **conftest.py** - Shared pytest fixtures and configuration
- **__init__.py** - Package initialization
- **simple_go.py** - Simple Go implementation for testing
- **test_optimized_go.py** - Legacy unittest tests (to be migrated)
- **test_alpha_go.py** - Legacy unittest tests (to be migrated)

## Running Tests

### Run all tests:
```bash
pytest
```

### Run specific test categories:
```bash
# Unit tests only
pytest -m unit

# Integration tests
pytest -m integration

# Slow tests
pytest -m slow

# GPU tests
pytest -m gpu
```

### Run specific test files:
```bash
pytest tests/test_go_rules.py
pytest tests/test_alpha_go_unified.py -v
```

### Run with coverage:
```bash
pytest --cov=. --cov-report=html
```

### Run performance benchmarks:
```bash
pytest tests/test_performance.py -v
```

## Test Markers

Tests are marked with the following markers:
- `@pytest.mark.unit` - Quick unit tests
- `@pytest.mark.integration` - Integration tests
- `@pytest.mark.slow` - Tests that take longer to run
- `@pytest.mark.gpu` - Tests that require GPU
- `@pytest.mark.training` - Tests involving model training
- `@pytest.mark.skip` - Tests that are temporarily skipped

## Configuration

Test configuration is in `pytest.ini` in the project root:
- Default timeout: 300 seconds
- Test discovery: `test_*.py` files
- Verbose output with short tracebacks

## Migration Notes

The test suite has been migrated from unittest to pytest for:
- Better fixtures and parametrization
- Cleaner test organization
- Better integration with CI/CD
- More detailed test reporting
- Easier test selection and filtering

Some tests from the original implementation are marked as skipped if the corresponding functionality is not yet implemented in the current codebase.