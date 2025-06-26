"""Shared pytest fixtures and configuration for all tests."""

import pytest
import sys
import os
import torch
import numpy as np

# Add parent directory to path so we can import modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

@pytest.fixture
def empty_board_9x9():
    """Fixture for empty 9x9 board."""
    return [[0 for _ in range(9)] for _ in range(9)]

@pytest.fixture
def empty_board_13x13():
    """Fixture for empty 13x13 board."""
    return [[0 for _ in range(13)] for _ in range(13)]

@pytest.fixture
def empty_board_19x19():
    """Fixture for empty 19x19 board."""
    return [[0 for _ in range(19)] for _ in range(19)]

@pytest.fixture
def simple_game_position():
    """Fixture for a simple game position with some stones."""
    board = [[0 for _ in range(9)] for _ in range(9)]
    # Add some stones
    board[3][3] = 1  # Black
    board[3][4] = 2  # White
    board[4][3] = 2  # White
    board[4][4] = 1  # Black
    return board

@pytest.fixture
def capture_position():
    """Fixture for a position where a capture is possible."""
    board = [[0 for _ in range(9)] for _ in range(9)]
    # White stone surrounded by black stones (except one liberty)
    board[4][4] = 2  # White stone
    board[3][4] = 1  # Black stone above
    board[5][4] = 1  # Black stone below
    board[4][3] = 1  # Black stone left
    # board[4][5] is empty - last liberty
    return board

@pytest.fixture
def ko_position():
    """Fixture for a ko situation."""
    board = [[0 for _ in range(9)] for _ in range(9)]
    # Set up a ko pattern
    board[3][3] = 1  # Black
    board[3][4] = 2  # White
    board[3][5] = 1  # Black
    board[4][3] = 2  # White
    board[4][5] = 2  # White
    board[5][3] = 1  # Black
    board[5][4] = 2  # White
    board[5][5] = 1  # Black
    return board

@pytest.fixture
def device():
    """Fixture for PyTorch device selection."""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")

@pytest.fixture
def random_seed():
    """Fixture to set random seeds for reproducibility."""
    seed = 42
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    return seed

@pytest.fixture(autouse=True)
def cleanup_gpu_memory():
    """Automatically cleanup GPU memory after each test."""
    yield
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

@pytest.fixture
def timeout_seconds():
    """Default timeout for tests that might hang."""
    return 30

@pytest.fixture
def small_board_size():
    """Small board size for quick tests."""
    return 9

@pytest.fixture
def medium_board_size():
    """Medium board size for integration tests."""
    return 13

@pytest.fixture
def large_board_size():
    """Large board size for performance tests."""
    return 19