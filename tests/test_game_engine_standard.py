"""
Standardized test suite for all Go game engine implementations.
This ensures consistency across all variants (Python, NumPy, Codon, etc.)
"""

import pytest
import numpy as np
import subprocess
import sys
import os
from abc import ABC, abstractmethod
from typing import List, Tuple, Optional, Any

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from server import GoGame
from optimized_go import OptimizedGoGame, EMPTY, BLACK, WHITE
from tests.simple_go import SimpleGoBoard


class GameEngineAdapter(ABC):
    """Abstract adapter to provide uniform interface for all game implementations."""
    
    @abstractmethod
    def create_game(self, size: int):
        """Create a new game instance."""
        pass
    
    @abstractmethod
    def get_board_value(self, x: int, y: int) -> int:
        """Get board value at position (x, y). Returns 0=empty, 1=black, 2=white."""
        pass
    
    @abstractmethod
    def make_move(self, x: int, y: int, color: int) -> bool:
        """Make a move. Color is 1=black, 2=white. Returns success."""
        pass
    
    @abstractmethod
    def get_current_player(self) -> int:
        """Get current player. Returns 1=black, 2=white."""
        pass
    
    @abstractmethod
    def is_game_over(self) -> bool:
        """Check if game is over."""
        pass
    
    @abstractmethod
    def get_captures(self) -> Tuple[int, int]:
        """Get captures for (black, white)."""
        pass
    
    @abstractmethod
    def pass_turn(self) -> bool:
        """Pass the turn."""
        pass
    
    @abstractmethod
    def get_valid_moves(self, color: int) -> List[Tuple[int, int]]:
        """Get list of valid moves for color."""
        pass
    
    @abstractmethod
    def get_board_size(self) -> int:
        """Get board size."""
        pass


class BasicGoGameAdapter(GameEngineAdapter):
    """Adapter for basic GoGame from server.py"""
    
    def create_game(self, size: int):
        self.game = GoGame(size)
        return self.game
    
    def get_board_value(self, x: int, y: int) -> int:
        value = self.game.board[y][x]
        if value == 'black':
            return 1
        elif value == 'white':
            return 2
        else:
            return 0
    
    def make_move(self, x: int, y: int, color: int) -> bool:
        color_str = 'black' if color == 1 else 'white'
        return self.game.make_move(x, y, color_str)
    
    def get_current_player(self) -> int:
        return 1 if self.game.current_player == 'black' else 2
    
    def is_game_over(self) -> bool:
        return self.game.game_over
    
    def get_captures(self) -> Tuple[int, int]:
        return (self.game.captures['black'], self.game.captures['white'])
    
    def pass_turn(self) -> bool:
        color_str = 'black' if self.get_current_player() == 1 else 'white'
        return self.game.pass_turn(color_str)
    
    def get_valid_moves(self, color: int) -> List[Tuple[int, int]]:
        color_str = 'black' if color == 1 else 'white'
        return self.game.get_valid_moves(color_str)
    
    def get_board_size(self) -> int:
        return self.game.size


class OptimizedGoGameAdapter(GameEngineAdapter):
    """Adapter for OptimizedGoGame"""
    
    def create_game(self, size: int):
        self.game = OptimizedGoGame(size)
        return self.game
    
    def get_board_value(self, x: int, y: int) -> int:
        return int(self.game.board[y, x])
    
    def make_move(self, x: int, y: int, color: int) -> bool:
        color_str = 'black' if color == 1 else 'white'
        return self.game.make_move(x, y, color_str)
    
    def get_current_player(self) -> int:
        return self.game.current_player
    
    def is_game_over(self) -> bool:
        return self.game.game_over
    
    def get_captures(self) -> Tuple[int, int]:
        return (self.game.captures[BLACK], self.game.captures[WHITE])
    
    def pass_turn(self) -> bool:
        return self.game.pass_turn()
    
    def get_valid_moves(self, color: int) -> List[Tuple[int, int]]:
        return self.game.get_valid_moves(color)
    
    def get_board_size(self) -> int:
        return self.game.size


class CodonGoGameAdapter(GameEngineAdapter):
    """Adapter for Codon-compiled game (if available)"""
    
    def __init__(self):
        self.available = os.path.exists('./go_game_fast_compiled')
        
    def create_game(self, size: int):
        if not self.available:
            pytest.skip("Codon compiled game not available")
        
        # Import the wrapper
        from codon_game_wrapper import CodonGoGame
        self.game = CodonGoGame(size)
        return self.game
    
    def get_board_value(self, x: int, y: int) -> int:
        return int(self.game.board[y, x])
    
    def make_move(self, x: int, y: int, color: int) -> bool:
        color_str = 'black' if color == 1 else 'white'
        return self.game.make_move(x, y, color_str)
    
    def get_current_player(self) -> int:
        return 1 if self.game.current_player == 'black' else 2
    
    def is_game_over(self) -> bool:
        return self.game.game_over
    
    def get_captures(self) -> Tuple[int, int]:
        return (self.game.captures['black'], self.game.captures['white'])
    
    def pass_turn(self) -> bool:
        return self.game.pass_turn()
    
    def get_valid_moves(self, color: int) -> List[Tuple[int, int]]:
        # Codon game doesn't have get_valid_moves, so we compute it
        moves = []
        for y in range(self.game.size):
            for x in range(self.game.size):
                if self.get_board_value(x, y) == 0:
                    # Try the move
                    saved_board = self.game.board.copy()
                    saved_player = self.game.current_player
                    saved_captures = self.game.captures.copy()
                    
                    if self.make_move(x, y, color):
                        moves.append((x, y))
                    
                    # Restore state
                    self.game.board = saved_board
                    self.game.current_player = saved_player
                    self.game.captures = saved_captures
        
        return moves
    
    def get_board_size(self) -> int:
        return self.game.size


# Test implementations
IMPLEMENTATIONS = [
    ("BasicGoGame", BasicGoGameAdapter),
    ("OptimizedGoGame", OptimizedGoGameAdapter),
    ("CodonGoGame", CodonGoGameAdapter),
]


@pytest.mark.parametrize("impl_name,adapter_class", IMPLEMENTATIONS)
class TestStandardGameLogic:
    """Standard tests that all implementations must pass."""
    
    @pytest.fixture
    def adapter(self, impl_name, adapter_class):
        """Create adapter instance."""
        return adapter_class()
    
    @pytest.fixture
    def game_9x9(self, adapter):
        """Create 9x9 game."""
        adapter.create_game(9)
        return adapter
    
    @pytest.mark.unit
    def test_initial_state(self, impl_name, adapter):
        """Test initial game state."""
        adapter.create_game(9)
        
        # Check empty board
        for x in range(9):
            for y in range(9):
                assert adapter.get_board_value(x, y) == 0, \
                    f"{impl_name}: Position ({x},{y}) should be empty"
        
        # Check initial player is black
        assert adapter.get_current_player() == 1, f"{impl_name}: Should start with black"
        
        # Check game not over
        assert not adapter.is_game_over(), f"{impl_name}: Game should not be over initially"
        
        # Check no captures
        captures = adapter.get_captures()
        assert captures == (0, 0), f"{impl_name}: Should have no captures initially"
    
    @pytest.mark.unit
    def test_basic_moves(self, impl_name, game_9x9):
        """Test basic move placement."""
        # Black move
        assert game_9x9.make_move(4, 4, 1), f"{impl_name}: Black move should succeed"
        assert game_9x9.get_board_value(4, 4) == 1, f"{impl_name}: Should place black stone"
        assert game_9x9.get_current_player() == 2, f"{impl_name}: Should switch to white"
        
        # White move
        assert game_9x9.make_move(3, 3, 2), f"{impl_name}: White move should succeed"
        assert game_9x9.get_board_value(3, 3) == 2, f"{impl_name}: Should place white stone"
        assert game_9x9.get_current_player() == 1, f"{impl_name}: Should switch to black"
        
        # Invalid move on occupied spot
        assert not game_9x9.make_move(4, 4, 1), f"{impl_name}: Move on occupied spot should fail"
    
    @pytest.mark.unit
    def test_single_stone_capture(self, impl_name, game_9x9):
        """Test capturing a single stone."""
        # Setup: White stone surrounded by black
        game_9x9.make_move(4, 4, 2)  # White
        game_9x9.make_move(3, 4, 1)  # Black left
        game_9x9.make_move(0, 0, 2)  # White dummy
        game_9x9.make_move(5, 4, 1)  # Black right
        game_9x9.make_move(0, 1, 2)  # White dummy
        game_9x9.make_move(4, 3, 1)  # Black top
        game_9x9.make_move(0, 2, 2)  # White dummy
        
        # Capture
        assert game_9x9.make_move(4, 5, 1), f"{impl_name}: Capturing move should succeed"
        
        # Check capture
        assert game_9x9.get_board_value(4, 4) == 0, f"{impl_name}: Captured stone should be removed"
        captures = game_9x9.get_captures()
        assert captures[0] == 1, f"{impl_name}: Black should have 1 capture"
    
    @pytest.mark.unit
    def test_group_capture(self, impl_name, game_9x9):
        """Test capturing a group of stones."""
        # Create white group
        moves = [
            (3, 3, 2), (2, 3, 1),  # W, B
            (3, 4, 2), (2, 4, 1),  # W, B
            (4, 3, 2), (5, 3, 1),  # W, B
            (4, 4, 2), (5, 4, 1),  # W, B
            (0, 0, 2), (3, 2, 1),  # W dummy, B
            (0, 1, 2), (4, 2, 1),  # W dummy, B
            (0, 2, 2), (3, 5, 1),  # W dummy, B
        ]
        
        for x, y, color in moves:
            assert game_9x9.make_move(x, y, color), \
                f"{impl_name}: Move ({x},{y}) color {color} should succeed"
        
        # Complete the capture
        assert game_9x9.make_move(4, 5, 1), f"{impl_name}: Final capturing move should succeed"
        
        # Check all stones captured
        for x, y in [(3, 3), (3, 4), (4, 3), (4, 4)]:
            assert game_9x9.get_board_value(x, y) == 0, \
                f"{impl_name}: Stone at ({x},{y}) should be captured"
        
        captures = game_9x9.get_captures()
        assert captures[0] == 4, f"{impl_name}: Black should have 4 captures"
    
    @pytest.mark.unit
    def test_suicide_prevention(self, impl_name, game_9x9):
        """Test that suicide moves are prevented."""
        # Surround a point with opponent stones
        game_9x9.make_move(3, 4, 1)  # Black
        game_9x9.make_move(0, 0, 2)  # White dummy
        game_9x9.make_move(5, 4, 1)  # Black
        game_9x9.make_move(0, 1, 2)  # White dummy
        game_9x9.make_move(4, 3, 1)  # Black
        game_9x9.make_move(0, 2, 2)  # White dummy
        game_9x9.make_move(4, 5, 1)  # Black
        
        # Try suicide move
        assert not game_9x9.make_move(4, 4, 2), \
            f"{impl_name}: Suicide move should be prevented"
    
    @pytest.mark.unit
    def test_ko_rule(self, impl_name, game_9x9):
        """Test ko rule implementation."""
        # Set up ko situation
        # . B W .
        # B . W .
        # W W . .
        moves = [
            (1, 0, 1), (2, 0, 2),  # B, W
            (0, 1, 1), (2, 1, 2),  # B, W
            (0, 2, 2), (1, 2, 2),  # W, W
            (1, 1, 1), (0, 0, 2),  # B in middle, W captures
        ]
        
        for i, (x, y, color) in enumerate(moves[:-1]):
            success = game_9x9.make_move(x, y, color)
            assert success, f"{impl_name}: Move {i} at ({x},{y}) should succeed"
        
        # White captures black
        assert game_9x9.make_move(0, 0, 2), f"{impl_name}: Capturing move should succeed"
        assert game_9x9.get_board_value(1, 1) == 0, f"{impl_name}: Black stone should be captured"
        
        # Black cannot immediately recapture (ko rule)
        assert not game_9x9.make_move(1, 1, 1), \
            f"{impl_name}: Immediate recapture should be prevented by ko rule"
        
        # After another move, ko is lifted
        assert game_9x9.make_move(8, 8, 1), f"{impl_name}: Move elsewhere should succeed"
        assert game_9x9.make_move(8, 7, 2), f"{impl_name}: White move should succeed"
        
        # Now black can recapture
        assert game_9x9.make_move(1, 1, 1), \
            f"{impl_name}: Recapture should be allowed after ko is lifted"
    
    @pytest.mark.unit
    def test_pass_and_game_end(self, impl_name, game_9x9):
        """Test passing and game end by double pass."""
        # Make a few moves
        game_9x9.make_move(4, 4, 1)
        game_9x9.make_move(5, 5, 2)
        
        # First pass
        assert game_9x9.pass_turn(), f"{impl_name}: First pass should succeed"
        assert not game_9x9.is_game_over(), f"{impl_name}: Game should not end on single pass"
        
        # Second pass
        assert game_9x9.pass_turn(), f"{impl_name}: Second pass should succeed"
        assert game_9x9.is_game_over(), f"{impl_name}: Game should end on double pass"
    
    @pytest.mark.unit
    def test_valid_moves_generation(self, impl_name, game_9x9):
        """Test valid moves list generation."""
        # Empty board should have 81 valid moves
        valid_moves = game_9x9.get_valid_moves(1)
        assert len(valid_moves) == 81, \
            f"{impl_name}: Empty 9x9 board should have 81 valid moves"
        
        # Place some stones
        game_9x9.make_move(4, 4, 1)
        game_9x9.make_move(5, 5, 2)
        
        # Should have 79 valid moves
        valid_moves = game_9x9.get_valid_moves(1)
        assert len(valid_moves) == 79, \
            f"{impl_name}: Should have 79 valid moves after 2 stones"
        
        # Occupied positions should not be in valid moves
        assert (4, 4) not in valid_moves, f"{impl_name}: Occupied position should not be valid"
        assert (5, 5) not in valid_moves, f"{impl_name}: Occupied position should not be valid"
    
    @pytest.mark.unit
    @pytest.mark.parametrize("board_size", [9, 13, 19])
    def test_different_board_sizes(self, impl_name, adapter_class, board_size):
        """Test different board sizes."""
        adapter = adapter_class()
        adapter.create_game(board_size)
        
        assert adapter.get_board_size() == board_size, \
            f"{impl_name}: Board size should be {board_size}"
        
        # Check all positions are empty
        empty_count = 0
        for x in range(board_size):
            for y in range(board_size):
                if adapter.get_board_value(x, y) == 0:
                    empty_count += 1
        
        assert empty_count == board_size * board_size, \
            f"{impl_name}: All positions should be empty initially"
        
        # Check valid moves count
        valid_moves = adapter.get_valid_moves(1)
        assert len(valid_moves) == board_size * board_size, \
            f"{impl_name}: Should have {board_size*board_size} valid moves"


class TestCrossValidation:
    """Cross-validation tests between implementations."""
    
    @pytest.mark.integration
    def test_identical_game_sequences(self):
        """Test that all implementations produce identical results for the same game sequence."""
        # Create all available adapters
        adapters = []
        for name, adapter_class in IMPLEMENTATIONS:
            try:
                adapter = adapter_class()
                adapter.create_game(9)
                adapters.append((name, adapter))
            except:
                pass  # Skip unavailable implementations
        
        if len(adapters) < 2:
            pytest.skip("Need at least 2 implementations for cross-validation")
        
        # Test sequence of moves
        test_moves = [
            (4, 4, 1), (3, 3, 2), (5, 5, 1), (3, 5, 2),
            (4, 3, 1), (4, 5, 2), (3, 4, 1), (5, 3, 2),
            (5, 4, 1), (2, 3, 2), (6, 3, 1), (2, 4, 2),
        ]
        
        # Apply moves to all implementations
        for move_idx, (x, y, color) in enumerate(test_moves):
            results = []
            for name, adapter in adapters:
                success = adapter.make_move(x, y, color)
                results.append((name, success))
            
            # All should have same success/failure
            first_result = results[0][1]
            for name, result in results[1:]:
                assert result == first_result, \
                    f"Move {move_idx} ({x},{y},{color}): {name} disagrees with {results[0][0]}"
        
        # Compare final board states
        reference_name, reference = adapters[0]
        for name, adapter in adapters[1:]:
            for x in range(9):
                for y in range(9):
                    ref_value = reference.get_board_value(x, y)
                    test_value = adapter.get_board_value(x, y)
                    assert ref_value == test_value, \
                        f"Board mismatch at ({x},{y}): {reference_name}={ref_value}, {name}={test_value}"
        
        # Compare captures
        ref_captures = reference.get_captures()
        for name, adapter in adapters[1:]:
            captures = adapter.get_captures()
            assert captures == ref_captures, \
                f"Capture mismatch: {reference_name}={ref_captures}, {name}={captures}"
    
    @pytest.mark.integration
    @pytest.mark.slow
    def test_random_game_consistency(self):
        """Test consistency across random games."""
        import random
        random.seed(42)
        
        # Create adapters
        adapters = []
        for name, adapter_class in IMPLEMENTATIONS:
            try:
                adapter = adapter_class()
                adapter.create_game(9)
                adapters.append((name, adapter))
            except:
                pass
        
        if len(adapters) < 2:
            pytest.skip("Need at least 2 implementations for cross-validation")
        
        # Play random game
        moves_played = 0
        max_moves = 100
        
        while moves_played < max_moves:
            # Get valid moves from first implementation
            current_player = adapters[0][1].get_current_player()
            valid_moves = adapters[0][1].get_valid_moves(current_player)
            
            if not valid_moves:
                # All should pass
                for name, adapter in adapters:
                    adapter.pass_turn()
                moves_played += 1
                
                # Check if game over
                if adapters[0][1].is_game_over():
                    break
            else:
                # Random move
                move = random.choice(valid_moves)
                
                # Apply to all
                for name, adapter in adapters:
                    success = adapter.make_move(move[0], move[1], current_player)
                    assert success, f"{name}: Move {move} should succeed"
                
                moves_played += 1
        
        # Final validation - all should have same state
        reference_name, reference = adapters[0]
        for name, adapter in adapters[1:]:
            # Same game over state
            assert adapter.is_game_over() == reference.is_game_over(), \
                f"Game over mismatch: {reference_name}={reference.is_game_over()}, {name}={adapter.is_game_over()}"
            
            # Same captures
            assert adapter.get_captures() == reference.get_captures(), \
                f"Final captures mismatch"
            
            # Same board
            for x in range(9):
                for y in range(9):
                    assert adapter.get_board_value(x, y) == reference.get_board_value(x, y), \
                        f"Final board mismatch at ({x},{y})"


@pytest.mark.benchmark
class TestPerformanceComparison:
    """Compare performance between implementations."""
    
    def test_move_generation_speed(self, benchmark):
        """Benchmark move generation speed."""
        # Skip this in CI - only run locally
        if os.environ.get('CI'):
            pytest.skip("Skipping benchmark in CI")
        
        results = {}
        
        for name, adapter_class in IMPLEMENTATIONS:
            try:
                adapter = adapter_class()
                adapter.create_game(9)
                
                # Benchmark valid move generation
                def generate_moves():
                    return adapter.get_valid_moves(1)
                
                result = benchmark.pedantic(generate_moves, rounds=100, iterations=10)
                results[name] = result
            except:
                pass
        
        # Print comparison (benchmark plugin will show detailed stats)
        print("\nMove generation performance:")
        for name, result in results.items():
            print(f"  {name}: {result}")