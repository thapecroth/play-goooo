"""
Comprehensive pytest tests for Go game rules implementation.
Tests both the basic GoGame and OptimizedGoGame implementations.
"""

import pytest
import numpy as np
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from server import GoGame
from optimized_go import OptimizedGoGame, EMPTY, BLACK, WHITE


class TestGoGameBasics:
    """Test basic Go game functionality for both implementations."""
    
    @pytest.fixture
    def basic_game(self):
        """Create a basic GoGame instance."""
        return GoGame(9)
    
    @pytest.fixture
    def optimized_game(self):
        """Create an OptimizedGoGame instance."""
        return OptimizedGoGame(9)
    
    @pytest.mark.unit
    def test_initial_state_basic(self, basic_game):
        """Test initial state of basic game."""
        assert basic_game.size == 9
        assert basic_game.current_player == 'black'
        assert basic_game.captures == {'black': 0, 'white': 0}
        assert not basic_game.game_over
        assert basic_game.winner is None
        assert basic_game.passes == 0
        
        # Check empty board
        for y in range(9):
            for x in range(9):
                assert basic_game.board[y][x] is None
    
    @pytest.mark.unit
    def test_initial_state_optimized(self, optimized_game):
        """Test initial state of optimized game."""
        assert optimized_game.size == 9
        assert optimized_game.current_player == BLACK
        assert optimized_game.captures == {BLACK: 0, WHITE: 0}
        assert not optimized_game.game_over
        assert optimized_game.winner is None
        assert optimized_game.passes == 0
        
        # Check empty board
        assert np.all(optimized_game.board == EMPTY)
    
    @pytest.mark.unit
    def test_basic_move_placement(self, basic_game, optimized_game):
        """Test basic stone placement."""
        # Basic game
        assert basic_game.make_move(4, 4, 'black')
        assert basic_game.board[4][4] == 'black'
        assert basic_game.current_player == 'white'
        
        # Optimized game
        assert optimized_game.make_move(4, 4, BLACK)
        assert optimized_game.board[4, 4] == BLACK
        assert optimized_game.current_player == WHITE
    
    @pytest.mark.unit
    def test_invalid_moves(self, basic_game, optimized_game):
        """Test invalid move detection."""
        # Place a black stone
        basic_game.make_move(4, 4, 'black')
        optimized_game.make_move(4, 4, BLACK)
        
        # Try to place on occupied position
        assert not basic_game.make_move(4, 4, 'white')
        assert not optimized_game.make_move(4, 4, WHITE)
        
        # Try out of bounds moves
        assert not basic_game.make_move(-1, 0, 'white')
        assert not basic_game.make_move(9, 0, 'white')
        assert not optimized_game.make_move(-1, 0, WHITE)
        assert not optimized_game.make_move(9, 0, WHITE)


class TestCaptures:
    """Test capture mechanics."""
    
    @pytest.fixture
    def capture_setup_basic(self):
        """Set up a basic game with a capture situation."""
        game = GoGame(9)
        # White stone surrounded by black stones
        game.make_move(4, 4, 'white')
        game.make_move(3, 4, 'black')
        game.make_move(0, 0, 'white')  # Dummy move
        game.make_move(5, 4, 'black')
        game.make_move(0, 1, 'white')  # Dummy move
        game.make_move(4, 3, 'black')
        return game
    
    @pytest.fixture
    def capture_setup_optimized(self):
        """Set up an optimized game with a capture situation."""
        game = OptimizedGoGame(9)
        # White stone surrounded by black stones
        game.make_move(4, 4, WHITE)
        game.make_move(3, 4, BLACK)
        game.make_move(0, 0, WHITE)  # Dummy move
        game.make_move(5, 4, BLACK)
        game.make_move(0, 1, WHITE)  # Dummy move
        game.make_move(4, 3, BLACK)
        return game
    
    @pytest.mark.unit
    def test_single_stone_capture_basic(self, capture_setup_basic):
        """Test capturing a single stone in basic game."""
        game = capture_setup_basic
        # Complete the capture
        assert game.make_move(4, 5, 'black')
        
        # Check stone is captured
        assert game.board[4][4] is None
        assert game.captures['black'] == 1
    
    @pytest.mark.unit
    def test_single_stone_capture_optimized(self, capture_setup_optimized):
        """Test capturing a single stone in optimized game."""
        game = capture_setup_optimized
        # Complete the capture
        assert game.make_move(4, 5, BLACK)
        
        # Check stone is captured
        assert game.board[4, 4] == EMPTY
        assert game.captures[BLACK] == 1
    
    @pytest.mark.unit
    def test_group_capture(self):
        """Test capturing a group of stones."""
        game = OptimizedGoGame(9)
        
        # Create a white group
        game.make_move(3, 3, WHITE)
        game.make_move(2, 3, BLACK)
        game.make_move(3, 4, WHITE)
        game.make_move(2, 4, BLACK)
        game.make_move(4, 3, WHITE)
        game.make_move(5, 3, BLACK)
        game.make_move(4, 4, WHITE)
        game.make_move(5, 4, BLACK)
        
        # Surround the group
        game.make_move(0, 0, WHITE)  # Dummy
        game.make_move(3, 2, BLACK)
        game.make_move(0, 1, WHITE)  # Dummy
        game.make_move(4, 2, BLACK)
        game.make_move(0, 2, WHITE)  # Dummy
        game.make_move(3, 5, BLACK)
        game.make_move(0, 3, WHITE)  # Dummy
        
        # Complete the capture
        assert game.make_move(4, 5, BLACK)
        
        # Check all stones are captured
        assert game.board[3, 3] == EMPTY
        assert game.board[3, 4] == EMPTY
        assert game.board[4, 3] == EMPTY
        assert game.board[4, 4] == EMPTY
        assert game.captures[BLACK] == 4


class TestKoRule:
    """Test ko rule implementation."""
    
    @pytest.mark.unit
    def test_ko_detection_basic(self):
        """Test ko rule in basic game."""
        game = GoGame(9)
        
        # Set up a ko situation
        moves = [
            (3, 3, 'black'), (3, 4, 'white'),
            (4, 2, 'black'), (4, 5, 'white'),
            (5, 3, 'black'), (5, 4, 'white'),
            (4, 4, 'black'), (2, 4, 'white'),
            (1, 0, 'black'), (4, 3, 'white')  # Capture black at (4,4)
        ]
        
        for x, y, color in moves:
            assert game.make_move(x, y, color)
        
        # Try to immediately recapture - should fail due to ko
        assert not game.make_move(4, 4, 'black')
        
        # After another move elsewhere, ko is lifted
        assert game.make_move(8, 8, 'black')
        assert game.make_move(8, 7, 'white')
        assert game.make_move(4, 4, 'black')  # Now legal
    
    @pytest.mark.unit
    def test_ko_detection_optimized(self):
        """Test ko rule in optimized game."""
        game = OptimizedGoGame(9)
        
        # Set up a ko situation
        game.board[3, 3] = BLACK
        game.board[3, 4] = WHITE
        game.board[4, 2] = BLACK
        game.board[4, 5] = WHITE
        game.board[5, 3] = BLACK
        game.board[5, 4] = WHITE
        game.board[4, 4] = BLACK
        game.board[2, 4] = WHITE
        
        # White captures black at (4,4)
        game.current_player = WHITE
        assert game.make_move(4, 3, WHITE)
        assert game.ko_point == (4, 4)
        
        # Try to immediately recapture - should fail due to ko
        assert not game.make_move(4, 4, BLACK)
        
        # After another move elsewhere, ko is lifted
        assert game.make_move(8, 8, BLACK)
        assert game.ko_point is None
        assert game.make_move(8, 7, WHITE)
        assert game.make_move(4, 4, BLACK)  # Now legal


class TestSelfCapture:
    """Test self-capture (suicide) rules."""
    
    @pytest.mark.unit
    def test_suicide_move_prevention(self):
        """Test that suicide moves are prevented."""
        game = OptimizedGoGame(9)
        
        # Surround a point with opponent stones
        game.make_move(3, 4, BLACK)
        game.make_move(0, 0, WHITE)
        game.make_move(5, 4, BLACK)
        game.make_move(0, 1, WHITE)
        game.make_move(4, 3, BLACK)
        game.make_move(0, 2, WHITE)
        game.make_move(4, 5, BLACK)
        
        # Try to play in the surrounded point - should fail
        assert not game.make_move(4, 4, WHITE)
    
    @pytest.mark.unit
    def test_suicide_allowed_if_captures(self):
        """Test that suicide is allowed if it captures opponent stones."""
        game = OptimizedGoGame(9)
        
        # Create a situation where white can capture by playing a "suicide" move
        # Black group with one liberty
        game.board[3, 3] = BLACK
        game.board[3, 4] = BLACK
        game.board[4, 3] = BLACK
        
        # White stones surrounding, leaving (4,4) as liberty
        game.board[2, 3] = WHITE
        game.board[2, 4] = WHITE
        game.board[3, 2] = WHITE
        game.board[4, 2] = WHITE
        game.board[5, 3] = WHITE
        game.board[5, 4] = WHITE
        game.board[3, 5] = WHITE
        game.board[4, 5] = WHITE
        
        game.current_player = WHITE
        
        # This move would be suicide except it captures the black group
        assert game.make_move(4, 4, WHITE)
        assert game.board[3, 3] == EMPTY
        assert game.board[3, 4] == EMPTY
        assert game.board[4, 3] == EMPTY


class TestGameEnd:
    """Test game ending conditions."""
    
    @pytest.mark.unit
    def test_double_pass_ends_game(self):
        """Test that two consecutive passes end the game."""
        game = OptimizedGoGame(9)
        
        # Make some moves
        game.make_move(4, 4, BLACK)
        game.make_move(4, 5, WHITE)
        
        # Both players pass
        game.pass_turn()
        assert game.passes == 1
        assert not game.game_over
        
        game.pass_turn()
        assert game.passes == 2
        assert game.game_over
    
    @pytest.mark.unit
    def test_resign_ends_game(self):
        """Test that resignation ends the game."""
        game = OptimizedGoGame(9)
        
        game.make_move(4, 4, BLACK)
        game.resign()
        
        assert game.game_over
        assert game.winner == WHITE  # Black resigned, so white wins
    
    @pytest.mark.unit
    def test_territory_scoring(self):
        """Test basic territory scoring."""
        game = OptimizedGoGame(9)
        
        # Create a simple position with clear territories
        # Black controls top-left
        for i in range(3):
            game.board[i, 3] = BLACK
            game.board[3, i] = BLACK
        
        # White controls bottom-right
        for i in range(5, 9):
            game.board[i, 5] = WHITE
            game.board[5, i] = WHITE
        
        scores = game.get_scores()
        assert scores['black'] > 0
        assert scores['white'] > 0


class TestValidMoves:
    """Test valid move generation."""
    
    @pytest.mark.unit
    def test_get_valid_moves(self):
        """Test that valid moves are correctly identified."""
        game = OptimizedGoGame(9)
        
        # Initially all positions should be valid
        valid_moves = game.get_valid_moves(BLACK)
        assert len(valid_moves) == 81
        
        # Place some stones
        game.make_move(4, 4, BLACK)
        game.make_move(4, 5, WHITE)
        
        # Should have 2 fewer valid moves
        valid_moves = game.get_valid_moves(BLACK)
        assert len(valid_moves) == 79
        
        # Create a suicide position
        game.board[0, 1] = WHITE
        game.board[1, 0] = WHITE
        
        # (0,0) should not be valid for white (suicide)
        valid_moves = game.get_valid_moves(WHITE)
        assert (0, 0) not in [(m[0], m[1]) for m in valid_moves]


class TestPerformance:
    """Performance tests for the game engines."""
    
    @pytest.mark.slow
    @pytest.mark.integration
    def test_large_game_performance(self):
        """Test performance with a 19x19 board."""
        import time
        
        game = OptimizedGoGame(19)
        
        start_time = time.time()
        
        # Play 100 random moves
        for i in range(100):
            valid_moves = game.get_valid_moves(game.current_player)
            if valid_moves:
                import random
                move = random.choice(valid_moves)
                game.make_move(move[0], move[1], game.current_player)
        
        elapsed = time.time() - start_time
        assert elapsed < 1.0  # Should complete in under 1 second
        
        # Test that game state is still valid
        assert not game.game_over
        assert game.current_player in [BLACK, WHITE]


@pytest.mark.parametrize("board_size", [9, 13, 19])
class TestMultipleBoardSizes:
    """Test game works correctly with different board sizes."""
    
    @pytest.mark.unit
    def test_board_initialization(self, board_size):
        """Test boards initialize correctly at different sizes."""
        game = OptimizedGoGame(board_size)
        assert game.size == board_size
        assert game.board.shape == (board_size, board_size)
        assert np.all(game.board == EMPTY)
    
    @pytest.mark.unit
    def test_valid_moves_count(self, board_size):
        """Test correct number of valid moves for empty board."""
        game = OptimizedGoGame(board_size)
        valid_moves = game.get_valid_moves(BLACK)
        assert len(valid_moves) == board_size * board_size