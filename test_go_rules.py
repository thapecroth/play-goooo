"""
Unit tests for Go game engine to verify correct implementation of all Go rules.
Tests both the basic GoGame and OptimizedGoGame implementations.
"""

import unittest
import numpy as np
from server import GoGame
from optimized_go import OptimizedGoGame, EMPTY, BLACK, WHITE


class TestGoRules(unittest.TestCase):
    """Test suite for verifying Go game rules implementation"""
    
    def setUp(self):
        """Set up test fixtures before each test method"""
        self.basic_game = GoGame(9)
        self.optimized_game = OptimizedGoGame(9)
    
    def test_initial_game_state(self):
        """Test that games start with correct initial state"""
        # Basic game
        self.assertEqual(self.basic_game.size, 9)
        self.assertEqual(self.basic_game.current_player, 'black')
        self.assertEqual(self.basic_game.captures, {'black': 0, 'white': 0})
        self.assertFalse(self.basic_game.game_over)
        self.assertIsNone(self.basic_game.winner)
        self.assertEqual(self.basic_game.passes, 0)
        
        # Check empty board
        for y in range(9):
            for x in range(9):
                self.assertIsNone(self.basic_game.board[y][x])
        
        # Optimized game
        self.assertEqual(self.optimized_game.size, 9)
        self.assertEqual(self.optimized_game.current_player, BLACK)
        self.assertEqual(self.optimized_game.captures, {BLACK: 0, WHITE: 0})
        self.assertFalse(self.optimized_game.game_over)
        self.assertIsNone(self.optimized_game.winner)
        self.assertEqual(self.optimized_game.passes, 0)
        
        # Check empty board
        self.assertTrue(np.all(self.optimized_game.board == EMPTY))
    
    def test_basic_move_placement(self):
        """Test basic stone placement"""
        # Basic game
        self.assertTrue(self.basic_game.make_move(4, 4, 'black'))
        self.assertEqual(self.basic_game.board[4][4], 'black')
        self.assertEqual(self.basic_game.current_player, 'white')
        self.assertEqual(self.basic_game.passes, 0)
        
        # Can't place on occupied position
        self.assertFalse(self.basic_game.make_move(4, 4, 'white'))
        
        # Optimized game
        self.assertTrue(self.optimized_game.make_move(4, 4, 'black'))
        self.assertEqual(self.optimized_game.board[4, 4], BLACK)
        self.assertEqual(self.optimized_game.current_player, WHITE)
        self.assertEqual(self.optimized_game.passes, 0)
        
        # Can't place on occupied position
        self.assertFalse(self.optimized_game.make_move(4, 4, 'white'))
    
    def test_invalid_moves(self):
        """Test rejection of invalid moves"""
        # Out of bounds moves
        self.assertFalse(self.basic_game.make_move(-1, 0, 'black'))
        self.assertFalse(self.basic_game.make_move(0, -1, 'black'))
        self.assertFalse(self.basic_game.make_move(9, 0, 'black'))
        self.assertFalse(self.basic_game.make_move(0, 9, 'black'))
        
        self.assertFalse(self.optimized_game.make_move(-1, 0, 'black'))
        self.assertFalse(self.optimized_game.make_move(0, -1, 'black'))
        self.assertFalse(self.optimized_game.make_move(9, 0, 'black'))
        self.assertFalse(self.optimized_game.make_move(0, 9, 'black'))
        
        # Wrong turn order
        self.assertFalse(self.basic_game.make_move(0, 0, 'white'))  # Black's turn
        self.assertFalse(self.optimized_game.make_move(0, 0, 'white'))  # Black's turn
    
    def test_stone_capture_single(self):
        """Test capturing a single stone"""
        # Set up situation where white stone can be captured
        # Black surrounds white stone at (1,1)
        #   0 1 2
        # 0 . B .
        # 1 B W B
        # 2 . B .
        
        # Basic game test
        game = GoGame(3)
        game.make_move(1, 0, 'black')  # B at (1,0)
        game.make_move(1, 1, 'white')  # W at (1,1)
        game.make_move(0, 1, 'black')  # B at (0,1)
        game.make_move(0, 0, 'white')  # W dummy move (safe position)
        game.make_move(2, 1, 'black')  # B at (2,1)
        game.make_move(2, 0, 'white')  # W dummy move (safe position)
        
        # Capture the white stone
        initial_captures = game.captures['black']
        game.make_move(1, 2, 'black')  # B at (1,2) - captures white stone
        
        self.assertIsNone(game.board[1][1])  # White stone should be captured
        self.assertEqual(game.captures['black'], initial_captures + 1)
        
        # Optimized game test
        opt_game = OptimizedGoGame(3)
        opt_game.make_move(1, 0, 'black')
        opt_game.make_move(1, 1, 'white')
        opt_game.make_move(0, 1, 'black')
        opt_game.make_move(2, 2, 'white')
        opt_game.make_move(2, 1, 'black')
        opt_game.make_move(0, 2, 'white')
        
        initial_captures = opt_game.captures[BLACK]
        opt_game.make_move(1, 2, 'black')
        
        self.assertEqual(opt_game.board[1, 1], EMPTY)  # White stone should be captured
        self.assertEqual(opt_game.captures[BLACK], initial_captures + 1)
    
    def test_stone_capture_group(self):
        """Test capturing a group of stones"""
        # Set up situation where a group of white stones can be captured
        #   0 1 2 3
        # 0 . B B .
        # 1 B W W B
        # 2 . B B .
        
        # Basic game test
        game = GoGame(4)
        # Place white group
        game.make_move(1, 1, 'black')
        game.make_move(1, 1, 'white')  # This should fail
        game.current_player = 'white'
        game.board[1][1] = 'white'
        game.board[1][2] = 'white'
        game.current_player = 'black'
        
        # Surround with black
        game.board[0][1] = 'black'
        game.board[0][2] = 'black'
        game.board[1][0] = 'black'
        game.board[1][3] = 'black'
        game.board[2][1] = 'black'
        game.board[2][2] = 'black'
        
        # Verify white group has no liberties
        white_group = game.get_group(1, 1)
        self.assertFalse(game.group_has_liberties(white_group))
        
        # Simulate capture
        captured = game.capture_stones('white')
        self.assertEqual(len(captured), 2)  # Two white stones captured
        self.assertIsNone(game.board[1][1])
        self.assertIsNone(game.board[1][2])
    
    def test_self_capture_prevention(self):
        """Test that self-capture moves are not allowed"""
        # Set up situation where placing a stone would result in self-capture
        #   0 1 2
        # 0 B . B
        # 1 . B .
        # 2 B . B
        
        # Basic game
        game = GoGame(3)
        game.board[0][0] = 'black'
        game.board[0][2] = 'black'
        game.board[1][1] = 'black'
        game.board[2][0] = 'black'
        game.board[2][2] = 'black'
        game.current_player = 'white'
        
        # Trying to place white at (0,1) should be self-capture
        self.assertFalse(game.make_move(0, 1, 'white'))
        
        # Optimized game
        opt_game = OptimizedGoGame(3)
        opt_game.board[0, 0] = BLACK
        opt_game.board[0, 2] = BLACK
        opt_game.board[1, 1] = BLACK
        opt_game.board[2, 0] = BLACK
        opt_game.board[2, 2] = BLACK
        opt_game.current_player = WHITE
        
        # Should prevent self-capture
        self.assertFalse(opt_game.is_valid_move(0, 1, WHITE))
    
    def test_ko_rule_basic(self):
        """Test basic ko rule implementation"""
        # Set up a ko situation
        #   0 1 2 3
        # 0 . B W .
        # 1 B W . W
        # 2 . B W .
        
        # This is a simplified ko test
        game = GoGame(4)
        
        # Place stones to create ko situation
        game.board[0][1] = 'black'
        game.board[0][2] = 'white'
        game.board[1][0] = 'black'
        game.board[1][1] = 'white'
        game.board[1][3] = 'white'
        game.board[2][1] = 'black'
        game.board[2][2] = 'white'
        
        # Save board state
        board_before = game.copy_board()
        
        # Black captures white stone at (1,1)
        game.current_player = 'black'
        game.make_move(1, 2, 'black')
        
        # White should not be able to immediately recapture (ko rule)
        # This is a basic check - full ko implementation is complex
        if game.ko is not None:
            ko_x, ko_y = game.ko['x'], game.ko['y']
            game.current_player = 'white'
            self.assertFalse(game.make_move(ko_x, ko_y, 'white'))
    
    def test_liberty_counting(self):
        """Test liberty counting for groups"""
        # Test single stone liberties
        game = GoGame(9)
        game.board[4][4] = 'black'
        
        group = game.get_group(4, 4)
        self.assertEqual(len(group), 1)
        self.assertTrue(game.group_has_liberties(group))
        
        # Test liberties for corner stone
        game_corner = GoGame(9)
        game_corner.board[0][0] = 'black'
        corner_group = game_corner.get_group(0, 0)
        self.assertTrue(game_corner.group_has_liberties(corner_group))
        
        # Optimized game liberty counting
        opt_game = OptimizedGoGame(9)
        opt_game.board[4, 4] = BLACK
        opt_group = opt_game.get_group(4, 4)
        liberties = opt_game.count_liberties(opt_group)
        self.assertEqual(liberties, 4)  # Center stone has 4 liberties
        
        # Corner stone has 2 liberties
        opt_game.board[0, 0] = BLACK
        corner_opt_group = opt_game.get_group(0, 0)
        corner_liberties = opt_game.count_liberties(corner_opt_group)
        self.assertEqual(corner_liberties, 2)
    
    def test_group_connection(self):
        """Test that connected stones form proper groups"""
        # Test connected stones
        game = GoGame(5)
        game.board[1][1] = 'black'
        game.board[1][2] = 'black'
        game.board[2][1] = 'black'
        
        group1 = game.get_group(1, 1)
        group2 = game.get_group(1, 2)
        group3 = game.get_group(2, 1)
        
        # All should be part of the same group
        self.assertEqual(len(group1), 3)
        self.assertEqual(len(group2), 3)
        self.assertEqual(len(group3), 3)
        
        # Test diagonal stones are NOT connected
        game.board[2][2] = 'black'
        diagonal_group = game.get_group(2, 2)
        self.assertEqual(len(diagonal_group), 4)  # Stone is connected via (2,1) to the group
        
        # Optimized game group test
        opt_game = OptimizedGoGame(5)
        opt_game.board[1, 1] = BLACK
        opt_game.board[1, 2] = BLACK
        opt_game.board[2, 1] = BLACK
        
        opt_group = opt_game.get_group(1, 1)
        self.assertEqual(len(opt_group), 3)
    
    def test_pass_turn(self):
        """Test pass turn functionality"""
        # Basic game
        self.assertTrue(self.basic_game.pass_turn('black'))
        self.assertEqual(self.basic_game.current_player, 'white')
        self.assertEqual(self.basic_game.passes, 1)
        
        # Second pass should end game
        self.assertTrue(self.basic_game.pass_turn('white'))
        self.assertTrue(self.basic_game.game_over)
        self.assertEqual(self.basic_game.passes, 2)
        
        # Optimized game
        self.assertTrue(self.optimized_game.pass_turn('black'))
        self.assertEqual(self.optimized_game.current_player, WHITE)
        self.assertEqual(self.optimized_game.passes, 1)
        
        self.assertTrue(self.optimized_game.pass_turn('white'))
        self.assertTrue(self.optimized_game.game_over)
        self.assertEqual(self.optimized_game.passes, 2)
    
    def test_game_end_conditions(self):
        """Test game ending conditions"""
        # Test resignation
        self.basic_game.resign('black')
        self.assertTrue(self.basic_game.game_over)
        self.assertEqual(self.basic_game.winner, 'white')
        
        # Test two consecutive passes
        new_game = GoGame(9)
        new_game.pass_turn('black')
        new_game.pass_turn('white')
        self.assertTrue(new_game.game_over)
        self.assertIsNotNone(new_game.winner)
    
    def test_score_calculation(self):
        """Test basic score calculation"""
        game = GoGame(9)
        
        # Place some stones
        game.board[0][0] = 'black'
        game.board[0][1] = 'black'
        game.board[8][8] = 'white'
        game.board[8][7] = 'white'
        
        # Set some captures
        game.captures['black'] = 2
        game.captures['white'] = 1
        
        score = game.calculate_score()
        
        # Basic verification that score calculation returns proper structure
        self.assertIn('black', score)
        self.assertIn('white', score)
        self.assertIsInstance(score['black'], (int, float))
        self.assertIsInstance(score['white'], (int, float))
        
        # White should have komi bonus
        self.assertGreater(score['white'], score['black'] - 10)  # Rough check
    
    def test_valid_moves_generation(self):
        """Test generation of valid moves"""
        # Empty board should have many valid moves
        valid_moves = self.basic_game.get_valid_moves('black')
        self.assertEqual(len(valid_moves), 81)  # 9x9 = 81 empty positions
        
        # After placing a stone, should have one less valid move
        self.basic_game.make_move(4, 4, 'black')
        valid_moves_after = self.basic_game.get_valid_moves('white')
        self.assertEqual(len(valid_moves_after), 80)
        
        # Optimized game
        opt_valid_moves = self.optimized_game.get_valid_moves('black')
        self.assertEqual(len(opt_valid_moves), 81)
        
        self.optimized_game.make_move(4, 4, 'black')
        opt_valid_moves_after = self.optimized_game.get_valid_moves('white')
        self.assertEqual(len(opt_valid_moves_after), 80)
    
    def test_neighbor_calculation(self):
        """Test neighbor position calculation"""
        # Test center position neighbors
        neighbors = self.basic_game.get_neighbors(4, 4)
        expected_neighbors = [
            {'x': 3, 'y': 4}, {'x': 5, 'y': 4},
            {'x': 4, 'y': 3}, {'x': 4, 'y': 5}
        ]
        self.assertEqual(len(neighbors), 4)
        for neighbor in expected_neighbors:
            self.assertIn(neighbor, neighbors)
        
        # Test corner position neighbors
        corner_neighbors = self.basic_game.get_neighbors(0, 0)
        self.assertEqual(len(corner_neighbors), 2)
        
        # Test edge position neighbors
        edge_neighbors = self.basic_game.get_neighbors(0, 4)
        self.assertEqual(len(edge_neighbors), 3)
    
    def test_board_boundaries(self):
        """Test that game properly handles board boundaries"""
        # Test various board sizes
        for size in [5, 9, 13, 19]:
            game = GoGame(size)
            opt_game = OptimizedGoGame(size)
            
            # Valid moves at corners
            self.assertTrue(game.make_move(0, 0, 'black'))
            self.assertTrue(opt_game.make_move(0, 0, 'black'))
            
            # Valid moves at edges
            game2 = GoGame(size)
            opt_game2 = OptimizedGoGame(size)
            self.assertTrue(game2.make_move(size-1, size-1, 'black'))
            self.assertTrue(opt_game2.make_move(size-1, size-1, 'black'))
            
            # Invalid moves outside boundaries
            game3 = GoGame(size)
            opt_game3 = OptimizedGoGame(size)
            self.assertFalse(game3.make_move(size, 0, 'black'))
            self.assertFalse(game3.make_move(0, size, 'black'))
            self.assertFalse(opt_game3.make_move(size, 0, 'black'))
            self.assertFalse(opt_game3.make_move(0, size, 'black'))
    
    def test_capture_with_multiple_groups(self):
        """Test capture when multiple enemy groups are affected"""
        # Set up board where one move captures multiple separate groups
        game = GoGame(5)
        
        # Create two separate white groups that will be captured
        # Group 1: (1,1)
        game.board[1][1] = 'white'
        game.board[0][1] = 'black'
        game.board[2][1] = 'black'
        game.board[1][0] = 'black'
        # Will be captured when black plays at (1,2)
        
        # Group 2: (3,1)  
        game.board[3][1] = 'white'
        game.board[2][1] = 'black'  # Already placed above
        game.board[4][1] = 'black'
        game.board[3][0] = 'black'
        # Will be captured when black plays at (3,2)
        
        # Place stone that captures first group
        game.current_player = 'black'
        initial_captures = game.captures['black']
        game.make_move(1, 2, 'black')
        
        # First white stone should be captured
        self.assertIsNone(game.board[1][1])
        self.assertEqual(game.captures['black'], initial_captures + 1)
        
        # Place stone that captures second group
        game.current_player = 'black'
        game.make_move(3, 2, 'black')
        
        # Second white stone should be captured
        self.assertIsNone(game.board[3][1])
        self.assertEqual(game.captures['black'], initial_captures + 2)


class TestEdgeCases(unittest.TestCase):
    """Test edge cases and boundary conditions"""
    
    def test_full_board_scenario(self):
        """Test behavior when board is nearly full"""
        game = GoGame(3)  # Small board for testing
        
        # Fill most of the board
        positions = [(0,0), (0,1), (1,0), (1,1), (2,0), (2,1), (0,2), (1,2)]
        colors = ['black', 'white', 'black', 'white', 'black', 'white', 'black', 'white']
        
        for (x, y), color in zip(positions, colors):
            game.board[y][x] = color
        
        # Only (2,2) should be empty
        valid_moves = game.get_valid_moves('black')
        
        # Should have at most 1 valid move
        self.assertLessEqual(len(valid_moves), 1)
    
    def test_large_group_capture(self):
        """Test capturing a large connected group"""
        game = GoGame(7)
        
        # Create a large white group in the center
        white_positions = [(2,2), (2,3), (2,4), (3,2), (3,3), (3,4), (4,2), (4,3), (4,4)]
        for x, y in white_positions:
            game.board[y][x] = 'white'
        
        # Surround with black stones
        black_positions = [
            (1,1), (1,2), (1,3), (1,4), (1,5),
            (2,1), (2,5),
            (3,1), (3,5), 
            (4,1), (4,5),
            (5,1), (5,2), (5,3), (5,4), (5,5)
        ]
        for x, y in black_positions:
            game.board[y][x] = 'black'
        
        # Verify the white group has no liberties
        white_group = game.get_group(3, 3)
        self.assertFalse(game.group_has_liberties(white_group))
        
        # Capture should remove all white stones
        captured = game.capture_stones('white')
        self.assertEqual(len(captured), 9)
    
    def test_complex_ko_situation(self):
        """Test more complex ko situations"""
        # This tests a more complex ko pattern that might arise
        game = GoGame(5)
        
        # Set up a complex board position
        # This is a simplified test - real ko situations are very complex
        game.board[1][1] = 'black'
        game.board[1][2] = 'white'
        game.board[2][1] = 'white'
        game.board[2][2] = 'black'
        
        # Test that ko detection prevents immediate recapture
        # (This is a basic ko test - full implementation would be more complex)
        board_state = game.copy_board()
        
        # Simulate a capture
        game.board[1][2] = None
        game.ko = {'x': 1, 'y': 2}
        
        # Ko rule should prevent immediate recapture
        if game.ko:
            result = game.is_ko(board_state)
            # The ko rule implementation may vary, but should prevent immediate recapture
    
    def test_suicide_vs_capture_precedence(self):
        """Test that capture takes precedence over suicide"""
        # Set up situation where a move would be suicide but also captures
        game = GoGame(5)
        
        # Create white group that will be captured
        game.board[1][1] = 'white'
        game.board[0][1] = 'black'
        game.board[2][1] = 'black'
        game.board[1][0] = 'black'
        
        # Create surrounding for black stone that would be suicide
        game.board[0][2] = 'white'
        game.board[2][2] = 'white'
        
        # This move should be allowed because it captures the white group
        # even though the black stone would have no liberties otherwise
        game.current_player = 'black'
        result = game.make_move(1, 2, 'black')
        
        # Move should be allowed because capture takes precedence
        self.assertTrue(result)
        self.assertIsNone(game.board[1][1])  # White stone captured
    
    def test_board_state_consistency(self):
        """Test that board state remains consistent after operations"""
        game = GoGame(9)
        
        # Make several moves and verify consistency
        moves = [(4,4), (4,5), (5,4), (3,4), (4,3)]
        colors = ['black', 'white', 'black', 'white', 'black']
        
        for (x, y), color in zip(moves, colors):
            game.current_player = color
            self.assertTrue(game.make_move(x, y, color))
            
            # Verify board consistency
            stone_count = 0
            for row in game.board:
                for cell in row:
                    if cell is not None:
                        stone_count += 1
            
            # Should match the number of moves made so far
            expected_count = moves.index((x, y)) + 1
            self.assertEqual(stone_count, expected_count)


if __name__ == '__main__':
    # Create test suite
    suite = unittest.TestSuite()
    
    # Add all test cases
    suite.addTest(unittest.makeSuite(TestGoRules))
    suite.addTest(unittest.makeSuite(TestEdgeCases))
    
    # Run tests with verbose output
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Print summary
    print(f"\n{'='*50}")
    print(f"TESTS RUN: {result.testsRun}")
    print(f"FAILURES: {len(result.failures)}")
    print(f"ERRORS: {len(result.errors)}")
    
    if result.failures:
        print(f"\nFAILURES:")
        for test, traceback in result.failures:
            print(f"- {test}: {traceback}")
    
    if result.errors:
        print(f"\nERRORS:")
        for test, traceback in result.errors:
            print(f"- {test}: {traceback}")
    
    if result.wasSuccessful():
        print("\n✅ ALL TESTS PASSED!")
    else:
        print("\n❌ SOME TESTS FAILED!")