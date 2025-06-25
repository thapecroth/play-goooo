import unittest
import numpy as np
from optimized_go import OptimizedGoGame, BLACK, WHITE, EMPTY

class TestOptimizedGoGame(unittest.TestCase):

    def setUp(self):
        self.game = OptimizedGoGame(size=5)

    def test_initial_state(self):
        self.assertEqual(self.game.size, 5)
        self.assertEqual(self.game.current_player, BLACK)
        self.assertFalse(self.game.game_over)
        self.assertEqual(np.sum(self.game.board), 0)

    def test_make_simple_move(self):
        self.assertTrue(self.game.make_move(2, 2, 'black'))
        self.assertEqual(self.game.board[2, 2], BLACK)
        self.assertEqual(self.game.current_player, WHITE)

    def test_invalid_move_occupied(self):
        self.game.make_move(2, 2, 'black')
        self.assertFalse(self.game.make_move(2, 2, 'white'))

    def test_simple_capture(self):
        #   B . .
        # B W B
        #   B . 
        self.game.board[0, 1] = BLACK
        self.game.board[1, 0] = BLACK
        self.game.board[2, 1] = BLACK
        self.game.board[1, 1] = WHITE
        self.game.current_player = BLACK
        
        # Black plays at (1,2) to capture the white stone
        self.assertTrue(self.game.make_move(1, 2, 'black'))
        self.assertEqual(self.game.board[1, 1], EMPTY, "Stone should be captured")
        self.assertEqual(self.game.captures[BLACK], 1)

    def test_self_capture_invalid(self):
        # W . .
        # W B .
        # . . .
        self.game.board[0, 0] = WHITE
        self.game.board[1, 0] = WHITE
        self.game.current_player = BLACK
        # Black attempts to play at (0,0) surrounded by white, which is suicide
        self.assertFalse(self.game.make_move(0, 1, 'black'), "Suicide move should be invalid")

    def test_self_capture_valid_by_capture(self):
        # . W .
        # W B W
        # . W .
        self.game.board[0, 1] = WHITE
        self.game.board[1, 0] = WHITE
        self.game.board[1, 2] = WHITE
        self.game.board[2, 1] = WHITE
        self.game.board[1, 1] = BLACK
        self.game.current_player = BLACK
        # Black plays in the middle of the white group, capturing it.
        # This is a valid move.
        self.assertTrue(self.game.make_move(0, 0, 'black'))
        self.assertEqual(self.game.board[0,1], EMPTY)
        self.assertEqual(self.game.board[1,0], EMPTY)
        self.assertEqual(self.game.board[1,2], EMPTY)
        self.assertEqual(self.game.board[2,1], EMPTY)
        self.assertEqual(self.game.captures[BLACK], 4)

    def test_ko_rule(self):
        # Setup a ko situation
        # . B . .
        # B W B .
        # . B W .
        # . . . .
        self.game.board[0, 1] = BLACK
        self.game.board[1, 0] = BLACK
        self.game.board[1, 2] = BLACK
        self.game.board[2, 1] = BLACK
        self.game.board[1, 1] = WHITE
        self.game.board[2, 2] = WHITE
        self.game.current_player = BLACK

        # Black captures the white stone at (1,1)
        self.assertTrue(self.game.make_move(2, 0, 'black'))
        self.assertEqual(self.game.board[1,1], EMPTY)
        self.assertIsNotNone(self.game.ko_point, "Ko point should be set")

        # White attempts to immediately recapture, which is illegal
        self.assertFalse(self.game.make_move(1, 1, 'white'), "Recapturing in a ko is illegal")

        # White plays elsewhere
        self.assertTrue(self.game.make_move(4, 4, 'white'))
        self.assertIsNone(self.game.ko_point, "Ko point should be reset after a move elsewhere")

        # Black plays elsewhere
        self.assertTrue(self.game.make_move(0, 0, 'black'))

        # Now white can legally recapture the ko position
        self.assertTrue(self.game.make_move(1, 1, 'white'))

    def test_pass_turn(self):
        self.assertTrue(self.game.pass_turn('black'))
        self.assertEqual(self.game.current_player, WHITE)
        self.assertEqual(self.game.passes, 1)

    def test_game_over_by_pass(self):
        self.game.pass_turn('black')
        self.game.pass_turn('white')
        self.assertTrue(self.game.game_over)

if __name__ == '__main__':
    unittest.main()