
import unittest
import torch
from alpha_go import AlphaGoPlayer, PolicyValueNet
from optimized_go import OptimizedGoGame

class TestAlphaGo(unittest.TestCase):

    def setUp(self):
        self.board_size = 5
        self.policy_value_net = PolicyValueNet(board_size=self.board_size)
        self.player = AlphaGoPlayer(self.policy_value_net, simulations=10)
        self.game = OptimizedGoGame(size=self.board_size)

    def test_player_initialization(self):
        self.assertIsNotNone(self.player)
        self.assertEqual(self.player.simulations, 10)

    def test_get_move(self):
        move = self.player.get_move(self.game, 'black')
        self.assertIsNotNone(move)
        self.assertIn(move, self.game.get_valid_moves('black'))

    def test_game_to_tensor(self):
        tensor = self.player._game_to_tensor(self.game)
        self.assertEqual(tensor.shape, (1, 3, self.board_size, self.board_size))

if __name__ == '__main__':
    unittest.main()
