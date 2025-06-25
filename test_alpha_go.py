import unittest
import torch
import numpy as np
import time
from alpha_go import PolicyValueNet, AlphaGoPlayer, AlphaGoMCTSNode
from optimized_go import OptimizedGoGame, BLACK, WHITE, EMPTY

class TestAlphaGo(unittest.TestCase):
    def setUp(self):
        self.board_size = 9
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.net = PolicyValueNet(board_size=self.board_size).to(self.device)
        self.net.eval()
        
    def test_policy_value_net_output_shapes(self):
        """Test that the network outputs correct shapes"""
        batch_size = 4
        input_tensor = torch.randn(batch_size, 3, self.board_size, self.board_size).to(self.device)
        
        policy, value = self.net(input_tensor)
        
        # Policy should be log probabilities for all board positions + pass
        expected_policy_shape = (batch_size, self.board_size * self.board_size + 1)
        self.assertEqual(policy.shape, expected_policy_shape)
        
        # Value should be a single value per batch
        expected_value_shape = (batch_size, 1)
        self.assertEqual(value.shape, expected_value_shape)
        
        # Check value is in [-1, 1] range (tanh output)
        self.assertTrue(torch.all(value >= -1.0))
        self.assertTrue(torch.all(value <= 1.0))
        
    def test_policy_value_net_valid_probabilities(self):
        """Test that policy outputs valid log probabilities"""
        input_tensor = torch.randn(1, 3, self.board_size, self.board_size).to(self.device)
        policy, _ = self.net(input_tensor)
        
        # Exponentiating log probabilities should give valid probabilities
        probs = torch.exp(policy)
        self.assertAlmostEqual(probs.sum().item(), 1.0, places=5)
        self.assertTrue(torch.all(probs >= 0))
        
    def test_alpha_go_player_get_move(self):
        """Test that AlphaGoPlayer returns valid moves"""
        player = AlphaGoPlayer(self.net, simulations=50, exploration_constant=1.0, is_self_play=True)
        game = OptimizedGoGame(self.board_size)
        
        # Test first move
        move = player.get_move(game, 'black', temperature=1.0)
        self.assertIsNotNone(move)
        
        # Make the move and test second move
        if move is not None:
            game.make_move(move[0], move[1], 'black')
        else:
            game.pass_turn('black')
            
        move2 = player.get_move(game, 'white', temperature=1.0)
        self.assertIsNotNone(move2)
        
    def test_alpha_go_player_zero_temperature(self):
        """Test deterministic play with temperature=0"""
        player = AlphaGoPlayer(self.net, simulations=100, exploration_constant=1.0, is_self_play=True)
        game = OptimizedGoGame(self.board_size)
        
        # With temperature=0, should select move with most visits
        moves = []
        for _ in range(3):
            move = player.get_move(game, 'black', temperature=0)
            moves.append(move)
            
        # All moves should be the same (deterministic)
        self.assertEqual(moves[0], moves[1])
        self.assertEqual(moves[1], moves[2])
        
    def test_mcts_node_edge_cases(self):
        """Test MCTS node behavior in edge cases"""
        game = OptimizedGoGame(self.board_size)
        root = AlphaGoMCTSNode(game)
        
        # Test with no children
        best_child = root.get_best_child(1.0)
        self.assertIsNone(best_child)
        
        # Add children with different priors
        game_copy1 = game.copy()
        child1 = AlphaGoMCTSNode(game_copy1, parent=root, move=(0, 0), color=BLACK, prior=0.8)
        root.children.append(child1)
        
        game_copy2 = game.copy()
        child2 = AlphaGoMCTSNode(game_copy2, parent=root, move=(0, 1), color=BLACK, prior=0.2)
        root.children.append(child2)
        
        # With no visits, should prefer higher prior
        best_child = root.get_best_child(1.0)
        self.assertEqual(best_child, child1)
        
        # Update visits and values
        child2.update(1.0)
        child2.update(1.0)
        child1.update(-1.0)
        
        # Now child2 has better value despite lower prior
        root.visits = 3
        best_child = root.get_best_child(0.1)  # Low exploration
        self.assertEqual(best_child, child2)
        
    def test_zero_visit_counts_handling(self):
        """Test that zero visit counts don't cause NaN errors"""
        player = AlphaGoPlayer(self.net, simulations=1, exploration_constant=1.0, is_self_play=True)
        game = OptimizedGoGame(self.board_size)
        
        # With only 1 simulation, some moves might have zero visits
        move = player.get_move(game, 'black', temperature=1.0)
        # Should not raise ValueError about NaN
        self.assertTrue(move is None or (isinstance(move, tuple) and len(move) == 2))
        
    def test_performance_benchmark(self):
        """Benchmark performance of move generation"""
        player = AlphaGoPlayer(self.net, simulations=100, exploration_constant=1.0, is_self_play=False)
        game = OptimizedGoGame(self.board_size)
        
        # Warm up
        player.get_move(game, 'black')
        
        # Benchmark
        start_time = time.time()
        num_moves = 10
        for i in range(num_moves):
            move = player.get_move(game, 'black' if i % 2 == 0 else 'white')
            if move is not None:
                game.make_move(move[0], move[1], 'black' if i % 2 == 0 else 'white')
            else:
                game.pass_turn('black' if i % 2 == 0 else 'white')
                
        elapsed = time.time() - start_time
        avg_time = elapsed / num_moves
        
        print(f"\nPerformance: {avg_time:.3f} seconds per move ({num_moves} moves in {elapsed:.3f}s)")
        self.assertLess(avg_time, 1.0)  # Should be under 1 second per move
        
    def test_game_to_tensor_correctness(self):
        """Test game state to tensor conversion"""
        player = AlphaGoPlayer(self.net, simulations=10)
        game = OptimizedGoGame(self.board_size)
        
        # Set up a specific board position
        game.make_move(0, 0, 'black')
        game.make_move(1, 1, 'white')
        game.make_move(2, 2, 'black')
        
        tensor = player._game_to_tensor(game)
        
        # Check tensor shape
        self.assertEqual(tensor.shape, (1, 3, self.board_size, self.board_size))
        
        # Check correct stone positions
        # Channel 0: current player stones (white)
        self.assertEqual(tensor[0, 0, 1, 1].item(), 1.0)  # White stone
        self.assertEqual(tensor[0, 0, 0, 0].item(), 0.0)  # Not white
        
        # Channel 1: opponent stones
        self.assertEqual(tensor[0, 1, 0, 0].item(), 1.0)  # Black stone
        self.assertEqual(tensor[0, 1, 2, 2].item(), 1.0)  # Black stone
        
        # Channel 2: current player indicator (all 1s for white)
        self.assertTrue(torch.all(tensor[0, 2, :, :] == 1.0))
        
    def test_parallel_games(self):
        """Test running multiple games in parallel for performance"""
        players = [AlphaGoPlayer(self.net, simulations=50, is_self_play=True) for _ in range(4)]
        games = [OptimizedGoGame(self.board_size) for _ in range(4)]
        
        start_time = time.time()
        moves_made = 0
        
        for _ in range(5):  # 5 moves per game
            for i, (player, game) in enumerate(zip(players, games)):
                move = player.get_move(game, 'black' if moves_made % 2 == 0 else 'white')
                if move is not None:
                    game.make_move(move[0], move[1], 'black' if moves_made % 2 == 0 else 'white')
                else:
                    game.pass_turn('black' if moves_made % 2 == 0 else 'white')
            moves_made += 1
            
        elapsed = time.time() - start_time
        print(f"\nParallel games: {elapsed:.3f}s for {len(games)} games, {moves_made} moves each")

if __name__ == '__main__':
    unittest.main()