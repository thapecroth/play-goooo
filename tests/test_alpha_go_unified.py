"""
Unified pytest tests for AlphaGo implementation.
Combines tests from various AlphaGo test files.
"""

import pytest
import torch
import numpy as np
import time
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from alpha_go import AlphaGoPlayer, PolicyValueNet, MCTSNode
from optimized_go import OptimizedGoGame, BLACK, WHITE, EMPTY


class TestPolicyValueNet:
    """Test the neural network component of AlphaGo."""
    
    @pytest.fixture
    def network(self, device):
        """Create a PolicyValueNet instance."""
        return PolicyValueNet(board_size=9, device=device)
    
    @pytest.mark.unit
    @pytest.mark.gpu
    def test_network_output_shapes(self, network, device):
        """Test that network outputs have correct shapes."""
        batch_size = 4
        board_size = 9
        
        # Create dummy input
        dummy_input = torch.randn(batch_size, 3, board_size, board_size).to(device)
        
        # Get network output
        policy, value = network(dummy_input)
        
        # Check shapes
        assert policy.shape == (batch_size, board_size * board_size)
        assert value.shape == (batch_size, 1)
    
    @pytest.mark.unit
    @pytest.mark.gpu
    def test_policy_probabilities(self, network, device):
        """Test that policy outputs are valid probabilities."""
        dummy_input = torch.randn(1, 3, 9, 9).to(device)
        
        policy, _ = network(dummy_input)
        
        # Check that policy sums to 1
        assert torch.allclose(policy.sum(dim=1), torch.ones(1).to(device), atol=1e-6)
        
        # Check that all probabilities are non-negative
        assert (policy >= 0).all()
    
    @pytest.mark.unit
    @pytest.mark.gpu
    def test_value_range(self, network, device):
        """Test that value outputs are in [-1, 1] range."""
        dummy_input = torch.randn(10, 3, 9, 9).to(device)
        
        _, value = network(dummy_input)
        
        # Check value range (tanh activation)
        assert (value >= -1).all()
        assert (value <= 1).all()


class TestMCTSNode:
    """Test Monte Carlo Tree Search node functionality."""
    
    @pytest.mark.unit
    def test_node_creation(self):
        """Test MCTS node initialization."""
        node = MCTSNode(parent=None, prior_prob=0.5, move=(4, 4))
        
        assert node.parent is None
        assert node.prior_prob == 0.5
        assert node.move == (4, 4)
        assert node.visits == 0
        assert node.value_sum == 0
        assert len(node.children) == 0
        assert not node.is_expanded
    
    @pytest.mark.unit
    def test_node_expansion(self):
        """Test MCTS node expansion."""
        node = MCTSNode(parent=None, prior_prob=1.0, move=None)
        
        # Create child nodes
        priors = [(0.3, (0, 0)), (0.5, (1, 1)), (0.2, (2, 2))]
        
        node.expand(priors)
        
        assert node.is_expanded
        assert len(node.children) == 3
        
        # Check children are properly created
        for (prior, move), child in zip(priors, node.children.values()):
            assert child.parent == node
            assert child.prior_prob == prior
            assert child.move == move
    
    @pytest.mark.unit
    def test_node_selection(self):
        """Test UCB-based node selection."""
        root = MCTSNode(parent=None, prior_prob=1.0, move=None)
        
        # Expand with some children
        priors = [(0.5, (0, 0)), (0.3, (1, 1)), (0.2, (2, 2))]
        root.expand(priors)
        
        # Update some children with visits
        children = list(root.children.values())
        children[0].update(0.5)  # Good value, should be selected more
        children[1].update(-0.5)  # Bad value
        children[2].update(0.0)   # Neutral
        
        # Select best child multiple times
        selections = []
        for _ in range(10):
            best_child = root.select(c_puct=1.0)
            selections.append(best_child.move)
            best_child.update(0.0)  # Update to change selection
        
        # First child should be selected most often initially
        assert selections[0] == (0, 0)
    
    @pytest.mark.unit
    def test_node_update(self):
        """Test node value updates and backpropagation."""
        root = MCTSNode(parent=None, prior_prob=1.0, move=None)
        child = MCTSNode(parent=root, prior_prob=0.5, move=(0, 0))
        root.children[(0, 0)] = child
        
        # Update child node
        child.update(0.8)
        
        assert child.visits == 1
        assert child.value_sum == 0.8
        assert child.get_value() == 0.8
        
        # Check backpropagation to parent
        assert root.visits == 1
        assert root.value_sum == -0.8  # Negated for opponent


class TestAlphaGoPlayer:
    """Test the complete AlphaGo player."""
    
    @pytest.fixture
    def player(self, device):
        """Create an AlphaGoPlayer instance."""
        return AlphaGoPlayer(board_size=9, device=device, n_simulations=50)
    
    @pytest.fixture
    def game(self):
        """Create a game instance."""
        return OptimizedGoGame(9)
    
    @pytest.mark.unit
    @pytest.mark.gpu
    def test_player_initialization(self, player):
        """Test player initializes correctly."""
        assert player.board_size == 9
        assert player.n_simulations == 50
        assert player.network is not None
        assert player.device is not None
    
    @pytest.mark.unit
    @pytest.mark.gpu
    def test_board_to_tensor_conversion(self, player, game):
        """Test conversion of board state to tensor."""
        # Add some stones
        game.make_move(4, 4, BLACK)
        game.make_move(4, 5, WHITE)
        
        tensor = player._board_to_tensor(game)
        
        # Check shape
        assert tensor.shape == (1, 3, 9, 9)
        
        # Check content
        assert tensor[0, 0, 4, 4] == 1  # Black stone
        assert tensor[0, 1, 4, 5] == 1  # White stone
        assert tensor[0, 2, 0, 0] == 1  # Empty position
    
    @pytest.mark.integration
    @pytest.mark.gpu
    @pytest.mark.timeout(30)
    def test_move_generation(self, player, game):
        """Test that player can generate valid moves."""
        move = player.get_move(game)
        
        assert move is not None
        assert isinstance(move, tuple)
        assert len(move) == 2
        assert 0 <= move[0] < 9
        assert 0 <= move[1] < 9
        
        # Verify move is legal
        assert game.make_move(move[0], move[1], BLACK)
    
    @pytest.mark.integration
    @pytest.mark.gpu
    @pytest.mark.slow
    def test_self_play_game(self, player, game):
        """Test playing a complete self-play game."""
        moves_played = 0
        max_moves = 200
        
        while not game.game_over and moves_played < max_moves:
            move = player.get_move(game)
            
            if move is None:
                game.pass_turn()
            else:
                success = game.make_move(move[0], move[1], game.current_player)
                assert success, f"Invalid move generated: {move}"
            
            moves_played += 1
        
        assert moves_played > 0
        assert moves_played < max_moves  # Game should end naturally
    
    @pytest.mark.unit
    @pytest.mark.gpu
    def test_no_valid_moves_handling(self, player):
        """Test behavior when no valid moves are available."""
        # Create a game where no moves are valid
        game = OptimizedGoGame(3)
        
        # Fill the board except one spot
        for i in range(3):
            for j in range(3):
                if (i, j) != (1, 1):
                    game.board[i, j] = BLACK if (i + j) % 2 == 0 else WHITE
        
        # Make the last spot a suicide move
        game.board[0, 1] = WHITE
        game.board[1, 0] = WHITE
        game.board[1, 2] = WHITE
        game.board[2, 1] = WHITE
        game.current_player = BLACK
        
        # Player should return None (pass)
        move = player.get_move(game)
        assert move is None
    
    @pytest.mark.unit
    @pytest.mark.gpu
    def test_game_over_handling(self, player):
        """Test behavior when game is already over."""
        game = OptimizedGoGame(9)
        game.game_over = True
        
        move = player.get_move(game)
        assert move is None


class TestMCTS:
    """Test Monte Carlo Tree Search functionality."""
    
    @pytest.mark.integration
    @pytest.mark.gpu
    def test_mcts_convergence(self, device):
        """Test that MCTS converges to good moves."""
        player = AlphaGoPlayer(board_size=9, device=device, n_simulations=100)
        game = OptimizedGoGame(9)
        
        # Create a position where center is clearly best
        # (empty board, center should be preferred)
        move = player.get_move(game)
        
        # Center region (3-5, 3-5) should be preferred
        assert 2 <= move[0] <= 6
        assert 2 <= move[1] <= 6
    
    @pytest.mark.unit
    @pytest.mark.gpu
    def test_mcts_with_temperature(self, device):
        """Test MCTS with different temperature settings."""
        player = AlphaGoPlayer(board_size=9, device=device, n_simulations=50)
        game = OptimizedGoGame(9)
        
        # Test with temperature = 0 (deterministic)
        moves_t0 = []
        for _ in range(3):
            move = player.get_move(game, temperature=0)
            moves_t0.append(move)
        
        # All moves should be the same with temperature=0
        assert all(m == moves_t0[0] for m in moves_t0)
        
        # Test with temperature = 1 (stochastic)
        moves_t1 = []
        for _ in range(10):
            move = player.get_move(game, temperature=1)
            moves_t1.append(move)
        
        # Should have some variety with temperature=1
        unique_moves = len(set(moves_t1))
        assert unique_moves > 1


class TestTraining:
    """Test training-related functionality."""
    
    @pytest.mark.unit
    @pytest.mark.gpu
    def test_network_save_load(self, device, tmp_path):
        """Test saving and loading network weights."""
        # Create and save network
        net1 = PolicyValueNet(board_size=9, device=device)
        save_path = tmp_path / "test_model.pth"
        torch.save(net1.state_dict(), save_path)
        
        # Load into new network
        net2 = PolicyValueNet(board_size=9, device=device)
        net2.load_state_dict(torch.load(save_path))
        
        # Compare outputs
        dummy_input = torch.randn(1, 3, 9, 9).to(device)
        
        net1.eval()
        net2.eval()
        
        with torch.no_grad():
            policy1, value1 = net1(dummy_input)
            policy2, value2 = net2(dummy_input)
        
        assert torch.allclose(policy1, policy2, atol=1e-6)
        assert torch.allclose(value1, value2, atol=1e-6)
    
    @pytest.mark.integration
    @pytest.mark.gpu
    @pytest.mark.slow
    def test_training_data_generation(self, device):
        """Test generation of training data from self-play."""
        player = AlphaGoPlayer(board_size=9, device=device, n_simulations=50)
        game = OptimizedGoGame(9)
        
        states = []
        policies = []
        
        # Play a few moves and collect data
        for _ in range(5):
            if not game.game_over:
                # Get move probabilities
                move_probs = player.get_action_probs(game, temperature=1)
                
                # Store state
                state = player._board_to_tensor(game).cpu().numpy()
                states.append(state)
                policies.append(move_probs)
                
                # Make move
                move = player.get_move(game, temperature=1)
                if move:
                    game.make_move(move[0], move[1], game.current_player)
        
        assert len(states) == 5
        assert len(policies) == 5
        assert all(s.shape == (1, 3, 9, 9) for s in states)
        assert all(len(p) == 81 for p in policies)


@pytest.mark.parametrize("board_size", [9, 13])
class TestDifferentBoardSizes:
    """Test AlphaGo with different board sizes."""
    
    @pytest.mark.unit
    @pytest.mark.gpu
    def test_initialization_with_size(self, board_size, device):
        """Test player initialization with different board sizes."""
        player = AlphaGoPlayer(board_size=board_size, device=device)
        assert player.board_size == board_size
        
        # Test network output shapes
        dummy_input = torch.randn(1, 3, board_size, board_size).to(device)
        policy, value = player.network(dummy_input)
        
        assert policy.shape == (1, board_size * board_size)
        assert value.shape == (1, 1)
    
    @pytest.mark.integration
    @pytest.mark.gpu
    def test_move_generation_with_size(self, board_size, device):
        """Test move generation with different board sizes."""
        player = AlphaGoPlayer(board_size=board_size, device=device, n_simulations=30)
        game = OptimizedGoGame(board_size)
        
        move = player.get_move(game)
        
        assert move is not None
        assert 0 <= move[0] < board_size
        assert 0 <= move[1] < board_size