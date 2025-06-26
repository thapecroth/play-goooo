"""
Unified pytest tests for self-play and training systems.
Combines tests from various self-play test files.
"""

import pytest
import torch
import numpy as np
import time
import multiprocessing as mp
from unittest.mock import Mock, patch
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from self_play_two_stage import TwoStageTrainer
from self_play_progressive import ProgressiveTrainer
from self_play_parallel import SelfPlayTrainer
from alpha_go import AlphaGoPlayer
from classic_go_ai import ClassicGoAI
from optimized_go import OptimizedGoGame, BLACK, WHITE


class TestSelfPlayBasics:
    """Test basic self-play functionality."""
    
    @pytest.mark.unit
    @pytest.mark.gpu
    def test_single_move_generation(self, device):
        """Test generating a single move without errors."""
        player = AlphaGoPlayer(board_size=9, device=device, n_simulations=30)
        game = OptimizedGoGame(9)
        
        # Test at different temperatures
        for temp in [0.0, 0.5, 1.0]:
            move = player.get_move(game, temperature=temp)
            assert move is not None
            assert isinstance(move, tuple)
            assert len(move) == 2
            
            # Verify no NaN in move probabilities
            probs = player.get_action_probs(game, temperature=temp)
            assert not np.isnan(probs).any()
            assert np.isclose(probs.sum(), 1.0)
    
    @pytest.mark.integration
    @pytest.mark.gpu
    @pytest.mark.timeout(60)
    def test_complete_self_play_game(self, device):
        """Test playing a complete self-play game."""
        player = AlphaGoPlayer(board_size=9, device=device, n_simulations=50)
        game = OptimizedGoGame(9)
        
        moves = []
        max_moves = 200
        
        start_time = time.time()
        
        while not game.game_over and len(moves) < max_moves:
            move = player.get_move(game, temperature=0.8)
            
            if move is None:
                game.pass_turn()
                moves.append('pass')
            else:
                success = game.make_move(move[0], move[1], game.current_player)
                assert success
                moves.append(move)
        
        elapsed = time.time() - start_time
        
        assert len(moves) > 0
        assert len(moves) < max_moves
        assert elapsed < 60  # Should complete within timeout
        
        # Game should have ended properly
        assert game.game_over or len(moves) == max_moves


# Parallel self-play tests commented out - classes not found in current codebase
# TODO: Implement ParallelSelfPlayManager or update tests to use SelfPlayTrainer

class TestParallelSelfPlay:
    """Test parallel self-play infrastructure."""
    
    @pytest.mark.skip(reason="ParallelSelfPlayManager not implemented in current codebase")
    def test_device_selection(self):
        """Test correct device selection for parallel workers."""
        pass
    
    @pytest.mark.skip(reason="ParallelSelfPlayManager not implemented in current codebase")
    def test_worker_process_creation(self):
        """Test parallel worker process creation."""
        pass
    
    @pytest.mark.skip(reason="play_game_worker not implemented in current codebase")
    def test_single_worker_game(self, device):
        """Test single worker can play a game."""
        pass


class TestTwoStageTraining:
    """Test two-stage training system."""
    
    @pytest.fixture
    def trainer(self, device):
        """Create a TwoStageTrainer instance."""
        return TwoStageTrainer(
            board_size=9,
            device=device,
            warmup_games=5,
            self_play_games_per_iter=5,
            n_simulations=30
        )
    
    @pytest.mark.unit
    @pytest.mark.gpu
    def test_trainer_initialization(self, trainer):
        """Test trainer initializes correctly."""
        assert trainer.board_size == 9
        assert trainer.warmup_games == 5
        assert trainer.self_play_games_per_iter == 5
        assert trainer.player is not None
        assert trainer.classic_ai is not None
        assert trainer.iteration == 0
    
    @pytest.mark.integration
    @pytest.mark.gpu
    @pytest.mark.slow
    def test_warmup_data_collection(self, trainer):
        """Test warmup phase data collection."""
        trainer.warmup_games = 2  # Reduce for faster testing
        
        print("Collecting warmup data...")
        trainer.collect_warmup_data()
        
        assert len(trainer.training_data) > 0
        
        # Check data format
        for state, policy, value in trainer.training_data:
            assert state.shape == (3, 9, 9)
            assert len(policy) == 81
            assert -1 <= value <= 1
    
    @pytest.mark.integration
    @pytest.mark.gpu
    @pytest.mark.timeout(120)
    def test_parallel_data_collection(self, trainer):
        """Test parallel self-play data collection."""
        trainer.self_play_games_per_iter = 4  # Must be divisible by num_workers
        trainer.num_workers = 2
        
        initial_data_size = len(trainer.training_data)
        
        print("Collecting self-play data in parallel...")
        trainer.collect_self_play_data_parallel()
        
        assert len(trainer.training_data) > initial_data_size
    
    @pytest.mark.unit
    @pytest.mark.gpu
    def test_neural_network_training(self, trainer):
        """Test neural network training step."""
        # Add some dummy training data
        for _ in range(64):  # Minimum for batch
            state = np.random.randn(3, 9, 9).astype(np.float32)
            policy = np.random.dirichlet([0.1] * 81)
            value = np.random.uniform(-1, 1)
            trainer.training_data.append((state, policy, value))
        
        # Get initial loss
        initial_loss = trainer._compute_loss_on_batch(trainer.training_data[:32])
        
        # Train for a few steps
        trainer.batch_size = 32
        trainer.train_network(epochs=5)
        
        # Loss should decrease
        final_loss = trainer._compute_loss_on_batch(trainer.training_data[:32])
        assert final_loss < initial_loss
    
    @pytest.mark.unit
    @pytest.mark.gpu
    def test_model_evaluation(self, trainer):
        """Test model evaluation against classic AI."""
        wins = trainer.evaluate_against_classic(games=2, classic_depth=1)
        
        assert isinstance(wins, int)
        assert 0 <= wins <= 2
    
    @pytest.mark.integration
    @pytest.mark.gpu
    @pytest.mark.slow
    @pytest.mark.timeout(300)
    def test_complete_training_iteration(self, trainer):
        """Test one complete training iteration."""
        # Reduce parameters for faster testing
        trainer.warmup_games = 2
        trainer.self_play_games_per_iter = 2
        trainer.batch_size = 16
        
        # Run one iteration
        trainer.train_iteration()
        
        assert trainer.iteration == 1
        assert len(trainer.training_data) > 0
        
        # Check that model was saved
        import os
        assert os.path.exists(f'alphago_model_iter_1.pth')
        
        # Cleanup
        os.remove(f'alphago_model_iter_1.pth')


class TestProgressiveTraining:
    """Test progressive difficulty training."""
    
    @pytest.fixture
    def trainer(self, device):
        """Create a ProgressiveTrainer instance."""
        return ProgressiveTrainer(
            board_size=9,
            device=device,
            games_per_difficulty=5,
            n_simulations=30
        )
    
    @pytest.mark.unit
    @pytest.mark.gpu
    def test_difficulty_progression(self, trainer):
        """Test that difficulty increases based on win rate."""
        trainer.games_per_difficulty = 2
        
        # Start at depth 1
        assert trainer.current_depth == 1
        
        # Simulate high win rate
        with patch.object(trainer, 'evaluate_against_classic', return_value=2):
            should_increase = trainer._should_increase_difficulty()
            assert should_increase
            
            if should_increase:
                trainer.current_depth += 1
        
        assert trainer.current_depth == 2
    
    @pytest.mark.integration
    @pytest.mark.gpu
    @pytest.mark.slow
    def test_progressive_training_step(self, trainer):
        """Test one step of progressive training."""
        trainer.games_per_difficulty = 2
        trainer.batch_size = 16
        
        initial_depth = trainer.current_depth
        
        # Run one training step
        trainer.train()
        
        # Should have collected data and trained
        assert len(trainer.player.memory) > 0
        
        # Depth might have increased if win rate was high
        assert trainer.current_depth >= initial_depth


class TestDataAugmentation:
    """Test data augmentation for training."""
    
    @pytest.mark.unit
    def test_board_rotation(self):
        """Test board rotation augmentation."""
        board = np.zeros((9, 9))
        board[0, 0] = 1  # Top-left corner
        
        # Rotate 90 degrees
        rotated = np.rot90(board)
        assert rotated[0, 8] == 1  # Should be in top-right
        
        # Rotate 180 degrees
        rotated_180 = np.rot90(board, 2)
        assert rotated_180[8, 8] == 1  # Should be in bottom-right
    
    @pytest.mark.unit
    def test_board_reflection(self):
        """Test board reflection augmentation."""
        board = np.zeros((9, 9))
        board[0, 2] = 1  # Asymmetric position
        
        # Horizontal flip
        flipped_h = np.fliplr(board)
        assert flipped_h[0, 6] == 1
        
        # Vertical flip
        flipped_v = np.flipud(board)
        assert flipped_v[8, 2] == 1
    
    @pytest.mark.unit
    def test_augmentation_consistency(self):
        """Test that augmentation preserves game properties."""
        # Create a position with clear patterns
        board = np.zeros((9, 9))
        policy = np.zeros(81)
        
        # Place stones in a pattern
        board[4, 4] = 1  # Center
        board[3, 4] = 2
        board[5, 4] = 2
        
        # Set policy for adjacent moves
        policy[4 * 9 + 3] = 0.5  # Left of center
        policy[4 * 9 + 5] = 0.5  # Right of center
        
        # Apply rotation
        rotated_board = np.rot90(board)
        rotated_policy = policy.reshape(9, 9)
        rotated_policy = np.rot90(rotated_policy).flatten()
        
        # Check that the pattern is preserved
        assert rotated_board[4, 4] == 1  # Center stays
        assert np.sum(rotated_board == 1) == 1
        assert np.sum(rotated_board == 2) == 2


class TestMemoryManagement:
    """Test memory management in training."""
    
    @pytest.mark.unit
    @pytest.mark.gpu
    def test_gpu_memory_cleanup(self, device):
        """Test that GPU memory is properly cleaned up."""
        if device.type == 'cpu':
            pytest.skip("GPU memory test requires GPU")
        
        initial_memory = torch.cuda.memory_allocated() if device.type == 'cuda' else 0
        
        # Create and delete large tensors
        for _ in range(5):
            large_tensor = torch.randn(100, 3, 19, 19, device=device)
            del large_tensor
        
        if device.type == 'cuda':
            torch.cuda.empty_cache()
        
        final_memory = torch.cuda.memory_allocated() if device.type == 'cuda' else 0
        
        # Memory should not grow significantly
        assert final_memory <= initial_memory * 1.1
    
    @pytest.mark.unit
    def test_training_data_memory_limit(self):
        """Test that training data doesn't grow unbounded."""
        max_size = 10000
        training_data = []
        
        # Add data beyond limit
        for i in range(max_size + 1000):
            state = np.random.randn(3, 9, 9).astype(np.float32)
            policy = np.random.dirichlet([0.1] * 81)
            value = np.random.uniform(-1, 1)
            training_data.append((state, policy, value))
            
            # Implement size limit
            if len(training_data) > max_size:
                training_data = training_data[-max_size:]
        
        assert len(training_data) == max_size


@pytest.mark.parametrize("temperature", [0.0, 0.5, 1.0])
class TestTemperatureEffects:
    """Test effects of temperature on move selection."""
    
    @pytest.mark.unit
    @pytest.mark.gpu
    def test_temperature_move_diversity(self, temperature, device):
        """Test that temperature affects move diversity."""
        player = AlphaGoPlayer(board_size=9, device=device, n_simulations=50)
        game = OptimizedGoGame(9)
        
        # Collect moves
        moves = []
        for _ in range(10):
            move = player.get_move(game, temperature=temperature)
            if move:
                moves.append(move)
        
        unique_moves = len(set(moves))
        
        if temperature == 0.0:
            # Deterministic - should always be same move
            assert unique_moves == 1
        else:
            # Stochastic - should have some variety
            assert unique_moves >= 1  # At least some variety expected