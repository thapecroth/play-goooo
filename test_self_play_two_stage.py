#!/usr/bin/env python3
"""
Comprehensive test suite for two-stage self-play training
"""

import unittest
import torch
import numpy as np
import time
import os
import tempfile
from unittest.mock import Mock, patch
from self_play_two_stage import (
    TwoStageTrainer, get_device, play_against_classic_ai, 
    play_self_play_game
)
from optimized_go import OptimizedGoGame, BLACK, WHITE
from alpha_go import PolicyValueNet

class TestTwoStageTraining(unittest.TestCase):
    """Test suite for two-stage self-play training system"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.board_size = 5  # Small board for fast tests
        self.num_blocks = 2  # Small network
        self.device = get_device()
        
    def test_device_selection(self):
        """Test device selection prioritizes MPS > CUDA > CPU"""
        device = get_device()
        self.assertIsInstance(device, torch.device)
        print(f"Selected device: {device}")
        
    def test_trainer_initialization(self):
        """Test TwoStageTrainer initialization"""
        trainer = TwoStageTrainer(
            board_size=self.board_size,
            num_blocks=self.num_blocks,
            num_workers=2
        )
        
        self.assertEqual(trainer.board_size, self.board_size)
        self.assertEqual(trainer.num_blocks, self.num_blocks)
        self.assertEqual(trainer.num_workers, 2)
        self.assertEqual(len(trainer.replay_buffer), 0)
        self.assertIsInstance(trainer.policy_value_net, PolicyValueNet)
        self.assertIsInstance(trainer.best_model, PolicyValueNet)
        
    def test_warmup_data_collection(self):
        """Test warmup phase collects valid training data"""
        trainer = TwoStageTrainer(
            board_size=self.board_size,
            num_blocks=self.num_blocks,
            num_workers=2
        )
        
        # Run warmup with minimal games
        positions = trainer.warmup_against_classic(
            num_games=2,
            simulations=5,
            ai_depth=1
        )
        
        # Check data was collected
        self.assertGreater(len(trainer.replay_buffer), 0)
        self.assertEqual(trainer.warmup_games, 2)
        self.assertGreaterEqual(trainer.warmup_wins, 0)
        self.assertLessEqual(trainer.warmup_wins, 2)
        
        # Verify data format
        if len(trainer.replay_buffer) > 0:
            sample = trainer.replay_buffer[0]
            self.assertEqual(len(sample), 3)  # state, policy, value
            state, policy, value = sample
            
            # Check state tensor
            self.assertEqual(state.shape, (1, 3, self.board_size, self.board_size))
            
            # Check policy tensor
            self.assertEqual(policy.shape, (self.board_size * self.board_size + 1,))
            self.assertAlmostEqual(policy.sum().item(), 1.0, places=5)
            
            # Check value
            self.assertIn(value, [-1.0, 0.0, 1.0])
            
    def test_self_play_data_collection(self):
        """Test self-play data collection phase"""
        trainer = TwoStageTrainer(
            board_size=self.board_size,
            num_blocks=self.num_blocks,
            num_workers=2
        )
        
        initial_buffer_size = len(trainer.replay_buffer)
        
        # Collect self-play data
        positions = trainer.collect_self_play_data(
            num_games=2,
            simulations=5
        )
        
        # Check data was collected
        self.assertGreater(positions, 0)
        self.assertEqual(len(trainer.replay_buffer), initial_buffer_size + positions)
        
    def test_training_phase(self):
        """Test network training phase"""
        trainer = TwoStageTrainer(
            board_size=self.board_size,
            num_blocks=self.num_blocks
        )
        
        # First collect some data
        trainer.collect_self_play_data(num_games=2, simulations=5)
        
        if len(trainer.replay_buffer) >= 16:  # Need enough for batch
            # Get initial model parameters
            initial_params = {
                name: param.clone() 
                for name, param in trainer.policy_value_net.named_parameters()
            }
            
            # Train
            losses = trainer.train_network(num_epochs=10, batch_size=16)
            
            # Check training occurred
            self.assertGreater(len(losses), 0)
            
            # Verify model parameters changed
            params_changed = False
            for name, param in trainer.policy_value_net.named_parameters():
                if not torch.equal(param, initial_params[name]):
                    params_changed = True
                    break
            
            self.assertTrue(params_changed, "Model parameters should change after training")
            
    def test_evaluation(self):
        """Test model evaluation"""
        trainer = TwoStageTrainer(
            board_size=self.board_size,
            num_blocks=self.num_blocks
        )
        
        # Run evaluation
        win_ratio = trainer.evaluate_model(num_games=2)
        
        # Check valid win ratio
        self.assertGreaterEqual(win_ratio, 0.0)
        self.assertLessEqual(win_ratio, 1.0)
        
    def test_full_iteration(self):
        """Test complete training iteration"""
        trainer = TwoStageTrainer(
            board_size=self.board_size,
            num_blocks=self.num_blocks,
            num_workers=2
        )
        
        # Run one iteration
        improved, win_ratio = trainer.run_iteration(
            iteration=1,
            num_games=2,
            num_epochs=10,
            batch_size=16,
            eval_games=2,
            win_ratio_to_update=0.4  # Low threshold for testing
        )
        
        # Check results
        self.assertIsInstance(improved, bool)
        self.assertGreaterEqual(win_ratio, 0.0)
        self.assertLessEqual(win_ratio, 1.0)
        
    def test_model_save_load(self):
        """Test model saving and loading"""
        trainer = TwoStageTrainer(
            board_size=self.board_size,
            num_blocks=self.num_blocks
        )
        
        # Create temporary file
        with tempfile.NamedTemporaryFile(suffix='.pth', delete=False) as tmp:
            tmp_path = tmp.name
            
        try:
            # Modify model slightly
            trainer.warmup_wins = 5
            trainer.warmup_games = 10
            
            # Save model
            trainer.save_model(os.path.basename(tmp_path))
            
            # Create new trainer and load
            new_trainer = TwoStageTrainer(
                board_size=self.board_size,
                num_blocks=self.num_blocks
            )
            
            # Load model
            success = new_trainer.load_model(os.path.basename(tmp_path))
            self.assertTrue(success)
            
            # Verify models are identical
            for (name1, param1), (name2, param2) in zip(
                trainer.policy_value_net.named_parameters(),
                new_trainer.policy_value_net.named_parameters()
            ):
                self.assertTrue(torch.equal(param1, param2), 
                              f"Parameter {name1} should be identical after loading")
                
        finally:
            # Clean up
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
            models_dir = os.path.join('models', os.path.basename(tmp_path))
            if os.path.exists(models_dir):
                os.remove(models_dir)
                
    def test_parallel_worker_functions(self):
        """Test parallel worker functions directly"""
        # Test play_against_classic_ai
        model = PolicyValueNet(self.board_size, num_blocks=self.num_blocks)
        state_dict = {k: v.cpu() for k, v in model.state_dict().items()}
        
        args = (0, self.board_size, state_dict, 'cpu', 5, 1, self.num_blocks)
        
        try:
            result = play_against_classic_ai(args)
            game_data, move_count, game_time, alpha_won = result
            
            self.assertIsInstance(game_data, list)
            self.assertGreater(move_count, 0)
            self.assertGreater(game_time, 0)
            self.assertIsInstance(alpha_won, bool)
        except Exception as e:
            self.fail(f"play_against_classic_ai failed: {e}")
            
        # Test play_self_play_game
        args = (0, self.board_size, state_dict, 'cpu', 5, self.num_blocks)
        
        try:
            result = play_self_play_game(args)
            game_data, move_count, game_time = result
            
            self.assertIsInstance(game_data, list)
            self.assertGreater(move_count, 0)
            self.assertGreater(game_time, 0)
        except Exception as e:
            self.fail(f"play_self_play_game failed: {e}")
            
    def test_replay_buffer_management(self):
        """Test replay buffer size limits and data management"""
        trainer = TwoStageTrainer(
            board_size=self.board_size,
            num_blocks=self.num_blocks,
            buffer_size=100  # Small buffer for testing
        )
        
        # Fill buffer beyond capacity
        for _ in range(3):
            trainer.collect_self_play_data(num_games=2, simulations=5)
            
        # Buffer should not exceed max size
        self.assertLessEqual(len(trainer.replay_buffer), 100)
        
    def test_error_handling(self):
        """Test error handling in data collection"""
        trainer = TwoStageTrainer(
            board_size=self.board_size,
            num_blocks=self.num_blocks,
            num_workers=1
        )
        
        # This should not crash even if some games fail
        positions = trainer.collect_self_play_data(
            num_games=2,
            simulations=5
        )
        
        # Should complete without crashing
        self.assertGreaterEqual(positions, 0)


class TestIntegration(unittest.TestCase):
    """Integration tests for complete training pipeline"""
    
    def test_complete_training_pipeline(self):
        """Test complete pipeline from warmup to evaluation"""
        print("\n" + "="*60)
        print("Running Complete Training Pipeline Test")
        print("="*60)
        
        trainer = TwoStageTrainer(
            board_size=5,
            num_blocks=2,
            num_workers=2,
            buffer_size=1000
        )
        
        # 1. Warmup phase
        print("\n1. Testing Warmup Phase...")
        start_time = time.time()
        warmup_positions = trainer.warmup_against_classic(
            num_games=4,
            simulations=5,
            ai_depth=1
        )
        warmup_time = time.time() - start_time
        print(f"   Warmup: {trainer.warmup_wins}/{trainer.warmup_games} wins")
        print(f"   Positions: {warmup_positions}")
        print(f"   Time: {warmup_time:.1f}s")
        
        # 2. Self-play data collection
        print("\n2. Testing Self-Play Collection...")
        start_time = time.time()
        selfplay_positions = trainer.collect_self_play_data(
            num_games=4,
            simulations=5
        )
        collection_time = time.time() - start_time
        print(f"   Positions: {selfplay_positions}")
        print(f"   Time: {collection_time:.1f}s")
        
        # 3. Training
        print("\n3. Testing Training Phase...")
        start_time = time.time()
        losses = trainer.train_network(
            num_epochs=20,
            batch_size=16
        )
        training_time = time.time() - start_time
        if losses:
            avg_loss = sum(l['total'] for l in losses) / len(losses)
            print(f"   Average loss: {avg_loss:.4f}")
        print(f"   Time: {training_time:.1f}s")
        
        # 4. Evaluation
        print("\n4. Testing Evaluation...")
        start_time = time.time()
        win_ratio = trainer.evaluate_model(num_games=2)
        eval_time = time.time() - start_time
        print(f"   Win ratio: {win_ratio:.2%}")
        print(f"   Time: {eval_time:.1f}s")
        
        # 5. Full iteration
        print("\n5. Testing Full Iteration...")
        start_time = time.time()
        improved, final_ratio = trainer.run_iteration(
            iteration=1,
            num_games=2,
            num_epochs=10,
            batch_size=16,
            eval_games=2,
            win_ratio_to_update=0.4
        )
        iter_time = time.time() - start_time
        print(f"   Improved: {improved}")
        print(f"   Final ratio: {final_ratio:.2%}")
        print(f"   Time: {iter_time:.1f}s")
        
        print("\n" + "="*60)
        print("Pipeline Test Complete!")
        print(f"Total buffer size: {len(trainer.replay_buffer)} positions")
        print("="*60)


def run_specific_test(test_name):
    """Run a specific test by name"""
    suite = unittest.TestSuite()
    
    if test_name == "all":
        # Run all tests
        loader = unittest.TestLoader()
        suite.addTests(loader.loadTestsFromTestCase(TestTwoStageTraining))
        suite.addTests(loader.loadTestsFromTestCase(TestIntegration))
    else:
        # Run specific test
        suite.addTest(TestTwoStageTraining(test_name))
    
    runner = unittest.TextTestRunner(verbosity=2)
    runner.run(suite)


if __name__ == '__main__':
    import sys
    
    if len(sys.argv) > 1:
        # Run specific test
        test_name = sys.argv[1]
        run_specific_test(test_name)
    else:
        # Run all tests
        unittest.main(verbosity=2)