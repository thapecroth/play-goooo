#!/usr/bin/env python3
"""
Test Summary for Two-Stage Self-Play Training
"""

import subprocess
import sys

def run_test(test_name, description):
    """Run a specific test and return status"""
    print(f"\n{'='*60}")
    print(f"Testing: {description}")
    print(f"{'='*60}")
    
    cmd = [sys.executable, "test_self_play_two_stage.py", test_name]
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    # Check if test passed
    if "OK" in result.stderr:
        print("✅ PASSED")
        return True
    else:
        print("❌ FAILED")
        print(result.stderr)
        return False

def main():
    """Run key tests and show summary"""
    print("Two-Stage Self-Play Training Test Summary")
    print("="*60)
    
    tests = [
        ("test_trainer_initialization", "Trainer Initialization"),
        ("test_warmup_data_collection", "Warmup vs Classic AI"),
        ("test_self_play_data_collection", "Parallel Data Collection"),
        ("test_training_phase", "Neural Network Training"),
        ("test_evaluation", "Model Evaluation"),
        ("test_full_iteration", "Complete Training Iteration"),
    ]
    
    results = []
    for test_name, description in tests:
        passed = run_test(test_name, description)
        results.append((description, passed))
    
    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    
    passed_count = sum(1 for _, passed in results if passed)
    total_count = len(results)
    
    for description, passed in results:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{description:.<45} {status}")
    
    print(f"\nTotal: {passed_count}/{total_count} tests passed")
    
    if passed_count == total_count:
        print("\n🎉 All tests passed! The two-stage training system is working correctly.")
    else:
        print(f"\n⚠️  {total_count - passed_count} tests failed. Please check the errors above.")

if __name__ == "__main__":
    main()