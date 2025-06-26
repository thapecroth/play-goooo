#!/usr/bin/env python3
"""
Script to run all tests with pytest.
"""

import subprocess
import sys

def run_tests():
    """Run the test suite."""
    print("Running Go game test suite with pytest...\n")
    
    # Test commands
    commands = [
        # Run all tests
        ["pytest", "-v", "--tb=short"],
        
        # Run specific test categories
        ["pytest", "-v", "-m", "unit", "--tb=short"],
        ["pytest", "-v", "-m", "integration", "--tb=short"],
        
        # Run with coverage if pytest-cov is installed
        ["pytest", "--cov=.", "--cov-report=term-missing", "--cov-config=.coveragerc"],
    ]
    
    # Try to run tests
    try:
        # First, just run all tests
        print("=" * 60)
        print("Running all tests...")
        print("=" * 60)
        result = subprocess.run(["pytest", "-v", "--tb=short"], check=False)
        
        if result.returncode != 0:
            print("\nSome tests failed. Run with -v for more details.")
            sys.exit(1)
        else:
            print("\nAll tests passed!")
            
    except FileNotFoundError:
        print("Error: pytest not found. Please install it with: pip install pytest")
        sys.exit(1)
    
    # Optionally run slow tests
    response = input("\nRun slow tests? (y/n): ")
    if response.lower() == 'y':
        print("\nRunning slow tests...")
        subprocess.run(["pytest", "-v", "-m", "slow", "--tb=short"])
    
    # Optionally run performance benchmarks
    response = input("\nRun performance benchmarks? (y/n): ")
    if response.lower() == 'y':
        print("\nRunning benchmarks...")
        subprocess.run(["pytest", "-v", "tests/test_performance.py", "--tb=short"])

if __name__ == "__main__":
    run_tests()