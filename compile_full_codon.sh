#!/bin/bash

echo "=== Compiling Full Go Game with Codon ==="
echo

# Check if Codon is installed
if ! command -v codon &> /dev/null; then
    echo "Error: Codon compiler not found!"
    echo "Please install Codon first: https://github.com/exaloop/codon"
    exit 1
fi

echo "Codon version:"
codon --version
echo

# Compile the full game implementation
echo "Compiling go_game_codon_full.py..."
if codon build -release -o go_game_codon_full_compiled go_game_codon_full.py; then
    echo "✓ Successfully compiled to go_game_codon_full_compiled"
    chmod +x go_game_codon_full_compiled
    
    # Test the compiled binary
    echo
    echo "Testing compiled binary..."
    echo '{"board_size": 9}' | ./go_game_codon_full_compiled reset
    if [ $? -eq 0 ]; then
        echo "✓ Binary test passed!"
    else
        echo "✗ Binary test failed!"
    fi
else
    echo "✗ Failed to compile go_game_codon_full.py"
    exit 1
fi

echo
echo "=== Compilation Complete ==="
echo "You can now use the compiled binary: ./go_game_codon_full_compiled" 