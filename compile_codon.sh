#!/bin/bash

# Compilation script for Codon Go AI engine

echo "=== Codon Go AI Compilation Script ==="
echo

# Check if Codon is installed
if ! command -v codon &> /dev/null; then
    echo "Error: Codon compiler not found!"
    echo "Please install Codon first: https://github.com/exaloop/codon"
    echo
    echo "Installation instructions:"
    echo "  1. Download Codon from https://github.com/exaloop/codon/releases"
    echo "  2. Extract and add to PATH"
    echo "  3. Or use: /bin/bash -c \"\$(curl -fsSL https://exaloop.io/install.sh)\""
    exit 1
fi

echo "Codon version:"
codon --version
echo

# Try to compile go_ai_codon_simple.py first (simplest version)
echo "Attempting to compile go_ai_codon_simple.py..."
if codon build -release -o go_ai_codon_compiled go_ai_codon_simple.py; then
    echo "✓ Successfully compiled go_ai_codon_simple.py to go_ai_codon_compiled"
else
    echo "✗ Failed to compile go_ai_codon_simple.py"
    echo
    echo "Creating a test program to verify Codon installation..."
    
    # Create a simple test program
    cat > codon_test.py << 'TESTEOF'
def main():
    print("Hello from Codon!")
    x = 42
    y = 58
    print(f"The answer is: {x + y}")

if __name__ == "__main__":
    main()
TESTEOF
    
    if codon build -release -o codon_test codon_test.py; then
        echo "✓ Codon is working correctly"
        ./codon_test
        rm -f codon_test codon_test.py
        echo
        echo "The Go AI code may need further simplification for Codon compatibility."
        echo "Please check the Codon documentation for supported Python features."
    else
        echo "✗ Codon installation appears to be broken"
        rm -f codon_test.py
        exit 1
    fi
fi

echo
echo "=== Compilation Complete ==="
if [ -f "go_ai_codon_compiled" ]; then
    echo "Generated files:"
    echo "  - go_ai_codon_compiled (native executable)"
    echo
    echo "To run benchmarks with Codon:"
    echo "  python benchmark_go_engines.py"
    echo
    echo "Note: The -march=native flag causes segmentation faults on some systems."
    echo "Use the default compilation or try:"
    echo "  codon build -release -o go_ai_codon_compiled go_ai_codon_simple.py"
else
    echo "No compiled output was generated."
    echo "The benchmark will fall back to using the Python version."
fi
echo