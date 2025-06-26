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

# Compile go_ai_codon.py
echo "Compiling go_ai_codon.py..."
if codon build -release -o go_ai_codon_compiled go_ai_codon.py; then
    echo "✓ Successfully compiled go_ai_codon.py to go_ai_codon_compiled"
else
    echo "✗ Failed to compile go_ai_codon.py"
    exit 1
fi

# Create a wrapper script for the compiled version
cat > go_ai_codon_wrapper.py << 'EOF'
#!/usr/bin/env python3
"""
Wrapper to use Codon-compiled Go AI in benchmarks
"""
import subprocess
import json
import sys
import os

def call_codon_ai(method, *args):
    """Call the Codon-compiled AI with method and arguments"""
    # Prepare the call
    script = f"""
import json
from go_ai_codon import GoAIOptimized

ai = GoAIOptimized({args[0] if args else 9})
result = ai.{method}(*{args[1:] if len(args) > 1 else ()})
print(json.dumps(result))
"""
    
    # Execute through Codon
    result = subprocess.run(['./go_ai_codon_compiled'], 
                           input=script, 
                           capture_output=True, 
                           text=True)
    
    if result.returncode == 0:
        return json.loads(result.stdout.strip())
    else:
        raise RuntimeError(f"Codon execution failed: {result.stderr}")

# Make it importable
class CodonGoAI:
    def __init__(self, board_size=9):
        self.board_size = board_size
        
    def get_best_move(self, board, color, captures_black, captures_white):
        return call_codon_ai('get_best_move', self.board_size, board, color, captures_black, captures_white)

if __name__ == "__main__":
    # Test the wrapper
    ai = CodonGoAI(9)
    print("Codon Go AI wrapper initialized successfully")
EOF

chmod +x go_ai_codon_wrapper.py

echo
echo "=== Compilation Complete ==="
echo "Generated files:"
echo "  - go_ai_codon_compiled (native executable)"
echo "  - go_ai_codon_wrapper.py (Python wrapper)"
echo
echo "To run benchmarks with Codon:"
echo "  python benchmark_go_engines.py"
echo
echo "To compile with specific optimizations:"
echo "  codon build -release -march=native -o go_ai_codon_compiled go_ai_codon.py"
echo