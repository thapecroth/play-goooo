# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Development Workflow

**IMPORTANT**: Always create a git commit after making code changes. This helps track progress and allows for easy rollback if needed.

### Commit Guidelines
- Create a commit after each logical change or feature implementation
- Use descriptive commit messages that explain what was changed and why
- Include the context of the change (e.g., "Add Q-value visualization for DQN", "Optimize training performance")
- Always commit before testing new features or major changes

### When to Commit
**Always create a commit after:**
- Adding new features (UI components, server endpoints, AI functionality)
- Fixing bugs or issues
- Optimizing performance or refactoring code
- Updating configuration or dependencies
- Adding or modifying documentation
- Making any code changes that could affect functionality

### Commit Process
1. Check current status: `git status`
2. Review changes: `git diff`
3. Add files: `git add .` (or specific files)
4. Create commit with descriptive message
5. Verify commit was created: `git log --oneline -5`

### Example Commit Messages
- "Add Q-value visualization feature for DQN models"
- "Optimize training performance with parallel episode generation"
- "Fix model loading issue in Python server"
- "Update UI to support AI type switching"
- "Improve error handling in DQN inference"

## Project Overview

This is a web-based Go game implementation with both traditional rule-based AI and deep learning components. The project supports dual backend architectures:

1. **Node.js backend** (`server.js`) - Express server with Socket.IO for real-time gameplay, using traditional minimax-style AI (`game/GoAI.js`)
2. **Python backend** (`server.py`) - FastAPI server with advanced Deep Q-Network (DQN) AI implementation

## Development Commands

### Node.js Backend (Default)
```bash
npm start              # Production server
npm run dev            # Development with nodemon
```

### Python Backend (AI Training)
**IMPORTANT**: Always activate the virtual environment first:
```bash
pyenv activate play-gooo        # Activate virtual environment (required)
npm run start:python           # Basic Python server
npm run server:python          # With pyenv environment activation

# Train DQN model with default optimized settings (board size 9, depth 6)
python train_dqn.py

# Train with custom parameters (optimized)
python train_dqn.py --board-size 9 --depth 4 --episodes 10000 --batch-size 64
python train_dqn.py --board-size 13 --depth 8 --episodes 50000 --episodes-per-batch 8
python train_dqn.py --depth 12 --learning-rate 0.0001 --batch-size 128 --num-workers 8

# Performance optimization options
python train_dqn.py --episodes-per-batch 8 --batch-size 128 --num-workers 8  # Maximum speed
python train_dqn.py --no-mixed-precision  # Disable mixed precision if issues occur
python train_dqn.py --buffer-capacity 100000  # Larger replay buffer for better learning
```

### Dependencies
```bash
npm install                     # Install Node.js dependencies
pyenv activate play-gooo        # Activate virtual environment first
pip install -r requirements.txt # Install Python dependencies
```

## Architecture

### Core Game Engine
- **GoGame.js**: Main game logic with move validation, capture detection, ko rule, and game state management
- **GoAI.js**: Traditional heuristic-based AI using position evaluation with weights for territory, captures, liberties, and influence

### Deep Learning Components
- **server.py**: Contains `GoDQN` class - deep convolutional neural network with residual blocks for Go position evaluation
- **train_dqn.py**: DQN training pipeline with experience replay buffer and epsilon-greedy exploration

### Client-Server Communication
- **Socket.IO events**: `newGame`, `makeMove`, `pass`, `resign`, `setAiDepth`
- Real-time game state synchronization between client and server
- AI move delays implemented for better user experience

### Frontend
- **public/**: Vanilla JavaScript client with canvas-based Go board rendering
- Dynamic board size support (9x9, 13x13, 19x19)
- Real-time score tracking and game status updates

## Key Implementation Details

- Both backends maintain game state in memory using Maps/dictionaries keyed by socket ID
- Go rules implementation includes: liberties calculation, stone capture, ko detection, territory scoring
- DQN uses 3-channel input (black stones, white stones, empty intersections) and outputs Q-values for all possible moves
- Training uses experience replay with prioritized sampling and target network updates