# Go Game

This is a web-based Go game with AI opponents and online multiplayer support.

🎮 **Play Now**: https://go-game-multiplayer.pages.dev/

> Note: The deployed version currently supports single-player vs AI only. For multiplayer functionality, you'll need to run your own backend server (see [Multiplayer Setup](./MULTIPLAYER_README.md)).

## Development Commands

### Node.js Backend (Default)
```bash
npm start              # Production server (single-player)
npm run dev            # Development with nodemon (single-player)
npm run start:multiplayer   # Production server (multiplayer)
npm run dev:multiplayer     # Development with nodemon (multiplayer)
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

# Train AlphaGo model
python self_play.py

# Train AlphaGo model with custom parameters
python self_play.py --board-size 13 --num-iterations 50
python self_play.py --learning-rate 1e-4 --buffer-size 20000
```

### Dependencies
```bash
npm install                     # Install Node.js dependencies
pyenv activate play-gooo        # Activate virtual environment first
pip install -r requirements.txt # Install Python dependencies
```
