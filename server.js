const express = require('express');
const http = require('http');
const socketIO = require('socket.io');
const path = require('path');
const fs = require('fs');
const GoGame = require('./game/GoGame');
const GoAI = require('./game/GoAI');

const app = express();
const server = http.createServer(app);
const io = socketIO(server);

app.use(express.static('public'));

app.get('/', (req, res) => {
    res.sendFile(path.join(__dirname, 'public', 'index.html'));
});

const games = new Map();

function getAvailableModels(callback) {
    const modelsDir = path.join(__dirname, 'models');
    
    // Check if models directory exists
    if (!fs.existsSync(modelsDir)) {
        callback([]);
        return;
    }
    
    fs.readdir(modelsDir, (err, files) => {
        if (err) {
            console.error('Error reading models directory:', err);
            callback([]);
            return;
        }
        
        const models = files
            .filter(file => file.endsWith('.pth'))
            .map(file => {
                // Parse model info from filename
                // Format: dqn_go_board{size}_depth{depth}_episode_{episode}.pth
                const match = file.match(/dqn_go_board(\d+)_depth(\d+)_(?:episode_(\d+)|final)\.pth/);
                if (match) {
                    return {
                        name: file,
                        boardSize: parseInt(match[1]),
                        depth: parseInt(match[2]),
                        episodes: match[3] ? parseInt(match[3]) : 'final'
                    };
                }
                
                // Fallback for other filename formats
                return {
                    name: file,
                    boardSize: 9,
                    depth: 6,
                    episodes: 'unknown'
                };
            });
        
        callback(models);
    });
}

function makeAiMove(gameData, socket) {
    const { game, ai, aiType, currentModel } = gameData;
    
    if (aiType === 'dqn' && currentModel) {
        // For DQN, we'll use a simple fallback to classic AI for now
        // In a full implementation, this would interface with Python DQN
        console.log(`Using DQN model ${currentModel} (fallback to classic AI)`);
        const aiMove = ai.getBestMove('white');
        if (aiMove) {
            game.makeMove(aiMove.x, aiMove.y, 'white');
        } else {
            game.pass('white');
        }
    } else {
        // Classic AI
        const aiMove = ai.getBestMove('white');
        if (aiMove) {
            game.makeMove(aiMove.x, aiMove.y, 'white');
        } else {
            game.pass('white');
        }
    }
    
    socket.emit('gameState', game.getState());
}

io.on('connection', (socket) => {
    console.log('New client connected');
    
    socket.on('newGame', (boardSize) => {
        const game = new GoGame(boardSize || 9);
        const ai = new GoAI(game);
        games.set(socket.id, { 
            game, 
            ai, 
            aiType: 'classic',
            classicAlgorithm: 'minimax',
            currentModel: null,
            mctsParams: {
                simulations: 1000,
                exploration: 1.414,
                timeLimit: 5
            }
        });
        socket.emit('gameState', game.getState());
    });
    
    socket.on('makeMove', (data) => {
        const gameData = games.get(socket.id);
        if (!gameData) {
            socket.emit('error', 'No game found');
            return;
        }
        
        const { game, ai } = gameData;
        const { x, y } = data;
        
        if (game.makeMove(x, y, 'black')) {
            socket.emit('gameState', game.getState());
            
            if (!game.gameOver) {
                setTimeout(() => {
                    makeAiMove(gameData, socket);
                }, 800);
            }
        } else {
            socket.emit('invalidMove', { x, y });
        }
    });
    
    socket.on('pass', () => {
        const gameData = games.get(socket.id);
        if (!gameData) return;
        
        const { game, ai } = gameData;
        game.pass('black');
        socket.emit('gameState', game.getState());
        
        if (!game.gameOver) {
            setTimeout(() => {
                makeAiMove(gameData, socket);
            }, 800);
        }
    });
    
    socket.on('setAiDepth', (depth) => {
        const gameData = games.get(socket.id);
        if (!gameData) return;
        
        const { ai } = gameData;
        ai.maxDepth = Math.max(1, Math.min(5, depth));
    });
    
    socket.on('setClassicAlgorithm', (algorithm) => {
        const gameData = games.get(socket.id);
        if (!gameData) return;
        
        // Store the algorithm preference - in full implementation this would
        // switch between minimax and MCTS algorithms in the AI
        gameData.classicAlgorithm = algorithm;
        console.log(`Classic algorithm set to ${algorithm} for client ${socket.id}`);
    });
    
    socket.on('setMctsParams', (params) => {
        const gameData = games.get(socket.id);
        if (!gameData) return;
        
        // Store MCTS parameters - in full implementation this would
        // update the MCTS algorithm parameters
        if (!gameData.mctsParams) {
            gameData.mctsParams = {
                simulations: 1000,
                exploration: 1.414,
                timeLimit: 5
            };
        }
        
        if (params.simulations !== undefined) {
            gameData.mctsParams.simulations = Math.max(100, Math.min(5000, params.simulations));
        }
        if (params.exploration !== undefined) {
            gameData.mctsParams.exploration = Math.max(0.5, Math.min(3.0, params.exploration));
        }
        if (params.timeLimit !== undefined) {
            gameData.mctsParams.timeLimit = Math.max(1, Math.min(30, params.timeLimit));
        }
        
        console.log(`MCTS parameters updated for client ${socket.id}:`, gameData.mctsParams);
    });
    
    socket.on('setAiType', (aiType) => {
        const gameData = games.get(socket.id);
        if (!gameData) return;
        
        gameData.aiType = aiType;
        console.log(`AI type set to ${aiType} for client ${socket.id}`);
    });
    
    socket.on('getAvailableModels', () => {
        getAvailableModels((models) => {
            socket.emit('availableModels', models);
        });
    });
    
    socket.on('setDqnModel', (modelName) => {
        const gameData = games.get(socket.id);
        if (!gameData) return;
        
        gameData.currentModel = modelName;
        console.log(`DQN model set to ${modelName} for client ${socket.id}`);
        
        // For Node.js server, we'll just store the model name
        // Actual DQN inference would require Python integration
        socket.emit('modelLoaded', {
            name: modelName,
            boardSize: 9, // Could parse from filename
            depth: 6 // Could parse from filename
        });
    });
    
    socket.on('getQValues', (data) => {
        const gameData = games.get(socket.id);
        if (!gameData) return;
        
        // Node.js server doesn't support DQN inference
        // Return error to prompt user to use Python server
        socket.emit('qValuesError', 'Q-values visualization requires Python server. Please use "npm run start:python" for DQN features.');
    });
    
    socket.on('resign', () => {
        const gameData = games.get(socket.id);
        if (!gameData) return;
        
        const { game } = gameData;
        game.resign('black');
        socket.emit('gameState', game.getState());
    });
    
    socket.on('disconnect', () => {
        console.log('Client disconnected');
        games.delete(socket.id);
    });
});

const PORT = process.env.PORT || 3000;
server.listen(PORT, () => {
    console.log(`Server running on port ${PORT}`);
});