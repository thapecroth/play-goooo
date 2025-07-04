const express = require('express');
const http = require('http');
const socketIO = require('socket.io');
const path = require('path');
const crypto = require('crypto');
const GoGame = require('./game/GoGame');
const GoAI = require('./game/GoAI');
const GoAIOptimized = require('./game/GoAIOptimized');

const app = express();
const server = http.createServer(app);
const io = socketIO(server);

app.use(express.static('public'));

app.get('/', (req, res) => {
    res.sendFile(path.join(__dirname, 'public', 'index.html'));
});

// Game room management
const rooms = new Map(); // roomId -> { game, players, spectators, settings }
const playerToRoom = new Map(); // socketId -> roomId

function generateRoomId() {
    return crypto.randomBytes(3).toString('hex').toUpperCase();
}

function createRoom(boardSize = 9, isPrivate = false) {
    const roomId = generateRoomId();
    const game = new GoGame(boardSize);
    
    rooms.set(roomId, {
        id: roomId,
        game: game,
        players: {
            black: null,
            white: null
        },
        spectators: new Set(),
        settings: {
            boardSize: boardSize,
            isPrivate: isPrivate,
            aiEnabled: false,
            ai: null,
            aiColor: null
        },
        created: Date.now()
    });
    
    return roomId;
}

function joinRoom(roomId, socketId, playerName) {
    const room = rooms.get(roomId);
    if (!room) return { success: false, error: 'Room not found' };
    
    // Check if player can join
    if (!room.players.black) {
        room.players.black = { id: socketId, name: playerName };
        playerToRoom.set(socketId, roomId);
        return { success: true, color: 'black', room: room };
    } else if (!room.players.white) {
        room.players.white = { id: socketId, name: playerName };
        playerToRoom.set(socketId, roomId);
        return { success: true, color: 'white', room: room };
    } else {
        // Room is full, join as spectator
        room.spectators.add(socketId);
        playerToRoom.set(socketId, roomId);
        return { success: true, spectator: true, room: room };
    }
}

function leaveRoom(socketId) {
    const roomId = playerToRoom.get(socketId);
    if (!roomId) return;
    
    const room = rooms.get(roomId);
    if (!room) return;
    
    // Remove player from room
    if (room.players.black && room.players.black.id === socketId) {
        room.players.black = null;
    } else if (room.players.white && room.players.white.id === socketId) {
        room.players.white = null;
    } else {
        room.spectators.delete(socketId);
    }
    
    playerToRoom.delete(socketId);
    
    // Delete room if empty
    if (!room.players.black && !room.players.white && room.spectators.size === 0) {
        rooms.delete(roomId);
    }
    
    return roomId;
}

function getRoomState(roomId) {
    const room = rooms.get(roomId);
    if (!room) return null;
    
    return {
        id: room.id,
        players: {
            black: room.players.black ? { name: room.players.black.name, connected: true } : null,
            white: room.players.white ? { name: room.players.white.name, connected: true } : null
        },
        spectators: room.spectators.size,
        gameState: room.game.getState(),
        settings: room.settings
    };
}

io.on('connection', (socket) => {
    console.log('New client connected:', socket.id);
    
    // Create a new multiplayer room
    socket.on('createRoom', (data) => {
        const { boardSize, playerName, vsAI } = data;
        const roomId = createRoom(boardSize || 9);
        const room = rooms.get(roomId);
        
        if (vsAI) {
            // Setup AI opponent
            room.settings.aiEnabled = true;
            room.settings.aiColor = 'white';
            room.ai = new GoAIOptimized(room.game);
            room.players.black = { id: socket.id, name: playerName };
            playerToRoom.set(socket.id, roomId);
            
            socket.join(roomId);
            socket.emit('roomJoined', {
                roomId: roomId,
                color: 'black',
                roomState: getRoomState(roomId)
            });
        } else {
            // Regular multiplayer
            const result = joinRoom(roomId, socket.id, playerName);
            socket.join(roomId);
            socket.emit('roomCreated', {
                roomId: roomId,
                color: result.color,
                roomState: getRoomState(roomId)
            });
        }
    });
    
    // Join an existing room
    socket.on('joinRoom', (data) => {
        const { roomId, playerName } = data;
        const result = joinRoom(roomId, socket.id, playerName);
        
        if (result.success) {
            socket.join(roomId);
            socket.emit('roomJoined', {
                roomId: roomId,
                color: result.color,
                spectator: result.spectator,
                roomState: getRoomState(roomId)
            });
            
            // Notify other players
            socket.to(roomId).emit('playerJoined', {
                color: result.color,
                spectator: result.spectator,
                playerName: playerName,
                roomState: getRoomState(roomId)
            });
        } else {
            socket.emit('joinError', result.error);
        }
    });
    
    // Get list of public rooms
    socket.on('getRooms', () => {
        const publicRooms = [];
        rooms.forEach((room, roomId) => {
            if (!room.settings.isPrivate) {
                publicRooms.push({
                    id: roomId,
                    players: {
                        black: room.players.black ? room.players.black.name : null,
                        white: room.players.white ? room.players.white.name : null
                    },
                    spectators: room.spectators.size,
                    boardSize: room.settings.boardSize,
                    created: room.created
                });
            }
        });
        socket.emit('roomsList', publicRooms);
    });
    
    // Make a move
    socket.on('makeMove', (data) => {
        const roomId = playerToRoom.get(socket.id);
        if (!roomId) {
            socket.emit('error', 'Not in a room');
            return;
        }
        
        const room = rooms.get(roomId);
        if (!room) {
            socket.emit('error', 'Room not found');
            return;
        }
        
        const { x, y } = data;
        const { game, players, settings } = room;
        
        // Determine player color
        let playerColor = null;
        if (players.black && players.black.id === socket.id) {
            playerColor = 'black';
        } else if (players.white && players.white.id === socket.id) {
            playerColor = 'white';
        }
        
        if (!playerColor) {
            socket.emit('error', 'You are a spectator');
            return;
        }
        
        // Check if it's player's turn
        if (game.currentPlayer !== playerColor) {
            socket.emit('invalidMove', { x, y, reason: 'Not your turn' });
            return;
        }
        
        // Make the move
        if (game.makeMove(x, y, playerColor)) {
            const gameState = game.getState();
            
            // Broadcast updated game state to all in room
            io.to(roomId).emit('gameState', gameState);
            
            // If playing against AI and game not over, make AI move
            if (settings.aiEnabled && !game.gameOver && game.currentPlayer === settings.aiColor) {
                setTimeout(() => {
                    const aiMove = room.ai.getBestMove(settings.aiColor);
                    if (aiMove) {
                        game.makeMove(aiMove.x, aiMove.y, settings.aiColor);
                    } else {
                        game.pass(settings.aiColor);
                    }
                    io.to(roomId).emit('gameState', game.getState());
                }, 800);
            }
        } else {
            socket.emit('invalidMove', { x, y });
        }
    });
    
    // Pass turn
    socket.on('pass', () => {
        const roomId = playerToRoom.get(socket.id);
        if (!roomId) return;
        
        const room = rooms.get(roomId);
        if (!room) return;
        
        const { game, players, settings } = room;
        
        // Determine player color
        let playerColor = null;
        if (players.black && players.black.id === socket.id) {
            playerColor = 'black';
        } else if (players.white && players.white.id === socket.id) {
            playerColor = 'white';
        }
        
        if (!playerColor || game.currentPlayer !== playerColor) return;
        
        game.pass(playerColor);
        io.to(roomId).emit('gameState', game.getState());
        
        // If playing against AI and game not over, make AI move
        if (settings.aiEnabled && !game.gameOver && game.currentPlayer === settings.aiColor) {
            setTimeout(() => {
                const aiMove = room.ai.getBestMove(settings.aiColor);
                if (aiMove) {
                    game.makeMove(aiMove.x, aiMove.y, settings.aiColor);
                } else {
                    game.pass(settings.aiColor);
                }
                io.to(roomId).emit('gameState', game.getState());
            }, 800);
        }
    });
    
    // Resign
    socket.on('resign', () => {
        const roomId = playerToRoom.get(socket.id);
        if (!roomId) return;
        
        const room = rooms.get(roomId);
        if (!room) return;
        
        const { game, players } = room;
        
        // Determine player color
        let playerColor = null;
        if (players.black && players.black.id === socket.id) {
            playerColor = 'black';
        } else if (players.white && players.white.id === socket.id) {
            playerColor = 'white';
        }
        
        if (!playerColor) return;
        
        game.resign(playerColor);
        io.to(roomId).emit('gameState', game.getState());
    });
    
    // Handle disconnection
    socket.on('disconnect', () => {
        console.log('Client disconnected:', socket.id);
        const roomId = leaveRoom(socket.id);
        
        if (roomId) {
            // Notify other players in the room
            io.to(roomId).emit('playerLeft', {
                roomState: getRoomState(roomId)
            });
        }
    });
    
    // Chat messages (optional)
    socket.on('chatMessage', (message) => {
        const roomId = playerToRoom.get(socket.id);
        if (!roomId) return;
        
        const room = rooms.get(roomId);
        if (!room) return;
        
        // Find player name
        let playerName = 'Spectator';
        if (room.players.black && room.players.black.id === socket.id) {
            playerName = room.players.black.name;
        } else if (room.players.white && room.players.white.id === socket.id) {
            playerName = room.players.white.name;
        }
        
        io.to(roomId).emit('chatMessage', {
            playerName: playerName,
            message: message,
            timestamp: Date.now()
        });
    });
});

const PORT = process.env.PORT || 3000;
server.listen(PORT, () => {
    console.log(`Multiplayer server running on port ${PORT}`);
});