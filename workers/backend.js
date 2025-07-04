// Cloudflare Worker backend for Go game with WebSocket support
// Uses Durable Objects for stateful game sessions

import { GoGame } from './game/GoGame.js';
import { GoAIOptimized } from './game/GoAIOptimized.js';

export default {
  async fetch(request, env) {
    const url = new URL(request.url);
    
    // Handle WebSocket upgrade
    if (request.headers.get('Upgrade') === 'websocket') {
      // Create or get a Durable Object instance for game coordination
      const id = env.GAME_COORDINATOR.idFromName('main');
      const coordinator = env.GAME_COORDINATOR.get(id);
      return coordinator.fetch(request);
    }
    
    // Handle regular HTTP requests
    if (url.pathname === '/health') {
      return new Response('OK', { status: 200 });
    }
    
    return new Response('Go Game WebSocket Server', {
      headers: {
        'Content-Type': 'text/plain',
      },
    });
  },
};

// Durable Object for game coordination
export class GameCoordinator {
  constructor(state, env) {
    this.state = state;
    this.env = env;
    this.rooms = new Map();
    this.playerToRoom = new Map();
    this.sockets = new Map();
  }
  
  async fetch(request) {
    if (request.headers.get('Upgrade') !== 'websocket') {
      return new Response('Expected WebSocket', { status: 400 });
    }
    
    const [client, server] = Object.values(new WebSocketPair());
    server.accept();
    
    const socketId = crypto.randomUUID();
    this.sockets.set(socketId, server);
    
    // Handle socket events
    server.addEventListener('message', async (event) => {
      try {
        const data = JSON.parse(event.data);
        await this.handleMessage(socketId, data);
      } catch (error) {
        console.error('Error handling message:', error);
        server.send(JSON.stringify({ type: 'error', message: error.message }));
      }
    });
    
    server.addEventListener('close', () => {
      this.handleDisconnect(socketId);
    });
    
    server.addEventListener('error', (error) => {
      console.error('WebSocket error:', error);
      this.handleDisconnect(socketId);
    });
    
    return new Response(null, { status: 101, webSocket: client });
  }
  
  generateRoomId() {
    return Math.random().toString(36).substr(2, 6).toUpperCase();
  }
  
  async handleMessage(socketId, data) {
    const socket = this.sockets.get(socketId);
    if (!socket) return;
    
    switch (data.type) {
      case 'createRoom':
        await this.handleCreateRoom(socketId, data);
        break;
      case 'joinRoom':
        await this.handleJoinRoom(socketId, data);
        break;
      case 'getRooms':
        await this.handleGetRooms(socketId);
        break;
      case 'makeMove':
        await this.handleMakeMove(socketId, data);
        break;
      case 'pass':
        await this.handlePass(socketId);
        break;
      case 'resign':
        await this.handleResign(socketId);
        break;
      case 'chatMessage':
        await this.handleChatMessage(socketId, data);
        break;
      default:
        console.log('Unknown message type:', data.type);
    }
  }
  
  async handleCreateRoom(socketId, data) {
    const socket = this.sockets.get(socketId);
    const roomId = this.generateRoomId();
    const { boardSize = 9, playerName = 'Anonymous', vsAI = false } = data;
    
    const room = {
      id: roomId,
      game: new GoGame(boardSize),
      players: {
        black: { id: socketId, name: playerName },
        white: vsAI ? { id: 'AI', name: 'AI' } : null
      },
      spectators: new Set(),
      settings: {
        boardSize: boardSize,
        isPrivate: false,
        aiEnabled: vsAI,
        ai: vsAI ? new GoAIOptimized(null) : null,
        aiColor: vsAI ? 'white' : null
      },
      created: Date.now()
    };
    
    // Set AI's game reference
    if (room.settings.ai) {
      room.settings.ai.game = room.game;
    }
    
    this.rooms.set(roomId, room);
    this.playerToRoom.set(socketId, roomId);
    
    socket.send(JSON.stringify({
      type: vsAI ? 'roomJoined' : 'roomCreated',
      roomId: roomId,
      color: 'black',
      roomState: this.getRoomState(roomId)
    }));
  }
  
  async handleJoinRoom(socketId, data) {
    const socket = this.sockets.get(socketId);
    const { roomId, playerName = 'Anonymous' } = data;
    const room = this.rooms.get(roomId);
    
    if (!room) {
      socket.send(JSON.stringify({
        type: 'joinError',
        error: 'Room not found'
      }));
      return;
    }
    
    let color = null;
    let isSpectator = false;
    
    if (!room.players.black) {
      room.players.black = { id: socketId, name: playerName };
      color = 'black';
    } else if (!room.players.white) {
      room.players.white = { id: socketId, name: playerName };
      color = 'white';
    } else {
      room.spectators.add(socketId);
      isSpectator = true;
    }
    
    this.playerToRoom.set(socketId, roomId);
    
    socket.send(JSON.stringify({
      type: 'roomJoined',
      roomId: roomId,
      color: color,
      spectator: isSpectator,
      roomState: this.getRoomState(roomId)
    }));
    
    // Notify other players
    this.broadcastToRoom(roomId, {
      type: 'playerJoined',
      color: color,
      spectator: isSpectator,
      playerName: playerName,
      roomState: this.getRoomState(roomId)
    }, socketId);
  }
  
  async handleGetRooms(socketId) {
    const socket = this.sockets.get(socketId);
    const publicRooms = [];
    
    this.rooms.forEach((room, roomId) => {
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
    
    socket.send(JSON.stringify({
      type: 'roomsList',
      rooms: publicRooms
    }));
  }
  
  async handleMakeMove(socketId, data) {
    const roomId = this.playerToRoom.get(socketId);
    if (!roomId) return;
    
    const room = this.rooms.get(roomId);
    if (!room) return;
    
    const { x, y } = data;
    const { game, players, settings } = room;
    
    // Determine player color
    let playerColor = null;
    if (players.black && players.black.id === socketId) {
      playerColor = 'black';
    } else if (players.white && players.white.id === socketId) {
      playerColor = 'white';
    }
    
    if (!playerColor || game.currentPlayer !== playerColor) {
      const socket = this.sockets.get(socketId);
      socket.send(JSON.stringify({
        type: 'invalidMove',
        x, y,
        reason: playerColor ? 'Not your turn' : 'You are a spectator'
      }));
      return;
    }
    
    // Make the move
    if (game.makeMove(x, y, playerColor)) {
      const gameState = game.getState();
      
      // Broadcast updated game state
      this.broadcastToRoom(roomId, {
        type: 'gameState',
        state: gameState
      });
      
      // If playing against AI and game not over, make AI move
      if (settings.aiEnabled && !game.gameOver && game.currentPlayer === settings.aiColor) {
        setTimeout(() => {
          const aiMove = settings.ai.getBestMove(settings.aiColor);
          if (aiMove) {
            game.makeMove(aiMove.x, aiMove.y, settings.aiColor);
          } else {
            game.pass(settings.aiColor);
          }
          this.broadcastToRoom(roomId, {
            type: 'gameState',
            state: game.getState()
          });
        }, 800);
      }
    } else {
      const socket = this.sockets.get(socketId);
      socket.send(JSON.stringify({
        type: 'invalidMove',
        x, y
      }));
    }
  }
  
  async handlePass(socketId) {
    const roomId = this.playerToRoom.get(socketId);
    if (!roomId) return;
    
    const room = this.rooms.get(roomId);
    if (!room) return;
    
    const { game, players, settings } = room;
    
    let playerColor = null;
    if (players.black && players.black.id === socketId) {
      playerColor = 'black';
    } else if (players.white && players.white.id === socketId) {
      playerColor = 'white';
    }
    
    if (!playerColor || game.currentPlayer !== playerColor) return;
    
    game.pass(playerColor);
    this.broadcastToRoom(roomId, {
      type: 'gameState',
      state: game.getState()
    });
    
    // If playing against AI and game not over, make AI move
    if (settings.aiEnabled && !game.gameOver && game.currentPlayer === settings.aiColor) {
      setTimeout(() => {
        const aiMove = settings.ai.getBestMove(settings.aiColor);
        if (aiMove) {
          game.makeMove(aiMove.x, aiMove.y, settings.aiColor);
        } else {
          game.pass(settings.aiColor);
        }
        this.broadcastToRoom(roomId, {
          type: 'gameState',
          state: game.getState()
        });
      }, 800);
    }
  }
  
  async handleResign(socketId) {
    const roomId = this.playerToRoom.get(socketId);
    if (!roomId) return;
    
    const room = this.rooms.get(roomId);
    if (!room) return;
    
    const { game, players } = room;
    
    let playerColor = null;
    if (players.black && players.black.id === socketId) {
      playerColor = 'black';
    } else if (players.white && players.white.id === socketId) {
      playerColor = 'white';
    }
    
    if (!playerColor) return;
    
    game.resign(playerColor);
    this.broadcastToRoom(roomId, {
      type: 'gameState',
      state: game.getState()
    });
  }
  
  async handleChatMessage(socketId, data) {
    const roomId = this.playerToRoom.get(socketId);
    if (!roomId) return;
    
    const room = this.rooms.get(roomId);
    if (!room) return;
    
    let playerName = 'Spectator';
    if (room.players.black && room.players.black.id === socketId) {
      playerName = room.players.black.name;
    } else if (room.players.white && room.players.white.id === socketId) {
      playerName = room.players.white.name;
    }
    
    this.broadcastToRoom(roomId, {
      type: 'chatMessage',
      playerName: playerName,
      message: data.message,
      timestamp: Date.now()
    });
  }
  
  handleDisconnect(socketId) {
    const roomId = this.playerToRoom.get(socketId);
    if (!roomId) return;
    
    const room = this.rooms.get(roomId);
    if (!room) return;
    
    // Remove player from room
    if (room.players.black && room.players.black.id === socketId) {
      room.players.black = null;
    } else if (room.players.white && room.players.white.id === socketId) {
      room.players.white = null;
    } else {
      room.spectators.delete(socketId);
    }
    
    this.playerToRoom.delete(socketId);
    this.sockets.delete(socketId);
    
    // Delete room if empty
    if (!room.players.black && !room.players.white && room.spectators.size === 0) {
      this.rooms.delete(roomId);
    } else {
      // Notify other players
      this.broadcastToRoom(roomId, {
        type: 'playerLeft',
        roomState: this.getRoomState(roomId)
      });
    }
  }
  
  getRoomState(roomId) {
    const room = this.rooms.get(roomId);
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
  
  broadcastToRoom(roomId, message, excludeSocketId = null) {
    const room = this.rooms.get(roomId);
    if (!room) return;
    
    const messageStr = JSON.stringify(message);
    
    // Send to players
    if (room.players.black && room.players.black.id !== excludeSocketId) {
      const socket = this.sockets.get(room.players.black.id);
      if (socket) socket.send(messageStr);
    }
    if (room.players.white && room.players.white.id !== excludeSocketId && room.players.white.id !== 'AI') {
      const socket = this.sockets.get(room.players.white.id);
      if (socket) socket.send(messageStr);
    }
    
    // Send to spectators
    room.spectators.forEach(spectatorId => {
      if (spectatorId !== excludeSocketId) {
        const socket = this.sockets.get(spectatorId);
        if (socket) socket.send(messageStr);
      }
    });
  }
}