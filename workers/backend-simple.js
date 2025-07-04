// Simplified Cloudflare Worker backend for Go game
// This version translates between Socket.IO client and Worker WebSocket

export default {
  async fetch(request, env) {
    const url = new URL(request.url);
    
    // Handle Socket.IO polling transport
    if (url.pathname.startsWith('/socket.io/')) {
      return handleSocketIO(request, env);
    }
    
    // Handle WebSocket upgrade
    if (request.headers.get('Upgrade') === 'websocket') {
      return handleWebSocket(request, env);
    }
    
    // Serve a simple status page
    return new Response('Go Game Backend Worker', {
      headers: { 'Content-Type': 'text/plain' }
    });
  }
};

async function handleSocketIO(request, env) {
  // Socket.IO uses polling as fallback when WebSocket fails
  // Return a response that tells the client to use WebSocket
  return new Response(JSON.stringify({
    sid: crypto.randomUUID(),
    upgrades: ['websocket'],
    pingInterval: 25000,
    pingTimeout: 20000
  }), {
    headers: {
      'Content-Type': 'application/json',
      'Access-Control-Allow-Origin': '*',
      'Access-Control-Allow-Methods': 'GET, POST, OPTIONS',
      'Access-Control-Allow-Headers': 'Content-Type'
    }
  });
}

async function handleWebSocket(request, env) {
  // Get or create the game coordinator
  const id = env.GAME_COORDINATOR.idFromName('main');
  const coordinator = env.GAME_COORDINATOR.get(id);
  return coordinator.fetch(request);
}

// Durable Object for game coordination
export class GameCoordinator {
  constructor(state, env) {
    this.state = state;
    this.env = env;
    // In-memory storage (will be reset when DO restarts)
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
    
    // Socket.IO message parser
    server.addEventListener('message', async (event) => {
      try {
        // Socket.IO messages format: "42[event,data]"
        const message = event.data;
        if (typeof message === 'string' && message.startsWith('42')) {
          const jsonStr = message.substring(2);
          const [eventName, data] = JSON.parse(jsonStr);
          await this.handleSocketIOEvent(socketId, eventName, data);
        } else if (message === '2') {
          // Socket.IO ping
          server.send('3'); // pong
        }
      } catch (error) {
        console.error('Error parsing message:', error);
      }
    });
    
    server.addEventListener('close', () => {
      this.handleDisconnect(socketId);
    });
    
    // Send Socket.IO handshake
    server.send('0{"sid":"' + socketId + '","upgrades":[],"pingInterval":25000,"pingTimeout":20000}');
    
    return new Response(null, { status: 101, webSocket: client });
  }
  
  async handleSocketIOEvent(socketId, eventName, data) {
    switch (eventName) {
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
    }
  }
  
  // Emit Socket.IO event to client
  emitToSocket(socketId, eventName, data) {
    const socket = this.sockets.get(socketId);
    if (socket) {
      const message = '42' + JSON.stringify([eventName, data]);
      socket.send(message);
    }
  }
  
  // Broadcast to all sockets in a room
  broadcastToRoom(roomId, eventName, data, excludeSocketId = null) {
    const room = this.rooms.get(roomId);
    if (!room) return;
    
    const message = '42' + JSON.stringify([eventName, data]);
    
    // Send to players
    if (room.players.black && room.players.black.id !== excludeSocketId) {
      const socket = this.sockets.get(room.players.black.id);
      if (socket) socket.send(message);
    }
    if (room.players.white && room.players.white.id !== excludeSocketId && room.players.white.id !== 'AI') {
      const socket = this.sockets.get(room.players.white.id);
      if (socket) socket.send(message);
    }
    
    // Send to spectators
    room.spectators.forEach(spectatorId => {
      if (spectatorId !== excludeSocketId) {
        const socket = this.sockets.get(spectatorId);
        if (socket) socket.send(message);
      }
    });
  }
  
  generateRoomId() {
    return Math.random().toString(36).substr(2, 6).toUpperCase();
  }
  
  async handleCreateRoom(socketId, data) {
    const roomId = this.generateRoomId();
    const { boardSize = 9, playerName = 'Anonymous', vsAI = false } = data;
    
    // Import game classes dynamically
    const { GoGame } = await import('./game/GoGame.js');
    const { GoAIOptimized } = await import('./game/GoAIOptimized.js');
    
    const game = new GoGame(boardSize);
    const room = {
      id: roomId,
      game: game,
      players: {
        black: { id: socketId, name: playerName },
        white: vsAI ? { id: 'AI', name: 'AI' } : null
      },
      spectators: new Set(),
      settings: {
        boardSize: boardSize,
        isPrivate: false,
        aiEnabled: vsAI,
        ai: vsAI ? new GoAIOptimized(game) : null,
        aiColor: vsAI ? 'white' : null
      },
      created: Date.now()
    };
    
    this.rooms.set(roomId, room);
    this.playerToRoom.set(socketId, roomId);
    
    this.emitToSocket(socketId, vsAI ? 'roomJoined' : 'roomCreated', {
      roomId: roomId,
      color: 'black',
      roomState: this.getRoomState(roomId)
    });
  }
  
  async handleJoinRoom(socketId, data) {
    const { roomId, playerName = 'Anonymous' } = data;
    const room = this.rooms.get(roomId);
    
    if (!room) {
      this.emitToSocket(socketId, 'joinError', 'Room not found');
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
    
    this.emitToSocket(socketId, 'roomJoined', {
      roomId: roomId,
      color: color,
      spectator: isSpectator,
      roomState: this.getRoomState(roomId)
    });
    
    // Notify other players
    this.broadcastToRoom(roomId, 'playerJoined', {
      color: color,
      spectator: isSpectator,
      playerName: playerName,
      roomState: this.getRoomState(roomId)
    }, socketId);
  }
  
  async handleGetRooms(socketId) {
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
    
    this.emitToSocket(socketId, 'roomsList', publicRooms);
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
      this.emitToSocket(socketId, 'invalidMove', {
        x, y,
        reason: playerColor ? 'Not your turn' : 'You are a spectator'
      });
      return;
    }
    
    // Make the move
    if (game.makeMove(x, y, playerColor)) {
      const gameState = game.getState();
      
      // Broadcast updated game state
      this.broadcastToRoom(roomId, 'gameState', gameState);
      
      // If playing against AI and game not over, make AI move
      if (settings.aiEnabled && !game.gameOver && game.currentPlayer === settings.aiColor) {
        setTimeout(async () => {
          const aiMove = settings.ai.getBestMove(settings.aiColor);
          if (aiMove) {
            game.makeMove(aiMove.x, aiMove.y, settings.aiColor);
          } else {
            game.pass(settings.aiColor);
          }
          this.broadcastToRoom(roomId, 'gameState', game.getState());
        }, 800);
      }
    } else {
      this.emitToSocket(socketId, 'invalidMove', { x, y });
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
    this.broadcastToRoom(roomId, 'gameState', game.getState());
    
    // If playing against AI and game not over, make AI move
    if (settings.aiEnabled && !game.gameOver && game.currentPlayer === settings.aiColor) {
      setTimeout(async () => {
        const aiMove = settings.ai.getBestMove(settings.aiColor);
        if (aiMove) {
          game.makeMove(aiMove.x, aiMove.y, settings.aiColor);
        } else {
          game.pass(settings.aiColor);
        }
        this.broadcastToRoom(roomId, 'gameState', game.getState());
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
    this.broadcastToRoom(roomId, 'gameState', game.getState());
  }
  
  async handleChatMessage(socketId, message) {
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
    
    this.broadcastToRoom(roomId, 'chatMessage', {
      playerName: playerName,
      message: message,
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
      this.broadcastToRoom(roomId, 'playerLeft', {
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
}