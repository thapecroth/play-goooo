// Cloudflare Worker backend using Durable Objects with SQLite for the free tier
// This implementation uses HTTP polling to simulate real-time multiplayer

import { GoGame } from './game/GoGame.js';
import { GoAIOptimized } from './game/GoAIOptimized.js';

// Rate limiting configuration
const RATE_LIMIT = {
  MAX_REQUESTS_PER_MINUTE: 30,  // 30 requests per minute per client
  MAX_POLLS_PER_MINUTE: 20,      // 20 poll requests per minute per client
  POLL_INTERVAL_MS: 3000,        // Minimum 3 seconds between polls
};

export default {
  async fetch(request, env, ctx) {
    const url = new URL(request.url);
    
    // CORS headers
    const corsHeaders = {
      'Access-Control-Allow-Origin': '*',
      'Access-Control-Allow-Methods': 'GET, POST, OPTIONS',
      'Access-Control-Allow-Headers': 'Content-Type',
    };
    
    // Handle CORS preflight
    if (request.method === 'OPTIONS') {
      return new Response(null, { headers: corsHeaders });
    }
    
    try {
      // API endpoints
      if (url.pathname === '/api/status') {
        return new Response(JSON.stringify({
          status: 'online',
          message: 'Go Game Backend (Durable Objects with SQLite)',
          features: [
            'HTTP-based multiplayer (polling)',
            'Rate-limited to prevent quota issues',
            'Persistent game state',
            'AI opponent support'
          ],
          rateLimit: RATE_LIMIT
        }), {
          headers: { ...corsHeaders, 'Content-Type': 'application/json' }
        });
      }
      
      // Route to game coordinator
      const coordinatorId = env.GAME_COORDINATOR.idFromName('main');
      const coordinator = env.GAME_COORDINATOR.get(coordinatorId);
      
      // Forward request to Durable Object
      const response = await coordinator.fetch(request.url, {
        method: request.method,
        headers: request.headers,
        body: request.body
      });
      
      // Add CORS headers to response
      const newHeaders = new Headers(response.headers);
      Object.entries(corsHeaders).forEach(([key, value]) => {
        newHeaders.set(key, value);
      });
      
      return new Response(response.body, {
        status: response.status,
        statusText: response.statusText,
        headers: newHeaders
      });
      
    } catch (error) {
      return new Response(JSON.stringify({
        error: 'Internal server error',
        message: error.message
      }), {
        status: 500,
        headers: { ...corsHeaders, 'Content-Type': 'application/json' }
      });
    }
  }
};

// Durable Object for game coordination
export class GameCoordinator {
  constructor(state, env) {
    this.state = state;
    this.env = env;
    this.sql = state.storage.sql;
    this.initializeDatabase();
  }
  
  async initializeDatabase() {
    // Create tables if they don't exist
    await this.sql.exec(`
      CREATE TABLE IF NOT EXISTS rooms (
        id TEXT PRIMARY KEY,
        game_state TEXT NOT NULL,
        players TEXT NOT NULL,
        settings TEXT NOT NULL,
        created_at INTEGER NOT NULL,
        updated_at INTEGER NOT NULL
      );
      
      CREATE TABLE IF NOT EXISTS rate_limits (
        client_id TEXT PRIMARY KEY,
        request_count INTEGER DEFAULT 0,
        poll_count INTEGER DEFAULT 0,
        last_reset INTEGER NOT NULL
      );
      
      CREATE TABLE IF NOT EXISTS chat_messages (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        room_id TEXT NOT NULL,
        player_name TEXT NOT NULL,
        message TEXT NOT NULL,
        timestamp INTEGER NOT NULL
      );
    `);
  }
  
  async fetch(request) {
    const url = new URL(request.url);
    const path = url.pathname;
    
    try {
      // Check rate limit
      const clientId = request.headers.get('CF-Connecting-IP') || 'unknown';
      if (!await this.checkRateLimit(clientId, path)) {
        return new Response(JSON.stringify({
          error: 'Rate limit exceeded',
          message: 'Please wait before making more requests',
          retryAfter: 60
        }), {
          status: 429,
          headers: { 'Content-Type': 'application/json' }
        });
      }
      
      // Route requests
      if (path === '/api/rooms/create' && request.method === 'POST') {
        return await this.handleCreateRoom(request);
      } else if (path === '/api/rooms/join' && request.method === 'POST') {
        return await this.handleJoinRoom(request);
      } else if (path === '/api/rooms/list' && request.method === 'GET') {
        return await this.handleListRooms();
      } else if (path.startsWith('/api/room/') && path.endsWith('/state') && request.method === 'GET') {
        return await this.handleGetRoomState(request);
      } else if (path.startsWith('/api/room/') && path.endsWith('/move') && request.method === 'POST') {
        return await this.handleMakeMove(request);
      } else if (path.startsWith('/api/room/') && path.endsWith('/pass') && request.method === 'POST') {
        return await this.handlePass(request);
      } else if (path.startsWith('/api/room/') && path.endsWith('/resign') && request.method === 'POST') {
        return await this.handleResign(request);
      } else if (path.startsWith('/api/room/') && path.endsWith('/chat') && request.method === 'POST') {
        return await this.handleChat(request);
      }
      
      return new Response('Not found', { status: 404 });
      
    } catch (error) {
      console.error('Error in GameCoordinator:', error);
      return new Response(JSON.stringify({
        error: 'Internal server error',
        message: error.message
      }), {
        status: 500,
        headers: { 'Content-Type': 'application/json' }
      });
    }
  }
  
  async checkRateLimit(clientId, path) {
    const now = Date.now();
    const minute = Math.floor(now / 60000);
    
    // Get current rate limit data
    const result = await this.sql.exec(
      `SELECT request_count, poll_count, last_reset FROM rate_limits WHERE client_id = ?`,
      clientId
    ).then(r => r.toArray()[0]);
    
    if (!result || result.last_reset < minute) {
      // Reset counters
      await this.sql.exec(
        `INSERT OR REPLACE INTO rate_limits (client_id, request_count, poll_count, last_reset) VALUES (?, 1, ?, ?)`,
        clientId, path.includes('/state') ? 1 : 0, minute
      );
      return true;
    }
    
    // Check limits
    if (result.request_count >= RATE_LIMIT.MAX_REQUESTS_PER_MINUTE) {
      return false;
    }
    
    if (path.includes('/state') && result.poll_count >= RATE_LIMIT.MAX_POLLS_PER_MINUTE) {
      return false;
    }
    
    // Update counters
    await this.sql.exec(
      `UPDATE rate_limits SET request_count = request_count + 1, poll_count = poll_count + ? WHERE client_id = ?`,
      path.includes('/state') ? 1 : 0, clientId
    );
    
    return true;
  }
  
  generateRoomId() {
    return Math.random().toString(36).substring(2, 8).toUpperCase();
  }
  
  async handleCreateRoom(request) {
    const body = await request.json();
    const { playerName, boardSize = 9, vsAI = false } = body;
    
    const roomId = this.generateRoomId();
    const game = new GoGame(boardSize);
    const now = Date.now();
    
    const roomData = {
      id: roomId,
      game_state: this.serializeGameState(game),
      players: {
        black: { name: playerName, id: request.headers.get('CF-Connecting-IP') },
        white: vsAI ? { name: 'AI', id: 'ai' } : null
      },
      settings: {
        boardSize,
        vsAI,
        created: now
      },
      chatMessages: [],
      lastActivity: now
    };
    
    await this.sql.exec(
      `INSERT INTO rooms (id, game_state, players, settings, created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?)`,
      roomId, 
      JSON.stringify(roomData.game_state),
      JSON.stringify(roomData.players),
      JSON.stringify(roomData.settings),
      now,
      now
    );
    
    return new Response(JSON.stringify({
      roomId,
      color: 'black',
      roomState: this.formatRoomState(roomData)
    }), {
      headers: { 'Content-Type': 'application/json' }
    });
  }
  
  async handleJoinRoom(request) {
    const body = await request.json();
    const { roomId, playerName } = body;
    
    const room = await this.getRoom(roomId);
    if (!room) {
      return new Response(JSON.stringify({
        error: 'Room not found'
      }), {
        status: 404,
        headers: { 'Content-Type': 'application/json' }
      });
    }
    
    const players = JSON.parse(room.players);
    const clientId = request.headers.get('CF-Connecting-IP');
    
    // Check if player can join
    let color = null;
    let isSpectator = false;
    
    if (!players.white && players.black.id !== clientId) {
      players.white = { name: playerName, id: clientId };
      color = 'white';
    } else if (players.black.id === clientId) {
      color = 'black';
    } else if (players.white && players.white.id === clientId) {
      color = 'white';
    } else {
      isSpectator = true;
    }
    
    // Update room
    await this.sql.exec(
      `UPDATE rooms SET players = ?, updated_at = ? WHERE id = ?`,
      JSON.stringify(players),
      Date.now(),
      roomId
    );
    
    return new Response(JSON.stringify({
      roomId,
      color,
      isSpectator,
      roomState: this.formatRoomState({
        ...room,
        players,
        game_state: JSON.parse(room.game_state),
        settings: JSON.parse(room.settings)
      })
    }), {
      headers: { 'Content-Type': 'application/json' }
    });
  }
  
  async handleListRooms() {
    const rooms = await this.sql.exec(
      `SELECT id, players, settings FROM rooms WHERE created_at > ? ORDER BY created_at DESC LIMIT 20`,
      Date.now() - 3600000 // Last hour
    ).then(r => r.toArray());
    
    const publicRooms = rooms.map(room => {
      const players = JSON.parse(room.players);
      const settings = JSON.parse(room.settings);
      
      return {
        id: room.id,
        players: {
          black: players.black ? players.black.name : null,
          white: players.white ? players.white.name : null
        },
        boardSize: settings.boardSize,
        vsAI: settings.vsAI
      };
    }).filter(room => !room.vsAI); // Only show public multiplayer rooms
    
    return new Response(JSON.stringify(publicRooms), {
      headers: { 'Content-Type': 'application/json' }
    });
  }
  
  async handleGetRoomState(request) {
    const url = new URL(request.url);
    const roomId = url.pathname.split('/')[3];
    
    const room = await this.getRoom(roomId);
    if (!room) {
      return new Response(JSON.stringify({
        error: 'Room not found'
      }), {
        status: 404,
        headers: { 'Content-Type': 'application/json' }
      });
    }
    
    // Get recent chat messages
    const chatMessages = await this.sql.exec(
      `SELECT player_name, message, timestamp FROM chat_messages 
       WHERE room_id = ? ORDER BY timestamp DESC LIMIT 50`,
      roomId
    ).then(r => r.toArray().reverse()); // Reverse to get chronological order
    
    return new Response(JSON.stringify({
      roomState: this.formatRoomState({
        ...room,
        game_state: JSON.parse(room.game_state),
        players: JSON.parse(room.players),
        settings: JSON.parse(room.settings)
      }),
      chatMessages: chatMessages,
      pollInterval: RATE_LIMIT.POLL_INTERVAL_MS
    }), {
      headers: { 
        'Content-Type': 'application/json',
        'Cache-Control': 'no-cache'
      }
    });
  }
  
  async handleMakeMove(request) {
    const url = new URL(request.url);
    const roomId = url.pathname.split('/')[3];
    const body = await request.json();
    const { x, y } = body;
    
    const room = await this.getRoom(roomId);
    if (!room) {
      return new Response(JSON.stringify({
        error: 'Room not found'
      }), {
        status: 404,
        headers: { 'Content-Type': 'application/json' }
      });
    }
    
    const gameState = JSON.parse(room.game_state);
    const players = JSON.parse(room.players);
    const settings = JSON.parse(room.settings);
    const clientId = request.headers.get('CF-Connecting-IP');
    
    // Determine player color
    let playerColor = null;
    if (players.black && players.black.id === clientId) {
      playerColor = 'black';
    } else if (players.white && players.white.id === clientId) {
      playerColor = 'white';
    }
    
    if (!playerColor) {
      return new Response(JSON.stringify({
        error: 'You are not a player in this game'
      }), {
        status: 403,
        headers: { 'Content-Type': 'application/json' }
      });
    }
    
    // Recreate game instance and make move
    const game = this.deserializeGameState(gameState, settings.boardSize);
    
    if (game.currentPlayer !== playerColor) {
      return new Response(JSON.stringify({
        error: 'Not your turn'
      }), {
        status: 400,
        headers: { 'Content-Type': 'application/json' }
      });
    }
    
    if (!game.makeMove(x, y, playerColor)) {
      return new Response(JSON.stringify({
        error: 'Invalid move'
      }), {
        status: 400,
        headers: { 'Content-Type': 'application/json' }
      });
    }
    
    // Update room state
    const newGameState = this.serializeGameState(game);
    await this.sql.exec(
      `UPDATE rooms SET game_state = ?, updated_at = ? WHERE id = ?`,
      JSON.stringify(newGameState),
      Date.now(),
      roomId
    );
    
    // If playing against AI, make AI move
    if (settings.vsAI && !game.gameOver && game.currentPlayer === 'white') {
      setTimeout(async () => {
        const ai = new GoAIOptimized(game);
        const aiMove = ai.getBestMove('white');
        if (aiMove) {
          game.makeMove(aiMove.x, aiMove.y, 'white');
        } else {
          game.pass('white');
        }
        
        const aiGameState = this.serializeGameState(game);
        await this.sql.exec(
          `UPDATE rooms SET game_state = ?, updated_at = ? WHERE id = ?`,
          JSON.stringify(aiGameState),
          Date.now(),
          roomId
        );
      }, 800);
    }
    
    return new Response(JSON.stringify({
      success: true,
      gameState: this.formatGameState(newGameState)
    }), {
      headers: { 'Content-Type': 'application/json' }
    });
  }
  
  async handlePass(request) {
    const url = new URL(request.url);
    const roomId = url.pathname.split('/')[3];
    
    const room = await this.getRoom(roomId);
    if (!room) {
      return new Response(JSON.stringify({
        error: 'Room not found'
      }), {
        status: 404,
        headers: { 'Content-Type': 'application/json' }
      });
    }
    
    const gameState = JSON.parse(room.game_state);
    const players = JSON.parse(room.players);
    const settings = JSON.parse(room.settings);
    const clientId = request.headers.get('CF-Connecting-IP');
    
    // Determine player color
    let playerColor = null;
    if (players.black && players.black.id === clientId) {
      playerColor = 'black';
    } else if (players.white && players.white.id === clientId) {
      playerColor = 'white';
    }
    
    if (!playerColor || gameState.currentPlayer !== playerColor) {
      return new Response(JSON.stringify({
        error: 'Not your turn'
      }), {
        status: 400,
        headers: { 'Content-Type': 'application/json' }
      });
    }
    
    // Recreate game and pass
    const game = this.deserializeGameState(gameState, settings.boardSize);
    game.pass(playerColor);
    
    const newGameState = this.serializeGameState(game);
    await this.sql.exec(
      `UPDATE rooms SET game_state = ?, updated_at = ? WHERE id = ?`,
      JSON.stringify(newGameState),
      Date.now(),
      roomId
    );
    
    return new Response(JSON.stringify({
      success: true,
      gameState: this.formatGameState(newGameState)
    }), {
      headers: { 'Content-Type': 'application/json' }
    });
  }
  
  async handleResign(request) {
    const url = new URL(request.url);
    const roomId = url.pathname.split('/')[3];
    
    const room = await this.getRoom(roomId);
    if (!room) {
      return new Response(JSON.stringify({
        error: 'Room not found'
      }), {
        status: 404,
        headers: { 'Content-Type': 'application/json' }
      });
    }
    
    const gameState = JSON.parse(room.game_state);
    const players = JSON.parse(room.players);
    const settings = JSON.parse(room.settings);
    const clientId = request.headers.get('CF-Connecting-IP');
    
    // Determine player color
    let playerColor = null;
    if (players.black && players.black.id === clientId) {
      playerColor = 'black';
    } else if (players.white && players.white.id === clientId) {
      playerColor = 'white';
    }
    
    if (!playerColor) {
      return new Response(JSON.stringify({
        error: 'You are not a player in this game'
      }), {
        status: 403,
        headers: { 'Content-Type': 'application/json' }
      });
    }
    
    // Recreate game and resign
    const game = this.deserializeGameState(gameState, settings.boardSize);
    game.resign(playerColor);
    
    const newGameState = this.serializeGameState(game);
    await this.sql.exec(
      `UPDATE rooms SET game_state = ?, updated_at = ? WHERE id = ?`,
      JSON.stringify(newGameState),
      Date.now(),
      roomId
    );
    
    return new Response(JSON.stringify({
      success: true,
      gameState: this.formatGameState(newGameState)
    }), {
      headers: { 'Content-Type': 'application/json' }
    });
  }
  
  async handleChat(request) {
    const url = new URL(request.url);
    const roomId = url.pathname.split('/')[3];
    const body = await request.json();
    const { message } = body;
    
    if (!message || message.trim() === '') {
      return new Response(JSON.stringify({
        error: 'Message cannot be empty'
      }), {
        status: 400,
        headers: { 'Content-Type': 'application/json' }
      });
    }
    
    const room = await this.getRoom(roomId);
    if (!room) {
      return new Response(JSON.stringify({
        error: 'Room not found'
      }), {
        status: 404,
        headers: { 'Content-Type': 'application/json' }
      });
    }
    
    const players = JSON.parse(room.players);
    const clientId = request.headers.get('CF-Connecting-IP');
    
    // Determine sender name
    let senderName = 'Spectator';
    if (players.black && players.black.id === clientId) {
      senderName = players.black.name;
    } else if (players.white && players.white.id === clientId) {
      senderName = players.white.name;
    }
    
    // Add message to room's chat history
    const chatMessage = {
      playerName: senderName,
      message: message.substring(0, 200), // Limit message length
      timestamp: Date.now()
    };
    
    // Store chat message in a new chat_messages table
    await this.sql.exec(
      `INSERT INTO chat_messages (room_id, player_name, message, timestamp) VALUES (?, ?, ?, ?)`,
      roomId, senderName, message.substring(0, 200), Date.now()
    );
    
    return new Response(JSON.stringify({
      success: true,
      chatMessage
    }), {
      headers: { 'Content-Type': 'application/json' }
    });
  }
  
  async getRoom(roomId) {
    const result = await this.sql.exec(
      `SELECT * FROM rooms WHERE id = ?`,
      roomId
    ).then(r => r.toArray()[0]);
    
    return result;
  }
  
  serializeGameState(game) {
    return {
      board: game.board,
      currentPlayer: game.currentPlayer,
      captures: game.captures,
      gameOver: game.gameOver,
      winner: game.winner,
      lastMove: game.lastMove,
      passes: game.passes,
      history: game.history.length // Store just the length to save space
    };
  }
  
  deserializeGameState(gameState, boardSize) {
    const game = new GoGame(boardSize);
    game.board = gameState.board;
    game.currentPlayer = gameState.currentPlayer;
    game.captures = gameState.captures;
    game.gameOver = gameState.gameOver;
    game.winner = gameState.winner;
    game.lastMove = gameState.lastMove;
    game.passes = gameState.passes;
    return game;
  }
  
  formatRoomState(roomData) {
    return {
      id: roomData.id,
      players: roomData.players,
      gameState: this.formatGameState(roomData.game_state),
      settings: roomData.settings
    };
  }
  
  formatGameState(gameState) {
    // Convert 2D board to 1D array for client
    const flatBoard = [];
    for (let y = 0; y < gameState.board.length; y++) {
      for (let x = 0; x < gameState.board[y].length; x++) {
        const stone = gameState.board[y][x];
        flatBoard.push(stone === 'black' ? 1 : stone === 'white' ? 2 : 0);
      }
    }
    
    return {
      board: flatBoard,
      boardSize: gameState.board.length,
      currentPlayer: gameState.currentPlayer,
      captures: gameState.captures,
      gameOver: gameState.gameOver,
      winner: gameState.winner,
      lastMove: gameState.lastMove
    };
  }
}