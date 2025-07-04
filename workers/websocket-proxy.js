// Cloudflare Worker for WebSocket proxy to handle multiplayer Go game

export default {
  async fetch(request, env) {
    const upgradeHeader = request.headers.get('Upgrade');
    
    if (upgradeHeader !== 'websocket') {
      // Handle regular HTTP requests
      return handleHttpRequest(request, env);
    }
    
    // Handle WebSocket upgrade
    return handleWebSocketUpgrade(request, env);
  }
};

async function handleHttpRequest(request, env) {
  const url = new URL(request.url);
  
  // Proxy HTTP requests to backend
  const backendUrl = env.BACKEND_URL || 'http://localhost:3000';
  const proxyUrl = backendUrl + url.pathname + url.search;
  
  const proxyRequest = new Request(proxyUrl, {
    method: request.method,
    headers: request.headers,
    body: request.body
  });
  
  try {
    const response = await fetch(proxyRequest);
    return new Response(response.body, {
      status: response.status,
      statusText: response.statusText,
      headers: response.headers
    });
  } catch (error) {
    return new Response('Backend server unavailable', { status: 503 });
  }
}

async function handleWebSocketUpgrade(request, env) {
  // Create a WebSocket pair for client connection
  const [client, server] = new WebSocketPair();
  
  // Accept the WebSocket connection
  server.accept();
  
  // Get or create Durable Object for this connection
  const id = env.GAME_SESSIONS.newUniqueId();
  const gameSession = env.GAME_SESSIONS.get(id);
  
  // Handle the WebSocket connection in the Durable Object
  const response = await gameSession.fetch(request, {
    headers: {
      'Upgrade': 'websocket',
    },
  });
  
  // Return the client socket
  return new Response(null, {
    status: 101,
    webSocket: client,
  });
}

// Durable Object for managing game sessions
export class GameSession {
  constructor(state, env) {
    this.state = state;
    this.env = env;
    this.sessions = new Map();
  }
  
  async fetch(request) {
    const upgradeHeader = request.headers.get('Upgrade');
    if (upgradeHeader !== 'websocket') {
      return new Response('Expected WebSocket', { status: 400 });
    }
    
    // Create WebSocket pair
    const [client, server] = new WebSocketPair();
    
    // Accept the connection
    server.accept();
    
    // Store session
    const sessionId = crypto.randomUUID();
    this.sessions.set(sessionId, {
      socket: server,
      connectedAt: new Date()
    });
    
    // Connect to backend WebSocket
    this.connectToBackend(server, sessionId);
    
    // Handle client messages
    server.addEventListener('message', async (event) => {
      await this.handleMessage(sessionId, event.data);
    });
    
    server.addEventListener('close', () => {
      this.handleClose(sessionId);
    });
    
    return new Response(null, {
      status: 101,
      webSocket: client,
    });
  }
  
  async connectToBackend(clientSocket, sessionId) {
    const backendUrl = this.env.BACKEND_URL || 'http://localhost:3000';
    const wsUrl = backendUrl.replace('http://', 'ws://').replace('https://', 'wss://');
    
    try {
      const backendSocket = new WebSocket(wsUrl);
      
      backendSocket.addEventListener('open', () => {
        console.log('Connected to backend');
      });
      
      backendSocket.addEventListener('message', (event) => {
        // Forward messages from backend to client
        clientSocket.send(event.data);
      });
      
      backendSocket.addEventListener('close', () => {
        console.log('Backend connection closed');
        clientSocket.close();
      });
      
      backendSocket.addEventListener('error', (error) => {
        console.error('Backend connection error:', error);
        clientSocket.close();
      });
      
      // Store backend socket reference
      const session = this.sessions.get(sessionId);
      if (session) {
        session.backendSocket = backendSocket;
      }
    } catch (error) {
      console.error('Failed to connect to backend:', error);
      clientSocket.send(JSON.stringify({
        type: 'error',
        message: 'Failed to connect to game server'
      }));
      clientSocket.close();
    }
  }
  
  async handleMessage(sessionId, message) {
    const session = this.sessions.get(sessionId);
    if (!session || !session.backendSocket) return;
    
    // Forward message to backend
    session.backendSocket.send(message);
  }
  
  handleClose(sessionId) {
    const session = this.sessions.get(sessionId);
    if (session) {
      if (session.backendSocket) {
        session.backendSocket.close();
      }
      this.sessions.delete(sessionId);
    }
  }
}