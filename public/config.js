// Configuration for different environments
const config = {
  // Backend API URL (for status checking)
  backendApiUrl: 'https://go-game-backend-free-production.thapecroth.workers.dev',
  
  // Check if running locally or in production
  getSocketUrl: function() {
    const hostname = window.location.hostname;
    
    // Local development
    if (hostname === 'localhost' || hostname === '127.0.0.1') {
      return window.location.origin;
    }
    
    // Production - Cloudflare Workers don't support WebSockets on free tier
    // Real-time multiplayer requires either:
    // 1. Running locally
    // 2. Deploying to a WebSocket-capable host
    // 3. Upgrading to Cloudflare paid plan
    return null;
  },
  
  // Check if multiplayer is available
  isMultiplayerAvailable: function() {
    const hostname = window.location.hostname;
    // Multiplayer only available on localhost
    return hostname === 'localhost' || hostname === '127.0.0.1';
  },
  
  hasBackendConfigured: function() {
    // WebSocket backend only available locally on free tier
    const socketUrl = this.getSocketUrl();
    return socketUrl !== null;
  },
  
  // Get backend status (for displaying info to users)
  getBackendStatus: async function() {
    try {
      const response = await fetch(this.backendApiUrl + '/api/status');
      return await response.json();
    } catch (error) {
      return {
        status: 'offline',
        error: error.message
      };
    }
  }
};

// Export for use in other scripts
window.gameConfig = config;