// Configuration for different environments
const config = {
  // Check if running locally or in production
  getSocketUrl: function() {
    const hostname = window.location.hostname;
    
    // Local development
    if (hostname === 'localhost' || hostname === '127.0.0.1') {
      return window.location.origin;
    }
    
    // Production - Cloudflare Pages
    // You'll need to update this with your actual backend URL
    return 'https://go-game-backend.example.com';
  },
  
  // Check if multiplayer is available
  isMultiplayerAvailable: function() {
    const hostname = window.location.hostname;
    // Multiplayer only available on localhost or with a configured backend
    return hostname === 'localhost' || hostname === '127.0.0.1' || this.hasBackendConfigured();
  },
  
  hasBackendConfigured: function() {
    // Check if a real backend URL is configured
    const socketUrl = this.getSocketUrl();
    return socketUrl && !socketUrl.includes('example.com');
  }
};

// Export for use in other scripts
window.gameConfig = config;