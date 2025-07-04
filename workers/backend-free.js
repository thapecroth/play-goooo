// Free-tier Cloudflare Worker backend for Go game
// This version uses KV storage instead of Durable Objects

export default {
  async fetch(request, env, ctx) {
    const url = new URL(request.url);
    
    // Add CORS headers
    const corsHeaders = {
      'Access-Control-Allow-Origin': '*',
      'Access-Control-Allow-Methods': 'GET, POST, OPTIONS',
      'Access-Control-Allow-Headers': 'Content-Type',
    };
    
    // Handle CORS preflight
    if (request.method === 'OPTIONS') {
      return new Response(null, { headers: corsHeaders });
    }
    
    // Simple REST API endpoints for game state
    if (url.pathname === '/api/status') {
      return new Response(JSON.stringify({
        status: 'online',
        message: 'Go Game Backend (Free Tier)',
        limitations: [
          'No real-time WebSocket support',
          'Use polling for game updates',
          'Limited to HTTP requests'
        ]
      }), {
        headers: {
          ...corsHeaders,
          'Content-Type': 'application/json'
        }
      });
    }
    
    // For now, return a message explaining the limitations
    return new Response(JSON.stringify({
      error: 'WebSocket not supported on free tier',
      message: 'Real-time multiplayer requires a paid Cloudflare plan or alternative hosting',
      alternatives: [
        'Run the server locally with npm run start:multiplayer',
        'Deploy to a VPS or cloud provider that supports WebSockets',
        'Upgrade to Cloudflare paid plan for Durable Objects support'
      ]
    }), {
      status: 200,
      headers: {
        ...corsHeaders,
        'Content-Type': 'application/json'
      }
    });
  }
};