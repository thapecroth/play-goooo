// Simple Cloudflare Worker proxy for Go game (without Durable Objects)

export default {
  async fetch(request, env) {
    const url = new URL(request.url);
    
    // Get the backend URL from environment or use default
    const backendUrl = env.BACKEND_URL || 'http://localhost:3000';
    
    // Handle CORS preflight
    if (request.method === 'OPTIONS') {
      return new Response(null, {
        headers: {
          'Access-Control-Allow-Origin': '*',
          'Access-Control-Allow-Methods': 'GET, POST, OPTIONS',
          'Access-Control-Allow-Headers': 'Content-Type',
        },
      });
    }
    
    // Proxy the request to the backend
    const proxyUrl = backendUrl + url.pathname + url.search;
    
    // Forward the request
    const proxyRequest = new Request(proxyUrl, {
      method: request.method,
      headers: {
        ...Object.fromEntries(request.headers),
        'X-Forwarded-For': request.headers.get('CF-Connecting-IP') || '',
        'X-Forwarded-Proto': 'https',
        'X-Real-IP': request.headers.get('CF-Connecting-IP') || '',
      },
      body: request.body,
    });
    
    try {
      const response = await fetch(proxyRequest);
      
      // Create new response with CORS headers
      const newResponse = new Response(response.body, {
        status: response.status,
        statusText: response.statusText,
        headers: {
          ...Object.fromEntries(response.headers),
          'Access-Control-Allow-Origin': '*',
          'Access-Control-Allow-Methods': 'GET, POST, OPTIONS',
          'Access-Control-Allow-Headers': 'Content-Type',
        },
      });
      
      return newResponse;
    } catch (error) {
      console.error('Proxy error:', error);
      return new Response('Backend server unavailable', { 
        status: 503,
        headers: {
          'Access-Control-Allow-Origin': '*',
        },
      });
    }
  },
};