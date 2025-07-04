# Go Game Multiplayer Setup

This guide explains how to set up and deploy the multiplayer version of the Go game.

## Live Demo

🎮 **Play Now**: https://go-game-multiplayer.pages.dev/

The game is deployed on Cloudflare Pages as a static site. Currently, it supports:
- ✅ Single-player vs AI (fully functional)
- ⚠️ Multiplayer (requires backend server - see below)

## Local Development

### 1. Start the Multiplayer Server

```bash
npm run start:multiplayer
# or for development with auto-reload
npm run dev:multiplayer
```

The multiplayer server will run on http://localhost:3000

### 2. Access the Game

- Single Player: http://localhost:3000/
- Multiplayer: http://localhost:3000/multiplayer.html

## Features

- **Create/Join Rooms**: Players can create new game rooms or join existing ones using a 6-character room code
- **Real-time Gameplay**: Uses Socket.IO for real-time move synchronization
- **Spectator Mode**: When a room is full, additional users can join as spectators
- **Chat System**: In-game chat for players and spectators
- **Multiple Board Sizes**: Support for 9x9, 13x13, and 19x19 boards
- **AI Opponent Option**: Can create rooms with AI opponents

## Cloudflare Pages Deployment (Static Frontend)

The frontend is already deployed at https://go-game-multiplayer.pages.dev/

To deploy your own version:

```bash
# Install Wrangler CLI
npm install -g wrangler

# Login to Cloudflare
wrangler login

# Create a new Pages project
wrangler pages project create your-project-name

# Deploy the static files
wrangler pages deploy public --project-name=your-project-name
```

Your game will be available at `https://your-project-name.pages.dev/`

## Cloudflare Workers Deployment (Backend Proxy)

### Prerequisites

1. Install Wrangler CLI:
```bash
npm install -g wrangler
```

2. Login to Cloudflare:
```bash
wrangler login
```

### Option 1: Simple HTTP Proxy (Free tier)

This option proxies HTTP requests but doesn't support WebSockets on Cloudflare's free tier.

1. Update the backend URL in `wrangler-simple.toml`:
```toml
[env.production.vars]
BACKEND_URL = "https://your-backend-server.com"
```

2. Deploy:
```bash
wrangler publish -c wrangler-simple.toml
```

### Option 2: WebSocket Support (Requires Cloudflare Pro)

For full WebSocket support through Cloudflare:

1. Ensure you have a Cloudflare Pro plan or higher
2. Update `wrangler.toml` with your domain and backend URL
3. Deploy:
```bash
wrangler publish
```

### Alternative: Cloudflare Tunnel

For the best WebSocket support without proxying through Workers:

1. Install cloudflared:
```bash
# macOS
brew install cloudflare/cloudflare/cloudflared

# Linux
wget -q https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64.deb
sudo dpkg -i cloudflared-linux-amd64.deb
```

2. Create a tunnel:
```bash
cloudflared tunnel create go-game
```

3. Create config file `~/.cloudflared/config.yml`:
```yaml
tunnel: <TUNNEL_ID>
credentials-file: /home/user/.cloudflared/<TUNNEL_ID>.json

ingress:
  - hostname: go-game.yourdomain.com
    service: http://localhost:3000
  - service: http_status:404
```

4. Route traffic:
```bash
cloudflared tunnel route dns go-game go-game.yourdomain.com
```

5. Run the tunnel:
```bash
cloudflared tunnel run go-game
```

## Production Deployment

### Backend Server Requirements

1. **Node.js Server**: The multiplayer server needs to run on a VPS or cloud provider that supports WebSockets
2. **Recommended Providers**:
   - DigitalOcean
   - Linode
   - AWS EC2
   - Google Cloud Platform
   - Heroku (with WebSocket support)

### Deployment Steps

1. Clone the repository on your server
2. Install dependencies:
```bash
npm install
```

3. Set up a process manager (PM2 recommended):
```bash
npm install -g pm2
pm2 start server-multiplayer.js --name go-game
pm2 save
pm2 startup
```

4. Configure nginx (if using):
```nginx
server {
    listen 80;
    server_name your-domain.com;

    location / {
        proxy_pass http://localhost:3000;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection 'upgrade';
        proxy_set_header Host $host;
        proxy_cache_bypass $http_upgrade;
    }
}
```

5. Set up SSL with Let's Encrypt:
```bash
sudo certbot --nginx -d your-domain.com
```

## Environment Variables

Create a `.env` file for production:

```env
PORT=3000
NODE_ENV=production
```

## Troubleshooting

### WebSocket Connection Issues

1. Ensure your firewall allows WebSocket connections
2. Check that your reverse proxy (nginx/Apache) is configured for WebSocket support
3. Verify that Cloudflare WebSocket support is enabled (Pro plan required)

### CORS Issues

The server includes CORS headers, but you may need to adjust them for your domain:

```javascript
const io = socketIO(server, {
  cors: {
    origin: "https://your-domain.com",
    methods: ["GET", "POST"]
  }
});
```

## Next Steps

1. Add user authentication
2. Implement game persistence (save/resume games)
3. Add player rankings and statistics
4. Create tournament mode
5. Add more AI difficulty levels