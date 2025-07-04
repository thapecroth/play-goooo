// Multiplayer Go Game Client

const socket = io(window.gameConfig ? window.gameConfig.getSocketUrl() : window.location.origin);

// Game state
let currentRoom = null;
let playerColor = null;
let isSpectator = false;
let gameState = null;
let boardSize = 9;
let boardElement = null;

// UI Elements
const lobbyView = document.getElementById('lobby-view');
const gameView = document.getElementById('game-view');
const playerNameInput = document.getElementById('player-name');
const newBoardSizeSelect = document.getElementById('new-board-size');
const vsAiCheckbox = document.getElementById('vs-ai');
const createRoomBtn = document.getElementById('create-room');
const roomCodeInput = document.getElementById('room-code');
const joinRoomBtn = document.getElementById('join-room');
const roomsListElement = document.getElementById('rooms-list');
const roomIdDisplay = document.getElementById('room-id-display');
const copyRoomCodeBtn = document.getElementById('copy-room-code');
const leaveGameBtn = document.getElementById('leave-game');
const passBtn = document.getElementById('pass');
const resignBtn = document.getElementById('resign');
const chatInput = document.getElementById('chat-input');
const sendChatBtn = document.getElementById('send-chat');
const chatMessages = document.getElementById('chat-messages');

// Initialize
document.addEventListener('DOMContentLoaded', () => {
    // Check if multiplayer is available
    if (!window.gameConfig || !window.gameConfig.hasBackendConfigured()) {
        showBackendNotice();
        return;
    }
    
    setupEventListeners();
    loadPlayerName();
    
    // Add connection status handling
    let connectionTimeout;
    
    socket.on('connect', () => {
        console.log('Connected to game server');
        clearTimeout(connectionTimeout);
        hideConnectionError();
        requestRoomsList();
        // Refresh rooms list every 5 seconds
        setInterval(requestRoomsList, 5000);
    });
    
    socket.on('connect_error', (error) => {
        console.error('Connection error:', error);
        showConnectionError();
    });
    
    // Set a timeout to show error if connection fails
    connectionTimeout = setTimeout(() => {
        if (!socket.connected) {
            showConnectionError();
        }
    }, 3000);
});

function setupEventListeners() {
    createRoomBtn.addEventListener('click', createRoom);
    joinRoomBtn.addEventListener('click', () => joinRoom(roomCodeInput.value));
    copyRoomCodeBtn.addEventListener('click', copyRoomCode);
    leaveGameBtn.addEventListener('click', leaveGame);
    passBtn.addEventListener('click', pass);
    resignBtn.addEventListener('click', resign);
    sendChatBtn.addEventListener('click', sendChat);
    chatInput.addEventListener('keypress', (e) => {
        if (e.key === 'Enter') sendChat();
    });
    playerNameInput.addEventListener('change', savePlayerName);
}

function loadPlayerName() {
    const savedName = localStorage.getItem('playerName');
    if (savedName) {
        playerNameInput.value = savedName;
    }
}

function savePlayerName() {
    localStorage.setItem('playerName', playerNameInput.value);
}

function getPlayerName() {
    return playerNameInput.value || 'Anonymous';
}

// Room Management
function createRoom() {
    const playerName = getPlayerName();
    const boardSize = parseInt(newBoardSizeSelect.value);
    const vsAI = vsAiCheckbox.checked;
    
    socket.emit('createRoom', {
        playerName: playerName,
        boardSize: boardSize,
        vsAI: vsAI
    });
}

function joinRoom(roomId) {
    if (!roomId) {
        showError('Please enter a room code');
        return;
    }
    
    const playerName = getPlayerName();
    socket.emit('joinRoom', {
        roomId: roomId.toUpperCase(),
        playerName: playerName
    });
}

function joinPublicRoom(roomId) {
    roomCodeInput.value = roomId;
    joinRoom(roomId);
}

function leaveGame() {
    if (confirm('Are you sure you want to leave the game?')) {
        location.reload(); // Simple way to disconnect and reset
    }
}

function requestRoomsList() {
    if (lobbyView.style.display !== 'none') {
        socket.emit('getRooms');
    }
}

// Socket Event Handlers
socket.on('roomCreated', (data) => {
    handleRoomJoined(data);
});

socket.on('roomJoined', (data) => {
    handleRoomJoined(data);
});

socket.on('joinError', (error) => {
    showError(error);
});

socket.on('roomsList', (rooms) => {
    updateRoomsList(rooms);
});

socket.on('playerJoined', (data) => {
    updateRoomState(data.roomState);
    addChatMessage('System', `${data.playerName} joined the game`);
});

socket.on('playerLeft', (data) => {
    updateRoomState(data.roomState);
    addChatMessage('System', 'A player left the game');
});

socket.on('gameState', (state) => {
    updateGameState(state);
});

socket.on('invalidMove', (data) => {
    if (data.reason) {
        showError(data.reason);
    }
});

socket.on('chatMessage', (data) => {
    addChatMessage(data.playerName, data.message, data.timestamp);
});

socket.on('error', (error) => {
    showError(error);
});

// UI Updates
function handleRoomJoined(data) {
    currentRoom = data.roomId;
    playerColor = data.color;
    isSpectator = data.spectator || false;
    
    roomIdDisplay.textContent = currentRoom;
    
    // Switch to game view
    lobbyView.style.display = 'none';
    gameView.style.display = 'block';
    
    // Update room state
    updateRoomState(data.roomState);
    
    // Initialize board
    if (data.roomState) {
        boardSize = data.roomState.gameState.boardSize;
        initializeBoard(boardSize);
        updateGameState(data.roomState.gameState);
    }
    
    // Show spectator banner if needed
    if (isSpectator) {
        showSpectatorBanner();
    }
}

function updateRoomState(roomState) {
    if (!roomState) return;
    
    // Update player names
    const blackPlayerName = document.getElementById('black-player-name');
    const whitePlayerName = document.getElementById('white-player-name');
    const blackStatus = document.getElementById('black-status');
    const whiteStatus = document.getElementById('white-status');
    
    if (roomState.players.black) {
        blackPlayerName.textContent = `Black (${roomState.players.black.name})`;
        blackStatus.textContent = roomState.players.black.connected ? '● Connected' : '○ Disconnected';
        blackStatus.className = `player-status ${roomState.players.black.connected ? 'connected' : 'disconnected'}`;
    } else {
        blackPlayerName.textContent = 'Black (Waiting...)';
        blackStatus.textContent = 'Waiting for player';
        blackStatus.className = 'player-status waiting';
    }
    
    if (roomState.players.white) {
        whitePlayerName.textContent = `White (${roomState.players.white.name})`;
        whiteStatus.textContent = roomState.players.white.connected ? '● Connected' : '○ Disconnected';
        whiteStatus.className = `player-status ${roomState.players.white.connected ? 'connected' : 'disconnected'}`;
    } else {
        whitePlayerName.textContent = 'White (Waiting...)';
        whiteStatus.textContent = 'Waiting for player';
        whiteStatus.className = 'player-status waiting';
    }
    
    // Add "You" indicator
    if (playerColor === 'black' && roomState.players.black) {
        blackPlayerName.innerHTML += ' <strong>(You)</strong>';
    } else if (playerColor === 'white' && roomState.players.white) {
        whitePlayerName.innerHTML += ' <strong>(You)</strong>';
    }
    
    // Update spectator count
    if (roomState.spectators > 0) {
        const spectatorInfo = document.createElement('div');
        spectatorInfo.className = 'spectator-info';
        spectatorInfo.textContent = `${roomState.spectators} spectator(s)`;
    }
}

function updateRoomsList(rooms) {
    if (rooms.length === 0) {
        roomsListElement.innerHTML = '<div class="loading">No public games available</div>';
        return;
    }
    
    roomsListElement.innerHTML = rooms.map(room => `
        <div class="room-item" onclick="joinPublicRoom('${room.id}')">
            <div class="room-item-header">
                <span class="room-item-id">${room.id}</span>
                <span class="room-item-board">${room.boardSize}x${room.boardSize}</span>
            </div>
            <div class="room-item-players">
                Black: ${room.players.black || 'Empty'} | 
                White: ${room.players.white || 'Empty'} | 
                Spectators: ${room.spectators}
            </div>
        </div>
    `).join('');
}

function updateGameState(state) {
    gameState = state;
    renderBoard();
    updateScores();
    updateCurrentPlayer();
    updateGameStatus();
}

function updateScores() {
    if (!gameState) return;
    
    document.getElementById('black-captures').textContent = gameState.captures.white;
    document.getElementById('white-captures').textContent = gameState.captures.black;
}

function updateCurrentPlayer() {
    if (!gameState) return;
    
    const currentPlayerElement = document.getElementById('current-player');
    currentPlayerElement.textContent = gameState.currentPlayer.charAt(0).toUpperCase() + gameState.currentPlayer.slice(1);
    
    // Highlight current player's score area
    document.querySelector('.black-score').classList.toggle('active', gameState.currentPlayer === 'black');
    document.querySelector('.white-score').classList.toggle('active', gameState.currentPlayer === 'white');
}

function updateGameStatus() {
    if (!gameState) return;
    
    const statusElement = document.getElementById('status-message');
    
    if (gameState.gameOver) {
        let message = '';
        if (gameState.resigned) {
            message = `${gameState.resigned} resigned. ${gameState.resigned === 'black' ? 'White' : 'Black'} wins!`;
        } else {
            const scores = calculateFinalScores();
            message = `Game Over! Black: ${scores.black}, White: ${scores.white}. `;
            message += scores.black > scores.white ? 'Black wins!' : 'White wins!';
        }
        statusElement.textContent = message;
        statusElement.className = 'game-over';
    } else if (gameState.lastMove && gameState.lastMove.pass) {
        statusElement.textContent = `${gameState.lastMove.player} passed`;
        statusElement.className = 'pass-notification';
    } else {
        statusElement.textContent = '';
    }
}

// Board Rendering
function initializeBoard(size) {
    boardSize = size;
    boardElement = document.getElementById('board');
    boardElement.className = `board board-${size}`;
    boardElement.innerHTML = '';
    
    // Create intersection elements
    for (let y = 0; y < size; y++) {
        for (let x = 0; x < size; x++) {
            const intersection = document.createElement('div');
            intersection.className = 'intersection';
            intersection.dataset.x = x;
            intersection.dataset.y = y;
            
            if (!isSpectator) {
                intersection.addEventListener('click', handleIntersectionClick);
                intersection.addEventListener('mouseenter', handleIntersectionHover);
                intersection.addEventListener('mouseleave', handleIntersectionLeave);
            }
            
            boardElement.appendChild(intersection);
        }
    }
}

function renderBoard() {
    if (!gameState || !boardElement) return;
    
    const intersections = boardElement.querySelectorAll('.intersection');
    
    intersections.forEach(intersection => {
        const x = parseInt(intersection.dataset.x);
        const y = parseInt(intersection.dataset.y);
        const index = y * boardSize + x;
        
        intersection.className = 'intersection';
        
        if (gameState.board[index] === 1) {
            intersection.classList.add('black');
        } else if (gameState.board[index] === 2) {
            intersection.classList.add('white');
        }
        
        // Highlight last move
        if (gameState.lastMove && !gameState.lastMove.pass &&
            gameState.lastMove.x === x && gameState.lastMove.y === y) {
            intersection.classList.add('last-move');
        }
    });
}

function handleIntersectionClick(e) {
    if (isSpectator || !gameState || gameState.gameOver) return;
    if (gameState.currentPlayer !== playerColor) return;
    
    const x = parseInt(e.target.dataset.x);
    const y = parseInt(e.target.dataset.y);
    
    socket.emit('makeMove', { x, y });
}

function handleIntersectionHover(e) {
    if (isSpectator || !gameState || gameState.gameOver) return;
    if (gameState.currentPlayer !== playerColor) return;
    
    const x = parseInt(e.target.dataset.x);
    const y = parseInt(e.target.dataset.y);
    const index = y * boardSize + x;
    
    if (gameState.board[index] === 0) {
        e.target.classList.add('hover', playerColor);
    }
}

function handleIntersectionLeave(e) {
    e.target.classList.remove('hover', 'black', 'white');
}

// Game Actions
function pass() {
    if (isSpectator || !gameState || gameState.gameOver) return;
    if (gameState.currentPlayer !== playerColor) return;
    
    socket.emit('pass');
}

function resign() {
    if (isSpectator || !gameState || gameState.gameOver) return;
    if (gameState.currentPlayer !== playerColor) return;
    
    if (confirm('Are you sure you want to resign?')) {
        socket.emit('resign');
    }
}

// Chat
function sendChat() {
    const message = chatInput.value.trim();
    if (!message) return;
    
    socket.emit('chatMessage', message);
    chatInput.value = '';
}

function addChatMessage(playerName, message, timestamp) {
    const messageElement = document.createElement('div');
    messageElement.className = 'chat-message';
    
    const time = timestamp ? new Date(timestamp).toLocaleTimeString() : new Date().toLocaleTimeString();
    
    messageElement.innerHTML = `
        <span class="chat-message-time">${time}</span>
        <span class="chat-message-header">${playerName}:</span>
        <span class="chat-message-text">${message}</span>
    `;
    
    chatMessages.appendChild(messageElement);
    chatMessages.scrollTop = chatMessages.scrollHeight;
}

// Utilities
function copyRoomCode() {
    navigator.clipboard.writeText(currentRoom).then(() => {
        const btn = copyRoomCodeBtn;
        const originalText = btn.textContent;
        btn.textContent = 'Copied!';
        setTimeout(() => {
            btn.textContent = originalText;
        }, 2000);
    });
}

function showError(message) {
    const errorDiv = document.createElement('div');
    errorDiv.className = 'error-message';
    errorDiv.textContent = message;
    
    const container = document.querySelector('.container');
    container.insertBefore(errorDiv, container.firstChild);
    
    setTimeout(() => {
        errorDiv.remove();
    }, 5000);
}

function showSpectatorBanner() {
    const banner = document.createElement('div');
    banner.className = 'spectator-banner';
    banner.textContent = 'You are spectating this game';
    
    const gameInfo = document.querySelector('.game-info');
    gameInfo.insertBefore(banner, gameInfo.firstChild);
}

function calculateFinalScores() {
    // Simple area scoring for now
    let blackScore = gameState.captures.white;
    let whiteScore = gameState.captures.black + 6.5; // Komi
    
    // Count controlled territory (simplified)
    for (let i = 0; i < gameState.board.length; i++) {
        if (gameState.board[i] === 1) blackScore++;
        else if (gameState.board[i] === 2) whiteScore++;
    }
    
    return { black: blackScore, white: whiteScore };
}

// Helper functions for connection status
async function showBackendNotice() {
    const lobbyView = document.getElementById('lobby-view');
    
    // Check backend status
    let backendInfo = '';
    if (window.gameConfig && window.gameConfig.getBackendStatus) {
        const status = await window.gameConfig.getBackendStatus();
        if (status.status === 'online') {
            backendInfo = `
                <div style="background: #fff3cd; padding: 15px; border-radius: 5px; margin: 20px 0;">
                    <strong>Backend Status:</strong> ${status.message}<br>
                    <strong>Limitations:</strong>
                    <ul style="margin: 10px 0; padding-left: 20px;">
                        ${status.limitations.map(l => `<li>${l}</li>`).join('')}
                    </ul>
                </div>
            `;
        }
    }
    
    lobbyView.innerHTML = `
        <div style="text-align: center; padding: 40px;">
            <h2>Multiplayer Mode</h2>
            <div style="background: #f8f8f8; padding: 30px; border-radius: 10px; margin: 20px auto; max-width: 600px;">
                <p style="font-size: 18px; color: #666; margin-bottom: 20px;">
                    Real-time multiplayer requires WebSocket support.
                </p>
                ${backendInfo}
                <p style="color: #999; margin-bottom: 20px;">
                    Cloudflare's free tier doesn't support WebSockets. To play multiplayer:
                </p>
                <ol style="text-align: left; display: inline-block; color: #666;">
                    <li>Clone the repository locally</li>
                    <li>Run <code>npm install</code></li>
                    <li>Run <code>npm run start:multiplayer</code></li>
                    <li>Open <code>http://localhost:3000/multiplayer.html</code></li>
                </ol>
                <div style="margin-top: 20px; padding: 15px; background: #e8f4fd; border-radius: 5px;">
                    <strong>Alternative Options:</strong>
                    <ul style="text-align: left; display: inline-block; margin: 10px 0;">
                        <li>Deploy to Heroku, Railway, or Render (free tiers available)</li>
                        <li>Use a VPS from DigitalOcean or Linode</li>
                        <li>Upgrade to Cloudflare paid plan for Durable Objects</li>
                    </ul>
                </div>
                <p style="margin-top: 30px;">
                    <a href="/" class="btn btn-primary">Play vs AI Instead</a>
                </p>
                <p style="margin-top: 15px; font-size: 14px; color: #999;">
                    Single-player vs AI works perfectly on this deployment!
                </p>
            </div>
        </div>
    `;
}

function showConnectionError() {
    const errorDiv = document.createElement('div');
    errorDiv.id = 'connection-error';
    errorDiv.className = 'error-message';
    errorDiv.style.cssText = 'position: fixed; top: 20px; left: 50%; transform: translateX(-50%); z-index: 1000;';
    errorDiv.innerHTML = `
        <strong>Connection Error</strong><br>
        Cannot connect to game server. Make sure the server is running.
    `;
    
    if (!document.getElementById('connection-error')) {
        document.body.appendChild(errorDiv);
    }
    
    // Disable buttons
    createRoomBtn.disabled = true;
    joinRoomBtn.disabled = true;
}

function hideConnectionError() {
    const errorDiv = document.getElementById('connection-error');
    if (errorDiv) {
        errorDiv.remove();
    }
    
    // Enable buttons
    createRoomBtn.disabled = false;
    joinRoomBtn.disabled = false;
}

// Export joinPublicRoom for onclick handler
window.joinPublicRoom = joinPublicRoom;