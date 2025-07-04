// HTTP-based Multiplayer Go Game Client
// Uses polling instead of WebSockets for Cloudflare free tier compatibility

// Configuration
const BACKEND_URL = window.gameConfig ? window.gameConfig.backendApiUrl : 'https://go-game-backend-sqlite-production.thapecroth.workers.dev';
const POLL_INTERVAL = 3000; // 3 seconds - respects rate limit
const MAX_POLL_FAILURES = 3;

// Game state
let currentRoom = null;
let playerColor = null;
let isSpectator = false;
let gameState = null;
let boardSize = 9;
let boardElement = null;
let pollTimer = null;
let pollFailures = 0;
let lastStateUpdate = 0;

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
const statusMessage = document.getElementById('status-message');

// Initialize
document.addEventListener('DOMContentLoaded', () => {
    setupEventListeners();
    loadPlayerName();
    checkBackendStatus();
    loadRoomsList();
});

function setupEventListeners() {
    createRoomBtn.addEventListener('click', createRoom);
    joinRoomBtn.addEventListener('click', () => joinRoom(roomCodeInput.value));
    copyRoomCodeBtn.addEventListener('click', copyRoomCode);
    leaveGameBtn.addEventListener('click', leaveGame);
    passBtn.addEventListener('click', pass);
    resignBtn.addEventListener('click', resign);
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

// Check backend status
async function checkBackendStatus() {
    try {
        const response = await fetch(`${BACKEND_URL}/api/status`);
        const data = await response.json();
        
        if (data.status === 'online') {
            console.log('Backend status:', data);
            showStatus('Connected to game server', 'success');
        }
    } catch (error) {
        console.error('Failed to connect to backend:', error);
        showStatus('Failed to connect to game server', 'error');
    }
}

// Room Management
async function createRoom() {
    const playerName = getPlayerName();
    const boardSize = parseInt(newBoardSizeSelect.value);
    const vsAI = vsAiCheckbox.checked;
    
    try {
        showStatus('Creating room...', 'info');
        
        const response = await fetch(`${BACKEND_URL}/api/rooms/create`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ playerName, boardSize, vsAI })
        });
        
        const data = await response.json();
        
        if (response.ok) {
            handleRoomJoined(data);
        } else {
            showStatus(data.error || 'Failed to create room', 'error');
        }
    } catch (error) {
        console.error('Error creating room:', error);
        showStatus('Failed to create room', 'error');
    }
}

async function joinRoom(roomId) {
    if (!roomId) {
        showStatus('Please enter a room code', 'error');
        return;
    }
    
    const playerName = getPlayerName();
    
    try {
        showStatus('Joining room...', 'info');
        
        const response = await fetch(`${BACKEND_URL}/api/rooms/join`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ roomId: roomId.toUpperCase(), playerName })
        });
        
        const data = await response.json();
        
        if (response.ok) {
            handleRoomJoined(data);
        } else {
            showStatus(data.error || 'Failed to join room', 'error');
        }
    } catch (error) {
        console.error('Error joining room:', error);
        showStatus('Failed to join room', 'error');
    }
}

async function loadRoomsList() {
    try {
        const response = await fetch(`${BACKEND_URL}/api/rooms/list`);
        const rooms = await response.json();
        
        if (rooms.length === 0) {
            roomsListElement.innerHTML = '<div class="loading">No public games available</div>';
        } else {
            roomsListElement.innerHTML = rooms.map(room => `
                <div class="room-item" onclick="joinPublicRoom('${room.id}')">
                    <div class="room-item-header">
                        <span class="room-item-id">${room.id}</span>
                        <span class="room-item-board">${room.boardSize}x${room.boardSize}</span>
                    </div>
                    <div class="room-item-players">
                        Black: ${room.players.black || 'Empty'} | 
                        White: ${room.players.white || 'Empty'}
                    </div>
                </div>
            `).join('');
        }
    } catch (error) {
        console.error('Error loading rooms:', error);
        roomsListElement.innerHTML = '<div class="loading">Failed to load rooms</div>';
    }
    
    // Refresh rooms list periodically
    setTimeout(loadRoomsList, 10000);
}

function joinPublicRoom(roomId) {
    roomCodeInput.value = roomId;
    joinRoom(roomId);
}

function handleRoomJoined(data) {
    currentRoom = data.roomId;
    playerColor = data.color;
    isSpectator = data.isSpectator || false;
    
    roomIdDisplay.textContent = currentRoom;
    
    // Switch to game view
    lobbyView.style.display = 'none';
    gameView.style.display = 'block';
    
    // Update room state
    updateRoomState(data.roomState);
    
    // Initialize board
    if (data.roomState && data.roomState.gameState) {
        boardSize = data.roomState.gameState.boardSize;
        initializeBoard(boardSize);
        updateGameState(data.roomState.gameState);
    }
    
    // Show spectator banner if needed
    if (isSpectator) {
        showSpectatorBanner();
    }
    
    // Start polling for updates
    startPolling();
}

function startPolling() {
    // Clear any existing timer
    stopPolling();
    
    // Poll for updates
    pollTimer = setInterval(async () => {
        if (!currentRoom) {
            stopPolling();
            return;
        }
        
        try {
            const response = await fetch(`${BACKEND_URL}/api/room/${currentRoom}/state`);
            const data = await response.json();
            
            if (response.ok) {
                pollFailures = 0;
                updateRoomState(data.roomState);
                
                // Update poll interval if provided
                if (data.pollInterval && data.pollInterval !== POLL_INTERVAL) {
                    stopPolling();
                    setTimeout(startPolling, data.pollInterval);
                }
            } else if (response.status === 429) {
                // Rate limited
                showStatus('Too many requests. Slowing down...', 'warning');
                stopPolling();
                setTimeout(startPolling, 10000); // Wait 10 seconds before retrying
            } else {
                pollFailures++;
                if (pollFailures >= MAX_POLL_FAILURES) {
                    showStatus('Lost connection to game', 'error');
                    stopPolling();
                }
            }
        } catch (error) {
            console.error('Polling error:', error);
            pollFailures++;
            if (pollFailures >= MAX_POLL_FAILURES) {
                showStatus('Lost connection to game', 'error');
                stopPolling();
            }
        }
    }, POLL_INTERVAL);
}

function stopPolling() {
    if (pollTimer) {
        clearInterval(pollTimer);
        pollTimer = null;
    }
}

function leaveGame() {
    if (confirm('Are you sure you want to leave the game?')) {
        stopPolling();
        location.reload();
    }
}

// Game Actions
async function makeMove(x, y) {
    if (isSpectator || !gameState || gameState.gameOver) return;
    if (gameState.currentPlayer !== playerColor) return;
    
    try {
        const response = await fetch(`${BACKEND_URL}/api/room/${currentRoom}/move`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ x, y })
        });
        
        const data = await response.json();
        
        if (response.ok) {
            updateGameState(data.gameState);
        } else {
            showStatus(data.error || 'Invalid move', 'error');
        }
    } catch (error) {
        console.error('Error making move:', error);
        showStatus('Failed to make move', 'error');
    }
}

async function pass() {
    if (isSpectator || !gameState || gameState.gameOver) return;
    if (gameState.currentPlayer !== playerColor) return;
    
    try {
        const response = await fetch(`${BACKEND_URL}/api/room/${currentRoom}/pass`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' }
        });
        
        const data = await response.json();
        
        if (response.ok) {
            updateGameState(data.gameState);
        } else {
            showStatus(data.error || 'Failed to pass', 'error');
        }
    } catch (error) {
        console.error('Error passing:', error);
        showStatus('Failed to pass', 'error');
    }
}

async function resign() {
    if (isSpectator || !gameState || gameState.gameOver) return;
    
    if (confirm('Are you sure you want to resign?')) {
        try {
            const response = await fetch(`${BACKEND_URL}/api/room/${currentRoom}/resign`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' }
            });
            
            const data = await response.json();
            
            if (response.ok) {
                updateGameState(data.gameState);
            } else {
                showStatus(data.error || 'Failed to resign', 'error');
            }
        } catch (error) {
            console.error('Error resigning:', error);
            showStatus('Failed to resign', 'error');
        }
    }
}

// UI Updates
function updateRoomState(roomState) {
    if (!roomState) return;
    
    // Update player names
    const blackPlayerName = document.getElementById('black-player-name');
    const whitePlayerName = document.getElementById('white-player-name');
    const blackStatus = document.getElementById('black-status');
    const whiteStatus = document.getElementById('white-status');
    
    if (roomState.players.black) {
        blackPlayerName.textContent = `Black (${roomState.players.black.name})`;
        blackStatus.textContent = '● Connected';
        blackStatus.className = 'player-status connected';
    } else {
        blackPlayerName.textContent = 'Black (Waiting...)';
        blackStatus.textContent = 'Waiting for player';
        blackStatus.className = 'player-status waiting';
    }
    
    if (roomState.players.white) {
        whitePlayerName.textContent = `White (${roomState.players.white.name})`;
        whiteStatus.textContent = '● Connected';
        whiteStatus.className = 'player-status connected';
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
    
    // Update game state
    if (roomState.gameState) {
        updateGameState(roomState.gameState);
    }
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
    
    const statusElement = document.getElementById('game-status-message');
    
    if (gameState.gameOver) {
        let message = '';
        if (gameState.winner) {
            message = `Game Over! ${gameState.winner.charAt(0).toUpperCase() + gameState.winner.slice(1)} wins!`;
        } else {
            message = 'Game Over! It\'s a tie!';
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
    
    makeMove(x, y);
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

function showStatus(message, type = 'info') {
    const statusDiv = document.createElement('div');
    statusDiv.className = `status-${type}`;
    statusDiv.textContent = message;
    
    statusMessage.innerHTML = '';
    statusMessage.appendChild(statusDiv);
    
    setTimeout(() => {
        if (statusMessage.firstChild === statusDiv) {
            statusMessage.innerHTML = '';
        }
    }, 3000);
}

function showSpectatorBanner() {
    const banner = document.createElement('div');
    banner.className = 'spectator-banner';
    banner.textContent = 'You are spectating this game';
    
    const gameInfo = document.querySelector('.game-info');
    gameInfo.insertBefore(banner, gameInfo.firstChild);
}

// Export joinPublicRoom for onclick handler
window.joinPublicRoom = joinPublicRoom;