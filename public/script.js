const socket = io();

let gameState = null;
let boardSize = 9;
let isAiThinking = false;

const elements = {
    board: document.getElementById('board'),
    newGameBtn: document.getElementById('new-game'),
    passBtn: document.getElementById('pass'),
    resignBtn: document.getElementById('resign'),
    currentPlayer: document.getElementById('current-player'),
    blackCaptures: document.getElementById('black-captures'),
    whiteCaptures: document.getElementById('white-captures'),
    statusMessage: document.getElementById('status-message'),
    boardSizeSelect: document.getElementById('board-size'),
    aiTypeSelect: document.getElementById('ai-type'),
    aiDepthSelect: document.getElementById('ai-depth'),
    dqnModelSelect: document.getElementById('dqn-model'),
    alphaGoModelSelect: document.getElementById('alpha-go-model'),
    classicAiSettings: document.querySelector('.classic-ai-settings'),
    dqnAiSettings: document.querySelector('.dqn-ai-settings'),
    alphaGoAiSettings: document.querySelector('.alpha-go-ai-settings'),
    modelDetails: document.getElementById('model-details'),
    showQValuesCheckbox: document.getElementById('show-q-values'),
    qValueLegend: document.getElementById('q-value-legend'),
    // New MCTS elements
    classicEngineSelect: document.getElementById('classic-engine'),
    classicAlgorithmSelect: document.getElementById('classic-algorithm'),
    minimaxSettings: document.querySelector('.minimax-settings'),
    mctsSettings: document.querySelector('.mcts-settings'),
    mctsSimulations: document.getElementById('mcts-simulations'),
    mctsSimulationsValue: document.getElementById('mcts-simulations-value'),
    mctsExploration: document.getElementById('mcts-exploration'),
    mctsExplorationValue: document.getElementById('mcts-exploration-value'),
    mctsTimeLimit: document.getElementById('mcts-time-limit'),
    mctsTimeLimitValue: document.getElementById('mcts-time-limit-value')
};

elements.newGameBtn.addEventListener('click', startNewGame);
elements.passBtn.addEventListener('click', pass);
elements.resignBtn.addEventListener('click', resign);
elements.boardSizeSelect.addEventListener('change', (e) => {
    boardSize = parseInt(e.target.value);
    startNewGame();
});

elements.aiTypeSelect.addEventListener('change', (e) => {
    const aiType = e.target.value;
    toggleAiSettings(aiType);
    socket.emit('setAiType', aiType);
    showStatus(`AI type set to ${aiType}`, 'info');
    
    // If switching to DQN and Q-values are enabled, request them
    if (aiType === 'dqn' && elements.showQValuesCheckbox.checked) {
        // Give the server a moment to switch AI type
        setTimeout(() => {
            requestQValues();
        }, 100);
    } else if (aiType !== 'dqn') {
        // Clear Q-values if switching away from DQN
        clearQValueVisualization();
    }
});

elements.aiDepthSelect.addEventListener('change', (e) => {
    const aiDepth = parseInt(e.target.value);
    socket.emit('setAiDepth', aiDepth);
    showStatus(`AI depth set to ${aiDepth}`, 'info');
});

elements.dqnModelSelect.addEventListener('change', (e) => {
    const modelName = e.target.value;
    if (modelName) {
        socket.emit('setDqnModel', modelName);
        updateModelDetails(modelName);
        showStatus(`DQN model set to ${modelName}`, 'info');
    }
});

elements.alphaGoModelSelect.addEventListener('change', (e) => {
    const modelName = e.target.value;
    if (modelName) {
        socket.emit('setAlphaGoModel', modelName);
        showStatus(`AlphaGo model set to ${modelName}`, 'info');
    }
});

elements.showQValuesCheckbox.addEventListener('change', (e) => {
    const showQValues = e.target.checked;
    elements.qValueLegend.style.display = showQValues ? 'flex' : 'none';
    
    if (showQValues) {
        if (elements.aiTypeSelect.value === 'dqn') {
            requestQValues();
            showStatus('Requesting Q-values...', 'info');
        } else {
            showStatus('Q-values only available for DQN AI type', 'warning');
            e.target.checked = false;
            elements.qValueLegend.style.display = 'none';
        }
    } else {
        clearQValueVisualization();
    }
});

// Classic engine selection
elements.classicEngineSelect.addEventListener('change', (e) => {
    const engine = e.target.value;
    socket.emit('setClassicEngine', engine);
    showStatus(`Classic engine set to ${engine}`, 'info');
});

// Classic algorithm selection
elements.classicAlgorithmSelect.addEventListener('change', (e) => {
    const algorithm = e.target.value;
    toggleClassicAlgorithmSettings(algorithm);
    socket.emit('setClassicAlgorithm', algorithm);
    showStatus(`Classic algorithm set to ${algorithm.toUpperCase()}`, 'info');
});

// MCTS parameter controls
elements.mctsSimulations.addEventListener('input', (e) => {
    const value = e.target.value;
    elements.mctsSimulationsValue.textContent = value;
    socket.emit('setMctsParams', { simulations: parseInt(value) });
});

elements.mctsExploration.addEventListener('input', (e) => {
    const value = e.target.value;
    elements.mctsExplorationValue.textContent = value;
    socket.emit('setMctsParams', { exploration: parseFloat(value) });
});

elements.mctsTimeLimit.addEventListener('input', (e) => {
    const value = e.target.value;
    elements.mctsTimeLimitValue.textContent = value;
    socket.emit('setMctsParams', { timeLimit: parseInt(value) });
});

socket.on('gameState', updateGameState);
socket.on('invalidMove', handleInvalidMove);
socket.on('error', (error) => showStatus(error, 'error'));

function startNewGame() {
    socket.emit('newGame', boardSize);
    showStatus('Starting new game...', 'info');
}

function makeMove(x, y) {
    if (!gameState || gameState.gameOver || gameState.currentPlayer !== 'black' || isAiThinking) {
        return;
    }
    
    socket.emit('makeMove', { x, y });
    startAiThinking();
}

function pass() {
    if (!gameState || gameState.gameOver || gameState.currentPlayer !== 'black' || isAiThinking) {
        return;
    }
    
    socket.emit('pass');
    startAiThinking();
}

function resign() {
    if (!gameState || gameState.gameOver) {
        return;
    }
    
    if (confirm('Are you sure you want to resign?')) {
        socket.emit('resign');
    }
}

function updateGameState(newGameState) {
    gameState = newGameState;
    stopAiThinking();
    renderBoard();
    updateUI();
}

function renderBoard() {
    if (!gameState) return;
    
    const board = gameState.board;
    const size = board.length;
    
    elements.board.innerHTML = '';
    elements.board.className = `board size-${size}`;
    
    for (let y = 0; y < size; y++) {
        for (let x = 0; x < size; x++) {
            const intersection = createIntersection(x, y, size);
            elements.board.appendChild(intersection);
        }
    }
}

function createIntersection(x, y, size) {
    const intersection = document.createElement('div');
    intersection.className = getIntersectionClasses(x, y, size);
    intersection.addEventListener('click', () => makeMove(x, y));
    
    if (isStarPoint(x, y, size)) {
        const star = document.createElement('div');
        star.className = 'star';
        intersection.appendChild(star);
    }
    
    const stone = gameState.board[y][x];
    if (stone) {
        const stoneElement = document.createElement('div');
        stoneElement.className = `stone ${stone}`;
        
        if (gameState.lastMove && gameState.lastMove.x === x && gameState.lastMove.y === y) {
            stoneElement.className += ' last-move';
        }
        
        intersection.appendChild(stoneElement);
    }
    
    return intersection;
}

function getIntersectionClasses(x, y, size) {
    let classes = ['intersection'];
    
    if (y === 0) classes.push('edge-top');
    if (y === size - 1) classes.push('edge-bottom');
    if (x === 0) classes.push('edge-left');
    if (x === size - 1) classes.push('edge-right');
    
    if (isStarPoint(x, y, size)) {
        classes.push('star-point');
    }
    
    return classes.join(' ');
}

function isStarPoint(x, y, size) {
    const starPoints = getStarPoints(size);
    return starPoints.some(point => point.x === x && point.y === y);
}

function getStarPoints(size) {
    const points = [];
    
    if (size === 9) {
        points.push(
            { x: 2, y: 2 }, { x: 6, y: 2 },
            { x: 4, y: 4 },
            { x: 2, y: 6 }, { x: 6, y: 6 }
        );
    } else if (size === 13) {
        points.push(
            { x: 3, y: 3 }, { x: 9, y: 3 },
            { x: 6, y: 6 },
            { x: 3, y: 9 }, { x: 9, y: 9 }
        );
    } else if (size === 19) {
        points.push(
            { x: 3, y: 3 }, { x: 9, y: 3 }, { x: 15, y: 3 },
            { x: 3, y: 9 }, { x: 9, y: 9 }, { x: 15, y: 9 },
            { x: 3, y: 15 }, { x: 9, y: 15 }, { x: 15, y: 15 }
        );
    }
    
    return points;
}

function updateUI() {
    if (!gameState) return;
    
    elements.currentPlayer.textContent = gameState.currentPlayer === 'black' ? 'Black (You)' : 'White (AI)';
    elements.blackCaptures.textContent = gameState.captures.black;
    elements.whiteCaptures.textContent = gameState.captures.white;
    
    elements.passBtn.disabled = gameState.gameOver || gameState.currentPlayer !== 'black';
    elements.resignBtn.disabled = gameState.gameOver;
    
    if (gameState.gameOver) {
        let message;
        if (gameState.winner) {
            const winnerName = gameState.winner === 'black' ? 'Black (You)' : 'White (AI)';
            message = `Game Over! ${winnerName} wins!`;
            
            if (gameState.score) {
                message += ` Final Score - Black: ${gameState.score.black.toFixed(1)}, White: ${gameState.score.white.toFixed(1)}`;
            }
        } else {
            message = 'Game Over! It\'s a tie!';
        }
        showStatus(message, gameState.winner === 'black' ? 'success' : 'info');
    } else {
        const currentPlayerName = gameState.currentPlayer === 'black' ? 'Your' : 'AI\'s';
        showStatus(`${currentPlayerName} turn`, 'info');
    }
}

function handleInvalidMove(data) {
    const intersection = elements.board.children[data.y * gameState.board.length + data.x];
    if (intersection) {
        intersection.classList.add('invalid-move');
        setTimeout(() => {
            intersection.classList.remove('invalid-move');
        }, 500);
    }
    showStatus('Invalid move! Try again.', 'error');
}

function showStatus(message, type = 'info') {
    elements.statusMessage.textContent = message;
    elements.statusMessage.className = `status-${type}`;
    
    setTimeout(() => {
        if (elements.statusMessage.textContent === message) {
            elements.statusMessage.textContent = '';
            elements.statusMessage.className = '';
        }
    }, 3000);
}

function startAiThinking() {
    isAiThinking = true;
    elements.board.classList.add('ai-computing');
    
    const thinkingMessage = document.createElement('div');
    thinkingMessage.className = 'ai-thinking';
    thinkingMessage.innerHTML = `
        AI is thinking
        <div class="thinking-dots">
            <div class="thinking-dot"></div>
            <div class="thinking-dot"></div>
            <div class="thinking-dot"></div>
        </div>
    `;
    
    elements.statusMessage.innerHTML = '';
    elements.statusMessage.appendChild(thinkingMessage);
    elements.statusMessage.className = 'status-thinking';
    
    elements.passBtn.disabled = true;
    elements.resignBtn.disabled = true;
}

function stopAiThinking() {
    isAiThinking = false;
    elements.board.classList.remove('ai-computing');
    
    elements.passBtn.disabled = gameState?.gameOver || gameState?.currentPlayer !== 'black';
    elements.resignBtn.disabled = gameState?.gameOver;
}

function toggleAiSettings(aiType) {
    elements.classicAiSettings.style.display = 'none';
    elements.dqnAiSettings.style.display = 'none';
    elements.alphaGoAiSettings.style.display = 'none';

    if (aiType === 'dqn') {
        elements.dqnAiSettings.style.display = 'block';
        loadAvailableModels('dqn');
        
        // Enable Q-values visualization if checked
        if (elements.showQValuesCheckbox.checked) {
            setTimeout(() => requestQValues(), 200);
        }
    } else if (aiType === 'alpha_go') {
        elements.alphaGoAiSettings.style.display = 'block';
        loadAvailableModels('alpha_go');
    } else {
        elements.classicAiSettings.style.display = 'block';
        
        // Disable Q-values visualization
        elements.showQValuesCheckbox.checked = false;
        elements.qValueLegend.style.display = 'none';
        clearQValueVisualization();
    }
}

function toggleClassicAlgorithmSettings(algorithm) {
    if (algorithm === 'mcts') {
        elements.minimaxSettings.style.display = 'none';
        elements.mctsSettings.style.display = 'block';
    } else {
        elements.minimaxSettings.style.display = 'block';
        elements.mctsSettings.style.display = 'none';
    }
}

function loadAvailableModels(modelType) {
    socket.emit('getAvailableModels', { modelType });
}

function updateModelDetails(modelName) {
    const option = elements.dqnModelSelect.querySelector(`option[value="${modelName}"]`);
    if (option && option.dataset.details) {
        elements.modelDetails.textContent = option.dataset.details;
    }
}

// Socket event handlers for AI management
socket.on('availableModels', (data) => {
    const { modelType, models } = data;
    let selectElement;

    if (modelType === 'dqn') {
        selectElement = elements.dqnModelSelect;
    } else if (modelType === 'alpha_go') {
        selectElement = elements.alphaGoModelSelect;
    } else {
        return;
    }

    selectElement.innerHTML = '<option value="">Select a model...</option>';
    
    models.forEach(model => {
        const option = document.createElement('option');
        option.value = model.name;
        option.textContent = `${model.name} (${model.boardSize}x${model.boardSize}, depth ${model.depth})`;
        option.dataset.details = `Board: ${model.boardSize}x${model.boardSize}, Depth: ${model.depth}, Episodes: ${model.episodes || 'Unknown'}`;
        selectElement.appendChild(option);
    });
    
    if (models.length === 0) {
        const option = document.createElement('option');
        option.value = '';
        option.textContent = 'No trained models available';
        selectElement.appendChild(option);
    }
});

socket.on('modelLoaded', (modelInfo) => {
    showStatus(`Model loaded: ${modelInfo.name}`, 'success');
    elements.modelDetails.textContent = `Loaded: ${modelInfo.boardSize}x${modelInfo.boardSize}, depth ${modelInfo.depth}`;
    
    // If Q-values visualization is enabled and we're in DQN mode, request Q-values
    if (elements.showQValuesCheckbox.checked && elements.aiTypeSelect.value === 'dqn') {
        setTimeout(() => {
            requestQValues();
        }, 100);
    }
});

socket.on('modelError', (error) => {
    showStatus(`Model error: ${error}`, 'error');
});

// Q-value visualization
let currentQValues = null;

function requestQValues() {
    if (gameState && elements.aiTypeSelect.value === 'dqn') {
        socket.emit('getQValues', { color: 'black' });
    }
}

function clearQValueVisualization() {
    const overlays = elements.board.querySelectorAll('.q-value-overlay, .q-value-text');
    overlays.forEach(overlay => overlay.remove());
    currentQValues = null;
}

function renderQValues(qValues) {
    if (!gameState || !qValues) return;
    
    // Clear existing overlays
    clearQValueVisualization();
    
    // Store current Q-values
    currentQValues = qValues;
    
    // Find min and max Q-values for normalization
    let minQ = Infinity;
    let maxQ = -Infinity;
    
    for (let y = 0; y < qValues.length; y++) {
        for (let x = 0; x < qValues[y].length; x++) {
            if (qValues[y][x] !== null) {
                minQ = Math.min(minQ, qValues[y][x]);
                maxQ = Math.max(maxQ, qValues[y][x]);
            }
        }
    }
    
    // Render overlays for each intersection
    for (let y = 0; y < qValues.length; y++) {
        for (let x = 0; x < qValues[y].length; x++) {
            const qValue = qValues[y][x];
            if (qValue !== null) {
                const intersectionIndex = y * gameState.board.length + x;
                const intersection = elements.board.children[intersectionIndex];
                
                if (intersection) {
                    // Create overlay
                    const overlay = document.createElement('div');
                    overlay.className = 'q-value-overlay';
                    
                    // Normalize Q-value to 0-1 range
                    const normalizedValue = maxQ > minQ ? (qValue - minQ) / (maxQ - minQ) : 0.5;
                    
                    // Create color from red (low) to green (high)
                    const red = Math.round(255 * (1 - normalizedValue));
                    const green = Math.round(255 * normalizedValue);
                    const blue = 0;
                    
                    overlay.style.backgroundColor = `rgba(${red}, ${green}, ${blue}, 0.6)`;
                    
                    // Create text element
                    const text = document.createElement('div');
                    text.className = 'q-value-text';
                    text.textContent = qValue.toFixed(2);
                    
                    // Add to intersection
                    intersection.appendChild(overlay);
                    intersection.appendChild(text);
                }
            }
        }
    }
}

socket.on('qValues', (data) => {
    if (elements.showQValuesCheckbox.checked) {
        renderQValues(data.qValues);
    }
});

socket.on('qValuesError', (error) => {
    console.error('Q-values error:', error);
    clearQValueVisualization();
    
    // Provide more helpful error messages
    if (error.includes('not available for current AI type')) {
        showStatus('Q-values only available for DQN AI type', 'error');
    } else if (error.includes('No DQN model loaded')) {
        showStatus('Please select a DQN model first', 'error');
    } else {
        showStatus(`Q-values error: ${error}`, 'error');
    }
});

// Update Q-values when game state changes
function updateGameStateWithQValues(newGameState) {
    updateGameState(newGameState);
    
    // Request new Q-values if visualization is enabled
    if (elements.showQValuesCheckbox.checked && elements.aiTypeSelect.value === 'dqn') {
        setTimeout(() => requestQValues(), 100); // Small delay to ensure board is rendered
    }
}

// Override the original socket handler
socket.off('gameState');
socket.on('gameState', updateGameStateWithQValues);

startNewGame();