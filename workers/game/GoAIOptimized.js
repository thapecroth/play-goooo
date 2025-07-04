// Optimized Go AI for Cloudflare Workers
export class GoAIOptimized {
    constructor(game) {
        this.game = game;
        this.maxDepth = 3;
        this.evaluationCache = new Map();
        
        // Pattern weights for evaluation
        this.weights = {
            territory: 10,
            captures: 30,
            liberties: 5,
            influence: 3,
            edgeControl: 2,
            cornerBonus: 15,
            eyeSpace: 20,
            connectionBonus: 8
        };
    }

    setDepth(depth) {
        this.maxDepth = Math.max(1, Math.min(5, depth));
    }

    getBestMove(player) {
        if (!this.game || this.game.gameOver) return null;
        
        this.evaluationCache.clear();
        const opponent = player === 'black' ? 'white' : 'black';
        
        // Get all valid moves
        const validMoves = this.getValidMoves(player);
        
        if (validMoves.length === 0) return null;
        
        // For depth 1, just use evaluation
        if (this.maxDepth === 1) {
            let bestMove = null;
            let bestScore = -Infinity;
            
            for (const move of validMoves) {
                const score = this.evaluateMove(move, player);
                if (score > bestScore) {
                    bestScore = score;
                    bestMove = move;
                }
            }
            
            return bestMove;
        }
        
        // For deeper search, use minimax
        let bestMove = null;
        let bestScore = -Infinity;
        const alpha = -Infinity;
        const beta = Infinity;
        
        // Sort moves by quick evaluation for better pruning
        const scoredMoves = validMoves.map(move => ({
            move,
            quickScore: this.quickEvaluateMove(move, player)
        }));
        scoredMoves.sort((a, b) => b.quickScore - a.quickScore);
        
        // Limit the number of moves to consider at higher depths
        const movesToConsider = this.maxDepth >= 3 ? 
            scoredMoves.slice(0, Math.min(15, scoredMoves.length)) : 
            scoredMoves;
        
        for (const { move } of movesToConsider) {
            const gameState = this.saveGameState();
            
            if (this.game.makeMove(move.x, move.y, player)) {
                const score = this.minimax(this.maxDepth - 1, opponent, alpha, beta, false);
                
                if (score > bestScore) {
                    bestScore = score;
                    bestMove = move;
                }
                
                this.restoreGameState(gameState);
                
                if (bestScore >= beta) break; // Beta cutoff
            }
        }
        
        return bestMove;
    }

    minimax(depth, player, alpha, beta, maximizing) {
        if (depth === 0 || this.game.gameOver) {
            return this.evaluatePosition(maximizing ? player : (player === 'black' ? 'white' : 'black'));
        }
        
        const validMoves = this.getValidMoves(player);
        
        if (validMoves.length === 0) {
            // Must pass
            const gameState = this.saveGameState();
            this.game.pass(player);
            const score = this.minimax(depth - 1, player === 'black' ? 'white' : 'black', alpha, beta, !maximizing);
            this.restoreGameState(gameState);
            return score;
        }
        
        // Limit moves to consider at deeper levels
        const movesToConsider = depth <= 1 ? validMoves : 
            validMoves.slice(0, Math.min(10, validMoves.length));
        
        if (maximizing) {
            let maxScore = -Infinity;
            
            for (const move of movesToConsider) {
                const gameState = this.saveGameState();
                
                if (this.game.makeMove(move.x, move.y, player)) {
                    const score = this.minimax(depth - 1, player === 'black' ? 'white' : 'black', alpha, beta, false);
                    maxScore = Math.max(maxScore, score);
                    alpha = Math.max(alpha, score);
                    
                    this.restoreGameState(gameState);
                    
                    if (beta <= alpha) break; // Alpha-beta pruning
                }
            }
            
            return maxScore;
        } else {
            let minScore = Infinity;
            
            for (const move of movesToConsider) {
                const gameState = this.saveGameState();
                
                if (this.game.makeMove(move.x, move.y, player)) {
                    const score = this.minimax(depth - 1, player === 'black' ? 'white' : 'black', alpha, beta, true);
                    minScore = Math.min(minScore, score);
                    beta = Math.min(beta, score);
                    
                    this.restoreGameState(gameState);
                    
                    if (beta <= alpha) break; // Alpha-beta pruning
                }
            }
            
            return minScore;
        }
    }

    getValidMoves(player) {
        const moves = [];
        const center = Math.floor(this.game.boardSize / 2);
        
        // Prioritize center and strategic points
        const priorityPoints = [
            {x: center, y: center},
            {x: 2, y: 2}, {x: this.game.boardSize - 3, y: 2},
            {x: 2, y: this.game.boardSize - 3}, {x: this.game.boardSize - 3, y: this.game.boardSize - 3},
            {x: center, y: 2}, {x: center, y: this.game.boardSize - 3},
            {x: 2, y: center}, {x: this.game.boardSize - 3, y: center}
        ];
        
        // Check priority points first
        for (const point of priorityPoints) {
            if (point.x < this.game.boardSize && point.y < this.game.boardSize) {
                if (this.game.isValidMove(point.x, point.y, player)) {
                    moves.push(point);
                }
            }
        }
        
        // Then check all other points
        for (let y = 0; y < this.game.boardSize; y++) {
            for (let x = 0; x < this.game.boardSize; x++) {
                if (!priorityPoints.some(p => p.x === x && p.y === y)) {
                    if (this.game.isValidMove(x, y, player)) {
                        moves.push({x, y});
                    }
                }
            }
        }
        
        return moves;
    }

    quickEvaluateMove(move, player) {
        // Quick heuristic evaluation without making the move
        let score = 0;
        
        // Prefer corners and edges early
        if ((move.x === 2 || move.x === this.game.boardSize - 3) && 
            (move.y === 2 || move.y === this.game.boardSize - 3)) {
            score += 20; // Star points
        }
        
        // Check adjacent stones
        const color = player === 'black' ? 1 : 2;
        const neighbors = this.game.getNeighbors(move.x, move.y);
        
        for (const neighbor of neighbors) {
            const index = neighbor.y * this.game.boardSize + neighbor.x;
            if (this.game.board[index] === color) {
                score += 5; // Adjacent to friendly stone
            } else if (this.game.board[index] !== 0) {
                score += 3; // Adjacent to enemy stone (attacking)
            }
        }
        
        return score;
    }

    evaluateMove(move, player) {
        const gameState = this.saveGameState();
        let score = 0;
        
        if (this.game.makeMove(move.x, move.y, player)) {
            score = this.evaluatePosition(player);
            this.restoreGameState(gameState);
        }
        
        return score;
    }

    evaluatePosition(player) {
        const cacheKey = this.getBoardHash() + player;
        if (this.evaluationCache.has(cacheKey)) {
            return this.evaluationCache.get(cacheKey);
        }
        
        const playerColor = player === 'black' ? 1 : 2;
        const opponentColor = playerColor === 1 ? 2 : 1;
        
        let score = 0;
        
        // Territory estimation
        const territory = this.estimateTerritory();
        score += this.weights.territory * (territory[playerColor] - territory[opponentColor]);
        
        // Captures
        const captures = player === 'black' ? 
            this.game.captures.white - this.game.captures.black :
            this.game.captures.black - this.game.captures.white;
        score += this.weights.captures * captures;
        
        // Liberty count
        const liberties = this.countLiberties(playerColor) - this.countLiberties(opponentColor);
        score += this.weights.liberties * liberties;
        
        // Influence
        const influence = this.calculateInfluence(playerColor) - this.calculateInfluence(opponentColor);
        score += this.weights.influence * influence;
        
        // Eye space potential
        const eyeSpace = this.countPotentialEyes(playerColor) - this.countPotentialEyes(opponentColor);
        score += this.weights.eyeSpace * eyeSpace;
        
        this.evaluationCache.set(cacheKey, score);
        return score;
    }

    estimateTerritory() {
        const territory = { 1: 0, 2: 0 };
        const visited = new Set();
        
        for (let y = 0; y < this.game.boardSize; y++) {
            for (let x = 0; x < this.game.boardSize; x++) {
                const key = `${x},${y}`;
                const index = y * this.game.boardSize + x;
                
                if (!visited.has(key) && this.game.board[index] === 0) {
                    const region = this.getEmptyRegion(x, y, visited);
                    const owner = this.estimateRegionOwner(region);
                    
                    if (owner === 1 || owner === 2) {
                        territory[owner] += region.length;
                    }
                }
            }
        }
        
        return territory;
    }

    getEmptyRegion(startX, startY, visited) {
        const region = [];
        const stack = [{x: startX, y: startY}];
        
        while (stack.length > 0) {
            const current = stack.pop();
            const key = `${current.x},${current.y}`;
            
            if (visited.has(key)) continue;
            
            const index = current.y * this.game.boardSize + current.x;
            if (this.game.board[index] === 0) {
                visited.add(key);
                region.push(current);
                
                const neighbors = this.game.getNeighbors(current.x, current.y);
                for (const neighbor of neighbors) {
                    stack.push(neighbor);
                }
            }
        }
        
        return region;
    }

    estimateRegionOwner(region) {
        const influence = { 1: 0, 2: 0 };
        
        for (const point of region) {
            const neighbors = this.game.getNeighbors(point.x, point.y);
            for (const neighbor of neighbors) {
                const index = neighbor.y * this.game.boardSize + neighbor.x;
                const color = this.game.board[index];
                if (color === 1 || color === 2) {
                    influence[color]++;
                }
            }
        }
        
        if (influence[1] > influence[2] * 1.5) return 1;
        if (influence[2] > influence[1] * 1.5) return 2;
        return 0;
    }

    countLiberties(color) {
        let totalLiberties = 0;
        const counted = new Set();
        
        for (let y = 0; y < this.game.boardSize; y++) {
            for (let x = 0; x < this.game.boardSize; x++) {
                const index = y * this.game.boardSize + x;
                if (this.game.board[index] === color) {
                    const group = this.game.getGroup(x, y);
                    const groupKey = group.map(p => `${p.x},${p.y}`).sort().join('|');
                    
                    if (!counted.has(groupKey)) {
                        counted.add(groupKey);
                        const liberties = this.getGroupLiberties(group);
                        totalLiberties += liberties.size;
                    }
                }
            }
        }
        
        return totalLiberties;
    }

    getGroupLiberties(group) {
        const liberties = new Set();
        
        for (const stone of group) {
            const neighbors = this.game.getNeighbors(stone.x, stone.y);
            for (const neighbor of neighbors) {
                const index = neighbor.y * this.game.boardSize + neighbor.x;
                if (this.game.board[index] === 0) {
                    liberties.add(`${neighbor.x},${neighbor.y}`);
                }
            }
        }
        
        return liberties;
    }

    calculateInfluence(color) {
        let influence = 0;
        
        for (let y = 0; y < this.game.boardSize; y++) {
            for (let x = 0; x < this.game.boardSize; x++) {
                const index = y * this.game.boardSize + x;
                if (this.game.board[index] === color) {
                    // Each stone exerts influence on nearby empty points
                    for (let dy = -2; dy <= 2; dy++) {
                        for (let dx = -2; dx <= 2; dx++) {
                            const nx = x + dx;
                            const ny = y + dy;
                            
                            if (nx >= 0 && nx < this.game.boardSize && 
                                ny >= 0 && ny < this.game.boardSize) {
                                const nindex = ny * this.game.boardSize + nx;
                                if (this.game.board[nindex] === 0) {
                                    const distance = Math.abs(dx) + Math.abs(dy);
                                    influence += 1 / (distance + 1);
                                }
                            }
                        }
                    }
                }
            }
        }
        
        return influence;
    }

    countPotentialEyes(color) {
        let eyeCount = 0;
        
        for (let y = 1; y < this.game.boardSize - 1; y++) {
            for (let x = 1; x < this.game.boardSize - 1; x++) {
                const index = y * this.game.boardSize + x;
                if (this.game.board[index] === 0) {
                    // Check if surrounded by friendly stones
                    const neighbors = this.game.getNeighbors(x, y);
                    let friendlyCount = 0;
                    
                    for (const neighbor of neighbors) {
                        const nindex = neighbor.y * this.game.boardSize + neighbor.x;
                        if (this.game.board[nindex] === color) {
                            friendlyCount++;
                        }
                    }
                    
                    if (friendlyCount >= 3) {
                        eyeCount++;
                    }
                }
            }
        }
        
        return eyeCount;
    }

    saveGameState() {
        return {
            board: [...this.game.board],
            currentPlayer: this.game.currentPlayer,
            lastMove: this.game.lastMove ? {...this.game.lastMove} : null,
            captures: {...this.game.captures},
            passCount: this.game.passCount,
            koPoint: this.game.koPoint ? {...this.game.koPoint} : null,
            gameOver: this.game.gameOver,
            winner: this.game.winner,
            resigned: this.game.resigned
        };
    }

    restoreGameState(state) {
        this.game.board = [...state.board];
        this.game.currentPlayer = state.currentPlayer;
        this.game.lastMove = state.lastMove ? {...state.lastMove} : null;
        this.game.captures = {...state.captures};
        this.game.passCount = state.passCount;
        this.game.koPoint = state.koPoint ? {...state.koPoint} : null;
        this.game.gameOver = state.gameOver;
        this.game.winner = state.winner;
        this.game.resigned = state.resigned;
    }

    getBoardHash() {
        return this.game.board.join('');
    }
}