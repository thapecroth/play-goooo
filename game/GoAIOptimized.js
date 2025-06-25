class GoAIOptimized {
    constructor(game) {
        this.game = game;
        this.maxDepth = 3;
        this.evaluationWeights = {
            territory: 1.0,
            captures: 10.0,
            liberties: 0.5,
            influence: 0.3
        };
        
        // Optimization: Pre-allocate arrays for better performance
        this.visitedArray = new Array(game.size * game.size);
        this.groupArray = new Array(game.size * game.size);
        
        // Optimization: Transposition table for memoization
        this.transpositionTable = new Map();
        this.maxTranspositionSize = 100000;
        
        // Optimization: Move ordering cache
        this.moveOrderingCache = new Map();
        
        // Optimization: Liberty cache
        this.libertyCache = new Map();
        
        // Pre-compute neighbor offsets
        this.neighborOffsets = [[-1, 0], [1, 0], [0, -1], [0, 1]];
        
        // Pre-compute star points
        this.starPoints = this.computeStarPoints();
    }
    
    computeStarPoints() {
        const size = this.game.size;
        if (size === 9) {
            return new Set(['2,2', '2,6', '6,2', '6,6', '4,4']);
        } else if (size === 13) {
            return new Set(['3,3', '3,9', '9,3', '9,9', '6,6']);
        } else if (size === 19) {
            return new Set(['3,3', '3,9', '3,15', '9,3', '9,9', '9,15', '15,3', '15,9', '15,15']);
        }
        return new Set();
    }
    
    getBestMove(color) {
        // Clear caches if they're too large
        if (this.transpositionTable.size > this.maxTranspositionSize) {
            this.transpositionTable.clear();
        }
        this.libertyCache.clear();
        
        const validMoves = this.getValidMovesFast(color);
        if (validMoves.length === 0) return null;
        
        // Sort moves by heuristic evaluation for better alpha-beta pruning
        const movesWithScores = validMoves.map(move => ({
            move,
            score: this.evaluateMoveHeuristic(move.x, move.y, color)
        }));
        
        movesWithScores.sort((a, b) => b.score - a.score);
        
        let bestMove = null;
        let bestScore = -Infinity;
        let alpha = -Infinity;
        const beta = Infinity;
        
        for (const { move } of movesWithScores) {
            const score = this.evaluateMove(move.x, move.y, color, alpha, beta);
            if (score > bestScore) {
                bestScore = score;
                bestMove = move;
            }
            alpha = Math.max(alpha, bestScore);
        }
        
        if (bestScore < -50) {
            return null;
        }
        
        return bestMove;
    }
    
    evaluateMoveHeuristic(x, y, color) {
        let score = 0;
        
        // Prefer center positions
        const center = (this.game.size - 1) / 2;
        const distanceToCenter = Math.abs(x - center) + Math.abs(y - center);
        score += 10 / (distanceToCenter + 1);
        
        // Prefer star points
        if (this.starPoints.has(`${x},${y}`)) {
            score += 5;
        }
        
        // Check for potential captures
        const oppositeColor = color === 'black' ? 'white' : 'black';
        for (const [dx, dy] of this.neighborOffsets) {
            const nx = x + dx;
            const ny = y + dy;
            if (nx >= 0 && nx < this.game.size && ny >= 0 && ny < this.game.size) {
                if (this.game.board[ny][nx] === oppositeColor) {
                    const liberties = this.countGroupLibertiesFast(nx, ny);
                    if (liberties === 1) {
                        score += 20;
                    } else if (liberties === 2) {
                        score += 10;
                    }
                }
            }
        }
        
        return score;
    }
    
    evaluateMove(x, y, color, alpha, beta) {
        // Create minimal board state backup
        const boardState = this.createBoardState();
        
        // Make the move
        this.game.currentPlayer = color;
        const moveSuccess = this.game.makeMove(x, y, color);
        
        if (!moveSuccess) {
            this.restoreBoardState(boardState);
            return -Infinity;
        }
        
        // Generate board hash for transposition table
        const boardHash = this.generateBoardHash();
        
        let score;
        if (this.transpositionTable.has(boardHash)) {
            score = this.transpositionTable.get(boardHash);
        } else {
            score = this.minimax(this.maxDepth - 1, alpha, beta, false, color);
            this.transpositionTable.set(boardHash, score);
        }
        
        this.restoreBoardState(boardState);
        
        return score;
    }
    
    minimax(depth, alpha, beta, isMaximizing, aiColor) {
        if (depth === 0 || this.game.gameOver) {
            return this.evaluatePositionFast(aiColor);
        }
        
        const currentColor = this.game.currentPlayer;
        const validMoves = this.getValidMovesFast(currentColor);
        
        if (validMoves.length === 0) {
            const boardState = this.createBoardState();
            this.game.pass(currentColor);
            const score = this.minimax(depth - 1, alpha, beta, !isMaximizing, aiColor);
            this.restoreBoardState(boardState);
            return score;
        }
        
        if (isMaximizing) {
            let maxScore = -Infinity;
            
            for (const move of validMoves) {
                const boardState = this.createBoardState();
                
                if (this.game.makeMove(move.x, move.y, currentColor)) {
                    const score = this.minimax(depth - 1, alpha, beta, false, aiColor);
                    maxScore = Math.max(maxScore, score);
                    alpha = Math.max(alpha, score);
                    
                    this.restoreBoardState(boardState);
                    
                    if (beta <= alpha) break;
                }
            }
            
            return maxScore;
        } else {
            let minScore = Infinity;
            
            for (const move of validMoves) {
                const boardState = this.createBoardState();
                
                if (this.game.makeMove(move.x, move.y, currentColor)) {
                    const score = this.minimax(depth - 1, alpha, beta, true, aiColor);
                    minScore = Math.min(minScore, score);
                    beta = Math.min(beta, score);
                    
                    this.restoreBoardState(boardState);
                    
                    if (beta <= alpha) break;
                }
            }
            
            return minScore;
        }
    }
    
    evaluatePositionFast(aiColor) {
        let score = 0;
        const oppositeColor = aiColor === 'black' ? 'white' : 'black';
        
        // Capture difference
        score += (this.game.captures[aiColor] - this.game.captures[oppositeColor]) * this.evaluationWeights.captures;
        
        // Territory estimation
        const territoryScore = this.estimateTerritoryFast(aiColor);
        score += territoryScore * this.evaluationWeights.territory;
        
        // Liberty difference
        const libertyScore = this.countAllLibertiesFast(aiColor) - this.countAllLibertiesFast(oppositeColor);
        score += libertyScore * this.evaluationWeights.liberties;
        
        // Influence
        const influenceScore = this.calculateInfluenceFast(aiColor);
        score += influenceScore * this.evaluationWeights.influence;
        
        return score;
    }
    
    getValidMovesFast(color) {
        const moves = [];
        const oppositeColor = color === 'black' ? 'white' : 'black';
        
        for (let y = 0; y < this.game.size; y++) {
            for (let x = 0; x < this.game.size; x++) {
                if (this.game.board[y][x] === null) {
                    // Quick check: if it has a liberty or friendly neighbor, it's likely valid
                    let hasLiberty = false;
                    let hasFriendlyNeighbor = false;
                    let canCapture = false;
                    
                    for (const [dx, dy] of this.neighborOffsets) {
                        const nx = x + dx;
                        const ny = y + dy;
                        if (nx >= 0 && nx < this.game.size && ny >= 0 && ny < this.game.size) {
                            const neighbor = this.game.board[ny][nx];
                            if (neighbor === null) {
                                hasLiberty = true;
                            } else if (neighbor === color) {
                                hasFriendlyNeighbor = true;
                            } else if (neighbor === oppositeColor) {
                                // Check if placing here would capture this group
                                const liberties = this.countGroupLibertiesFast(nx, ny);
                                if (liberties === 1) {
                                    canCapture = true;
                                }
                            }
                        }
                    }
                    
                    if (hasLiberty || hasFriendlyNeighbor || canCapture) {
                        moves.push({ x, y });
                    }
                }
            }
        }
        
        return moves;
    }
    
    countGroupLibertiesFast(x, y) {
        const cacheKey = `${x},${y}`;
        if (this.libertyCache.has(cacheKey)) {
            return this.libertyCache.get(cacheKey);
        }
        
        const color = this.game.board[y][x];
        if (!color) return 0;
        
        const group = this.getGroupFast(x, y);
        const liberties = new Set();
        
        for (const stone of group) {
            for (const [dx, dy] of this.neighborOffsets) {
                const nx = stone.x + dx;
                const ny = stone.y + dy;
                if (nx >= 0 && nx < this.game.size && ny >= 0 && ny < this.game.size) {
                    if (this.game.board[ny][nx] === null) {
                        liberties.add(`${nx},${ny}`);
                    }
                }
            }
        }
        
        const libertyCount = liberties.size;
        this.libertyCache.set(cacheKey, libertyCount);
        return libertyCount;
    }
    
    getGroupFast(x, y) {
        const color = this.game.board[y][x];
        if (!color) return [];
        
        const group = [];
        const stack = [{x, y}];
        const visited = new Set();
        
        while (stack.length > 0) {
            const pos = stack.pop();
            const key = `${pos.x},${pos.y}`;
            
            if (visited.has(key)) continue;
            visited.add(key);
            
            if (this.game.board[pos.y][pos.x] === color) {
                group.push(pos);
                
                for (const [dx, dy] of this.neighborOffsets) {
                    const nx = pos.x + dx;
                    const ny = pos.y + dy;
                    if (nx >= 0 && nx < this.game.size && ny >= 0 && ny < this.game.size) {
                        const nkey = `${nx},${ny}`;
                        if (!visited.has(nkey)) {
                            stack.push({x: nx, y: ny});
                        }
                    }
                }
            }
        }
        
        return group;
    }
    
    countAllLibertiesFast(color) {
        let totalLiberties = 0;
        const counted = new Set();
        
        for (let y = 0; y < this.game.size; y++) {
            for (let x = 0; x < this.game.size; x++) {
                if (this.game.board[y][x] === color && !counted.has(`${x},${y}`)) {
                    const liberties = this.countGroupLibertiesFast(x, y);
                    totalLiberties += liberties;
                    
                    // Mark all stones in this group as counted
                    const group = this.getGroupFast(x, y);
                    for (const stone of group) {
                        counted.add(`${stone.x},${stone.y}`);
                    }
                }
            }
        }
        
        return totalLiberties;
    }
    
    estimateTerritoryFast(color) {
        let territory = 0;
        const oppositeColor = color === 'black' ? 'white' : 'black';
        
        // Use a simpler, faster territory estimation
        for (let y = 0; y < this.game.size; y++) {
            for (let x = 0; x < this.game.size; x++) {
                if (this.game.board[y][x] === null) {
                    const influence = this.getPointInfluenceFast(x, y);
                    if (influence[color] > influence[oppositeColor] * 1.5) {
                        territory++;
                    } else if (influence[oppositeColor] > influence[color] * 1.5) {
                        territory--;
                    }
                }
            }
        }
        
        return territory;
    }
    
    getPointInfluenceFast(x, y) {
        const influence = { black: 0, white: 0 };
        const maxDistance = 3; // Reduced from 4 for speed
        
        for (let dy = -maxDistance; dy <= maxDistance; dy++) {
            for (let dx = -maxDistance; dx <= maxDistance; dx++) {
                const nx = x + dx;
                const ny = y + dy;
                
                if (nx >= 0 && nx < this.game.size && ny >= 0 && ny < this.game.size) {
                    const stone = this.game.board[ny][nx];
                    if (stone) {
                        const distance = Math.abs(dx) + Math.abs(dy);
                        influence[stone] += 1 / (distance + 1);
                    }
                }
            }
        }
        
        return influence;
    }
    
    calculateInfluenceFast(color) {
        let influence = 0;
        
        for (let y = 0; y < this.game.size; y++) {
            for (let x = 0; x < this.game.size; x++) {
                if (this.game.board[y][x] === color) {
                    influence += this.getStoneValueFast(x, y);
                }
            }
        }
        
        return influence;
    }
    
    getStoneValueFast(x, y) {
        let value = 1;
        
        // Distance to edge
        const distanceToEdge = Math.min(x, y, this.game.size - 1 - x, this.game.size - 1 - y);
        if (distanceToEdge === 0) {
            value *= 0.7;
        } else if (distanceToEdge === 1) {
            value *= 0.85;
        }
        
        // Star points
        if (this.starPoints.has(`${x},${y}`)) {
            value *= 1.2;
        }
        
        return value;
    }
    
    createBoardState() {
        return {
            board: this.game.board.map(row => [...row]),
            currentPlayer: this.game.currentPlayer,
            captures: { ...this.game.captures },
            passes: this.game.passes,
            lastMove: this.game.lastMove,
            ko: this.game.ko
        };
    }
    
    restoreBoardState(state) {
        this.game.board = state.board;
        this.game.currentPlayer = state.currentPlayer;
        this.game.captures = state.captures;
        this.game.passes = state.passes;
        this.game.lastMove = state.lastMove;
        this.game.ko = state.ko;
    }
    
    generateBoardHash() {
        let hash = 0;
        for (let y = 0; y < this.game.size; y++) {
            for (let x = 0; x < this.game.size; x++) {
                const cell = this.game.board[y][x];
                const value = cell === 'black' ? 1 : cell === 'white' ? 2 : 0;
                hash = hash * 3 + value;
            }
        }
        return hash;
    }
    
    setDepth(depth) {
        this.maxDepth = depth;
    }
}

module.exports = GoAIOptimized;