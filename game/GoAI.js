class GoAI {
    constructor(game) {
        this.game = game;
        this.maxDepth = 3;
        this.evaluationWeights = {
            territory: 1.0,
            captures: 10.0,
            liberties: 0.5,
            influence: 0.3
        };
    }
    
    getBestMove(color) {
        const validMoves = this.game.getValidMoves(color);
        if (validMoves.length === 0) return null;
        
        let bestMove = null;
        let bestScore = -Infinity;
        
        for (const move of validMoves) {
            const score = this.evaluateMove(move.x, move.y, color);
            if (score > bestScore) {
                bestScore = score;
                bestMove = move;
            }
        }
        
        if (bestScore < -50) {
            return null;
        }
        
        return bestMove;
    }
    
    evaluateMove(x, y, color) {
        const originalBoard = this.game.copyBoard();
        const originalCaptures = { ...this.game.captures };
        const originalPlayer = this.game.currentPlayer;
        const originalLastMove = this.game.lastMove;
        const originalKo = this.game.ko;
        
        this.game.currentPlayer = color;
        const moveSuccess = this.game.makeMove(x, y, color);
        
        if (!moveSuccess) {
            this.game.board = originalBoard;
            this.game.captures = originalCaptures;
            this.game.currentPlayer = originalPlayer;
            this.game.lastMove = originalLastMove;
            this.game.ko = originalKo;
            return -Infinity;
        }
        
        const score = this.minimax(this.maxDepth - 1, -Infinity, Infinity, false, color);
        
        this.game.board = originalBoard;
        this.game.captures = originalCaptures;
        this.game.currentPlayer = originalPlayer;
        this.game.lastMove = originalLastMove;
        this.game.ko = originalKo;
        
        return score;
    }
    
    minimax(depth, alpha, beta, isMaximizing, aiColor) {
        if (depth === 0 || this.game.gameOver) {
            return this.evaluatePosition(aiColor);
        }
        
        const currentColor = this.game.currentPlayer;
        const validMoves = this.game.getValidMoves(currentColor);
        
        if (validMoves.length === 0) {
            const originalPasses = this.game.passes;
            this.game.pass(currentColor);
            const score = this.minimax(depth - 1, alpha, beta, !isMaximizing, aiColor);
            this.game.passes = originalPasses;
            this.game.currentPlayer = currentColor;
            return score;
        }
        
        if (isMaximizing) {
            let maxScore = -Infinity;
            
            for (const move of validMoves) {
                const boardCopy = this.game.copyBoard();
                const capturesCopy = { ...this.game.captures };
                const playerCopy = this.game.currentPlayer;
                
                if (this.game.makeMove(move.x, move.y, currentColor)) {
                    const score = this.minimax(depth - 1, alpha, beta, false, aiColor);
                    maxScore = Math.max(maxScore, score);
                    alpha = Math.max(alpha, score);
                    
                    this.game.board = boardCopy;
                    this.game.captures = capturesCopy;
                    this.game.currentPlayer = playerCopy;
                    
                    if (beta <= alpha) break;
                }
            }
            
            return maxScore;
        } else {
            let minScore = Infinity;
            
            for (const move of validMoves) {
                const boardCopy = this.game.copyBoard();
                const capturesCopy = { ...this.game.captures };
                const playerCopy = this.game.currentPlayer;
                
                if (this.game.makeMove(move.x, move.y, currentColor)) {
                    const score = this.minimax(depth - 1, alpha, beta, true, aiColor);
                    minScore = Math.min(minScore, score);
                    beta = Math.min(beta, score);
                    
                    this.game.board = boardCopy;
                    this.game.captures = capturesCopy;
                    this.game.currentPlayer = playerCopy;
                    
                    if (beta <= alpha) break;
                }
            }
            
            return minScore;
        }
    }
    
    evaluatePosition(aiColor) {
        let score = 0;
        const oppositeColor = aiColor === 'black' ? 'white' : 'black';
        
        score += (this.game.captures[aiColor] - this.game.captures[oppositeColor]) * this.evaluationWeights.captures;
        
        const territoryScore = this.estimateTerritory(aiColor);
        score += territoryScore * this.evaluationWeights.territory;
        
        const libertyScore = this.countLiberties(aiColor) - this.countLiberties(oppositeColor);
        score += libertyScore * this.evaluationWeights.liberties;
        
        const influenceScore = this.calculateInfluence(aiColor);
        score += influenceScore * this.evaluationWeights.influence;
        
        return score;
    }
    
    estimateTerritory(color) {
        let territory = 0;
        const oppositeColor = color === 'black' ? 'white' : 'black';
        
        for (let y = 0; y < this.game.size; y++) {
            for (let x = 0; x < this.game.size; x++) {
                if (this.game.board[y][x] === null) {
                    const influence = this.getPointInfluence(x, y);
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
    
    getPointInfluence(x, y) {
        const influence = { black: 0, white: 0 };
        const maxDistance = 4;
        
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
    
    countLiberties(color) {
        let totalLiberties = 0;
        const counted = new Set();
        
        for (let y = 0; y < this.game.size; y++) {
            for (let x = 0; x < this.game.size; x++) {
                if (this.game.board[y][x] === color && !counted.has(`${x},${y}`)) {
                    const group = this.game.getGroup(x, y);
                    const liberties = new Set();
                    
                    for (const stone of group) {
                        counted.add(`${stone.x},${stone.y}`);
                        const neighbors = this.game.getNeighbors(stone.x, stone.y);
                        
                        for (const n of neighbors) {
                            if (this.game.board[n.y][n.x] === null) {
                                liberties.add(`${n.x},${n.y}`);
                            }
                        }
                    }
                    
                    totalLiberties += liberties.size;
                }
            }
        }
        
        return totalLiberties;
    }
    
    calculateInfluence(color) {
        let influence = 0;
        
        for (let y = 0; y < this.game.size; y++) {
            for (let x = 0; x < this.game.size; x++) {
                if (this.game.board[y][x] === color) {
                    influence += this.getStoneValue(x, y);
                }
            }
        }
        
        return influence;
    }
    
    getStoneValue(x, y) {
        let value = 1;
        
        const distanceToEdge = Math.min(x, y, this.game.size - 1 - x, this.game.size - 1 - y);
        if (distanceToEdge === 0) {
            value *= 0.7;
        } else if (distanceToEdge === 1) {
            value *= 0.85;
        }
        
        if ((x === 3 || x === this.game.size - 4) && (y === 3 || y === this.game.size - 4)) {
            value *= 1.2;
        }
        
        return value;
    }
}

module.exports = GoAI;