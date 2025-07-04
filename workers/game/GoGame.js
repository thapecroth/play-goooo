// Go Game logic for Cloudflare Workers
export class GoGame {
    constructor(boardSize = 9) {
        this.boardSize = boardSize;
        this.board = Array(boardSize * boardSize).fill(0); // 0: empty, 1: black, 2: white
        this.currentPlayer = 'black';
        this.lastMove = null;
        this.captures = { black: 0, white: 0 };
        this.passCount = 0;
        this.moveHistory = [];
        this.koPoint = null; // Stores the point that cannot be played due to ko rule
        this.gameOver = false;
        this.winner = null;
        this.resigned = null;
    }

    makeMove(x, y, player) {
        if (this.gameOver) return false;
        if (player !== this.currentPlayer) return false;
        
        const index = y * this.boardSize + x;
        if (this.board[index] !== 0) return false;
        
        // Check ko rule
        if (this.koPoint && this.koPoint.x === x && this.koPoint.y === y) {
            return false;
        }
        
        // Place stone
        const color = player === 'black' ? 1 : 2;
        this.board[index] = color;
        
        // Check for captures
        const capturedStones = this.checkCaptures(x, y, color);
        const totalCaptured = capturedStones.length;
        
        // Check for self-capture (suicide)
        if (totalCaptured === 0 && !this.hasLiberties(x, y, color)) {
            this.board[index] = 0; // Remove the stone
            return false;
        }
        
        // Update captures
        if (totalCaptured > 0) {
            if (player === 'black') {
                this.captures.black += totalCaptured;
            } else {
                this.captures.white += totalCaptured;
            }
        }
        
        // Check for ko
        this.koPoint = null;
        if (totalCaptured === 1 && this.isSingleStoneCapture(capturedStones[0])) {
            const capturedStone = capturedStones[0];
            // Check if this creates a ko situation
            if (this.wouldBeKo(capturedStone.x, capturedStone.y, color === 1 ? 2 : 1)) {
                this.koPoint = { x: capturedStone.x, y: capturedStone.y };
            }
        }
        
        // Update game state
        this.lastMove = { x, y, player, pass: false };
        this.moveHistory.push({ ...this.lastMove, board: [...this.board] });
        this.passCount = 0;
        this.currentPlayer = this.currentPlayer === 'black' ? 'white' : 'black';
        
        return true;
    }

    pass(player) {
        if (this.gameOver) return false;
        if (player !== this.currentPlayer) return false;
        
        this.passCount++;
        this.lastMove = { player, pass: true };
        this.moveHistory.push({ ...this.lastMove, board: [...this.board] });
        this.currentPlayer = this.currentPlayer === 'black' ? 'white' : 'black';
        
        // Two consecutive passes end the game
        if (this.passCount >= 2) {
            this.endGame();
        }
        
        return true;
    }

    resign(player) {
        if (this.gameOver) return false;
        
        this.gameOver = true;
        this.resigned = player;
        this.winner = player === 'black' ? 'white' : 'black';
        
        return true;
    }

    checkCaptures(x, y, color) {
        const capturedStones = [];
        const opponentColor = color === 1 ? 2 : 1;
        const neighbors = this.getNeighbors(x, y);
        
        for (const neighbor of neighbors) {
            const index = neighbor.y * this.boardSize + neighbor.x;
            if (this.board[index] === opponentColor) {
                const group = this.getGroup(neighbor.x, neighbor.y);
                if (!this.groupHasLiberties(group)) {
                    // Capture the group
                    for (const stone of group) {
                        this.board[stone.y * this.boardSize + stone.x] = 0;
                        capturedStones.push(stone);
                    }
                }
            }
        }
        
        return capturedStones;
    }

    hasLiberties(x, y, color) {
        const group = this.getGroup(x, y);
        return this.groupHasLiberties(group);
    }

    groupHasLiberties(group) {
        for (const stone of group) {
            const neighbors = this.getNeighbors(stone.x, stone.y);
            for (const neighbor of neighbors) {
                const index = neighbor.y * this.boardSize + neighbor.x;
                if (this.board[index] === 0) {
                    return true;
                }
            }
        }
        return false;
    }

    getGroup(x, y) {
        const index = y * this.boardSize + x;
        const color = this.board[index];
        if (color === 0) return [];
        
        const group = [];
        const visited = new Set();
        const stack = [{x, y}];
        
        while (stack.length > 0) {
            const current = stack.pop();
            const currentIndex = current.y * this.boardSize + current.x;
            const key = `${current.x},${current.y}`;
            
            if (visited.has(key)) continue;
            visited.add(key);
            
            if (this.board[currentIndex] === color) {
                group.push(current);
                const neighbors = this.getNeighbors(current.x, current.y);
                for (const neighbor of neighbors) {
                    const neighborKey = `${neighbor.x},${neighbor.y}`;
                    if (!visited.has(neighborKey)) {
                        stack.push(neighbor);
                    }
                }
            }
        }
        
        return group;
    }

    getNeighbors(x, y) {
        const neighbors = [];
        if (x > 0) neighbors.push({x: x - 1, y});
        if (x < this.boardSize - 1) neighbors.push({x: x + 1, y});
        if (y > 0) neighbors.push({x, y: y - 1});
        if (y < this.boardSize - 1) neighbors.push({x, y: y + 1});
        return neighbors;
    }

    isSingleStoneCapture(stone) {
        const neighbors = this.getNeighbors(stone.x, stone.y);
        let emptyNeighbors = 0;
        
        for (const neighbor of neighbors) {
            const index = neighbor.y * this.boardSize + neighbor.x;
            if (this.board[index] === 0) {
                emptyNeighbors++;
            }
        }
        
        return emptyNeighbors === 3; // Single stone surrounded by 3 empty + 1 capturing stone
    }

    wouldBeKo(x, y, color) {
        // Check if playing at (x, y) would immediately capture a single stone
        // that was just played in the previous move
        if (!this.lastMove || this.lastMove.pass) return false;
        
        const neighbors = this.getNeighbors(x, y);
        let wouldCapture = null;
        
        for (const neighbor of neighbors) {
            if (neighbor.x === this.lastMove.x && neighbor.y === this.lastMove.y) {
                const index = neighbor.y * this.boardSize + neighbor.x;
                if (this.board[index] === (color === 1 ? 2 : 1)) {
                    wouldCapture = neighbor;
                    break;
                }
            }
        }
        
        return wouldCapture !== null;
    }

    endGame() {
        this.gameOver = true;
        const scores = this.calculateScore();
        this.winner = scores.black > scores.white ? 'black' : 'white';
    }

    calculateScore() {
        // Simple area scoring
        const territory = this.calculateTerritory();
        return {
            black: this.captures.white + territory.black,
            white: this.captures.black + territory.white + 6.5 // Komi
        };
    }

    calculateTerritory() {
        const visited = new Set();
        const territory = { black: 0, white: 0 };
        
        for (let y = 0; y < this.boardSize; y++) {
            for (let x = 0; x < this.boardSize; x++) {
                const key = `${x},${y}`;
                const index = y * this.boardSize + x;
                
                if (!visited.has(key) && this.board[index] === 0) {
                    const region = this.getEmptyRegion(x, y, visited);
                    const owner = this.getRegionOwner(region);
                    
                    if (owner === 'black') {
                        territory.black += region.length;
                    } else if (owner === 'white') {
                        territory.white += region.length;
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
            
            const index = current.y * this.boardSize + current.x;
            if (this.board[index] === 0) {
                visited.add(key);
                region.push(current);
                
                const neighbors = this.getNeighbors(current.x, current.y);
                for (const neighbor of neighbors) {
                    const neighborKey = `${neighbor.x},${neighbor.y}`;
                    if (!visited.has(neighborKey)) {
                        stack.push(neighbor);
                    }
                }
            }
        }
        
        return region;
    }

    getRegionOwner(region) {
        const borderingColors = new Set();
        
        for (const point of region) {
            const neighbors = this.getNeighbors(point.x, point.y);
            for (const neighbor of neighbors) {
                const index = neighbor.y * this.boardSize + neighbor.x;
                const color = this.board[index];
                if (color !== 0) {
                    borderingColors.add(color);
                }
            }
        }
        
        if (borderingColors.size === 1) {
            return borderingColors.has(1) ? 'black' : 'white';
        }
        
        return 'neutral';
    }

    getState() {
        return {
            board: [...this.board],
            currentPlayer: this.currentPlayer,
            lastMove: this.lastMove,
            captures: { ...this.captures },
            boardSize: this.boardSize,
            gameOver: this.gameOver,
            winner: this.winner,
            resigned: this.resigned,
            passCount: this.passCount
        };
    }

    isValidMove(x, y, player) {
        if (this.gameOver) return false;
        if (player !== this.currentPlayer) return false;
        if (x < 0 || x >= this.boardSize || y < 0 || y >= this.boardSize) return false;
        
        const index = y * this.boardSize + x;
        if (this.board[index] !== 0) return false;
        
        // Check ko rule
        if (this.koPoint && this.koPoint.x === x && this.koPoint.y === y) {
            return false;
        }
        
        // Temporarily place the stone to check if it's a valid move
        const color = player === 'black' ? 1 : 2;
        this.board[index] = color;
        
        // Check for captures
        const wouldCapture = this.checkWouldCapture(x, y, color);
        
        // Check for self-capture (suicide)
        const isValid = wouldCapture || this.hasLiberties(x, y, color);
        
        // Restore the board
        this.board[index] = 0;
        
        return isValid;
    }

    checkWouldCapture(x, y, color) {
        const opponentColor = color === 1 ? 2 : 1;
        const neighbors = this.getNeighbors(x, y);
        
        for (const neighbor of neighbors) {
            const index = neighbor.y * this.boardSize + neighbor.x;
            if (this.board[index] === opponentColor) {
                const group = this.getGroup(neighbor.x, neighbor.y);
                if (!this.groupHasLiberties(group)) {
                    return true;
                }
            }
        }
        
        return false;
    }
}