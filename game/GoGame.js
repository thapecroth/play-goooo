class GoGame {
    constructor(size = 9) {
        this.size = size;
        this.board = Array(size).fill(null).map(() => Array(size).fill(null));
        this.currentPlayer = 'black';
        this.captures = { black: 0, white: 0 };
        this.history = [];
        this.passes = 0;
        this.gameOver = false;
        this.winner = null;
        this.lastMove = null;
        this.ko = null;
    }
    
    makeMove(x, y, color) {
        if (this.gameOver || x < 0 || x >= this.size || y < 0 || y >= this.size) {
            return false;
        }
        
        if (this.board[y][x] !== null) {
            return false;
        }
        
        if (color !== this.currentPlayer) {
            return false;
        }
        
        const boardCopy = this.copyBoard();
        this.board[y][x] = color;
        
        const oppositeColor = color === 'black' ? 'white' : 'black';
        const capturedStones = this.captureStones(oppositeColor);
        
        if (!this.hasLiberties(x, y, color)) {
            this.board = boardCopy;
            return false;
        }
        
        if (this.isKo(boardCopy)) {
            this.board = boardCopy;
            return false;
        }
        
        this.captures[color] += capturedStones.length;
        this.history.push({
            board: boardCopy,
            move: { x, y, color },
            captures: { ...this.captures }
        });
        
        this.lastMove = { x, y };
        this.passes = 0;
        this.currentPlayer = oppositeColor;
        this.ko = capturedStones.length === 1 ? capturedStones[0] : null;
        
        return true;
    }
    
    captureStones(color) {
        const captured = [];
        
        for (let y = 0; y < this.size; y++) {
            for (let x = 0; x < this.size; x++) {
                if (this.board[y][x] === color) {
                    const group = this.getGroup(x, y);
                    if (!this.groupHasLiberties(group)) {
                        group.forEach(stone => {
                            this.board[stone.y][stone.x] = null;
                            captured.push(stone);
                        });
                    }
                }
            }
        }
        
        return captured;
    }
    
    getGroup(x, y) {
        const color = this.board[y][x];
        if (!color) return [];
        
        const group = [];
        const visited = new Set();
        const stack = [{x, y}];
        
        while (stack.length > 0) {
            const pos = stack.pop();
            const key = `${pos.x},${pos.y}`;
            
            if (visited.has(key)) continue;
            visited.add(key);
            
            if (this.board[pos.y][pos.x] === color) {
                group.push(pos);
                
                const neighbors = this.getNeighbors(pos.x, pos.y);
                neighbors.forEach(n => {
                    if (!visited.has(`${n.x},${n.y}`)) {
                        stack.push(n);
                    }
                });
            }
        }
        
        return group;
    }
    
    getNeighbors(x, y) {
        const neighbors = [];
        if (x > 0) neighbors.push({x: x - 1, y});
        if (x < this.size - 1) neighbors.push({x: x + 1, y});
        if (y > 0) neighbors.push({x, y: y - 1});
        if (y < this.size - 1) neighbors.push({x, y: y + 1});
        return neighbors;
    }
    
    groupHasLiberties(group) {
        for (const stone of group) {
            const neighbors = this.getNeighbors(stone.x, stone.y);
            for (const n of neighbors) {
                if (this.board[n.y][n.x] === null) {
                    return true;
                }
            }
        }
        return false;
    }
    
    hasLiberties(x, y, color) {
        const tempColor = this.board[y][x];
        this.board[y][x] = color;
        const group = this.getGroup(x, y);
        const hasLib = this.groupHasLiberties(group);
        this.board[y][x] = tempColor;
        return hasLib;
    }
    
    isKo(previousBoard) {
        if (this.history.length < 1) return false;
        
        for (let y = 0; y < this.size; y++) {
            for (let x = 0; x < this.size; x++) {
                if (this.board[y][x] !== previousBoard[y][x]) {
                    return false;
                }
            }
        }
        return true;
    }
    
    copyBoard() {
        return this.board.map(row => [...row]);
    }
    
    pass(color) {
        if (color !== this.currentPlayer) return false;
        
        this.passes++;
        this.currentPlayer = color === 'black' ? 'white' : 'black';
        
        if (this.passes >= 2) {
            this.endGame();
        }
        
        return true;
    }
    
    resign(color) {
        this.gameOver = true;
        this.winner = color === 'black' ? 'white' : 'black';
    }
    
    endGame() {
        this.gameOver = true;
        const score = this.calculateScore();
        this.winner = score.black > score.white ? 'black' : 'white';
    }
    
    calculateScore() {
        const territory = { black: 0, white: 0 };
        const visited = new Set();
        
        for (let y = 0; y < this.size; y++) {
            for (let x = 0; x < this.size; x++) {
                const key = `${x},${y}`;
                if (this.board[y][x] === null && !visited.has(key)) {
                    const region = this.getEmptyRegion(x, y, visited);
                    const owner = this.getRegionOwner(region);
                    if (owner) {
                        territory[owner] += region.length;
                    }
                }
            }
        }
        
        const stones = { black: 0, white: 0 };
        for (let y = 0; y < this.size; y++) {
            for (let x = 0; x < this.size; x++) {
                if (this.board[y][x]) {
                    stones[this.board[y][x]]++;
                }
            }
        }
        
        return {
            black: stones.black + territory.black + this.captures.black,
            white: stones.white + territory.white + this.captures.white + 6.5
        };
    }
    
    getEmptyRegion(startX, startY, visited) {
        const region = [];
        const stack = [{x: startX, y: startY}];
        
        while (stack.length > 0) {
            const pos = stack.pop();
            const key = `${pos.x},${pos.y}`;
            
            if (visited.has(key)) continue;
            visited.add(key);
            
            if (this.board[pos.y][pos.x] === null) {
                region.push(pos);
                const neighbors = this.getNeighbors(pos.x, pos.y);
                neighbors.forEach(n => stack.push(n));
            }
        }
        
        return region;
    }
    
    getRegionOwner(region) {
        const borderColors = new Set();
        
        for (const pos of region) {
            const neighbors = this.getNeighbors(pos.x, pos.y);
            for (const n of neighbors) {
                const color = this.board[n.y][n.x];
                if (color) {
                    borderColors.add(color);
                }
            }
        }
        
        if (borderColors.size === 1) {
            return [...borderColors][0];
        }
        return null;
    }
    
    getValidMoves(color) {
        const moves = [];
        for (let y = 0; y < this.size; y++) {
            for (let x = 0; x < this.size; x++) {
                if (this.board[y][x] === null) {
                    const boardCopy = this.copyBoard();
                    const currentPlayerCopy = this.currentPlayer;
                    this.currentPlayer = color;
                    
                    if (this.makeMove(x, y, color)) {
                        moves.push({x, y});
                        this.board = boardCopy;
                        this.currentPlayer = currentPlayerCopy;
                    } else {
                        this.board = boardCopy;
                        this.currentPlayer = currentPlayerCopy;
                    }
                }
            }
        }
        return moves;
    }
    
    getState() {
        return {
            board: this.board,
            currentPlayer: this.currentPlayer,
            captures: this.captures,
            gameOver: this.gameOver,
            winner: this.winner,
            lastMove: this.lastMove,
            score: this.gameOver ? this.calculateScore() : null
        };
    }
}

module.exports = GoGame;