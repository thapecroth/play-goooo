const GoGame = require('./game/GoGame');
const GoAI = require('./game/GoAI');
const GoAIOptimized = require('./game/GoAIOptimized');

function runBenchmark(aiClass, aiName, boardSize, depth, numMoves) {
    console.log(`\n=== Benchmarking ${aiName} ===`);
    console.log(`Board size: ${boardSize}x${boardSize}, Depth: ${depth}, Moves: ${numMoves}`);
    
    const game = new GoGame(boardSize);
    const ai = new aiClass(game);
    if (ai.setDepth) {
        ai.setDepth(depth);
    } else {
        ai.maxDepth = depth;
    }
    
    const times = [];
    let totalMoves = 0;
    let currentPlayer = 'black';
    
    for (let i = 0; i < numMoves && !game.gameOver; i++) {
        const startTime = process.hrtime.bigint();
        const move = ai.getBestMove(currentPlayer);
        const endTime = process.hrtime.bigint();
        
        const timeMs = Number(endTime - startTime) / 1000000;
        times.push(timeMs);
        
        if (move) {
            game.makeMove(move.x, move.y, currentPlayer);
            totalMoves++;
        } else {
            game.pass(currentPlayer);
        }
        
        currentPlayer = currentPlayer === 'black' ? 'white' : 'black';
    }
    
    if (times.length > 0) {
        const avgTime = times.reduce((a, b) => a + b, 0) / times.length;
        const minTime = Math.min(...times);
        const maxTime = Math.max(...times);
        
        console.log(`Total moves made: ${totalMoves}`);
        console.log(`Average time per move: ${avgTime.toFixed(2)}ms`);
        console.log(`Min time: ${minTime.toFixed(2)}ms`);
        console.log(`Max time: ${maxTime.toFixed(2)}ms`);
        console.log(`Total time: ${times.reduce((a, b) => a + b, 0).toFixed(2)}ms`);
    }
    
    return {
        name: aiName,
        boardSize,
        depth,
        moves: totalMoves,
        avgTime: times.length > 0 ? times.reduce((a, b) => a + b, 0) / times.length : 0,
        totalTime: times.reduce((a, b) => a + b, 0)
    };
}

function compareBenchmarks() {
    console.log('Starting AI Performance Benchmarks...\n');
    
    const testConfigs = [
        { boardSize: 9, depth: 2, numMoves: 10 },
        { boardSize: 9, depth: 3, numMoves: 10 },
        { boardSize: 13, depth: 2, numMoves: 10 },
        { boardSize: 13, depth: 3, numMoves: 5 }
    ];
    
    const results = [];
    
    for (const config of testConfigs) {
        // Test original AI
        const originalResult = runBenchmark(
            GoAI, 
            'Original GoAI', 
            config.boardSize, 
            config.depth, 
            config.numMoves
        );
        results.push(originalResult);
        
        // Test optimized AI
        const optimizedResult = runBenchmark(
            GoAIOptimized, 
            'Optimized GoAI', 
            config.boardSize, 
            config.depth, 
            config.numMoves
        );
        results.push(optimizedResult);
        
        // Calculate speedup
        const speedup = originalResult.avgTime / optimizedResult.avgTime;
        console.log(`\nSpeedup: ${speedup.toFixed(2)}x faster`);
        console.log(`Time saved: ${(originalResult.totalTime - optimizedResult.totalTime).toFixed(2)}ms`);
    }
    
    // Summary
    console.log('\n=== SUMMARY ===');
    console.log('Configuration | Original AI | Optimized AI | Speedup');
    console.log('--------------------------------------------------------');
    
    for (let i = 0; i < results.length; i += 2) {
        const original = results[i];
        const optimized = results[i + 1];
        const speedup = original.avgTime / optimized.avgTime;
        
        console.log(
            `${original.boardSize}x${original.boardSize}, depth ${original.depth} | ` +
            `${original.avgTime.toFixed(2)}ms | ` +
            `${optimized.avgTime.toFixed(2)}ms | ` +
            `${speedup.toFixed(2)}x`
        );
    }
}

// Run the benchmarks
compareBenchmarks();