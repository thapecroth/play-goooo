const GoGame = require('./game/GoGame');
const GoAI = require('./game/GoAI');
const GoAIOptimized = require('./game/GoAIOptimized');

function quickBenchmark() {
    console.log('Quick Performance Comparison\n');
    
    // Test with depth 2, 9x9 board
    const game1 = new GoGame(9);
    const originalAI = new GoAI(game1);
    originalAI.maxDepth = 2;
    
    const game2 = new GoGame(9);
    const optimizedAI = new GoAIOptimized(game2);
    optimizedAI.setDepth(2);
    
    // Measure original AI
    console.log('Testing Original AI (5 moves)...');
    let originalTotal = 0;
    for (let i = 0; i < 5; i++) {
        const start = process.hrtime.bigint();
        const move = originalAI.getBestMove('black');
        const end = process.hrtime.bigint();
        const timeMs = Number(end - start) / 1000000;
        originalTotal += timeMs;
        if (move) {
            game1.makeMove(move.x, move.y, 'black');
            console.log(`  Move ${i+1}: ${timeMs.toFixed(2)}ms at (${move.x}, ${move.y})`);
        }
    }
    
    console.log(`\nOriginal AI average: ${(originalTotal / 5).toFixed(2)}ms per move`);
    
    // Measure optimized AI
    console.log('\nTesting Optimized AI (5 moves)...');
    let optimizedTotal = 0;
    for (let i = 0; i < 5; i++) {
        const start = process.hrtime.bigint();
        const move = optimizedAI.getBestMove('black');
        const end = process.hrtime.bigint();
        const timeMs = Number(end - start) / 1000000;
        optimizedTotal += timeMs;
        if (move) {
            game2.makeMove(move.x, move.y, 'black');
            console.log(`  Move ${i+1}: ${timeMs.toFixed(2)}ms at (${move.x}, ${move.y})`);
        }
    }
    
    console.log(`\nOptimized AI average: ${(optimizedTotal / 5).toFixed(2)}ms per move`);
    
    const speedup = originalTotal / optimizedTotal;
    console.log(`\n=== RESULTS ===`);
    console.log(`Speedup: ${speedup.toFixed(2)}x faster`);
    console.log(`Time saved per move: ${((originalTotal - optimizedTotal) / 5).toFixed(2)}ms`);
}

quickBenchmark();