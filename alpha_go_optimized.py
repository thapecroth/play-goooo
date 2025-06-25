import torch
import torch.nn as nn
import torch.nn.functional as F
from optimized_go import OptimizedGoGame, BLACK, WHITE, EMPTY
from mcts_go import MCTSNode, MCTSPlayer

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = self.relu(out)
        return out

class PolicyValueNet(nn.Module):
    def __init__(self, board_size=9, in_channels=3, num_blocks=5):
        super(PolicyValueNet, self).__init__()
        self.board_size = board_size
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        
        self.layers = self._make_layer(ResidualBlock, 64, num_blocks, stride=1)
        
        # Policy head
        self.policy_conv = nn.Conv2d(64, 2, kernel_size=1)
        self.policy_bn = nn.BatchNorm2d(2)
        self.policy_fc = nn.Linear(2 * board_size * board_size, board_size * board_size + 1)
        
        # Value head
        self.value_conv = nn.Conv2d(64, 1, kernel_size=1)
        self.value_bn = nn.BatchNorm2d(1)
        self.value_fc1 = nn.Linear(board_size * board_size, 256)
        self.value_fc2 = nn.Linear(256, 1)

    def _make_layer(self, block, out_channels, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(64, out_channels, stride))
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.layers(out)
        
        # Policy head
        policy = self.relu(self.policy_bn(self.policy_conv(out)))
        policy = policy.view(policy.size(0), -1)
        policy = F.log_softmax(self.policy_fc(policy), dim=1)
        
        # Value head
        value = self.relu(self.value_bn(self.value_conv(out)))
        value = value.view(value.size(0), -1)
        value = self.relu(self.value_fc1(value))
        value = torch.tanh(self.value_fc2(value))
        
        return policy, value

class AlphaGoMCTSNode(MCTSNode):
    def __init__(self, game_state, parent=None, move=None, color=None, prior=0.0):
        super().__init__(game_state, parent, move, color)
        self.prior = prior
        self.value_sum = 0.0

    def get_best_child(self, exploration_constant):
        best_value = -float('inf')
        best_child = None
        
        sqrt_visits = self.visits ** 0.5
        
        for child in self.children:
            if child.visits == 0:
                q_value = 0.0
            else:
                q_value = child.value_sum / child.visits
            
            u_value = exploration_constant * child.prior * sqrt_visits / (1 + child.visits)
            value = q_value + u_value
            
            if value > best_value:
                best_value = value
                best_child = child
        
        return best_child

    def update(self, result):
        self.visits += 1
        self.value_sum += result

class AlphaGoPlayer(MCTSPlayer):
    def __init__(self, policy_value_net, simulations=400, exploration_constant=1.0, is_self_play=False, device=None):
        super().__init__(simulations=simulations, exploration_constant=exploration_constant)
        self.policy_value_net = policy_value_net
        self.is_self_play = is_self_play
        # Default to MPS on Mac, then CUDA, then CPU
        if device is None:
            if torch.backends.mps.is_available():
                self.device = torch.device('mps')
            elif torch.cuda.is_available():
                self.device = torch.device('cuda')
            else:
                self.device = torch.device('cpu')
        else:
            self.device = device
        self.policy_value_net.to(self.device)
        
        # Pre-allocate tensors for efficiency
        self.board_size = policy_value_net.board_size
        self._tensor_cache = torch.zeros((1, 3, self.board_size, self.board_size), 
                                       dtype=torch.float32, device=self.device)

    def get_move(self, game, color, temperature=1.0):
        if game.game_over:
            return None

        root = AlphaGoMCTSNode(game.copy())
        
        # Run simulations
        for _ in range(self.simulations):
            self._run_simulation(root)

        if self.is_self_play:
            # Get moves and visit counts using list comprehension for speed
            moves = [child.move for child in root.children]
            visit_counts = torch.tensor([child.visits for child in root.children], 
                                       dtype=torch.float32, device=self.device)
            
            if temperature == 0:
                # Deterministic: choose most visited
                action_probs = torch.zeros_like(visit_counts)
                action_probs[torch.argmax(visit_counts)] = 1.0
            else:
                # Apply temperature and normalize
                if visit_counts.sum() == 0:
                    # Uniform distribution if no visits
                    action_probs = torch.ones_like(visit_counts) / len(visit_counts)
                else:
                    visit_counts = visit_counts.pow(1.0 / temperature)
                    action_probs = visit_counts / visit_counts.sum()

            if len(moves) > 0:
                # Use torch.multinomial for sampling
                move_idx = torch.multinomial(action_probs, 1).item()
                best_move = moves[move_idx]
            else:
                best_move = None
        else:
            # Choose move with most visits
            if root.children:
                best_child = max(root.children, key=lambda c: c.visits)
                best_move = best_child.move
            else:
                best_move = None

        return best_move

    def _run_simulation(self, node):
        # Selection
        while not node.is_terminal() and node.is_fully_expanded():
            node = node.get_best_child(self.exploration_constant)

        # Expansion
        if not node.is_terminal() and not node.is_fully_expanded():
            # Convert game state to tensor
            game_state_tensor = self._game_to_tensor_fast(node.game_state)
            
            with torch.no_grad():
                policy, value = self.policy_value_net(game_state_tensor)
            
            # Keep on GPU for faster processing
            policy = torch.exp(policy[0])  # Remove batch dimension and exp
            value = value[0, 0].item()  # Extract scalar value
            
            # Get valid moves
            current_color = 'black' if node.game_state.current_player == BLACK else 'white'
            valid_moves = node.game_state.get_valid_moves(current_color)
            
            # Pre-calculate board positions for all valid moves
            for move in valid_moves:
                if move is None:
                    action = self.board_size * self.board_size
                else:
                    action = move[1] * self.board_size + move[0]
                
                # Create child node
                game_copy = node.game_state.copy()
                if move is None:
                    game_copy.pass_turn(current_color)
                else:
                    game_copy.make_move(move[0], move[1], current_color)
                
                prior = policy[action].item()
                child_node = AlphaGoMCTSNode(game_copy, parent=node, move=move, 
                                           color=node.game_state.current_player, prior=prior)
                node.children.append(child_node)
            
            # Backpropagation
            self._backpropagate(node, value)
        else:
            # Terminal node
            if node.game_state.winner == 'black':
                value = 1.0 if node.color == BLACK else -1.0
            elif node.game_state.winner == 'white':
                value = 1.0 if node.color == WHITE else -1.0
            else:
                value = 0.0
            self._backpropagate(node, value)

    def _backpropagate(self, node, value):
        while node is not None:
            node.update(value)
            value = -value
            node = node.parent

    def _game_to_tensor_fast(self, game):
        """Optimized tensor conversion using pre-allocated tensor"""
        # Reset tensor
        self._tensor_cache.zero_()
        
        board = game.board
        current_player = game.current_player
        
        # Use torch operations for faster conversion
        # Channel 0: current player stones
        self._tensor_cache[0, 0] = torch.from_numpy((board == current_player).astype(float))
        
        # Channel 1: opponent stones
        opponent = WHITE if current_player == BLACK else BLACK
        self._tensor_cache[0, 1] = torch.from_numpy((board == opponent).astype(float))
        
        # Channel 2: current player indicator
        if current_player == WHITE:
            self._tensor_cache[0, 2].fill_(1.0)
            
        return self._tensor_cache

    def _game_to_tensor(self, game):
        """Original tensor conversion for compatibility"""
        board = game.board
        size = game.size
        tensor = torch.zeros((1, 3, size, size), dtype=torch.float32, device=self.device)
        
        current_player = game.current_player
        opponent = WHITE if current_player == BLACK else BLACK
        
        # Convert board to tensor
        board_tensor = torch.from_numpy(board).to(self.device)
        
        # Channel 0: current player stones
        tensor[0, 0] = (board_tensor == current_player).float()
        
        # Channel 1: opponent stones  
        tensor[0, 1] = (board_tensor == opponent).float()
        
        # Channel 2: current player indicator
        if current_player == WHITE:
            tensor[0, 2].fill_(1.0)
            
        return tensor

# Batch processing version for self-play
class BatchAlphaGoPlayer(AlphaGoPlayer):
    """Optimized version that can process multiple game positions in batches"""
    
    def __init__(self, policy_value_net, simulations=400, exploration_constant=1.0, 
                 is_self_play=False, device=None, batch_size=8):
        super().__init__(policy_value_net, simulations, exploration_constant, is_self_play, device)
        self.batch_size = batch_size
        self._batch_tensor = torch.zeros((batch_size, 3, self.board_size, self.board_size), 
                                       dtype=torch.float32, device=self.device)
        
    def get_batch_moves(self, games, colors, temperature=1.0):
        """Get moves for multiple games in a single batch"""
        moves = []
        
        # Process games in batches
        for i in range(0, len(games), self.batch_size):
            batch_games = games[i:i+self.batch_size]
            batch_colors = colors[i:i+self.batch_size]
            
            batch_moves = self._process_batch(batch_games, batch_colors, temperature)
            moves.extend(batch_moves)
            
        return moves
    
    def _process_batch(self, games, colors, temperature):
        """Process a batch of games"""
        roots = [AlphaGoMCTSNode(game.copy()) for game in games]
        
        # Run simulations for all games
        for _ in range(self.simulations):
            # Could potentially parallelize this further
            for root in roots:
                self._run_simulation(root)
        
        # Extract moves
        moves = []
        for root in roots:
            if self.is_self_play:
                child_moves = [child.move for child in root.children]
                visit_counts = torch.tensor([child.visits for child in root.children], 
                                          dtype=torch.float32, device=self.device)
                
                if temperature == 0:
                    action_probs = torch.zeros_like(visit_counts)
                    if len(visit_counts) > 0:
                        action_probs[torch.argmax(visit_counts)] = 1.0
                else:
                    if visit_counts.sum() == 0:
                        action_probs = torch.ones_like(visit_counts) / len(visit_counts)
                    else:
                        visit_counts = visit_counts.pow(1.0 / temperature)
                        action_probs = visit_counts / visit_counts.sum()

                if len(child_moves) > 0:
                    move_idx = torch.multinomial(action_probs, 1).item()
                    best_move = child_moves[move_idx]
                else:
                    best_move = None
            else:
                if root.children:
                    best_child = max(root.children, key=lambda c: c.visits)
                    best_move = best_child.move
                else:
                    best_move = None
                    
            moves.append(best_move)
            
        return moves