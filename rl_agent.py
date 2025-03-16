import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import random
from collections import deque
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
import sys

# 添加2048目录到路径
sys.path.append(os.path.join(os.path.dirname(__file__), '2048'))
from game import Game2048

# Set random seeds for reproducibility
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)

# Check if CUDA is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

class DQN(nn.Module):
    def __init__(self):
        super(DQN, self).__init__()
        # Input: 4x4 grid with one-hot encoding of tile values (17 possible values: 0, 2, 4, 8, ..., 65536)
        # We use a 1D representation of the grid (16 cells) with 17 channels per cell
        self.fc1 = nn.Linear(16 * 17, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, 4)  # 4 actions: up, right, down, left
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        return self.fc4(x)

class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
    
    def add(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)
    
    def __len__(self):
        return len(self.buffer)

class DQNAgent:
    def __init__(self, state_size=16*17, action_size=4, buffer_size=10000, batch_size=64, gamma=0.99, 
                 learning_rate=0.001, epsilon_start=1.0, epsilon_end=0.01, epsilon_decay=0.995):
        self.state_size = state_size
        self.action_size = action_size
        self.batch_size = batch_size
        self.gamma = gamma  # discount factor
        self.epsilon = epsilon_start  # exploration rate
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        
        # Q-Networks
        self.policy_net = DQN().to(device)
        self.target_net = DQN().to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=learning_rate)
        self.memory = ReplayBuffer(buffer_size)
        
        # For tracking progress
        self.scores = []
        self.max_tiles = []
        self.epsilon_history = []
    
    def encode_state(self, grid):
        """One-hot encode the grid state"""
        # Flatten the grid
        flat_grid = grid.flatten()
        
        # Create one-hot encoding
        encoded = np.zeros((16, 17), dtype=np.float32)
        
        for i, val in enumerate(flat_grid):
            # Convert value to index (0 -> 0, 2 -> 1, 4 -> 2, etc.)
            if val == 0:
                idx = 0
            else:
                idx = int(np.log2(val))
                
            # Ensure we don't exceed our encoding size
            if idx < 17:
                encoded[i, idx] = 1
        
        return encoded.flatten()
    
    def select_action(self, state, valid_actions):
        """Select an action using epsilon-greedy policy"""
        if random.random() < self.epsilon:
            # Random action from valid actions
            return random.choice(valid_actions)
        else:
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
                q_values = self.policy_net(state_tensor)
                
                # Mask invalid actions with large negative values
                action_mask = torch.ones(self.action_size) * float('-inf')
                for action in valid_actions:
                    action_mask[action] = 0
                
                masked_q_values = q_values + action_mask.to(device)
                return masked_q_values.argmax().item()
    
    def update_epsilon(self):
        """Decay epsilon for less exploration over time"""
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
    
    def learn(self):
        """Update the network weights using a batch of experiences"""
        if len(self.memory) < self.batch_size:
            return
        
        # Sample a batch of experiences
        batch = self.memory.sample(self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        
        # Convert to tensors
        states = torch.FloatTensor(np.array(states)).to(device)
        actions = torch.LongTensor(np.array(actions)).unsqueeze(1).to(device)
        rewards = torch.FloatTensor(np.array(rewards)).unsqueeze(1).to(device)
        next_states = torch.FloatTensor(np.array(next_states)).to(device)
        dones = torch.FloatTensor(np.array(dones, dtype=np.float32)).unsqueeze(1).to(device)
        
        # Compute Q values
        current_q_values = self.policy_net(states).gather(1, actions)
        next_q_values = self.target_net(next_states).max(1, keepdim=True)[0]
        target_q_values = rewards + (1 - dones) * self.gamma * next_q_values
        
        # Compute loss and update
        loss = F.smooth_l1_loss(current_q_values, target_q_values.detach())
        self.optimizer.zero_grad()
        loss.backward()
        # Gradient clipping to prevent exploding gradients
        for param in self.policy_net.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()
    
    def update_target_network(self):
        """Update the target network with the policy network's weights"""
        self.target_net.load_state_dict(self.policy_net.state_dict())
    
    def save_model(self, filename):
        """Save the model weights"""
        torch.save({
            'policy_net': self.policy_net.state_dict(),
            'target_net': self.target_net.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'scores': self.scores,
            'max_tiles': self.max_tiles,
            'epsilon_history': self.epsilon_history
        }, filename)
    
    def load_model(self, filename):
        """Load the model weights"""
        if os.path.exists(filename):
            try:
                # 使用weights_only=False加载模型
                checkpoint = torch.load(filename, weights_only=False)
                self.policy_net.load_state_dict(checkpoint['policy_net'])
                self.target_net.load_state_dict(checkpoint['target_net'])
                self.optimizer.load_state_dict(checkpoint['optimizer'])
                self.scores = checkpoint['scores']
                self.max_tiles = checkpoint['max_tiles']
                self.epsilon_history = checkpoint['epsilon_history']
                print(f"Model loaded from {filename}")
            except Exception as e:
                print(f"Error loading model: {e}")
                print("Initializing new model...")
        else:
            print(f"No model found at {filename}")

def get_valid_actions(game):
    """Get list of valid actions for the current game state"""
    valid_actions = []
    
    # Create a copy of the game to test moves
    for action in range(4):
        test_game = Game2048(enable_recording=False)
        test_game.grid = game.grid.copy()
        test_game.score = game.score
        
        # If the move changes the grid, it's valid
        if test_game.move(action):
            valid_actions.append(action)
    
    # If no valid actions, return all actions (game will end anyway)
    if not valid_actions:
        valid_actions = list(range(4))
    
    return valid_actions

def get_max_tile(grid):
    """Get the maximum tile value on the grid"""
    return np.max(grid)

def train_agent(episodes=1000, target_update=10, save_interval=100, model_path="rl_model.pth"):
    agent = DQNAgent()
    
    # Try to load existing model
    agent.load_model(model_path)
    
    progress_bar = tqdm(range(episodes), desc="Training")
    
    for episode in progress_bar:
        game = Game2048(enable_recording=False)
        state = agent.encode_state(game.grid)
        total_reward = 0
        
        while not game.is_game_over():
            valid_actions = get_valid_actions(game)
            action = agent.select_action(state, valid_actions)
            
            # Get current score before move
            prev_score = game.score
            
            # Take action
            game.move(action)
            
            # Calculate reward (score increase)
            reward = game.score - prev_score
            
            # Additional reward for reaching higher tiles
            max_tile = get_max_tile(game.grid)
            
            # Encode next state
            next_state = agent.encode_state(game.grid)
            
            # Store transition in memory
            agent.memory.add(state, action, reward, next_state, game.is_game_over())
            
            # Learn from experiences
            agent.learn()
            
            # Update state
            state = next_state
            total_reward += reward
        
        # Update target network periodically
        if episode % target_update == 0:
            agent.update_target_network()
        
        # Update exploration rate
        agent.update_epsilon()
        
        # Track progress
        agent.scores.append(game.score)
        agent.max_tiles.append(get_max_tile(game.grid))
        agent.epsilon_history.append(agent.epsilon)
        
        # Update progress bar
        progress_bar.set_postfix({
            'Score': game.score,
            'Max Tile': get_max_tile(game.grid),
            'Epsilon': f"{agent.epsilon:.2f}"
        })
        
        # Save model periodically
        if (episode + 1) % save_interval == 0:
            agent.save_model(model_path)
            
            # Plot progress
            plot_training_progress(agent, episode + 1)
    
    # Save final model
    agent.save_model(model_path)
    
    # Final plot
    plot_training_progress(agent, episodes)
    
    return agent

def plot_training_progress(agent, episodes):
    """Plot the training progress"""
    plt.figure(figsize=(15, 5))
    
    # Plot scores
    plt.subplot(1, 3, 1)
    plt.plot(agent.scores)
    plt.title('Score per Episode')
    plt.xlabel('Episode')
    plt.ylabel('Score')
    
    # Plot max tiles (log scale)
    plt.subplot(1, 3, 2)
    plt.plot([np.log2(x) if x > 0 else 0 for x in agent.max_tiles])
    plt.title('Max Tile per Episode (log2 scale)')
    plt.xlabel('Episode')
    plt.ylabel('log2(Max Tile)')
    
    # Plot epsilon
    plt.subplot(1, 3, 3)
    plt.plot(agent.epsilon_history)
    plt.title('Epsilon per Episode')
    plt.xlabel('Episode')
    plt.ylabel('Epsilon')
    
    plt.tight_layout()
    plt.savefig(f"training_progress_{episodes}.png")
    plt.close()

def play_game(agent, model_path="rl_model.pth", render=True):
    """Play a single game with the trained agent"""
    # Load the trained model
    agent.load_model(model_path)
    
    # Set to evaluation mode (no exploration)
    agent.epsilon = 0
    
    game = Game2048(enable_recording=True)
    
    if render:
        print("Initial state:")
        print(game.grid)
    
    while not game.is_game_over():
        state = agent.encode_state(game.grid)
        valid_actions = get_valid_actions(game)
        action = agent.select_action(state, valid_actions)
        
        # Take action
        game.move(action)
        
        if render:
            print(f"\nAction: {['Up', 'Right', 'Down', 'Left'][action]}")
            print(f"Score: {game.score}")
            print(game.grid)
    
    print(f"\nGame Over! Final Score: {game.score}")
    print(f"Max Tile: {get_max_tile(game.grid)}")
    
    # Save the game record
    record_file = game.save_record()
    if record_file:
        print(f"Game record saved to: {record_file}")
    
    return game.score, get_max_tile(game.grid)

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='2048 RL Agent')
    parser.add_argument('--train', action='store_true', help='Train the agent')
    parser.add_argument('--play', action='store_true', help='Play a game with the trained agent')
    parser.add_argument('--episodes', type=int, default=1000, help='Number of episodes for training')
    parser.add_argument('--model', type=str, default="rl_model.pth", help='Path to save/load the model')
    
    args = parser.parse_args()
    
    if args.train:
        print(f"Training for {args.episodes} episodes...")
        agent = train_agent(episodes=args.episodes, model_path=args.model)
        print("Training completed!")
    
    if args.play:
        print("Playing a game with the trained agent...")
        agent = DQNAgent()
        play_game(agent, model_path=args.model)
    
    # If no arguments provided, train and then play
    if not (args.train or args.play):
        print("No arguments provided. Training and then playing...")
        agent = train_agent(episodes=args.episodes, model_path=args.model)
        play_game(agent, model_path=args.model) 