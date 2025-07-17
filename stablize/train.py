import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Normal
import matplotlib.pyplot as plt
from collections import deque
import random
import pickle
import os

class PolicyNetwork(nn.Module):
    def __init__(self, obs_dim, action_dim, hidden_dim=256):
        super(PolicyNetwork, self).__init__()
        
        self.fc1 = nn.Linear(obs_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        
        # Policy head (mean and log_std)
        self.policy_mean = nn.Linear(hidden_dim, action_dim)
        self.policy_log_std = nn.Linear(hidden_dim, action_dim)
        
        # Value head
        self.value_head = nn.Linear(hidden_dim, 1)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        
        # Policy output
        action_mean = self.policy_mean(x)
        action_log_std = self.policy_log_std(x)
        action_log_std = torch.clamp(action_log_std, -20, 2)
        
        # Value output
        value = self.value_head(x)
        
        return action_mean, action_log_std, value

class PPOAgent:
    def __init__(self, obs_dim, action_dim, lr=3e-4, gamma=0.99, eps_clip=0.2, k_epochs=4):
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.k_epochs = k_epochs
        
        self.policy = PolicyNetwork(obs_dim, action_dim)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        
        self.policy_old = PolicyNetwork(obs_dim, action_dim)
        self.policy_old.load_state_dict(self.policy.state_dict())
        
        self.mse_loss = nn.MSELoss()
        
    def get_action(self, state):
        state = torch.FloatTensor(state).unsqueeze(0)
        
        with torch.no_grad():
            action_mean, action_log_std, _ = self.policy_old(state)
            action_std = torch.exp(action_log_std)
            
            dist = Normal(action_mean, action_std)
            action = dist.sample()
            action_log_prob = dist.log_prob(action).sum(dim=-1)
            
        return action.numpy().flatten(), action_log_prob.numpy().flatten()
    
    def update(self, states, actions, rewards, log_probs, next_states, dones):
        # Convert to tensors
        states = torch.FloatTensor(states)
        actions = torch.FloatTensor(actions)
        rewards = torch.FloatTensor(rewards)
        log_probs = torch.FloatTensor(log_probs)
        next_states = torch.FloatTensor(next_states)
        dones = torch.FloatTensor(dones)
        
        # Calculate returns and advantages
        returns = []
        advantages = []
        
        with torch.no_grad():
            _, _, values = self.policy(states)
            _, _, next_values = self.policy(next_states)
            
            # Calculate returns
            returns = torch.zeros_like(rewards)
            advantages = torch.zeros_like(rewards)
            
            gae = 0
            for t in reversed(range(len(rewards))):
                if t == len(rewards) - 1:
                    next_value = next_values[-1] * (1 - dones[t])
                else:
                    next_value = values[t + 1]
                
                delta = rewards[t] + self.gamma * next_value - values[t]
                gae = delta + self.gamma * 0.95 * (1 - dones[t]) * gae
                advantages[t] = gae
                returns[t] = advantages[t] + values[t]
        
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # Update policy for K epochs
        for _ in range(self.k_epochs):
            # Get current policy outputs
            action_mean, action_log_std, values = self.policy(states)
            action_std = torch.exp(action_log_std)
            
            dist = Normal(action_mean, action_std)
            new_log_probs = dist.log_prob(actions).sum(dim=-1)
            entropy = dist.entropy().sum(dim=-1)
            
            # Calculate ratio
            ratio = torch.exp(new_log_probs - log_probs)
            
            # Calculate surrogate loss
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - self.eps_clip, 1 + self.eps_clip) * advantages
            policy_loss = -torch.min(surr1, surr2).mean()
            
            # Calculate value loss
            value_loss = self.mse_loss(values.squeeze(), returns)
            
            # Calculate total loss
            total_loss = policy_loss + 0.5 * value_loss - 0.01 * entropy.mean()
            
            # Update
            self.optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 0.5)
            self.optimizer.step()
        
        # Update old policy
        self.policy_old.load_state_dict(self.policy.state_dict())
        
        return policy_loss.item(), value_loss.item(), entropy.mean().item()

class ValkyrieTrainer:
    def __init__(self, env, save_dir="./valkyrie_models"):
        self.env = env
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        
        # Initialize agent
        obs_dim = env.observation_space.shape[0]
        action_dim = env.action_space.shape[0]
        self.agent = PPOAgent(obs_dim, action_dim)
        
        # Training parameters
        self.batch_size = 2048
        self.max_episodes = 10000
        self.max_steps = 1000
        self.update_frequency = 2048
        
        # Logging
        self.episode_rewards = []
        self.episode_lengths = []
        self.standing_times = []
        
        # Buffer for collecting experiences
        self.buffer = {
            'states': [],
            'actions': [],
            'rewards': [],
            'log_probs': [],
            'next_states': [],
            'dones': []
        }
        
    def collect_experience(self, state, action, reward, log_prob, next_state, done):
        """Add experience to buffer"""
        self.buffer['states'].append(state)
        self.buffer['actions'].append(action)
        self.buffer['rewards'].append(reward)
        self.buffer['log_probs'].append(log_prob)
        self.buffer['next_states'].append(next_state)
        self.buffer['dones'].append(done)
        
    def clear_buffer(self):
        """Clear experience buffer"""
        for key in self.buffer:
            self.buffer[key] = []
    
    def train(self):
        """Main training loop"""
        print("Starting Valkyrie training...")
        
        step_count = 0
        best_reward = -float('inf')
        
        for episode in range(self.max_episodes):
            state, _ = self.env.reset()
            episode_reward = 0
            episode_length = 0
            max_standing_time = 0
            
            for step in range(self.max_steps):
                # Get action
                action, log_prob = self.agent.get_action(state)
                
                # Take step
                next_state, reward, terminated, truncated, info = self.env.step(action)
                done = terminated or truncated
                
                # Store experience
                self.collect_experience(state, action, reward, log_prob, next_state, done)
                
                episode_reward += reward
                episode_length += 1
                step_count += 1
                max_standing_time = max(max_standing_time, info.get('standing_time', 0))
                
                state = next_state
                
                # Update policy
                if step_count % self.update_frequency == 0:
                    if len(self.buffer['states']) > 0:
                        policy_loss, value_loss, entropy = self.agent.update(
                            self.buffer['states'],
                            self.buffer['actions'],
                            self.buffer['rewards'],
                            self.buffer['log_probs'],
                            self.buffer['next_states'],
                            self.buffer['dones']
                        )
                        self.clear_buffer()
                        
                        print(f"Step {step_count}: Policy Loss: {policy_loss:.4f}, "
                              f"Value Loss: {value_loss:.4f}, Entropy: {entropy:.4f}")
                
                if done:
                    break
            
            # Log episode results
            self.episode_rewards.append(episode_reward)
            self.episode_lengths.append(episode_length)
            self.standing_times.append(max_standing_time)
            
            # Print progress
            if episode % 10 == 0:
                avg_reward = np.mean(self.episode_rewards[-10:])
                avg_length = np.mean(self.episode_lengths[-10:])
                avg_standing = np.mean(self.standing_times[-10:])
                
                print(f"Episode {episode}: Avg Reward: {avg_reward:.2f}, "
                      f"Avg Length: {avg_length:.1f}, Avg Standing Time: {avg_standing:.1f}")
                
                # Save best model
                if avg_reward > best_reward:
                    best_reward = avg_reward
                    self.save_model(f"best_model_episode_{episode}.pth")
            
            # Save checkpoint every 100 episodes
            if episode % 100 == 0:
                self.save_model(f"checkpoint_episode_{episode}.pth")
                self.save_training_data()
        
        print("Training completed!")
        self.save_model("final_model.pth")
        self.save_training_data()
        self.plot_training_results()
    
    def save_model(self, filename):
        """Save model checkpoint"""
        filepath = os.path.join(self.save_dir, filename)
        torch.save({
            'policy_state_dict': self.agent.policy.state_dict(),
            'optimizer_state_dict': self.agent.optimizer.state_dict(),
            'episode_rewards': self.episode_rewards,
            'episode_lengths': self.episode_lengths,
            'standing_times': self.standing_times
        }, filepath)
        print(f"Model saved to {filepath}")
    
    def load_model(self, filename):
        """Load model checkpoint"""
        filepath = os.path.join(self.save_dir, filename)
        checkpoint = torch.load(filepath)
        
        self.agent.policy.load_state_dict(checkpoint['policy_state_dict'])
        self.agent.policy_old.load_state_dict(checkpoint['policy_state_dict'])
        self.agent.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        self.episode_rewards = checkpoint.get('episode_rewards', [])
        self.episode_lengths = checkpoint.get('episode_lengths', [])
        self.standing_times = checkpoint.get('standing_times', [])
        
        print(f"Model loaded from {filepath}")
    
    def save_training_data(self):
        """Save training data"""
        data = {
            'episode_rewards': self.episode_rewards,
            'episode_lengths': self.episode_lengths,
            'standing_times': self.standing_times
        }
        
        filepath = os.path.join(self.save_dir, "training_data.pkl")
        with open(filepath, 'wb') as f:
            pickle.dump(data, f)
        print(f"Training data saved to {filepath}")
    
    def plot_training_results(self):
        """Plot training results"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Episode rewards
        axes[0, 0].plot(self.episode_rewards)
        axes[0, 0].set_title('Episode Rewards')
        axes[0, 0].set_xlabel('Episode')
        axes[0, 0].set_ylabel('Reward')
        
        # Episode lengths
        axes[0, 1].plot(self.episode_lengths)
        axes[0, 1].set_title('Episode Lengths')
        axes[0, 1].set_xlabel('Episode')
        axes[0, 1].set_ylabel('Steps')
        
        # Standing times
        axes[1, 0].plot(self.standing_times)
        axes[1, 0].set_title('Max Standing Time per Episode')
        axes[1, 0].set_xlabel('Episode')
        axes[1, 0].set_ylabel('Standing Time')
        
        # Moving averages
        window = 100
        if len(self.episode_rewards) >= window:
            moving_avg = np.convolve(self.episode_rewards, np.ones(window)/window, mode='valid')
            axes[1, 1].plot(moving_avg)
            axes[1, 1].set_title(f'Moving Average Rewards (window={window})')
            axes[1, 1].set_xlabel('Episode')
            axes[1, 1].set_ylabel('Reward')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, 'training_results.png'))
        plt.show()

def test_trained_model(env, model_path):
    """Test a trained model"""
    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    
    # Load trained model
    agent = PPOAgent(obs_dim, action_dim)
    checkpoint = torch.load(model_path)
    agent.policy.load_state_dict(checkpoint['policy_state_dict'])
    
    # Test for multiple episodes
    for episode in range(5):
        state, _ = env.reset()
        episode_reward = 0
        
        for step in range(1000):
            action, _ = agent.get_action(state)
            next_state, reward, terminated, truncated, info = env.step(action)
            
            episode_reward += reward
            state = next_state
            
            if terminated or truncated:
                break
        
        print(f"Test Episode {episode + 1}: Reward = {episode_reward:.2f}, "
              f"Standing Time = {info.get('standing_time', 0)}")

if __name__ == "__main__":
    # Import the environment
    from valkyrie_env import ValkyrieEnv
    
    # Create environment
    env = ValkyrieEnv(render_mode=None)  # Set to "human" for visualization
    
    # Create trainer
    trainer = ValkyrieTrainer(env)
    
    # Start training
    trainer.train()
    
    # Test trained model
    test_trained_model(env, "./valkyrie_models/best_model.pth")