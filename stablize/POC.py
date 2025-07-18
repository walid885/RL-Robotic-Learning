import pybullet as p
import pybullet_data
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import time
import math
from robot_descriptions.loaders.pybullet import load_robot_description
from typing import List, Dict, Tuple, Optional
from functools import partial
from dataclasses import dataclass
import matplotlib.pyplot as plt
import json
from datetime import datetime
import pickle
import numpy as np
from collections import defaultdict, deque
import random

class ImprovedQLearningAgent:
    def __init__(self, action_dim, state_dim, learning_rate=0.001, epsilon=0.9, epsilon_decay=0.995, epsilon_min=0.01):
        self.action_dim = action_dim
        self.state_dim = state_dim
        self.learning_rate = learning_rate
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.gamma = 0.95
        
        # Initialize weights for policy network (state -> action)
        self.weights = np.random.randn(action_dim, state_dim) * 0.01

        self.bias = np.zeros(action_dim)
        
        # Experience replay
        self.memory = deque(maxlen=10000)
        self.batch_size = 32
        
        # Training stats
        self.episode_rewards = []
        self.losses = []
        
        # Stability tracking
        self.action_history = deque(maxlen=10)
        self.reward_history = deque(maxlen=100)
        
    def get_action(self, state):
        """Get action with proper exploration"""
        state = np.nan_to_num(state, nan=0.0, posinf=1.0, neginf=-1.0)
        
        if random.random() < self.epsilon:
            # Structured exploration: small perturbations around current policy
            base_action = self.predict(state)
            noise = np.random.normal(0, 0.1, self.action_dim)
            return np.clip(base_action + noise, -1.0, 1.0)
        else:
            return self.predict(state)
    
    def predict(self, state):
        """Predict action from current policy"""
        state = np.nan_to_num(state, nan=0.0, posinf=1.0, neginf=-1.0)
        
        try:
            raw_action = np.dot(self.weights, state) + self.bias
            # Use tanh activation for bounded outputs
            action = np.tanh(raw_action)
            return np.clip(action, -1.0, 1.0)
        except:
            return np.zeros(self.action_dim)
    
    def remember(self, state, action, reward, next_state, done):
        """Store experience"""
        self.memory.append((state, action, reward, next_state, done))
    
    def replay(self):
        """Improved training with policy gradient-style updates"""
        if len(self.memory) < self.batch_size:
            return
        
        batch = random.sample(self.memory, self.batch_size)
        total_loss = 0
        
        for state, action, reward, next_state, done in batch:
            state = np.nan_to_num(state, nan=0.0, posinf=1.0, neginf=-1.0)
            next_state = np.nan_to_num(next_state, nan=0.0, posinf=1.0, neginf=-1.0)
            
            # Calculate advantage (simplified)
            if done:
                advantage = reward
            else:
                # Estimate future value
                next_action = self.predict(next_state)
                next_value = np.sum(next_action * self.weights @ next_state)
                advantage = reward + self.gamma * next_value
            
            # Current policy output
            current_action = self.predict(state)
            
            # Policy gradient update (move policy toward better actions)
            if advantage > 0:  # If this was a good action
                # Move policy toward this action
                error = action - current_action
                self.weights += self.learning_rate * advantage * np.outer(error, state)
                self.bias += self.learning_rate * advantage * error
            else:  # If this was a bad action
                # Move policy away from this action
                error = current_action - action
                self.weights += self.learning_rate * abs(advantage) * 0.5 * np.outer(error, state)
                self.bias += self.learning_rate * abs(advantage) * 0.5 * error
            
            total_loss += abs(advantage)
        
        self.losses.append(total_loss / len(batch))
        
        # Decay epsilon more gradually
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
    
    def save_model(self, filename):
        """Save model"""
        model_data = {
            'weights': self.weights,
            'bias': self.bias,
            'epsilon': self.epsilon,
            'episode_rewards': self.episode_rewards,
            'losses': self.losses
        }
        
        with open(filename, 'wb') as f:
            pickle.dump(model_data, f)
        print(f"Model saved to {filename}")
    
    def load_model(self, filename):
        """Load model"""
        try:
            with open(filename, 'rb') as f:
                model_data = pickle.load(f)
            
            self.weights = model_data['weights']
            self.bias = model_data['bias']
            self.epsilon = model_data['epsilon']
            self.episode_rewards = model_data.get('episode_rewards', [])
            self.losses = model_data.get('losses', [])
            
            print(f"Model loaded from {filename}")
            return True
        except FileNotFoundError:
            print(f"No saved model found at {filename}")
            return False        
    



@dataclass
class JointInfo:
    id: int
    name: str
    type: int
    limits: Tuple[float, float]
    current_position: float

@dataclass
class TrainingMetrics:
    episode_rewards: List[float]
    standing_times: List[float]
    heights: List[float]
    orientations: List[float]
    energy_consumption: List[float]
    stability_scores: List[float]
    success_rate: float
    avg_reward: float
    best_episode: int
    
    def __post_init__(self):
        if self.episode_rewards:
            self.avg_reward = np.mean(self.episode_rewards)
            self.best_episode = int(np.argmax(self.episode_rewards))  # Convert to int
        else:
            self.avg_reward = 0.0
            self.best_episode = -1



class ValkyrieEnv(gym.Env):
    def __init__(self, render_mode=None, collect_metrics=True):
        super().__init__()
        
        self.render_mode = render_mode
        self.collect_metrics = collect_metrics
        self.time_step = 1./240.
        self.max_episode_steps = 2000
        self.current_step = 0
        
        # Physics setup
        if render_mode == "human":
            self.physics_client = p.connect(p.GUI)
        else:
            self.physics_client = p.connect(p.DIRECT)
            
        # Set search path immediately after connection
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        
        # Robot parameters
        self.robot_id = None
        self.plane_id = None
        self.initial_height = 2.0
        self.target_height = 1.95
        
        # Momentum tracking for reward improvement
        self.prev_reward = 0.0
        self.reward_momentum = 0.0
        self.momentum_decay = 0.95
        self.momentum_boost = 2.0
        self.improvement_threshold = 0.01
        
        # Action momentum tracking
        self.prev_action = None
        self.action_momentum = None
        self.action_momentum_decay = 0.9
        
        # Reward history for trend analysis
        self.reward_history = []
        self.reward_window = 10
        
        # Metrics tracking
        self.episode_metrics = []
        self.current_episode_data = {
            'rewards': [],
            'heights': [],
            'orientations': [],
            'energy': [],
            'stability': []
        }
        
        # Initialize spaces (will be updated after robot loading)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(10,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(50,), dtype=np.float32)
        
        # Reward weights
        self.height_weight = 2.0
        self.orientation_weight = 3.0
        self.energy_weight = 0.01
        self.stability_weight = 1.0
        self.progress_weight = 0.5
        
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Save previous episode metrics
        if self.collect_metrics and self.current_episode_data['rewards']:
            self.episode_metrics.append({
                'total_reward': float(sum(self.current_episode_data['rewards'])),  # Ensure float
                'avg_height': float(np.mean(self.current_episode_data['heights'])),
                'avg_orientation': float(np.mean(self.current_episode_data['orientations'])),
                'total_energy': float(sum(self.current_episode_data['energy'])),
                'avg_stability': float(np.mean(self.current_episode_data['stability'])),
                'episode_length': int(len(self.current_episode_data['rewards']))
            })
        
        # Reset episode data
        self.current_episode_data = {
            'rewards': [],
            'heights': [],
            'orientations': [],
            'energy': [],
            'stability': []
        }
        
        # Reset momentum tracking
        self.prev_reward = 0.0
        self.reward_momentum = 0.0
        self.prev_action = None
        self.action_momentum = None
        self.reward_history = []
        
        # Reset simulation
        p.resetSimulation(self.physics_client)
        
        # Re-set search path and physics after reset
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.81)
        p.setTimeStep(self.time_step)
        
        # Load environment
        self.plane_id = p.loadURDF("plane.urdf")
        p.changeDynamics(self.plane_id, -1, lateralFriction=1.0, spinningFriction=0.1, rollingFriction=0.1)
        
        # Load Valkyrie robot
        try:
            self.robot_id = load_robot_description("valkyrie_description")
            start_pos = [0, 0, self.initial_height]
            start_orientation = p.getQuaternionFromEuler([0, 0, 0])
            p.resetBasePositionAndOrientation(self.robot_id, start_pos, start_orientation)
            
            # Set robot dynamics
            for i in range(p.getNumJoints(self.robot_id)):
                p.changeDynamics(self.robot_id, i, 
                               linearDamping=0.1, 
                               angularDamping=0.1,
                               maxJointVelocity=2.0)
            
        except Exception as e:
            print(f"Failed to load Valkyrie: {e}")
            # Fallback to R2D2
            self.robot_id = p.loadURDF("r2d2.urdf", [0, 0, self.initial_height])
        
        # Discover and setup joints
        self._setup_joints()
        
        # Initialize robot pose
        self._set_initial_pose()
        
        # Stabilize robot
        for _ in range(100):
            p.stepSimulation()
        
        # Reset counters
        self.current_step = 0
        self.standing_time = 0
        
        obs = self._get_observation()
        return obs, {}
    
    def _setup_joints(self):
        """Setup joint information and action/observation spaces"""
        num_joints = p.getNumJoints(self.robot_id)
        
        # Find controllable joints
        self.controllable_joints = []
        self.arm_joints = []
        self.balance_joints = []
        
        for i in range(num_joints):
            joint_info = p.getJointInfo(self.robot_id, i)
            joint_name = joint_info[1].decode('utf-8').lower()
            joint_type = joint_info[2]
            
            if joint_type in [p.JOINT_REVOLUTE, p.JOINT_PRISMATIC]:
                self.controllable_joints.append(i)
                
                # Categorize joints
                if any(keyword in joint_name for keyword in ['arm', 'shoulder', 'elbow', 'wrist']):
                    self.arm_joints.append(i)
                elif any(keyword in joint_name for keyword in ['leg', 'hip', 'knee', 'ankle', 'torso']):
                    self.balance_joints.append(i)
        
        print(f"Found {len(self.controllable_joints)} controllable joints")
        print(f"Arm joints: {len(self.arm_joints)}, Balance joints: {len(self.balance_joints)}")
        
        # Update action space
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, 
            shape=(len(self.controllable_joints),), 
            dtype=np.float32
        )

        
    # FIXED: Correct observation space calculation
    # joints * 2 (pos + vel) + base_pos(3) + base_orn(4) + base_vel(3) + base_ang_vel(3) + contacts(3)
        obs_dim = len(self.controllable_joints) * 2 + 3 + 4 + 3 + 3 + 3  # = joints*2 + 16
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, 
            shape=(obs_dim,), 
            dtype=np.float32
        )
    
        print(f"Observation space dimension: {obs_dim}")
        print(f"Action space dimension: {len(self.controllable_joints)}")  

    def _set_initial_pose(self):
        """Set robot to stable initial pose"""
        # IMPROVED: Set joints to more stable positions
        joint_targets = {
            'hip': 0.0,
            'knee': 0.1,    # Slightly bent for stability
            'ankle': 0.0,
            'torso': 0.0,
            'arm': 0.0,
            'shoulder': 0.0
        }
        
        for joint_id in self.controllable_joints:
            joint_info = p.getJointInfo(self.robot_id, joint_id)
            joint_name = joint_info[1].decode('utf-8').lower()
            
            target_pos = 0.0
            for keyword, pos in joint_targets.items():
                if keyword in joint_name:
                    target_pos = pos
                    break
            
            p.resetJointState(self.robot_id, joint_id, target_pos)
            
            # IMPROVED: Add position control to maintain pose initially
            p.setJointMotorControl2(
                self.robot_id, joint_id, p.POSITION_CONTROL, 
                targetPosition=target_pos,
                force=100.0
            )
    
    def step(self, action):
        # Debug: Check action for NaN
        if np.any(np.isnan(action)):
            print(f"NaN detected in action: {action}")
            action = np.zeros_like(action)
        
        # Apply action
        self._apply_action(action)
        
        # Step simulation
        p.stepSimulation()
        
        # Get observation and calculate metrics
        obs = self._get_observation()
        
        # Debug: Check observation for NaN
        if np.any(np.isnan(obs)):
            print(f"NaN detected in observation at step {self.current_step}")
            print(f"NaN indices: {np.where(np.isnan(obs))}")
        
        reward, metrics = self._calculate_base_reward(action)

        
        # Debug: Check reward for NaN
        if np.isnan(reward):
            print(f"NaN detected in reward at step {self.current_step}")
            print(f"Metrics: {metrics}")
        
        # Check termination
        terminated = self._is_terminated()
        truncated = self.current_step >= self.max_episode_steps
        
        self.current_step += 1
        
        # Update metrics
        if self.collect_metrics:
            self._update_metrics(reward, metrics)
        
        info = {
            'height': metrics['height'],
            'orientation_penalty': metrics['orientation_penalty'],
            'stability_score': metrics['stability_score'],
            'energy_consumption': metrics['energy_consumption'],
            'standing_time': self.standing_time,
            'reward_momentum': self.reward_momentum,
            'reward_trend': self._get_reward_trend()
        }
        
        if self.render_mode == "human":
            time.sleep(self.time_step)
            
        return obs, reward, terminated, truncated, info    
    def _apply_action(self, action):
        """Apply joint control actions with NaN protection"""
        max_torque = 20.0  # Reduced further for stability
        
        # NaN protection for actions
        action = np.nan_to_num(action, nan=0.0, posinf=1.0, neginf=-1.0)
        
        for i, joint_id in enumerate(self.controllable_joints):
            if i < len(action):
                # Safe action processing
                safe_action = np.clip(action[i], -1.0, 1.0)
                torque = np.tanh(safe_action) * max_torque
                
                # Apply different control for different joint types
                if joint_id in self.balance_joints:
                    torque *= 0.5  # Even more conservative
                elif joint_id in self.arm_joints:
                    torque *= 0.2
                
                # Additional safety check
                if np.isnan(torque) or np.isinf(torque):
                    torque = 0.0
                
                p.setJointMotorControl2(
                    self.robot_id, joint_id, p.TORQUE_CONTROL, force=torque
                )
    
    def get_initial_action(self, obs):
        """Get stabilizing action for early training"""
        # Extract orientation
        base_orn_start = len(self.controllable_joints) * 2 + 3
        base_orn = obs[base_orn_start:base_orn_start + 4]
        euler = p.getEulerFromQuaternion(base_orn)
        
        action = np.zeros(self.action_dim)
        
        # Simple PD controller for initial stability
        roll_correction = -euler[0] * 0.5
        pitch_correction = -euler[1] * 0.5
        
        # Apply to relevant joints (hip/ankle joints)
        for i in range(min(6, len(action))):  # First 6 joints often leg joints
            if i % 2 == 0:  # Even indices for roll
                action[i] = roll_correction
            else:  # Odd indices for pitch
                action[i] = pitch_correction
        
        return np.clip(action, -0.5, 0.5)

    def train_with_learning(self, episodes=200, save_interval=50):
        agent = ImprovedQLearningAgent(
            action_dim=self.action_space.shape[0],
            state_dim=self.observation_space.shape[0],
            learning_rate=0.001,  # Lower learning rate
            epsilon=0.9,
            epsilon_decay=0.998,  # Slower decay
            epsilon_min=0.05      # Higher minimum
        )
        
        # Training with curriculum learning
        for episode in range(episodes):
            obs, _ = self.reset()
            episode_reward = 0
            
            for step in range(self.max_episode_steps):
                if episode < 20:  # Early episodes: more structured exploration
                    action = agent.get_initial_action(obs)
                    action += np.random.normal(0, 0.1, agent.action_dim)
                else:
                    action = agent.get_action(obs)
                
                next_obs, reward, terminated, truncated, info = self.step(action)
                agent.remember(obs, action, reward, next_obs, terminated or truncated)
                
                # Train every 4 steps
                if step % 4 == 0:
                    agent.replay()
                
                obs = next_obs
                episode_reward += reward
                
                if terminated or truncated:
                    break
            
            agent.episode_rewards.append(episode_reward)
            
            if episode % 10 == 0:
                avg_reward = np.mean(agent.episode_rewards[-10:])
                print(f"Episode {episode}: Reward={episode_reward:.2f}, "
                    f"Avg={avg_reward:.2f}, ε={agent.epsilon:.3f}")
    
    def train_with_learning(self, episodes=200, save_interval=50, model_filename="valkyrie_model.pkl"):
        """Train with actual learning agent"""
        print("Starting training with learning agent...")
        
        # FIXED: Get actual observation to ensure correct dimensions
        obs_sample, _ = self.reset()
        actual_obs_dim = len(obs_sample)
        actual_action_dim = self.action_space.shape[0]
        
        print(f"Actual observation dimension: {actual_obs_dim}")
        print(f"Actual action dimension: {actual_action_dim}")
        
        # Create learning agent with correct dimensions
        agent = ImprovedQLearningAgent(
            action_dim=actual_action_dim,
            state_dim=actual_obs_dim  # Use actual observation dimension
        )
        
        # Try to load existing model
        agent.load_model(model_filename)
        
        best_reward = float('-inf')
        recent_rewards = deque(maxlen=100)
        
        for episode in range(episodes):
            obs, _ = self.reset()
            episode_reward = 0
            step_count = 0
            
            print(f"\nEpisode {episode + 1}/{episodes} (ε={agent.epsilon:.3f})")
            
            while True:
                # Get action from agent
                action = agent.get_action(obs)
                
                # Take step
                next_obs, reward, terminated, truncated, info = self.step(action)
                episode_reward += reward
                step_count += 1
                
                # Store experience
                agent.remember(obs, action, reward, next_obs, terminated or truncated)
                
                # Train agent
                if step_count % 4 == 0:  # Train every 4 steps
                    agent.replay()
                
                obs = next_obs
                
                if terminated or truncated:
                    break
            
            # Track performance
            agent.episode_rewards.append(episode_reward)
            recent_rewards.append(episode_reward)
            
            # Check for improvement
            if episode_reward > best_reward:
                best_reward = episode_reward
                print(f"  NEW BEST! Reward: {episode_reward:.2f}")
            
            avg_recent = np.mean(recent_rewards) if recent_rewards else 0
            print(f"  Episode reward: {episode_reward:.2f}, Recent avg: {avg_recent:.2f}")
            print(f"  Steps: {step_count}, Height: {info['height']:.3f}")
            
            # Save model periodically
            if (episode + 1) % save_interval == 0:
                agent.save_model(model_filename)
                print(f"  Model saved at episode {episode + 1}")
            
            # Plot progress
            if (episode + 1) % 50 == 0:
                self.plot_training_progress()
        
        # Final save
        agent.save_model(model_filename)
        
        print(f"\nTraining completed!")
        print(f"Best reward: {best_reward:.2f}")
        print(f"Final epsilon: {agent.epsilon:.3f}")
        
        return agent
    
    def _get_observation(self):
        """Get current state observation with NaN protection"""
        obs = []
        
        # Joint states with NaN protection
        for joint_id in self.controllable_joints:
            joint_state = p.getJointState(self.robot_id, joint_id)
            pos = joint_state[0] if not np.isnan(joint_state[0]) else 0.0
            vel = joint_state[1] if not np.isnan(joint_state[1]) else 0.0
            obs.extend([pos, vel])
        
        # Base state with NaN protection
        base_pos, base_orn = p.getBasePositionAndOrientation(self.robot_id)
        base_vel, base_ang_vel = p.getBaseVelocity(self.robot_id)
        
        # Check for NaN values and replace with safe defaults
        base_pos = [x if not np.isnan(x) else 0.0 for x in base_pos]
        base_orn = [x if not np.isnan(x) else 0.0 for x in base_orn]
        base_vel = [x if not np.isnan(x) else 0.0 for x in base_vel]
        base_ang_vel = [x if not np.isnan(x) else 0.0 for x in base_ang_vel]
        
        obs.extend(base_pos)
        obs.extend(base_orn)
        obs.extend(base_vel)
        obs.extend(base_ang_vel)
        
        # Contact information
        contacts = p.getContactPoints(self.robot_id, self.plane_id)
        contact_force = sum([contact[9] for contact in contacts]) if contacts else 0.0
        num_contacts = len(contacts)
        obs.extend([contact_force, num_contacts, float(num_contacts > 0)])
        
        # Final NaN check
        obs = np.array(obs, dtype=np.float32)
        obs = np.nan_to_num(obs, nan=0.0, posinf=1.0, neginf=-1.0)
        
        return obs

    
    def _calculate_reward(self, action):
        """Simplified, stable reward function"""
        base_pos, base_orn = p.getBasePositionAndOrientation(self.robot_id)
        height = base_pos[2]
        
        # Simple, stable rewards
        height_reward = 2.0 * max(0, 1.0 - abs(self.target_height - height))
        
        euler = p.getEulerFromQuaternion(base_orn)
        orientation_reward = 1.0 * max(0, 1.0 - (abs(euler[0]) + abs(euler[1])))
        
        # Survival bonus
        survival_bonus = 0.1
        
        # Penalize excessive actions
        energy_penalty = -0.001 * np.sum(np.square(action))
        
        return height_reward + orientation_reward + survival_bonus + energy_penalty
    def _calculate_reward_with_momentum(self, action):
        """Calculate reward with momentum-based improvements"""
        # Calculate base reward
        base_reward, metrics = self._calculate_base_reward(action)
        
        # Update reward history
        self.reward_history.append(base_reward)
        if len(self.reward_history) > self.reward_window:
            self.reward_history.pop(0)
        
        # Calculate reward improvement
        reward_improvement = base_reward - self.prev_reward
        
        # Update momentum based on improvement
        if reward_improvement > self.improvement_threshold:
            # Reward is improving, boost momentum
            self.reward_momentum = self.reward_momentum * self.momentum_decay + reward_improvement * self.momentum_boost
            
            # Apply action momentum if we have previous action
            if self.prev_action is not None:
                if self.action_momentum is None:
                    self.action_momentum = np.zeros_like(action)
                
                # Calculate action difference that led to improvement
                action_diff = action - self.prev_action
                self.action_momentum = self.action_momentum * self.action_momentum_decay + action_diff * 0.1
        else:
            # Reward is not improving, decay momentum
            self.reward_momentum *= self.momentum_decay
            if self.action_momentum is not None:
                self.action_momentum *= self.action_momentum_decay
        
        # Calculate trend bonus
        trend_bonus = self._calculate_trend_bonus()
        
        # Apply momentum bonus
        momentum_bonus = max(0, self.reward_momentum)
        
        # Final reward with momentum
        final_reward = base_reward + momentum_bonus + trend_bonus
        
        # Update for next step
        self.prev_reward = base_reward
        self.prev_action = action.copy()
        
        # Add momentum info to metrics
        metrics['momentum_bonus'] = momentum_bonus
        metrics['trend_bonus'] = trend_bonus
        metrics['reward_improvement'] = reward_improvement
        
        return final_reward, metrics
    
    def _calculate_base_reward(self, action):
        """Calculate base reward with NaN protection"""
        try:
            # Get current state
            base_pos, base_orn = p.getBasePositionAndOrientation(self.robot_id)
            
            # NaN protection for base position
            height = base_pos[2] if not np.isnan(base_pos[2]) else 1.0
            
            # Safe quaternion to euler conversion
            try:
                euler = p.getEulerFromQuaternion(base_orn)
                orientation_penalty = abs(euler[0]) + abs(euler[1])
            except:
                orientation_penalty = 0.0
            
            # NaN protection for orientation
            if np.isnan(orientation_penalty):
                orientation_penalty = 0.0
            
            # Safe energy calculation
            energy_consumption = np.sum(np.square(action))
            if np.isnan(energy_consumption):
                energy_consumption = 0.0
            
            # Safe reward calculations
            height_reward = max(0, 1.0 - abs(self.target_height - height))
            orientation_reward = max(0, 1.0 - orientation_penalty)
            energy_penalty = -energy_consumption * 0.001
            
            # Stability reward
            stability_score = self._calculate_stability_score(height, orientation_penalty)
            
            if stability_score > 0.6:
                self.standing_time += 1
                stability_reward = 1.0
            else:
                self.standing_time = 0
                stability_reward = 0.0
            
            if self.standing_time > 100:
                stability_reward += 5.0
            
            survival_bonus = 0.1
            
            # Total reward with NaN protection
            total_reward = (
                self.height_weight * height_reward +
                self.orientation_weight * orientation_reward +
                self.energy_weight * energy_penalty +
                self.stability_weight * stability_reward +
                survival_bonus
            )
            
            # Final NaN check
            if np.isnan(total_reward):
                total_reward = 0.0
            
            metrics = {
                'height': height,
                'orientation_penalty': orientation_penalty,
                'energy_consumption': energy_consumption,
                'stability_score': stability_score,
                'height_reward': height_reward,
                'orientation_reward': orientation_reward,
                'energy_penalty': energy_penalty,
                'stability_reward': stability_reward,
                'survival_bonus': survival_bonus
            }
            
            return total_reward, metrics
            
        except Exception as e:
            print(f"Error in reward calculation: {e}")
            # Return safe default values
            return 0.0, {
                'height': 1.0,
                'orientation_penalty': 0.0,
                'energy_consumption': 0.0,
                'stability_score': 0.0,
                'height_reward': 0.0,
                'orientation_reward': 0.0,
                'energy_penalty': 0.0,
                'stability_reward': 0.0,
                'survival_bonus': 0.0
            }




    def simple_balance_policy(self, obs):
        """Simple policy that tries to maintain balance"""
        # Extract base orientation from observation
        base_orn_start = len(self.controllable_joints) * 2 + 3  # After joint states and position
        base_orn = obs[base_orn_start:base_orn_start + 4]  # quaternion
        
        # Convert to euler
        euler = p.getEulerFromQuaternion(base_orn)
        roll, pitch, yaw = euler
        
        # Simple PD control for balance
        action = np.zeros(len(self.controllable_joints))
        
        # Apply corrective torques to balance joints
        for i, joint_id in enumerate(self.controllable_joints):
            if joint_id in self.balance_joints:
                joint_info = p.getJointInfo(self.robot_id, joint_id)
                joint_name = joint_info[1].decode('utf-8').lower()
                
                if 'hip' in joint_name:
                    if 'roll' in joint_name or 'x' in joint_name:
                        action[i] = -roll * 2.0  # Counteract roll
                    elif 'pitch' in joint_name or 'y' in joint_name:
                        action[i] = -pitch * 2.0  # Counteract pitch
                elif 'ankle' in joint_name:
                    if 'roll' in joint_name or 'x' in joint_name:
                        action[i] = -roll * 1.0
                    elif 'pitch' in joint_name or 'y' in joint_name:
                        action[i] = -pitch * 1.0
        
        # Clip actions
        action = np.clip(action, -1.0, 1.0)
        
        return action

    def _calculate_trend_bonus(self):
        """Calculate bonus based on reward trend"""
        if len(self.reward_history) < 3:
            return 0.0
        
        # Calculate trend over recent rewards
        recent_rewards = self.reward_history[-3:]
        if len(recent_rewards) >= 2:
            # Simple trend: compare last reward with average of previous
            trend = recent_rewards[-1] - np.mean(recent_rewards[:-1])
            return max(0, trend * 0.1)  # Small bonus for positive trend
        
        return 0.0
    
    def _get_reward_trend(self):
        """Get current reward trend indicator"""
        if len(self.reward_history) < 2:
            return 0.0
        
        return self.reward_history[-1] - self.reward_history[-2]
    
    def _calculate_stability_score(self, height, orientation_penalty):
        """Calculate stability score (0-1)"""
        height_score = max(0, min(1, (height - 1.0) / 1.0))
        orientation_score = max(0, min(1, (0.5 - orientation_penalty) / 0.5))
        return (height_score + orientation_score) / 2.0
    
    def _is_terminated(self):
        """Check if episode should terminate"""
        base_pos, base_orn = p.getBasePositionAndOrientation(self.robot_id)
        euler = p.getEulerFromQuaternion(base_orn)
        
        height = base_pos[2]
        orientation_penalty = abs(euler[0]) + abs(euler[1])
        
        # Terminate if robot falls or tips over severely
        return height < 1.0 or orientation_penalty > 1.5
    
    def _is_terminated(self):
        """Check if episode should terminate"""
        try:
            base_pos, base_orn = p.getBasePositionAndOrientation(self.robot_id)
            
            # Check for NaN values
            if any(np.isnan(base_pos)) or any(np.isnan(base_orn)):
                print(f"NaN detected in base position/orientation: pos={base_pos}, orn={base_orn}")
                return True
            
            euler = p.getEulerFromQuaternion(base_orn)
            height = base_pos[2]
            orientation_penalty = abs(euler[0]) + abs(euler[1])
            
            # Check for NaN in calculations
            if np.isnan(height) or np.isnan(orientation_penalty):
                print(f"NaN detected in height or orientation: height={height}, orientation_penalty={orientation_penalty}")
                return True
            
            # Terminate if robot falls or tips over severely
            return height < 0.5 or orientation_penalty > 2.0
        except Exception as e:
            print(f"Error in termination check: {e}")
            return True  # Terminate on any error

    def _update_metrics(self, reward, metrics):
        """Update episode metrics"""
        self.current_episode_data['rewards'].append(reward)
        self.current_episode_data['heights'].append(metrics['height'])
        self.current_episode_data['orientations'].append(metrics['orientation_penalty'])
        self.current_episode_data['energy'].append(metrics['energy_consumption'])
        self.current_episode_data['stability'].append(metrics['stability_score'])
    
    def get_training_metrics(self) -> TrainingMetrics:
        """Get comprehensive training metrics"""
        if not self.episode_metrics:
            return TrainingMetrics([], [], [], [], [], [], 0.0, 0.0, -1)
        
        episode_rewards = [float(ep['total_reward']) for ep in self.episode_metrics]
        standing_times = [float(ep['episode_length']) for ep in self.episode_metrics]
        heights = [float(ep['avg_height']) for ep in self.episode_metrics]
        orientations = [float(ep['avg_orientation']) for ep in self.episode_metrics]
        energy_consumption = [float(ep['total_energy']) for ep in self.episode_metrics]
        stability_scores = [float(ep['avg_stability']) for ep in self.episode_metrics]
        
        # Calculate success rate (episodes where robot stayed up > 75% of time)
        successful_episodes = sum(1 for ep in self.episode_metrics 
                                if ep['avg_stability'] > 0.75)
        success_rate = float(successful_episodes / len(self.episode_metrics))
        
        return TrainingMetrics(
            episode_rewards=episode_rewards,
            standing_times=standing_times,
            heights=heights,
            orientations=orientations,
            energy_consumption=energy_consumption,
            stability_scores=stability_scores,
            success_rate=success_rate,
            avg_reward=0.0,  # Will be calculated in __post_init__
            best_episode=-1   # Will be calculated in __post_init__
        )
    
    def plot_training_progress(self, save_path=None):
        """Plot training progress"""
        if not self.episode_metrics:
            print("No metrics to plot")
            return
        
        metrics = self.get_training_metrics()
        episodes = range(len(metrics.episode_rewards))
        
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle('Valkyrie Training Progress', fontsize=16)
        
        # Episode rewards
        axes[0, 0].plot(episodes, metrics.episode_rewards)
        axes[0, 0].set_title('Episode Rewards')
        axes[0, 0].set_xlabel('Episode')
        axes[0, 0].set_ylabel('Total Reward')
        axes[0, 0].grid(True)
        
        # Average heights
        axes[0, 1].plot(episodes, metrics.heights)
        axes[0, 1].axhline(y=self.target_height, color='r', linestyle='--', label='Target')
        axes[0, 1].set_title('Average Height')
        axes[0, 1].set_xlabel('Episode')
        axes[0, 1].set_ylabel('Height (m)')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        # Stability scores
        axes[0, 2].plot(episodes, metrics.stability_scores)
        axes[0, 2].set_title('Stability Scores')
        axes[0, 2].set_xlabel('Episode')
        axes[0, 2].set_ylabel('Stability Score')
        axes[0, 2].grid(True)
        
        # Orientation penalties
        axes[1, 0].plot(episodes, metrics.orientations)
        axes[1, 0].set_title('Orientation Penalties')
        axes[1, 0].set_xlabel('Episode')
        axes[1, 0].set_ylabel('Orientation Penalty')
        axes[1, 0].grid(True)
        
        # Energy consumption
        axes[1, 1].plot(episodes, metrics.energy_consumption)
        axes[1, 1].set_title('Energy Consumption')
        axes[1, 1].set_xlabel('Episode')
        axes[1, 1].set_ylabel('Energy')
        axes[1, 1].grid(True)
        
        # Success rate (rolling average)
        window_size = min(10, len(episodes))
        if window_size > 1:
            rolling_success = np.convolve(
                [1 if s > 0.75 else 0 for s in metrics.stability_scores], 
                np.ones(window_size)/window_size, mode='valid'
            )
            axes[1, 2].plot(episodes[window_size-1:], rolling_success)
        axes[1, 2].set_title(f'Success Rate (Rolling {window_size})')
        axes[1, 2].set_xlabel('Episode')
        axes[1, 2].set_ylabel('Success Rate')
        axes[1, 2].grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def save_metrics(self, filename):
        """Save metrics to JSON file"""
        metrics = self.get_training_metrics()
        data = {
            'timestamp': datetime.now().isoformat(),
            'total_episodes': len(metrics.episode_rewards),
            'success_rate': float(metrics.success_rate),  # Ensure float
            'average_reward': float(metrics.avg_reward),
            'best_episode': int(metrics.best_episode),  # Ensure int
            'episode_data': self.episode_metrics
        }
        
        with open(filename, 'w') as f:
            json.dump(data, f, indent=2)
        
        print(f"Metrics saved to {filename}")
    
    def close(self):
        p.disconnect(self.physics_client)
    
def train_valkyrie_improved(episodes=100, render=False):
    """Train Valkyrie robot with improved setup"""
    print("Starting improved Valkyrie training...")
    
    env = ValkyrieEnv(render_mode="human" if render else None)
    
    for episode in range(episodes):
        obs, _ = env.reset()
        episode_reward = 0
        step_count = 0
        
        print(f"\nEpisode {episode + 1}/{episodes}")
        
        while True:
            # IMPROVED: Use simple balance policy instead of random
            if episode < 20:  # First 20 episodes use balance policy
                action = env.simple_balance_policy(obs)
            else:
                # Mix of balance policy and exploration
                balance_action = env.simple_balance_policy(obs)
                random_action = env.action_space.sample() * 0.1
                action = 0.7 * balance_action + 0.3 * random_action
            
            obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            step_count += 1
            
            # Print progress
            if step_count % 500 == 0:
                print(f"  Step {step_count}: Reward={reward:.3f}, "
                      f"Height={info['height']:.3f}, "
                      f"Stability={info['stability_score']:.3f}")
            
            if terminated or truncated:
                break
        
        print(f"Episode {episode + 1} completed: "
              f"Reward={episode_reward:.2f}, "
              f"Steps={step_count}")
        
        # Plot progress every 25 episodes
        if (episode + 1) % 25 == 0:
            env.plot_training_progress()
    
    # Final metrics
    metrics = env.get_training_metrics()
    print(f"\nTraining completed!")
    print(f"Success rate: {metrics.success_rate:.2%}")
    print(f"Average reward: {metrics.avg_reward:.2f}")
    
    env.close()
    return metrics


def train_valkyrie(episodes=100, render=False):
    """Train Valkyrie robot with random policy"""
    print("Starting Valkyrie training...")
    
    env = ValkyrieEnv(render_mode="human" if render else None)
    
    for episode in range(episodes):
        obs, _ = env.reset()
        episode_reward = 0
        step_count = 0
        
        print(f"\nEpisode {episode + 1}/{episodes}")
        
        while True:
            # Random action (replace with RL algorithm)
            action = env.action_space.sample() * 0.3  # Reduced action magnitude
            
            obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            step_count += 1
            
            # Print progress
            if step_count % 500 == 0:
                print(f"  Step {step_count}: Reward={reward:.3f}, "
                      f"Height={info['height']:.3f}, "
                      f"Stability={info['stability_score']:.3f}, "
                      f"Momentum={info['reward_momentum']:.3f}")
            
            if terminated or truncated:
                break
        
        print(f"Episode {episode + 1} completed: "
              f"Reward={episode_reward:.2f}, "
              f"Steps={step_count}, "
              f"Standing time={info['standing_time']}")
        
        # Plot progress every 10 episodes
        if (episode + 1) % 10 == 0:
            env.plot_training_progress()
    
    # Final metrics
    metrics = env.get_training_metrics()
    print(f"\nTraining completed!")
    print(f"Success rate: {metrics.success_rate:.2%}")
    print(f"Average reward: {metrics.avg_reward:.2f}")
    print(f"Best episode: {metrics.best_episode}")
    
    # Save final metrics
    env.save_metrics(f"valkyrie_metrics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    env.plot_training_progress(f"valkyrie_progress_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png")
    
    env.close()
    return metrics


def test_environment():
    """Test the environment setup"""
    print("Testing Valkyrie environment...")
    
    env = ValkyrieEnv(render_mode="human")
    
    # Test reset
    obs, _ = env.reset()
    print(f"Observation shape: {obs.shape}")
    print(f"Action space: {env.action_space.shape}")
    
    # Test a few steps
    for i in range(100):
        action = env.action_space.sample() * 0.1
        obs, reward, terminated, truncated, info = env.step(action)
        
        if i % 20 == 0:
            print(f"Step {i}: Reward={reward:.3f}, Height={info['height']:.3f}, "
                  f"Momentum={info['reward_momentum']:.3f}")
        
        if terminated or truncated:
            break
    
    env.close()

# Updated main training function
def train_valkyrie_with_learning(episodes=200, render=False):
    """Train Valkyrie with actual learning"""
    env = ValkyrieEnv(render_mode="human" if render else None)
    
    
    # Train with learning
    agent = env.train_with_learning(episodes=episodes)

    
    # Test the trained agent
    print("\nTesting trained agent...")
    obs, _ = env.reset()
    total_reward = 0
    
    for step in range(1000):
        action = agent.predict(obs)  # Use learned policy (no exploration)
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
        if step % 100 == 0:
            print(f"Test step {step}: Reward={reward:.3f}, Height={info['height']:.3f}")
        
        if terminated or truncated:
            break
    
    print(f"Test completed: Total reward={total_reward:.2f}")
    
    env.close()
    return agent

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Valkyrie Robot Training')
    parser.add_argument('--mode', choices=['test', 'train'], default='test',
                      help='Mode: test or train')
    parser.add_argument('--episodes', type=int, default=50,
                      help='Number of training episodes')
    parser.add_argument('--render', action='store_true',
                      help='Render training')
    
    args = parser.parse_args()
    
    if args.mode == 'test':
        test_environment()
    else:
        train_valkyrie_with_learning(episodes=args.episodes, render=args.render)
        