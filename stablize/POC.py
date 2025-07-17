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
            self.best_episode = np.argmax(self.episode_rewards)
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
            
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        
        # Robot parameters
        self.robot_id = None
        self.plane_id = None
        self.initial_height = 2.0
        self.target_height = 1.95
        
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
                'total_reward': sum(self.current_episode_data['rewards']),
                'avg_height': np.mean(self.current_episode_data['heights']),
                'avg_orientation': np.mean(self.current_episode_data['orientations']),
                'total_energy': sum(self.current_episode_data['energy']),
                'avg_stability': np.mean(self.current_episode_data['stability']),
                'episode_length': len(self.current_episode_data['rewards'])
            })
        
        # Reset episode data
        self.current_episode_data = {
            'rewards': [],
            'heights': [],
            'orientations': [],
            'energy': [],
            'stability': []
        }
        
        # Reset simulation
        p.resetSimulation(self.physics_client)
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
        
        # Update observation space
        obs_dim = len(self.controllable_joints) * 2 + 13  # joints + base state + contacts
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, 
            shape=(obs_dim,), 
            dtype=np.float32
        )
    
    def _set_initial_pose(self):
        """Set robot to stable initial pose"""
        for joint_id in self.controllable_joints:
            p.resetJointState(self.robot_id, joint_id, 0.0)
    
    def step(self, action):
        # Apply action
        self._apply_action(action)
        
        # Step simulation
        p.stepSimulation()
        
        # Get observation and calculate metrics
        obs = self._get_observation()
        reward, metrics = self._calculate_reward(action)
        
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
            'standing_time': self.standing_time
        }
        
        if self.render_mode == "human":
            time.sleep(self.time_step)
            
        return obs, reward, terminated, truncated, info
    
    def _apply_action(self, action):
        """Apply joint control actions"""
        max_torque = 200.0
        
        for i, joint_id in enumerate(self.controllable_joints):
            if i < len(action):
                # Scale action to torque
                torque = action[i] * max_torque
                
                # Apply different control for different joint types
                if joint_id in self.balance_joints:
                    # More conservative control for balance joints
                    torque *= 0.5
                
                p.setJointMotorControl2(
                    self.robot_id, joint_id, p.TORQUE_CONTROL, force=torque
                )
    
    def _get_observation(self):
        """Get current state observation"""
        obs = []
        
        # Joint states
        for joint_id in self.controllable_joints:
            joint_state = p.getJointState(self.robot_id, joint_id)
            obs.extend([joint_state[0], joint_state[1]])  # position, velocity
        
        # Base state
        base_pos, base_orn = p.getBasePositionAndOrientation(self.robot_id)
        base_vel, base_ang_vel = p.getBaseVelocity(self.robot_id)
        
        obs.extend(base_pos)  # 3D position
        obs.extend(base_orn)  # 4D quaternion
        obs.extend(base_vel)  # 3D linear velocity
        obs.extend(base_ang_vel)  # 3D angular velocity
        
        # Contact information
        contacts = p.getContactPoints(self.robot_id, self.plane_id)
        contact_force = sum([contact[9] for contact in contacts]) if contacts else 0.0
        num_contacts = len(contacts)
        obs.extend([contact_force, num_contacts, float(num_contacts > 0)])
        
        return np.array(obs, dtype=np.float32)
    
    def _calculate_reward(self, action):
        """Calculate reward and metrics"""
        # Get current state
        base_pos, base_orn = p.getBasePositionAndOrientation(self.robot_id)
        euler = p.getEulerFromQuaternion(base_orn)
        
        height = base_pos[2]
        orientation_penalty = abs(euler[0]) + abs(euler[1])  # roll + pitch
        energy_consumption = np.sum(np.square(action))
        
        # Height reward
        height_reward = -abs(self.target_height - height)
        
        # Orientation reward
        orientation_reward = -orientation_penalty
        
        # Energy penalty
        energy_penalty = -energy_consumption
        
        # Stability reward
        stability_score = self._calculate_stability_score(height, orientation_penalty)
        
        if stability_score > 0.8:
            self.standing_time += 1
            stability_reward = 0.1
        else:
            self.standing_time = 0
            stability_reward = -0.1
        
        # Bonus for sustained standing
        if self.standing_time > 500:
            stability_reward += 2.0
        
        # Total reward
        total_reward = (
            self.height_weight * height_reward +
            self.orientation_weight * orientation_reward +
            self.energy_weight * energy_penalty +
            self.stability_weight * stability_reward
        )
        
        metrics = {
            'height': height,
            'orientation_penalty': orientation_penalty,
            'energy_consumption': energy_consumption,
            'stability_score': stability_score,
            'height_reward': height_reward,
            'orientation_reward': orientation_reward,
            'energy_penalty': energy_penalty,
            'stability_reward': stability_reward
        }
        
        return total_reward, metrics
    
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
        
        episode_rewards = [ep['total_reward'] for ep in self.episode_metrics]
        standing_times = [ep['episode_length'] for ep in self.episode_metrics]
        heights = [ep['avg_height'] for ep in self.episode_metrics]
        orientations = [ep['avg_orientation'] for ep in self.episode_metrics]
        energy_consumption = [ep['total_energy'] for ep in self.episode_metrics]
        stability_scores = [ep['avg_stability'] for ep in self.episode_metrics]
        
        # Calculate success rate (episodes where robot stayed up > 75% of time)
        successful_episodes = sum(1 for ep in self.episode_metrics 
                                if ep['avg_stability'] > 0.75)
        success_rate = successful_episodes / len(self.episode_metrics)
        
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
            'success_rate': metrics.success_rate,
            'average_reward': metrics.avg_reward,
            'best_episode': metrics.best_episode,
            'episode_data': self.episode_metrics
        }
        
        with open(filename, 'w') as f:
            json.dump(data, f, indent=2)
        
        print(f"Metrics saved to {filename}")
    
    def close(self):
        p.disconnect(self.physics_client)


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
                      f"Stability={info['stability_score']:.3f}")
            
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
            print(f"Step {i}: Reward={reward:.3f}, Height={info['height']:.3f}")
        
        if terminated or truncated:
            break
    
    env.close()


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
        train_valkyrie(episodes=args.episodes, render=args.render)