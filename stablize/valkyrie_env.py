import pybullet as p
import pybullet_data
import numpy as np
import gym
from gym import spaces
import time

class ValkyrieEnv(gym.Env):
    def __init__(self, render_mode=None):
        super().__init__()
        
        # Environment parameters
        self.render_mode = render_mode
        self.time_step = 1./240.
        self.max_episode_steps = 1000
        self.current_step = 0
        
        # Physics client
        if render_mode == "human":
            self.physics_client = p.connect(p.GUI)
        else:
            self.physics_client = p.connect(p.DIRECT)
            
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        
        # Robot parameters
        self.valkyrie_id = None
        self.initial_height = 1.0
        self.target_height = 0.95
        
        # Joint indices (adjust based on your Valkyrie URDF)
        self.leg_joints = list(range(6, 18))  # Assuming 12 leg joints
        self.torso_joints = list(range(0, 6))  # Torso joints
        self.controllable_joints = self.leg_joints + self.torso_joints
        
        # Action and observation spaces
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, 
            shape=(len(self.controllable_joints),), 
            dtype=np.float32
        )
        
        # Observation: joint pos, joint vel, base orientation, base velocity, contact forces
        obs_dim = len(self.controllable_joints) * 2 + 7 + 6 + 4  # joints + base_pose + base_vel + contacts
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, 
            shape=(obs_dim,), 
            dtype=np.float32
        )
        
        # Reward weights
        self.height_weight = 1.0
        self.orientation_weight = 2.0
        self.energy_weight = 0.001
        self.stability_weight = 0.5
        
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Reset simulation
        p.resetSimulation(self.physics_client)
        p.setGravity(0, 0, -9.81)
        p.setTimeStep(self.time_step)
        
        # Load ground and robot
        self.plane_id = p.loadURDF("plane.urdf")
        
        # Load Valkyrie (replace with actual path to Valkyrie URDF)
        start_pos = [0, 0, self.initial_height]
        start_orientation = p.getQuaternionFromEuler([0, 0, 0])
        
        # You need to provide the path to Valkyrie URDF file
        self.valkyrie_id = p.loadURDF(
            "valkyrie/valkyrie.urdf",  # Replace with actual path
            start_pos, 
            start_orientation,
            useFixedBase=False
        )
        
        # Set initial joint positions (standing pose)
        self._set_initial_pose()
        
        # Reset counters
        self.current_step = 0
        self.standing_time = 0
        
        # Get initial observation
        obs = self._get_observation()
        info = {}
        
        return obs, info
    
    def step(self, action):
        # Apply action (joint torques)
        self._apply_action(action)
        
        # Step simulation
        p.stepSimulation()
        
        # Get observation
        obs = self._get_observation()
        
        # Calculate reward
        reward = self._calculate_reward(action)
        
        # Check termination
        terminated = self._is_terminated()
        truncated = self.current_step >= self.max_episode_steps
        
        self.current_step += 1
        
        info = {
            'height': self._get_base_height(),
            'orientation': self._get_base_orientation(),
            'standing_time': self.standing_time
        }
        
        if self.render_mode == "human":
            time.sleep(self.time_step)
            
        return obs, reward, terminated, truncated, info
    
    def _set_initial_pose(self):
        """Set robot to initial standing pose"""
        # Set joint positions for standing pose
        initial_positions = [0.0] * len(self.controllable_joints)
        
        # Adjust specific joints for standing (customize based on your robot)
        # Example: slightly bend knees and hips
        if len(self.leg_joints) >= 12:
            initial_positions[6] = -0.2   # Hip pitch
            initial_positions[7] = 0.4    # Knee pitch
            initial_positions[8] = -0.2   # Ankle pitch
            initial_positions[9] = -0.2   # Hip pitch (other leg)
            initial_positions[10] = 0.4   # Knee pitch (other leg)
            initial_positions[11] = -0.2  # Ankle pitch (other leg)
        
        for i, joint_idx in enumerate(self.controllable_joints):
            p.resetJointState(self.valkyrie_id, joint_idx, initial_positions[i])
    
    def _apply_action(self, action):
        """Apply joint torques"""
        # Scale action to reasonable torque range
        max_torque = 100.0  # Adjust based on robot specs
        torques = action * max_torque
        
        for i, joint_idx in enumerate(self.controllable_joints):
            p.setJointMotorControl2(
                self.valkyrie_id,
                joint_idx,
                p.TORQUE_CONTROL,
                force=torques[i]
            )
    
    def _get_observation(self):
        """Get current observation"""
        obs = []
        
        # Joint positions and velocities
        for joint_idx in self.controllable_joints:
            joint_state = p.getJointState(self.valkyrie_id, joint_idx)
            obs.extend([joint_state[0], joint_state[1]])  # position, velocity
        
        # Base pose and velocity
        base_pos, base_orn = p.getBasePositionAndOrientation(self.valkyrie_id)
        base_vel, base_ang_vel = p.getBaseVelocity(self.valkyrie_id)
        
        obs.extend(base_pos)      # x, y, z
        obs.extend(base_orn)      # quaternion
        obs.extend(base_vel)      # linear velocity
        obs.extend(base_ang_vel)  # angular velocity
        
        # Contact forces (simplified - check foot contacts)
        contact_forces = self._get_contact_forces()
        obs.extend(contact_forces)
        
        return np.array(obs, dtype=np.float32)
    
    def _get_contact_forces(self):
        """Get contact forces for both feet"""
        # Get contact points for feet (adjust link indices for your robot)
        left_foot_contacts = p.getContactPoints(self.valkyrie_id, self.plane_id, linkIndexA=17)  # Left foot
        right_foot_contacts = p.getContactPoints(self.valkyrie_id, self.plane_id, linkIndexA=23)  # Right foot
        
        left_force = sum([contact[9] for contact in left_foot_contacts]) if left_foot_contacts else 0.0
        right_force = sum([contact[9] for contact in right_foot_contacts]) if right_foot_contacts else 0.0
        
        return [left_force, right_force, float(len(left_foot_contacts) > 0), float(len(right_foot_contacts) > 0)]
    
    def _calculate_reward(self, action):
        """Calculate reward for current state"""
        reward = 0.0
        
        # Height reward
        current_height = self._get_base_height()
        height_reward = -abs(self.target_height - current_height)
        reward += self.height_weight * height_reward
        
        # Orientation reward (keep upright)
        orientation_reward = -self._get_orientation_penalty()
        reward += self.orientation_weight * orientation_reward
        
        # Energy penalty
        energy_penalty = -np.sum(np.square(action))
        reward += self.energy_weight * energy_penalty
        
        # Stability reward
        if self._is_stable():
            self.standing_time += 1
            reward += self.stability_weight * 0.1
        else:
            self.standing_time = 0
        
        # Bonus for standing for extended time
        if self.standing_time > 100:
            reward += 1.0
        
        return reward
    
    def _get_base_height(self):
        """Get robot base height"""
        base_pos, _ = p.getBasePositionAndOrientation(self.valkyrie_id)
        return base_pos[2]
    
    def _get_base_orientation(self):
        """Get robot base orientation"""
        _, base_orn = p.getBasePositionAndOrientation(self.valkyrie_id)
        return p.getEulerFromQuaternion(base_orn)
    
    def _get_orientation_penalty(self):
        """Calculate orientation penalty (deviation from upright)"""
        euler = self._get_base_orientation()
        roll, pitch, yaw = euler
        return abs(roll) + abs(pitch)  # Penalize tilting
    
    def _is_stable(self):
        """Check if robot is stable"""
        height = self._get_base_height()
        orientation_penalty = self._get_orientation_penalty()
        
        # Robot is stable if height is reasonable and not tilted too much
        return height > 0.8 and orientation_penalty < 0.3
    
    def _is_terminated(self):
        """Check if episode should terminate"""
        height = self._get_base_height()
        orientation_penalty = self._get_orientation_penalty()
        
        # Terminate if robot falls or tips over too much
        return height < 0.5 or orientation_penalty > 1.0
    
    def render(self):
        """Render the environment"""
        if self.render_mode == "human":
            # Camera follows robot
            base_pos, _ = p.getBasePositionAndOrientation(self.valkyrie_id)
            p.resetDebugVisualizerCamera(
                cameraDistance=3.0,
                cameraYaw=45,
                cameraPitch=-30,
                cameraTargetPosition=base_pos
            )
    
    def close(self):
        """Close the environment"""
        p.disconnect(self.physics_client)