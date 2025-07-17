# valkyrie_env.py
import gymnasium as gym
from gymnasium import spaces
import pybullet as p
import numpy as np
import os

# Import your simulation components
from robot_descriptions.loaders.pybullet import load_robot_description

from sim.physics_engine import PhysicsEngine
from sim.robot_loader import RobotLoader
from sim.simulation_config import SimulationConfig
# from simulation.joint_utils import JointAnalyzer # If you need this for complex parsing later


class ValkyrieStandEnv(gym.Env):
    metadata = {'render_modes': ['human', 'rgb_array'], 'render_fps': 50}

    def __init__(self, render_mode='human'):
        super(ValkyrieStandEnv, self).__init__()
        self.render_mode = render_mode
        self.config = SimulationConfig()

        # Initialize your simulation components
        self.physics_engine = PhysicsEngine(p.GUI if render_mode == 'human' else p.DIRECT)
        self.robot_loader = RobotLoader(self.physics_engine, self.config)

        self.humanoid_id = -1
        self.plane_id = -1
        self.controlled_joint_indices = []
        self.joint_limits_low = []
        self.joint_limits_high = []
        self.joint_ranges = []
        self.max_forces = []
        self.initial_joint_target_positions = {} # Dictionary to store initial positions by name

        # Define Valkyrie specific joint names and foot links
        # IMPORTANT: These must EXACTLY match the joint names in your Valkyrie URDF.
        # Run the `list_valkyrie_joints` helper at the end of this file to verify.
        self.VALKYRIE_CONTROLLABLE_JOINT_NAMES = [
            'leftHipYaw', 'leftHipRoll', 'leftHipPitch',
            'leftKneePitch',
            'leftAnklePitch', 'leftAnkleRoll',
            'rightHipYaw', 'rightHipRoll', 'rightHipPitch',
            'rightKneePitch',
            'rightAnklePitch', 'rightAnkleRoll',
            'torsoYaw', 'torsoPitch', 'torsoRoll' # Added torso roll
            # Add more as needed, e.g., shoulder/elbow/wrist for upper body balance
        ]
        self.LEFT_FOOT_LINK_NAME = 'leftFoot' # Common link name for the foot
        self.RIGHT_FOOT_LINK_NAME = 'rightFoot'
        self.left_foot_link_id = -1
        self.right_foot_link_id = -1
        self.torso_link_name = 'torso' # Or 'pelvis', 'base_link' depending on URDF
        self.torso_link_id = -1

        self.physics_engine.initialize() # Connect PyBullet client
        self._load_robot_initial() # Load robot once to parse joint info for spaces

        # Define Observation Space
        # Base orientation (quaternion: 4 values)
        # Base angular velocity (3 values)
        # Base linear velocity (3 values)
        # Joint positions (N values)
        # Joint velocities (N values)
        num_observations = 4 + 3 + 3 + len(self.controlled_joint_indices) * 2
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(num_observations,), dtype=np.float32)

        # Define Action Space (target joint positions, normalized [-1, 1])
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(len(self.controlled_joint_indices),), dtype=np.float32)

        # Initial joint target positions (e.g., a standing or slightly bent pose)
        # You'll need to define a good initial pose for Valkyrie here.
        # This dictionary maps joint names to target angles in radians.
        # This is a critical part for stable learning!
        self.initial_joint_target_positions_dict = {
            # Example values, TUNE THESE FOR VALKYRIE
            'leftHipYaw': 0.0, 'leftHipRoll': 0.0, 'leftHipPitch': -0.7,
            'leftKneePitch': 1.4,
            'leftAnklePitch': -0.7, 'leftAnkleRoll': 0.0,
            'rightHipYaw': 0.0, 'rightHipRoll': 0.0, 'rightHipPitch': -0.7,
            'rightKneePitch': 1.4,
            'rightAnklePitch': -0.7, 'rightAnkleRoll': 0.0,
            'torsoYaw': 0.0, 'torsoPitch': 0.0, 'torsoRoll': 0.0,
        }
        # Convert to an array based on controlled_joint_indices order later in _get_joint_info
        self._initial_joint_target_positions_array = None


    def _load_robot_initial(self):
        # This is called once to get robot ID and parse joint info
        # We use a temporary connection for this if not in human mode
        if self.humanoid_id == -1: # Only load if not already loaded
            self.plane_id = p.loadURDF("plane.urdf", physicsClientId=self.physics_engine.client_id)
            self.humanoid_id = self.robot_loader.load_robot("valkyrie_description", 
                                                            position=[0, 0, self.config.robot_height])
            self._get_joint_info()
            self._get_foot_link_ids()


    def _get_joint_info(self):
        self.controlled_joint_indices = []
        self.joint_limits_low = []
        self.joint_limits_high = []
        self.joint_ranges = []
        self.max_forces = []
        
        # Populate these for easy lookup
        all_joint_info = {}
        link_name_to_id = {}

        num_joints = p.getNumJoints(self.humanoid_id, self.physics_engine.client_id)
        for i in range(num_joints):
            joint_info = p.getJointInfo(self.humanoid_id, i, self.physics_engine.client_id)
            joint_id = joint_info[0]
            joint_name = joint_info[1].decode("utf-8")
            joint_type = joint_info[2]
            link_name = joint_info[12].decode("utf-8")
            
            all_joint_info[joint_name] = {
                "id": joint_id, "type": joint_type,
                "lower_limit": joint_info[8], "upper_limit": joint_info[9],
                "max_force": joint_info[10], "link_name": link_name
            }
            link_name_to_id[link_name] = joint_id

            if joint_name in self.VALKYRIE_CONTROLLABLE_JOINT_NAMES:
                if joint_type in [p.JOINT_REVOLUTE, p.JOINT_PRISMATIC]:
                    self.controlled_joint_indices.append(joint_id)
                    self.joint_limits_low.append(joint_info[8])
                    self.joint_limits_high.append(joint_info[9])
                    self.joint_ranges.append(joint_info[9] - joint_info[8])
                    # Use default_joint_force if URDF specifies 0 or small force
                    self.max_forces.append(joint_info[10] if joint_info[10] > 0 else self.config.default_joint_force)

        # Sort controlled joints by their ID for consistent ordering in observation/action spaces
        # This is important as `getJointStates` returns in ID order.
        temp_joint_data = sorted([(idx, all_joint_info[p.getJointInfo(self.humanoid_id, idx, self.physics_engine.client_id)[1].decode("utf-8")])
                                   for idx in self.controlled_joint_indices], key=lambda x: x[0])
        
        self.controlled_joint_indices = [x[0] for x in temp_joint_data]
        self.joint_limits_low = np.array([x[1]["lower_limit"] for x in temp_joint_data])
        self.joint_limits_high = np.array([x[1]["upper_limit"] for x in temp_joint_data])
        self.joint_ranges = np.array([x[1]["upper_limit"] - x[1]["lower_limit"] for x in temp_joint_data])
        self.max_forces = np.array([x[1]["max_force"] if x[1]["max_force"] > 0 else self.config.default_joint_force for x in temp_joint_data])

        # Populate the initial joint target positions array based on the sorted indices
        self._initial_joint_target_positions_array = np.array([
            self.initial_joint_target_positions_dict.get(
                p.getJointInfo(self.humanoid_id, joint_idx, self.physics_engine.client_id)[1].decode("utf-8"),
                0.0 # Default to 0.0 if not specified
            ) for joint_idx in self.controlled_joint_indices
        ])
        
        # Get torso link ID
        self.torso_link_id = link_name_to_id.get(self.torso_link_name, -1)
        if self.torso_link_id == -1:
             print(f"WARNING: Torso link '{self.torso_link_name}' not found. Orientation reward might be inaccurate.")


        # print(f"Detected {len(self.controlled_joint_indices)} controllable joints for Valkyrie.")
        # print(f"Controlled Joint IDs: {self.controlled_joint_indices}")
        # print(f"Joint Names: {[p.getJointInfo(self.humanoid_id, idx, self.physics_engine.client_id)[1].decode('utf-8') for idx in self.controlled_joint_indices]}")


    def _get_foot_link_ids(self):
        # Iterate through all links to find foot links
        for i in range(p.getNumJoints(self.humanoid_id, self.physics_engine.client_id)):
            link_name = p.getJointInfo(self.humanoid_id, i, self.physics_engine.client_id)[12].decode("utf-8")
            if link_name == self.LEFT_FOOT_LINK_NAME:
                self.left_foot_link_id = i
            elif link_name == self.RIGHT_FOOT_LINK_NAME:
                self.right_foot_link_id = i
        if self.left_foot_link_id == -1 or self.right_foot_link_id == -1:
            print(f"WARNING: Could not find one or both foot link IDs ({self.LEFT_FOOT_LINK_NAME}, {self.RIGHT_FOOT_LINK_NAME}). Contact detection might be inaccurate.")


    def _get_obs(self):
        # Base state (using torso link as proxy for CoM/base if available, else base link)
        if self.torso_link_id != -1:
            torso_state = p.getLinkState(self.humanoid_id, self.torso_link_id, computeLinkVelocity=True, physicsClientId=self.physics_engine.client_id)
            base_pos = torso_state[0] # CoM position approximation
            base_orn = torso_state[1] # Orientation
            base_lin_vel = torso_state[6] # Linear velocity
            base_ang_vel = torso_state[7] # Angular velocity
        else: # Fallback to base position if torso link not found
            base_pos, base_orn = p.getBasePositionAndOrientation(self.humanoid_id, self.physics_engine.client_id)
            base_lin_vel, base_ang_vel = p.getBaseVelocity(self.humanoid_id, self.physics_engine.client_id)


        # Joint states
        joint_states = p.getJointStates(self.humanoid_id, self.controlled_joint_indices, self.physics_engine.client_id)
        joint_positions = [s[0] for s in joint_states]
        joint_velocities = [s[1] for s in joint_states]

        # Concatenate all observations
        obs = np.concatenate([
            np.array(base_orn),       # 4 (quaternion x,y,z,w)
            np.array(base_ang_vel),   # 3 (rad/s)
            np.array(base_lin_vel),   # 3 (m/s)
            np.array(joint_positions), # N (rad)
            np.array(joint_velocities) # N (rad/s)
        ]).astype(np.float32)

        return obs

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # Disconnect and reconnect to reset simulation fully for each episode
        # This is important for robust resets in PyBullet
        self.physics_engine.disconnect()
        self.physics_engine.initialize()
        
        self.plane_id = p.loadURDF("plane.urdf", physicsClientId=self.physics_engine.client_id)
        
        # Use robot_loader to load the robot, which handles its initial setup
        self.humanoid_id = self.robot_loader.load_robot("valkyrie_description", 
                                                        position=[0, 0, self.config.robot_height])
        
        # After loading, get the joint and link info again as IDs might change
        self._get_joint_info() 
        self._get_foot_link_ids()

        # Set initial joint positions for the agent to start from a specific pose
        for i, joint_idx in enumerate(self.controlled_joint_indices):
            # Use the pre-defined initial_joint_target_positions_array
            initial_pos = self._initial_joint_target_positions_array[i]
            p.resetJointState(self.humanoid_id, joint_idx, initial_pos, physicsClientId=self.physics_engine.client_id)
            # Set motors to hold this initial pose
            p.setJointMotorControl2(self.humanoid_id, joint_idx, p.POSITION_CONTROL,
                                    targetPosition=initial_pos,
                                    force=self.max_forces[i],
                                    physicsClientId=self.physics_engine.client_id)
        
        # Let the simulation settle for a few steps after reset
        for _ in range(self.config.stabilization_steps):
            self.physics_engine.step_simulation()

        observation = self._get_obs()
        info = {}
        return observation, info

    def step(self, action):
        # Map normalized actions [-1, 1] to actual joint angle ranges
        target_joint_positions = self.joint_limits_low + (action + 1.0) * 0.5 * self.joint_ranges

        # Apply actions to joints
        p.setJointMotorControlArray(self.humanoid_id, self.controlled_joint_indices,
                                    p.POSITION_CONTROL,
                                    targetPositions=target_joint_positions,
                                    forces=self.max_forces,
                                    physicsClientId=self.physics_engine.client_id)

        # Simulate multiple steps for each RL action
        for _ in range(self.config.sim_steps_per_action):
            self.physics_engine.step_simulation()

        observation = self._get_obs()
        reward, terminated = self._calculate_reward()
        truncated = False # For now, we don't have truncation based on time limits, only termination based on fall.
        info = {}

        return observation, reward, terminated, truncated, info

    def _calculate_reward(self):
        # Get base link information (using torso link as proxy for CoM/base)
        if self.torso_link_id != -1:
            torso_state = p.getLinkState(self.humanoid_id, self.torso_link_id, computeLinkVelocity=True, physicsClientId=self.physics_engine.client_id)
            base_pos = torso_state[0] # CoM position approximation
            base_orn = torso_state[1] # Orientation
            base_lin_vel = torso_state[6] # Linear velocity
        else: # Fallback to base position if torso link not found
            base_pos, base_orn = p.getBasePositionAndOrientation(self.humanoid_id, self.physics_engine.client_id)
            base_lin_vel, _ = p.getBaseVelocity(self.humanoid_id, self.physics_engine.client_id)
        
        # Convert quaternion to rotation matrix and get "up" vector
        rot_mat = np.array(p.getMatrixFromQuaternion(base_orn)).reshape(3, 3)
        up_vector = rot_mat[:, 2] # Z-axis of the robot's base frame in world coordinates

        # --- Reward Components ---
        reward = 0.0
        terminated = False

        # 1. Height Reward (CoM height)
        com_height = base_pos[2]
        height_reward = self.config.reward_weight_height * min(com_height, self.config.target_standing_height)

        # 2. Uprightness Reward (alignment with world Z-axis)
        upright_reward = self.config.reward_weight_uprightness * np.dot(up_vector, [0, 0, 1])

        # 3. Stability Penalty (penalize horizontal velocity of base CoM)
        horizontal_velocity = np.linalg.norm(base_lin_vel[:2]) # X, Y components
        stability_penalty = self.config.penalty_weight_horizontal_velocity * horizontal_velocity**2

        # 4. Action Regularization Penalty (penalize deviation from initial/neutral pose)
        current_joint_positions = np.array([s[0] for s in p.getJointStates(self.humanoid_id, self.controlled_joint_indices, self.physics_engine.client_id)])
        action_penalty = self.config.penalty_weight_action_magnitude * np.sum(np.square(current_joint_positions - self._initial_joint_target_positions_array))

        # 5. Contact Penalty (penalize if non-foot parts touch the ground)
        contact_points = p.getContactPoints(self.humanoid_id, self.plane_id, physicsClientId=self.physics_engine.client_id)
        non_foot_contact_penalty = 0.0
        for contact in contact_points:
            link_index_a = contact[3] # Link ID of the robot involved in contact
            if link_index_a != self.left_foot_link_id and link_index_a != self.right_foot_link_id and link_index_a != -1:
                non_foot_contact_penalty = self.config.penalty_weight_non_foot_contact
                break # Only one non-foot contact needed to apply penalty

        # --- Termination Condition ---
        # If CoM falls below a certain threshold or is too tilted
        if com_height < self.config.min_stable_height or np.dot(up_vector, [0, 0, 1]) < self.config.max_tilt_angle_cos:
            terminated = True
            reward += self.config.penalty_episode_termination # Large penalty for falling

        # --- Combine Rewards ---
        reward += height_reward + upright_reward + stability_penalty + action_penalty + non_foot_contact_penalty

        return reward, terminated

    def render(self):
        # PyBullet GUI handles rendering automatically when render_mode='human'
        if self.render_mode == 'human':
            pass # No explicit rendering needed here
        elif self.render_mode == 'rgb_array':
            # Implement capturing image frames if needed for video recording
            # Adjust camera parameters for Valkyrie's scale
            view_matrix = p.computeViewMatrixFromYawPitchRoll(
                cameraTargetPosition=[0, 0, 1.0], # Focus on humanoid CoM height
                distance=3.0,
                yaw=50,
                pitch=-35,
                roll=0,
                upAxisIndex=2,
                physicsClientId=self.physics_engine.client_id
            )
            proj_matrix = p.computeProjectionMatrixFOV(
                fov=60,
                aspect=float(960)/720,
                nearVal=0.1,
                farVal=100.0,
                physicsClientId=self.physics_engine.client_id
            )
            (_, _, px, _, _) = p.getCameraImage(
                width=960,
                height=720,
                viewMatrix=view_matrix,
                projectionMatrix=proj_matrix,
                renderer=p.ER_BULLET_HARDWARE_OPENGL,
                physicsClientId=self.physics_engine.client_id
            )
            rgb_array = np.array(px).reshape(720, 960, 4)[:, :, :3]
            return rgb_array

    def close(self):
        self.physics_engine.disconnect()

def load_robot(description: str, position: list[float]) -> int:
    """Load robot from description and set initial position."""
    robot_id = load_robot_description(description)
    p.resetBasePositionAndOrientation(robot_id, position, [0, 0, 0, 1])
    for i in range(p.getNumJoints(robot_id)):
        p.changeDynamics(robot_id, i, 
                        linearDamping=0.1, 
                        angularDamping=0.1,
                        maxJointVelocity=1.0)
    return robot_id


# --- Helper to list Valkyrie joints for configuration ---
def list_valkyrie_joints_from_description():
    cid = p.connect(p.DIRECT)
    try:
        valkyrie = load_robot("valkyrie_description", [0, 0, config.robot_height])

        num_joints = p.getNumJoints(valkyrie, physicsClientId=cid)
        print(f"\n--- Joints and Links for Valkyrie from robot_descriptions ---")
        for i in range(num_joints):
            joint_info = p.getJointInfo(valkyrie, i, physicsClientId=cid)
            joint_id = joint_info[0]
            joint_name = joint_info[1].decode("utf-8")
            joint_type = joint_info[2]
            link_name = joint_info[12].decode("utf-8") # Link name associated with this joint
            print(f"ID: {joint_id}, Name: {joint_name}, Type: {joint_type} ({'Revolute' if joint_type == p.JOINT_REVOLUTE else 'Prismatic' if joint_type == p.JOINT_PRISMATIC else 'Fixed'}), Link: {link_name}")
    except Exception as e:
        print(f"Error loading Valkyrie description to list joints: {e}")
        print("Please ensure 'robot_descriptions' library is installed and 'valkyrie_description' is available.")
    finally:
        p.disconnect(cid)
    print("\n--- End Joint List ---")
    print("Carefully select `VALKYRIE_CONTROLLABLE_JOINT_NAMES` and foot/torso link names based on this output.")


if __name__ == '__main__':
    # Run this first to identify correct joint names and foot/torso links!
    list_valkyrie_joints_from_description()
    input("\nReview joint list and update VALKYRIE_CONTROLLABLE_JOINT_NAMES and foot/torso link names in valkyrie_env.py. Press Enter to test environment...")

    # Minimal test of the environment
    env = ValkyrieStandEnv(render_mode='human')
    obs, info = env.reset()
    print(f"Observation space: {env.observation_space.shape}")
    print(f"Action space: {env.action_space.shape}")
    print(f"Initial observation shape: {obs.shape}")

    # Simple random action loop to test the environment
    print("Running random actions for 500 steps...")
    for i in range(500):
        action = env.action_space.sample() # Random action
        obs, reward, terminated, truncated, info = env.step(action)
        if i % 50 == 0:
            print(f"Step {i}: Reward: {reward:.4f}, Terminated: {terminated}")
        if terminated or truncated:
            print(f"Episode ended at step {i}.")
            obs, info = env.reset()
    env.close()