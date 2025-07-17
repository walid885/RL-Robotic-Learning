# simulation/robot_loader.py - Fixed robot loading and configuration
import pybullet as p
import time
from typing import List, Dict
# Ensure this import works:
from robot_descriptions.loaders.pybullet import load_robot_description 
from sim.physics_engine import PhysicsEngine
from sim.simulation_config import SimulationConfig
# from simulation.joint_utils import JointAnalyzer # Not directly used in this snippet

class RobotLoader:
    """Handles robot loading and initial configuration with proper spawning."""
    
    def __init__(self, physics_engine: PhysicsEngine, config: SimulationConfig):
        self.physics_engine = physics_engine
        self.config = config
    
    def get_robot_bounding_box(self, robot_id: int) -> tuple[float, float]:
        """Get robot's bounding box to determine proper spawn height."""
        min_z = float('inf')
        max_z = float('-inf')
        
        # Check base link
        base_pos, _ = p.getBasePositionAndOrientation(robot_id, physicsClientId=self.physics_engine.client_id)
        min_z = min(min_z, base_pos[2])
        max_z = max(max_z, base_pos[2])
        
        # Check all links
        num_joints = p.getNumJoints(robot_id, physicsClientId=self.physics_engine.client_id)
        for i in range(num_joints):
            link_state = p.getLinkState(robot_id, i, physicsClientId=self.physics_engine.client_id)
            link_pos = link_state[0]
            min_z = min(min_z, link_pos[2])
            max_z = max(max_z, link_pos[2])
        
        return min_z, max_z
    
    def load_robot(self, description: str, position: List[float] = None, orientation: List[float] = None) -> int:
        """Load robot from description and set proper initial position."""
        if position is None:
            position = [0, 0, self.config.robot_height]
        if orientation is None:
            orientation = [0, 0, 0, 1] # Identity quaternion
            
        robot_id = load_robot_description(description, physicsClient=self.physics_engine.client_id)
        p.resetBasePositionAndOrientation(robot_id, position, orientation, physicsClientId=self.physics_engine.client_id)
        
        self._configure_robot_dynamics(robot_id)
        self._verify_ground_clearance(robot_id)
        self._initialize_joint_positions(robot_id)
        self._stabilize_robot(robot_id)
        
        return robot_id
    
    def _configure_robot_dynamics(self, robot_id: int) -> None:
        """Configure robot dynamics for stability."""
        for i in range(p.getNumJoints(robot_id, physicsClientId=self.physics_engine.client_id)):
            p.changeDynamics(
                robot_id, i,
                linearDamping=self.config.linear_damping,
                angularDamping=self.config.angular_damping,
                maxJointVelocity=self.config.max_joint_velocity,
                physicsClientId=self.physics_engine.client_id
            )
    
    def _verify_ground_clearance(self, robot_id: int) -> None:
        """Verify robot has proper ground clearance."""
        min_z, _ = self.get_robot_bounding_box(robot_id)
        base_pos, _ = p.getBasePositionAndOrientation(robot_id, physicsClientId=self.physics_engine.client_id)
        
        if min_z <= 0.05: # Threshold for too close to ground
            print(f"WARNING: Robot may be too close to ground (min_z={min_z:.3f})")
            # Adjust position to ensure clearance, relative to current base_pos
            new_pos = [base_pos[0], base_pos[1], base_pos[2] + (0.05 - min_z) + 0.01] # Add a small buffer
            _, base_orn = p.getBasePositionAndOrientation(robot_id, physicsClientId=self.physics_engine.client_id)
            p.resetBasePositionAndOrientation(robot_id, new_pos, base_orn, physicsClientId=self.physics_engine.client_id)
            print(f"Adjusted robot position to: {new_pos}")
    
    def _initialize_joint_positions(self, robot_id: int) -> None:
        """Initialize all joints to position control with current positions."""
        for i in range(p.getNumJoints(robot_id, physicsClientId=self.physics_engine.client_id)):
            joint_info = p.getJointInfo(robot_id, i, physicsClientId=self.physics_engine.client_id)
            if joint_info[2] in [p.JOINT_REVOLUTE, p.JOINT_PRISMATIC]:
                current_position = p.getJointState(robot_id, i, physicsClientId=self.physics_engine.client_id)[0]
                p.setJointMotorControl2(
                    robot_id, i, p.POSITION_CONTROL,
                    targetPosition=current_position,
                    force=self.config.default_joint_force, # Use force from config
                    physicsClientId=self.physics_engine.client_id
                )
    
    def _stabilize_robot(self, robot_id: int) -> None:
        """Stabilize robot by stepping simulation."""
        for _ in range(self.config.stabilization_steps):
            self.physics_engine.step_simulation()
            # No time.sleep here, as PyBullet handles simulation speed automatically
            # and `time.sleep` would block RL training (unless it's for visual debugging)
            # if self.physics_engine.connection_mode == p.GUI:
            #    time.sleep(1.0 / self.config.simulation_rate) # Only for visual GUI


# Note: The `setup_simulation` method provided in your snippet seems to be part of another class,
# perhaps a `SimulationManager` or `EnvironmentManager`. We'll focus on passing these
# instances to the RL environment.