# simulation/robot_loader.py - Fixed robot loading and configuration
import pybullet as p
import time
from typing import List, Dict
from robot_descriptions.loaders.pybullet import load_robot_description
from simulation.physics_engine import PhysicsEngine
from simulation.simulation_config import SimulationConfig
from simulation.joint_utils import JointAnalyzer


class RobotLoader:
    """Handles robot loading and initial configuration with proper spawning."""
    
    def __init__(self, physics_engine: PhysicsEngine):
        self.physics_engine = physics_engine
        self.config = SimulationConfig()
    
    def get_robot_bounding_box(self, robot_id: int) -> tuple[float, float]:
        """Get robot's bounding box to determine proper spawn height."""
        min_z = float('inf')
        max_z = float('-inf')
        
        # Check base link
        base_pos, _ = p.getBasePositionAndOrientation(robot_id)
        min_z = min(min_z, base_pos[2])
        max_z = max(max_z, base_pos[2])
        
        # Check all links
        num_joints = p.getNumJoints(robot_id)
        for i in range(num_joints):
            link_state = p.getLinkState(robot_id, i)
            link_pos = link_state[0]
            min_z = min(min_z, link_pos[2])
            max_z = max(max_z, link_pos[2])
        
        return min_z, max_z
    
    def load_robot(self, description: str, position: List[float] = None) -> int:
        """Load robot from description and set proper initial position."""
        if position is None:
            position = [0, 0, self.config.robot_height]
        
        robot_id = load_robot_description(description)
        p.resetBasePositionAndOrientation(robot_id, position, [0, 0, 0, 1])
        
        self._configure_robot_dynamics(robot_id)
        self._verify_ground_clearance(robot_id)
        self._initialize_joint_positions(robot_id)
        self._stabilize_robot(robot_id)
        
        return robot_id
    
    def _configure_robot_dynamics(self, robot_id: int) -> None:
        """Configure robot dynamics for stability."""
        for i in range(p.getNumJoints(robot_id)):
            p.changeDynamics(
                robot_id, i,
                linearDamping=self.config.linear_damping,
                angularDamping=self.config.angular_damping,
                maxJointVelocity=self.config.max_joint_velocity
            )
    
    def _verify_ground_clearance(self, robot_id: int) -> None:
        """Verify robot has proper ground clearance."""
        min_z, _ = self.get_robot_bounding_box(robot_id)
        base_pos, _ = p.getBasePositionAndOrientation(robot_id)
        
        if min_z <= 0.05:
            print(f"WARNING: Robot may be too close to ground (min_z={min_z:.3f})")
            new_pos = [base_pos[0], base_pos[1], base_pos[2] + 0.3]
            p.resetBasePositionAndOrientation(robot_id, new_pos, [0, 0, 0, 1])
            print(f"Adjusted robot position to: {new_pos}")
    
    def _initialize_joint_positions(self, robot_id: int) -> None:
        """Initialize all joints to position control with current positions."""
        for i in range(p.getNumJoints(robot_id)):
            joint_info = p.getJointInfo(robot_id, i)
            if joint_info[2] in [p.JOINT_REVOLUTE, p.JOINT_PRISMATIC]:
                current_position = p.getJointState(robot_id, i)[0]
                p.setJointMotorControl2(
                    robot_id, i, p.POSITION_CONTROL,
                    targetPosition=current_position,
                    force=200
                )
    
    def _stabilize_robot(self, robot_id: int) -> None:
        """Stabilize robot by stepping simulation."""
        for _ in range(self.config.stabilization_steps):
            p.stepSimulation()
            time.sleep(1.0 / self.config.simulation_rate)