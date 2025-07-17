# simulation/robot_loader.py - Robot loading and configuration
import pybullet as p
from typing import List
from robot_descriptions.loaders.pybullet import load_robot_description
from simulation.physics_engine import PhysicsEngine
from simulation.simulation_config import SimulationConfig


class RobotLoader:
    """Handles robot loading and initial configuration."""
    
    def __init__(self, physics_engine: PhysicsEngine):
        self.physics_engine = physics_engine
        self.config = SimulationConfig()
    
    def load_robot(self, description: str, position: List[float]) -> int:
        """Load robot from description and set initial position."""
        robot_id = load_robot_description(description)
        p.resetBasePositionAndOrientation(robot_id, position, [0, 0, 0, 1])
        
        self._configure_robot_dynamics(robot_id)
        return robot_id
    
    def _configure_robot_dynamics(self, robot_id: int) -> None:
        """Configure robot dynamics for stability."""
        num_joints = p.getNumJoints(robot_id)
        
        for i in range(num_joints):
            p.changeDynamics(
                robot_id, i,
                linearDamping=self.config.linear_damping,
                angularDamping=self.config.angular_damping,
                maxJointVelocity=self.config.max_joint_velocity
            )


