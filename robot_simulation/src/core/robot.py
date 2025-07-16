# src/core/robot.py
import pybullet as p
from robot_descriptions.loaders.pybullet import load_robot_description
from typing import List, Dict
from ..utils.data_structures import JointInfo

class Robot:
    """Main robot class handling loading and basic operations."""
    
    def __init__(self, description: str, position: List[float]):
        self.description = description
        self.position = position
        self.robot_id = None
        self.joints = {}
        
    def load(self, physics_client) -> int:
        """Load robot with stability settings."""
        self.robot_id = load_robot_description(self.description)
        p.resetBasePositionAndOrientation(self.robot_id, self.position, [0, 0, 0, 1])
        
        self._configure_dynamics()
        return self.robot_id
    
    def _configure_dynamics(self):
        """Configure robot dynamics for stability."""
        num_joints = p.getNumJoints(self.robot_id)
        
        # Enhanced base dynamics
        p.changeDynamics(self.robot_id, -1, 
                        linearDamping=0.4,
                        angularDamping=0.4,
                        mass=100.0,
                        localInertiaDiagonal=[2, 2, 2])
        
        # Enhanced joint dynamics
        for i in range(num_joints):
            joint_info = p.getJointInfo(self.robot_id, i)
            joint_name = joint_info[1].decode('utf-8').lower()
            
            if 'leg' in joint_name or 'ankle' in joint_name or 'hip' in joint_name:
                p.changeDynamics(self.robot_id, i, 
                                linearDamping=0.5, 
                                angularDamping=0.5,
                                maxJointVelocity=0.5,
                                jointDamping=0.2,
                                frictionAnchor=1)
            else:
                p.changeDynamics(self.robot_id, i, 
                                linearDamping=0.3, 
                                angularDamping=0.3,
                                maxJointVelocity=0.8,
                                jointDamping=0.15,
                                frictionAnchor=1)
