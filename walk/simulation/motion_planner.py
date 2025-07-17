# simulation/motion_planner.py - Motion planning and calculations
import math
from typing import List, Tuple
from models.joint_info import JointInfo
from simulation.simulation_config import SimulationConfig


class MotionPlanner:
    """Handles motion planning and trajectory calculations."""
    
    def __init__(self, config: SimulationConfig):
        self.config = config
    
    def calculate_safe_target_position(self, joint: JointInfo, offset: float = 0.05) -> float:
        """Calculate safe target position within joint limits."""
        current = joint.current_position
        lower, upper = joint.limits
        
        if joint.is_valid_limits():
            safe_range = joint.get_safe_range()
            target = current + safe_range * 0.3
            return max(lower, min(upper, target))
        else:
            return current
    
    def calculate_wave_motion(self, step: int, frequency: float = None, 
                            amplitude: float = None) -> float:
        """Calculate wave motion offset."""
        frequency = frequency or self.config.wave_frequency
        amplitude = amplitude or self.config.wave_amplitude
        
        t = step * 0.01
        return math.sin(t * frequency * 2 * math.pi) * amplitude
    
    def calculate_target_positions(self, arm_joints: List[JointInfo]) -> List[float]:
        """Calculate target positions for arm joints."""
        return [
            self.calculate_safe_target_position(joint) 
            for joint in arm_joints[:3]
        ]
    
    def get_joint_limits(self, arm_joints: List[JointInfo]) -> List[Tuple[float, float]]:
        """Extract joint limits for arm joints."""
        return [joint.limits for joint in arm_joints[:3]]


