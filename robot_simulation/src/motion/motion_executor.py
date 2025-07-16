# src/motion/motion_executor.py
import time
from typing import Dict, Callable
from control.joint_controller import JointController

class MotionExecutor:
    """Execute complex motions with timing and coordination."""
    
    def __init__(self, robot_id: int, simulation_rate: float = 240.0):
        self.robot_id = robot_id
        self.simulation_rate = simulation_rate
        self.joint_controller = JointController(robot_id)
        self.motion_callbacks = {}
        
    def register_motion_callback(self, name: str, callback: Callable):
        """Register a motion callback function."""
        self.motion_callbacks[name] = callback
    
    def execute_motion_sequence(self, sequence: List[Dict], step_count: int) -> None:
        """Execute a sequence of motions."""
        for motion in sequence:
            motion_name = motion.get('name')
            if motion_name in self.motion_callbacks:
                self.motion_callbacks[motion_name](motion, step_count)
    
    def smooth_transition(self, joint_id: int, current_pos: float, target_pos: float, 
                         transition_steps: int, current_step: int) -> float:
        """Calculate smooth transition between positions."""
        if current_step >= transition_steps:
            return target_pos
        
        # Smooth interpolation
        t = current_step / transition_steps
        smooth_t = 3 * t * t - 2 * t * t * t  # Smooth step function
        return current_pos + (target_pos - current_pos) * smooth_t

