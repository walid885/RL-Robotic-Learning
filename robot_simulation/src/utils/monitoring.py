# src/utils/monitoring.py
import pybullet as p
from typing import Tuple

class StabilityMonitor:
    """Robot stability monitoring and assessment."""
    
    def __init__(self, robot_id: int):
        self.robot_id = robot_id
        
    def check_stability(self, step: int) -> bool:
        """Check if robot is stable and upright."""
        pos, orn = p.getBasePositionAndOrientation(self.robot_id)
        linear_vel, angular_vel = p.getBaseVelocity(self.robot_id)
        
        euler = p.getEulerFromQuaternion(orn)
        
        # Stability criteria
        height_ok = pos[2] > 0.8
        tilt_ok = abs(euler[0]) < 0.3 and abs(euler[1]) < 0.3
        vel_ok = sum(v**2 for v in linear_vel)**0.5 < 1.0
        
        if step % 2000 == 0:
            print(f"Step {step}: H={pos[2]:.3f}, Tilt=({euler[0]:.3f},{euler[1]:.3f}), Vel={sum(v**2 for v in linear_vel)**0.5:.3f}")
        
        return height_ok and tilt_ok and vel_ok
    
    def get_robot_state(self) -> Tuple[list, list, list, list]:
        """Get current robot state."""
        pos, orn = p.getBasePositionAndOrientation(self.robot_id)
        linear_vel, angular_vel = p.getBaseVelocity(self.robot_id)
        return pos, orn, linear_vel, angular_vel