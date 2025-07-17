# simulation/stabilizer.py - Robot stabilization
import pybullet as p
import time
from typing import List
from models.joint_info import JointInfo
from simulation.simulation_config import SimulationConfig


class RobotStabilizer:
    """Handles robot stabilization during simulation."""
    
    def __init__(self, config: SimulationConfig):
        self.config = config
    
    def stabilize_robot(self, robot_id: int, steps: int = None, rate: float = None) -> None:
        """Stabilize robot for specified number of steps."""
        steps = steps or self.config.stabilization_steps
        rate = rate or self.config.simulation_rate
        
        print("Stabilizing robot...")
        
        for i in range(steps):
            p.stepSimulation()
            
            if i % 500 == 0:
                pos, _ = p.getBasePositionAndOrientation(robot_id)
                print(f"Stabilization step {i}: Robot Z position = {pos[2]:.3f}")
            
            time.sleep(1.0 / rate)
        
        print("Robot stabilization complete.")


