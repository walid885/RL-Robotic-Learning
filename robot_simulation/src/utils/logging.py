# src/utils/logging.py
import time
from typing import Dict, Any
from dataclasses import asdict

class SimulationLogger:
    """Logging and debugging utilities for simulation."""
    
    def __init__(self, log_level: str = "INFO"):
        self.log_level = log_level
        self.start_time = time.time()
        self.logs = []
        
    def log(self, level: str, message: str, data: Dict[Any, Any] = None) -> None:
        """Log a message with optional data."""
        timestamp = time.time() - self.start_time
        log_entry = {
            "timestamp": timestamp,
            "level": level,
            "message": message,
            "data": data or {}
        }
        
        self.logs.append(log_entry)
        
        if self._should_print(level):
            print(f"[{timestamp:.3f}] {level}: {message}")
            if data:
                print(f"  Data: {data}")
    
    def _should_print(self, level: str) -> bool:
        """Determine if message should be printed based on log level."""
        levels = {"DEBUG": 0, "INFO": 1, "WARNING": 2, "ERROR": 3}
        return levels.get(level, 1) >= levels.get(self.log_level, 1)
    
    def log_robot_state(self, robot_id: int, step: int) -> None:
        """Log current robot state."""
        import pybullet as p
        pos, orn = p.getBasePositionAndOrientation(robot_id)
        linear_vel, angular_vel = p.getBaseVelocity(robot_id)
        euler = p.getEulerFromQuaternion(orn)
        
        state_data = {
            "step": step,
            "position": pos,
            "orientation": euler,
            "linear_velocity": linear_vel,
            "angular_velocity": angular_vel
        }
        
        self.log("DEBUG", f"Robot state at step {step}", state_data)
    
    def log_joint_states(self, robot_id: int, joint_ids: List[int]) -> None:
        """Log joint states for debugging."""
        import pybullet as p
        joint_states = {}
        
        for joint_id in joint_ids:
            joint_state = p.getJointState(robot_id, joint_id)
            joint_info = p.getJointInfo(robot_id, joint_id)
            joint_name = joint_info[1].decode('utf-8')
            
            joint_states[joint_name] = {
                "position": joint_state[0],
                "velocity": joint_state[1],
                "force": joint_state[3]
            }
        
        self.log("DEBUG", "Joint states", joint_states)
    
    def save_logs(self, filename: str) -> None:
        """Save logs to file."""
        import json
        with open(filename, 'w') as f:
            json.dump(self.logs, f, indent=2)
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics."""
        total_time = time.time() - self.start_time
        total_logs = len(self.logs)
        
        return {
            "total_time": total_time,
            "total_logs": total_logs,
            "logs_per_second": total_logs / total_time if total_time > 0 else 0
        }