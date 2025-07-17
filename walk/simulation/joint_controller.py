# simulation/joint_controller.py - Joint control operations
import pybullet as p
from typing import List, Dict, Tuple
from models.joint_info import JointInfo
from simulation.simulation_config import SimulationConfig


class JointController:
    """Handles joint position control and movements."""
    
    def __init__(self):
        self.config = SimulationConfig()
    
    def set_joint_position_control(self, robot_id: int, joint_id: int, target_position: float,
                                 force: float = None, position_gain: float = None, 
                                 velocity_gain: float = None) -> None:
        """Set position control for a single joint."""
        force = force or self.config.default_force
        position_gain = position_gain or self.config.position_gain
        velocity_gain = velocity_gain or self.config.velocity_gain
        
        p.setJointMotorControl2(
            robot_id, joint_id, p.POSITION_CONTROL,
            targetPosition=target_position,
            force=force,
            positionGain=position_gain,
            velocityGain=velocity_gain
        )
    
    def initialize_joint_positions(self, robot_id: int, joints: List[JointInfo]) -> None:
        """Initialize all joints to position control with current positions."""
        for joint in joints:
            self.set_joint_position_control(
                robot_id, joint.id, joint.current_position, force=200
            )
    
    def apply_balance_control(self, robot_id: int, balance_joints: List[JointInfo], 
                            initial_positions: Dict[int, float]) -> None:
        """Apply strong position control to balance joints."""
        for joint in balance_joints:
            self.set_joint_position_control(
                robot_id, joint.id, initial_positions[joint.id],
                force=self.config.balance_force,
                position_gain=self.config.balance_position_gain,
                velocity_gain=self.config.balance_velocity_gain
            )
    
    def apply_stabilization_control(self, robot_id: int, joints: List[JointInfo],
                                  arm_joints: List[JointInfo], balance_joints: List[JointInfo],
                                  initial_positions: Dict[int, float]) -> None:
        """Apply stabilization control to non-arm, non-balance joints."""
        arm_ids = {j.id for j in arm_joints}
        balance_ids = {j.id for j in balance_joints}
        
        for joint in joints:
            if joint.id not in arm_ids and joint.id not in balance_ids:
                self.set_joint_position_control(
                    robot_id, joint.id, initial_positions[joint.id], force=800
                )
    
    def apply_wave_motion(self, robot_id: int, arm_joints: List[JointInfo],
                         target_positions: List[float], wave_offset: float,
                         joint_limits: List[Tuple[float, float]]) -> None:
        """Apply gentle wave motion to arm joints."""
        wave_multipliers = [1.0, -0.5, 0.2]
        
        for i, joint in enumerate(arm_joints[:3]):
            if i < len(target_positions):
                multiplier = wave_multipliers[i] if i < len(wave_multipliers) else 0.1
                target_pos = target_positions[i] + wave_offset * multiplier
                
                if i < len(joint_limits):
                    lower, upper = joint_limits[i]
                    if lower < upper and abs(lower) < 100 and abs(upper) < 100:
                        target_pos = max(lower, min(upper, target_pos))
                
                self.set_joint_position_control(
                    robot_id, joint.id, target_pos, force=200,
                    position_gain=0.3, velocity_gain=0.1
                )


