# src/control/balance_controller.py
import pybullet as p
from typing import Dict, List, Tuple
from ..utils.data_structures import JointInfo, WaveMotionConfig
from ..utils.monitoring import StabilityMonitor

class BalanceController:
    """Advanced balance control and recovery system."""
    
    def __init__(self, robot_id: int, wave_config: WaveMotionConfig):
        self.robot_id = robot_id
        self.wave_config = wave_config
        self.stability_monitor = StabilityMonitor(robot_id)
        self.balance_state = "stable"
        
    def assess_balance(self, step: int) -> str:
        """Assess current balance state."""
        pos, orn, linear_vel, angular_vel = self.stability_monitor.get_robot_state()
        euler = p.getEulerFromQuaternion(orn)
        
        # Calculate balance metrics
        tilt_magnitude = (euler[0]**2 + euler[1]**2)**0.5
        velocity_magnitude = sum(v**2 for v in linear_vel)**0.5
        
        # Determine balance state
        if tilt_magnitude > 0.4 or velocity_magnitude > 1.5:
            self.balance_state = "unstable"
        elif tilt_magnitude > 0.2 or velocity_magnitude > 0.8:
            self.balance_state = "marginal"
        else:
            self.balance_state = "stable"
        
        return self.balance_state
    
    def apply_corrective_forces(self, joint_categories: Dict[str, List[JointInfo]], 
                               stable_positions: Dict[int, float]) -> None:
        """Apply corrective forces based on balance state."""
        force_multiplier = {
            "stable": 1.0,
            "marginal": 1.5,
            "unstable": 2.0
        }
        
        multiplier = force_multiplier.get(self.balance_state, 1.0)
        
        # Apply stronger forces to critical joints
        for joint in joint_categories['hip'] + joint_categories['knee'] + joint_categories['ankle']:
            enhanced_force = self.wave_config.leg_force * multiplier
            p.setJointMotorControl2(
                self.robot_id, joint.id, p.POSITION_CONTROL,
                targetPosition=stable_positions[joint.id],
                force=enhanced_force,
                positionGain=0.9 * multiplier,
                velocityGain=0.6 * multiplier
            )
    
    def emergency_stabilization(self, joint_categories: Dict[str, List[JointInfo]], 
                              stable_positions: Dict[int, float]) -> None:
        """Emergency stabilization when robot is falling."""
        print("Emergency stabilization activated!")
        
        # Maximum forces to all critical joints
        for joint in joint_categories['hip'] + joint_categories['knee'] + joint_categories['ankle']:
            p.setJointMotorControl2(
                self.robot_id, joint.id, p.POSITION_CONTROL,
                targetPosition=stable_positions[joint.id],
                force=self.wave_config.leg_force * 3.0,
                positionGain=1.0,
                velocityGain=0.8
            )
        
        # Stop all arm motions
        for joint in joint_categories['shoulder_right'] + joint_categories['elbow_right'] + joint_categories['wrist_right']:
            p.setJointMotorControl2(
                self.robot_id, joint.id, p.POSITION_CONTROL,
                targetPosition=stable_positions[joint.id],
                force=self.wave_config.base_stabilization_force * 2.0,
                positionGain=0.8,
                velocityGain=0.5
            )

