import pybullet as p

class JointController:
    """Low-level joint control operations."""
    
    def __init__(self, robot_id: int):
        self.robot_id = robot_id
        
    def set_position_control(self, joint_id: int, target_position: float, 
                           force: float = 500, position_gain: float = 0.3, 
                           velocity_gain: float = 0.1) -> None:
        """Enhanced joint position control with velocity limiting."""
        p.setJointMotorControl2(
            self.robot_id, joint_id, p.POSITION_CONTROL,
            targetPosition=target_position,
            force=force,
            positionGain=position_gain,
            velocityGain=velocity_gain,
            maxVelocity=0.3
        )
