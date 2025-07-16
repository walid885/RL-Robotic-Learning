# src/control/stabilizer.py
import time
from typing import Dict, List
from ..core.joint_manager import JointManager
from ..core.joint_controller import JointController
from ..utils.data_structures import JointInfo, WaveMotionConfig, SimulationConfig
from ..utils.monitoring import StabilityMonitor

class RobotStabilizer:
    """Multi-phase stabilization system for robot balance."""
    
    def __init__(self, robot, config: SimulationConfig):
        self.robot = robot
        self.config = config
        self.joint_manager = JointManager(robot.robot_id)
        self.joint_controller = JointController(robot.robot_id)
        self.stability_monitor = StabilityMonitor(robot.robot_id)
        self.wave_config = WaveMotionConfig()
        
        # Initialize joints
        self.movable_joints = self.joint_manager.discover_joints()
        self.joint_categories = self.joint_manager.categorize_joints(self.movable_joints)
        self.stable_positions = self._calculate_stable_positions()
        
    def _calculate_stable_positions(self) -> Dict[int, float]:
        """Calculate optimal standing pose for maximum stability."""
        positions = {}
        
        # Hip joints: slight forward lean
        for joint in self.joint_categories['hip']:
            if 'pitch' in joint.name.lower() or 'y' in joint.name.lower():
                positions[joint.id] = -0.1
            else:
                positions[joint.id] = 0.0
        
        # Knee joints: slight bend
        for joint in self.joint_categories['knee']:
            positions[joint.id] = 0.05
        
        # Ankle joints: flat stance
        for joint in self.joint_categories['ankle']:
            positions[joint.id] = 0.0
        
        # Torso: upright
        for joint in self.joint_categories['torso']:
            positions[joint.id] = 0.0
        
        # Arms: neutral positions
        for joint in self.joint_categories['shoulder_left'] + self.joint_categories['shoulder_right']:
            positions[joint.id] = 0.0
        for joint in self.joint_categories['elbow_left'] + self.joint_categories['elbow_right']:
            positions[joint.id] = 0.1
        for joint in self.joint_categories['wrist_right']:
            positions[joint.id] = 0.0
        
        # Other joints: current position
        for joint in self.joint_categories['other']:
            positions[joint.id] = joint.current_position
        
        return positions
    
    def stabilize(self, steps: int) -> None:
        """Run enhanced stabilization phase."""
        print("Enhanced stabilization starting...")
        
        for i in range(steps):
            self._apply_stabilization_control(i)
            import pybullet as p
            p.stepSimulation()
            
            if i % 1000 == 0:
                is_stable = self.stability_monitor.check_stability(i)
                if not is_stable and i > 2000:
                    print(f"Warning: Robot unstable at step {i}")
            
            time.sleep(1.0 / self.config.simulation_rate)
        
        print("Stabilization complete!")
    
    def _apply_stabilization_control(self, step: int) -> None:
        """Apply progressive stabilization control."""
        force_ramp = min(1.0, step / 2000.0)
        
        # Critical balance joints
        for joint in self.joint_categories['hip']:
            self.joint_controller.set_position_control(
                joint.id, self.stable_positions[joint.id], 
                force=self.wave_config.leg_force * force_ramp,
                position_gain=0.8, velocity_gain=0.5
            )
        
        for joint in self.joint_categories['knee']:
            self.joint_controller.set_position_control(
                joint.id, self.stable_positions[joint.id], 
                force=self.wave_config.leg_force * force_ramp,
                position_gain=0.8, velocity_gain=0.5
            )
        
        for joint in self.joint_categories['ankle']:
            self.joint_controller.set_position_control(
                joint.id, self.stable_positions[joint.id], 
                force=self.wave_config.leg_force * force_ramp,
                position_gain=0.9, velocity_gain=0.6
            )
        
        for joint in self.joint_categories['torso']:
            self.joint_controller.set_position_control(
                joint.id, self.stable_positions[joint.id], 
                force=self.wave_config.torso_force * force_ramp,
                position_gain=0.7, velocity_gain=0.4
            )
        
        # Left arm stabilization
        for joint in self.joint_categories['shoulder_left']:
            self.joint_controller.set_position_control(
                joint.id, self.stable_positions[joint.id], 
                force=self.wave_config.base_stabilization_force * force_ramp,
                position_gain=0.6, velocity_gain=0.3
            )
        
        for joint in self.joint_categories['elbow_left']:
            self.joint_controller.set_position_control(
                joint.id, self.stable_positions[joint.id], 
                force=self.wave_config.base_stabilization_force * force_ramp,
                position_gain=0.6, velocity_gain=0.3
            )
    
    def maintain_balance(self, step: int) -> None:
        """Maintain balance during motion."""
        self._apply_stabilization_control(step)

