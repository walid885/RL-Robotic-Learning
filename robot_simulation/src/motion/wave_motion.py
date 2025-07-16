# src/motion/wave_motion.py
import math
import time
import pybullet as p
from ..control.stabilizer import RobotStabilizer
from ..utils.data_structures import SimulationConfig

class WaveMotion:
    """Waving motion implementation with stability management."""
    
    def __init__(self, robot, config: SimulationConfig):
        self.robot = robot
        self.config = config
        self.stabilizer = None
        
    def run_simulation(self, stabilizer: RobotStabilizer) -> None:
        """Run the complete waving simulation."""
        self.stabilizer = stabilizer
        
        print(f"Found {len(stabilizer.movable_joints)} movable joints")
        for category, joints in stabilizer.joint_categories.items():
            if joints:
                print(f"  {category}: {len(joints)} joints")
        
        # Stabilization phase
        stabilizer.stabilize(self.config.stabilization_steps)
        
        # Motion phase
        self._execute_waving_motion()
    
    def _execute_waving_motion(self) -> None:
        """Execute the waving motion with balance maintenance."""
        print("Starting goodbye waving motion...")
        
        step_count = 0
        warmup_steps = 3000
        
        try:
            while True:
                # Always maintain balance
                self.stabilizer.maintain_balance(step_count)
                
                # Add waving motion after warmup
                if step_count > warmup_steps:
                    self._apply_wave_motion(step_count - warmup_steps)
                
                p.stepSimulation()
                time.sleep(1.0 / self.config.simulation_rate)
                step_count += 1
                
                # Stability monitoring
                if step_count % 2000 == 0:
                    is_stable = self.stabilizer.stability_monitor.check_stability(step_count)
                    if not is_stable:
                        print("Robot becoming unstable - increasing stabilization")
                    
                    if step_count == warmup_steps:
                        print("Beginning waving motion...")
        
        except KeyboardInterrupt:
            print("Simulation stopped by user")
        except Exception as e:
            print(f"Simulation error: {e}")
    
    def _apply_wave_motion(self, motion_step: int) -> None:
        """Apply waving motion to right arm."""
        t = motion_step * 0.01
        wave_factor = min(1.0, motion_step / 2000.0)
        
        # Shoulder lift
        shoulder_wave = 0.3 * wave_factor
        for joint in self.stabilizer.joint_categories['shoulder_right']:
            if 'pitch' in joint.name.lower() or 'y' in joint.name.lower():
                target = self.stabilizer.stable_positions[joint.id] + shoulder_wave
                self.stabilizer.joint_controller.set_position_control(
                    joint.id, target, force=400, position_gain=0.3, velocity_gain=0.15
                )
        
        # Elbow motion
        elbow_wave = math.sin(t * self.config.wave_frequency * 2 * math.pi) * self.config.elbow_wave_amplitude
        for joint in self.stabilizer.joint_categories['elbow_right']:
            target = self.stabilizer.stable_positions[joint.id] + (elbow_wave * wave_factor)
            self.stabilizer.joint_controller.set_position_control(
                joint.id, target, force=300, position_gain=0.25, velocity_gain=0.12
            )
        
        # Wrist motion
        wrist_wave = math.sin(t * 1.2 * 2 * math.pi) * self.config.wrist_wave_amplitude
        for joint in self.stabilizer.joint_categories['wrist_right']:
            target = self.stabilizer.stable_positions[joint.id] + (wrist_wave * wave_factor)
            self.stabilizer.joint_controller.set_position_control(
                joint.id, target, force=200, position_gain=0.4, velocity_gain=0.1
            )

