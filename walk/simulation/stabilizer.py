# simulation/stabilizer.py - Enhanced robot stabilization with ground contact
import pybullet as p
import time
from typing import List, Dict
from models.joint_info import JointInfo
from simulation.simulation_config import SimulationConfig


class RobotStabilizer:
    """Handles robot stabilization during simulation with ground contact verification."""
    
    def __init__(self, config: SimulationConfig):
        self.config = config
    
    def check_ground_contact(self, robot_id: int) -> bool:
        """Check if robot has proper ground contact."""
        contact_points = p.getContactPoints(bodyA=robot_id, bodyB=0)  # 0 is ground plane
        
        # Check for foot contact
        foot_contact = False
        for contact in contact_points:
            link_index = contact[3]
            if link_index >= 0:
                joint_info = p.getJointInfo(robot_id, link_index)
                joint_name = joint_info[1].decode('utf-8').lower()
                if 'foot' in joint_name or 'ankle' in joint_name:
                    foot_contact = True
                    break
        
        return foot_contact or len(contact_points) > 0
    
    def get_robot_stability_metrics(self, robot_id: int) -> Dict[str, float]:
        """Get stability metrics for the robot."""
        base_pos, base_orn = p.getBasePositionAndOrientation(robot_id)
        base_vel, base_ang_vel = p.getBaseVelocity(robot_id)
        
        # Calculate stability metrics
        return {
            'height': base_pos[2],
            'velocity': sum(v**2 for v in base_vel)**0.5,
            'angular_velocity': sum(v**2 for v in base_ang_vel)**0.5,
            'tilt': abs(base_orn[0]) + abs(base_orn[1])  # Roll and pitch
        }
    
    def apply_emergency_stabilization(self, robot_id: int) -> None:
        """Apply emergency stabilization if robot is falling."""
        base_pos, base_orn = p.getBasePositionAndOrientation(robot_id)
        
        # If robot is too low or tilted, reset position
        if base_pos[2] < 0.5:
            print("Emergency: Robot too low, resetting position")
            new_pos = [base_pos[0], base_pos[1], 1.5]
            p.resetBasePositionAndOrientation(robot_id, new_pos, [0, 0, 0, 1])
        
        # Apply strong damping to stop violent motion
        p.changeDynamics(robot_id, -1, linearDamping=0.9, angularDamping=0.9)
    
    def stabilize_robot(self, robot_id: int, steps: int = None, rate: float = None) -> None:
        """Enhanced stabilization with ground contact and stability monitoring."""
        steps = steps or self.config.stabilization_steps
        rate = rate or self.config.simulation_rate
        
        print("Enhanced robot stabilization starting...")
        
        # Phase 1: Initial settling (25% of steps)
        settling_steps = steps // 4
        print(f"Phase 1: Initial settling ({settling_steps} steps)")
        
        for i in range(settling_steps):
            # Apply strong damping initially
            if i < settling_steps // 2:
                p.changeDynamics(robot_id, -1, linearDamping=0.8, angularDamping=0.8)
            else:
                p.changeDynamics(robot_id, -1, linearDamping=0.3, angularDamping=0.3)
            
            p.stepSimulation()
            time.sleep(1.0 / rate)
            
            # Check for instability
            metrics = self.get_robot_stability_metrics(robot_id)
            if metrics['height'] < 0.3 or metrics['velocity'] > 2.0:
                self.apply_emergency_stabilization(robot_id)
        
        # Phase 2: Ground contact establishment (50% of steps)
        contact_steps = steps // 2
        print(f"Phase 2: Ground contact establishment ({contact_steps} steps)")
        
        for i in range(contact_steps):
            p.stepSimulation()
            time.sleep(1.0 / rate)
            
            # Monitor ground contact
            if i % 200 == 0:
                has_contact = self.check_ground_contact(robot_id)
                metrics = self.get_robot_stability_metrics(robot_id)
                
                print(f"Step {settling_steps + i}: "
                      f"Height={metrics['height']:.3f}, "
                      f"Ground contact={has_contact}")
                
                if not has_contact and metrics['height'] > 2.0:
                    print("Warning: Robot losing ground contact")
        
        # Phase 3: Fine stabilization (25% of steps)
        fine_steps = steps - settling_steps - contact_steps
        print(f"Phase 3: Fine stabilization ({fine_steps} steps)")
        
        # Restore normal damping
        p.changeDynamics(robot_id, -1, 
                        linearDamping=self.config.linear_damping,
                        angularDamping=self.config.angular_damping)
        
        for i in range(fine_steps):
            p.stepSimulation()
            time.sleep(1.0 / rate)
            
            if i % 500 == 0:
                metrics = self.get_robot_stability_metrics(robot_id)
                contact = self.check_ground_contact(robot_id)
                
                print(f"Final stabilization step {i}: "
                      f"Height={metrics['height']:.3f}, "
                      f"Velocity={metrics['velocity']:.3f}, "
                      f"Contact={contact}")
        
        # Final verification
        final_metrics = self.get_robot_stability_metrics(robot_id)
        final_contact = self.check_ground_contact(robot_id)
        
        print(f"Stabilization complete!")
        print(f"Final height: {final_metrics['height']:.3f}")
        print(f"Final velocity: {final_metrics['velocity']:.3f}")
        print(f"Ground contact: {final_contact}")
        
        if not final_contact or final_metrics['height'] < 0.5:
            print("WARNING: Robot may not be properly stabilized!")
        else:
            print("Robot successfully stabilized.")