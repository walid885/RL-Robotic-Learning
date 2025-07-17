# simulation/simulation_runner.py - Main simulation orchestration
import pybullet as p
import time
from typing import Dict, List
from models.joint_info import JointInfo
from simulation.simulation_config import SimulationConfig
from models.joint_utils import JointAnalyzer
from simulation.joint_controller import JointController
from simulation.motion_planner import MotionPlanner
from simulation.stabilizer import RobotStabilizer
from simulation.physics_engine import PhysicsEngine


class SimulationRunner:
    """Orchestrates the main simulation loop."""
    
    def __init__(self, robot_id: int, config: SimulationConfig, 
                 joint_analyzer: JointAnalyzer, joint_controller: JointController,
                 motion_planner: MotionPlanner, physics_engine: PhysicsEngine):
        self.robot_id = robot_id
        self.config = config
        self.joint_analyzer = joint_analyzer
        self.joint_controller = joint_controller
        self.motion_planner = motion_planner
        self.physics_engine = physics_engine
        self.stabilizer = RobotStabilizer(config)
        
        # Initialize simulation state
        self.step_count = 0
        self.movable_joints: List[JointInfo] = []
        self.arm_joints: List[JointInfo] = []
        self.balance_joints: List[JointInfo] = []
        self.initial_positions: Dict[int, float] = {}
        self.target_positions: List[float] = []
        self.joint_limits: List[tuple] = []
    
    def setup_simulation(self) -> None:
        """Setup the simulation with joint analysis and initial positions."""
        # Analyze joints
        self.movable_joints = self.joint_analyzer.get_movable_joints(self.robot_id)
        self.arm_joints, self.balance_joints = self.joint_analyzer.find_joint_groups(self.movable_joints)
        
        print(f"Controlling {len(self.movable_joints)} movable joints")
        print(f"Found {len(self.arm_joints)} arm joints")
        print(f"Found {len(self.balance_joints)} balance joints")
        
        # Initialize joint positions
        self.joint_controller.initialize_joint_positions(self.robot_id, self.movable_joints)
        
        # Stabilize robot
        self.stabilizer.stabilize_robot(self.robot_id)
        
        # Calculate motion parameters
        self.target_positions = self.motion_planner.calculate_target_positions(self.arm_joints)
        self.joint_limits = self.motion_planner.get_joint_limits(self.arm_joints)
        
        # Store initial positions for balance reference
        self.initial_positions = {
            joint.id: joint.current_position 
            for joint in self.movable_joints
        }
    
    def run_simulation_step(self) -> None:
        """Execute a single simulation step."""
        # Calculate wave motion (only after warmup)
        if self.step_count > self.config.warmup_steps:
            wave_offset = self.motion_planner.calculate_wave_motion(
                self.step_count - self.config.warmup_steps
            )
        else:
            wave_offset = 0.0
        
        # Apply controls
        self.joint_controller.apply_balance_control(
            self.robot_id, self.balance_joints, self.initial_positions
        )
        
        self.joint_controller.apply_stabilization_control(
            self.robot_id, self.movable_joints, self.arm_joints, 
            self.balance_joints, self.initial_positions
        )
        
        if self.step_count > self.config.warmup_steps:
            self.joint_controller.apply_wave_motion(
                self.robot_id, self.arm_joints, self.target_positions, 
                wave_offset, self.joint_limits
            )
        
        # Step simulation
        p.stepSimulation()
        time.sleep(1.0 / self.config.simulation_rate)
        self.step_count += 1
        
        # Debug output
        if self.step_count % self.config.debug_interval == 0:
            pos, _ = p.getBasePositionAndOrientation(self.robot_id)
            print(f"Step {self.step_count}: Robot Z position = {pos[2]:.3f}")
            
            if self.step_count == self.config.warmup_steps:
                print("Starting wave motion...")
    
    def run(self) -> None:
        """Run the main simulation loop."""
        self.setup_simulation()
        
        print("Starting gentle wave animation with balance compensation...")
        
        try:
            while True:
                self.run_simulation_step()
        except KeyboardInterrupt:
            print("\nSimulation interrupted by user.")
        finally:
            self.cleanup()
    
    def cleanup(self) -> None:
        """Clean up simulation resources."""
        print("Cleaning up simulation...")
        self.physics_engine.disconnect()