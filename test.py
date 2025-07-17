import pybullet as p
import pybullet_data
import time
import math
from robot_descriptions.loaders.pybullet import load_robot_description
from typing import List, Dict, Tuple, Optional
from functools import partial
from dataclasses import dataclass

@dataclass
class JointInfo:
    id: int
    name: str
    type: int
    limits: Tuple[float, float]
    current_position: float

@dataclass
class SimulationConfig:
    gravity: float = -9.81
    robot_height: float = 2.0  # Reduced height for stability
    stabilization_steps: int = 2000  # Increased stabilization
    simulation_rate: float = 240.0
    wave_frequency: float = 0.2  # Much slower wave
    wave_amplitude: float = 0.05  # Reduced amplitude

def initialize_physics_engine() -> None:
    """Initialize PyBullet physics engine with default settings."""
    p.connect(p.GUI)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, -9.81)
    p.loadURDF("plane.urdf")
    # Add friction to the ground
    p.changeDynamics(0, -1, lateralFriction=1.0, spinningFriction=0.1, rollingFriction=0.1)

def load_robot(description: str, position: List[float]) -> int:
    """Load robot from description and set initial position."""
    robot_id = load_robot_description(description)
    p.resetBasePositionAndOrientation(robot_id, position, [0, 0, 0, 1])
    
    # Set robot dynamics for stability
    for i in range(p.getNumJoints(robot_id)):
        p.changeDynamics(robot_id, i, 
                        linearDamping=0.1, 
                        angularDamping=0.1,
                        maxJointVelocity=1.0)  # Limit joint velocities
    
    return robot_id

def get_joint_info(robot_id: int, joint_id: int) -> JointInfo:
    """Extract joint information from robot."""
    joint_info = p.getJointInfo(robot_id, joint_id)
    joint_state = p.getJointState(robot_id, joint_id)
    
    return JointInfo(
        id=joint_id,
        name=joint_info[1].decode('utf-8'),
        type=joint_info[2],
        limits=(joint_info[8], joint_info[9]),
        current_position=joint_state[0]
    )

def get_movable_joints(robot_id: int) -> List[JointInfo]:
    """Get all movable joints from robot."""
    num_joints = p.getNumJoints(robot_id)
    movable_types = [p.JOINT_REVOLUTE, p.JOINT_PRISMATIC]
    
    return [
        get_joint_info(robot_id, i) 
        for i in range(num_joints)
        if get_joint_info(robot_id, i).type in movable_types
    ]

def filter_joints_by_keywords(joints: List[JointInfo], keywords: List[str]) -> List[JointInfo]:
    """Filter joints by name keywords."""
    return [
        joint for joint in joints
        if any(keyword in joint.name.lower() for keyword in keywords)
    ]

def find_joint_groups(joints: List[JointInfo]) -> Tuple[List[JointInfo], List[JointInfo]]:
    """Find arm and balance joint groups."""
    arm_keywords = ['rightarm', 'right_arm', 'rightshoulder', 'rightelbow', 'rightwrist', 'r_arm', 'r_shoulder', 'r_elbow']
    balance_keywords = ['leg', 'ankle', 'hip', 'knee', 'foot', 'torso', 'waist']
    
    arm_joints = filter_joints_by_keywords(joints, arm_keywords)
    if not arm_joints:
        arm_joints = filter_joints_by_keywords(joints, ['arm', 'shoulder'])[:3]  # Limit to 3 joints
    
    balance_joints = filter_joints_by_keywords(joints, balance_keywords)
    
    return arm_joints, balance_joints

def calculate_safe_target_position(joint: JointInfo, offset: float = 0.05) -> float:
    """Calculate safe target position within joint limits."""
    current, (lower, upper) = joint.current_position, joint.limits
    
    if lower < upper and abs(lower) < 100 and abs(upper) < 100:  # Check for reasonable limits
        # Stay closer to current position
        safe_range = min(0.2, (upper - lower) * 0.1)
        target = current + safe_range * 0.3
        return max(lower, min(upper, target))
    else:
        return current  # Don't move joints with invalid limits

def set_joint_position_control(robot_id: int, joint_id: int, target_position: float, force: float = 500, 
                             position_gain: float = 0.3, velocity_gain: float = 0.1) -> None:
    """Set position control for a single joint with gentler defaults."""
    p.setJointMotorControl2(
        robot_id, joint_id, p.POSITION_CONTROL,
        targetPosition=target_position,
        force=force,
        positionGain=position_gain,
        velocityGain=velocity_gain
    )

def initialize_joint_positions(robot_id: int, joints: List[JointInfo]) -> None:
    """Initialize all joints to position control with current positions."""
    for joint in joints:
        set_joint_position_control(robot_id, joint.id, joint.current_position, force=200)

def stabilize_robot(steps: int, robot_id: int, rate: float = 240.0) -> None:
    """Stabilize robot for specified number of steps."""
    print("Stabilizing robot...")
    for i in range(steps):
        p.stepSimulation()
        if i % 500 == 0:
            pos, _ = p.getBasePositionAndOrientation(robot_id)
            print(f"Stabilization step {i}: Robot Z position = {pos[2]:.3f}")
        time.sleep(1.0 / rate)

def calculate_wave_motion(step: int, frequency: float, amplitude: float) -> float:
    """Calculate wave motion offset."""
    t = step * 0.01
    return math.sin(t * frequency * 2 * math.pi) * amplitude

def apply_balance_control(robot_id: int, balance_joints: List[JointInfo], initial_positions: Dict[int, float]) -> None:
    """Apply strong position control to balance joints."""
    for joint in balance_joints:
        set_joint_position_control(
            robot_id, joint.id, initial_positions[joint.id], 
            force=1000, position_gain=0.5, velocity_gain=0.2
        )

def apply_stabilization_control(robot_id: int, joints: List[JointInfo], arm_joints: List[JointInfo], 
                              balance_joints: List[JointInfo], initial_positions: Dict[int, float]) -> None:
    """Apply stabilization control to non-arm, non-balance joints."""
    arm_ids = {j.id for j in arm_joints}
    balance_ids = {j.id for j in balance_joints}
    
    for joint in joints:
        if joint.id not in arm_ids and joint.id not in balance_ids:
            set_joint_position_control(robot_id, joint.id, initial_positions[joint.id], force=800)

def apply_wave_motion(robot_id: int, arm_joints: List[JointInfo], target_positions: List[float], 
                     wave_offset: float, joint_limits: List[Tuple[float, float]]) -> None:
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
            
            set_joint_position_control(
                robot_id, joint.id, target_pos, force=200,  # Reduced force
                position_gain=0.3, velocity_gain=0.1
            )

def run_simulation(robot_id: int, config: SimulationConfig) -> None:
    """Main simulation loop."""
    # Get joint information
    movable_joints = get_movable_joints(robot_id)
    arm_joints, balance_joints = find_joint_groups(movable_joints)
    
    print(f"Controlling {len(movable_joints)} movable joints")
    print(f"Found {len(arm_joints)} arm joints")
    print(f"Found {len(balance_joints)} balance joints")
    
    # Initialize positions
    initialize_joint_positions(robot_id, movable_joints)
    stabilize_robot(config.stabilization_steps, robot_id, config.simulation_rate)
    
    # Calculate target positions for arm movement
    target_positions = [calculate_safe_target_position(joint) for joint in arm_joints[:3]]
    joint_limits = [joint.limits for joint in arm_joints[:3]]
    
    # Store initial positions for balance reference
    initial_positions = {joint.id: joint.current_position for joint in movable_joints}
    
    print("Starting gentle wave animation with balance compensation...")
    
    step_count = 0
    warmup_steps = 5000  # Warmup period before starting wave motion
    
    while True:
        # Calculate wave motion (only after warmup)
        if step_count > warmup_steps:
            wave_offset = calculate_wave_motion(step_count - warmup_steps, config.wave_frequency, config.wave_amplitude)
        else:
            wave_offset = 0.0  # No wave motion during warmup
        
        # Apply controls
        apply_balance_control(robot_id, balance_joints, initial_positions)
        apply_stabilization_control(robot_id, movable_joints, arm_joints, balance_joints, initial_positions)
        
        if step_count > warmup_steps:
            apply_wave_motion(robot_id, arm_joints, target_positions, wave_offset, joint_limits)
        
        # Step simulation
        p.stepSimulation()
        time.sleep(1.0 / config.simulation_rate)
        step_count += 1
        
        # Debug output
        if step_count % 2000 == 0:
            pos, _ = p.getBasePositionAndOrientation(robot_id)
            print(f"Step {step_count}: Robot Z position = {pos[2]:.3f}")
            if step_count == warmup_steps:
                print("Starting wave motion...")

def main():
    """Main entry point."""
    config = SimulationConfig()
    
    # Initialize simulation
    initialize_physics_engine()
    robot_id = load_robot("valkyrie_description", [0, 0, config.robot_height])
    
    print(f"Robot loaded with ID: {robot_id}")
    
    # Run simulation
    run_simulation(robot_id, config)

if __name__ == "__main__":
    main()