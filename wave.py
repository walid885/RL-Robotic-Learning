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
    robot_height: float = 1.0  # Lower spawn height
    stabilization_steps: int = 10000  # More stabilization steps
    simulation_rate: float = 240.0
    wave_frequency: float = 0.4  # Even slower
    wave_amplitude: float = 0.2  # Smaller amplitude
    elbow_wave_amplitude: float = 0.3
    wrist_wave_amplitude: float = 0.15

@dataclass
class WaveMotionConfig:
    shoulder_lift: float = 0.2  # Further reduced
    elbow_bend_base: float = 0.15
    wrist_wave_speed: float = 0.8
    compensation_force: float = 5000  # Much stronger
    base_stabilization_force: float = 3000
    leg_force: float = 8000  # Dedicated leg force
    torso_force: float = 6000  # Dedicated torso force

def initialize_physics_engine() -> None:
    """Initialize PyBullet with enhanced stability settings."""
    p.connect(p.GUI)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, -9.81)
    
    # High-friction ground
    plane_id = p.loadURDF("plane.urdf")
    p.changeDynamics(plane_id, -1, lateralFriction=2.0, spinningFriction=0.5, rollingFriction=0.2)
    
    # More stable physics
    p.setPhysicsEngineParameter(
        fixedTimeStep=1.0/240.0,
        numSolverIterations=100,  # Doubled iterations
        numSubSteps=4,  # More substeps
        contactBreakingThreshold=0.0005,
        enableConeFriction=True,
        erp=0.8,  # Error reduction parameter
        contactERP=0.8,
        frictionERP=0.2
    )

def load_robot(description: str, position: List[float]) -> int:
    """Load robot with maximum stability settings."""
    robot_id = load_robot_description(description)
    
    # Set position lower and add slight forward lean for stability
    p.resetBasePositionAndOrientation(robot_id, position, [0, 0, 0, 1])
    
    num_joints = p.getNumJoints(robot_id)
    
    # Enhanced base dynamics
    p.changeDynamics(robot_id, -1, 
                    linearDamping=0.4,  # Higher damping
                    angularDamping=0.4,
                    mass=100.0,  # Heavier for stability
                    localInertiaDiagonal=[2, 2, 2])
    
    # Enhanced joint dynamics
    for i in range(num_joints):
        joint_info = p.getJointInfo(robot_id, i)
        joint_name = joint_info[1].decode('utf-8').lower()
        
        # Different settings for different joint types
        if 'leg' in joint_name or 'ankle' in joint_name or 'hip' in joint_name:
            p.changeDynamics(robot_id, i, 
                            linearDamping=0.5, 
                            angularDamping=0.5,
                            maxJointVelocity=0.5,
                            jointDamping=0.2,
                            frictionAnchor=1)
        else:
            p.changeDynamics(robot_id, i, 
                            linearDamping=0.3, 
                            angularDamping=0.3,
                            maxJointVelocity=0.8,
                            jointDamping=0.15,
                            frictionAnchor=1)
    
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

def find_critical_balance_joints(joints: List[JointInfo]) -> Dict[str, List[JointInfo]]:
    """Find and categorize critical balance joints."""
    joint_categories = {
        'hip': [],
        'knee': [],
        'ankle': [],
        'torso': [],
        'shoulder_left': [],
        'elbow_left': [],
        'shoulder_right': [],
        'elbow_right': [],
        'wrist_right': [],
        'other': []
    }
    
    for joint in joints:
        name = joint.name.lower()
        
        if 'hip' in name:
            joint_categories['hip'].append(joint)
        elif 'knee' in name:
            joint_categories['knee'].append(joint)
        elif 'ankle' in name:
            joint_categories['ankle'].append(joint)
        elif 'torso' in name or 'spine' in name or 'waist' in name:
            joint_categories['torso'].append(joint)
        elif ('shoulder' in name or 'arm' in name) and ('left' in name or 'l_' in name):
            joint_categories['shoulder_left'].append(joint)
        elif ('elbow' in name or 'forearm' in name) and ('left' in name or 'l_' in name):
            joint_categories['elbow_left'].append(joint)
        elif ('shoulder' in name or 'arm' in name) and ('right' in name or 'r_' in name):
            joint_categories['shoulder_right'].append(joint)
        elif ('elbow' in name or 'forearm' in name) and ('right' in name or 'r_' in name):
            joint_categories['elbow_right'].append(joint)
        elif ('wrist' in name or 'hand' in name) and ('right' in name or 'r_' in name):
            joint_categories['wrist_right'].append(joint)
        else:
            joint_categories['other'].append(joint)
    
    return joint_categories

def calculate_optimal_standing_pose(joint_categories: Dict[str, List[JointInfo]]) -> Dict[int, float]:
    """Calculate optimal standing pose for maximum stability."""
    stable_positions = {}
    
    # Hip joints: slight forward lean for stability
    for joint in joint_categories['hip']:
        if 'pitch' in joint.name.lower() or 'y' in joint.name.lower():
            stable_positions[joint.id] = -0.1  # Slight forward lean
        else:
            stable_positions[joint.id] = 0.0
    
    # Knee joints: slight bend for shock absorption
    for joint in joint_categories['knee']:
        stable_positions[joint.id] = 0.05  # Very slight bend
    
    # Ankle joints: flat foot stance
    for joint in joint_categories['ankle']:
        stable_positions[joint.id] = 0.0
    
    # Torso: upright
    for joint in joint_categories['torso']:
        stable_positions[joint.id] = 0.0
    
    # Left arm: neutral position
    for joint in joint_categories['shoulder_left']:
        stable_positions[joint.id] = 0.0
    for joint in joint_categories['elbow_left']:
        stable_positions[joint.id] = 0.1  # Slight bend
    
    # Right arm: ready for waving
    for joint in joint_categories['shoulder_right']:
        stable_positions[joint.id] = 0.0
    for joint in joint_categories['elbow_right']:
        stable_positions[joint.id] = 0.1
    for joint in joint_categories['wrist_right']:
        stable_positions[joint.id] = 0.0
    
    # Other joints: current position
    for joint in joint_categories['other']:
        stable_positions[joint.id] = joint.current_position
    
    return stable_positions

def enhanced_stabilization_control(robot_id: int, joint_categories: Dict[str, List[JointInfo]], 
                                 stable_positions: Dict[int, float], wave_config: WaveMotionConfig,
                                 step: int) -> None:
    """Apply enhanced stabilization with progressive force ramping."""
    
    # Progressive force ramp-up
    force_ramp = min(1.0, step / 2000.0)
    
    # Critical balance joints get maximum force
    for joint in joint_categories['hip']:
        set_joint_position_control(
            robot_id, joint.id, stable_positions[joint.id], 
            force=wave_config.leg_force * force_ramp,
            position_gain=0.8, velocity_gain=0.5
        )
    
    for joint in joint_categories['knee']:
        set_joint_position_control(
            robot_id, joint.id, stable_positions[joint.id], 
            force=wave_config.leg_force * force_ramp,
            position_gain=0.8, velocity_gain=0.5
        )
    
    for joint in joint_categories['ankle']:
        set_joint_position_control(
            robot_id, joint.id, stable_positions[joint.id], 
            force=wave_config.leg_force * force_ramp,
            position_gain=0.9, velocity_gain=0.6
        )
    
    for joint in joint_categories['torso']:
        set_joint_position_control(
            robot_id, joint.id, stable_positions[joint.id], 
            force=wave_config.torso_force * force_ramp,
            position_gain=0.7, velocity_gain=0.4
        )
    
    # Left arm stabilization
    for joint in joint_categories['shoulder_left']:
        set_joint_position_control(
            robot_id, joint.id, stable_positions[joint.id], 
            force=wave_config.base_stabilization_force * force_ramp,
            position_gain=0.6, velocity_gain=0.3
        )
    
    for joint in joint_categories['elbow_left']:
        set_joint_position_control(
            robot_id, joint.id, stable_positions[joint.id], 
            force=wave_config.base_stabilization_force * force_ramp,
            position_gain=0.6, velocity_gain=0.3
        )

def set_joint_position_control(robot_id: int, joint_id: int, target_position: float, 
                             force: float = 500, position_gain: float = 0.3, 
                             velocity_gain: float = 0.1) -> None:
    """Enhanced joint position control with velocity limiting."""
    p.setJointMotorControl2(
        robot_id, joint_id, p.POSITION_CONTROL,
        targetPosition=target_position,
        force=force,
        positionGain=position_gain,
        velocityGain=velocity_gain,
        maxVelocity=0.3  # Further reduced for stability
    )

def check_robot_stability(robot_id: int, step: int) -> bool:
    """Check if robot is stable and upright."""
    pos, orn = p.getBasePositionAndOrientation(robot_id)
    linear_vel, angular_vel = p.getBaseVelocity(robot_id)
    
    # Calculate orientation from quaternion
    euler = p.getEulerFromQuaternion(orn)
    
    # Check stability criteria
    height_ok = pos[2] > 0.8  # Minimum height
    tilt_ok = abs(euler[0]) < 0.3 and abs(euler[1]) < 0.3  # Not tilted too much
    vel_ok = sum(v**2 for v in linear_vel)**0.5 < 1.0  # Not moving too fast
    
    if step % 2000 == 0:
        print(f"Step {step}: H={pos[2]:.3f}, Tilt=({euler[0]:.3f},{euler[1]:.3f}), Vel={sum(v**2 for v in linear_vel)**0.5:.3f}")
    
    return height_ok and tilt_ok and vel_ok

def stabilize_robot_enhanced(steps: int, robot_id: int, joint_categories: Dict[str, List[JointInfo]], 
                           stable_positions: Dict[int, float], wave_config: WaveMotionConfig, 
                           rate: float = 240.0) -> None:
    """Enhanced multi-phase stabilization."""
    print("Enhanced stabilization starting...")
    
    for i in range(steps):
        # Apply enhanced stabilization control
        enhanced_stabilization_control(robot_id, joint_categories, stable_positions, wave_config, i)
        
        # Step simulation
        p.stepSimulation()
        
        # Check stability
        if i % 1000 == 0:
            is_stable = check_robot_stability(robot_id, i)
            if not is_stable and i > 2000:
                print(f"Warning: Robot unstable at step {i}")
        
        time.sleep(1.0 / rate)
    
    print("Stabilization complete!")

def run_goodbye_waving_simulation(robot_id: int, config: SimulationConfig) -> None:
    """Enhanced simulation with better stability management."""
    wave_config = WaveMotionConfig()
    
    # Get and categorize joints
    movable_joints = get_movable_joints(robot_id)
    joint_categories = find_critical_balance_joints(movable_joints)
    
    print(f"Found {len(movable_joints)} movable joints")
    for category, joints in joint_categories.items():
        if joints:
            print(f"  {category}: {len(joints)} joints")
    
    # Calculate optimal standing pose
    stable_positions = calculate_optimal_standing_pose(joint_categories)
    
    # Enhanced stabilization
    stabilize_robot_enhanced(config.stabilization_steps, robot_id, joint_categories, 
                           stable_positions, wave_config, config.simulation_rate)
    
    print("Starting goodbye waving motion...")
    
    step_count = 0
    warmup_steps = 3000
    
    try:
        while True:
            # Always maintain balance
            enhanced_stabilization_control(robot_id, joint_categories, stable_positions, 
                                         wave_config, step_count)
            
            # Add waving motion after warmup
            if step_count > warmup_steps:
                # Simple waving motion for right arm
                t = (step_count - warmup_steps) * 0.01
                wave_factor = min(1.0, (step_count - warmup_steps) / 2000.0)
                
                # Gentle shoulder lift
                shoulder_wave = 0.3 * wave_factor
                for joint in joint_categories['shoulder_right']:
                    if 'pitch' in joint.name.lower() or 'y' in joint.name.lower():
                        target = stable_positions[joint.id] + shoulder_wave
                        set_joint_position_control(robot_id, joint.id, target, 
                                                 force=400, position_gain=0.3, velocity_gain=0.15)
                
                # Gentle elbow motion
                elbow_wave = math.sin(t * config.wave_frequency * 2 * math.pi) * config.elbow_wave_amplitude
                for joint in joint_categories['elbow_right']:
                    target = stable_positions[joint.id] + (elbow_wave * wave_factor)
                    set_joint_position_control(robot_id, joint.id, target,
                                             force=300, position_gain=0.25, velocity_gain=0.12)
                
                # Gentle wrist motion
                wrist_wave = math.sin(t * 1.2 * 2 * math.pi) * config.wrist_wave_amplitude
                for joint in joint_categories['wrist_right']:
                    target = stable_positions[joint.id] + (wrist_wave * wave_factor)
                    set_joint_position_control(robot_id, joint.id, target,
                                             force=200, position_gain=0.4, velocity_gain=0.1)
            
            p.stepSimulation()
            time.sleep(1.0 / config.simulation_rate)
            step_count += 1
            
            # Stability monitoring
            if step_count % 2000 == 0:
                is_stable = check_robot_stability(robot_id, step_count)
                if not is_stable:
                    print("Robot becoming unstable - increasing stabilization")
                    # Could add emergency stabilization here
                
                if step_count == warmup_steps:
                    print("Beginning waving motion...")
    
    except KeyboardInterrupt:
        print("Simulation stopped by user")
    except Exception as e:
        print(f"Simulation error: {e}")

def main():
    """Main entry point."""
    config = SimulationConfig()
    
    try:
        initialize_physics_engine()
        robot_id = load_robot("valkyrie_description", [0, 0, config.robot_height])
        
        print(f"Robot loaded with ID: {robot_id}")
        run_goodbye_waving_simulation(robot_id, config)
        
    except Exception as e:
        print(f"Error in main: {e}")
    finally:
        p.disconnect()

if __name__ == "__main__":
    main()