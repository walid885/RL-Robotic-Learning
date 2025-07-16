import pybullet as p
import pybullet_data
import time
import math
from robot_descriptions.loaders.pybullet import load_robot_description
from typing import List, Dict, Tuple, Optional
from functools import partial
from dataclasses import dataclass
import threading

# Import the biomechanical analyzer
from biomechanical_analyzer import run_biomechanical_analysis

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
    robot_height: float = 0.93  # Reduced from 1.0
    stabilization_steps: int = 10000  # Increased
    simulation_rate: float = 240.0
    wave_frequency: float = 0.2
    wave_amplitude: float = 0.3
    elbow_wave_amplitude: float = 0.4
    wrist_wave_amplitude: float = 0.2
    analysis_duration: float = 60.0

@dataclass
class WaveMotionConfig:
    shoulder_lift: float = 0.4
    elbow_bend_base: float = 0.2
    wrist_wave_speed: float = 0.8
    compensation_force: float = 3000
    base_stabilization_force: float = 2000
    leg_force: float = 3000
    torso_force: float = 2000

def initialize_physics_engine() -> None:
    """Initialize PyBullet with more stable settings."""
    p.connect(p.GUI)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, -9.81)
    
    # Create ground plane with better friction
    plane_id = p.loadURDF("plane.urdf")
    p.changeDynamics(plane_id, -1, 
                    lateralFriction=2.0,  # Increased
                    spinningFriction=1.0, 
                    rollingFriction=0.5,
                    restitution=0.1)  # Slight bounce
    
    # More stable physics parameters
    p.setPhysicsEngineParameter(
        fixedTimeStep=1.0/240.0,
        numSolverIterations=150,  # Increased
        numSubSteps=4,  # Increased
        contactBreakingThreshold=0.001,
        enableConeFriction=True,
        erp=0.2,  # Increased
        contactERP=0.2,
        frictionERP=0.1,
        enableFileCaching=0,
        restitutionVelocityThreshold=0.1,
        deterministicOverlappingPairs=1,
        allowedCcdPenetration=0.0005
    )

def load_robot(description: str, position: List[float]) -> int:
    """Load robot with proper grounding and conservative settings."""
    robot_id = load_robot_description(description)
    
    # Start lower and let it settle
    initial_position = [0, 0, 0.5]  # Much lower initial position
    p.resetBasePositionAndOrientation(robot_id, initial_position, [0, 0, 0, 1])
    
    # Get all joints and set them to neutral positions immediately
    num_joints = p.getNumJoints(robot_id)
    for i in range(num_joints):
        joint_info = p.getJointInfo(robot_id, i)
        joint_name = joint_info[1].decode('utf-8').lower()
        
        # Set initial joint positions to stable values
        if 'knee' in joint_name:
            p.resetJointState(robot_id, i, 0.1)  # Slight knee bend
        elif 'hip' in joint_name and ('pitch' in joint_name or 'y' in joint_name):
            p.resetJointState(robot_id, i, -0.05)  # Slight forward lean
        elif 'ankle' in joint_name and ('pitch' in joint_name or 'y' in joint_name):
            p.resetJointState(robot_id, i, 0.02)  # Slight forward ankle
        else:
            p.resetJointState(robot_id, i, 0.0)
    
    # Set base dynamics for stability
    p.changeDynamics(robot_id, -1, 
                    linearDamping=2.0,  # Increased
                    angularDamping=2.0,  # Increased
                    mass=65.0,  # Slightly heavier
                    contactStiffness=10000,  # Increased
                    contactDamping=500)  # Increased
    
    # Set joint dynamics for all joints
    for i in range(num_joints):
        p.changeDynamics(robot_id, i, 
                        linearDamping=1.0,  # Increased
                        angularDamping=1.0,  # Increased
                        maxJointVelocity=0.3,  # Reduced
                        jointDamping=0.5)  # Increased
    
    # Extended settling with active control
    print("Settling robot with active control...")
    for step in range(5000):
        # Apply gentle upward force to prevent falling
        pos, orn = p.getBasePositionAndOrientation(robot_id)
        if pos[2] < 0.8:  # If robot is too low
            p.applyExternalForce(robot_id, -1, [0, 0, 100], [0, 0, 0], p.WORLD_FRAME)
        
        # Stabilize critical joints
        for i in range(num_joints):
            joint_info = p.getJointInfo(robot_id, i)
            joint_name = joint_info[1].decode('utf-8').lower()
            
            if 'knee' in joint_name:
                p.setJointMotorControl2(robot_id, i, p.POSITION_CONTROL, 
                                      targetPosition=0.1, force=500)
            elif 'hip' in joint_name and ('pitch' in joint_name or 'y' in joint_name):
                p.setJointMotorControl2(robot_id, i, p.POSITION_CONTROL, 
                                      targetPosition=-0.05, force=500)
            elif 'ankle' in joint_name and ('pitch' in joint_name or 'y' in joint_name):
                p.setJointMotorControl2(robot_id, i, p.POSITION_CONTROL, 
                                      targetPosition=0.02, force=500)
            else:
                p.setJointMotorControl2(robot_id, i, p.POSITION_CONTROL, 
                                      targetPosition=0.0, force=300)
        
        p.stepSimulation()
        time.sleep(1.0/240.0)
        
        if step % 1000 == 0:
            print(f"Settling step {step}: Height = {pos[2]:.3f}m")
    
    print("Robot settled and stable!")
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
        'hip': [], 'knee': [], 'ankle': [], 'torso': [],
        'shoulder_left': [], 'elbow_left': [], 'shoulder_right': [], 
        'elbow_right': [], 'wrist_right': [], 'other': []
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
    """Calculate biomechanically stable standing pose."""
    stable_positions = {}
    
    # Hip joints
    for joint in joint_categories['hip']:
        name = joint.name.lower()
        if 'pitch' in name or 'y' in name:
            stable_positions[joint.id] = -0.05
        elif 'roll' in name or 'x' in name:
            stable_positions[joint.id] = 0.0
        else:
            stable_positions[joint.id] = 0.0
    
    # Knee joints
    for joint in joint_categories['knee']:
        stable_positions[joint.id] = 0.1
    
    # Ankle joints
    for joint in joint_categories['ankle']:
        name = joint.name.lower()
        if 'pitch' in name or 'y' in name:
            stable_positions[joint.id] = 0.02
        else:
            stable_positions[joint.id] = 0.0
    
    # Other joints
    for category in ['torso', 'shoulder_left', 'shoulder_right', 'elbow_left', 'elbow_right', 'wrist_right', 'other']:
        for joint in joint_categories[category]:
            stable_positions[joint.id] = 0.0
    
    return stable_positions

def set_joint_position_control(robot_id: int, joint_id: int, target_position: float, 
                             force: float = 300, position_gain: float = 0.3, 
                             velocity_gain: float = 0.1) -> None:
    """Set joint position with conservative control."""
    p.setJointMotorControl2(
        robot_id, joint_id, p.POSITION_CONTROL,
        targetPosition=target_position,
        force=force,
        positionGain=position_gain,
        velocityGain=velocity_gain,
        maxVelocity=0.3  # Increased slightly
    )

def stabilize_robot(robot_id: int, joint_categories: Dict[str, List[JointInfo]], 
                   stable_positions: Dict[int, float], steps: int) -> None:
    """Stabilize robot with gentle, adaptive control."""
    print("Final stabilization phase...")
    
    for i in range(steps):
        pos, orn = p.getBasePositionAndOrientation(robot_id)
        euler = p.getEulerFromQuaternion(orn)
        
        # Stronger corrections for stability
        pitch_correction = -euler[1] * 0.2
        roll_correction = -euler[0] * 0.2
        
        # Apply base stabilization force if needed
        if abs(euler[0]) > 0.1 or abs(euler[1]) > 0.1:
            stabilizing_force = [
                -euler[0] * 1000,  # Roll correction
                -euler[1] * 1000,  # Pitch correction
                max(0, (0.93 - pos[2]) * 2000)  # Height correction
            ]
            p.applyExternalForce(robot_id, -1, stabilizing_force, [0, 0, 0], p.WORLD_FRAME)
        
        for joint_id, target_pos in stable_positions.items():
            joint_info = p.getJointInfo(robot_id, joint_id)
            joint_name = joint_info[1].decode('utf-8').lower()
            
            corrected_target = target_pos
            force = 300
            
            if 'ankle' in joint_name:
                if 'pitch' in joint_name or 'y' in joint_name:
                    corrected_target += pitch_correction
                elif 'roll' in joint_name or 'x' in joint_name:
                    corrected_target += roll_correction
                force = 800  # Increased ankle control
            elif 'hip' in joint_name or 'knee' in joint_name:
                force = 600  # Increased leg control
            elif 'torso' in joint_name:
                force = 400
            
            set_joint_position_control(robot_id, joint_id, corrected_target, 
                                     force=force, position_gain=0.3, velocity_gain=0.1)
        
        p.stepSimulation()
        
        if i % 2000 == 0:
            print(f"Step {i}: Height = {pos[2]:.3f}m, Pitch = {math.degrees(euler[1]):.1f}°, Roll = {math.degrees(euler[0]):.1f}°")
        
        time.sleep(1.0/240.0)
    
    print("Stabilization complete!")

def run_goodbye_waving_simulation(robot_id: int, config: SimulationConfig) -> None:
    """Run waving simulation with biomechanical analysis."""
    # Get joints and calculate stable pose
    movable_joints = get_movable_joints(robot_id)
    joint_categories = find_critical_balance_joints(movable_joints)
    stable_positions = calculate_optimal_standing_pose(joint_categories)
    
    print(f"Found {len(movable_joints)} movable joints")
    for category, joints in joint_categories.items():
        if joints:
            print(f"  {category}: {len(joints)} joints")
    
    # Final stabilization
    stabilize_robot(robot_id, joint_categories, stable_positions, config.stabilization_steps)
    
    # Enable joint feedback for force measurements
    num_joints = p.getNumJoints(robot_id)
    for i in range(num_joints):
        p.enableJointFeedback(robot_id, i, True)
    
    # Start biomechanical analysis in separate thread
    print("Starting biomechanical analysis...")
    analysis_thread = threading.Thread(
        target=run_biomechanical_analysis,
        args=(robot_id, config.analysis_duration)
    )
    analysis_thread.daemon = True
    analysis_thread.start()
    
    print("Starting waving motion...")
    
    step_count = 0
    try:
        while True:
            # Monitor and correct balance
            pos, orn = p.getBasePositionAndOrientation(robot_id)
            euler = p.getEulerFromQuaternion(orn)
            
            # Active balance correction
            if abs(euler[0]) > 0.15 or abs(euler[1]) > 0.15:
                correction_force = [
                    -euler[0] * 500,
                    -euler[1] * 500,
                    max(0, (0.93 - pos[2]) * 1000)
                ]
                p.applyExternalForce(robot_id, -1, correction_force, [0, 0, 0], p.WORLD_FRAME)
            
            # Gentle corrections for joints
            pitch_correction = -euler[1] * 0.1
            roll_correction = -euler[0] * 0.1
            
            # Maintain base stability
            for category in ['hip', 'knee', 'ankle', 'torso']:
                for joint in joint_categories[category]:
                    target = stable_positions[joint.id]
                    force = 300
                    
                    if 'ankle' in joint.name.lower():
                        if 'pitch' in joint.name.lower() or 'y' in joint.name.lower():
                            target += pitch_correction
                        elif 'roll' in joint.name.lower() or 'x' in joint.name.lower():
                            target += roll_correction
                        force = 600
                    elif 'hip' in joint.name.lower() or 'knee' in joint.name.lower():
                        force = 500
                    
                    set_joint_position_control(robot_id, joint.id, target,
                                             force=force, position_gain=0.3, velocity_gain=0.1)
            
            # Left arm stays stable
            for joint in joint_categories['shoulder_left'] + joint_categories['elbow_left']:
                set_joint_position_control(robot_id, joint.id, stable_positions[joint.id],
                                         force=300, position_gain=0.3, velocity_gain=0.1)
            
            # Right arm waves gently
            t = step_count * 0.01
            
            for joint in joint_categories['shoulder_right']:
                if 'pitch' in joint.name.lower() or 'y' in joint.name.lower():
                    wave_offset = math.sin(t * config.wave_frequency * 2 * math.pi) * config.wave_amplitude
                    target = stable_positions[joint.id] + 0.4 + wave_offset
                    set_joint_position_control(robot_id, joint.id, target, 
                                             force=400, position_gain=0.3, velocity_gain=0.1)
                else:
                    set_joint_position_control(robot_id, joint.id, stable_positions[joint.id],
                                             force=300, position_gain=0.3, velocity_gain=0.1)
            
            for joint in joint_categories['elbow_right']:
                elbow_wave = math.sin(t * config.wave_frequency * 2 * math.pi) * config.elbow_wave_amplitude
                target = stable_positions[joint.id] + 0.2 + elbow_wave
                set_joint_position_control(robot_id, joint.id, target,
                                         force=350, position_gain=0.3, velocity_gain=0.1)
            
            # Wrist motion
            for joint in joint_categories['wrist_right']:
                wrist_wave = math.sin(t * config.wave_frequency * 4 * math.pi) * config.wrist_wave_amplitude
                target = stable_positions[joint.id] + wrist_wave
                set_joint_position_control(robot_id, joint.id, target,
                                         force=200, position_gain=0.3, velocity_gain=0.1)
            
            p.stepSimulation()
            time.sleep(1.0/config.simulation_rate)
            step_count += 1
            
            # Status update
            if step_count % 2400 == 0:
                print(f"Wave cycle {step_count//2400}: Height = {pos[2]:.3f}m, Tilt = {math.degrees(max(abs(euler[0]), abs(euler[1]))):.1f}°")
            
    except KeyboardInterrupt:
        print("Simulation stopped")
    finally:
        # Wait for analysis thread to complete
        if analysis_thread.is_alive():
            analysis_thread.join(timeout=2.0)

def main():
    """Main entry point with integrated biomechanical analysis."""
    config = SimulationConfig()
    
    print("="*60)
    print("INTEGRATED ROBOT SIMULATION WITH BIOMECHANICAL ANALYSIS")
    print("="*60)
    print(f"Analysis Duration: {config.analysis_duration}s")
    print(f"Wave Frequency: {config.wave_frequency}Hz")
    print(f"Simulation Rate: {config.simulation_rate}Hz")
    print("="*60)
    
    try:
        initialize_physics_engine()
        robot_id = load_robot("valkyrie_description", [0, 0, config.robot_height])
        
        print(f"Robot loaded with ID: {robot_id}")
        print("\nINSTRUCTIONS:")
        print("1. The robot will first settle and stabilize")
        print("2. Additional stabilization phase will run")
        print("3. Biomechanical analysis will start automatically")
        print("4. Real-time charts will show stability metrics")
        print("5. Press Ctrl+C to stop and generate final report")
        print("\nStarting simulation...")
        
        run_goodbye_waving_simulation(robot_id, config)
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        p.disconnect()

if __name__ == "__main__":
    main()