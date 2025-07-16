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
    robot_height: float = 1.0  # Start higher, will settle down
    stabilization_steps: int = 5000
    simulation_rate: float = 240.0
    wave_frequency: float = 0.3
    wave_amplitude: float = 0.4
    elbow_wave_amplitude: float = 0.6
    wrist_wave_amplitude: float = 0.3

@dataclass
class WaveMotionConfig:
    shoulder_lift: float = 0.4
    elbow_bend_base: float = 0.2
    wrist_wave_speed: float = 0.8
    compensation_force: float = 5000  # Reduced from 8000
    base_stabilization_force: float = 3000  # Reduced from 6000
    leg_force: float = 5000  # Reduced from 8000
    torso_force: float = 4000  # Reduced from 8000

def initialize_physics_engine() -> None:
    """Initialize PyBullet with stable settings."""
    p.connect(p.GUI)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, -9.81)
    
    # Create ground plane
    plane_id = p.loadURDF("plane.urdf")
    p.changeDynamics(plane_id, -1, 
                    lateralFriction=1.0, 
                    spinningFriction=0.5, 
                    rollingFriction=0.1,
                    restitution=0.0)
    
    # More conservative physics settings
    p.setPhysicsEngineParameter(
        fixedTimeStep=1.0/240.0,
        numSolverIterations=50,  # Reduced from 150
        numSubSteps=4,  # Reduced from 8
        contactBreakingThreshold=0.001,
        enableConeFriction=True,
        erp=0.2,  # Reduced from 0.9
        contactERP=0.2,  # Reduced from 0.9
        frictionERP=0.1,  # Reduced from 0.3
        enableFileCaching=0,
        restitutionVelocityThreshold=0.1,
        deterministicOverlappingPairs=1,
        allowedCcdPenetration=0.005
    )

def load_robot(description: str, position: List[float]) -> int:
    """Load robot with proper grounding."""
    robot_id = load_robot_description(description)
    
    # Initial placement - start higher and let it settle
    initial_position = [0, 0, 1.5]  # Start at 1.5m height
    p.resetBasePositionAndOrientation(robot_id, initial_position, [0, 0, 0, 1])
    
    # Set conservative base dynamics
    p.changeDynamics(robot_id, -1, 
                    linearDamping=0.8,
                    angularDamping=0.8,
                    mass=80.0,  # Reduced from 120
                    contactStiffness=10000,  # Reduced from 30000
                    contactDamping=500)  # Reduced from 1000
    
    # Set all joints to zero initially
    num_joints = p.getNumJoints(robot_id)
    for i in range(num_joints):
        p.resetJointState(robot_id, i, 0.0)
        
        # Conservative joint dynamics
        p.changeDynamics(robot_id, i, 
                        linearDamping=0.5, 
                        angularDamping=0.5,
                        maxJointVelocity=1.0,
                        jointDamping=0.1)
    
    # Let robot settle naturally for a moment
    for _ in range(1000):
        p.stepSimulation()
        time.sleep(1.0/240.0)
    
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
    """Calculate stable standing pose."""
    stable_positions = {}
    
    # Keep all joints near zero for stability
    for category, joints in joint_categories.items():
        for joint in joints:
            if 'knee' in joint.name.lower():
                stable_positions[joint.id] = 0.05  # Very slight bend
            else:
                stable_positions[joint.id] = 0.0
    
    return stable_positions

def set_joint_position_control(robot_id: int, joint_id: int, target_position: float, 
                             force: float = 500, position_gain: float = 0.3, 
                             velocity_gain: float = 0.1) -> None:
    """Set joint position with conservative control."""
    p.setJointMotorControl2(
        robot_id, joint_id, p.POSITION_CONTROL,
        targetPosition=target_position,
        force=force,
        positionGain=position_gain,
        velocityGain=velocity_gain,
        maxVelocity=0.3  # Reduced from 0.5
    )

def stabilize_robot(robot_id: int, joint_categories: Dict[str, List[JointInfo]], 
                   stable_positions: Dict[int, float], steps: int) -> None:
    """Stabilize robot with gentle control."""
    print("Stabilizing robot...")
    
    for i in range(steps):
        # Apply gentle stabilization to all joints
        for joint_id, target_pos in stable_positions.items():
            set_joint_position_control(robot_id, joint_id, target_pos, 
                                     force=1000, position_gain=0.2, velocity_gain=0.1)
        
        p.stepSimulation()
        
        if i % 1000 == 0:
            pos, orn = p.getBasePositionAndOrientation(robot_id)
            print(f"Step {i}: Height = {pos[2]:.3f}m")
        
        time.sleep(1.0/240.0)
    
    print("Stabilization complete!")

def run_goodbye_waving_simulation(robot_id: int, config: SimulationConfig) -> None:
    """Run waving simulation with stable base."""
    # Get joints and calculate stable pose
    movable_joints = get_movable_joints(robot_id)
    joint_categories = find_critical_balance_joints(movable_joints)
    stable_positions = calculate_optimal_standing_pose(joint_categories)
    
    print(f"Found {len(movable_joints)} movable joints")
    
    # Stabilize first
    stabilize_robot(robot_id, joint_categories, stable_positions, config.stabilization_steps)
    
    print("Starting waving motion...")
    
    step_count = 0
    try:
        while True:
            # Maintain base stability
            for category in ['hip', 'knee', 'ankle', 'torso']:
                for joint in joint_categories[category]:
                    set_joint_position_control(robot_id, joint.id, stable_positions[joint.id],
                                             force=2000, position_gain=0.3, velocity_gain=0.2)
            
            # Left arm stays stable
            for joint in joint_categories['shoulder_left'] + joint_categories['elbow_left']:
                set_joint_position_control(robot_id, joint.id, stable_positions[joint.id],
                                         force=1000, position_gain=0.2, velocity_gain=0.1)
            
            # Right arm waves
            t = step_count * 0.01
            wave_amplitude = 0.3
            
            for joint in joint_categories['shoulder_right']:
                if 'pitch' in joint.name.lower() or 'y' in joint.name.lower():
                    wave_offset = math.sin(t * config.wave_frequency * 2 * math.pi) * wave_amplitude
                    target = stable_positions[joint.id] + 0.5 + wave_offset
                    set_joint_position_control(robot_id, joint.id, target, 
                                             force=800, position_gain=0.3, velocity_gain=0.1)
            
            for joint in joint_categories['elbow_right']:
                elbow_wave = math.sin(t * config.wave_frequency * 2 * math.pi) * 0.4
                target = stable_positions[joint.id] + 0.3 + elbow_wave
                set_joint_position_control(robot_id, joint.id, target,
                                         force=600, position_gain=0.3, velocity_gain=0.1)
            
            p.stepSimulation()
            time.sleep(1.0/config.simulation_rate)
            step_count += 1
            
    except KeyboardInterrupt:
        print("Simulation stopped")

def main():
    """Main entry point."""
    config = SimulationConfig()
    
    try:
        initialize_physics_engine()
        robot_id = load_robot("valkyrie_description", [0, 0, config.robot_height])
        
        print(f"Robot loaded with ID: {robot_id}")
        run_goodbye_waving_simulation(robot_id, config)
        
    except Exception as e:
        print(f"Error: {e}")
    finally:
        p.disconnect()

if __name__ == "__main__":
    main()