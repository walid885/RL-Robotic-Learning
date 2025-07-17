# simulation/simulation_config.py - Updated configuration for stable spawning
from dataclasses import dataclass


@dataclass
class SimulationConfig:
    """Configuration parameters for the robot simulation with stability improvements."""
    
    # Physics settings
    gravity: float = -9.81
    robot_height: float = 1.0  # Increased spawn height
    stabilization_steps: int = 8000  # More stabilization steps
    simulation_rate: float = 240.0
    physics_timestep: float = 1.0/240.0
    robot_height: float = 0.8  # Optimal starting height for humanoids
    stabilization_steps: int = 3000
    simulation_rate: float = 240.0

    # Wave motion parameters
    wave_frequency: float = 0.15  # Reduced frequency for stability
    wave_amplitude: float = 0.03  # Reduced amplitude
    warmup_steps: int = 8000  # Longer warmup
    debug_interval: int = 1000
    
    # Joint control parameters
    default_force: float = 400.0  # Reduced default force
    position_gain: float = 0.2  # Reduced gain for smoother motion
    velocity_gain: float = 0.05
    balance_force: float = 800.0  # Reduced balance force
    balance_position_gain: float = 0.4
    balance_velocity_gain: float = 0.2
    
    # Robot dynamics - improved stability
    linear_damping: float = 0.2  # Increased damping
    angular_damping: float = 0.2
    max_joint_velocity: float = 0.5  # Reduced max velocity
    joint_damping: float = 0.1
    joint_friction: float = 0.1
    
    # Ground friction - better grip
    lateral_friction: float = 1.5  # Increased friction
    spinning_friction: float = 1.5
    rolling_friction: float = 0.5
    
    # Stability thresholds
    min_height: float = 0.5  # Minimum acceptable height
    max_velocity: float = 1.0  # Maximum velocity before intervention
    max_angular_velocity: float = 2.0
    
    # Performance settings
    enable_gui: bool = True
    enable_real_time: bool = False
    collision_margin: float = 0.01
    contact_breaking_threshold: float = 0.02

    robot_height: float = 2.0
    stabilization_steps: int = 2000
    simulation_rate: float = 240.0
2