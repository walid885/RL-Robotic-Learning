# config/simulation_config.py - Configuration management
from dataclasses import dataclass


@dataclass
class SimulationConfig:
    """Configuration parameters for the robot simulation."""
    gravity: float = -9.81
    robot_height: float = 1.0
    stabilization_steps: int = 2000
    simulation_rate: float = 240.0
    wave_frequency: float = 0.2
    wave_amplitude: float = 0.05
    warmup_steps: int = 5000
    debug_interval: int = 2000
    
    # Joint control parameters
    default_force: float = 500.0
    position_gain: float = 0.3
    velocity_gain: float = 0.1
    balance_force: float = 1000.0
    balance_position_gain: float = 0.5
    balance_velocity_gain: float = 0.2
    
    # Robot dynamics
    linear_damping: float = 0.1
    angular_damping: float = 0.1
    max_joint_velocity: float = 1.0
    
    # Ground friction
    lateral_friction: float = 1.0
    spinning_friction: float = 0.1
    rolling_friction: float = 0.1
