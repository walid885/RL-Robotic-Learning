# src/utils/data_structures.py
from dataclasses import dataclass
from typing import Tuple, List, Dict, Optional

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
    robot_height: float = 1.0
    stabilization_steps: int = 10000
    simulation_rate: float = 240.0
    wave_frequency: float = 0.4
    wave_amplitude: float = 0.2
    elbow_wave_amplitude: float = 0.3
    wrist_wave_amplitude: float = 0.15

@dataclass
class WaveMotionConfig:
    shoulder_lift: float = 0.2
    elbow_bend_base: float = 0.15
    wrist_wave_speed: float = 0.8
    compensation_force: float = 5000
    base_stabilization_force: float = 3000
    leg_force: float = 8000
    torso_force: float = 6000
