# performance_config.py - Optimized configuration for Ryzen 5 5600H
import pybullet as p
import numpy as np
from dataclasses import dataclass
from typing import Optional


@dataclass
class PerformanceConfig:
    """Performance configuration optimized for Ryzen 5 5600H processor."""
    
    # CPU-specific optimizations
    cpu_cores: int = 6
    cpu_threads: int = 12
    recommended_workers: int = 8  # Optimal for 5600H
    
    # Physics engine optimizations
    physics_timestep: float = 1.0/240.0  # High frequency for smooth simulation
    render_timestep: float = 1.0/60.0    # 60 FPS rendering
    max_sub_steps: int = 1               # Reduce for speed
    solver_iterations: int = 10          # Balanced performance/accuracy
    
    # Memory optimizations
    enable_memory_pooling: bool = True
    buffer_size: int = 1024
    garbage_collection_frequency: int = 100
    
    # Rendering optimizations
    enable_shadows: bool = False         # Disable for speed
    enable_wireframe: bool = False
    enable_gui: bool = True             # Set False for headless mode
    camera_distance: float = 3.0
    camera_pitch: float = -30
    camera_yaw: float = 0
    
    # Collision detection optimizations
    collision_margin: float = 0.001
    contact_breaking_threshold: float = 0.001
    enable_cone_friction: bool = False   # Disable for speed
    
    # Parallel processing settings
    enable_parallel_physics: bool = True
    enable_parallel_rendering: bool = True
    enable_async_loading: bool = True

