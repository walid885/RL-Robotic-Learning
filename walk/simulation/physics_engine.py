# simulation/physics_engine.py - Physics engine management
import pybullet as p
import pybullet_data
from simulation.simulation_config import SimulationConfig


class PhysicsEngine:
    """Manages PyBullet physics engine initialization and configuration."""
    
    def __init__(self):
        self.config = SimulationConfig()
        self.connected = False
    
    def initialize(self) -> None:
        """Initialize PyBullet physics engine with default settings."""
        if not self.connected:
            p.connect(p.GUI)
            self.connected = True
        
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, self.config.gravity)
        
        # Load ground plane
        ground_id = p.loadURDF("plane.urdf")
        self._configure_ground_friction(ground_id)
    
    def _configure_ground_friction(self, ground_id: int) -> None:
        """Configure ground friction properties."""
        p.changeDynamics(
            ground_id, -1,
            lateralFriction=self.config.lateral_friction,
            spinningFriction=self.config.spinning_friction,
            rollingFriction=self.config.rolling_friction
        )
    
    def disconnect(self) -> None:
        """Disconnect from PyBullet."""
        if self.connected:
            p.disconnect()
            self.connected = False


