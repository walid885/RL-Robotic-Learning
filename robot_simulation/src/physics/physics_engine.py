# src/physics/physics_engine.py
import pybullet as p
import pybullet_data
from ..utils.data_structures import SimulationConfig

class PhysicsEngine:
    """PyBullet engine initialization and configuration."""
    
    def __init__(self, config: SimulationConfig):
        self.config = config
        self.client = None
        self.plane_id = None
        
    def initialize(self):
        """Initialize PyBullet with enhanced stability settings."""
        self.client = p.connect(p.GUI)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, self.config.gravity)
        
        self._setup_ground()
        self._configure_physics()
        
    def _setup_ground(self):
        """Setup high-friction ground plane."""
        self.plane_id = p.loadURDF("plane.urdf")
        p.changeDynamics(self.plane_id, -1, 
                        lateralFriction=2.0, 
                        spinningFriction=0.5, 
                        rollingFriction=0.2)
    
    def _configure_physics(self):
        """Configure physics parameters for stability."""
        p.setPhysicsEngineParameter(
            fixedTimeStep=1.0/self.config.simulation_rate,
            numSolverIterations=100,
            numSubSteps=4,
            contactBreakingThreshold=0.0005,
            enableConeFriction=True,
            erp=0.8,
            contactERP=0.8,
            frictionERP=0.2
        )
    
    def disconnect(self):
        """Disconnect from physics engine."""
        if self.client:
            p.disconnect()

# src/control/joint_controller.py
import pybullet as p

class JointController:
    """Low-level joint control operations."""
    
    def __init__(self, robot_id: int):
        self.robot_id = robot_id
        
    def set_position_control(self, joint_id: int, target_position: float, 
                           force: float = 500, position_gain: float = 0.3, 
                           velocity_gain: float = 0.1) -> None:
        """Enhanced joint position control with velocity limiting."""
        p.setJointMotorControl2(
            self.robot_id, joint_id, p.POSITION_CONTROL,
            targetPosition=target_position,
            force=force,
            positionGain=position_gain,
            velocityGain=velocity_gain,
            maxVelocity=0.3
        )
