"""Main simulation entry point."""
from core.robot import Robot
from physics.physics_engine import PhysicsEngine
from control.stabilizer import RobotStabilizer
from motion.wave_motion import WaveMotion
from utils.data_structures import SimulationConfig

def main():
    """Main simulation entry point."""
    config = SimulationConfig()
    
    try:
        # Initialize physics
        physics = PhysicsEngine(config)
        physics.initialize()
        
        # Load robot
        robot = Robot("valkyrie_description", [0, 0, config.robot_height])
        robot.load(physics.client)
        
        # Initialize stabilizer
        stabilizer = RobotStabilizer(robot, config)
        
        # Initialize wave motion
        wave_motion = WaveMotion(robot, config)
        
        # Run simulation
        wave_motion.run_simulation(stabilizer)
        
    except Exception as e:
        print(f"Error in main: {e}")
    finally:
        physics.disconnect()

if __name__ == "__main__":
    main()
