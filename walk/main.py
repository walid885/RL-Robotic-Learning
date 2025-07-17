# main.py - Entry point and orchestration
import pybullet as p
import time
from typing import Dict, List

from simulation.physics_engine import PhysicsEngine
from simulation.robot_loader import RobotLoader
from simulation.joint_controller import JointController
from simulation.motion_planner import MotionPlanner
from simulation.simulation_runner import SimulationRunner
from simulation.simulation_config import SimulationConfig
from models.joint_utils import JointAnalyzer


def main():
    """Main entry point for the robot simulation."""
    config = SimulationConfig()
    
    # Initialize components
    physics_engine = PhysicsEngine()
    robot_loader = RobotLoader(physics_engine)
    joint_analyzer = JointAnalyzer()
    joint_controller = JointController()
    motion_planner = MotionPlanner(config)
    
    # Setup simulation
    physics_engine.initialize()
    robot_id = robot_loader.load_robot("valkyrie_description", [0, 0, config.robot_height])
    
    print(f"Robot loaded with ID: {robot_id}")
    
    # Run simulation
    simulation_runner = SimulationRunner(
        robot_id=robot_id,
        config=config,
        joint_analyzer=joint_analyzer,
        joint_controller=joint_controller,
        motion_planner=motion_planner,
        physics_engine=physics_engine
    )
    
    simulation_runner.run()


if __name__ == "__main__":
    main()




