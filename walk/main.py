# main.py - Optimized Entry point and orchestration
import pybullet as p
import time
import multiprocessing as mp
from typing import Dict, List
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from simulation.physics_engine import PhysicsEngine
from simulation.robot_loader import RobotLoader
from simulation.joint_controller import JointController
from simulation.motion_planner import MotionPlanner
from simulation.simulation_runner import SimulationRunner
from simulation.simulation_config import SimulationConfig
from models.joint_utils import JointAnalyzer
from config.PerformanceConfig import PerformanceConfig
from utils.PerformanceOptimizer import PerformanceOptimizer

class OptimizedSimulation:
    """Optimized simulation class with multi-threading and performance enhancements."""
    
    def __init__(self):
        # Initialize performance configuration
        self.perf_config = PerformanceConfig()
        self.performance_optimizer = PerformanceOptimizer(self.perf_config)
        
        # Apply Ryzen 5 5600H optimizations
        self.performance_optimizer.optimize_for_ryzen_5600h()
        
        # Initialize simulation config with performance settings
        self.config = SimulationConfig()
        self.setup_performance_config()
        
        # Initialize thread pool for parallel processing
        self.thread_pool = ThreadPoolExecutor(max_workers=self.perf_config.recommended_workers)
        
        # Pre-allocate arrays for better memory management
        self.joint_states_buffer = np.zeros(100)  # Adjust size based on your robot
        self.control_buffer = np.zeros(100)
        
        # Initialize components
        self.physics_engine = None
        self.robot_loader = None
        self.joint_analyzer = None
        self.joint_controller = None
        self.motion_planner = None
        
    def setup_performance_config(self):
        """Configure simulation parameters using performance config."""
        # Use performance config values
        self.config.physics_timestep = self.perf_config.physics_timestep
        self.config.render_timestep = self.perf_config.render_timestep
        
        # Enable multi-threading in PyBullet
        self.config.enable_real_time = False
        self.config.enable_gui = self.perf_config.enable_gui
        
        # Optimize collision detection
        self.config.collision_margin = self.perf_config.collision_margin
        self.config.contact_breaking_threshold = self.perf_config.contact_breaking_threshold
        
    def initialize_components(self):
        """Initialize all simulation components with optimizations."""
        try:
            # Initialize physics engine with performance settings
            self.physics_engine = PhysicsEngine()
            
            # Pre-load components for faster access
            self.robot_loader = RobotLoader(self.physics_engine)
            self.joint_analyzer = JointAnalyzer()
            self.joint_controller = JointController()
            self.motion_planner = MotionPlanner(self.config)
            
            print("All components initialized successfully")
            return True
            
        except Exception as e:
            print(f"Error initializing components: {e}")
            return False
        
    def setup_simulation(self):
        """Setup simulation with performance optimizations."""
        try:
            # Initialize physics engine with optimized settings
            self.physics_engine.initialize()
            
            # Use performance optimizer to configure PyBullet
            self.performance_optimizer.configure_pybullet()
            
            # Load robot with optimizations
            robot_id = self.robot_loader.load_robot(
                "valkyrie_description", 
                [0, 0, self.config.robot_height]
            )
            
            return robot_id
            
        except Exception as e:
            print(f"Error setting up simulation: {e}")
            return None
        
    def run_parallel_tasks(self, robot_id):
        """Run simulation tasks in parallel for better CPU utilization."""
        # Submit tasks to thread pool
        futures = []
        
        try:
            # Motion planning task - check if method exists
            if hasattr(self.motion_planner, 'update_motion_plan'):
                motion_future = self.thread_pool.submit(
                    self.motion_planner.update_motion_plan
                )
                futures.append(motion_future)
            elif hasattr(self.motion_planner, 'update'):
                motion_future = self.thread_pool.submit(
                    self.motion_planner.update
                )
                futures.append(motion_future)
            else:
                print("Warning: No motion planning update method found")
            
            # Joint analysis task - check if method exists
            if hasattr(self.joint_analyzer, 'analyze_joints'):
                joint_future = self.thread_pool.submit(
                    self.joint_analyzer.analyze_joints, robot_id
                )
                futures.append(joint_future)
            elif hasattr(self.joint_analyzer, 'update'):
                joint_future = self.thread_pool.submit(
                    self.joint_analyzer.update, robot_id
                )
                futures.append(joint_future)
            else:
                print("Warning: No joint analysis method found")
            
            # Control task - check if method exists
            if hasattr(self.joint_controller, 'update_control_signals'):
                control_future = self.thread_pool.submit(
                    self.joint_controller.update_control_signals
                )
                futures.append(control_future)
            elif hasattr(self.joint_controller, 'update'):
                control_future = self.thread_pool.submit(
                    self.joint_controller.update
                )
                futures.append(control_future)
            else:
                print("Warning: No control update method found")
            
            # Wait for all tasks to complete
            for future in futures:
                try:
                    future.result(timeout=0.001)  # Quick timeout for real-time performance
                except Exception as e:
                    print(f"Task failed: {e}")
                    
        except Exception as e:
            print(f"Error in parallel tasks: {e}")
                
    def run_optimized_simulation(self):
        """Run the main simulation loop with optimizations."""
        robot_id = self.setup_simulation()
        
        if robot_id is None:
            print("Failed to setup simulation")
            return
            
        print(f"Robot loaded with ID: {robot_id}")
        
        # Create optimized simulation runner
        try:
            simulation_runner = SimulationRunner(
                robot_id=robot_id,
                config=self.config,
                joint_analyzer=self.joint_analyzer,
                joint_controller=self.joint_controller,
                motion_planner=self.motion_planner,
                physics_engine=self.physics_engine
            )
        except Exception as e:
            print(f"Error creating simulation runner: {e}")
            return
        
        # Performance monitoring
        frame_count = 0
        start_time = time.time()
        last_fps_time = start_time
        last_performance_check = start_time
        
        try:
            while True:
                loop_start = time.time()
                
                # Run parallel tasks every few frames to reduce overhead
                if frame_count % 4 == 0:  # Every 4th frame
                    self.run_parallel_tasks(robot_id)
                
                # Main simulation step
                try:
                    simulation_runner.run_simulation_step()

                except Exception as e:
                    print(f"Error in simulation step: {e}")
                    break
                
                # Adaptive sleep for consistent frame rate
                elapsed = time.time() - loop_start
                target_frame_time = self.config.render_timestep
                
                if elapsed < target_frame_time:
                    time.sleep(target_frame_time - elapsed)
                
                frame_count += 1
                
                # FPS monitoring every second
                current_time = time.time()
                if current_time - last_fps_time >= 1.0:
                    fps = frame_count / (current_time - start_time)
                    print(f"FPS: {fps:.1f}")
                    last_fps_time = current_time
                
                # Performance monitoring every 5 seconds
                if current_time - last_performance_check >= 5.0:
                    perf_stats = self.performance_optimizer.monitor_performance()
                    if 'cpu_percent' in perf_stats:
                        print(f"CPU: {perf_stats['cpu_percent']:.1f}% | "
                              f"Memory: {perf_stats['memory_percent']:.1f}% | "
                              f"Temp: {perf_stats['cpu_temperature']}")
                    last_performance_check = current_time
                    
        except KeyboardInterrupt:
            print("Simulation stopped by user")
        except Exception as e:
            print(f"Simulation error: {e}")
        finally:
            self.cleanup()
            
    def cleanup(self):
        """Clean up resources."""
        try:
            # Shutdown thread pool
            if hasattr(self, 'thread_pool'):
                self.thread_pool.shutdown(wait=True)
                print("Thread pool shutdown complete")
            
            # Cleanup physics engine
            if self.physics_engine:
                if hasattr(self.physics_engine, 'cleanup'):
                    self.physics_engine.cleanup()
                elif hasattr(self.physics_engine, 'disconnect'):
                    self.physics_engine.disconnect()
                else:
                    # Fallback to PyBullet disconnect
                    try:
                        p.disconnect()
                        print("PyBullet disconnected")
                    except:
                        pass
                        
            print("Cleanup complete")
            
        except Exception as e:
            print(f"Error during cleanup: {e}")


def main():
    """Optimized main entry point for the robot simulation."""
    print("Starting optimized simulation for Ryzen 5 5600H...")
    print("=" * 50)
    
    try:
        # Initialize and run optimized simulation
        optimized_sim = OptimizedSimulation()
        
        if optimized_sim.initialize_components():
            optimized_sim.run_optimized_simulation()
        else:
            print("Failed to initialize simulation components")
            
    except Exception as e:
        print(f"Fatal error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    # Enable memory optimization
    import gc
    gc.set_threshold(700, 10, 10)  # Adjust garbage collection for better performance
    
    main()