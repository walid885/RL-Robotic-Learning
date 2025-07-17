import pybullet as p
import numpy as np
from dataclasses import dataclass
from typing import Optional

from config import PerformanceConfig
class PerformanceOptimizer:
    """Handles performance optimizations for the simulation."""
    
    def __init__(self, config: PerformanceConfig):
        self.config = config
        self.setup_numpy_optimizations()
        
    def setup_numpy_optimizations(self):
        """Configure NumPy for optimal performance on Ryzen 5 5600H."""
        # Set NumPy to use all available threads
        import os
        os.environ['OMP_NUM_THREADS'] = str(self.config.cpu_threads)
        os.environ['MKL_NUM_THREADS'] = str(self.config.cpu_threads)
        os.environ['NUMEXPR_NUM_THREADS'] = str(self.config.cpu_threads)
        
        # Enable fast math optimizations
        np.seterr(all='ignore')  # Ignore minor numerical errors for speed
        
    def configure_pybullet(self):
        """Configure PyBullet for optimal performance."""
        # Set physics engine parameters
        p.setPhysicsEngineParameter(
            numSolverIterations=self.config.solver_iterations,
            numSubSteps=self.config.max_sub_steps,
            constraintSolverType=p.CONSTRAINT_SOLVER_LCP_DANTZIG,
            erp=0.1,
            contactERP=0.1,
            frictionERP=0.1,
            enableConeFriction=int(self.config.enable_cone_friction),
            deterministicOverlappingPairs=1,
            allowedCcdPenetration=0.01,
            enableFileCaching=1
        )
        
        # Configure time step
        p.setTimeStep(self.config.physics_timestep)
        
        # Disable unnecessary features for speed
        p.configureDebugVisualizer(
            p.COV_ENABLE_SHADOWS, int(self.config.enable_shadows)
        )
        p.configureDebugVisualizer(
            p.COV_ENABLE_WIREFRAME, int(self.config.enable_wireframe)
        )
        p.configureDebugVisualizer(
            p.COV_ENABLE_GUI, int(self.config.enable_gui)
        )
        
        # Set camera for optimal viewing
        p.resetDebugVisualizerCamera(
            cameraDistance=self.config.camera_distance,
            cameraPitch=self.config.camera_pitch,
            cameraYaw=self.config.camera_yaw,
            cameraTargetPosition=[0, 0, 1]
        )
        
    def setup_memory_management(self):
        """Setup memory management for optimal performance."""
        import gc
        
        # Configure garbage collection for real-time performance
        gc.set_threshold(
            self.config.garbage_collection_frequency,
            10,
            10
        )
        
        # Enable memory pooling if supported
        if self.config.enable_memory_pooling:
            try:
                import psutil
                # Monitor memory usage
                memory_info = psutil.virtual_memory()
                if memory_info.percent > 80:
                    print("Warning: High memory usage detected")
                    gc.collect()
            except ImportError:
                pass
                
    def setup_cpu_affinity(self):
        """Set CPU affinity for optimal performance on Ryzen 5 5600H."""
        try:
            import psutil
            import os
            
            # Set process priority
            current_process = psutil.Process()
            if hasattr(psutil, 'HIGH_PRIORITY_CLASS'):
                current_process.nice(psutil.HIGH_PRIORITY_CLASS)
            else:
                current_process.nice(-10)  # Linux/Mac
                
            # Set CPU affinity to use all cores
            if hasattr(os, 'sched_setaffinity'):
                os.sched_setaffinity(0, range(self.config.cpu_cores))
                
        except (ImportError, AttributeError, OSError) as e:
            print(f"Could not set CPU affinity: {e}")
            
    def optimize_for_ryzen_5600h(self):
        """Apply all optimizations for Ryzen 5 5600H."""
        print("Applying Ryzen 5 5600H optimizations...")
        
        self.setup_cpu_affinity()
        self.configure_pybullet()
        self.setup_memory_management()
        
        # Additional processor-specific optimizations
        self._optimize_cache_usage()
        self._setup_thread_scheduling()
        
        print("Optimizations applied successfully!")
        
    def _optimize_cache_usage(self):
        """Optimize for L3 cache size (32MB on 5600H)."""
        # Adjust buffer sizes to fit in L3 cache
        optimal_buffer_size = min(self.config.buffer_size, 8192)  # 32MB / 4KB per buffer
        self.config.buffer_size = optimal_buffer_size
        
    def _setup_thread_scheduling(self):
        """Setup thread scheduling for optimal performance."""
        import threading
        
        # Set thread stack size for better memory usage
        threading.stack_size(1024 * 1024)  # 1MB stack size
        
    def monitor_performance(self):
        """Monitor performance metrics."""
        try:
            import psutil
            
            # CPU usage
            cpu_percent = psutil.cpu_percent(interval=1)
            
            # Memory usage
            memory = psutil.virtual_memory()
            
            # Temperature monitoring (if available)
            try:
                temperatures = psutil.sensors_temperatures()
                cpu_temp = temperatures.get('coretemp', [{}])[0].get('current', 'N/A')
            except:
                cpu_temp = 'N/A'
                
            return {
                'cpu_percent': cpu_percent,
                'memory_percent': memory.percent,
                'memory_available': memory.available // (1024**3),  # GB
                'cpu_temperature': cpu_temp
            }
            
        except ImportError:
            return {'status': 'psutil not available for monitoring'}


# Example usage
def create_optimized_config():
    """Create an optimized configuration for Ryzen 5 5600H."""
    config = PerformanceConfig()
    optimizer = PerformanceOptimizer(config)
    
    return config, optimizer


# Performance tips for Ryzen 5 5600H
PERFORMANCE_TIPS = """
Performance Tips for Ryzen 5 5600H:

1. Ensure your system is in High Performance power mode
2. Close unnecessary applications to free up CPU resources
3. Make sure your system has adequate cooling
4. Consider enabling XMP/DOCP for RAM if available
5. Use an SSD for better I/O performance
6. Monitor CPU temperatures to avoid thermal throttling
7. Consider disabling Windows Game Mode if it causes issues
8. Ensure your PyBullet is compiled with proper optimization flags
9. Use headless mode (enable_gui=False) for maximum performance
10. Profile your specific simulation to identify bottlenecks
"""