# utils/PerformanceOptimizer.py
import psutil
import time
import pybullet as p
from typing import Dict, Any
import numpy as np

class PerformanceOptimizer:
    """Performance optimization utilities for PyBullet simulation."""
    
    def __init__(self, config):
        self.config = config
        self.last_performance_check = time.time()
        
    def optimize_for_ryzen_5600h(self):
        """Optimize settings specifically for Ryzen 5 5600H processor."""
        print("Applying Ryzen 5 5600H optimizations...")
        
        # Ryzen 5 5600H has 6 cores, 12 threads
        # Optimize thread count for this CPU
        self.config.recommended_workers = min(6, psutil.cpu_count())
        
        # Optimize physics timestep for this CPU performance
        self.config.physics_timestep = 1.0 / 240.0  # 240 Hz for good performance
        self.config.render_timestep = 1.0 / 60.0    # 60 FPS rendering
        
        # Memory optimizations
        self.config.collision_margin = 0.001
        self.config.contact_breaking_threshold = 0.001
        
        # Try to set CPU affinity and priority (optional optimizations)
        self.setup_cpu_optimizations()
        
        print(f"Optimized for Ryzen 5 5600H: {self.config.recommended_workers} workers")
        
    def setup_cpu_optimizations(self):
        """Setup CPU optimizations with proper error handling."""
        try:
            current_process = psutil.Process()
            
            # Set CPU affinity to use all available cores
            available_cores = list(range(psutil.cpu_count()))
            current_process.cpu_affinity(available_cores)
            print(f"Set CPU affinity to cores: {available_cores}")
            
            # Try to set higher priority (this might fail without sudo)
            try:
                import os
                if os.name == 'posix':  # Linux/Mac
                    current_process.nice(-5)  # Less aggressive than -10
                else:  # Windows
                    current_process.nice(psutil.HIGH_PRIORITY_CLASS)
                print("Successfully set higher process priority")
            except (psutil.AccessDenied, PermissionError):
                print("Warning: Cannot set higher priority without elevated permissions")
                print("Running with default priority (this is fine for most cases)")
                
        except Exception as e:
            print(f"CPU optimization warning: {e}")
            print("Continuing with default CPU settings")
        
    def configure_pybullet(self):
        """Configure PyBullet for optimal performance."""
        try:
            # Set number of solver iterations for balance of accuracy and speed
            p.setPhysicsEngineParameter(numSolverIterations=50)
            
            # Enable parallel processing
            p.setPhysicsEngineParameter(enableConeFriction=1)
            
            # Set contact processing threshold
            p.setPhysicsEngineParameter(
                contactBreakingThreshold=self.config.contact_breaking_threshold
            )
            
            # Enable file caching for better performance
            p.setPhysicsEngineParameter(enableFileCaching=1)
            
            # Set deterministic overlapping pairs
            p.setPhysicsEngineParameter(deterministicOverlappingPairs=1)
            
            # Optimize solver type
            p.setPhysicsEngineParameter(constraintSolverType=p.CONSTRAINT_SOLVER_LCP_PGS)
            
            print("PyBullet configured for optimal performance")
            
        except Exception as e:
            print(f"Warning: Some PyBullet optimizations failed: {e}")
            print("Continuing with default PyBullet settings")
        
    def monitor_performance(self) -> Dict[str, Any]:
        """Monitor system performance metrics."""
        try:
            # CPU usage
            cpu_percent = psutil.cpu_percent(interval=0.1)
            
            # Memory usage
            memory = psutil.virtual_memory()
            memory_percent = memory.percent
            
            # Temperature (if available)
            cpu_temperature = "N/A"
            try:
                temps = psutil.sensors_temperatures()
                if temps:
                    # Try to get CPU temperature
                    for name, entries in temps.items():
                        if 'cpu' in name.lower() or 'core' in name.lower():
                            cpu_temperature = f"{entries[0].current}°C"
                            break
            except:
                pass
            
            return {
                'cpu_percent': cpu_percent,
                'memory_percent': memory_percent,
                'cpu_temperature': cpu_temperature,
                'available_memory': memory.available / (1024**3),  # GB
                'timestamp': time.time()
            }
            
        except Exception as e:
            print(f"Performance monitoring error: {e}")
            return {}
            
    def optimize_memory_usage(self):
        """Optimize memory usage for better performance."""
        import gc
        
        # Force garbage collection
        gc.collect()
        
        # Set aggressive garbage collection thresholds
        gc.set_threshold(500, 5, 5)
        
        print("Memory optimization applied")
        
    def adjust_quality_for_performance(self, target_fps: float = 60.0):
        """Dynamically adjust simulation quality based on performance."""
        perf_stats = self.monitor_performance()
        
        if not perf_stats:
            return
            
        cpu_usage = perf_stats.get('cpu_percent', 0)
        memory_usage = perf_stats.get('memory_percent', 0)
        
        # Adjust physics timestep based on CPU usage
        if cpu_usage > 80:
            # Reduce physics accuracy for better performance
            self.config.physics_timestep = 1.0 / 120.0
            print("Reduced physics timestep due to high CPU usage")
        elif cpu_usage < 50:
            # Increase physics accuracy if CPU can handle it
            self.config.physics_timestep = 1.0 / 240.0
            
        # Adjust rendering based on memory usage
        if memory_usage > 80:
            self.config.enable_gui = False
            print("Disabled GUI due to high memory usage")
            
    def get_optimization_recommendations(self) -> Dict[str, str]:
        """Get recommendations for optimization based on current performance."""
        perf_stats = self.monitor_performance()
        recommendations = {}
        
        if not perf_stats:
            return recommendations
        
    def set_process_priority_safely(self, priority_level: str = "normal"):
        """Safely set process priority with proper error handling."""
        try:
            current_process = psutil.Process()
            
            if priority_level == "high":
                try:
                    import os
                    if os.name == 'posix':  # Linux/Mac
                        current_process.nice(-5)
                    else:  # Windows
                        current_process.nice(psutil.HIGH_PRIORITY_CLASS)
                    print("Process priority set to HIGH")
                except (psutil.AccessDenied, PermissionError):
                    print("Cannot set high priority without elevated permissions")
                    return False
            elif priority_level == "low":
                current_process.nice(10)
                print("Process priority set to LOW")
            else:
                print("Process priority remains at DEFAULT")
                
            return True
            
        except Exception as e:
            print(f"Priority setting error: {e}")
            return False
            
        cpu_usage = perf_stats.get('cpu_percent', 0)
        memory_usage = perf_stats.get('memory_percent', 0)
        
        if cpu_usage > 80:
            recommendations['cpu'] = "Consider reducing physics timestep or number of objects"
            
        if memory_usage > 80:
            recommendations['memory'] = "Consider reducing simulation complexity or disabling GUI"
            
        if cpu_usage < 30:
            recommendations['performance'] = "CPU has headroom - can increase simulation complexity"
            
        return recommendations