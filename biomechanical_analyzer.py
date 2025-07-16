import pybullet as p
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import time
from collections import deque
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional
import threading
import queue

@dataclass
class ForceData:
    joint_id: int
    joint_name: str
    force: float
    torque: float
    timestamp: float

@dataclass
class COGData:
    position: np.ndarray
    velocity: np.ndarray
    timestamp: float

@dataclass
class StabilityMetrics:
    base_tilt: float
    cog_displacement: float
    total_force: float
    balance_score: float
    timestamp: float

class BiomechanicalAnalyzer:
    def __init__(self, robot_id: int, analysis_duration: float = 60.0):
        self.robot_id = robot_id
        self.analysis_duration = analysis_duration
        self.data_queue = queue.Queue()
        
        # Data storage
        self.force_history = deque(maxlen=1000)
        self.cog_history = deque(maxlen=1000)
        self.stability_history = deque(maxlen=1000)
        self.joint_forces_history = {}
        
        # Analysis parameters
        self.ground_level = 0.0
        self.nominal_height = 1.0
        self.analysis_running = False
        
        # Get robot joint info
        self.joint_info = self._get_joint_info()
        self.link_masses = self._get_link_masses()
        
        # Initialize force tracking for each joint
        for joint_id in self.joint_info:
            self.joint_forces_history[joint_id] = deque(maxlen=1000)
    
    def _get_joint_info(self) -> Dict[int, str]:
        """Get joint information from robot."""
        joint_info = {}
        num_joints = p.getNumJoints(self.robot_id)
        
        for i in range(num_joints):
            info = p.getJointInfo(self.robot_id, i)
            joint_info[i] = info[1].decode('utf-8')
        
        return joint_info
    
    def _get_link_masses(self) -> Dict[int, float]:
        """Get mass information for each link."""
        link_masses = {}
        num_joints = p.getNumJoints(self.robot_id)
        
        # Base link
        base_info = p.getDynamicsInfo(self.robot_id, -1)
        link_masses[-1] = base_info[0]
        
        # Joint links
        for i in range(num_joints):
            dynamics_info = p.getDynamicsInfo(self.robot_id, i)
            link_masses[i] = dynamics_info[0]
        
        return link_masses
    
    def calculate_center_of_gravity(self) -> COGData:
        """Calculate the center of gravity of the robot."""
        total_mass = 0.0
        weighted_position = np.zeros(3)
        
        # Base link
        base_pos, base_orn = p.getBasePositionAndOrientation(self.robot_id)
        base_mass = self.link_masses[-1]
        total_mass += base_mass
        weighted_position += np.array(base_pos) * base_mass
        
        # Joint links
        num_joints = p.getNumJoints(self.robot_id)
        for i in range(num_joints):
            link_state = p.getLinkState(self.robot_id, i)
            link_pos = np.array(link_state[0])
            link_mass = self.link_masses[i]
            
            total_mass += link_mass
            weighted_position += link_pos * link_mass
        
        cog_position = weighted_position / total_mass if total_mass > 0 else np.zeros(3)
        
        # Calculate velocity (approximate using position history)
        velocity = np.zeros(3)
        if len(self.cog_history) > 0:
            dt = time.time() - self.cog_history[-1].timestamp
            if dt > 0:
                velocity = (cog_position - self.cog_history[-1].position) / dt
        
        return COGData(
            position=cog_position,
            velocity=velocity,
            timestamp=time.time()
        )
    
    def calculate_joint_forces(self) -> List[ForceData]:
        """Calculate forces and torques at each joint."""
        forces = []
        
        for joint_id in self.joint_info:
            joint_state = p.getJointState(self.robot_id, joint_id)
            
            # Get applied motor torque
            applied_torque = joint_state[3]  # Applied joint motor torque
            
            # Get joint reaction forces (if available)
            try:
                # This requires enabling joint feedback
                joint_info = p.getJointInfo(self.robot_id, joint_id)
                if joint_info[2] != p.JOINT_FIXED:  # Only for movable joints
                    force_magnitude = abs(applied_torque)
                    
                    forces.append(ForceData(
                        joint_id=joint_id,
                        joint_name=self.joint_info[joint_id],
                        force=force_magnitude,
                        torque=applied_torque,
                        timestamp=time.time()
                    ))
            except:
                pass
        
        return forces
    
    def calculate_stability_metrics(self, cog_data: COGData, forces: List[ForceData]) -> StabilityMetrics:
        """Calculate stability metrics."""
        # Base orientation
        base_pos, base_orn = p.getBasePositionAndOrientation(self.robot_id)
        euler = p.getEulerFromQuaternion(base_orn)
        base_tilt = np.sqrt(euler[0]**2 + euler[1]**2)  # Total tilt
        
        # COG displacement from nominal
        nominal_cog = np.array([0.0, 0.0, self.nominal_height])
        cog_displacement = np.linalg.norm(cog_data.position - nominal_cog)
        
        # Total force magnitude
        total_force = sum(f.force for f in forces)
        
        # Balance score (0-100, higher is better)
        tilt_penalty = min(100, np.degrees(base_tilt) * 10)
        cog_penalty = min(100, cog_displacement * 50)
        force_penalty = min(100, total_force / 100)
        
        balance_score = max(0, 100 - tilt_penalty - cog_penalty - force_penalty)
        
        return StabilityMetrics(
            base_tilt=base_tilt,
            cog_displacement=cog_displacement,
            total_force=total_force,
            balance_score=balance_score,
            timestamp=time.time()
        )
    
    def analyze_frame(self):
        """Analyze one frame of the simulation."""
        if not self.analysis_running:
            return
        
        # Calculate COG
        cog_data = self.calculate_center_of_gravity()
        self.cog_history.append(cog_data)
        
        # Calculate forces
        forces = self.calculate_joint_forces()
        self.force_history.extend(forces)
        
        # Update joint force history
        for force_data in forces:
            if force_data.joint_id in self.joint_forces_history:
                self.joint_forces_history[force_data.joint_id].append(force_data)
        
        # Calculate stability metrics
        stability = self.calculate_stability_metrics(cog_data, forces)
        self.stability_history.append(stability)
        
        # Put data in queue for visualization
        self.data_queue.put({
            'cog': cog_data,
            'forces': forces,
            'stability': stability
        })
    
    def start_analysis(self):
        """Start the biomechanical analysis."""
        self.analysis_running = True
        print("Starting biomechanical analysis...")
        
        start_time = time.time()
        while self.analysis_running and (time.time() - start_time) < self.analysis_duration:
            self.analyze_frame()
            time.sleep(1.0/60.0)  # 60 Hz analysis
        
        print("Analysis completed.")
    
    def stop_analysis(self):
        """Stop the analysis."""
        self.analysis_running = False
    
    def generate_report(self):
        """Generate analysis report."""
        if not self.cog_history:
            print("No data available for report generation.")
            return
        
        print("\n" + "="*60)
        print("BIOMECHANICAL ANALYSIS REPORT")
        print("="*60)
        
        # COG Analysis
        cog_positions = np.array([cog.position for cog in self.cog_history])
        cog_velocities = np.array([cog.velocity for cog in self.cog_history])
        
        print(f"\nCENTER OF GRAVITY ANALYSIS:")
        print(f"  Average COG Position: [{cog_positions.mean(axis=0)[0]:.3f}, {cog_positions.mean(axis=0)[1]:.3f}, {cog_positions.mean(axis=0)[2]:.3f}]")
        print(f"  COG Range X: {cog_positions[:,0].max() - cog_positions[:,0].min():.3f}m")
        print(f"  COG Range Y: {cog_positions[:,1].max() - cog_positions[:,1].min():.3f}m")
        print(f"  COG Range Z: {cog_positions[:,2].max() - cog_positions[:,2].min():.3f}m")
        print(f"  Average COG Velocity: {np.linalg.norm(cog_velocities.mean(axis=0)):.3f}m/s")
        
        # Force Analysis
        if self.force_history:
            total_forces = [f.force for f in self.force_history]
            print(f"\nFORCE ANALYSIS:")
            print(f"  Average Total Force: {np.mean(total_forces):.1f}N")
            print(f"  Max Force: {np.max(total_forces):.1f}N")
            print(f"  Force Standard Deviation: {np.std(total_forces):.1f}N")
        
        # Stability Analysis
        if self.stability_history:
            tilts = [s.base_tilt for s in self.stability_history]
            displacements = [s.cog_displacement for s in self.stability_history]
            scores = [s.balance_score for s in self.stability_history]
            
            print(f"\nSTABILITY ANALYSIS:")
            print(f"  Average Base Tilt: {np.degrees(np.mean(tilts)):.2f}°")
            print(f"  Max Base Tilt: {np.degrees(np.max(tilts)):.2f}°")
            print(f"  Average COG Displacement: {np.mean(displacements):.3f}m")
            print(f"  Average Balance Score: {np.mean(scores):.1f}/100")
        
        # Joint Force Analysis
        print(f"\nJOINT FORCE ANALYSIS:")
        for joint_id, force_data in self.joint_forces_history.items():
            if force_data:
                forces = [f.force for f in force_data]
                joint_name = self.joint_info[joint_id]
                print(f"  {joint_name}: Avg={np.mean(forces):.1f}N, Max={np.max(forces):.1f}N")
        
        print("="*60)

class BiomechanicalVisualizer:
    def __init__(self, analyzer: BiomechanicalAnalyzer):
        self.analyzer = analyzer
        self.fig, self.axes = plt.subplots(2, 2, figsize=(15, 10))
        self.fig.suptitle('Real-time Biomechanical Analysis', fontsize=16)
        
        # Initialize plots
        self.setup_plots()
        
        # Data for animation
        self.time_data = deque(maxlen=200)
        self.cog_x_data = deque(maxlen=200)
        self.cog_y_data = deque(maxlen=200)
        self.cog_z_data = deque(maxlen=200)
        self.force_data = deque(maxlen=200)
        self.tilt_data = deque(maxlen=200)
        self.balance_data = deque(maxlen=200)
        
        self.start_time = time.time()
    
    def setup_plots(self):
        """Setup the visualization plots."""
        # COG Position plot
        self.axes[0,0].set_title('Center of Gravity Position')
        self.axes[0,0].set_xlabel('Time (s)')
        self.axes[0,0].set_ylabel('Position (m)')
        self.axes[0,0].legend(['X', 'Y', 'Z'])
        self.axes[0,0].grid(True)
        
        # Force plot
        self.axes[0,1].set_title('Total Joint Forces')
        self.axes[0,1].set_xlabel('Time (s)')
        self.axes[0,1].set_ylabel('Force (N)')
        self.axes[0,1].grid(True)
        
        # Stability plot
        self.axes[1,0].set_title('Base Tilt Angle')
        self.axes[1,0].set_xlabel('Time (s)')
        self.axes[1,0].set_ylabel('Tilt (degrees)')
        self.axes[1,0].grid(True)
        
        # Balance score plot
        self.axes[1,1].set_title('Balance Score')
        self.axes[1,1].set_xlabel('Time (s)')
        self.axes[1,1].set_ylabel('Score (0-100)')
        self.axes[1,1].grid(True)
    
    def update_plots(self, frame):
        """Update plots with new data."""
        # Get data from analyzer
        try:
            while not self.analyzer.data_queue.empty():
                data = self.analyzer.data_queue.get_nowait()
                
                current_time = time.time() - self.start_time
                self.time_data.append(current_time)
                
                # COG data
                cog = data['cog']
                self.cog_x_data.append(cog.position[0])
                self.cog_y_data.append(cog.position[1])
                self.cog_z_data.append(cog.position[2])
                
                # Force data
                total_force = sum(f.force for f in data['forces'])
                self.force_data.append(total_force)
                
                # Stability data
                stability = data['stability']
                self.tilt_data.append(np.degrees(stability.base_tilt))
                self.balance_data.append(stability.balance_score)
        except queue.Empty:
            pass
        
        # Clear and update plots
        for ax in self.axes.flat:
            ax.clear()
        
        self.setup_plots()
        
        if len(self.time_data) > 1:
            # COG plot
            self.axes[0,0].plot(self.time_data, self.cog_x_data, 'r-', label='X', linewidth=2)
            self.axes[0,0].plot(self.time_data, self.cog_y_data, 'g-', label='Y', linewidth=2)
            self.axes[0,0].plot(self.time_data, self.cog_z_data, 'b-', label='Z', linewidth=2)
            self.axes[0,0].legend()
            
            # Force plot
            self.axes[0,1].plot(self.time_data, self.force_data, 'purple', linewidth=2)
            
            # Tilt plot
            self.axes[1,0].plot(self.time_data, self.tilt_data, 'orange', linewidth=2)
            
            # Balance plot
            self.axes[1,1].plot(self.time_data, self.balance_data, 'green', linewidth=2)
            self.axes[1,1].set_ylim(0, 100)
    
    def start_visualization(self):
        """Start the real-time visualization."""
        self.animation = FuncAnimation(self.fig, self.update_plots, interval=100, blit=False)
        plt.tight_layout()
        plt.show()

def run_biomechanical_analysis(robot_id: int, duration: float = 60.0):
    """Main function to run biomechanical analysis."""
    # Enable joint feedback for force measurements
    p.enableJointFeedback(robot_id, range(p.getNumJoints(robot_id)))
    
    # Create analyzer
    analyzer = BiomechanicalAnalyzer(robot_id, duration)
    
    # Create visualizer
    visualizer = BiomechanicalVisualizer(analyzer)
    
    # Start analysis in separate thread
    analysis_thread = threading.Thread(target=analyzer.start_analysis)
    analysis_thread.daemon = True
    analysis_thread.start()
    
    try:
        # Start visualization (blocking)
        visualizer.start_visualization()
    except KeyboardInterrupt:
        print("Analysis interrupted by user")
    finally:
        analyzer.stop_analysis()
        analysis_thread.join(timeout=1.0)
        
        # Generate final report
        analyzer.generate_report()

# Example usage - call this after loading your robot
if __name__ == "__main__":
    # This would be called with your robot_id from the main simulation
    # run_biomechanical_analysis(robot_id, duration=60.0)
    print("Biomechanical analyzer ready. Call run_biomechanical_analysis(robot_id) after loading your robot.")