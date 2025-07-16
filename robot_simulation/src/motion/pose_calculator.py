# src/motion/pose_calculator.py
from typing import Dict, List
from ..utils.data_structures import JointInfo

class PoseCalculator:
    """Calculate optimal poses for different robot states."""
    
    def __init__(self):
        pass
    
    def calculate_standing_pose(self, joint_categories: Dict[str, List[JointInfo]]) -> Dict[int, float]:
        """Calculate stable standing pose."""
        positions = {}
        
        # Hip joints: slight forward lean for stability
        for joint in joint_categories['hip']:
            if 'pitch' in joint.name.lower() or 'y' in joint.name.lower():
                positions[joint.id] = -0.1
            else:
                positions[joint.id] = 0.0
        
        # Knee joints: slight bend for shock absorption
        for joint in joint_categories['knee']:
            positions[joint.id] = 0.05
        
        # Ankle joints: flat foot stance
        for joint in joint_categories['ankle']:
            positions[joint.id] = 0.0
        
        # Torso: upright
        for joint in joint_categories['torso']:
            positions[joint.id] = 0.0
        
        # Arms: neutral positions
        for joint in joint_categories['shoulder_left'] + joint_categories['shoulder_right']:
            positions[joint.id] = 0.0
        for joint in joint_categories['elbow_left'] + joint_categories['elbow_right']:
            positions[joint.id] = 0.1
        for joint in joint_categories['wrist_right']:
            positions[joint.id] = 0.0
        
        # Other joints: current position
        for joint in joint_categories['other']:
            positions[joint.id] = joint.current_position
        
        return positions
    
    def calculate_wave_ready_pose(self, joint_categories: Dict[str, List[JointInfo]]) -> Dict[int, float]:
        """Calculate pose ready for waving motion."""
        positions = self.calculate_standing_pose(joint_categories)
        
        # Modify right arm for waving readiness
        for joint in joint_categories['shoulder_right']:
            if 'pitch' in joint.name.lower() or 'y' in joint.name.lower():
                positions[joint.id] = 0.3  # Lift shoulder
        
        return positions

