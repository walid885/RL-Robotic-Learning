# src/core/joint_manager.py
import pybullet as p
from typing import List, Dict
from ..utils.data_structures import JointInfo

class JointManager:
    """Joint discovery, categorization, and information management."""
    
    def __init__(self, robot_id: int):
        self.robot_id = robot_id
        self.joints = {}
        self.joint_categories = {}
        
    def discover_joints(self) -> List[JointInfo]:
        """Get all movable joints from robot."""
        num_joints = p.getNumJoints(self.robot_id)
        movable_types = [p.JOINT_REVOLUTE, p.JOINT_PRISMATIC]
        
        movable_joints = []
        for i in range(num_joints):
            joint_info = self._get_joint_info(i)
            if joint_info.type in movable_types:
                movable_joints.append(joint_info)
                self.joints[joint_info.id] = joint_info
                
        return movable_joints
    
    def _get_joint_info(self, joint_id: int) -> JointInfo:
        """Extract joint information from robot."""
        joint_info = p.getJointInfo(self.robot_id, joint_id)
        joint_state = p.getJointState(self.robot_id, joint_id)
        
        return JointInfo(
            id=joint_id,
            name=joint_info[1].decode('utf-8'),
            type=joint_info[2],
            limits=(joint_info[8], joint_info[9]),
            current_position=joint_state[0]
        )
    
    def categorize_joints(self, joints: List[JointInfo]) -> Dict[str, List[JointInfo]]:
        """Find and categorize critical balance joints."""
        categories = {
            'hip': [], 'knee': [], 'ankle': [], 'torso': [],
            'shoulder_left': [], 'elbow_left': [], 'shoulder_right': [],
            'elbow_right': [], 'wrist_right': [], 'other': []
        }
        
        for joint in joints:
            name = joint.name.lower()
            category = self._classify_joint(name)
            categories[category].append(joint)
        
        self.joint_categories = categories
        return categories
    
    def _classify_joint(self, name: str) -> str:
        """Classify joint based on name."""
        if 'hip' in name:
            return 'hip'
        elif 'knee' in name:
            return 'knee'
        elif 'ankle' in name:
            return 'ankle'
        elif 'torso' in name or 'spine' in name or 'waist' in name:
            return 'torso'
        elif ('shoulder' in name or 'arm' in name) and ('left' in name or 'l_' in name):
            return 'shoulder_left'
        elif ('elbow' in name or 'forearm' in name) and ('left' in name or 'l_' in name):
            return 'elbow_left'
        elif ('shoulder' in name or 'arm' in name) and ('right' in name or 'r_' in name):
            return 'shoulder_right'
        elif ('elbow' in name or 'forearm' in name) and ('right' in name or 'r_' in name):
            return 'elbow_right'
        elif ('wrist' in name or 'hand' in name) and ('right' in name or 'r_' in name):
            return 'wrist_right'
        else:
            return 'other'
