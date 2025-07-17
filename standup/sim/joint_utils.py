# utils/joint_utils.py - Joint analysis utilities
import pybullet as p
from typing import List, Tuple
from models.joint_info import JointInfo


class JointAnalyzer:
    """Analyzes and categorizes robot joints."""
    
    def get_joint_info(self, robot_id: int, joint_id: int) -> JointInfo:
        """Extract joint information from robot."""
        joint_info = p.getJointInfo(robot_id, joint_id)
        joint_state = p.getJointState(robot_id, joint_id)
        
        return JointInfo(
            id=joint_id,
            name=joint_info[1].decode('utf-8'),
            type=joint_info[2],
            limits=(joint_info[8], joint_info[9]),
            current_position=joint_state[0]
        )
    def get_movable_joints(self, robot_id: int) -> List[JointInfo]:
        """Get all movable joints from robot."""
        num_joints = p.getNumJoints(robot_id)
        movable_types = [p.JOINT_REVOLUTE, p.JOINT_PRISMATIC]
        
        return [
            self.get_joint_info(robot_id, i)
            for i in range(num_joints)
            if self.get_joint_info(robot_id, i).type in movable_types
        ]
    
    def filter_joints_by_keywords(self, joints: List[JointInfo], keywords: List[str]) -> List[JointInfo]:
        """Filter joints by name keywords."""
        return [
            joint for joint in joints
            if any(keyword in joint.name.lower() for keyword in keywords)
        ]
    
    def find_joint_groups(self, joints: List[JointInfo]) -> Tuple[List[JointInfo], List[JointInfo]]:
        """Find arm and balance joint groups."""
        arm_keywords = [
            'rightarm', 'right_arm', 'rightshoulder', 'rightelbow', 
            'rightwrist', 'r_arm', 'r_shoulder', 'r_elbow'
        ]
        balance_keywords = [
            'leg', 'ankle', 'hip', 'knee', 'foot', 'torso', 'waist'
        ]
        
        arm_joints = self.filter_joints_by_keywords(joints, arm_keywords)
        if not arm_joints:
            arm_joints = self.filter_joints_by_keywords(joints, ['arm', 'shoulder'])[:3]
        
        balance_joints = self.filter_joints_by_keywords(joints, balance_keywords)
        
        return arm_joints, balance_joints


