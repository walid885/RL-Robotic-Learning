# models/joint_info.py - Data models
from dataclasses import dataclass
from typing import Tuple


@dataclass
class JointInfo:
    """Information about a robot joint."""
    id: int
    name: str
    type: int
    limits: Tuple[float, float]
    current_position: float
    
    def is_valid_limits(self) -> bool:
        """Check if joint limits are valid and reasonable."""
        lower, upper = self.limits
        return lower < upper and abs(lower) < 100 and abs(upper) < 100
    
    def get_safe_range(self) -> float:
        """Get a safe range for joint movement."""
        if self.is_valid_limits():
            return min(0.2, (self.limits[1] - self.limits[0]) * 0.1)
        return 0.0


