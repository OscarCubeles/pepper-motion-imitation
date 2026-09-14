import kinematics.constants as constants

import threading


class KinematicsSingleton:
    """
    Singleton for managing kinematic configuration values.
    Allows dynamic adjustment of IK branch switch margin based on conditions.
    """
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialize()
        return cls._instance
    
    def _initialize(self):
        """Initialize default values."""
        self._ik_branch_switch_margin_mm = 25.0  # Default value in mm
    
    def get_ik_branch_switch_margin(self):
        """Get the current IK branch switch margin."""
        return self._ik_branch_switch_margin_mm
    
    def set_ik_branch_switch_margin(self, value):
        """
        Set the IK branch switch margin.
        
        Args:
            value: Margin in millimeters (float)
        """
        if not isinstance(value, (int, float)) or value < 0:
            raise ValueError(f"IK_BRANCH_SWITCH_MARGIN_MM must be a non-negative number, got {value}")
        self._ik_branch_switch_margin_mm = value
    

# Global singleton instance
_kinematics_config = KinematicsSingleton()

def get_kinematics_config():
    """Get the global kinematics configuration singleton."""
    return _kinematics_config