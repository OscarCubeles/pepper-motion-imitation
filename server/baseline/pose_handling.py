import settings
from dual_singularity_fsm import DualSingularityFSM


class PoseHandler:
    """
    Manages pose constraint handling and singularity FSM instances.
    
    Maintains separate FSM instances for each arm and singularity type,
    providing a unified interface for constraint application and state tracking.
    """
    
    def __init__(self):
        """Initialize PoseHandler with FSM instances for each arm and singularity type."""
        # Dual singularity FSM instances (one per arm)
        self.dual_singularity_fsm_right = DualSingularityFSM(arm='right')
        self.dual_singularity_fsm_left = DualSingularityFSM(arm='left')
        
        # TODO: Elbow and shoulder singularity FSM instances will be added here
        # self.elbow_singularity_fsm_right = ElbowSingularityFSM(arm='right')
        # self.elbow_singularity_fsm_left = ElbowSingularityFSM(arm='left')
        # self.shoulder_singularity_fsm_right = ShoulderSingularityFSM(arm='right')
        # self.shoulder_singularity_fsm_left = ShoulderSingularityFSM(arm='left')
    
    def _get_dual_singularity_fsm(self, arm='right'):
        """Get the appropriate DualSingularityFSM instance for the arm."""
        if arm.lower() == 'left':
            return self.dual_singularity_fsm_left
        else:
            return self.dual_singularity_fsm_right
    
    def apply_singularity_constraints(
        self, 
        singularity, 
        pepper_angles, 
        orientation_labels=None, 
        arm='right'
    ):
        """
        Apply appropriate singularity constraints based on singularity type for a single arm.
        
        Always updates FSMs (even when no singularity), allowing state tracking across frames.
        
        Args:
            singularity: Singularity dict for the arm (or None if no singularity)
            pepper_angles: Dictionary of pepper joint angles (will be modified)
            orientation_labels: Optional dict with hand orientation labels
            arm: 'right' or 'left' to identify which arm's singularity is being processed
            
        Returns:
            Modified pepper_angles dictionary with constraints applied
        """
        if not singularity:
            singularity = {}
        
        has_dual_singularity = singularity.get("has_singularity", False)
        singularity_type = singularity.get("singularity_type", "")
        
        # Update dual singularity FSM regardless of type (provides state info)
        if "dual" in str(singularity_type):
            fsm = self._get_dual_singularity_fsm(arm)
            pepper_angles = fsm.update(pepper_angles, has_dual_singularity, orientation_labels)
        
        # TODO: Add elbow and shoulder FSM updates here
        # elif "elbow" in str(singularity_type):
        #     fsm = self._get_elbow_singularity_fsm(arm)
        #     pepper_angles = fsm.update(pepper_angles, has_elbow_singularity, orientation_labels)
        # elif "shoulder" in str(singularity_type):
        #     fsm = self._get_shoulder_singularity_fsm(arm)
        #     pepper_angles = fsm.update(pepper_angles, has_shoulder_singularity, orientation_labels)
        
        return pepper_angles
    
    def get_singularity_fsm_state(self, arm='right'):
        """
        Get current FSM state information for debugging/visualization.
        
        Args:
            arm: 'right' or 'left'
            
        Returns:
            Dict with FSM state information
        """
        fsm = self._get_dual_singularity_fsm(arm)
        return {
            "dual_singularity": fsm.get_state_info(),
        }
    
    @staticmethod
    def compute_joint_speeds(pepper_angles_prev, pepper_angles_curr, dt=0.033):
        """
        Compute joint speeds from angle differences.
        
        Args:
            pepper_angles_prev: Previous frame angles dict
            pepper_angles_curr: Current frame angles dict
            dt: Time delta in seconds (default ~30ms for 30 FPS)
            
        Returns:
            Dictionary with speeds for each joint (rad/s)
        """
        if not pepper_angles_prev or not pepper_angles_curr:
            return {}
        
        speeds = {}
        for joint_name, curr_angle in pepper_angles_curr.items():
            if curr_angle is None:
                speeds[joint_name] = 0.5  # Default speed
                continue
            
            prev_angle = pepper_angles_prev.get(joint_name)
            if prev_angle is None:
                speeds[joint_name] = 0.5
                continue
            
            # Compute angular velocity: rad/s
            angle_diff = abs(curr_angle - prev_angle)
            angular_velocity = angle_diff / dt if dt > 0 else 0.0
            
            # Clamp speed to reasonable range [0.1, 1.0]
            speed = max(0.1, min(1.0, angular_velocity))
            speeds[joint_name] = speed
        
        return speeds


# Deprecated: These functions are kept for backward compatibility
# Use PoseHandler class methods instead

# Deprecated: These functions are kept for backward compatibility
# Use PoseHandler class methods instead

def apply_dual_singularity_constraints(singularity, pepper_angles, orientation_labels=None, arm='right'):
    """Deprecated: Use PoseHandler.apply_singularity_constraints() instead."""
    # For backward compatibility, create a temporary singularity dict if needed
    if not isinstance(singularity, dict):
        singularity = {"has_singularity": bool(singularity), "singularity_type": "dual"}
    
    # Apply dual singularity constraints with orientation labels
    arm_label = arm.capitalize()
    arm_orientation_dict = orientation_labels.get(arm_label) if orientation_labels else None
    
    shoulder_pitch_key = f"ShoulderPitch_{arm_label}"
    shoulder_roll_key = f"ShoulderRoll_{arm_label}"
    elbow_yaw_key = f"ElbowYaw_{arm_label}"
    wrist_yaw_key = f"WristYaw_{arm_label}"
    elbow_roll_key = f"ElbowRoll_{arm_label}"
    
    arm_orientation = arm_orientation_dict.get('primary') if arm_orientation_dict else None

    if arm_orientation and arm_orientation in settings.HAND_ORIENTATION_TO_SHOULDER_PITCH:
        pepper_angles[shoulder_pitch_key] = settings.HAND_ORIENTATION_TO_SHOULDER_PITCH[arm_orientation]
    
    pepper_angles[shoulder_roll_key] = settings.SHOULDER_ROLL_BY_ARM[arm.lower()]
    pepper_angles[elbow_yaw_key] = 0.0
    pepper_angles[wrist_yaw_key] = 0.0
    pepper_angles[elbow_roll_key] = settings.ELBOW_ROLL_SAFE_ANGLE

    return pepper_angles


def apply_elbow_singularity_constraints(singularity, pepper_angles, orientation_labels=None, arm='right'):
    """Deprecated: Use PoseHandler.apply_singularity_constraints() instead."""
    if not singularity or not singularity.get("has_singularity"):
        return pepper_angles
    
    arm_label = arm.capitalize()
    shoulder_roll_key = f"ShoulderRoll_{arm_label}"
    elbow_roll_key = f"ElbowRoll_{arm_label}"
    
    pepper_angles[shoulder_roll_key] = settings.SHOULDER_ROLL_BY_ARM[arm.lower()]
    pepper_angles[elbow_roll_key] = settings.ELBOW_ROLL_SAFE_ANGLE
    
    return pepper_angles


def apply_shoulder_singularity_constraints(singularity, pepper_angles, orientation_labels=None, arm='right'):
    """Deprecated: Use PoseHandler.apply_singularity_constraints() instead."""
    if not singularity or not singularity.get("has_singularity"):
        return pepper_angles
    
    arm_label = arm.capitalize()
    shoulder_pitch_key = f"ShoulderPitch_{arm_label}"
    
    arm_orientation_dict = orientation_labels.get(arm_label) if orientation_labels else None
    arm_orientation = arm_orientation_dict.get('primary') if arm_orientation_dict else None
    
    if arm_orientation and arm_orientation in settings.HAND_ORIENTATION_TO_SHOULDER_PITCH:
        pepper_angles[shoulder_pitch_key] = settings.HAND_ORIENTATION_TO_SHOULDER_PITCH[arm_orientation]
    
    return pepper_angles


def apply_singularity_constraints(singularity, pepper_angles, orientation_labels=None, arm='right'):
    """Deprecated: Use PoseHandler.apply_singularity_constraints() instead."""
    if not singularity or not singularity.get("has_singularity"):
        return pepper_angles
    
    singularity_type = singularity.get("singularity_type", "")
    
    if "dual" in str(singularity_type):
        pepper_angles = apply_dual_singularity_constraints(singularity, pepper_angles, orientation_labels, arm=arm)
    elif "elbow" in str(singularity_type):
        pepper_angles = apply_elbow_singularity_constraints(singularity, pepper_angles, orientation_labels, arm=arm)
    elif "shoulder" in str(singularity_type):
        pepper_angles = apply_shoulder_singularity_constraints(singularity, pepper_angles, orientation_labels, arm=arm)
    
    return pepper_angles


def compute_joint_speeds(pepper_angles_prev, pepper_angles_curr, dt=0.033):
    """Deprecated: Use PoseHandler.compute_joint_speeds() instead."""
    return PoseHandler.compute_joint_speeds(pepper_angles_prev, pepper_angles_curr, dt=dt)