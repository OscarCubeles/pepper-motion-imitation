"""
Dual Singularity FSM - handles gradual constraint application during dual singularity.

Uses a 3-state FSM to manage smooth transitions and constraint application:
    NO_SINGULAR: No dual singularity detected
    TRANSITION_TO_SINGULAR: Singularity detected for 1-4 frames (gradual reduction phase)
    STEADY_SINGULAR: Singularity detected for 5+ frames (steady constraint phase)
"""
from enum import Enum
from typing import Dict, Optional
import numpy as np
from numpy import isnan
import settings


class DualSingularityState(Enum):
    """Dual singularity FSM states."""
    NO_SINGULAR = "no_singular"
    TRANSITION_TO_SINGULAR = "transition_to_singular"
    STEADY_SINGULAR = "steady_singular"


class DualSingularityFSM:
    """
    FSM for handling dual singularity constraints with gradual angle adjustments.
    
    States and Transitions:
        NO_SINGULAR:
            - Stay: No singularity detected
            - → TRANSITION_TO_SINGULAR: Singularity detected for 2 consecutive frames
        
        TRANSITION_TO_SINGULAR (frames 1-4):
            - Gradually reduce ElbowRoll, ElbowYaw, WristYaw to 0 over 5 frames
            - → STEADY_SINGULAR: Singularity detected for 5 consecutive frames total
            - → NO_SINGULAR: No singularity for 2 consecutive frames
        
        STEADY_SINGULAR (frames 5+):
            - Gradually move ShoulderRoll to ±89.5° over 5 frames
            - Apply orientation-based ShoulderPitch constraint
            - → TRANSITION_TO_SINGULAR: No singularity for 1 frame
    """
    
    def __init__(self, arm: str = "right"):
        """
        Initialize Dual Singularity FSM.
        
        Args:
            arm: "right" or "left"
        """
        self.arm = arm.lower()
        if self.arm not in ("right", "left"):
            raise ValueError("arm must be 'right' or 'left'")
        
        self.state = DualSingularityState.NO_SINGULAR
        self.singularity_frame_count = 0  # Counts consecutive frames with singularity
        self.clear_frame_count = 0  # Counts consecutive frames without singularity
        
        # For tracking gradual angle changes
        self.frame_in_state = 0  # Frame counter within current state
        self.start_angles = {}  # Starting angles for interpolation
        
        # Transition thresholds
        self.singularity_transition_threshold = 2  # frames to enter TRANSITION_TO_SINGULAR
        self.singularity_steady_threshold = 5  # frames to enter STEADY_SINGULAR
        self.clear_transition_threshold = 2  # frames in TRANSITION to go back to NO_SINGULAR
        self.clear_steady_threshold = 1  # frames in STEADY to go back to TRANSITION
        
        # Interpolation parameters
        self.transition_duration = 5  # frames to reduce joints to 0
        self.steady_duration = 5  # frames to reach target ShoulderRoll
    
    def _get_target_shoulder_roll(self) -> float:
        """Get target ShoulderRoll angle in degrees based on arm."""
        if self.arm == "left":
            return 89.5  # degrees
        else:  # right
            return -89.5  # degrees
    
    def _interpolate_angle(
        self, 
        start_angle: float, 
        target_angle: float, 
        current_frame: int, 
        total_frames: int
    ) -> float:
        """
        Linearly interpolate from start_angle to target_angle.
        
        Args:
            start_angle: Starting angle value
            target_angle: Target angle value
            current_frame: Current frame number (0-indexed)
            total_frames: Total frames for interpolation
            
        Returns:
            Interpolated angle
        """
        if current_frame >= total_frames:
            return target_angle
        
        progress = current_frame / total_frames
        return start_angle + (target_angle - start_angle) * progress
    
    def _compute_constrained_angles(
        self, 
        pepper_angles: Dict[str, float],
        has_dual_singularity: bool
    ) -> Dict[str, float]:
        """
        Compute constrained angles based on current FSM state.
        
        Args:
            pepper_angles: Current pepper joint angles dict
            has_dual_singularity: Whether dual singularity is currently detected
            
        Returns:
            Modified pepper_angles with constraints applied
        """
        arm_label = self.arm.capitalize()
        
        # Joint keys
        elbow_roll_key = f"ElbowRoll_{arm_label}"
        elbow_yaw_key = f"ElbowYaw_{arm_label}"
        wrist_yaw_key = f"WristYaw_{arm_label}"
        shoulder_roll_key = f"ShoulderRoll_{arm_label}"
        shoulder_pitch_key = f"ShoulderPitch_{arm_label}"
        
        if self.state == DualSingularityState.NO_SINGULAR:
            # No constraints applied
            return pepper_angles
        
        elif self.state == DualSingularityState.TRANSITION_TO_SINGULAR:
            # Gradually reduce ElbowRoll, ElbowYaw, WristYaw to 0 over 5 frames
            
            # Initialize start angles on first frame of this state
            if self.frame_in_state == 0:
                self.start_angles = {
                    elbow_roll_key: pepper_angles.get(elbow_roll_key, 0.0),
                    elbow_yaw_key: pepper_angles.get(elbow_yaw_key, 0.0),
                    wrist_yaw_key: pepper_angles.get(wrist_yaw_key, 0.0),
                }
            
            # Interpolate toward 0
            pepper_angles[elbow_roll_key] = self._interpolate_angle(
                self.start_angles.get(elbow_roll_key, 0.0),
                0.0,
                self.frame_in_state,
                self.transition_duration
            )
            pepper_angles[elbow_yaw_key] = self._interpolate_angle(
                self.start_angles.get(elbow_yaw_key, 0.0),
                0.0,
                self.frame_in_state,
                self.transition_duration
            )
            pepper_angles[wrist_yaw_key] = self._interpolate_angle(
                self.start_angles.get(wrist_yaw_key, 0.0),
                0.0,
                self.frame_in_state,
                self.transition_duration
            )
        
        elif self.state == DualSingularityState.STEADY_SINGULAR:
            # Set ElbowRoll, ElbowYaw, WristYaw to 0 (already at target from TRANSITION)
            pepper_angles[elbow_yaw_key] = 0.0
            pepper_angles[wrist_yaw_key] = 0.0
            pepper_angles[elbow_roll_key] = settings.ELBOW_ROLL_SAFE_ANGLE
            
            # Gradually move ShoulderRoll to target over 5 frames
            if self.frame_in_state == 0:
                self.start_angles[shoulder_roll_key] = pepper_angles.get(shoulder_roll_key, 0.0)
            
            # Convert target to radians
            target_shoulder_roll_deg = self._get_target_shoulder_roll()
            target_shoulder_roll_rad = np.radians(target_shoulder_roll_deg)
            
            pepper_angles[shoulder_roll_key] = self._interpolate_angle(
                self.start_angles.get(shoulder_roll_key, 0.0),
                target_shoulder_roll_rad,
                self.frame_in_state,
                self.steady_duration
            )
        
        return pepper_angles
    
    def update(
        self, 
        pepper_angles: Dict[str, float],
        has_dual_singularity: bool,
        orientation_labels: Optional[Dict] = None
    ) -> Dict[str, float]:
        """
        Update FSM state and compute constrained angles.
        
        Args:
            pepper_angles: Current pepper joint angles dict
            has_dual_singularity: Whether dual singularity is detected this frame
            orientation_labels: Optional hand orientation labels for ShoulderPitch constraint
            
        Returns:
            Modified pepper_angles with constraints applied based on FSM state
        """
        # Update frame counters
        if has_dual_singularity:
            self.singularity_frame_count += 1
            self.clear_frame_count = 0
        else:
            self.clear_frame_count += 1
            self.singularity_frame_count = 0
        
        # State machine transitions
        if self.state == DualSingularityState.NO_SINGULAR:
            if self.singularity_frame_count >= self.singularity_transition_threshold:
                self.state = DualSingularityState.TRANSITION_TO_SINGULAR
                self.frame_in_state = 0
                self.start_angles = {}
        
        elif self.state == DualSingularityState.TRANSITION_TO_SINGULAR:
            if self.singularity_frame_count >= self.singularity_steady_threshold:
                self.state = DualSingularityState.STEADY_SINGULAR
                self.frame_in_state = 0
                self.start_angles = {}
            elif self.clear_frame_count >= self.clear_transition_threshold:
                self.state = DualSingularityState.NO_SINGULAR
                self.frame_in_state = 0
                self.start_angles = {}
        
        elif self.state == DualSingularityState.STEADY_SINGULAR:
            if self.clear_frame_count >= self.clear_steady_threshold:
                self.state = DualSingularityState.TRANSITION_TO_SINGULAR
                self.frame_in_state = 0
                self.start_angles = {}
        
        # Increment frame counter within state if singularity is detected
        if has_dual_singularity and self.state != DualSingularityState.NO_SINGULAR:
            self.frame_in_state += 1
        
        # Compute constrained angles based on current state
        pepper_angles = self._compute_constrained_angles(pepper_angles, has_dual_singularity)
        
        # Apply orientation-based ShoulderPitch constraint in TRANSITION and STEADY states
        if self.state != DualSingularityState.NO_SINGULAR and orientation_labels:
            arm_label = self.arm.capitalize()
            shoulder_pitch_key = f"ShoulderPitch_{arm_label}"
            
            arm_orientation_dict = orientation_labels.get(arm_label)
            arm_orientation = arm_orientation_dict.get('primary') if arm_orientation_dict else None
            
            if arm_orientation and arm_orientation in settings.HAND_ORIENTATION_TO_SHOULDER_PITCH:
                pepper_angles[shoulder_pitch_key] = settings.HAND_ORIENTATION_TO_SHOULDER_PITCH[arm_orientation]
        
        return pepper_angles
    
    def get_state_name(self) -> str:
        """Get human-readable state name."""
        return self.state.value
    
    def get_state_info(self) -> Dict:
        """Get detailed state information."""
        return {
            "state": self.get_state_name(),
            "singularity_frames": self.singularity_frame_count,
            "clear_frames": self.clear_frame_count,
            "frame_in_state": self.frame_in_state,
        }
