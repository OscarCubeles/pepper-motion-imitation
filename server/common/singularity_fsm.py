from enum import Enum
import numpy as np
from numpy import isnan
from typing import Dict, Optional, List
from server.common.kinematics.kinematics_singleton import get_kinematics_config
import server.common.kinematics.constants as constants

class SingularityState(Enum):
    """Singularity detection states."""
    NON_SINGULAR = "non_singular"
    SINGULAR = "singular"
    CONFIDENT_SINGULAR = "confident_singular"


class ElbowRollStateMachine:
    """
    3-state FSM for ElbowRoll singularity detection (singularity at 0°).
    
    States:
        NON_SINGULAR: |elbow_roll| >= 15° - safe zone
        SINGULAR: 7.5° <= |elbow_roll| < 15° - warning zone
        CONFIDENT_SINGULAR: |elbow_roll| < 7.5° - critical zone
    
    Hysteresis prevents oscillation between states.
    No direct transition from NON_SINGULAR to CONFIDENT_SINGULAR.
    """
    
    def __init__(self):
        """Initialize ElbowRoll FSM."""
        self.state = SingularityState.NON_SINGULAR
        self.warning_threshold = 23.0  # degrees
        self.critical_threshold = 7.5  # degrees
    
    @staticmethod
    def _radians_to_degrees(value: Optional[float]) -> Optional[float]:
        """Safely convert radians to degrees."""
        if value is None:
            return None
        try:
            if isnan(value):
                return None
        except (TypeError, ValueError):
            pass
        return np.degrees(float(value))
    
    def update(self, elbow_roll_rad: Optional[float]) -> SingularityState:
        """
        Update FSM state based on ElbowRoll angle.
        
        Args:
            elbow_roll_rad: ElbowRoll angle in radians
            
        Returns:
            Current state
        """
        if elbow_roll_rad is None:
            return self.state
        
        elbow_roll_deg = abs(self._radians_to_degrees(elbow_roll_rad))
        
        if self.state == SingularityState.NON_SINGULAR:
            if elbow_roll_deg < self.warning_threshold:
                self.state = SingularityState.SINGULAR
        
        elif self.state == SingularityState.SINGULAR:
            if elbow_roll_deg < self.critical_threshold:
                self.state = SingularityState.CONFIDENT_SINGULAR
            elif elbow_roll_deg >= self.warning_threshold:
                self.state = SingularityState.NON_SINGULAR
        
        elif self.state == SingularityState.CONFIDENT_SINGULAR:
            if elbow_roll_deg >= self.critical_threshold:
                self.state = SingularityState.SINGULAR

        # Adjust IK margin based on current state
        adjust_for_singularity(self.state)
        return self.state
    
    def get_state_name(self) -> str:
        """Get human-readable state name."""
        return self.state.value
    
    def is_singular(self) -> bool:
        """Check if in any singular state."""
        return self.state != SingularityState.NON_SINGULAR
    
    def is_confident(self) -> bool:
        """Check if in confident singular state."""
        return self.state == SingularityState.CONFIDENT_SINGULAR


class ShoulderRollStateMachine:
    """
    3-state FSM for ShoulderRoll singularity detection (singularity at ±90°).
    
    States based on distance from ±90° singularity point:
        NON_SINGULAR: distance from 90° > 30° - safe zone
        SINGULAR: 15° < distance from 90° <= 30° - warning zone
        CONFIDENT_SINGULAR: distance from 90° <= 15° - critical zone (very close to singularity)
    
    Hysteresis prevents oscillation between states.
    No direct transition from NON_SINGULAR to CONFIDENT_SINGULAR.
    """
    
    def __init__(self):
        """Initialize ShoulderRoll FSM."""
        self.state = SingularityState.NON_SINGULAR
        self.warning_threshold = 30.0  # degrees distance from 90°
        self.critical_threshold = 15.0  # degrees distance from 90°
    
    @staticmethod
    def _radians_to_degrees(value: Optional[float]) -> Optional[float]:
        """Safely convert radians to degrees."""
        if value is None:
            return None
        try:
            if isnan(value):
                return None
        except (TypeError, ValueError):
            pass
        return np.degrees(float(value))
    
    def update(self, shoulder_roll_rad: Optional[float]) -> SingularityState:
        """
        Update FSM state based on ShoulderRoll angle.
        
        Based on distance from ±90° singularity point.
        
        Args:
            shoulder_roll_rad: ShoulderRoll angle in radians
            
        Returns:
            Current state
        """
        if shoulder_roll_rad is None:
            return self.state
        
        shoulder_roll_deg = self._radians_to_degrees(shoulder_roll_rad)
        dist_from_90 = abs(abs(shoulder_roll_deg) - 90.0)
        
        if self.state == SingularityState.NON_SINGULAR:
            if dist_from_90 <= self.warning_threshold:
                self.state = SingularityState.SINGULAR
        
        elif self.state == SingularityState.SINGULAR:
            if dist_from_90 <= self.critical_threshold:
                self.state = SingularityState.CONFIDENT_SINGULAR
            elif dist_from_90 > self.warning_threshold:
                self.state = SingularityState.NON_SINGULAR
        
        elif self.state == SingularityState.CONFIDENT_SINGULAR:
            if dist_from_90 > self.critical_threshold:
                self.state = SingularityState.SINGULAR
        # Adjust IK margin based on current state
        adjust_for_singularity(self.state)
        return self.state
    
    def get_state_name(self) -> str:
        """Get human-readable state name."""
        return self.state.value
    
    def is_singular(self) -> bool:
        """Check if in any singular state."""
        return self.state != SingularityState.NON_SINGULAR
    
    def is_confident(self) -> bool:
        """Check if in confident singular state."""
        return self.state == SingularityState.CONFIDENT_SINGULAR


class SingularityFSM:
    """
    FSM to track singularity for one arm.
    
    Uses two sub-FSMs: ElbowRollStateMachine and ShoulderRollStateMachine.
    Detects dual singularity when both joints are in singular state.
    """
    
    def __init__(self, arm: str = "right"):
        """
        Initialize FSM for one arm.
        
        Args:
            arm: "right" or "left"
        """
        self.arm = arm.lower()
        if self.arm not in ("right", "left"):
            raise ValueError("arm must be 'right' or 'left'")
        
        self.elbow_fsm = ElbowRollStateMachine()
        self.shoulder_fsm = ShoulderRollStateMachine()
    
    @staticmethod
    def _radians_to_degrees(value: Optional[float]) -> Optional[float]:
        """Safely convert radians to degrees."""
        if value is None:
            return None
        try:
            if isnan(value):
                return None
        except (TypeError, ValueError):
            pass
        return np.degrees(float(value))
    
    def _detect_singularity(self, angles: Dict[str, float]) -> Dict:
        """
        Detect singularity using ElbowRoll and ShoulderRoll state machines.
        
        Args:
            angles: Dictionary with joint angles in RADIANS
            
        Returns:
            Dictionary with singularity information:
            {
                "has_singularity": bool,
                "singularity_type": None or "elbow_roll" or "shoulder_roll" or "dual",
                "confidence": None or "singular" or "confident_singular",
                "warnings": [list of warning strings],
                "details": {...}
            }
        """
        shoulder_roll_key = f"ShoulderRoll_{self.arm.capitalize()}"
        elbow_roll_key = f"ElbowRoll_{self.arm.capitalize()}"
        
        shoulder_roll_rad = angles.get(shoulder_roll_key)
        elbow_roll_rad = angles.get(elbow_roll_key)
        

        # Update state machines
        elbow_state = self.elbow_fsm.update(elbow_roll_rad)
        shoulder_state = self.shoulder_fsm.update(shoulder_roll_rad)
        
        # Get values in degrees for reporting
        elbow_roll_deg = self._radians_to_degrees(elbow_roll_rad)
        shoulder_roll_deg = self._radians_to_degrees(shoulder_roll_rad)
        
        warnings = []
        singularity_type = None
        confidence = None
        
        elbow_is_singular = self.elbow_fsm.is_singular()
        shoulder_is_singular = self.shoulder_fsm.is_singular()
        elbow_is_confident = self.elbow_fsm.is_confident()
        shoulder_is_confident = self.shoulder_fsm.is_confident()
        
        arm_side = self.arm.upper()
        
        # ElbowRoll singularity warnings
        if elbow_is_singular:
            confidence_level = "confident" if elbow_is_confident else "near"
            warnings.append(
                f"{arm_side}: ElbowRoll at {abs(elbow_roll_deg):.1f}° "
                f"({confidence_level} singularity at ≈0°)"
            )
        
        # ShoulderRoll singularity warnings
        if shoulder_is_singular:
            confidence_level = "confident" if shoulder_is_confident else "near"
            warnings.append(
                f"{arm_side}: ShoulderRoll at {shoulder_roll_deg:.1f}° "
                f"({confidence_level} singularity)"
            )
        
        # Determine singularity type and confidence
        if elbow_is_singular and shoulder_is_singular:
            singularity_type = "dual"
            # Dual confidence: use the higher confidence level
            if elbow_is_confident or shoulder_is_confident:
                confidence = "confident_singular"
            else:
                confidence = "singular"
            warnings.insert(
                0,
                f"{arm_side}: DUAL SINGULARITY - both ElbowRoll and ShoulderRoll are "
                f"{'confidently ' if confidence == 'confident_singular' else ''}singular"
            )
        elif elbow_is_singular:
            singularity_type = "elbow_roll"
            confidence = "confident_singular" if elbow_is_confident else "singular"
        elif shoulder_is_singular:
            singularity_type = "shoulder_roll"
            confidence = "confident_singular" if shoulder_is_confident else "singular"
        
        return {
            "has_singularity": singularity_type is not None,
            "singularity_type": singularity_type,
            "confidence": confidence,
            "warnings": warnings,
            "details": {
                "elbow_roll": elbow_roll_rad,
                "elbow_state": self.elbow_fsm.get_state_name(),
                "shoulder_roll": shoulder_roll_rad,
                "shoulder_state": self.shoulder_fsm.get_state_name(),
            },
        }
    
    def update(self, angles: Dict[str, float], pose_feasibility: Dict) -> SingularityState:
        """
        Update singularity detection for this arm.
        
        Args:
            angles: Dictionary with joint angles in RADIANS
            pose_feasibility: Dict with feasibility info
            
        Returns:
            Overall singularity state
        """
        singularity_info = self._detect_singularity(angles)
        has_singularity = singularity_info.get("has_singularity", False)
        
        # Store for later retrieval
        self._last_singularity_info = singularity_info
        
        # Determine return state
        if singularity_info.get("confidence") == "confident_singular":
            return_state = SingularityState.CONFIDENT_SINGULAR
        elif has_singularity:
            return_state = SingularityState.SINGULAR
        else:
            return_state = SingularityState.NON_SINGULAR

        # Adjust IK margin based on singularity state
        adjust_for_singularity(return_state)
        
        return return_state
    
    def get_singularity_info(self) -> Dict:
        """Get the last computed singularity information."""
        return getattr(self, "_last_singularity_info", {
            "has_singularity": False,
            "singularity_type": None,
            "confidence": None,
            "warnings": [],
        })
    

def adjust_for_singularity(singularity_state: SingularityState):
    """
    Adjust IK branch switch margin based on singularity state.
    
    When in any singular state (SINGULAR or CONFIDENT_SINGULAR),
    uses a tighter margin for more precise branch selection.
    When NON_SINGULAR, uses default margin to reduce jitter.
    
    Args:
        singularity_state: Current singularity state
    """
    kinematics_config = get_kinematics_config()
    if singularity_state != SingularityState.NON_SINGULAR:
        # Use lower margin when singular to prioritize precision
        kinematics_config.set_ik_branch_switch_margin(
            constants.IK_BRANCH_SWITCH_MARGIN_MM_SINGULARITY
        )
        #print(f"Setting IK branch switch margin to {constants.IK_BRANCH_SWITCH_MARGIN_MM_SINGULARITY} mm due to singularity")
    else:
        # Use default margin when not singular
        kinematics_config.set_ik_branch_switch_margin(
            constants.IK_BRANCH_SWITCH_MARGIN_MM
        )
        #print(f"Setting IK branch switch margin to {constants.IK_BRANCH_SWITCH_MARGIN_MM} mm due to non-singularity")
