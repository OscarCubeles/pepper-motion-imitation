"""
Angle classifier - validates and classifies computed joint angles.

Takes pre-computed joint angles and performs validation/classification:
- Feasibility checking (workspace constraints, joint limits)
- Singularity detection
- Discrete wrist orientation validation
"""
import settings
import numpy as np
from numpy import isnan
from typing import Dict, Optional, List
from singularity_fsm import SingularityFSM
from pepper_feasible_fsm import PepperFeasibilityFSM
import numpy as np



class AngleClassifier:
    """
    Validates and classifies pre-computed joint angles.
    
    Checks feasibility, detects singularities, and validates wrist orientation.
    Does NOT compute angles - use AngleCalculator for that.
    """
    
    def __init__(self):
        """Initialize classifier."""
        # Create feasibility FSMs for both arms (includes wrist FSM)
        self._fsm_right = PepperFeasibilityFSM(arm="right")
        self._fsm_left = PepperFeasibilityFSM(arm="left")
    
    @staticmethod
    def _radians_to_degrees(value: Optional[float]) -> Optional[float]:
        """Safely convert angle value from radians to degrees, handling None and NaN."""
        if value is None:
            return None
        try:
            if isnan(value):
                return None
        except (TypeError, ValueError):
            pass
        return np.degrees(float(value))
    
    # ============================================================================
    # FEASIBILITY CHECKING
    # ============================================================================
    
    def check_feasibility(
        self,
        angles: Dict[str, float],
        forearm_direction_right: Optional[str] = None,
        forearm_direction_left: Optional[str] = None,
        orientation_labels: Optional[Dict[str, Dict[str, str]]] = None,
    ) -> Dict:
        """
        Check if computed angles are feasible for Pepper.
        
        Args:
            angles: Dictionary with joint angles in RADIANS
            forearm_direction_right: Optional forearm direction label
            forearm_direction_left: Optional forearm direction label
            orientation_labels: Optional hand orientation labels
            
        Returns:
            Dictionary with feasibility status and violations (angle violations shown in degrees)
        """
        violations = []
        
        # Convert critical angles from radians to degrees for constraint checking
        left_roll_deg = self._radians_to_degrees(angles.get("ShoulderRoll_Left"))
        if left_roll_deg is not None and left_roll_deg <= 1.5:
            violations.append(f"ShoulderRoll_Left: {left_roll_deg:.1f}° (must be > 1.5°)")
        
        right_roll_deg = self._radians_to_degrees(angles.get("ShoulderRoll_Right"))
        if right_roll_deg is not None and right_roll_deg >= -1.5:
            violations.append(f"ShoulderRoll_Right: {right_roll_deg:.1f}° (must be < -1.5°)")
        
        elbow_roll_right_deg = self._radians_to_degrees(angles.get("ElbowRoll_Right"))
        if elbow_roll_right_deg is not None and elbow_roll_right_deg >= 88:
            violations.append(f"ElbowRoll_Right: {elbow_roll_right_deg:.1f}° (must be < 88°)")
        
        elbow_roll_left_deg = self._radians_to_degrees(angles.get("ElbowRoll_Left"))
        if elbow_roll_left_deg is not None and elbow_roll_left_deg <= -88:
            violations.append(f"ElbowRoll_Left: {elbow_roll_left_deg:.1f}° (must be > -88°)")
        
        wrist_feasibility = self.check_wrist_feasibility(
            forearm_direction_right,
            forearm_direction_left,
            orientation_labels,
        )
        # Add wrist violations (already separated by hand)
        wrist_violations = wrist_feasibility.get("violations", {})
        violations.extend(wrist_violations.get("left", []))
        violations.extend(wrist_violations.get("right", []))
        
        return {
            "is_feasible": len(violations) == 0,
            "violations": violations,
        }
    
    def check_feasibility_fsm(
        self,
        angles: Dict[str, float],
        forearm_direction_right: Optional[str] = None,
        forearm_direction_left: Optional[str] = None,
        orientation_labels: Optional[Dict[str, Dict[str, str]]] = None,
    ) -> Dict:
        """
        Check if computed angles are feasible for Pepper using FSM-based checks.
        
        Uses ElbowRollFeasibilityStateMachine and ShoulderRollFeasibilityStateMachine 
        for arm joint checking. Other checks (wrist) use the same logic as check_feasibility.
        
        Args:
            angles: Dictionary with joint angles in RADIANS
            forearm_direction_right: Optional forearm direction label
            forearm_direction_left: Optional forearm direction label
            orientation_labels: Optional hand orientation labels
            
        Returns:
            Dictionary with feasibility status and violations dict with "left" and "right" keys
        """
        violations = {"left": [], "right": []}
        
        # Check wrist feasibility first and update wrist FSM states
        wrist_feasibility = self.check_wrist_feasibility(
            forearm_direction_right,
            forearm_direction_left,
            orientation_labels,
        )
        
        # Update wrist FSM states based on violations (before arm FSM updates)
        # Right arm wrist: has violation if violations list is not empty
        right_wrist_has_violation = len(wrist_feasibility.get("violations", {}).get("right", [])) > 0
        self._fsm_right.update_wrist(right_wrist_has_violation)
        
        # Left arm wrist: has violation if violations list is not empty
        left_wrist_has_violation = len(wrist_feasibility.get("violations", {}).get("left", [])) > 0
        self._fsm_left.update_wrist(left_wrist_has_violation)
        
        # ElbowRoll and ShoulderRoll checking using FSM for both arms
        # (now includes wrist FSM state violations)
        elbow_roll_right_rad = angles.get("ElbowRoll_Right")
        elbow_roll_left_rad = angles.get("ElbowRoll_Left")
        
        # Check right arm using FSM (both elbow, shoulder, and wrist)
        if elbow_roll_right_rad is not None:
            feasibility_right = self._fsm_right.update(angles)
            violations["right"].extend(feasibility_right.get("violations", []))
        
        # Check left arm using FSM (both elbow, shoulder, and wrist)
        if elbow_roll_left_rad is not None:
            feasibility_left = self._fsm_left.update(angles)
            violations["left"].extend(feasibility_left.get("violations", []))
        
        #print(f"FSM-based feasibility check - Right arm violations: {violations['right']}")
        #print(f"FSM-based feasibility check - Left arm violations: {violations['left']}")

        is_feasible = len(violations["left"]) == 0 and len(violations["right"]) == 0
        return {
            "is_feasible": is_feasible,
            "violations": violations,
        }
    


    def _normalize_discrete_direction_label(self, label: Optional[str]) -> Optional[str]:
        """Normalize direction labels to WRIST_AXIS_MAP vocabulary."""
        if label is None:
            return None
        label_norm = str(label).strip().lower()
        alias = {
            "front": "forward",
            "back": "backward",
        }
        label_norm = alias.get(label_norm, label_norm)
        if label_norm in settings.WRIST_AXIS_MAP:
            return label_norm
        return None

    def check_wrist_feasibility(
        self,
        forearm_direction_right: Optional[str],
        forearm_direction_left: Optional[str],
        orientation_labels: Optional[Dict[str, Dict[str, str]]],
    ) -> Dict:
        """Check discrete wrist feasibility for both hands from orientation labels.
        
        Returns:
            Dictionary with violations organized by hand:
            {
                "violations": {
                    "left": [list of violation strings],
                    "right": [list of violation strings]
                }
            }
        """

        def _axis_of(label: Optional[str]) -> Optional[str]:
            normalized = self._normalize_discrete_direction_label(label)
            if normalized in ("left", "right"):
                return "x"
            if normalized in ("up", "down"):
                return "y"
            if normalized in ("forward", "backward"):
                return "z"
            return None

        def _extract_palm_label(hand_data: Optional[Dict[str, str]]) -> Optional[str]:
            if not isinstance(hand_data, dict):
                return None
            for key in ("primary", "palm", "x_axis", "y_axis", "z_axis"):
                candidate = hand_data.get(key)
                normalized = self._normalize_discrete_direction_label(candidate)
                if normalized is not None:
                    return normalized
            return None

        orientation_labels = orientation_labels or {}
        violations = {"left": [], "right": []}

        hand_specs = (
            ("right", forearm_direction_right),
            ("left", forearm_direction_left),
        )

        for hand_side, forearm_raw in hand_specs:
            forearm_label = self._normalize_discrete_direction_label(forearm_raw)
            palm_label = _extract_palm_label(orientation_labels.get(hand_side.capitalize()))

            if forearm_label is None or palm_label is None:
                continue

            forearm_axis = _axis_of(forearm_label)
            palm_axis = _axis_of(palm_label)

            if forearm_axis is None or palm_axis is None:
                continue

            if forearm_axis == palm_axis:
                violation_text = (
                    f"WristDiscrete: forearm={forearm_label}, palm={palm_label} "
                    f"(same axis -> not feasible)"
                )
                violations[hand_side].append(violation_text)

        return {
            "violations": violations,
        }
    
    # ============================================================================
    # SINGULARITY CHECKING
    # ============================================================================
    
    def check_singularity_poses(
        self,
        angles: Dict[str, float],
        arm: str = 'right',
        shoulder_roll_threshold: float = 10.0,
        elbow_roll_threshold: float = 10.0,
    ) -> Dict:
        """Check if the arm configuration is near a singularity pose.
        
        Args:
            angles: Dictionary with joint angles in RADIANS
            arm: Arm to check ('right' or 'left')
            shoulder_roll_threshold: Threshold in degrees
            elbow_roll_threshold: Threshold in degrees
            
        Returns:
            Dictionary with singularity information
        """
        warnings = []

        shoulder_roll_key = f"ShoulderRoll_{arm.capitalize()}"
        elbow_roll_key = f"ElbowRoll_{arm.capitalize()}"

        shoulder_roll_rad = angles.get(shoulder_roll_key)
        elbow_roll_rad = angles.get(elbow_roll_key)
        
        # Convert from radians to degrees for singularity checking
        shoulder_roll_deg = self._radians_to_degrees(shoulder_roll_rad)
        elbow_roll_deg = self._radians_to_degrees(elbow_roll_rad)

        singularity_type = None
        near_shoulder_roll_singularity = False
        near_elbow_roll_singularity = False
        arm_side = arm.upper()

        if shoulder_roll_deg is not None:
            dist_from_90 = abs(abs(shoulder_roll_deg) - 90.0)
            if dist_from_90 < 5.0:
                warnings.append(f"{arm_side}: ShoulderRoll at {shoulder_roll_deg:.1f}° (SINGULARITY at ≈90°)")
                near_shoulder_roll_singularity = True
            elif dist_from_90 < shoulder_roll_threshold:
                warnings.append(
                    f"{arm_side}: ShoulderRoll at {shoulder_roll_deg:.1f}° (near singularity, within {shoulder_roll_threshold}°)"
                )
                near_shoulder_roll_singularity = True

        if elbow_roll_deg is not None:
            dist_from_0 = abs(elbow_roll_deg)
            if dist_from_0 < 5.0:
                warnings.append(f"{arm_side}: ElbowRoll at {elbow_roll_deg:.1f}° (SINGULARITY at ≈0°)")
                near_elbow_roll_singularity = True
            elif dist_from_0 < elbow_roll_threshold:
                warnings.append(
                    f"{arm_side}: ElbowRoll at {elbow_roll_deg:.1f}° (near singularity, within {elbow_roll_threshold}°)"
                )
                near_elbow_roll_singularity = True

        if near_shoulder_roll_singularity and near_elbow_roll_singularity:
            singularity_type = "dual"
            warnings.insert(0, f"{arm_side}: DUAL SINGULARITY - both ShoulderRoll and ElbowRoll are critical")
        elif near_shoulder_roll_singularity:
            singularity_type = "shoulder_roll"
        elif near_elbow_roll_singularity:
            singularity_type = "elbow_roll"

        return {
            "has_singularity": singularity_type is not None,
            "singularity_type": singularity_type,
            "warnings": warnings,
            "details": {
                "shoulder_roll": shoulder_roll_rad,
                "elbow_roll": elbow_roll_rad
                
                ,
                "near_shoulder_roll_singularity": near_shoulder_roll_singularity,
                "near_elbow_roll_singularity": near_elbow_roll_singularity,
            },
        }
    
    def check_singularity_poses_fsm(
        self,
        angles: Dict[str, float],
        arm: str = 'right'
    ) -> Dict:
        """
        Check if the arm configuration is near a singularity pose using FSM.
        
        Uses ElbowRollStateMachine and ShoulderRollStateMachine for state-based 
        singularity detection with hysteresis and confidence levels.
        
        Args:
            angles: Dictionary with joint angles in RADIANS
            arm: Arm to check ('right' or 'left')
            shoulder_roll_threshold: Threshold in degrees (unused, kept for compatibility)
            elbow_roll_threshold: Threshold in degrees (unused, kept for compatibility)
            
        Returns:
            Dictionary with singularity information compatible with check_singularity_poses()
        """
        # Create FSM for this arm
        fsm = SingularityFSM(arm=arm)
        
        # Get singularity info from FSM
        singularity_info = fsm._detect_singularity(angles)
        #print(f"FSM-based singularity detection for {arm} arm: {singularity_info}")
        # Convert FSM output to original format for compatibility
        warnings = singularity_info.get("warnings", [])
        singularity_type = singularity_info.get("singularity_type")
        confidence = singularity_info.get("confidence")
        
        # Add confidence level to singularity type if present
        if singularity_type and confidence:
            singularity_type_with_confidence = f"{singularity_type} ({confidence})"
        else:
            singularity_type_with_confidence = singularity_type
        
        return {
            "has_singularity": singularity_info.get("has_singularity", False),
            "singularity_type": singularity_type_with_confidence,
            "warnings": warnings,
            "details": singularity_info.get("details", {}),
            "confidence": confidence,
        }



