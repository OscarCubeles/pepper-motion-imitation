"""
Feasibility FSM - validates robot pose feasibility using state machines.

Uses ElbowRoll, ShoulderRoll, and Wrist state machines to detect when Pepper's 
configuration becomes infeasible or unfeasible.
"""
from enum import Enum
import numpy as np
from numpy import isnan
from typing import Dict, Optional


class FeasibilityState(Enum):
    """Feasibility detection states."""
    OK = "ok"
    TRANSITION = "transition"
    UNFEASIBLE = "unfeasible"


class ElbowRollFeasibilityStateMachine:
    """
    3-state FSM for ElbowRoll feasibility detection.
    
    States:
        OK: |elbow_roll| <= 90° - fully feasible zone
        TRANSITION: 90° < |elbow_roll| <= 100° - warning zone
        UNFEASIBLE: |elbow_roll| > 100° - infeasible zone
    
    Hysteresis prevents oscillation between states.
    No direct transition from OK to UNFEASIBLE.
    """
    
    def __init__(self):
        """Initialize ElbowRoll Feasibility FSM."""
        self.state = FeasibilityState.OK
        self.transition_threshold = 95.0  # degrees (absolute value) - OK to TRANSITION
        self.unfeasible_threshold = 100.0  # degrees (absolute value) - TRANSITION to UNFEASIBLE
        self.recovery_threshold = 90.0  # degrees (absolute value) - UNFEASIBLE/TRANSITION back to OK
    
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
    
    def update(self, elbow_roll_rad: Optional[float]) -> FeasibilityState:
        """
        Update FSM state based on ElbowRoll angle (absolute value).
        
        Args:
            elbow_roll_rad: ElbowRoll angle in radians (will be converted to absolute value)
            
        Returns:
            Current feasibility state
        """
        if elbow_roll_rad is None:
            return self.state
        
        elbow_roll_deg = abs(self._radians_to_degrees(elbow_roll_rad))
        
        if self.state == FeasibilityState.OK:
            if elbow_roll_deg > self.transition_threshold:
                self.state = FeasibilityState.TRANSITION
        
        elif self.state == FeasibilityState.TRANSITION:
            if elbow_roll_deg > self.unfeasible_threshold:
                self.state = FeasibilityState.UNFEASIBLE
            elif elbow_roll_deg <= self.recovery_threshold:
                self.state = FeasibilityState.OK
        
        elif self.state == FeasibilityState.UNFEASIBLE:
            if elbow_roll_deg <= self.recovery_threshold:
                self.state = FeasibilityState.TRANSITION
        
        return self.state
    
    def get_state_name(self) -> str:
        """Get human-readable state name."""
        return self.state.value
    
    def is_feasible(self) -> bool:
        """Check if currently feasible."""
        return self.state == FeasibilityState.OK
    
    def is_unfeasible(self) -> bool:
        """Check if currently unfeasible."""
        return self.state == FeasibilityState.UNFEASIBLE


class ShoulderRollFeasibilityStateMachine:
    """
    3-state FSM for ShoulderRoll feasibility detection (arm-specific).
    
    States:
        OK: fully feasible zone
        TRANSITION: warning zone (entering unfeasible region)
        UNFEASIBLE: infeasible zone
    
    LEFT ARM (positive angles):
        OK: angle > 5°
        TRANSITION: 0.5° < angle <= 1.5°
        UNFEASIBLE: angle <= 0.5°
    
    RIGHT ARM (negative angles):
        OK: angle < -5°
        TRANSITION: -1.5° <= angle < -0.5°
        UNFEASIBLE: angle >= -0.5°
    
    Hysteresis prevents oscillation between states.
    No direct transition from OK to UNFEASIBLE.
    """
    
    def __init__(self, arm: str = "right"):
        """
        Initialize ShoulderRoll Feasibility FSM.
        
        Args:
            arm: "right" or "left"
        """
        self.arm = arm.lower()
        if self.arm not in ("right", "left"):
            raise ValueError("arm must be 'right' or 'left'")
        
        self.state = FeasibilityState.OK
        
        if self.arm == "left":
            # LEFT ARM: positive angles
            self.ok_threshold = 5.0  # angle > 5°
            self.transition_threshold = 1.5  # OK to TRANSITION entry at 1.5°
            self.unfeasible_threshold = 0.5  # TRANSITION to UNFEASIBLE entry at 0.5°
            self.recovery_threshold = 5.0  # recovery back to OK at 5°
        else:
            # RIGHT ARM: negative angles (mirror of left)
            self.ok_threshold = -5.0  # angle < -5°
            self.transition_threshold = -1.5  # OK to TRANSITION entry at -1.5°
            self.unfeasible_threshold = -0.5  # TRANSITION to UNFEASIBLE entry at -0.5°
            self.recovery_threshold = -5.0  # recovery back to OK at -5°
    
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
    
    def update(self, shoulder_roll_rad: Optional[float]) -> FeasibilityState:
        """
        Update FSM state based on ShoulderRoll angle.
        
        Args:
            shoulder_roll_rad: ShoulderRoll angle in radians
            
        Returns:
            Current feasibility state
        """
        if shoulder_roll_rad is None:
            return self.state
        
        shoulder_roll_deg = self._radians_to_degrees(shoulder_roll_rad)
        
        if self.arm == "left":
            # LEFT ARM: positive angles, decreasing towards zero is bad
            if self.state == FeasibilityState.OK:
                if shoulder_roll_deg <= self.transition_threshold:
                    self.state = FeasibilityState.TRANSITION
            
            elif self.state == FeasibilityState.TRANSITION:
                if shoulder_roll_deg <= self.unfeasible_threshold:
                    self.state = FeasibilityState.UNFEASIBLE
                elif shoulder_roll_deg > self.recovery_threshold:
                    self.state = FeasibilityState.OK
            
            elif self.state == FeasibilityState.UNFEASIBLE:
                if shoulder_roll_deg > self.recovery_threshold:
                    self.state = FeasibilityState.TRANSITION
        
        else:
            # RIGHT ARM: negative angles, increasing towards zero is bad
            if self.state == FeasibilityState.OK:
                if shoulder_roll_deg >= self.transition_threshold:
                    self.state = FeasibilityState.TRANSITION
            
            elif self.state == FeasibilityState.TRANSITION:
                if shoulder_roll_deg >= self.unfeasible_threshold:
                    self.state = FeasibilityState.UNFEASIBLE
                elif shoulder_roll_deg < self.recovery_threshold:
                    self.state = FeasibilityState.OK
            
            elif self.state == FeasibilityState.UNFEASIBLE:
                if shoulder_roll_deg < self.recovery_threshold:
                    self.state = FeasibilityState.TRANSITION
        
        return self.state
    
    def get_state_name(self) -> str:
        """Get human-readable state name."""
        return self.state.value
    
    def is_feasible(self) -> bool:
        """Check if currently feasible."""
        return self.state == FeasibilityState.OK
    
    def is_unfeasible(self) -> bool:
        """Check if currently unfeasible."""
        return self.state == FeasibilityState.UNFEASIBLE


class WristFeasibilityStateMachine:
    """
    3-state FSM for Wrist feasibility detection (frame-count based).
    
    States:
        OK: No wrist violations
        TRANSITION: 1 frame of violation (entering unfeasible)
        UNFEASIBLE: 2+ frames of violations
    
    Transitions use frame counting instead of angle thresholds:
        OK -> TRANSITION: 1 frame with violation
        TRANSITION -> UNFEASIBLE: 2 consecutive frames with violations
        UNFEASIBLE -> TRANSITION: 2 consecutive frames without violations
        TRANSITION -> OK: 1 frame without violation
    """
    
    def __init__(self, hand: str = "right"):
        """
        Initialize Wrist Feasibility FSM.
        
        Args:
            hand: "right" or "left"
        """
        self.hand = hand.lower()
        if self.hand not in ("right", "left"):
            raise ValueError("hand must be 'right' or 'left'")
        
        self.state = FeasibilityState.OK
        self.violation_frame_count = 0  # Counts consecutive frames with violations
        self.clear_frame_count = 0  # Counts consecutive frames without violations
    
    def update(self, has_violation: bool) -> FeasibilityState:
        """
        Update FSM state based on presence/absence of wrist violation.
        
        Args:
            has_violation: True if wrist discrete violation detected in this frame
            
        Returns:
            Current feasibility state
        """
        if has_violation:
            # Increment violation counter, reset clear counter
            self.violation_frame_count += 1
            self.clear_frame_count = 0
            
            if self.state == FeasibilityState.OK:
                # OK -> TRANSITION after 1 frame with violation
                if self.violation_frame_count >= 1:
                    self.state = FeasibilityState.TRANSITION
            
            elif self.state == FeasibilityState.TRANSITION:
                # TRANSITION -> UNFEASIBLE after 2 frames with violations
                if self.violation_frame_count >= 2:
                    self.state = FeasibilityState.UNFEASIBLE
            
            # UNFEASIBLE stays in UNFEASIBLE while violations continue
        
        else:
            # No violation: increment clear counter, reset violation counter
            self.clear_frame_count += 1
            self.violation_frame_count = 0
            
            if self.state == FeasibilityState.UNFEASIBLE:
                # UNFEASIBLE -> TRANSITION after 2 frames without violations
                if self.clear_frame_count >= 2:
                    self.state = FeasibilityState.TRANSITION
            
            elif self.state == FeasibilityState.TRANSITION:
                # TRANSITION -> OK after 1 frame without violation
                if self.clear_frame_count >= 1:
                    self.state = FeasibilityState.OK
            
            # OK stays in OK while no violations
        
        return self.state
    
    def get_state_name(self) -> str:
        """Get human-readable state name."""
        return self.state.value
    
    def is_feasible(self) -> bool:
        """Check if currently feasible."""
        return self.state == FeasibilityState.OK
    
    def is_unfeasible(self) -> bool:
        """Check if currently unfeasible."""
        return self.state == FeasibilityState.UNFEASIBLE


class PepperFeasibilityFSM:
    """
    FSM to track feasibility for Pepper poses.
    
    Uses sub-FSMs: ElbowRollFeasibilityStateMachine, ShoulderRollFeasibilityStateMachine, 
    and WristFeasibilityStateMachine (to be implemented).
    
    Detects when arm configuration becomes infeasible.
    """
    
    def __init__(self, arm: str = "right"):
        """
        Initialize feasibility FSM for one arm.
        
        Args:
            arm: "right" or "left"
        """
        self.arm = arm.lower()
        if self.arm not in ("right", "left"):
            raise ValueError("arm must be 'right' or 'left'")
        
        self.elbow_fsm = ElbowRollFeasibilityStateMachine()
        self.shoulder_fsm = ShoulderRollFeasibilityStateMachine(arm=self.arm)
        self.wrist_fsm = WristFeasibilityStateMachine(hand=self.arm)
    
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
    
    def _detect_feasibility(self, angles: Dict[str, float]) -> Dict:
        """
        Detect feasibility issues using ElbowRoll and ShoulderRoll state machines.
        
        Args:
            angles: Dictionary with joint angles in RADIANS
            
        Returns:
            Dictionary with feasibility information:
            {
                "is_feasible": bool,
                "violations": [list of violation strings],
                "details": {...}
            }
        """
        violations = []
        arm_side = self.arm.upper()
        
        # ===== ElbowRoll checking =====
        elbow_roll_key = f"ElbowRoll_{self.arm.capitalize()}"
        elbow_roll_rad = angles.get(elbow_roll_key)
        
        # Update state machine
        elbow_state = self.elbow_fsm.update(elbow_roll_rad)
        
        # Get value in degrees for reporting
        elbow_roll_deg = self._radians_to_degrees(elbow_roll_rad)
        abs_elbow_roll = abs(elbow_roll_deg) if elbow_roll_deg is not None else None
        
        # ElbowRoll feasibility violations
        if self.elbow_fsm.is_unfeasible():
            violations.append(
                f"ElbowRoll_{arm_side}: {abs_elbow_roll:.1f}° (UNFEASIBLE - must be < 100°)"
            )
        elif self.elbow_fsm.state == FeasibilityState.TRANSITION:
            violations.append(
                f"ElbowRoll_{arm_side}: {abs_elbow_roll:.1f}° (WARNING - approaching limit at 100°)"
            )
        
        # ===== ShoulderRoll checking =====
        shoulder_roll_key = f"ShoulderRoll_{self.arm.capitalize()}"
        shoulder_roll_rad = angles.get(shoulder_roll_key)
        
        # Update state machine
        shoulder_state = self.shoulder_fsm.update(shoulder_roll_rad)
        
        # Get value in degrees for reporting
        shoulder_roll_deg = self._radians_to_degrees(shoulder_roll_rad)
        
        if shoulder_roll_deg is not None:
            # ShoulderRoll feasibility violations
            if self.shoulder_fsm.is_unfeasible():
                violations.append(
                    f"ShoulderRoll_{arm_side}: {shoulder_roll_deg:.1f}° (UNFEASIBLE - "
                    f"left must be > 0.5°, right must be < -0.5°)"
                )
            elif self.shoulder_fsm.state == FeasibilityState.TRANSITION:
                violations.append(
                    f"ShoulderRoll_{arm_side}: {shoulder_roll_deg:.1f}° (WARNING - "
                    f"left approaching 0°, right approaching 0°)"
                )
        
        # ===== Wrist checking (frame-count based) =====
        # Note: Wrist violations are detected separately in AngleClassifier
        # and passed via update_wrist() method. This section reports wrist state.
        if self.wrist_fsm.is_unfeasible():
            violations.append(
                f"WristDiscrete: UNFEASIBLE (2+ consecutive frames with violations)"
            )
        elif self.wrist_fsm.state == FeasibilityState.TRANSITION:
            violations.append(
                f"WristDiscrete: WARNING (1+ frame with violations)"
            )
        
        return {
            "is_feasible": len(violations) == 0,
            "violations": violations,
            "details": {
                "elbow_roll": elbow_roll_rad,
                "elbow_state": self.elbow_fsm.get_state_name(),
                "shoulder_roll": shoulder_roll_rad,
                "shoulder_state": self.shoulder_fsm.get_state_name(),
                "wrist_state": self.wrist_fsm.get_state_name(),
            },
        }
    
    def update_wrist(self, has_violation: bool) -> FeasibilityState:
        """
        Update wrist FSM state based on presence of wrist violations.
        
        Args:
            has_violation: True if wrist discrete violation detected in this frame
            
        Returns:
            Current wrist feasibility state
        """
        return self.wrist_fsm.update(has_violation)
    
    def update(self, angles: Dict[str, float]) -> Dict:
        """
        Update feasibility detection for this arm.
        
        Args:
            angles: Dictionary with joint angles in RADIANS
            
        Returns:
            Feasibility information dictionary
        """
        feasibility_info = self._detect_feasibility(angles)
        
        # Store for later retrieval
        self._last_feasibility_info = feasibility_info
        
        return feasibility_info
    
    def get_feasibility_info(self) -> Dict:
        """Get the last computed feasibility information."""
        return getattr(self, "_last_feasibility_info", {
            "is_feasible": True,
            "violations": [],
            "details": {},
        })