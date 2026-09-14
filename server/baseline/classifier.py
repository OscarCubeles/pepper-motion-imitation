"""
Pose classifier for rule-based classification using keypoints.
Computes geometric vectors and features from 3D keypoints.
"""

import numpy as np
from typing import Dict, Optional
from settings import KINECTV2_25_JOINTS

# Metrabs bounding box is 2200mm (±1100mm from center)
METRABS_HALF_BOX_MM = 1100.0

class PoseVectorClassifier:
    """
    Receives 3D keypoints and computes vectors/features for rule-based classification.
    
    Input: keypoints of shape (J, 3) where J is number of joints
           and 3 are [x, y, z] coordinates
    """
    
    def __init__(self, joint_names=None):
        """
        Initialize classifier.
        
        Args:
            joint_names: List of joint name strings (e.g., from Metrabs model)
        """
        self.keypoints = None
        self.joint_names = joint_names or []
        self.vectors = {}
        
    def set_keypoints(self, keypoints_3d: np.ndarray, normalize: bool = False) -> None:
        """
        Set the keypoints to analyze.
        
        Args:
            keypoints_3d: Array of shape (J, 3) with [x, y, z] coordinates (in mm from Metrabs)
            normalize: If True, normalize to [-1, 1] range based on Metrabs bounding box
        """
        if keypoints_3d.shape[1] != 3:
            raise ValueError(f"Expected shape (J, 3), got {keypoints_3d.shape}")
        
        keypoints_3d = keypoints_3d.astype(np.float32)
        
        # Normalize keypoints from Metrabs mm coordinates to [-1, 1] range
        # This improves numerical stability for angle computations
        if normalize:
            keypoints_3d = keypoints_3d / METRABS_HALF_BOX_MM
        
        self.keypoints = keypoints_3d
        self.vectors = {}
    
    def _normalize_vector(self, vector: np.ndarray) -> Optional[np.ndarray]:
        """Normalize a vector to unit length. Returns None if degenerate."""
        magnitude = np.linalg.norm(vector)
        if magnitude < 1e-8:
            return None
        return vector / magnitude
    
    def _compute_angle(self, vec1: np.ndarray, vec2: np.ndarray) -> Optional[float]:
        """Compute angle between two vectors in degrees."""
        normalized_v1 = self._normalize_vector(vec1)
        normalized_v2 = self._normalize_vector(vec2)
        
        if normalized_v1 is None or normalized_v2 is None:
            return None
        
        dot_product = np.clip(np.dot(normalized_v1, normalized_v2), -1.0, 1.0)
        angle_rad = np.arccos(dot_product)
        angle_deg = np.degrees(angle_rad)
        return angle_deg
    
    def _build_torso_frame(self) -> Optional[Dict[str, np.ndarray]]:
        """
        Build normalized torso reference frame from keypoints.
        
        Returns:
            Dict with normalized frame vectors:
            - torso_up: normalized vector from SpineChest to Neck
            - torso_right: normalized vector from ShoulderLeft to ShoulderRight
            - torso_forward: normalized cross product (torso_up × torso_right)
            
            Returns None if frame is degenerate
        """
        if self.keypoints is None:
            raise RuntimeError("Keypoints not set. Call set_keypoints() first.")
        
        try:
            neck_idx = KINECTV2_25_JOINTS["Neck"]
            spine_idx = KINECTV2_25_JOINTS["SpineChest"]
            shoulder_left_idx = KINECTV2_25_JOINTS["ShoulderLeft"]
            shoulder_right_idx = KINECTV2_25_JOINTS["ShoulderRight"]
            
            # Build frame vectors
            torso_up = self.keypoints[neck_idx] - self.keypoints[spine_idx]
            torso_right = self.keypoints[shoulder_right_idx] - self.keypoints[shoulder_left_idx]
            torso_forward = np.cross(torso_up, torso_right)
            
            # Normalize
            torso_up_norm = self._normalize_vector(torso_up)
            torso_right_norm = self._normalize_vector(torso_right)
            torso_forward_norm = self._normalize_vector(torso_forward)
            
            if any(v is None for v in [torso_up_norm, torso_right_norm, torso_forward_norm]):
                return None
            
            return {
                "torso_up": torso_up_norm,
                "torso_right": torso_right_norm,
                "torso_forward": torso_forward_norm,
            }
        except KeyError as e:
            print(f"Warning: Missing joint index in KINECTV2_25_JOINTS: {e}")
            return None
    
    def _compute_elbow_roll(self, side: str) -> Optional[float]:
        """
        Compute ElbowRoll angle for a specific arm (elbow flexion/bend).
        
        Measures how much the forearm is bent relative to the upper arm,
        independent of pronation (arm twisting).
        
        - 0° = arm fully folded (biceps touches forearm)
        - 90° = arm in L-shape 
        - 180° = arm fully straight/extended
        
        Method:
        1. Project lower arm onto plane perpendicular to upper arm (removes twist)
        2. Compute angle from the "straight extension" direction
        
        Args:
            side: "Left" or "Right"
            
        Returns:
            Angle in degrees or None if computation fails
        """
        if self.keypoints is None:
            raise RuntimeError("Keypoints not set. Call set_keypoints() first.")
        
        try:
            shoulder_idx = KINECTV2_25_JOINTS[f"Shoulder{side}"]
            elbow_idx = KINECTV2_25_JOINTS[f"Elbow{side}"]
            wrist_idx = KINECTV2_25_JOINTS[f"Wrist{side}"]
            
            upper_arm = self.keypoints[elbow_idx] - self.keypoints[shoulder_idx]
            lower_arm = self.keypoints[wrist_idx] - self.keypoints[elbow_idx]
            
            # Normalize upper arm to define the flexion axis
            upper_arm_norm = self._normalize_vector(upper_arm)
            if upper_arm_norm is None:
                return None
            
            # Project lower arm onto plane perpendicular to upper arm
            # This removes the effect of forearm rotation (pronation/supination)
            projection_along_axis = np.dot(lower_arm, upper_arm_norm)
            lower_arm_perpendicular = lower_arm - projection_along_axis * upper_arm_norm
            
            # Compute the flexion angle using atan2
            # projection_along_axis: positive = arm extended, negative = arm folded
            # lower_arm_perpendicular magnitude: 0 = straight line, large = bent
            #
            # Map to [0, 180]:
            # - Straight arm (proj=+, perp≈0): angle = 180°
            # - 90° bend (proj≈0, perp=+): angle = 90°
            # - Folded arm (proj=-, perp≈0): angle = 0°
            raw_angle = np.degrees(np.arctan2(
                np.linalg.norm(lower_arm_perpendicular),
                projection_along_axis
            ))
            
            # atan2 returns [-180, 180], convert to [0, 180] flexion angle
            # atan2(perp, proj) where proj=extended_arm gives 0° for straight
            # We want 180° for straight, so: elbow_roll = 180 - raw_angle (with adjustments)
            elbow_roll = 180.0 - raw_angle
            if elbow_roll < 0:
                elbow_roll += 360.0
            
            return elbow_roll
        except KeyError as e:
            print(f"Warning: Missing joint for ElbowRoll_{side}: {e}")
            return None
    
    def _compute_head_pitch(self, torso_frame: Dict[str, np.ndarray]) -> Optional[float]:
        """
        Compute HeadPitch angle.
        
        HeadPitch = angle(head_dir, torso_up)
        
        Meaning:
        - Head tilt forward/backward relative to torso
        
        Args:
            torso_frame: Dict with normalized torso frame vectors
            
        Returns:
            Angle in degrees or None if computation fails
        """
        if self.keypoints is None:
            raise RuntimeError("Keypoints not set. Call set_keypoints() first.")
        
        try:
            neck_idx = KINECTV2_25_JOINTS["Neck"]
            spine_idx = KINECTV2_25_JOINTS["SpineChest"]
            head_idx = KINECTV2_25_JOINTS["Head"]
            
            head_dir = self.keypoints[head_idx] - self.keypoints[neck_idx]
            torso_up_vec = self.keypoints[neck_idx] - self.keypoints[spine_idx]
            
            return self._compute_angle(head_dir, torso_up_vec)
        except KeyError as e:
            print(f"Warning: Missing joint for HeadPitch: {e}")
            return None
    
    def _compute_shoulder_roll(self, side: str, torso_frame: Dict[str, np.ndarray]) -> Optional[float]:
        """
        Compute ShoulderRoll angle for a specific arm.
        
        ShoulderRoll = rotation around the forward axis (lifts arm to the side)
        - 0° = arm pointing down/up
        - 90° = arm pointing to the side
        - -90° = arm pointing to the opposite side
        
        Computed by projecting arm onto UP-RIGHT plane (removing forward component).
        
        Args:
            side: "Left" or "Right"
            torso_frame: Dict with normalized torso frame vectors
            
        Returns:
            Angle in degrees or None if computation fails
        """
        if self.keypoints is None:
            raise RuntimeError("Keypoints not set. Call set_keypoints() first.")
        
        try:
            shoulder_idx = KINECTV2_25_JOINTS[f"Shoulder{side}"]
            elbow_idx = KINECTV2_25_JOINTS[f"Elbow{side}"]
            
            upper_arm = self.keypoints[elbow_idx] - self.keypoints[shoulder_idx]
            u = self._normalize_vector(upper_arm)
            
            if u is None:
                return None
            
            # Project arm onto UP-RIGHT plane by removing FORWARD component
            forward_component = np.dot(u, torso_frame["torso_forward"])
            u_on_up_right = u - forward_component * torso_frame["torso_forward"]
            u_on_up_right = self._normalize_vector(u_on_up_right)
            
            if u_on_up_right is None:
                return None
            
            # Compute angle in UP-RIGHT plane
            shoulder_roll = np.degrees(np.arctan2(
                np.dot(u_on_up_right, torso_frame["torso_right"]),
                np.dot(u_on_up_right, torso_frame["torso_up"])
            ))
            return shoulder_roll
        except KeyError as e:
            print(f"Warning: Missing joint for ShoulderRoll_{side}: {e}")
            return None
    
    def _compute_shoulder_pitch(self, side: str, torso_frame: Dict[str, np.ndarray]) -> Optional[float]:
        """
        Compute ShoulderPitch angle for a specific arm.
        
        ShoulderPitch = rotation around the right axis (raises arm forward/backward)
        - 0° = arm pointing down
        - 90° = arm pointing forward
        - -90° = arm pointing backward
        
        Computed by projecting arm onto UP-FORWARD plane (removing side component).
        
        Args:
            side: "Left" or "Right"
            torso_frame: Dict with normalized torso frame vectors
            
        Returns:
            Angle in degrees or None if computation fails
        """
        if self.keypoints is None:
            raise RuntimeError("Keypoints not set. Call set_keypoints() first.")
        
        try:
            shoulder_idx = KINECTV2_25_JOINTS[f"Shoulder{side}"]
            elbow_idx = KINECTV2_25_JOINTS[f"Elbow{side}"]
            
            upper_arm = self.keypoints[elbow_idx] - self.keypoints[shoulder_idx]
            u = self._normalize_vector(upper_arm)
            
            if u is None:
                return None
            
            # Project arm onto UP-FORWARD plane by removing RIGHT component
            right_component = np.dot(u, torso_frame["torso_right"])
            u_on_up_forward = u - right_component * torso_frame["torso_right"]
            u_on_up_forward = self._normalize_vector(u_on_up_forward)
            
            if u_on_up_forward is None:
                return None
            
            # Compute angle in UP-FORWARD plane
            shoulder_pitch = np.degrees(np.arctan2(
                np.dot(u_on_up_forward, torso_frame["torso_forward"]),
                np.dot(u_on_up_forward, torso_frame["torso_up"])
            ))
            return shoulder_pitch
        except KeyError as e:
            print(f"Warning: Missing joint for ShoulderPitch_{side}: {e}")
            return None
    
    def compute_joint_angles(self) -> Dict[str, float]:
        """
        Compute all joint angles for Pepper feasibility classification.
        
        Orchestrates independent angle computation methods:
        - ElbowRoll: bending angle at elbow (180° = straight, smaller = more bent)
        - ShoulderRoll: side lifting angle
        - ShoulderPitch: forward/back lifting angle
        - HeadPitch: head tilt relative to torso
        
        Returns:
            Dictionary with angle keys (e.g., "ElbowRoll_Left", "ShoulderRoll_Right") 
            and values in degrees
        """
        if self.keypoints is None:
            raise RuntimeError("Keypoints not set. Call set_keypoints() first.")
        
        angles = {}
        
        # Build torso frame
        torso_frame = self._build_torso_frame()
        if torso_frame is None:
            return angles
        
        # Head pitch
        head_pitch = self._compute_head_pitch(torso_frame)
        if head_pitch is not None:
            angles["HeadPitch"] = head_pitch
        
        # Process both arms
        for side in ["Left", "Right"]:
            elbow_roll = self._compute_elbow_roll(side)
            if elbow_roll is not None:
                angles[f"ElbowRoll_{side}"] = elbow_roll
            
            shoulder_roll = self._compute_shoulder_roll(side, torso_frame)
            if shoulder_roll is not None:
                angles[f"ShoulderRoll_{side}"] = shoulder_roll
            
            shoulder_pitch = self._compute_shoulder_pitch(side, torso_frame)
            if shoulder_pitch is not None:
                angles[f"ShoulderPitch_{side}"] = shoulder_pitch
        
        return angles
    
    def check_feasibility(self, angles: Dict[str, float]) -> Dict[str, bool]:
        """
        Check if a pose is feasible for Pepper based on computed angles.
        
        Pepper constraints:
        - ShoulderRoll_Left must be <= 0° (arm cannot go up to the right)
        - ShoulderRoll_Right must be >= 0° (arm cannot go up to the left)
        
        Args:
            angles: Dictionary of computed angles
            
        Returns:
            Dictionary with:
            - "is_feasible": bool, True if all constraints satisfied
            - "violations": list of constraint violations
        """
        violations = []
        
        # Check left shoulder roll (should not be positive)
        left_roll = angles.get("ShoulderRoll_Left")
        if left_roll is not None and left_roll > -25:
            violations.append(f"ShoulderRoll_Left: {left_roll:.1f}° (must be ≤ 0°)")
        
        # Check right shoulder roll (should not be negative)
        right_roll = angles.get("ShoulderRoll_Right")
        if right_roll is not None and right_roll < 25:
            violations.append(f"ShoulderRoll_Right: {right_roll:.1f}° (must be ≥ 0°)")
        
        return {
            "is_feasible": len(violations) == 0,
            "violations": violations,
        }
    