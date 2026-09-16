"""
Angle calculator - computes joint angles from pose keypoints or payload data.

This module handles all angle computation logic including:
- Coordinate transformations
- Forward/Inverse kinematics
- Workspace fitting
- Spherical coordinate conversions
"""
from server.common import settings
import numpy as np
import server.common.kinematics.forward_kinematics as fk
import server.common.kinematics.inverse_kinematics as ik
import server.common.kinematics.math_functions as mathik
import server.common.kinematics.transformation_matrices as tm
import server.common.kinematics.scaling_spherical as scaling
import server.common.kinematics.workspace_fitting as workspace
from server.common import pose_mapping as pose_map
from server.common import hand_orientation as ho
from typing import Dict, Optional, Tuple
from server.common.settings import KINECTV2_25_JOINTS
import base64
            
from numpy import pi, arctan, arctan2, arcsin, arccos, sqrt, sin, cos, transpose, radians, real, isnan, nan, linalg, array

class AngleCalculator:
    """
    Computes joint angles from human pose keypoints.
    
    Handles coordinate transformations, IK computation, and workspace fitting
    to calculate the exact t1, t2, t3, t4 angles that would be sent to Pepper.
    """
    
    def __init__(self):
        """Initialize the shared analytical calculator."""
        self.keypoints = None
    
    @staticmethod
    def _safe_angle_to_degrees(value):
        """Safely convert angle value to degrees, handling None and NaN."""
        if value is None:
            return None
        try:
            if isnan(value):
                return None
        except (TypeError, ValueError):
            pass
        return np.degrees(float(value))
    
    @staticmethod
    def _safe_angle_to_radians(value):
        """Safely keep angle value in radians, handling None and NaN.
        
        This method keeps the angle as-is (IK functions already return radians),
        but safely handles None and NaN values.
        """
        if value is None:
            return None
        try:
            if isnan(value):
                return None
        except (TypeError, ValueError):
            pass
        return float(value)
    
    @staticmethod
    def _safe_convert_degrees_to_radians(value):  
        """Safely convert angle value from degrees to radians, handling None and NaN.
        
        Converts an angle in degrees to radians, safely handling None and NaN values.
        """
        if value is None:
            return None
        try:
            if isnan(value):
                return None
        except (TypeError, ValueError):
            pass
        return np.radians(float(value))
        
    def set_keypoints(self, keypoints_3d: np.ndarray) -> None:
        """
        Set the keypoints to analyze.
        
        Args:
            keypoints_3d: Array of shape (25, 3) with [x, y, z] coordinates from Metrabs
        """
        if keypoints_3d.shape[1] != 3:
            raise ValueError(f"Expected shape (J, 3), got {keypoints_3d.shape}")
        self.keypoints = keypoints_3d.astype(np.float64)

    
    def _get_keypoint(self, joint_name):
        """Get keypoint by name."""
        try:
            idx = KINECTV2_25_JOINTS[joint_name]
            return self.keypoints[idx]
        except KeyError:
            return None
    
    # ============================================================================
    # PUBLIC INTERFACE - ANGLE COMPUTATION
    # ============================================================================
    
    def compute_joint_angles(self) -> Dict[str, float]:
        """
        Compute all 4 arm joint angles (t1, t2, t3, t4) for both arms.
        
        Returns:
            Dictionary with keys like "ShoulderPitch_Right", "ShoulderRoll_Right", etc.
            All values in degrees.
        """
        if self.keypoints is None:
            raise RuntimeError("Keypoints not set. Call set_keypoints() first.")
        
        angles = {}
        
        try:
            torso = self._get_keypoint("SpineChest")
            
            # Process right arm
            r_shoulder = self._get_keypoint("ShoulderRight")
            r_elbow = self._get_keypoint("ElbowRight")
            r_wrist = self._get_keypoint("WristRight")
            l_shoulder = self._get_keypoint("ShoulderLeft")
            
            if all(x is not None for x in [torso, r_shoulder, r_elbow, r_wrist, l_shoulder]):
                t1, t2, t3, t4, hand_angle = scaling.compute_arm_targets(torso, r_shoulder, r_elbow, r_wrist, l_shoulder, 'right', None, use_human_mode=False)
                angles["ShoulderPitch_Right"] = self._safe_angle_to_degrees(t1)
                angles["ShoulderRoll_Right"] = self._safe_angle_to_degrees(t2)
                angles["ElbowYaw_Right"] = self._safe_angle_to_degrees(t3)
                angles["ElbowRoll_Right"] = self._safe_angle_to_degrees(t4)
                angles["RHand"] = hand_angle if hand_angle is not None else None
            
            # Process left arm
            l_elbow = self._get_keypoint("ElbowLeft")
            l_wrist = self._get_keypoint("WristLeft")
            
            if all(x is not None for x in [torso, l_shoulder, l_elbow, l_wrist, r_shoulder]):
                t1, t2, t3, t4, hand_angle = scaling.compute_arm_targets(torso, l_shoulder, l_elbow, l_wrist, r_shoulder, 'left', None, use_human_mode=False)
                angles["ShoulderPitch_Left"] = self._safe_angle_to_degrees(t1)
                angles["ShoulderRoll_Left"] = self._safe_angle_to_degrees(t2)
                angles["ElbowYaw_Left"] = self._safe_angle_to_degrees(t3)
                angles["ElbowRoll_Left"] = self._safe_angle_to_degrees(t4)
                angles["LHand"] = hand_angle if hand_angle is not None else None
        
        except Exception as e:
            pass
        
        return angles
    
    def compute_joint_angles_from_payload(self, payload: Dict) -> Dict[str, float]:
        """
        Compute joint angles from network payload.
        
        Args:
            payload: Dictionary with "pose_keypoints" (base64-encoded array)
        
        Returns:
            Dictionary with computed joint angles in degrees (Pepper mode with limits)
        """
        try:
            pose_b64 = payload.get("pose_keypoints")
            if pose_b64 is None:
                raise ValueError("payload missing 'pose_keypoints' field")
            
            pose_bytes = base64.b64decode(pose_b64)
            pose_array = np.frombuffer(pose_bytes, dtype=np.float32).reshape(26, 3)
            
            scale_to_mm = 1000.0
            r_shoulder = pose_array[2, :] * scale_to_mm
            r_elbow = pose_array[3, :] * scale_to_mm
            r_wrist = pose_array[4, :] * scale_to_mm
            l_shoulder = pose_array[5, :] * scale_to_mm
            l_elbow = pose_array[6, :] * scale_to_mm
            l_wrist = pose_array[7, :] * scale_to_mm
            torso = pose_array[8, :] * scale_to_mm
            
            # Extract hand tips (indices 10-11 are right and left hand tips)
            r_hand_tip = pose_array[10, :] * scale_to_mm if pose_array.shape[0] > 10 else None
            l_hand_tip = pose_array[11, :] * scale_to_mm if pose_array.shape[0] > 11 else None
            
            angles = {}
            
            # Compute right arm angles
            if all(x is not None for x in [torso, r_shoulder, r_elbow, r_wrist, l_shoulder]):
                t1, t2, t3, t4, hand_angle = scaling.compute_arm_targets(torso, r_shoulder, r_elbow, r_wrist, l_shoulder, 'right', r_hand_tip, use_human_mode=False)
                angles["ShoulderPitch_Right"] = self._safe_angle_to_degrees(t1)
                angles["ShoulderRoll_Right"] = self._safe_angle_to_degrees(t2)
                angles["ElbowYaw_Right"] = self._safe_angle_to_degrees(t3)
                angles["ElbowRoll_Right"] = self._safe_angle_to_degrees(t4)
                angles["RHand"] = hand_angle if hand_angle is not None else None
            
            # Compute left arm angles
            if all(x is not None for x in [torso, l_shoulder, l_elbow, l_wrist, r_shoulder]):
                t1, t2, t3, t4, hand_angle = scaling.compute_arm_targets(torso, l_shoulder, l_elbow, l_wrist, r_shoulder, 'left', l_hand_tip, use_human_mode=False)
                angles["ShoulderPitch_Left"] = self._safe_angle_to_degrees(t1)
                angles["ShoulderRoll_Left"] = self._safe_angle_to_degrees(t2)
                angles["ElbowYaw_Left"] = self._safe_angle_to_degrees(t3)
                angles["ElbowRoll_Left"] = self._safe_angle_to_degrees(t4)
                angles["LHand"] = hand_angle if hand_angle is not None else None
            

            #print(f"Right Hand: {angles['RHand']}, Left Hand: {angles['LHand']}")
            return angles
        
        except Exception as e:
            return {}
    

    def detect_missing_keypoints(self, pose_array: np.ndarray) -> list:
        """
        Detect missing or invalid keypoints in pose array.
        
        Missing keypoints are represented as [0, 0, 0] (zero vectors).
        
        Args:
            pose_array: Array of shape (26, 3) with pose keypoints
        
        Returns:
            List of keypoint names that are missing/invalid (zero vectors)
        """
        #print(f"[DEBUG] Detecting missing keypoints in pose array of shape \n{pose_array}")
        missing_keypoints = []
        keypoint_names = [
            "Nose", "Neck", "ShoulderRight", "ElbowRight", "WristRight",
            "ShoulderLeft", "ElbowLeft", "WristLeft", "SpineChest", "Pelvis",
            "HandTipRight", "HandTipLeft",
            "HandWristRight", "ThumbCMCRight", "IndexMCPRight", "PinkyMCPRight",
            "HandWristLeft", "ThumbCMCLeft", "IndexMCPLeft", "PinkyMCPLeft",
            "PalmXRight", "PalmYRight", "PalmZRight",
            "PalmXLeft", "PalmYLeft", "PalmZLeft",
        ]

        for idx in range(min(26, pose_array.shape[0])):
            keypoint = pose_array[idx, :]
            # Check if keypoint is a zero vector (not detected) or NaN/Inf
            is_zero = np.allclose(keypoint, 0, atol=1e-8)
            is_invalid = not np.all(np.isfinite(keypoint))
            
            if is_zero or is_invalid:
                missing_keypoints.append(f"{keypoint_names[idx]}(idx:{idx})")

        #if missing_keypoints:
        #    print(f"[MISSING_KEYPOINTS] Detected {len(missing_keypoints)} missing keypoints: {', '.join(missing_keypoints)}\n\n")
        
        return missing_keypoints


    def compute_both_angles_from_payload(self, payload: Dict) -> Dict[str, Dict]:
        """
        Compute BOTH human and Pepper angles from payload, plus forearm directions and wristyaw.
        
        Returns angles computed with two modes:
        - "human": Full human motion angles (no angle limits) in RADIANS
        - "pepper": Pepper robot angles (with all constraints) in RADIANS
        - Forearm and upper arm directions for visualization
        - WristYaw angles computed from hand orientation in RADIANS
        - missing_keypoints: List of keypoints with NaN/Inf values
        
        Args:
            payload: Dictionary with "pose_keypoints" (base64-encoded array) and "hand_orientation"
        
        Returns:
            Dictionary with structure:
            {
                "human": {"ShoulderPitch_Right": ... (radians), "RWristYaw": ..., ...},
                "pepper": {"ShoulderPitch_Right": ... (radians), "RWristYaw": ..., ...},
                "forearm_direction_right": "forward"|"back"|"up"|"down"|"left"|"right"|None,
                "forearm_direction_left": "forward"|"back"|"up"|"down"|"left"|"right"|None,
                "upper_arm_direction_right": "forward"|"back"|"up"|"down"|"left"|"right"|None,
                "upper_arm_direction_left": "forward"|"back"|"up"|"down"|"left"|"right"|None,
                "missing_keypoints": ["Nose(idx:0)", "WristLeft(idx:7)", ...],
            }
        """
        try:

            pose_b64 = payload.get("pose_keypoints")
            if pose_b64 is None:
                raise ValueError("payload missing 'pose_keypoints' field")
            
            pose_bytes = base64.b64decode(pose_b64)
            pose_array = np.frombuffer(pose_bytes, dtype=np.float32).reshape(26, 3)
            
            # Detect missing keypoints early
            missing_keypoints = self.detect_missing_keypoints(pose_array)
            
            scale_to_mm = 1000.0
            
            # Extract all keypoints needed for head, torso, and arm angles
            nose = pose_array[0, :] * scale_to_mm           # Index 0
            neck = pose_array[1, :] * scale_to_mm           # Index 1
            r_shoulder = pose_array[2, :] * scale_to_mm     # Index 2
            r_elbow = pose_array[3, :] * scale_to_mm        # Index 3
            r_wrist = pose_array[4, :] * scale_to_mm        # Index 4
            l_shoulder = pose_array[5, :] * scale_to_mm     # Index 5
            l_elbow = pose_array[6, :] * scale_to_mm        # Index 6
            l_wrist = pose_array[7, :] * scale_to_mm        # Index 7
            torso = pose_array[8, :] * scale_to_mm          # Index 8 (SpineChest)
            spine_base = pose_array[9, :] * scale_to_mm     # Index 9 (Pelvis)
            
            # Extract hand tips (indices 10-11 are right and left hand tips)
            r_hand_tip = pose_array[10, :]  if pose_array.shape[0] > 10 else None
            l_hand_tip = pose_array[11, :]  if pose_array.shape[0] > 11 else None
            
            human_angles = {}
            pepper_angles = {}
            
            # Compute right arm angles (both modes) - returns radians
            if all(x is not None for x in [torso, r_shoulder, r_elbow, r_wrist, l_shoulder]):
                t1_h, t2_h, t3_h, t4_h, hand_h = scaling.compute_arm_targets(torso, r_shoulder, r_elbow, r_wrist, l_shoulder, 'right', r_hand_tip, use_human_mode=True)
                human_angles["ShoulderPitch_Right"] = self._safe_angle_to_radians(t1_h)
                human_angles["ShoulderRoll_Right"] = self._safe_angle_to_radians(t2_h)
                human_angles["ElbowYaw_Right"] = self._safe_angle_to_radians(t3_h)
                human_angles["ElbowRoll_Right"] = self._safe_angle_to_radians(t4_h)
                human_angles["RHand"] = hand_h if hand_h is not None else None
                
                t1_p, t2_p, t3_p, t4_p, hand_p = scaling.compute_arm_targets(torso, r_shoulder, r_elbow, r_wrist, l_shoulder, 'right', r_hand_tip, use_human_mode=False)
                pepper_angles["ShoulderPitch_Right"] = self._safe_angle_to_radians(t1_p)
                pepper_angles["ShoulderRoll_Right"] = self._safe_angle_to_radians(t2_p)
                pepper_angles["ElbowYaw_Right"] = self._safe_angle_to_radians(t3_p)
                pepper_angles["ElbowRoll_Right"] = self._safe_angle_to_radians(t4_p)
                pepper_angles["RHand"] = hand_p if hand_p is not None else None
            
            # Compute left arm angles (both modes) - returns radians
            if all(x is not None for x in [torso, l_shoulder, l_elbow, l_wrist, r_shoulder]):
                t1_h, t2_h, t3_h, t4_h, hand_h = scaling.compute_arm_targets(torso, l_shoulder, l_elbow, l_wrist, r_shoulder, 'left', l_hand_tip, use_human_mode=True)
                human_angles["ShoulderPitch_Left"] = self._safe_angle_to_radians(t1_h)
                human_angles["ShoulderRoll_Left"] = self._safe_angle_to_radians(t2_h)
                human_angles["ElbowYaw_Left"] = self._safe_angle_to_radians(t3_h)
                human_angles["ElbowRoll_Left"] = self._safe_angle_to_radians(t4_h)
                human_angles["LHand"] = hand_h if hand_h is not None else None
                
                t1_p, t2_p, t3_p, t4_p, hand_p = scaling.compute_arm_targets(torso, l_shoulder, l_elbow, l_wrist, r_shoulder, 'left', l_hand_tip, use_human_mode=False)
                pepper_angles["ShoulderPitch_Left"] = self._safe_angle_to_radians(t1_p)
                pepper_angles["ShoulderRoll_Left"] = self._safe_angle_to_radians(t2_p)
                pepper_angles["ElbowYaw_Left"] = self._safe_angle_to_radians(t3_p)
                pepper_angles["ElbowRoll_Left"] = self._safe_angle_to_radians(t4_p)
                pepper_angles["LHand"] = hand_p if hand_p is not None else None

            
            #TODO: check if this is correct in terms of values for both torso and head
            if all(x is not None for x in [torso, neck, nose, r_shoulder, l_shoulder]):
                head_names, head_angles = scaling.compute_head_targets(
                    torso, neck, nose, r_shoulder, l_shoulder
                )
                pepper_angles["HeadYaw"] = head_angles[0]
                pepper_angles["HeadPitch"] = head_angles[1]
                human_angles["HeadYaw"] = head_angles[0]
                human_angles["HeadPitch"] = head_angles[1]

            if all(x is not None for x in [spine_base, torso]):
                torso_names, torso_angles = scaling.compute_torso_targets(spine_base, torso)
                pepper_angles["TorsoPitch"] = torso_angles[0]
                pepper_angles["TorsoRoll"] = torso_angles[1]
                human_angles["TorsoPitch"] = torso_angles[0]
                human_angles["TorsoRoll"] = torso_angles[1]

            # Compute forearm and upper arm directions for visualization
            forearm_direction_right = None
            forearm_direction_left = None
            upper_arm_direction_right = None
            upper_arm_direction_left = None
            wrisyaw_right = 0.0
            wrisyaw_left = 0.0
            orientation_labels = payload.get("hand_orientation", {})
            
            
            # Create pose_out array (20-joint Pepper format) for direction computation
            pose_out = np.zeros((settings.OUT_JOINT_COUNT, settings.PARAMS_PER_JOINT), dtype=np.float32)
            
            # Map the full 26-keypoint pose to Pepper's 20-joint format
            pose_map._map_pose_20(pose_array.astype(np.float32), {}, pose_out)
            
            # Compute forearm and upper arm directions
            forearm_direction_right = pose_map._compute_discrete_direction_label(r_wrist - r_elbow)
            forearm_direction_left = pose_map._compute_discrete_direction_label(l_wrist - l_elbow)
            upper_arm_direction_right = pose_map._compute_discrete_direction_label(r_elbow - r_shoulder)
            upper_arm_direction_left = pose_map._compute_discrete_direction_label(l_elbow - l_shoulder)
            
            # Compute wristyaw angles from arm orientations
            wrisyaw_right, wrisyaw_left = ho.get_wrist_yaws(
                upper_arm_direction_left,
                forearm_direction_left,
                upper_arm_direction_right,
                forearm_direction_right,
                orientation_labels,
                wrisyaw_right,
                wrisyaw_left
            )

            wrisyaw_right = self._safe_convert_degrees_to_radians(wrisyaw_right)
            wrisyaw_left = self._safe_convert_degrees_to_radians(wrisyaw_left)

            # Add wristyaw to angles
            pepper_angles["RWristYaw"] = wrisyaw_right
            pepper_angles["LWristYaw"] = wrisyaw_left
            human_angles["RWristYaw"] = wrisyaw_right
            human_angles["LWristYaw"] = wrisyaw_left
            
            #print(f"\n\nRight Hand tip: {r_hand_tip}")
            #print(f"wrist position right : {r_wrist}")
            #print(f"\n\nRight Hand: {pepper_angles['RHand']}\n\n")
#
            #print(f"\n\nLeft Hand tip: {l_hand_tip}")
            #print(f"wrist position left : {l_wrist}")
            #print(f"\n\nLeft Hand: {pepper_angles['LHand']}\n\n")

            return {
                "human": human_angles,
                "pepper": pepper_angles,
                "forearm_direction_right": forearm_direction_right,
                "forearm_direction_left": forearm_direction_left,
                "upper_arm_direction_right": upper_arm_direction_right,
                "upper_arm_direction_left": upper_arm_direction_left,
                "missing_keypoints": missing_keypoints,
            }
        
        except Exception as e:
            print(f"Error in compute_both_angles_from_payload: {e}")
            return {
                "human": {},
                "pepper": {},
                "missing_keypoints": [],
            }
