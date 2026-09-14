# -*- coding: utf-8 -*-
from __future__ import print_function
import numpy as np
import math
from forward_kinematics import get_wrist_position 
from forward_kinematics import get_elbow_position
from math_functions import computeT_dh, rx_calc, ry_calc, rz_calc


# Toggle this to fix palm/back inversion without changing the core math.
# True: invert target palm normal (often fixes palm<->back swap).
# NOTE: This should be FALSE now that the server computes orientation correctly
TARGET_PALM_NORMAL_FLIPPED = False

# Toggle between discrete and continuous wrist orientation adjustment
# False: use continuous angle-based approach (compute_wristyaw_from_angles)
# True: use discrete 6-position approach (compute_wristyaw_from_discrete_orientation)
USE_DISCRETE_WRIST_ORIENTATION = True

# Toggle between two hand orientation computation methods
# False: use numerical differentiation approach (delta_t4 perturbation + cross products)
# True: use rotation matrix chain approach (proper DH chain with all 5 joints)
USE_ROTATION_MATRIX_METHOD = True


def set_target_palm_normal_flipped(enabled):
    global TARGET_PALM_NORMAL_FLIPPED
    TARGET_PALM_NORMAL_FLIPPED = bool(enabled)


def set_discrete_wrist_orientation(enabled):
    """Enable/disable discrete wrist orientation mode."""
    global USE_DISCRETE_WRIST_ORIENTATION
    USE_DISCRETE_WRIST_ORIENTATION = bool(enabled)


def get_discrete_wrist_orientation():
    """Check if discrete wrist orientation mode is enabled."""
    return USE_DISCRETE_WRIST_ORIENTATION


def set_rotation_matrix_method(enabled):
    """Enable/disable rotation matrix method for hand orientation computation."""
    global USE_ROTATION_MATRIX_METHOD
    USE_ROTATION_MATRIX_METHOD = bool(enabled)


def get_rotation_matrix_method():
    """Check if rotation matrix method is enabled."""
    return USE_ROTATION_MATRIX_METHOD


def _safe_normalize(vec, fallback):
    norm = np.linalg.norm(vec)
    if norm > 1e-6:
        return vec / norm
    return np.array(fallback, dtype=float)


def compute_robot_hand_orientation_from_rotation_matrix(t1, t2, t3, t4, t5, arm):
    """
    Compute hand orientation by chaining all 5 arm joint rotations using proper DH parameters.
    This is the mathematically exact method using rotation matrices.
    
    Accounts for left/right arm angle convention differences:
    - ShoulderPitch: same range both arms
    - ShoulderRoll: opposite ranges (negate for consistency)
    - ElbowYaw: opposite ranges (negate for consistency)
    - ElbowRoll: opposite ranges (negate for consistency)
    - WristYaw: same range both arms
    
    Args:
        t1, t2, t3, t4, t5: Arm joint angles in radians
            - t1: ShoulderPitch
            - t2: ShoulderRoll
            - t3: ElbowYaw
            - t4: ElbowRoll
            - t5: WristYaw
        arm: 'right' or 'left'
        
    Returns:
        palm_x, palm_y, palm_z: Unit vectors representing the wrist frame axes
    """
    try:
        # DH parameters extracted from arm_forward_matrix in forward_kinematics.py
        # These define the kinematic chain structure
        
        # Normalize joint angles for left arm (opposite conventions for some joints)
        # Right arm ranges: ShoulderRoll (−89.5° to −0.5°), ElbowYaw (+0.5° to +89.5°), ElbowRoll (+0.5° to +89.5°)
        # Left arm ranges:  ShoulderRoll (+0.5° to +89.5°), ElbowYaw (−89.5° to −0.5°), ElbowRoll (−89.5° to −0.5°)
        # To make them equivalent, negate the angles for left arm
        if arm == "left":
            t2_normalized = -t2  # ShoulderRoll: negate for left
            t3_normalized = -t3  # ElbowYaw: negate for left
            t4_normalized = -t4  # ElbowRoll: negate for left
        else:
            t2_normalized = t2
            t3_normalized = t3
            t4_normalized = t4
        
        # Adjust t2 (ShoulderRoll) as done in forward kinematics
        t2_adjusted = t2_normalized - np.pi / 2.0
        
        # Build transformation matrices for each joint
        # T = Rx(alpha) * Tx(a) * Rz(theta) * Tz(d)
        
        # T0: ShoulderPitch (rotates around Z by t1, with alpha=-pi/2)
        T0 = computeT_dh(-np.pi / 2.0, 0, t1, 0)
        
        # T1: ShoulderRoll (rotates around Z by t2_adjusted, with alpha=pi/2)
        T1 = computeT_dh(np.pi / 2.0, 0, t2_adjusted, 0)
        
        # T2: ElbowYaw (9-degree offset + theta3 rotation)
        # This combines the fixed offset and the joint angle
        a3 = 15.0 if arm == 'right' else -15.0
        d3 = 181.2
        z3 = 0.13
        T2 = computeT_dh(-np.pi / 2.0, a3, 0, d3)
        T2_offset = computeT_dh(np.radians(9.0), 0, t3_normalized, z3)
        
        # T3: ElbowRoll (rotates around Z by t4, with alpha=pi/2)
        T3 = computeT_dh(np.pi / 2.0, 0, t4_normalized, 0)
        
        # T4: WristYaw (rotates around Z by t5, with alpha=0)
        # Assuming WristYaw is a pure rotation around the wrist frame Z-axis
        # WristYaw has same range for both arms, but needs negation for left arm to follow convention
        t5_normalized = -t5 if arm == "left" else t5
        T4 = computeT_dh(0, 0, t5_normalized, 0)
        
        # Chain all transformations: T_total = T0 * T1 * T2 * T2_offset * T3 * T4
        T_total = T0.dot(T1).dot(T2).dot(T2_offset).dot(T3).dot(T4)
        
        # Extract the rotation part (upper-left 3x3)
        R_total = T_total[:3, :3]
        
        # The columns of the rotation matrix are the wrist frame axes
        # palm_x = 1st column (X-axis of wrist frame)
        # palm_y = 2nd column (Y-axis of wrist frame)
        # palm_z = 3rd column (Z-axis of wrist frame)
        palm_x = _safe_normalize(R_total[:, 0], [1.0, 0.0, 0.0])
        palm_y = _safe_normalize(R_total[:, 1], [0.0, 1.0, 0.0])
        palm_z = _safe_normalize(R_total[:, 2], [0.0, 0.0, 1.0])
        
        return palm_x, palm_y, palm_z
        
    except Exception as e:
        print("Error computing robot hand orientation from rotation matrix: {0}".format(e))
        return np.array([1, 0, 0], dtype=float), np.array([0, 1, 0], dtype=float), np.array([0, 0, 1], dtype=float)



def orientation_to_euler(palm_x, palm_y, palm_z):
    """
    Convert palm axes to Euler angles (yaw, pitch, roll) in radians.
    Same as the hand detection version but works with arrays directly.
    
    Args:
        palm_x, palm_y, palm_z: Unit vector axes of the palm
        
    Returns:
        yaw, pitch, roll: Euler angles in radians
    """
    yaw = math.atan2(palm_x[1], palm_x[0])
    pitch = math.atan2(-palm_z[0], np.sqrt(palm_z[1]**2 + palm_z[2]**2))
    roll = math.atan2(palm_z[1], palm_z[2])
    return yaw, pitch, roll


def orientation_to_euler_hand_detection_style(x_axis, y_axis, z_axis, in_degrees=False):
    """
    Compute yaw/pitch/roll exactly like yaw_new/pitch_new/roll_new in
    hand_detection_orientation.py:
        R_hand = column_stack((x_axis, y_axis, z_axis))
        yaw   = atan2(R[1,0], R[0,0])
        pitch = atan2(-R[2,0], sqrt(R[2,1]^2 + R[2,2]^2))
        roll  = atan2(R[2,1], R[2,2])
    """
    x_axis = _safe_normalize(np.array(x_axis, dtype=float), [1.0, 0.0, 0.0])
    y_axis = _safe_normalize(np.array(y_axis, dtype=float), [0.0, 1.0, 0.0])
    z_axis = _safe_normalize(np.array(z_axis, dtype=float), [0.0, 0.0, 1.0])

    r_hand = np.column_stack((x_axis, y_axis, z_axis))
    yaw = math.atan2(r_hand[1, 0], r_hand[0, 0])
    pitch = math.atan2(-r_hand[2, 0], math.sqrt(r_hand[2, 1]**2 + r_hand[2, 2]**2))
    roll = math.atan2(r_hand[2, 1], r_hand[2, 2])

    if in_degrees:
        return math.degrees(yaw), math.degrees(pitch), math.degrees(roll)
    return yaw, pitch, roll


def compute_robot_hand_orientation(t1, t2, t3, t4, arm, use_rotation_matrix=None, t5=0.0):
    """
    Compute the robot hand (palm) orientation given arm joint angles.
    
    Can use two methods:
    1. Numerical differentiation (default): Uses perturbation of t4 and cross products
    2. Rotation matrix chain: Uses proper DH chain with all 5 joints
    
    Args:
        t1, t2, t3, t4: Arm joint angles (radians)
            - t1: ShoulderPitch
            - t2: ShoulderRoll
            - t3: ElbowYaw
            - t4: ElbowRoll
        arm: 'right' or 'left'
        use_rotation_matrix: bool or None
            - None: Use the global USE_ROTATION_MATRIX_METHOD flag
            - True: Use rotation matrix chain method
            - False: Use numerical differentiation method
        t5: WristYaw angle in radians (only used if use_rotation_matrix=True)
        
    Returns:
        palm_x, palm_y, palm_z: Unit vectors representing hand axes
    """
    try:
        # Determine which method to use
        if use_rotation_matrix is None:
            use_rotation_matrix = USE_ROTATION_MATRIX_METHOD
        
        # Use rotation matrix method if enabled
        if use_rotation_matrix:
            return compute_robot_hand_orientation_from_rotation_matrix(t1, t2, t3, t4, t5, arm)
        
        # Default: Numerical differentiation method (original approach)
        # Get wrist position at current angles
        wrist_pos = get_wrist_position(t1, t2, t3, t4, arm)
        
        # Get wrist position at slightly different t4 to get orientation vector
        delta_t4 = 0.2  # Small angle change to compute orientation
        wrist_pos_rotated = get_wrist_position(t1, t2, t3, t4 + delta_t4, arm)
        
        # Vector representing the wrist roll axis (t4 rotation)
        wrist_roll_axis = wrist_pos_rotated - wrist_pos
        if np.linalg.norm(wrist_roll_axis) > 1e-6:
            wrist_roll_axis = wrist_roll_axis / np.linalg.norm(wrist_roll_axis)
        else:
            # Fallback if positions are too close
            wrist_roll_axis = np.array([0, 0, 1], dtype=float)
        
        # Get elbow position to compute wrist approach vector
        elbow_pos = get_elbow_position(t1, t2, arm)
        
        # Vector from shoulder to wrist (rough approach direction)
        approach_vector = wrist_pos - elbow_pos
        if np.linalg.norm(approach_vector) > 1e-6:
            approach_vector = approach_vector / np.linalg.norm(approach_vector)
        else:
            approach_vector = np.array([1, 0, 0], dtype=float)
        
        # Construct hand frame similar to hand detection:
        # palm_z: normal to palm (wrist approach direction)
        palm_z = approach_vector
        
        # palm_x: perpendicular to both palm_z and wrist_roll_axis
        palm_x = np.cross(wrist_roll_axis, palm_z)
        if np.linalg.norm(palm_x) > 1e-6:
            palm_x = palm_x / np.linalg.norm(palm_x)
        else:
            palm_x = np.array([1, 0, 0], dtype=float)
        
        # palm_y: complete the frame
        palm_y = np.cross(palm_z, palm_x)
        if np.linalg.norm(palm_y) > 1e-6:
            palm_y = palm_y / np.linalg.norm(palm_y)
        else:
            palm_y = np.array([0, 1, 0], dtype=float)
        
        return palm_x, palm_y, palm_z
        
    except Exception as e:
        print("Error computing robot hand orientation: {0}".format(e))
        # Return default frame
        return np.array([1, 0, 0], dtype=float), np.array([0, 1, 0], dtype=float), np.array([0, 0, 1], dtype=float)


def get_current_hand_orientation_report(t1, t2, t3, t4, arm, t5=0.0, use_rotation_matrix=None):
    """
    Get current hand orientation for diagnostic reporting.
    
    Args:
        t1, t2, t3, t4: Arm joint angles in radians
        arm: 'right' or 'left'
        t5: WristYaw angle in radians (only used if use_rotation_matrix=True)
        use_rotation_matrix: bool or None, which method to use
        
    Returns:
        dict with current orientation info (yaw_deg, palm vector, closest discrete label)
    """
    try:
        robot_palm_x, robot_palm_y, robot_palm_z = compute_robot_hand_orientation(t1, t2, t3, t4, arm, use_rotation_matrix=use_rotation_matrix, t5=t5)
        robot_yaw, robot_pitch, robot_roll = orientation_to_euler_hand_detection_style(
            robot_palm_x, robot_palm_y, robot_palm_z, in_degrees=False
        )
        
        # Convert to degrees for reporting
        robot_yaw_deg = math.degrees(robot_yaw)
        robot_pitch_deg = math.degrees(robot_pitch)
        robot_roll_deg = math.degrees(robot_roll)
        
        # Find closest discrete direction
        discrete_directions = {
            'forward': np.array([1.0, 0.0, 0.0]),
            'back': np.array([-1.0, 0.0, 0.0]),
            'up': np.array([0.0, 0.0, 1.0]),
            'down': np.array([0.0, 0.0, -1.0]),
            'left': np.array([0.0, -1.0, 0.0]),
            'right': np.array([0.0, 1.0, 0.0]),
        }
        
        # Current palm normal in world frame
        current_normal = np.array([robot_palm_x[0], robot_palm_y[0], robot_palm_z[0]], dtype=float)
        current_normal = current_normal / np.linalg.norm(current_normal)
        
        # Find closest direction using dot product
        similarities = {}
        for label, direction in discrete_directions.items():
            similarity = np.dot(current_normal, direction)
            similarities[label] = similarity
        
        closest_label = max(similarities, key=lambda k: similarities[k])
        closest_similarity = similarities[closest_label]
        
        return {
            'arm': arm,
            'yaw_deg': robot_yaw_deg,
            'pitch_deg': robot_pitch_deg,
            'roll_deg': robot_roll_deg,
            'palm_normal': [float(robot_palm_x[0]), float(robot_palm_y[0]), float(robot_palm_z[0])],
            'closest_discrete': closest_label,
            'similarity': float(closest_similarity),
        }
    except Exception as e:
        print("Error getting hand orientation report: {0}".format(e))
        return None


def compute_wristyaw_from_discrete_orientation(discrete_label, t1, t2, t3, t4, arm, t5=0.0, use_rotation_matrix=None):
    """
    Map discrete orientation label to WristYaw angle accounting for current arm position.
    Context-aware: same discrete label gets different WristYaw based on arm configuration.
    
    Args:
        discrete_label: string, one of 'forward', 'back', 'up', 'down', 'left', 'right'
        t1, t2, t3, t4: Arm joint angles in radians (ShoulderPitch, ShoulderRoll, ElbowYaw, ElbowRoll)
        arm: 'right' or 'left'
        t5: WristYaw angle in radians (only used if use_rotation_matrix=True)
        use_rotation_matrix: bool or None, which method to use
        
    Returns:
        tuple: (wrist_yaw, diagnostic_info)
        - wrist_yaw: The WristYaw angle (in radians) clamped to [-104.5°, +104.5°]
        - diagnostic_info: dict with current/target orientation for logging
    """
    try:
        # Map discrete labels to target direction vectors (in Pepper's world frame)
        # These represent the desired palm normal direction
        orientation_map = {
            'forward': np.array([1.0, 0.0, 0.0], dtype=float),   # +X (forward)
            'back': np.array([-1.0, 0.0, 0.0], dtype=float),     # -X (backward)
            'up': np.array([0.0, 0.0, 1.0], dtype=float),        # +Z (upward)
            'down': np.array([0.0, 0.0, -1.0], dtype=float),     # -Z (downward)
            'left': np.array([0.0, -1.0, 0.0], dtype=float),     # -Y (left)
            'right': np.array([0.0, 1.0, 0.0], dtype=float),     # +Y (right)
        }
        
        if discrete_label not in orientation_map:
            return 0.0, None
        
        target_direction = orientation_map[discrete_label]
        
        # Get current robot hand orientation from FK
        robot_palm_x, robot_palm_y, robot_palm_z = compute_robot_hand_orientation(t1, t2, t3, t4, arm, use_rotation_matrix=use_rotation_matrix, t5=t5)
        robot_yaw, _, _ = orientation_to_euler_hand_detection_style(
            robot_palm_x, robot_palm_y, robot_palm_z, in_degrees=False
        )
        
        # Calculate target yaw from target direction
        # atan2(y, x) gives the yaw angle of the direction vector
        target_yaw = math.atan2(target_direction[1], target_direction[0])
        
        # Compute the difference
        yaw_difference = target_yaw - robot_yaw
        
        # Normalize angle to [-pi, pi]
        while yaw_difference > np.pi:
            yaw_difference -= 2 * np.pi
        while yaw_difference < -np.pi:
            yaw_difference += 2 * np.pi
        
        # Clamp to Pepper's WristYaw joint limits: [-104.5°, +104.5°] = [-1.823 rad, +1.823 rad]
        WRIST_YAW_MIN_RAD = -(104.5 * np.pi / 180.0)
        WRIST_YAW_MAX_RAD = (104.5 * np.pi / 180.0)
        yaw_difference_clamped = np.clip(yaw_difference, WRIST_YAW_MIN_RAD, WRIST_YAW_MAX_RAD)
        
        # For left arm, negate due to opposite rotation convention
        if arm == "left":
            yaw_difference_clamped = -yaw_difference_clamped
        
        # Build diagnostic info
        diagnostic_info = {
            'arm': arm,
            'discrete_label': discrete_label,
            'current_yaw_deg': math.degrees(robot_yaw),
            'target_yaw_deg': math.degrees(target_yaw),
            'yaw_delta_deg': math.degrees(yaw_difference),
            'yaw_delta_clamped_deg': math.degrees(yaw_difference_clamped),
            'was_clamped': abs(yaw_difference) > abs(yaw_difference_clamped),
        }
        
        return yaw_difference_clamped, diagnostic_info
        
    except Exception as e:
        print("Error computing WristYaw from discrete orientation: {0}".format(e))
        return 0.0, None


def compute_target_hand_orientation_from_keypoints(wrist, thumb_cmc, index_mcp, pinky_mcp, is_left_hand=False):
    """
    Compute target hand orientation from 4 detected keypoints.
    Matches the corrected algorithm from hand_detection_orientation.py
    
    Args:
        wrist, thumb_cmc, index_mcp, pinky_mcp: 3D keypoints (numpy arrays or array-like)
        is_left_hand: bool, whether this is a left hand (affects cross product order)
        
    Returns:
        palm_x, palm_y, palm_z: Unit vectors representing target hand orientation
    """
    wrist = np.array(wrist, dtype=float)
    thumb_cmc = np.array(thumb_cmc, dtype=float)
    index_mcp = np.array(index_mcp, dtype=float)
    pinky_mcp = np.array(pinky_mcp, dtype=float)

    # Compute palm normal with correct cross product order for each hand chirality
    v1 = index_mcp - wrist
    v2 = pinky_mcp - wrist
    
    # Due to MediaPipe's mirrored coordinates for left/right hands, use different cross product orders
    if is_left_hand:
        # For LEFT hand: v2 × v1 gives outward normal
        palm_z = np.cross(v2, v1)
    else:
        # For RIGHT hand: v1 × v2 gives outward normal (opposite order due to chirality)
        palm_z = np.cross(v1, v2)
    
    palm_z = _safe_normalize(palm_z, [0.0, 0.0, 1.0])

    # X axis: use thumb as consistent anatomical reference (always on same side relative to palm)
    v_thumb = thumb_cmc - wrist
    # Project thumb vector onto palm plane to get consistent X direction
    palm_x = v_thumb - np.dot(v_thumb, palm_z) * palm_z
    palm_x = _safe_normalize(palm_x, [1.0, 0.0, 0.0])

    # Y axis along palm (perpendicular to both X and Z)
    palm_y = np.cross(palm_z, palm_x)
    palm_y = _safe_normalize(palm_y, [0.0, 1.0, 0.0])

    return palm_x, palm_y, palm_z





def validate_hand_orientation_sources(
    server_palm_x, server_palm_y, server_palm_z,
    target_palm_x, target_palm_y, target_palm_z,
    robot_palm_x, robot_palm_y, robot_palm_z,
    arm="right"
):
    """
    Compare three sources of hand orientation to detect discrepancies.
    
    Validates that:
    1. Server-computed vectors match target vectors (from same keypoints)
    2. Target vectors match robot vectors (from FK)
    
    Args:
        server_palm_x, server_palm_y, server_palm_z: Palm vectors computed on server
        target_palm_x, target_palm_y, target_palm_z: Palm vectors computed on client from keypoints
        robot_palm_x, robot_palm_y, robot_palm_z: Palm vectors from robot forward kinematics
        arm: 'right' or 'left'
        
    Returns:
        dict with comparison results:
            - 'arm': arm side
            - 'server_yaw', 'target_yaw', 'robot_yaw': yaw angles in radians
            - 'server_target_diff': difference between server and target yaw
            - 'target_robot_diff': difference between target and robot yaw
            - 'mismatch': bool, True if differences exceed threshold
            - 'mismatch_details': list of any warnings/issues found
    """
    try:
        angle_diff_threshold = 0.15  # radians (~8.6 degrees)
        mismatch_details = []
        
        # Safe conversion to Euler angles
        server_yaw, server_pitch, server_roll = orientation_to_euler_hand_detection_style(
            server_palm_x, server_palm_y, server_palm_z, in_degrees=False
        )
        target_yaw, target_pitch, target_roll = orientation_to_euler_hand_detection_style(
            target_palm_x, target_palm_y, target_palm_z, in_degrees=False
        )
        robot_yaw, robot_pitch, robot_roll = orientation_to_euler_hand_detection_style(
            robot_palm_x, robot_palm_y, robot_palm_z, in_degrees=False
        )
        
        # Normalize angle differences to [-pi, pi]
        def normalize_angle_diff(a1, a2):
            diff = a1 - a2
            while diff > np.pi:
                diff -= 2 * np.pi
            while diff < -np.pi:
                diff += 2 * np.pi
            return diff
        
        server_target_diff = normalize_angle_diff(server_yaw, target_yaw)
        target_robot_diff = normalize_angle_diff(target_yaw, robot_yaw)
        
        # Check for mismatches
        has_mismatch = False
        if abs(server_target_diff) > angle_diff_threshold:
            mismatch_details.append(
                "Server-Target mismatch: {0:.2f} rad ({1:.1f} deg)".format(
                    server_target_diff, math.degrees(server_target_diff)
                )
            )
            has_mismatch = True
        
        if abs(target_robot_diff) > angle_diff_threshold:
            mismatch_details.append(
                "Target-Robot mismatch: {0:.2f} rad ({1:.1f} deg)".format(
                    target_robot_diff, math.degrees(target_robot_diff)
                )
            )
            has_mismatch = True
        
        return {
            'arm': arm,
            'server_yaw': server_yaw,
            'target_yaw': target_yaw,
            'robot_yaw': robot_yaw,
            'server_yaw_deg': math.degrees(server_yaw),
            'target_yaw_deg': math.degrees(target_yaw),
            'robot_yaw_deg': math.degrees(robot_yaw),
            'server_target_diff': server_target_diff,
            'target_robot_diff': target_robot_diff,
            'server_target_diff_deg': math.degrees(server_target_diff),
            'target_robot_diff_deg': math.degrees(target_robot_diff),
            'mismatch': has_mismatch,
            'mismatch_details': mismatch_details,
        }
        
    except Exception as e:
        return {
            'arm': arm,
            'mismatch': True,
            'mismatch_details': ["Error during validation: {0}".format(e)],
        }


def compute_wristyaw_from_angles(names, angles, handwrist, thumbcmc, indexmcp, pinkymcp, arm,
                                 server_palm_x=None, server_palm_y=None, server_palm_z=None,
                                 enable_validation_log=False, discrete_orientation=None, 
                                 use_rotation_matrix=None, t5=0.0):
    """
    Compute the WristYaw angle that aligns the robot hand orientation with the target
    hand orientation from the detected keypoints.
    
    Can operate in two modes:
    1. Continuous mode (default): aligns with target hand orientation from keypoints
    2. Discrete mode (if flag enabled): uses discrete orientation labels (forward/back/up/down/left/right)
    
    Args:
        names: List of joint names from the arm chain
        angles: List of angles corresponding to the names
        handwrist, thumbcmc, indexmcp, pinkymcp: Hand keypoints for this arm
        arm: 'right' or 'left'
        server_palm_x, server_palm_y, server_palm_z: Server-computed palm vectors (optional)
        enable_validation_log: If True, log validation results when server vectors provided
        discrete_orientation: discrete label string ('forward', 'back', 'up', 'down', 'left', 'right')
                             Used if USE_DISCRETE_WRIST_ORIENTATION flag is True
        use_rotation_matrix: bool or None, which method to use for hand orientation
        t5: WristYaw angle in radians (only used if use_rotation_matrix=True)
        
    Returns:
        adjusted_wrist_yaw: The WristYaw angle (in radians) to align with target orientation
    """
    try:
        # Extract arm angles from the names and angles lists.
        # Find the indices of arm joints in the names list
        t1_idx = None
        t2_idx = None
        t3_idx = None
        t4_idx = None
        
        arm_prefix = "R" if arm == "right" else "L"
        
        for i, name in enumerate(names):
            # Map joint names: t1=ShoulderPitch, t2=ShoulderRoll, t3=ElbowYaw, t4=ElbowRoll
            if "{0}ShoulderPitch".format(arm_prefix) in name:
                t1_idx = i
            elif "{0}ShoulderRoll".format(arm_prefix) in name:
                t2_idx = i
            elif "{0}ElbowYaw".format(arm_prefix) in name:
                t3_idx = i
            elif "{0}ElbowRoll".format(arm_prefix) in name:
                t4_idx = i
        
        # If we found all necessary angles, compute robot hand orientation
        if t1_idx is not None and t2_idx is not None and t3_idx is not None and t4_idx is not None:
            t1 = angles[t1_idx]
            t2 = angles[t2_idx]
            t3 = angles[t3_idx]
            t4 = angles[t4_idx]
            
            # Check if we should use discrete orientation mode
            if USE_DISCRETE_WRIST_ORIENTATION and discrete_orientation is not None:
                wrist_yaw, diag_info = compute_wristyaw_from_discrete_orientation(
                    discrete_orientation, t1, t2, t3, t4, arm, t5=t5, use_rotation_matrix=use_rotation_matrix
                )
                if diag_info and enable_validation_log:
                    print("[{0}] Target: {1} | Current: {2:.1f}° | Target: {3:.1f}° | Delta: {4:.1f}° | Clamped: {5}".format(
                        arm.upper(),
                        diag_info['discrete_label'],
                        diag_info['current_yaw_deg'],
                        diag_info['target_yaw_deg'],
                        diag_info['yaw_delta_deg'],
                        diag_info['was_clamped']
                    ))
                return wrist_yaw
            
            # Default: Continuous orientation mode
            # Compute current robot hand orientation
            robot_palm_x, robot_palm_y, robot_palm_z = compute_robot_hand_orientation(
                t1, t2, t3, t4, arm, use_rotation_matrix=use_rotation_matrix, t5=t5
            )
            robot_yaw, robot_pitch, robot_roll = orientation_to_euler_hand_detection_style(
                robot_palm_x,
                robot_palm_y,
                robot_palm_z,
                in_degrees=False,
            )
            
            # Get target hand keypoints
            hand_keypoints = (handwrist, thumbcmc, indexmcp, pinkymcp)
            
            # Check if keypoints are valid (not all zeros)
            if all(np.linalg.norm(kp) > 1e-6 for kp in hand_keypoints):
                # Compute target hand orientation from detected keypoints
                target_palm_x, target_palm_y, target_palm_z = compute_target_hand_orientation_from_keypoints(*hand_keypoints)
                target_yaw, target_pitch, target_roll = orientation_to_euler_hand_detection_style(
                    target_palm_x,
                    target_palm_y,
                    target_palm_z,
                    in_degrees=False,
                )
                
                # Compute the WristYaw angle needed to match target yaw
                # WristYaw rotates around the wrist frame Z-axis, which affects the yaw angle
                yaw_difference = target_yaw - robot_yaw
                
                # Normalize angle to [-pi, pi]
                while yaw_difference > np.pi:
                    yaw_difference -= 2 * np.pi
                while yaw_difference < -np.pi:
                    yaw_difference += 2 * np.pi
                
                # Clamp to Pepper's WristYaw joint limits: [-104.5°, +104.5°] = [-1.823 rad, +1.823 rad]
                WRIST_YAW_MIN_RAD = -(104.5 * np.pi / 180.0)  # -1.823 rad
                WRIST_YAW_MAX_RAD = (104.5 * np.pi / 180.0)   # +1.823 rad
                yaw_difference = np.clip(yaw_difference, WRIST_YAW_MIN_RAD, WRIST_YAW_MAX_RAD)
                
                # Validate hand orientation sources if server vectors provided
                if server_palm_x is not None and server_palm_y is not None and server_palm_z is not None:
                    try:
                        validation_result = validate_hand_orientation_sources(
                            server_palm_x, server_palm_y, server_palm_z,
                            target_palm_x, target_palm_y, target_palm_z,
                            robot_palm_x, robot_palm_y, robot_palm_z,
                            arm=arm
                        )
                        if enable_validation_log:
                            arm_label = "[RIGHT] " if arm == "right" else "[LEFT]  "
                            #print("{0}Hand Orientation Validation:".format(arm_label))
                            #print("  Server: {0:.2f} | Target: {1:.2f} | Robot: {2:.2f}".format(
                            #    validation_result.get('server_yaw_deg', 0),
                            #    validation_result.get('target_yaw_deg', 0),
                            #    validation_result.get('robot_yaw_deg', 0)
                            #))
                            if validation_result.get('mismatch'):
                                #print("  WARNING - Mismatch detected:")
                                for detail in validation_result.get('mismatch_details', []):
                                    print("    - {0}".format(detail))
                    except Exception as val_e:
                        if enable_validation_log:
                            print("  Validation error: {0}".format(val_e))
                
                # For left arm, negate the value due to opposite rotation convention
                if arm == "left":
                    yaw_difference = -yaw_difference
                
                return yaw_difference
        
        # If we couldn't compute, return 0
        return 0.0
        
    except Exception as e:
        print("Error computing WristYaw from angles: {0}".format(e))
        return 0.0
