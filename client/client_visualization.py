# -*- coding: utf-8 -*-
"""
Client-side hand orientation debug visualization.

Displays received hand keypoints and compares:
- Orientation computed from received detection points
- Wrist orientation command sent to Pepper robot
"""
from __future__ import print_function
import cv2
import numpy as np
import math
import matplotlib.pyplot as plt

HAND_CONNECTIONS_2D = [
    (0, 1), (1, 5), (5, 17), (17, 0),
]

HAND_DEBUG_WINDOW_NAME = "Hand Orientation Debug (Client)"
ENABLE_CLIENT_HAND_VIS = True

_window_open = False
_fig = None


def _safe_normalize(vec, fallback=(1.0, 0.0, 0.0)):
    """Safely normalize a vector or return fallback."""
    vec = np.array(vec, dtype=float)
    norm = np.linalg.norm(vec)
    if norm > 1e-6:
        return vec / norm
    return np.array(fallback, dtype=float)


def _compute_orientation_from_keypoints(keypoints_dict):
    """
    Compute hand orientation (yaw, pitch, roll) from 4 keypoints.
    
    Args:
        keypoints_dict: dict mapping index -> 3D point:
            0: wrist, 1: thumb_cmc, 5: index_mcp, 17: pinky_mcp
    
    Returns:
        (yaw_deg, pitch_deg, roll_deg, valid): Euler angles in degrees, and validity flag
    """
    try:
        if not all(idx in keypoints_dict for idx in (0, 1, 5, 17)):
            return (0.0, 0.0, 0.0, False)
        
        wrist = np.array(keypoints_dict[0], dtype=float)
        thumb_cmc = np.array(keypoints_dict[1], dtype=float)
        index_mcp = np.array(keypoints_dict[5], dtype=float)
        pinky_mcp = np.array(keypoints_dict[17], dtype=float)
        
        # Check if any are zero vectors
        if (np.linalg.norm(wrist) < 1e-6 or np.linalg.norm(thumb_cmc) < 1e-6 or
            np.linalg.norm(index_mcp) < 1e-6 or np.linalg.norm(pinky_mcp) < 1e-6):
            return (0.0, 0.0, 0.0, False)
        
        # Compute palm axes (same as hand_orientation.py)
        v1 = index_mcp - wrist
        v2 = pinky_mcp - wrist
        palm_normal = _safe_normalize(np.cross(v1, v2), (0.0, 0.0, 1.0))
        
        mid_top = (wrist + thumb_cmc) / 2.0
        mid_bottom = (index_mcp + pinky_mcp) / 2.0
        vertical = _safe_normalize(mid_bottom - mid_top, (0.0, 1.0, 0.0))
        
        third_axis = _safe_normalize(np.cross(palm_normal, vertical), (1.0, 0.0, 0.0))
        
        # Compute yaw/pitch/roll from axes (same as hand_orientation.py)
        r_hand = np.column_stack((third_axis, vertical, palm_normal))
        yaw = math.atan2(r_hand[1, 0], r_hand[0, 0])
        pitch = math.atan2(-r_hand[2, 0], math.sqrt(r_hand[2, 1]**2 + r_hand[2, 2]**2))
        roll = math.atan2(r_hand[2, 1], r_hand[2, 2])
        
        yaw_deg = math.degrees(yaw)
        pitch_deg = math.degrees(pitch)
        roll_deg = math.degrees(roll)
        
        return (yaw_deg, pitch_deg, roll_deg, True)
        
    except Exception as e:
        print("Error computing orientation from keypoints: {0}".format(e))
        return (0.0, 0.0, 0.0, False)





def update_hand_orientation_debug(payload):
    """
    Update and display hand orientation debug visualization with 2x2 subplots.
    
    Top row: Hand keypoints received from server (right and left)
    Bottom row: Palm orientation vectors (right and left)
    
    Args:
        payload: dict with keys:
            'right': dict with 'keypoints', 'received_ypr', 'wrist_yaw_cmd', 'valid'
            'left': dict with 'keypoints', 'received_ypr', 'wrist_yaw_cmd', 'valid'
            'timestamp': optional frame timestamp
    """
    global _window_open, _fig
    
    if not ENABLE_CLIENT_HAND_VIS:
        return
    
    try:
        # Print keypoint values to console
        #_print_keypoint_values(payload)
        
        # Create figure only once on first call
        if _fig is None:
            _fig = plt.figure(figsize=(14, 10))
        else:
            # Clear previous content from all subplots
            for ax in _fig.get_axes():
                ax.clear()
        _fig.suptitle(u'Hand Orientation Debug - Server to Robot', fontsize=14, fontweight='bold')
        
        # Create 2x2 grid: 2 rows, 2 columns
        ax_right_keypoints = plt.subplot(2, 2, 1)
        ax_left_keypoints = plt.subplot(2, 2, 2)
        ax_right_orient = plt.subplot(2, 2, 3)
        ax_left_orient = plt.subplot(2, 2, 4)
        
        right_data = payload.get('right', {})
        left_data = payload.get('left', {})
        
        # Top row: Keypoints
        _draw_keypoints_subplot(ax_right_keypoints, right_data, u"Right Hand Keypoints")
        _draw_keypoints_subplot(ax_left_keypoints, left_data, u"Left Hand Keypoints")
        
        # Bottom row: Orientation vectors
        _draw_orientation_subplot(ax_right_orient, right_data, u"Right Hand Orientation")
        _draw_orientation_subplot(ax_left_orient, left_data, u"Left Hand Orientation")
        
        plt.tight_layout()
        plt.draw()
        plt.pause(0.001)
        _window_open = True
        
    except Exception as e:
        print("Error in hand orientation debug visualization: {0}".format(e))


def _print_keypoint_values(payload):
    """
    Print hand keypoint coordinate values to console.
    
    Args:
        payload: dict with 'right' and 'left' hand data
    """
    try:
        print(u"\n" + u"="*80)
        print(u"HAND KEYPOINT VALUES")
        print(u"="*80)
        
        # Right hand
        right_data = payload.get('right', {})
        if right_data.get('valid', False):
            right_kpts = right_data.get('keypoints', {})
            print(u"\nRIGHT HAND:")
            print(u"  Wrist (0):    {0}".format(right_kpts.get(0, 'N/A')))
            print(u"  Thumb (1):    {0}".format(right_kpts.get(1, 'N/A')))
            print(u"  Index (5):    {0}".format(right_kpts.get(5, 'N/A')))
            print(u"  Pinky (17):   {0}".format(right_kpts.get(17, 'N/A')))
            r_yaw, r_pitch, r_roll = right_data.get('received_ypr', (0.0, 0.0, 0.0))
            r_wrist = math.degrees(right_data.get('wrist_yaw_cmd', 0.0))
            print(u"  YPR: {0:.2f}, {1:.2f}, {2:.2f}".format(r_yaw, r_pitch, r_roll))
            print(u"  WristYaw Cmd: {0:.2f}".format(r_wrist))
        else:
            print(u"\nRIGHT HAND: No valid data")
        
        # Left hand
        left_data = payload.get('left', {})
        if left_data.get('valid', False):
            left_kpts = left_data.get('keypoints', {})
            print(u"\nLEFT HAND:")
            print(u"  Wrist (0):    {0}".format(left_kpts.get(0, 'N/A')))
            print(u"  Thumb (1):    {0}".format(left_kpts.get(1, 'N/A')))
            print(u"  Index (5):    {0}".format(left_kpts.get(5, 'N/A')))
            print(u"  Pinky (17):   {0}".format(left_kpts.get(17, 'N/A')))
            l_yaw, l_pitch, l_roll = left_data.get('received_ypr', (0.0, 0.0, 0.0))
            l_wrist = math.degrees(left_data.get('wrist_yaw_cmd', 0.0))
            print(u"  YPR: {0:.2f}, {1:.2f}, {2:.2f}".format(l_yaw, l_pitch, l_roll))
            print(u"  WristYaw Cmd: {0:.2f}".format(l_wrist))
        else:
            print(u"\nLEFT HAND: No valid data")
        
        print(u"="*80 + u"\n")
        
    except Exception as e:
        print("Error printing keypoint values: {0}".format(e))


def _draw_keypoints_subplot(ax, hand_data, title):
    """
    Draw hand keypoints on a subplot.
    
    Args:
        ax: matplotlib axes
        hand_data: dict with 'keypoints', 'received_ypr', 'wrist_yaw_cmd', 'valid'
        title: subplot title
    """
    ax.clear()
    ax.set_aspect('equal')
    ax.set_title(title, fontweight='bold')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.grid(True, alpha=0.3)
    
    valid = hand_data.get('valid', False)
    keypoints = hand_data.get('keypoints', {})
    received_ypr = hand_data.get('received_ypr', (0.0, 0.0, 0.0))
    wrist_yaw_cmd = hand_data.get('wrist_yaw_cmd', 0.0)
    
    if not valid or not keypoints:
        ax.text(0.5, 0.5, u'No hand data', ha='center', va='center', 
                fontsize=12, color='red', transform=ax.transAxes)
        return
    
    # Collect valid keypoints and compute centroid for centering
    valid_points = {}
    for idx in (0, 1, 5, 17):
        if idx in keypoints:
            pt = keypoints[idx]
            if pt is not None and len(pt) >= 2:
                valid_points[idx] = np.array(pt[:2], dtype=float)
    
    if not valid_points:
        ax.text(0.5, 0.5, u'No valid keypoints', ha='center', va='center', 
                fontsize=12, color='red', transform=ax.transAxes)
        return
    
    # Compute centroid to center the display
    centroid = np.mean([p for p in valid_points.values()], axis=0)
    
    # Compute scale based on point spread
    max_dist = np.max([np.linalg.norm(p - centroid) for p in valid_points.values()])
    scale = 0.4 / (max_dist + 1e-6)  # Scale to fit in [-0.4, 0.4] range
    
    # Draw keypoints centered and scaled
    keypoint_colors = {0: 'green', 1: 'yellow', 5: 'blue', 17: 'red'}
    keypoint_labels = {0: 'Wrist', 1: 'Thumb', 5: 'Index', 17: 'Pinky'}
    
    for idx in (0, 1, 5, 17):
        if idx in valid_points:
            pt_scaled = (valid_points[idx] - centroid) * scale
            x, y = pt_scaled[0], pt_scaled[1]
            color = keypoint_colors.get(idx, 'white')
            label = keypoint_labels.get(idx, u'KP{0}'.format(idx))
            ax.plot(x, y, 'o', color=color, markersize=12, label=label)
            ax.text(x + 0.03, y + 0.03, label, fontsize=10, color=color, fontweight='bold')
    
    # Draw connections
    for start_idx, end_idx in HAND_CONNECTIONS_2D:
        if start_idx in valid_points and end_idx in valid_points:
            pt1_scaled = (valid_points[start_idx] - centroid) * scale
            pt2_scaled = (valid_points[end_idx] - centroid) * scale
            ax.plot([pt1_scaled[0], pt2_scaled[0]], [pt1_scaled[1], pt2_scaled[1]], 'k-', linewidth=2)
    
    # Set auto limits with some padding
    ax.autoscale_view()
    
    # Display orientation info and keypoint coordinates
    yaw_recv, pitch_recv, roll_recv = received_ypr
    wrist_yaw_deg = math.degrees(wrist_yaw_cmd)
    
    # Build keypoint coordinates text
    kpt_text = u"Keypoint Coords:\n"
    keypoint_names = {0: u'Wrist', 1: u'Thumb', 5: u'Index', 17: u'Pinky'}
    for idx in (0, 1, 5, 17):
        if idx in valid_points:
            pt = valid_points[idx]
            kpt_text += u"  {0}: ({1:.3f}, {2:.3f})\n".format(keypoint_names[idx], pt[0], pt[1])
    
    info_text = (
        u"Received YPR:\n"
        u"  Y: {0:.1f}\u00b0 P: {1:.1f}\u00b0 R: {2:.1f}\u00b0\n"
        u"Robot WristYaw: {3:.1f}\u00b0\n\n"
        u"{4}"
    ).format(yaw_recv, pitch_recv, roll_recv, wrist_yaw_deg, kpt_text)
    
    ax.text(0.02, 0.98, info_text, transform=ax.transAxes, fontsize=8, 
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.9),
            family='monospace')


def _draw_orientation_subplot(ax, hand_data, title):
    """
    Draw palm orientation vectors on a subplot.
    
    Args:
        ax: matplotlib axes
        hand_data: dict with 'keypoints', 'received_ypr', 'wrist_yaw_cmd', 'valid'
        title: subplot title
    """
    ax.clear()
    ax.set_xlim(-1.2, 1.2)
    ax.set_ylim(-1.2, 1.2)
    ax.set_aspect('equal')
    ax.set_title(title, fontweight='bold')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.grid(True, alpha=0.3)
    
    valid = hand_data.get('valid', False)
    keypoints = hand_data.get('keypoints', {})
    received_ypr = hand_data.get('received_ypr', (0.0, 0.0, 0.0))
    wrist_yaw_cmd = hand_data.get('wrist_yaw_cmd', 0.0)
    
    if not valid:
        ax.text(0.5, 0.5, 'No hand data', ha='center', va='center', 
                fontsize=12, color='red', transform=ax.transAxes)
        return
    
    # Compute orientation vectors from keypoints
    orient_vectors = _compute_orientation_vectors(keypoints)
    
    if orient_vectors is None:
        ax.text(0.5, 0.5, 'Cannot compute\norientation', ha='center', va='center', 
                fontsize=12, color='orange', transform=ax.transAxes)
        return
    
    # Draw coordinate frame at origin
    ax.plot(0, 0, 'ko', markersize=8, label='Palm origin')
    
    # Draw axes vectors (X, Y, Z) projected to 2D
    palm_x = orient_vectors.get('palm_x', np.array([1, 0, 0]))
    palm_y = orient_vectors.get('palm_y', np.array([0, 1, 0]))
    palm_z = orient_vectors.get('palm_z', np.array([0, 0, 1]))
    
    # Project 3D vectors to 2D (using XY plane, ignoring Z for now)
    # Draw X-axis (red)
    ax.arrow(0, 0, palm_x[0], palm_x[1], head_width=0.08, head_length=0.08, 
             fc='red', ec='red', linewidth=2, label='X-axis (palm_x)')
    ax.text(palm_x[0] * 1.15, palm_x[1] * 1.15, 'X', fontsize=10, color='red', fontweight='bold')
    
    # Draw Y-axis (green)
    ax.arrow(0, 0, palm_y[0], palm_y[1], head_width=0.08, head_length=0.08, 
             fc='green', ec='green', linewidth=2, label='Y-axis (palm_y)')
    ax.text(palm_y[0] * 1.15, palm_y[1] * 1.15, 'Y', fontsize=10, color='green', fontweight='bold')
    
    # Draw Z-axis (blue) - shown as Z-component influence on XY
    ax.arrow(0, 0, palm_z[0], palm_z[1], head_width=0.08, head_length=0.08, 
             fc='blue', ec='blue', linewidth=2, label='Z-axis (palm_z)')
    ax.text(palm_z[0] * 1.15, palm_z[1] * 1.15, 'Z', fontsize=10, color='blue', fontweight='bold')
    
    # Display YPR and wrist command
    yaw_recv, pitch_recv, roll_recv = received_ypr
    wrist_yaw_deg = math.degrees(wrist_yaw_cmd)
    
    info_text = (
        u"Computed Orientation:\n"
        u"  Yaw: {0:.1f}\u00b0\n"
        u"  Pitch: {1:.1f}\u00b0\n"
        u"  Roll: {2:.1f}\u00b0\n"
        u"Robot Cmd: {3:.1f}\u00b0"
    ).format(yaw_recv, pitch_recv, roll_recv, wrist_yaw_deg)
    
    ax.text(0.02, 0.98, info_text, transform=ax.transAxes, fontsize=9,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8),
            family='monospace')
    
    ax.legend(loc='lower right', fontsize=8)


def _compute_orientation_vectors(keypoints_dict):
    """
    Compute hand orientation vectors (palm_x, palm_y, palm_z) from 4 keypoints.
    
    Args:
        keypoints_dict: dict mapping index -> 3D point:
            0: wrist, 1: thumb_cmc, 5: index_mcp, 17: pinky_mcp
    
    Returns:
        dict with 'palm_x', 'palm_y', 'palm_z' unit vectors, or None if invalid
    """
    try:
        if not all(idx in keypoints_dict for idx in (0, 1, 5, 17)):
            return None
        
        wrist = np.array(keypoints_dict[0], dtype=float)
        thumb_cmc = np.array(keypoints_dict[1], dtype=float)
        index_mcp = np.array(keypoints_dict[5], dtype=float)
        pinky_mcp = np.array(keypoints_dict[17], dtype=float)
        
        # Check if any are zero vectors
        if (np.linalg.norm(wrist) < 1e-6 or np.linalg.norm(thumb_cmc) < 1e-6 or
            np.linalg.norm(index_mcp) < 1e-6 or np.linalg.norm(pinky_mcp) < 1e-6):
            return None
        
        # Compute palm axes
        v1 = index_mcp - wrist
        v2 = pinky_mcp - wrist
        palm_normal = _safe_normalize(np.cross(v1, v2), (0.0, 0.0, 1.0))
        
        mid_top = (wrist + thumb_cmc) / 2.0
        mid_bottom = (index_mcp + pinky_mcp) / 2.0
        vertical = _safe_normalize(mid_bottom - mid_top, (0.0, 1.0, 0.0))
        
        palm_x = _safe_normalize(np.cross(vertical, palm_normal), (1.0, 0.0, 0.0))
        palm_y = vertical
        palm_z = palm_normal
        
        return {
            'palm_x': palm_x,
            'palm_y': palm_y,
            'palm_z': palm_z,
        }
        
    except Exception as e:
        print("Error computing orientation vectors: {0}".format(e))
        return None


def close():
    """Close visualization window."""
    global _window_open, _fig
    try:
        if _fig is not None:
            plt.close(_fig)
            _fig = None
        _window_open = False
    except Exception:
        pass