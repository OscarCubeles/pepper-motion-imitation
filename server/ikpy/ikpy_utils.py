import cv2
import torch
import numpy as np
import os
import sys
import argparse
import time
from pathlib import Path
from server.common import settings
from ikpy.chain import Chain
import json

from server.common import pose_mapping as pose_map


def _resolve_workspace_calibration_path(json_path):
    if json_path == "metrabs_workspace.json":
        return str(settings.IKPY_WORKSPACE_PATH)

    path = Path(json_path)
    if not path.is_absolute():
        path = Path(__file__).resolve().parent / path
    return str(path)


def load_metrabs_calibration(
    json_path="metrabs_workspace.json"
):
    """
    Returns calibration dictionary or None.
    """

    json_path = _resolve_workspace_calibration_path(json_path)
    if not os.path.exists(json_path):
        return None

    with open(json_path, "r") as f:
        return json.load(f)

def load_pepper_chains(script_dir=None):
    """Load Pepper robot arm chains from the shared resource directory.

    ``script_dir`` is retained for compatibility with older callers, but paths
    are resolved centrally so loading does not depend on the current working
    directory or the caller's location.
    """
    resources_dir = settings.PEPPER_RESOURCES_DIR
    left_arm = Chain.from_json_file(str(resources_dir / "pepper_left_arm.json"))
    right_arm = Chain.from_json_file(str(resources_dir / "pepper_right_arm.json"))

    print("Pepper left arm joints:")

    print(left_arm.links)
    fk = left_arm.forward_kinematics([0]*len(left_arm.links))
    print("Pepper left arm fk:")
    print(fk[:3,3])

    for i, link in enumerate(left_arm.links):
        print(i, link.name)


    return left_arm, right_arm


def compute_ik_from_metrabs_coordinate_adjusted_2(
    metrabs_pose_mm,
    left_arm_chain,
    right_arm_chain,
    arm='right',
    joint_indices=None,
    calibration_file="metrabs_workspace.json"
):
    """
    Convert MeTRAbs wrist coordinates into the calibrated IKPy
    workspace and compute inverse kinematics.

    MeTRAbs:
        X: left(+), right(-)
        Y: down(+), up(-)
        Z: depth

    IKPy:
        X: front/back
        Y: lateral
        Z: up/down
    """

    if joint_indices is None:
        raise ValueError("joint_indices must be provided")

    calibration = load_metrabs_calibration(
        calibration_file
    )

    if calibration is None:
        raise RuntimeError(
            "IKPy workspace calibration not found at {}. "
            "Generate it with `python -m server.ikpy.run_ikpy_no_client`.".format(
                _resolve_workspace_calibration_path(calibration_file)
            )
        )

    IKPY_WORKSPACE = {
        "right": {
            "x": (-0.20, 0.35),
            "y": (-0.52, 0.025),
            "z": (-0.32, 0.50)
        },
        "left": {
            "x": (-0.20, 0.35),
            "y": (-0.025, 0.50),
            "z": (-0.32, 0.50)
        }
    }

    wrist_key = (
        "rwri_smpl"
        if arm == "right"
        else "lwri_smpl"
    )

    wrist_idx = joint_indices[wrist_key]
    thor_idx = joint_indices["thor_smpl"]

    if wrist_idx is None or thor_idx is None:
        raise ValueError(
            f"Could not find wrist or thor_smpl joints for {arm} arm"
        )

    # MeTRAbs positions in meters
    wrist_m = metrabs_pose_mm[wrist_idx] / 1000.0
    thor_m = metrabs_pose_mm[thor_idx] / 1000.0

    # Thorax becomes origin
    dx, dy, dz = wrist_m - thor_m

    arm_cal = calibration[arm]
    workspace = IKPY_WORKSPACE[arm]

    # -------------------------
    # Coordinate transformation
    # -------------------------
    #
    # MeTRAbs -> IKPy
    #
    # IK X = -dz
    # IK Y =  dx
    # IK Z = -dy
    #

    ik_x = map_range(
        -dz,
        -arm_cal["dz_max"],
        -arm_cal["dz_min"],
        workspace["x"][0],
        workspace["x"][1]
    )

    ik_y = map_range(
        dx,
        arm_cal["dx_min"],
        arm_cal["dx_max"],
        workspace["y"][0],
        workspace["y"][1]
    )

    ik_z = map_range(
        -dy,
        -arm_cal["dy_max"],
        -arm_cal["dy_min"],
        workspace["z"][0],
        workspace["z"][1]
    )

    wrist_target = np.array([
        ik_x,
        ik_y,
        ik_z
    ])

    frame_target = np.eye(4)
    frame_target[:3, 3] = wrist_target

    chain = (
        right_arm_chain
        if arm == "right"
        else left_arm_chain
    )

    ik_result = chain.inverse_kinematics_frame(
        frame_target,
        optimizer="scalar"
    )

    return ik_result, chain


def compute_ik_from_wrist_thor_coordinate_adjusted(
    wrist_m,
    thor_m,
    left_arm_chain,
    right_arm_chain,
    arm='right',
    joint_indices=None,
    calibration_file="metrabs_workspace.json"
):
    """
    Convert MeTRAbs wrist coordinates into the calibrated IKPy
    workspace and compute inverse kinematics.

    MeTRAbs:
        X: left(+), right(-)
        Y: down(+), up(-)
        Z: depth

    IKPy:
        X: front/back
        Y: lateral
        Z: up/down
    """

    if joint_indices is None:
        raise ValueError("joint_indices must be provided")

    calibration = load_metrabs_calibration(
        calibration_file
    )

    if calibration is None:
        raise RuntimeError(
            "IKPy workspace calibration not found at {}. "
            "Generate it with `python -m server.ikpy.run_ikpy_no_client`.".format(
                _resolve_workspace_calibration_path(calibration_file)
            )
        )

    IKPY_WORKSPACE = {
        "right": {
            "x": (-0.20, 0.35),
            "y": (-0.52, 0.025),
            "z": (-0.32, 0.50)
        },
        "left": {
            "x": (-0.20, 0.35),
            "y": (-0.025, 0.50),
            "z": (-0.32, 0.50)
        }
    }


    # Thorax becomes origin
    dx, dy, dz = wrist_m - thor_m

    arm_cal = calibration[arm]
    workspace = IKPY_WORKSPACE[arm]

    # -------------------------
    # Coordinate transformation
    # -------------------------
    #
    # MeTRAbs -> IKPy
    #
    # IK X = -dz
    # IK Y =  dx
    # IK Z = -dy
    #

    ik_x = map_range(
        -dz,
        -arm_cal["dz_max"],
        -arm_cal["dz_min"],
        workspace["x"][0],
        workspace["x"][1]
    )

    ik_y = map_range(
        dx,
        arm_cal["dx_min"],
        arm_cal["dx_max"],
        workspace["y"][0],
        workspace["y"][1]
    )

    ik_z = map_range(
        -dy,
        -arm_cal["dy_max"],
        -arm_cal["dy_min"],
        workspace["z"][0],
        workspace["z"][1]
    )

    wrist_target = np.array([
        ik_x,
        ik_y,
        ik_z
    ])

    frame_target = np.eye(4)
    frame_target[:3, 3] = wrist_target

    chain = (
        right_arm_chain
        if arm == "right"
        else left_arm_chain
    )

    ik_result = chain.inverse_kinematics_frame(
        frame_target,
        optimizer="scalar"
    )

    return ik_result, chain


def map_range(
    value,
    src_min,
    src_max,
    dst_min,
    dst_max
):
    """
    Affine mapping from one interval to another.
    """

    if abs(src_max - src_min) < 1e-8:
        return (dst_min + dst_max) / 2

    normalized = (
        (value - src_min)
        / (src_max - src_min)
    )

    normalized = np.clip(normalized, 0.0, 1.0)

    return (
        dst_min
        + normalized * (dst_max - dst_min)
    )


def load_metrabs_model():
    """Load Metrabs inference model and calibration using pose_mapping utilities."""
    try:
        # Use pose_mapping's loader which handles CUDA device setup correctly
        intrinsic_matrix, distortion_coeffs, model, joint_names, joint_edges = pose_map._load_metrabs_model()
        return model, intrinsic_matrix, distortion_coeffs, joint_names, joint_edges
    except Exception as e:
        print(f"Error loading Metrabs model: {e}")
        sys.exit(1)

def get_joint_indices(model, skeleton_name):
    joint_names = model.per_skeleton_joint_names[skeleton_name]

    indices = {
        str(name): idx
        for idx, name in enumerate(joint_names)
    }

    return indices

def get_joint_indices_from_edges(joint_names):
    indices = {
        str(name): idx
        for idx, name in enumerate(joint_names)
    }

    return indices



def update_metrabs_calibration(
    metrabs_pose_mm,
    joint_indices,
    json_path="metrabs_workspace.json"
):
    """
    Updates min/max wrist-thorax distances and stores them to disk.
    """

    json_path = _resolve_workspace_calibration_path(json_path)

    try:
        with open(json_path, "r") as f:
            data = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):

        data = {
            "right": {
                "dx_min": float("inf"),
                "dx_max": float("-inf"),
                "dy_min": float("inf"),
                "dy_max": float("-inf"),
                "dz_min": float("inf"),
                "dz_max": float("-inf")
            },
            "left": {
                "dx_min": float("inf"),
                "dx_max": float("-inf"),
                "dy_min": float("inf"),
                "dy_max": float("-inf"),
                "dz_min": float("inf"),
                "dz_max": float("-inf")
            }
        }

    thor_idx = joint_indices["thor_smpl"]

    thor = metrabs_pose_mm[thor_idx] / 1000.0

    for arm, wrist_key in [
        ("right", "rwri_smpl"),
        ("left", "lwri_smpl")
    ]:

        wrist_idx = joint_indices[wrist_key]

        wrist = metrabs_pose_mm[wrist_idx] / 1000.0

        dx, dy, dz = wrist - thor

        data[arm]["dx_min"] = min(data[arm]["dx_min"], float(dx))
        data[arm]["dx_max"] = max(data[arm]["dx_max"], float(dx))

        data[arm]["dy_min"] = min(data[arm]["dy_min"], float(dy))
        data[arm]["dy_max"] = max(data[arm]["dy_max"], float(dy))

        data[arm]["dz_min"] = min(data[arm]["dz_min"], float(dz))
        data[arm]["dz_max"] = max(data[arm]["dz_max"], float(dz))

    with open(json_path, "w") as f:
        json.dump(data, f, indent=4)


def get_controllable_joints(chain, ik_result):
    """Extract only controllable joints from IK result.
    
    Filters out fixed joints (no bounds or infinite bounds).
    
    Args:
        chain: ikpy Chain
        ik_result: Full IK result array
        
    Returns:
        tuple: (filtered_angles, filtered_names, filtered_indices)
    """
    controllable_angles = []
    controllable_names = []
    controllable_indices = []
    
    for idx, link in enumerate(chain.links):
        if idx >= len(ik_result):
            continue
        
        # Check if this is a controllable joint (has bounds and not infinity)
        is_controllable = (
            link.bounds is not None and 
            link.bounds[0] != float('-inf') and 
            link.bounds[1] != float('inf')
        )
        
        if is_controllable:
            controllable_angles.append(ik_result[idx])
            controllable_names.append(link.name)
            controllable_indices.append(idx)
    
    return controllable_angles, controllable_names, controllable_indices


def draw_skeleton_keypoints(
    frame,
    poses2d,
    joint_edges,
    joint_names,
    keypoint_radius=5,
    line_color=(0, 255, 0),
    point_color=(0, 0, 255)
):
    """Draw 2D skeleton keypoints and connections on frame."""

    # Convert GPU tensor to numpy if needed
    if hasattr(poses2d, 'cpu'):
        poses2d = poses2d.cpu().numpy()
    elif not isinstance(poses2d, np.ndarray):
        poses2d = np.array(poses2d)

    frame_h, frame_w = frame.shape[:2]

    # Joints to highlight in blue
    highlighted_joints = {
        "lwri_smpl",
        "rwri_smpl",
        "thor_smpl"
    }

    # Draw skeleton lines
    if joint_edges is not None:
        for edge in joint_edges:
            i, j = edge

            if i < len(poses2d[0]) and j < len(poses2d[0]):
                pt1 = poses2d[0, i].astype(int)
                pt2 = poses2d[0, j].astype(int)

                if (
                    0 <= pt1[0] < frame_w and 0 <= pt1[1] < frame_h and
                    0 <= pt2[0] < frame_w and 0 <= pt2[1] < frame_h
                ):
                    cv2.line(frame, tuple(pt1), tuple(pt2), line_color, 2)

    # Draw keypoint circles
    for joint_idx in range(poses2d[0].shape[0]):
        pt = poses2d[0, joint_idx].astype(int)

        if 0 <= pt[0] < frame_w and 0 <= pt[1] < frame_h:

            joint_name = str(joint_names[joint_idx]).lower()

            if joint_name in highlighted_joints:
                color = (255, 0, 0)  # Blue in BGR
            else:
                color = point_color

            cv2.circle(frame, tuple(pt), keypoint_radius, color, -1)
