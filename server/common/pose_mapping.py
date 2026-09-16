"""
Pose mapping and transformation module.

Handles coordinate transformations, alignment of hand points to body joints,
and conversion of 3D pose data to the standardized 20-joint output format
with hand orientation vectors.
"""

import os
import numpy as np
import torch
from server.common import settings
from server.common import hand_orientation
from server.common import websocket_client as clientws
from server.common.calibration_utils import load_metrabs_calibration
from server.common.metrabs_pytorch.inference import metrabsInference
from server.common.hand_keypoints import _get_hand_joint_indices, _open_hand_landmarker, HandDetectionWorker


def _get_metrabs_dir():
    """
    Get the absolute path to the Metrabs model directory.
    
    Returns:
        str: Absolute path to metrabs_eff2l_384px_800k_28ds_pytorch directory
    """
    return str(settings.METRABS_MODEL_DIR)


def setup_metrabs():
    """Initialize Metrabs model, hand detection, and WebSocket server.
    
    Returns:
        tuple: (model, joint_names, joint_edges, suppress_joint_indices, 
                wrist_joint_indices, hand_worker, hand_landmarker, server, 
                server_thread, intrinsic_matrix, distortion_coeffs)
    """
    # Setup model and calibration
    intrinsic_matrix, distortion_coeffs, model, joint_names, joint_edges = _load_metrabs_model()
    
    # Setup joint indices
    suppress_joint_indices, wrist_joint_indices = _setup_metrabs_joints(joint_names)
    
    # Setup hand detection and server
    hand_worker, hand_landmarker, server, server_thread = _setup_hand_and_server()
    
    return model, joint_names, joint_edges, suppress_joint_indices, wrist_joint_indices, hand_worker, hand_landmarker, server, server_thread, intrinsic_matrix, distortion_coeffs


def _load_metrabs_model(skeleton=settings.SKELETON):
    """Load Metrabs model and camera calibration.
    
    Returns:
        tuple: (intrinsic_matrix, distortion_coeffs, model, joint_names, joint_edges)
    """
    intrinsic_matrix, distortion_coeffs = load_metrabs_calibration()
    model_dir = _get_metrabs_dir()
    checkpoint_path = os.path.join(model_dir, "ckpt.pt")
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(
            "MetrAbs checkpoint not found. Expected the untracked model file at: "
            + checkpoint_path
        )
    metrabs_inference_model = metrabsInference.metrabs_inference(model_dir)
    model = metrabs_inference_model.load_model()
    joint_names = model.per_skeleton_joint_names[skeleton]
    joint_edges = model.per_skeleton_joint_edges[skeleton].cpu().numpy()
    
    return intrinsic_matrix, distortion_coeffs, model, joint_names, joint_edges


def _setup_metrabs_joints(joint_names):
    """Extract joint indices and hand suppression info from model.
    
    Args:
        joint_names: List of joint names from model
        
    Returns:
        tuple: (suppress_joint_indices, wrist_joint_indices)
    """
    suppress_joint_indices = _get_hand_joint_indices(joint_names)
    wrist_joint_indices = _get_wrist_joint_indices(joint_names)
    
    if "Left" not in wrist_joint_indices or "Right" not in wrist_joint_indices:
        print(f"[warn] Could not resolve both wrist joints for {settings.SKELETON}: {wrist_joint_indices}")
    
    return suppress_joint_indices, wrist_joint_indices


def _setup_hand_and_server():
    """Setup hand detection worker and WebSocket server.
    
    Returns:
        tuple: (hand_worker, hand_landmarker, server, server_thread)
    """
    hand_landmarker = _open_hand_landmarker()
    hand_worker = HandDetectionWorker(hand_landmarker)
    server, server_thread = clientws._create_server(host="127.0.0.1", port=8080)
    
    return hand_worker, hand_landmarker, server, server_thread


def _to_cpu_numpy(tensor):
    """Convert GPU tensor to CPU numpy array.
    
    Performs non-blocking transfer to CPU and syncs CUDA stream.
    
    Args:
        tensor: GPU tensor or None
        
    Returns:
        Numpy array or None
    """
    if tensor is None:
        return None
    cpu_tensor = tensor.detach().to(device="cpu", non_blocking=True)
    torch.cuda.current_stream().synchronize()
    return cpu_tensor.numpy()


def _detect_poses(model, image, intrinsic_matrix, distortion_coeffs, skeleton=settings.SKELETON):
    """Run pose detection inference on an image.
    
    Args:
        model: Metrabs model instance
        image: Input image as numpy array (BGR)
        intrinsic_matrix: Camera intrinsic matrix
        distortion_coeffs: Camera distortion coefficients
        skeleton: Skeleton type to use for pose detection
    Returns:
        Prediction dict with 'poses3d', 'poses2d' keys
    """
    with torch.inference_mode(), torch.device("cuda"):
        image_pt = torch.from_numpy(image).permute(2, 0, 1).cuda(non_blocking=True)
        pred = model.detect_poses(
            image_pt,
            intrinsic_matrix=intrinsic_matrix,
            distortion_coeffs=distortion_coeffs,
            detector_threshold=settings.DETECTOR_THRESHOLD,
            detector_nms_iou_threshold=settings.DETECTOR_NMS_IOU,
            max_detections=settings.MAX_DETECTIONS,
            skeleton=skeleton,
            num_aug=settings.NUM_AUG,
            antialias_factor=settings.ANTIALIAS_FACTOR,
            internal_batch_size=settings.INTERNAL_BATCH_SIZE,
            average_aug=settings.AVERAGE_AUG,
            suppress_implausible_poses=settings.SUPPRESS_IMPLAUSIBLE_POSES,
            detector_flip_aug=settings.DETECTOR_FLIP_AUG,
        )
    return pred


def _start_async_pose_transfer(poses3d_gpu, current_frame_id, frame_ready_ts, copy_stream):
    """Start asynchronous GPU→CPU transfer of pose data.
    
    Initiates non-blocking transfer using separate CUDA stream and pinned memory
    for optimal performance. Synchronization is deferred to later.
    
    Args:
        poses3d_gpu: GPU tensor with 3D poses
        current_frame_id: Frame identifier
        frame_ready_ts: Timestamp when frame was ready
        copy_stream: CUDA stream for copying
        
    Returns:
        dict with 'cpu_tensor', 'done_event', 'src_tensor', 'frame_id', 'frame_ready_ts'
    """
    # OP-2: start async GPU→CPU transfer (sync deferred below)
    pose3d_first_gpu = poses3d_gpu[0].detach().contiguous()
    pose3d_first_cpu = torch.empty(
        pose3d_first_gpu.shape,
        dtype=pose3d_first_gpu.dtype,
        device="cpu",
        pin_memory=True,
    )
    infer_done_event = torch.cuda.Event()
    copy_done_event = torch.cuda.Event()
    infer_done_event.record(torch.cuda.current_stream())
    with torch.cuda.stream(copy_stream):
        copy_stream.wait_event(infer_done_event)
        pose3d_first_cpu.copy_(pose3d_first_gpu, non_blocking=True)
        copy_done_event.record(copy_stream)
    
    return {
        "cpu_tensor": pose3d_first_cpu,
        "done_event": copy_done_event,
        "src_tensor": pose3d_first_gpu,
        "frame_id": current_frame_id,
        "frame_ready_ts": frame_ready_ts,
    }


def _safe_normalize(vec):
    """
    Safely normalize a vector with a small threshold.
    
    Args:
        vec: Input vector to normalize
        
    Returns:
        Normalized vector or None if norm is too small
    """
    norm = float(np.linalg.norm(vec))
    if norm <= 1e-8:
        return None
    return vec / norm


def _get_wrist_joint_indices(joint_names):
    """
    Extract left and right wrist joint indices from skeleton joint names.
    
    Args:
        joint_names: List of joint name strings from model
        
    Returns:
        dict with 'Left' and/or 'Right' keys mapping to joint indices
    """
    wrist_joint_indices = {}
    for joint_idx, joint_name in enumerate(joint_names):
        name = str(joint_name).lower()
        if "wrist" not in name and "wri" not in name:
            continue
        if ("left" in name) or ("lwri" in name) or name.startswith("l"):
            wrist_joint_indices.setdefault("Left", int(joint_idx))
        elif ("right" in name) or ("rwri" in name) or name.startswith("r"):
            wrist_joint_indices.setdefault("Right", int(joint_idx))
    return wrist_joint_indices


def _anchor_hand_points_to_metrabs_wrist(
    pose3d_metrabs,
    hand_points_world_3d: dict[str, dict[int, np.ndarray]],
    wrist_joint_indices: dict[str, int],
    units_to_meters: float = settings.METRABS_UNITS_TO_METERS,
) -> dict[str, dict[int, np.ndarray]]:
    """
    Align hand keypoints to the Metrabs body wrist joint position.
    
    Converts hand keypoints from MediaPipe world coordinates (which are 
    relative to the hand) to body coordinates by anchoring them to the
    Metrabs-detected wrist position.
    
    Args:
        pose3d_metrabs: 3D pose array from Metrabs model (shape: [num_joints, 3])
        hand_points_world_3d: Hand keypoint dict with 'Left'/'Right' keys,
                             each containing {landmark_id: [x, y, z]} arrays
        wrist_joint_indices: Dict mapping 'Left'/'Right' to wrist joint indices
        units_to_meters: Conversion factor from pose units to meters
        
    Returns:
        dict with anchored hand points in body coordinate system
    """
    if pose3d_metrabs is None:
        return {}
    pose3d_arr = np.asarray(pose3d_metrabs, dtype=np.float32)
    if pose3d_arr.ndim != 2 or pose3d_arr.shape[0] == 0:
        return {}
    if units_to_meters <= 0:
        return {}

    meters_to_pose_units = float(1.0 / units_to_meters)
    anchored_points: dict[str, dict[int, np.ndarray]] = {}
    
    for label, label_points in hand_points_world_3d.items():
        if not label_points or 0 not in label_points:
            continue

        wrist_joint_idx = wrist_joint_indices.get(label)
        if wrist_joint_idx is None or wrist_joint_idx >= pose3d_arr.shape[0]:
            continue

        # Get wrist positions in both coordinate systems
        metrabs_wrist = np.asarray(pose3d_arr[wrist_joint_idx, :3], dtype=np.float32)
        mp_wrist = np.asarray(label_points[0], dtype=np.float32).reshape(-1)
        
        # Validate data
        if mp_wrist.size < 3 or not np.isfinite(metrabs_wrist).all() or not np.isfinite(mp_wrist[:3]).all():
            continue

        # Align each hand point to body coordinate system
        anchored_label_points: dict[int, np.ndarray] = {}
        for point_idx, point_world in label_points.items():
            point_world_arr = np.asarray(point_world, dtype=np.float32).reshape(-1)
            if point_world_arr.size < 3 or not np.isfinite(point_world_arr[:3]).all():
                continue
                
            # Transform: anchor hand point relative to body wrist
            delta_pose_units = (point_world_arr[:3] - mp_wrist[:3]) * meters_to_pose_units
            anchored_label_points[int(point_idx)] = (metrabs_wrist + delta_pose_units).astype(np.float32)

        if anchored_label_points:
            anchored_points[str(label)] = anchored_label_points

    return anchored_points


def _map_pose_20(pose3d_cpu: np.ndarray, hand_points_3d: dict[str, dict[int, np.ndarray]], pose_out: np.ndarray):
    """
    Map 3D pose to standardized 20-joint output format with hand data.
    
    Converts a Metrabs pose to a 20-joint output with:
    - Indices 0-9: Body joints (mapped from Metrabs)
    - Indices 10-11: Hand tips (right, left)
    - Indices 12-15: Right hand orientation keypoints (wrist, thumb, index, pinky)
    - Indices 16-19: Left hand orientation keypoints (wrist, thumb, index, pinky)
    - Indices 20-22: Right hand palm vectors (x, y, z components)
    - Indices 23-25: Left hand palm vectors (x, y, z components)
    
    Args:
        pose3d_cpu: Input 3D pose from Metrabs (shape: [num_joints, 3])
        hand_points_3d: Hand keypoints dict with 'Left'/'Right' keys
        pose_out: Output array to fill (shape: [26, 3])
    """
    pose_out.fill(0.0)
    
    


    # Map body joints: indices 0-9
    pose_out[settings.TARGET_BODY_IDXS] = pose3d_cpu[settings.SOURCE_BODY_IDXS]
    pose_out *= 0.001  # Convert to meters

    # Stream fingertip and orientation keypoints for each hand
    # Indices 10-11 (tips) and 12-15 (right) / 16-19 (left)
    right_hand = hand_points_3d.get("Right")
    left_hand = hand_points_3d.get("Left")
    
    if right_hand:
        # Right hand tip (index 10)
        if settings.HAND_TIP_ID in right_hand:
            pose_out[settings.RIGHT_TIP_OUT_IDX, :] = right_hand[settings.HAND_TIP_ID] * 0.001
        # Right hand orientation keypoints (indices 12-15)
        for source_idx, out_idx in zip(settings.HAND_ORIENTATION_IDS, settings.RIGHT_ORIENTATION_OUT_IDXS):
            if source_idx in right_hand:
                pose_out[out_idx, :] = right_hand[source_idx] * 0.001
                
    if left_hand:
        # Left hand tip (index 11)
        if settings.HAND_TIP_ID in left_hand:
            pose_out[settings.LEFT_TIP_OUT_IDX, :] = left_hand[settings.HAND_TIP_ID] * 0.001
        # Left hand orientation keypoints (indices 16-19)
        for source_idx, out_idx in zip(settings.HAND_ORIENTATION_IDS, settings.LEFT_ORIENTATION_OUT_IDXS):
            if source_idx in left_hand:
                pose_out[out_idx, :] = left_hand[source_idx] * 0.001
    
    # Compute and append palm vectors (indices 20-25)
    # These represent the normal vector to the palm plane for hand orientation
    palm_vectors = hand_orientation.compute_hand_palm_vectors(hand_points_3d)
    
    # Right hand palm vectors (indices 20, 21, 22 for x, y, z)
    if "Right" in palm_vectors:
        palm_x, palm_y, palm_z = palm_vectors["Right"]
        pose_out[20, :] = palm_x
        pose_out[21, :] = palm_y
        pose_out[22, :] = palm_z
    
    # Left hand palm vectors (indices 23, 24, 25 for x, y, z)
    if "Left" in palm_vectors:
        palm_x, palm_y, palm_z = palm_vectors["Left"]
        pose_out[23, :] = palm_x
        pose_out[24, :] = palm_y
        pose_out[25, :] = palm_z


def _get_hand_orientation_labels(hand_points_3d: dict[str, dict[int, np.ndarray]]) -> dict:
    """Get discrete orientation labels for both hands.
    
    Args:
        hand_points_3d: Hand keypoints dict with 'Left'/'Right' keys
        
    Returns:
        dict with 'Right' and 'Left' keys containing:
            - 'primary': dominant orientation label
            - 'x_axis': left/right label
            - 'y_axis': front/back label  
            - 'z_axis': up/down label
    """
    if not hand_points_3d:
        return {}
    
    orientation_labels = hand_orientation.compute_hand_orientation_labels(hand_points_3d)
    
    # Extract the 4 labels per hand (primary, x_axis, y_axis, z_axis)
    result = {}
    for hand in ["Right", "Left"]:
        if hand in orientation_labels:
            labels = orientation_labels[hand]
            result[hand] = {
                "primary": labels.get("primary", "UNKNOWN"),
                "x_axis": labels.get("x_label", "UNKNOWN"),
                "y_axis": labels.get("y_label", "UNKNOWN"),
                "z_axis": labels.get("z_label", "UNKNOWN"),
            }
    
    return result


def _compute_discrete_direction_label(vec: np.ndarray):
    """Classify a 3D direction vector into one of the dominant axis labels."""
    vec = np.asarray(vec, dtype=np.float32).reshape(-1)
    if vec.size < 3:
        return None

    norm = float(np.linalg.norm(vec[:3]))
    if norm < 1e-6:
        return None

    vec = vec[:3] / norm
    ax, ay, az = np.abs(vec)

    if ax >= ay and ax >= az:
        return "left" if vec[0] > 0 else "right"
    if ay >= ax and ay >= az:
        return "down" if vec[1] > 0 else "up"
    return "backward" if vec[2] > 0 else "forward"


def prepare_full_pose_data(pose_keypoints: np.ndarray, hand_orientation_labels: dict) -> dict:
    """Prepare full pose data with base64-encoded keypoints and orientation labels.
    
    Combines pose keypoints (encoded as base64) with discrete orientation labels
    for transmission over network.
    
    Args:
        pose_keypoints: (26, 3) array of pose keypoints
        hand_orientation_labels: dict with orientation labels from _get_hand_orientation_labels
        
    Returns:
        JSON-serializable dict with:
            - 'pose_keypoints': base64-encoded pose array
            - 'hand_orientation': dict with 'Right' and 'Left' orientation labels
    """
    import base64
    
    # Encode pose as base64
    pose_bytes = pose_keypoints.astype(np.float32).tobytes()
    pose_base64 = base64.b64encode(pose_bytes).decode('utf-8')
    
    return {
        "pose_keypoints": pose_base64,
        "hand_orientation": hand_orientation_labels,
    }


def compute_forearm_direction(pose3d_cpu: np.ndarray, hand: str):
    """
    Returns:
        label (one of: left/right/up/down/forward/backward) or None
    """

    if hand == "left":
        elbow_idx, wrist_idx = 6, 7
    elif hand == "right":
        elbow_idx, wrist_idx = 3, 4
    else:
        return None

    elbow = pose3d_cpu[elbow_idx]
    wrist = pose3d_cpu[wrist_idx]
    return _compute_discrete_direction_label(wrist - elbow)





def compute_upper_arm_direction(pose3d_cpu: np.ndarray, hand: str):
    """
    Returns:
        label (one of: left/right/up/down/forward/backward) or None
    """

    if hand == "left":
        shoulder_idx, elbow_idx = 5, 6
    elif hand == "right":
        shoulder_idx, elbow_idx = 2, 3
    else:
        return None

    shoulder = pose3d_cpu[shoulder_idx]
    elbow = pose3d_cpu[elbow_idx]
    return _compute_discrete_direction_label(elbow - shoulder)
