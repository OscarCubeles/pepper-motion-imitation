"""Shared visualization module for the Exercise Motivation System pose servers.

This module provides the ``PoseVisualizer`` class used by all server variants
(MetrAbs webcam, ZED camera, and their MediaPipe-augmented counterparts) to
render a real-time debug UI consisting of:

  * A 2D camera-feed panel (left) with overlaid skeleton keypoints, edges,
    bounding boxes / polygons, and optional extra hand landmarks.
  * A 3D matplotlib scatter-plot panel (right) showing the same skeleton in
    an intuitive upright coordinate system together with depth/distance info.
  * A title bar showing source name, visualization FPS, and inference FPS.

The module is intentionally **not** imported in the default streaming path.
Visualization is a debugging aid and is disabled by default in the maintained
server scripts (``ENABLE_VISUALIZATION = False``) to keep latency low.

Key design choices
------------------
* All helper functions are module-private (prefixed with ``_``) so that
  callers only interact with ``PoseVisualizer`` and ``VisualizationResult``.
* Input data is accepted in flexible forms (PyTorch tensors, NumPy arrays,
  ZED SDK enum values, plain lists) and uniformly coerced to NumPy internally.
* The 3D view converts camera-convention coordinates (Y-down, Z-forward) to
  display-convention coordinates (Y-depth, Z-up) for intuitive viewing.
* Bounding boxes can be supplied as ``[x, y, w, h]`` rectangles **or**
  arbitrary polygons (e.g. ZED body bounding polygons).
"""

import time
from dataclasses import dataclass
from collections.abc import Iterable, Sequence
import re

import cv2
import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg  # Off-screen Agg renderer
from matplotlib.figure import Figure

# ---------------------------------------------------------------------------
# Angles panel rendering
# ---------------------------------------------------------------------------


def _render_angles_panel(human_angles: dict, width: int = 1200, height: int = 500, pose_feasibility: dict = None,
                         pepper_feasibility: dict = None, singularity_right: dict = None, singularity_left: dict = None, 
                         missing_keypoints: list = None, show_angle_values: bool = True, bg_color=(30, 30, 30)) -> np.ndarray:
    """
    Render panel showing pose status (feasibility + singularity), missing keypoints, and optionally joint angles.
    
    Args:
        human_angles: Dictionary with angle keys like "ElbowRoll_Left", "ShoulderRoll_Right", etc.
        width: Panel width in pixels
        height: Panel height in pixels
        pose_feasibility: Dict with "is_feasible" bool and "violations" list (human pose)
        pepper_feasibility: Dict with "is_feasible" bool and "violations" list (pepper pose)
        singularity_right: Dict with singularity info for right arm from check_singularity_poses()
        singularity_left: Dict with singularity info for left arm from check_singularity_poses()
        missing_keypoints: List of keypoint names that are missing/invalid
        show_angle_values: If True, display angle values. If False, show only feasibility + singularity status
        bg_color: BGR background color
        
    Returns:
        Image array of shape (height, width, 3)
    """
    # Create background
    panel = np.full((height, width, 3), bg_color, dtype=np.uint8)
    
    # Font settings
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.6
    font_color = (0, 255, 0)  # Green
    thickness = 1
    line_height = 28

    def _extract_joint_name(text: str) -> str:
        base = text.split(":", 1)[0].strip()
        if base.startswith("[") and "]" in base:
            base = base.split("]", 1)[1].strip()
        if not base:
            return "Unknown"
        return base
    
    def _extract_joint_name_singularity(text: str) -> str:
        """
        Extract joint name from warning strings like:
        'RIGHT: ElbowRoll at 15.8° (near singularity, within 20.0°)'
        """
        match = re.search(r":\s*([A-Za-z_]+)", text)
        if match:
            return match.group(1)

        return "Unknown"

    def _side_from_text(text: str) -> str | None:
        upper = text.upper()
        if "LEFT" in upper:
            return "left"
        if "RIGHT" in upper:
            return "right"
        return None

    # Header status line.
    y_header = 28
    if pose_feasibility is not None:
        is_feasible = pose_feasibility.get("is_feasible", False)
        status_text = "HUMAN: FEASIBLE" if is_feasible else "HUMAN: NOT FEASIBLE"
        status_color = (0, 255, 0) if is_feasible else (0, 0, 255)
        cv2.putText(panel, status_text, (10, y_header), font, font_scale + 0.2, status_color, thickness + 1)
    else:
        is_feasible = True
    
    # Add Pepper feasibility status next to human status
    if pepper_feasibility is not None:
        pepper_feasible = pepper_feasibility.get("is_feasible", False)
        pepper_status_text = "PEPPER: FEASIBLE" if pepper_feasible else "PEPPER: NOT FEASIBLE"
        pepper_status_color = (0, 255, 0) if pepper_feasible else (0, 0, 255)
        cv2.putText(panel, pepper_status_text, (350, y_header), font, font_scale + 0.2, pepper_status_color, thickness + 1)

    # Two-column layout anchors.
    column_gap = 20
    col_w = max((width - (column_gap * 3)) // 2, 200)
    left_x = column_gap
    right_x = left_x + col_w + column_gap
    y_start = 70

    cv2.putText(panel, "LEFT ARM", (left_x, y_start), font, font_scale + 0.15, (255, 255, 255), thickness + 1)
    cv2.putText(panel, "RIGHT ARM", (right_x, y_start), font, font_scale + 0.15, (255, 255, 255), thickness + 1)

    y_left = y_start + 28
    y_right = y_start + 28

    def _draw_section_title(x: int, y: int, title: str, color: tuple[int, int, int]) -> int:
        cv2.putText(panel, title, (x, y), font, font_scale + 0.05, color, thickness + 1)
        return y + 24

    def _draw_lines(x: int, y: int, lines: list[str], color: tuple[int, int, int], max_lines: int = 6) -> int:
        shown = lines[:max_lines]
        if not shown:
            cv2.putText(panel, "-", (x, y), font, font_scale - 0.05, (150, 150, 150), thickness)
            return y + line_height
        for line in shown:
            cv2.putText(panel, line, (x, y), font, font_scale - 0.05, color, thickness)
            y += line_height
        return y

    # Split violations by side and keep only joint names.
    left_violations = []
    right_violations = []
    #print(f"[DEBUG] pose_feasibility={pose_feasibility}\n\n")
    if pose_feasibility is not None:
        # New structure: violations is a dict with "left" and "right" keys
        violations_dict = pose_feasibility.get("violations", {})
        if isinstance(violations_dict, dict):
            # Extract joint names from left violations
            for violation in violations_dict.get("left", []):
                joint_name = _extract_joint_name(str(violation))
                left_violations.append(joint_name)
            # Extract joint names from right violations
            for violation in violations_dict.get("right", []):
                joint_name = _extract_joint_name(str(violation))
                right_violations.append(joint_name)
        else:
            # Fallback for old structure (flat list)
            for violation in violations_dict if isinstance(violations_dict, list) else []:
                side = _side_from_text(str(violation))
                joint_name = _extract_joint_name(str(violation))
                if side == "left" or "left" in joint_name.lower():
                    left_violations.append(joint_name)
                elif side == "right" or "right" in joint_name.lower():
                    right_violations.append(joint_name)

    y_left = _draw_section_title(left_x, y_left, "Violations", (0, 0, 255))
    y_left = _draw_lines(left_x, y_left, left_violations, (0, 0, 255)) + 8
    y_right = _draw_section_title(right_x, y_right, "Violations", (0, 0, 255))
    y_right = _draw_lines(right_x, y_right, right_violations, (0, 0, 255)) + 8

    # Split singularity warnings by side and keep only compact labels.
    left_sing = []
    right_sing = []
    if singularity_left is not None and singularity_left.get("has_singularity"):
        for warning in singularity_left.get("warnings", []):
            left_sing.append(_extract_joint_name_singularity(str(warning)))
    if singularity_right is not None and singularity_right.get("has_singularity"):
        for warning in singularity_right.get("warnings", []):
            right_sing.append(_extract_joint_name_singularity(str(warning)))

    y_left = _draw_section_title(left_x, y_left, "Singularities", (0, 165, 255))
    y_left = _draw_lines(left_x, y_left, left_sing, (0, 165, 255)) + 8
    y_right = _draw_section_title(right_x, y_right, "Singularities", (0, 165, 255))
    y_right = _draw_lines(right_x, y_right, right_sing, (0, 165, 255)) + 8

    # Only show angle values if enabled and feasible.
    #if not show_angle_values or not is_feasible:
    #    return panel

    left_angles = []
    right_angles = []
    for angle_name, angle_value in human_angles.items():
        if "ElbowYaw" in angle_name:
            continue
        if angle_value is None:
            continue
        text = f"{angle_name}: {angle_value:.1f}deg"
        if "Left" in angle_name:
            left_angles.append(text)
        elif "Right" in angle_name:
            right_angles.append(text)

    left_angles.sort()
    right_angles.sort()

   #y_left = _draw_section_title(left_x, y_left, "Angles", font_color)
   #y_left = _draw_lines(left_x, y_left, left_angles, font_color, max_lines=10)
   #y_right = _draw_section_title(right_x, y_right, "Angles", font_color)
   #y_right = _draw_lines(right_x, y_right, right_angles, font_color, max_lines=10)
    # Add missing keypoints section at the bottom
    #if missing_keypoints:
    #    # Determine y position for missing keypoints section (leave some room at bottom)
    #    y_bottom_start = max(y_left, y_right) + 32
    #    
    #    if y_bottom_start < height - 80:
    #        # Split missing keypoints by left/right
    #        left_missing = []
    #        right_missing = []
    #        for kp in missing_keypoints:
    #            kp_upper = str(kp).upper()
    #            # Extract just the keypoint name without the index
    #            kp_name = str(kp).split("(")[0] if "(" in str(kp) else str(kp)
    #            
    #            if "LEFT" in kp_upper:
    #                left_missing.append(kp_name)
    #            elif "RIGHT" in kp_upper:
    #                right_missing.append(kp_name)
    #            else:
    #                # If not clearly left or right, add to both sides
    #                left_missing.append(kp_name)
    #        
    #        #print(f"[DEBUG] left_missing={left_missing}, right_missing={right_missing}")
    #        y_left_bottom = _draw_section_title(left_x, y_bottom_start, "Missing Keypoints", (0, 0, 255))
    #        y_left_bottom = _draw_lines(left_x, y_left_bottom, left_missing, (0, 0, 255), max_lines=4)
    #        
    #        y_right_bottom = _draw_section_title(right_x, y_bottom_start, "Missing Keypoints", (0, 0, 255))
    #        y_right_bottom = _draw_lines(right_x, y_right_bottom, right_missing, (0, 0, 255), max_lines=4)
    #    else:
    #        print(f"[DEBUG] NOT ENOUGH SPACE for missing keypoints (y_bottom_start={y_bottom_start} >= height-80={height-80})")

    
    return panel


# ---------------------------------------------------------------------------
# Skeleton topology: BODY_38 (ZED Body Tracking default)
# ---------------------------------------------------------------------------
# Default BODY_38 edge list used when no custom joint connectivity is provided.
# Each tuple ``(start, end)`` defines one bone connecting two keypoint indices.
# This topology matches the ZED SDK BODY_38 format with 38 keypoints covering
# the full body, face, feet, and hand extremities.
BODY38_JOINT_EDGES = [
    (5, 4),  # NOSE -> NECK
    (4, 6),  # NECK -> LEFT_EYE
    (4, 7),  # NECK -> RIGHT_EYE
    (6, 8),  # LEFT_EYE -> LEFT_EAR
    (7, 9),  # RIGHT_EYE -> RIGHT_EAR
    (0, 1),  # PELVIS -> SPINE_1
    (1, 2),  # SPINE_1 -> SPINE_2
    (2, 3),  # SPINE_2 -> SPINE_3
    (3, 4),  # SPINE_3 -> NECK
    (4, 10),  # NECK -> LEFT_CLAVICLE
    (4, 11),  # NECK -> RIGHT_CLAVICLE
    (10, 12),  # LEFT_CLAVICLE -> LEFT_SHOULDER
    (12, 14),  # LEFT_SHOULDER -> LEFT_ELBOW
    (14, 16),  # LEFT_ELBOW -> LEFT_WRIST
    (11, 13),  # RIGHT_CLAVICLE -> RIGHT_SHOULDER
    (13, 15),  # RIGHT_SHOULDER -> RIGHT_ELBOW
    (15, 17),  # RIGHT_ELBOW -> RIGHT_WRIST
    (0, 18),  # PELVIS -> LEFT_HIP
    (0, 19),  # PELVIS -> RIGHT_HIP
    (18, 20),  # LEFT_HIP -> LEFT_KNEE
    (20, 22),  # LEFT_KNEE -> LEFT_ANKLE
    (22, 24),  # LEFT_ANKLE -> LEFT_BIG_TOE
    (22, 26),  # LEFT_ANKLE -> LEFT_SMALL_TOE
    (22, 28),  # LEFT_ANKLE -> LEFT_HEEL
    (19, 21),  # RIGHT_HIP -> RIGHT_KNEE
    (21, 23),  # RIGHT_KNEE -> RIGHT_ANKLE
    (23, 25),  # RIGHT_ANKLE -> RIGHT_BIG_TOE
    (23, 27),  # RIGHT_ANKLE -> RIGHT_SMALL_TOE
    (23, 29),  # RIGHT_ANKLE -> RIGHT_HEEL
    (16, 30),  # LEFT_WRIST -> LEFT_HAND_THUMB_4
    (16, 32),  # LEFT_WRIST -> LEFT_HAND_INDEX_1
    (16, 34),  # LEFT_WRIST -> LEFT_HAND_MIDDLE_4
    (16, 36),  # LEFT_WRIST -> LEFT_HAND_PINKY_1
    (17, 31),  # RIGHT_WRIST -> RIGHT_HAND_THUMB_4
    (17, 33),  # RIGHT_WRIST -> RIGHT_HAND_INDEX_1
    (17, 35),  # RIGHT_WRIST -> RIGHT_HAND_MIDDLE_4
    (17, 37),  # RIGHT_WRIST -> RIGHT_HAND_PINKY_1
]


# ---------------------------------------------------------------------------
# Internal helper functions
# ---------------------------------------------------------------------------


def _to_numpy_array(data, dtype=np.float32):
    """Convert arbitrary tensor-like data to a NumPy array.

    Handles PyTorch tensors (with ``.detach().cpu().numpy()`` chain), plain
    lists, and existing NumPy arrays.  Returns ``None`` when ``data`` is
    ``None`` so callers can propagate missing data without extra checks.
    """
    if data is None:
        return None
    if hasattr(data, "detach"):
        data = data.detach()
    if hasattr(data, "cpu"):
        data = data.cpu()
    if hasattr(data, "numpy"):
        data = data.numpy()
    return np.asarray(data, dtype=dtype)


def _letterbox(image: np.ndarray, target_w: int, target_h: int, fill_color: tuple[int, int, int] = (20, 20, 20)) -> np.ndarray:
    """Resize ``image`` to fit inside ``(target_w, target_h)`` while preserving
    aspect ratio, centering the result on a solid-color canvas.

    This is the standard "letterbox" technique used in computer-vision
    pipelines to avoid distortion when the source and target aspect ratios
    differ.  Empty borders are filled with ``fill_color`` (dark grey by
    default to match the dark UI theme).

    Parameters
    ----------
    image : np.ndarray
        Source BGR image (H x W x 3).
    target_w, target_h : int
        Desired output dimensions in pixels.
    fill_color : tuple[int, int, int]
        BGR color for the padding border.

    Returns
    -------
    np.ndarray
        Letterboxed image of shape ``(target_h, target_w, 3)``.
    """
    src_h, src_w = image.shape[:2]

    # Guard against degenerate inputs (empty frames, zero-sized targets).
    if src_h <= 0 or src_w <= 0 or target_w <= 0 or target_h <= 0:
        return np.zeros((max(target_h, 1), max(target_w, 1), 3), dtype=np.uint8)

    # Compute uniform scale factor that makes the image fit entirely inside
    # the target rectangle without cropping.
    scale = min(target_w / float(src_w), target_h / float(src_h))
    new_w = max(1, int(round(src_w * scale)))
    new_h = max(1, int(round(src_h * scale)))
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)

    # Create a solid-color canvas and paste the resized image centered.
    canvas = np.full((target_h, target_w, 3), fill_color, dtype=np.uint8)
    x_off = (target_w - new_w) // 2
    y_off = (target_h - new_h) // 2
    canvas[y_off:y_off + new_h, x_off:x_off + new_w] = resized
    return canvas


def _camera_to_display_axes(points_3d: np.ndarray) -> np.ndarray:
    """Re-map camera-convention 3D coordinates to a display-friendly system.

    Most cameras (webcam, ZED) use the convention:
        * X = left / right
        * Y = vertical, **down** is positive
        * Z = depth (forward away from the camera)

    For an intuitive 3D scatter plot where "up" looks upward on screen, we
    rearrange to:
        * X_display = X            (left / right unchanged)
        * Y_display = Z            (depth becomes the plot's Y axis)
        * Z_display = -Y           (height, flipped so up is positive)

    Parameters
    ----------
    points_3d : np.ndarray
        Array of shape ``(N, 3)`` in camera coordinates.

    Returns
    -------
    np.ndarray
        Array of shape ``(N, 3)`` in display coordinates.
    """
    pts = np.asarray(points_3d, dtype=np.float32)
    display_pts = np.empty_like(pts)
    display_pts[:, 0] = pts[:, 0]   # X stays as-is
    display_pts[:, 1] = pts[:, 2]   # camera Z -> display Y (depth)
    display_pts[:, 2] = -pts[:, 1]  # camera -Y -> display Z (height up)
    return display_pts


def _normalize_edges(joint_edges: Iterable[tuple[int, int]] | None) -> list[tuple[int, int]]:
    """Coerce an iterable of joint-edge pairs into a clean list of ``(int, int)``.

    This handles inputs where the edge indices might be ZED SDK enum values
    (which expose a ``.value`` attribute) or other non-int types.  Invalid
    entries are silently dropped so that callers don't need to sanitize
    upstream data.
    """
    if joint_edges is None:
        return []
    normalized: list[tuple[int, int]] = []
    for edge in joint_edges:
        if len(edge) != 2:
            continue
        start, end = edge
        # ZED SDK enum values expose a .value attribute; unwrap them.
        if hasattr(start, "value"):
            start = start.value
        if hasattr(end, "value"):
            end = end.value
        try:
            normalized.append((int(start), int(end)))
        except (TypeError, ValueError):
            continue
    return normalized


def _normalize_index_set(indices: Iterable[int] | None) -> set[int]:
    """Coerce an iterable of joint indices into a ``set[int]``.

    Used to build the set of suppressed (hidden) joint indices for the
    visualization.  Like ``_normalize_edges``, this handles ZED SDK enums
    and silently discards non-convertible values.
    """
    if indices is None:
        return set()
    normalized: set[int] = set()
    for idx in indices:
        # Unwrap enum values (e.g. ``sl.BODY_38_PARTS.NOSE.value``).
        if hasattr(idx, "value"):
            idx = idx.value
        try:
            normalized.add(int(idx))
        except (TypeError, ValueError):
            continue
    return normalized


def _default_edges_for_joint_count(joint_count: int) -> list[tuple[int, int]]:
    """Return a default skeleton edge list based on the number of keypoints.

    If the pose has 38 or more joints, we assume ZED BODY_38 format and
    return the full ``BODY38_JOINT_EDGES`` topology.  For smaller skeletons
    (e.g. the 10-joint or 12-joint compact Pepper output format) there is no
    built-in default, so an empty list is returned and no edges are drawn.
    """
    # BODY_38 is the default fallback for ZED-style 38-keypoint input.
    if joint_count >= 38:
        return list(BODY38_JOINT_EDGES)
    return []


def _to_pose_batch(poses) -> np.ndarray:
    """Normalize a pose array into a 3-D batch of shape ``(N, J, C)``.

    Accepts:
    * A single pose ``(J, C)`` -- reshaped to ``(1, J, C)``.
    * A batch ``(N, J, C)``.
    * A PyTorch tensor or ZED SDK object -- converted via ``_to_numpy_array``.
    * ``None`` or empty -- returns a sentinel ``(0, 0, 3)`` array.

    Where ``N`` = number of detected persons, ``J`` = joints, ``C`` = coords
    (typically 2 for 2D or 3 for 3D).
    """
    arr = _to_numpy_array(poses)
    if arr is None or arr.size == 0:
        return np.empty((0, 0, 3), dtype=np.float32)
    # If a single person pose was passed, wrap it in a batch dimension.
    if arr.ndim == 2:
        arr = arr.reshape(1, arr.shape[0], arr.shape[1])
    return arr.astype(np.float32, copy=False)


def _to_boxes_list(boxes) -> list | None:
    """Convert bounding-box data into a plain Python list of per-person boxes.

    Each element in the returned list corresponds to one detected person and
    may be:
    * A flat ``[x, y, w, h]`` rectangle (MetrAbs / YOLO style).
    * A polygon array of shape ``(V, 2)`` (ZED body bounding polygon).

    Returns ``None`` when no valid data is available.
    """
    if boxes is None:
        return None
    # Handle PyTorch tensors.
    if hasattr(boxes, "detach"):
        boxes = boxes.detach()
    if hasattr(boxes, "cpu"):
        boxes = boxes.cpu()
    if hasattr(boxes, "numpy"):
        boxes = boxes.numpy()

    if isinstance(boxes, np.ndarray):
        if boxes.size == 0:
            return None
        # Single box supplied as a flat 1-D array -> wrap in a list.
        if boxes.ndim == 1:
            return [boxes]
        # Multi-box array: one row per person.
        return [boxes[i] for i in range(len(boxes))]

    try:
        box_list = list(boxes)
    except TypeError:
        return None
    return box_list if box_list else None


def _is_valid_point2d(point: np.ndarray, width: int, height: int) -> bool:
    """Return ``True`` if ``point`` has finite x/y within the frame bounds.

    Used to skip drawing joints that fall outside the camera image or that
    contain ``NaN`` / ``Inf`` values (which happen when a keypoint is not
    detected by the pose estimator).
    """
    if point.shape[0] < 2:
        return False
    x, y = float(point[0]), float(point[1])
    if not np.isfinite(x) or not np.isfinite(y):
        return False
    return 0 <= x < width and 0 <= y < height


def _compute_bbox_from_points(points_2d: np.ndarray, width: int, height: int) -> tuple[int, int, int, int] | None:
    """Compute a tight axis-aligned bounding box from 2D keypoints.

    This is the fallback used when no bounding box is supplied by the
    detection model.  It encloses all finite keypoints of the tracked person
    and clips the result to the image dimensions.

    Returns
    -------
    tuple[int, int, int, int] or None
        ``(x_min, y_min, x_max, y_max)`` pixel coordinates, or ``None`` if
        no valid points exist.
    """
    if points_2d is None or points_2d.size == 0:
        return None
    # Only consider keypoints with finite coordinates.
    valid = np.isfinite(points_2d[:, :2]).all(axis=1)
    if not np.any(valid):
        return None
    pts = points_2d[valid, :2]
    # Compute tight bounding rectangle, clipped to image edges.
    x_min = max(0, int(np.floor(np.min(pts[:, 0]))))
    y_min = max(0, int(np.floor(np.min(pts[:, 1]))))
    x_max = min(width - 1, int(np.ceil(np.max(pts[:, 0]))))
    y_max = min(height - 1, int(np.ceil(np.max(pts[:, 1]))))
    if x_max <= x_min or y_max <= y_min:
        return None
    return x_min, y_min, x_max, y_max


def _normalize_polygon(box_entry) -> np.ndarray | None:
    """Try to interpret ``box_entry`` as a polygon with >= 3 vertices.

    The ZED SDK provides body bounding boxes as convex polygons (a list of
    ``(x, y)`` vertex pairs) rather than axis-aligned rectangles.  This
    function attempts to reshape / validate the input into an ``(V, 2)``
    array of polygon vertices.  Returns ``None`` if the data doesn't
    represent a valid polygon (e.g. it's a simple ``[x, y, w, h]`` box).
    """
    if box_entry is None:
        return None
    try:
        arr = np.asarray(box_entry, dtype=np.float32)
    except (TypeError, ValueError):
        return None
    if arr.size == 0:
        return None

    # A flat array with >= 6 elements and even count can be reshaped to (V, 2).
    if arr.ndim == 1 and arr.size >= 6 and (arr.size % 2 == 0):
        arr = arr.reshape(-1, 2)
    # Need at least 3 vertices (triangle) with x,y columns.
    if arr.ndim != 2 or arr.shape[0] < 3 or arr.shape[1] < 2:
        return None
    pts = arr[:, :2]
    # Drop any vertices with non-finite coordinates.
    finite_mask = np.isfinite(pts).all(axis=1)
    pts = pts[finite_mask]
    if pts.shape[0] < 3:
        return None
    return pts


def _select_tracked_person_index(poses2d_batch: np.ndarray, poses3d_batch: np.ndarray) -> int | None:
    """Choose which person in the batch to visualize.

    The upstream pose-estimation pipeline (server scripts) is expected to
    have already selected the target person (e.g. the closest body, or the
    single-person output of MetrAbs).  If multiple detections still remain
    in the batch, we deterministically pick index 0 rather than applying
    complex distance-based heuristics -- keeping this function simple and
    predictable.

    Returns ``None`` when no person is detected at all.
    """
    # Prefer 3D data if available; fall back to 2D.
    if poses3d_batch.ndim == 3 and poses3d_batch.shape[0] > 0:
        return 0
    if poses2d_batch.ndim == 3 and poses2d_batch.shape[0] > 0:
        return 0
    return None


def _compute_torso_distance(
    tracked_points_3d: np.ndarray | None,
    torso_index: int,
    units_to_meters: float,
) -> float | None:
    """Compute the Euclidean distance from the camera origin to the torso joint.

    The torso keypoint (e.g. pelvis or spine-base depending on the skeleton
    model) serves as a rough proxy for "how far away is the person."  The
    distance is displayed in the 3D panel as "Distance to camera: X.XX m".

    Parameters
    ----------
    tracked_points_3d : np.ndarray or None
        ``(J, 3+)`` array of the tracked person's 3D keypoints in camera
        coordinates.  ``None`` if no person is tracked.
    torso_index : int
        Which joint index to use as the torso reference.  Clamped to valid
        range internally.
    units_to_meters : float
        Conversion factor from the native coordinate units to meters.

    Returns
    -------
    float or None
        Distance in meters, or ``None`` if the torso is invalid.
    """
    if tracked_points_3d is None or tracked_points_3d.ndim != 2 or tracked_points_3d.shape[0] == 0:
        return None
    # Clamp the index so out-of-range values don't crash.
    torso_index = max(0, min(int(torso_index), tracked_points_3d.shape[0] - 1))
    torso = tracked_points_3d[torso_index, :3]
    if not np.isfinite(torso).all():
        return None
    # Distance = magnitude of the 3D position vector (camera at origin).
    return float(np.linalg.norm(torso) * float(units_to_meters))


# ---------------------------------------------------------------------------
# Public data classes
# ---------------------------------------------------------------------------


@dataclass
class VisualizationResult:
    """Return value from ``PoseVisualizer.update()``.

    Attributes
    ----------
    tracked_index : int or None
        Index into the pose batch of the person being visualized.  ``None``
        when no person was detected in the current frame.
    distance_m : float or None
        Estimated distance from the camera to the tracked person's torso in
        meters.  ``None`` when unavailable.
    fps : float
        The most recently computed visualization FPS (may lag behind the
        actual frame rate by up to ``fps_update_interval_sec``).
    key : int
        The key code returned by ``cv2.waitKey()``.  Callers can check this
        to detect quit keys (e.g. ``ord('q')`` or ``27`` for Escape).
    frame : np.ndarray
        The final composited BGR image that was displayed in the window.
        Can be used for recording or further processing.
    """
    tracked_index: int | None
    distance_m: float | None
    fps: float
    key: int
    frame: np.ndarray


# ---------------------------------------------------------------------------
# Main visualizer class
# ---------------------------------------------------------------------------


class PoseVisualizer:
    """Real-time debug visualizer for 2D + 3D pose estimation results.

    The visualizer renders a split-screen OpenCV window:

    +---------------------------+---------------------------+
    |     Title bar (source, FPS, inference FPS)             |
    +---------------------------+---------------------------+
    |  Left panel:              |  Right panel:             |
    |  Camera frame with 2D     |  Matplotlib 3D scatter    |
    |  skeleton overlay,        |  plot of the body in      |
    |  bounding box, and extra  |  display coordinates      |
    |  hand landmarks           |  with edges and distance  |
    +---------------------------+---------------------------+

    Usage pattern (called once per frame in the server loop)::

        viz = PoseVisualizer(source_name="MetrAbs")
        while running:
            result = viz.update(frame, poses2d, poses3d, ...)
            if result.key == ord('q'):
                break
        viz.close()

    Parameters
    ----------
    source_name : str
        Label shown in the title bar (e.g. "MetrAbs", "ZED").
    window_name : str
        OpenCV window title.
    fullscreen : bool
        If ``True``, the window is set to fullscreen mode.
    window_size : tuple[int, int]
        ``(width, height)`` of the OpenCV window in pixels.
    fps_update_interval_sec : float
        How often (in seconds) the displayed FPS value is refreshed.
        A longer interval gives a more stable reading.
    """

    def __init__(
        self,
        source_name: str,
        window_name: str = "Pose Visualization",
        fullscreen: bool = False,
        window_size: tuple[int, int] = (1600, 900),
        fps_update_interval_sec: float = 2.0,
    ):
        self.source_name = source_name
        self.window_name = window_name
        self.fullscreen = fullscreen
        self.window_size = (int(window_size[0]), int(window_size[1]))

        # Height in pixels reserved for the title bar at the top of the window.
        self.title_height = 78

        # --- FPS tracking state ---
        self._fps = 0.0  # Most recently computed FPS value.
        self._fps_update_interval_sec = max(float(fps_update_interval_sec), 0.1)
        self._fps_window_start = None      # Start timestamp of the current measurement window.
        self._fps_window_frames = 0        # Frames counted in the current window.
        self._fps_external_latest = None   # Latest FPS value supplied by the caller.

        # --- Matplotlib off-screen rendering objects ---
        # We reuse a single Figure and Axes3D across frames to avoid the
        # overhead of creating new matplotlib objects every iteration.
        self._figure = Figure(figsize=(6.4, 4.8), dpi=100)
        self._canvas = FigureCanvasAgg(self._figure)  # Agg backend for off-screen rendering.
        self._ax3d = self._figure.add_subplot(111, projection="3d")
        self._3d_artists: list = []  # Per-frame 3D artists for efficient removal.
        # Configure axis labels and appearance once (persists across frames).
        self._ax3d.set_title("3D Pose")
        self._ax3d.set_xlabel("X (m)")
        self._ax3d.set_ylabel("Depth (m)")
        self._ax3d.set_zlabel("Height (m)")
        self._ax3d.view_init(elev=20, azim=-70)
        self._ax3d.grid(True)

        # Tracks whether the OpenCV window has been created yet.
        self._window_initialized = False

    def _ensure_window(self):
        """Lazily create the OpenCV display window on first use.

        We defer window creation to the first ``update()`` call rather than
        ``__init__`` so that importing or constructing a ``PoseVisualizer``
        has no visible side-effects (important when visualization is disabled).
        """
        if self._window_initialized:
            return
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.window_name, self.window_size[0], self.window_size[1])
        if self.fullscreen:
            cv2.setWindowProperty(self.window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        else:
            cv2.setWindowProperty(self.window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_NORMAL)
        self._window_initialized = True

    def _update_fps(self, external_fps: float | None = None):
        """Update the displayed FPS counter.

        FPS is computed over a sliding time window of length
        ``_fps_update_interval_sec``.  Two sources are supported:

        1. **External FPS** -- if the server supplies a pre-computed FPS
           value (e.g. from its own inference timer), that value takes
           priority because it reflects the true pipeline throughput.
        2. **Self-measured FPS** -- if no external value is available, we
           count how many ``update()`` calls occurred within the window
           and divide by elapsed time.

        The value is only refreshed when the window elapses, so the
        on-screen number stays stable and readable.
        """
        now = time.perf_counter()
        if self._fps_window_start is None:
            self._fps_window_start = now

        self._fps_window_frames += 1

        # Latch the latest externally provided FPS if valid.
        if external_fps is not None:
            try:
                fps_value = float(external_fps)
            except (TypeError, ValueError):
                fps_value = None
            if fps_value is not None and np.isfinite(fps_value) and fps_value > 0:
                self._fps_external_latest = fps_value

        elapsed = now - self._fps_window_start
        # Only update the displayed value when the measurement window has
        # fully elapsed.  This avoids jittery numbers.
        if elapsed < self._fps_update_interval_sec:
            return

        if self._fps_external_latest is not None:
            # Prefer externally reported FPS (e.g. inference throughput).
            self._fps = self._fps_external_latest
        elif elapsed > 0:
            # Fall back to self-measured visualization FPS.
            self._fps = self._fps_window_frames / elapsed

        # Reset the counters for the next measurement window.
        self._fps_window_start = now
        self._fps_window_frames = 0

    def _render_3d_view(
        self,
        target_shape: tuple[int, int],
        points_3d: np.ndarray | None,
        joint_edges: list[tuple[int, int]],
        extra_points_3d: dict[str, dict[int, Sequence[float]]] | None,
        suppress_joint_indices: set[int],
        units_to_meters: float,
        torso_index: int,
        distance_m: float | None,
    ) -> np.ndarray:
        """Render the right-side 3D scatter-plot panel and return it as a BGR image.

        The method clears the reusable Axes3D, plots body keypoints as a
        colored scatter, draws bone edges as line segments, highlights the
        torso reference joint in red, optionally overlays extra hand points,
        and auto-scales the axes to keep the skeleton centered.  The
        matplotlib figure is rendered off-screen via the Agg backend and
        converted to a BGR NumPy array for compositing with OpenCV.

        Parameters
        ----------
        target_shape : tuple[int, int]
            ``(height, width)`` of the desired output image.
        points_3d : np.ndarray or None
            ``(J, 3+)`` tracked person 3D keypoints in camera coords.
        joint_edges : list[tuple[int, int]]
            Bone connectivity for skeleton drawing.
        extra_points_3d : dict or None
            Additional labeled 3D point groups (e.g. hand landmarks)
            keyed by label string ("Left", "Right").
        suppress_joint_indices : set[int]
            Joints to hide from the plot.
        units_to_meters : float
            Conversion factor applied to raw coordinates.
        torso_index : int
            Index of the torso keypoint (highlighted in red).
        distance_m : float or None
            Pre-computed camera-to-torso distance for the overlay text.

        Returns
        -------
        np.ndarray
            BGR image of shape ``(target_h, target_w, 3)``.
        """
        target_h, target_w = target_shape

        # Size the matplotlib figure to match the target panel dimensions.
        dpi = 100
        width_in = max(target_w / dpi, 2.0)
        height_in = max(target_h / dpi, 2.0)
        self._figure.set_size_inches(width_in, height_in, forward=False)

        ax = self._ax3d
        # Remove previous frame's artists instead of ax.cla(), which
        # tears down the entire axis structure each frame.
        for _artist in self._3d_artists:
            try:
                _artist.remove()
            except (ValueError, NotImplementedError):
                pass
        self._3d_artists.clear()
        _old_legend = ax.get_legend()
        if _old_legend is not None:
            _old_legend.remove()

        # ---------------------------------------------------------------
        # Pre-process extra 3D points (e.g. MediaPipe hand landmarks).
        # Convert them from camera coords to display coords so they
        # appear correctly alongside the body skeleton.
        # ---------------------------------------------------------------
        extra_display_points: dict[str, np.ndarray] = {}
        if extra_points_3d:
            for label, point_map in extra_points_3d.items():
                if not isinstance(point_map, dict) or not point_map:
                    continue
                label_points = []
                for point in point_map.values():
                    try:
                        point_arr = np.asarray(point, dtype=np.float32).reshape(-1)
                    except (TypeError, ValueError):
                        continue
                    if point_arr.size < 3:
                        continue
                    point_xyz = point_arr[:3]
                    if not np.isfinite(point_xyz).all():
                        continue
                    label_points.append(point_xyz)
                if not label_points:
                    continue
                # Scale to meters and convert to display axes.
                label_points_np = np.asarray(label_points, dtype=np.float32) * float(units_to_meters)
                extra_display_points[str(label)] = _camera_to_display_axes(label_points_np)

        # ---------------------------------------------------------------
        # Plot the body skeleton (keypoints + edges) in 3D.
        # ---------------------------------------------------------------
        plotted_any_points = False
        bounds_points = []  # Collects all plotted points for auto-scaling.

        if points_3d is not None and points_3d.size > 0:
            # Convert raw coordinates to meters and then to display axes.
            pts = np.asarray(points_3d[:, :3], dtype=np.float32) * float(units_to_meters)
            pts = _camera_to_display_axes(pts)

            # Build a boolean mask of joints that are both finite and not
            # suppressed (hidden) by the caller.
            valid = np.isfinite(pts).all(axis=1)
            if suppress_joint_indices and pts.shape[0] > 0:
                visible = np.ones(pts.shape[0], dtype=bool)
                for joint_idx in suppress_joint_indices:
                    if 0 <= joint_idx < pts.shape[0]:
                        visible[joint_idx] = False
                valid &= visible

            if np.any(valid):
                valid_pts = pts[valid]
                # Color-code joints along a gradient so they're visually
                # distinguishable (head joints lighter, leg joints darker).
                colors = np.linspace(0.0, 1.0, valid_pts.shape[0], dtype=np.float32)
                _sc = ax.scatter(
                    valid_pts[:, 0],
                    valid_pts[:, 1],
                    valid_pts[:, 2],
                    c=colors,
                    cmap="viridis",
                    s=24,
                    depthshade=True,
                )
                self._3d_artists.append(_sc)
                plotted_any_points = True
                bounds_points.append(valid_pts)

                # Draw bone edges as line segments between connected joints.
                for edge_start, edge_end in joint_edges:
                    if edge_start >= len(pts) or edge_end >= len(pts):
                        continue
                    p0 = pts[edge_start]
                    p1 = pts[edge_end]
                    if np.isfinite(p0).all() and np.isfinite(p1).all():
                        _ln = ax.plot(
                            [p0[0], p1[0]],
                            [p0[1], p1[1]],
                            [p0[2], p1[2]],
                            color="#1f77b4",
                            linewidth=1.5,
                        )
                        self._3d_artists.extend(_ln)

                # Highlight the torso reference joint in red so the user can
                # quickly verify which point the distance measurement refers to.
                if (
                    0 <= torso_index < len(pts)
                    and torso_index not in suppress_joint_indices
                    and np.isfinite(pts[torso_index]).all()
                ):
                    torso = pts[torso_index]
                    _tsc = ax.scatter([torso[0]], [torso[1]], [torso[2]], c="red", s=50, label="Torso")
                    self._3d_artists.append(_tsc)
                    ax.legend(loc="upper right")

        # ---------------------------------------------------------------
        # Overlay extra 3D points (e.g. hand landmarks from MediaPipe).
        # Left hand = red-ish, Right hand = blue-ish, other = orange.
        # ---------------------------------------------------------------
        extra_colors = {"Left": "#ff6b6b", "Right": "#4dabf7"}
        for label, label_pts in extra_display_points.items():
            if label_pts.size == 0:
                continue
            _esc = ax.scatter(
                label_pts[:, 0],
                label_pts[:, 1],
                label_pts[:, 2],
                c=extra_colors.get(label, "#ff922b"),
                s=42,
                depthshade=True,
                marker="o",
                label=f"{label} hand",
            )
            self._3d_artists.append(_esc)
            plotted_any_points = True
            bounds_points.append(label_pts)

        # ---------------------------------------------------------------
        # Auto-scale the 3D axes to keep the skeleton centered and visible.
        # We compute a bounding sphere from all plotted points and set
        # equal limits on all three axes so the aspect ratio is preserved.
        # ---------------------------------------------------------------
        if bounds_points:
            all_bounds_pts = np.vstack(bounds_points)
            center = np.mean(all_bounds_pts, axis=0)
            span = np.max(all_bounds_pts, axis=0) - np.min(all_bounds_pts, axis=0)
            # Radius is 60% of the largest span, with a 0.5 m minimum so
            # the view doesn't zoom in excessively on a nearly-static pose.
            radius = float(max(np.max(span) * 0.6, 0.5))
            ax.set_xlim(center[0] - radius, center[0] + radius)
            ax.set_ylim(center[1] - radius, center[1] + radius)
            ax.set_zlim(center[2] - radius, center[2] + radius)
        elif points_3d is not None and points_3d.size > 0:
            # Points exist but all were non-finite or suppressed.
            _txt = ax.text2D(0.1, 0.5, "No valid 3D keypoints", transform=ax.transAxes)
            self._3d_artists.append(_txt)
        else:
            # No person detected at all.
            _txt = ax.text2D(0.1, 0.5, "No tracked person", transform=ax.transAxes)
            self._3d_artists.append(_txt)

        # Force equal aspect ratio so the skeleton isn't distorted.
        try:
            ax.set_box_aspect((1, 1, 1))
        except Exception:
            pass  # Older matplotlib versions may not support set_box_aspect.
        ax.grid(True)

        # ---------------------------------------------------------------
        # Render the matplotlib figure to an off-screen RGBA buffer and
        # convert it to a BGR OpenCV image for compositing.
        # ---------------------------------------------------------------
        self._canvas.draw()

        rgba = np.asarray(self._canvas.buffer_rgba())  # (H, W, 4) uint8
        rgb = rgba[:, :, :3]                           # Drop alpha channel.
        right_view = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        right_view = cv2.resize(right_view, (target_w, target_h), interpolation=cv2.INTER_AREA)

        # Overlay the camera-to-torso distance text on the 3D panel.
        distance_text = "Distance to camera: n/a"
        if distance_m is not None and np.isfinite(distance_m):
            distance_text = f"Distance to camera: {distance_m:.2f} m"
        cv2.putText(
            right_view,
            distance_text,
            (20, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 0),
            2,
            cv2.LINE_AA,
        )
        return right_view

    def update(
        self,
        frame_bgr: np.ndarray,
        poses2d,
        poses3d,
        joint_edges: Iterable[tuple[int, int]] | None = None,
        boxes=None,
        extra_points_2d: dict[str, dict[int, tuple[int, int]]] | None = None,
        extra_points_3d: dict[str, dict[int, Sequence[float]]] | None = None,
        suppress_joint_indices: Iterable[int] | None = None,
        torso_index: int = 0,
        units_to_meters: float = 1.0,
        fps: float | None = None,
        inference_fps: float | None = None,
        wait_key_delay: int = 1,
        source_name: str | None = None,
        draw_bounding_box: bool = True,
    ) -> VisualizationResult:
        """Process one frame and display the updated visualization window.

        This is the main per-frame entry point.  It:

        1. Normalizes all inputs (poses, boxes, edges) to consistent formats.
        2. Selects which person in the batch to visualize.
        3. Draws the 2D overlay (keypoints, edges, bounding box, hand
           landmarks) on a copy of the camera frame.
        4. Renders the 3D scatter plot via ``_render_3d_view``.
        5. Composites the left (2D) and right (3D) panels side-by-side,
           adds a title bar, and displays everything in the OpenCV window.

        Parameters
        ----------
        frame_bgr : np.ndarray
            Raw camera frame in BGR format.
        poses2d, poses3d
            Per-person 2D / 3D keypoints.  Accepted as NumPy arrays,
            PyTorch tensors, or compatible objects.
        joint_edges : iterable of (int, int), optional
            Skeleton bone connectivity.  Falls back to BODY_38 if ``None``
            and the joint count is >= 38.
        boxes : optional
            Per-person bounding boxes (rectangles or polygons).
        extra_points_2d, extra_points_3d : dict, optional
            Additional labeled point groups (e.g. hand landmarks) for
            overlay on 2D and 3D panels respectively.
        suppress_joint_indices : iterable of int, optional
            Joints to hide from both panels (e.g. face joints).
        torso_index : int
            Index of the torso/pelvis keypoint for distance calculation.
        units_to_meters : float
            Multiply raw 3D coordinates by this to get meters.
        fps : float, optional
            Externally measured FPS to display.
        inference_fps : float, optional
            Separate inference-only FPS shown in the title bar.
        wait_key_delay : int
            Milliseconds to wait in ``cv2.waitKey`` (1 = minimal blocking).
        source_name : str, optional
            Override the source label in the title bar.
        draw_bounding_box : bool
            Whether to draw bounding boxes / polygons on the 2D panel.

        Returns
        -------
        VisualizationResult
            Struct containing tracked index, distance, FPS, key press, and
            the final composited frame.
        """
        # Create the OpenCV window if it doesn't exist yet.
        self._ensure_window()
        self._update_fps(external_fps=fps)

        # Allow the caller to dynamically change the source label.
        if source_name:
            self.source_name = source_name

        # --- Normalize inputs into uniform NumPy representations ---
        poses2d_np = _to_pose_batch(poses2d)      # (N, J, 2+)
        poses3d_np = _to_pose_batch(poses3d)      # (N, J, 3+)
        boxes_list = _to_boxes_list(boxes) if draw_bounding_box else None
        tracked_index = _select_tracked_person_index(poses2d_np, poses3d_np)
        suppress_joint_indices_set = _normalize_index_set(suppress_joint_indices)

        # Work on a copy of the camera frame so we don't mutate the caller's data.
        left_view = frame_bgr.copy()
        frame_h, frame_w = left_view.shape[:2]

        # Each side panel takes half the window width; the panel height is
        # the window height minus the title bar.
        panel_w = max(self.window_size[0] // 2, 320)
        panel_h = max(self.window_size[1] - self.title_height, 240)

        # Extract the tracked person's keypoints from the batch.
        tracked_points_2d = None
        tracked_points_3d = None
        if tracked_index is not None and tracked_index < len(poses2d_np):
            tracked_points_2d = poses2d_np[tracked_index]
        if tracked_index is not None and tracked_index < len(poses3d_np):
            tracked_points_3d = poses3d_np[tracked_index]

        # Compute camera-to-torso distance for the 3D panel overlay.
        distance_m = _compute_torso_distance(tracked_points_3d, torso_index, units_to_meters)

        # --- Resolve skeleton edge topology ---
        # Use caller-supplied edges if available; otherwise fall back to
        # a default based on the number of keypoints (BODY_38 for >= 38).
        edges = _normalize_edges(joint_edges)
        if not edges:
            joint_count = 0
            if tracked_points_2d is not None:
                joint_count = int(tracked_points_2d.shape[0])
            elif tracked_points_3d is not None:
                joint_count = int(tracked_points_3d.shape[0])
            elif poses2d_np.ndim == 3 and poses2d_np.shape[1] > 0:
                joint_count = int(poses2d_np.shape[1])
            elif poses3d_np.ndim == 3 and poses3d_np.shape[1] > 0:
                joint_count = int(poses3d_np.shape[1])
            edges = _default_edges_for_joint_count(joint_count)
        # Remove edges that touch suppressed joints so we don't draw
        # dangling lines to hidden keypoints.
        visible_edges = [
            (edge_start, edge_end)
            for edge_start, edge_end in edges
            if edge_start not in suppress_joint_indices_set and edge_end not in suppress_joint_indices_set
        ]

        # ---------------------------------------------------------------
        # Draw the 2D skeleton overlay on the left (camera) panel.
        # ---------------------------------------------------------------
        if tracked_points_2d is not None:
            # Draw bone edges as cyan lines between connected keypoints.
            for edge_start, edge_end in visible_edges:
                if edge_start >= len(tracked_points_2d) or edge_end >= len(tracked_points_2d):
                    continue
                p0 = tracked_points_2d[edge_start]
                p1 = tracked_points_2d[edge_end]
                if _is_valid_point2d(p0, frame_w, frame_h) and _is_valid_point2d(p1, frame_w, frame_h):
                    cv2.line(
                        left_view,
                        (int(p0[0]), int(p0[1])),
                        (int(p1[0]), int(p1[1])),
                        (0, 255, 255),  # Cyan
                        2,
                    )

            # Draw each visible keypoint as a green dot with its index label.
            for joint_idx, joint in enumerate(tracked_points_2d):
                if joint_idx in suppress_joint_indices_set:
                    continue
                if not _is_valid_point2d(joint, frame_w, frame_h):
                    continue
                x, y = int(joint[0]), int(joint[1])
                cv2.circle(left_view, (x, y), 4, (0, 220, 0), -1)  # Green filled circle
                cv2.putText(
                    left_view,
                    str(joint_idx),
                    (x + 6, y - 6),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.45,
                    (255, 255, 255),
                    1,
                    cv2.LINE_AA,
                )

            # -----------------------------------------------------------
            # Draw bounding box or polygon around the tracked person.
            # The box source can be:
            #   1. A polygon (ZED bounding_box_2d) with >= 3 vertices.
            #   2. A rectangle [x, y, w, h] from a YOLO / MetrAbs detector.
            #   3. A fallback tight bbox computed from the 2D keypoints.
            # -----------------------------------------------------------
            if draw_bounding_box:
                bbox = None
                label_anchor = None  # Top-left corner for the "Tracked" label.
                polygon = None

                if boxes_list is not None and tracked_index < len(boxes_list):
                    box_entry = boxes_list[tracked_index]
                    # Try interpreting as a polygon first (ZED style).
                    polygon = _normalize_polygon(box_entry)
                    if polygon is None:
                        # Fall back to [x, y, w, h] rectangle format.
                        try:
                            flat_box = np.asarray(box_entry, dtype=np.float32).reshape(-1)
                        except (TypeError, ValueError):
                            flat_box = None
                        if flat_box is not None and flat_box.size >= 4 and np.isfinite(flat_box[:4]).all():
                            x, y, w, h = flat_box[:4]
                            bbox = (
                                int(max(0, x)),
                                int(max(0, y)),
                                int(min(frame_w - 1, x + w)),
                                int(min(frame_h - 1, y + h)),
                            )

                # Draw polygon outline if available...
                if polygon is not None:
                    # Clip polygon vertices to frame boundaries.
                    polygon[:, 0] = np.clip(polygon[:, 0], 0, frame_w - 1)
                    polygon[:, 1] = np.clip(polygon[:, 1], 0, frame_h - 1)
                    poly_i = np.round(polygon).astype(np.int32)
                    cv2.polylines(left_view, [poly_i], True, (0, 128, 255), 2)  # Orange outline
                    label_anchor = (
                        int(np.min(poly_i[:, 0])),
                        int(np.min(poly_i[:, 1])),
                    )
                else:
                    # ...otherwise draw a rectangle (supplied or computed).
                    if bbox is None:
                        bbox = _compute_bbox_from_points(tracked_points_2d, frame_w, frame_h)
                    if bbox is not None:
                        cv2.rectangle(left_view, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 128, 255), 2)
                        label_anchor = (bbox[0], bbox[1])

                # Place a "Tracked" label above the bounding region.
                if label_anchor is not None:
                    cv2.putText(
                        left_view,
                        "Tracked",
                        (label_anchor[0], max(20, label_anchor[1] - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0, 128, 255),
                        2,
                        cv2.LINE_AA,
                    )

        # ---------------------------------------------------------------
        # Draw extra 2D points (e.g. MediaPipe hand landmarks) on the
        # camera frame as magenta dots with "Label:Index" annotations.
        # ---------------------------------------------------------------
        if extra_points_2d:
            for label, points in extra_points_2d.items():
                for point_idx, (x, y) in points.items():
                    if not (0 <= int(x) < frame_w and 0 <= int(y) < frame_h):
                        continue
                    cv2.circle(left_view, (int(x), int(y)), 5, (255, 0, 180), -1)  # Magenta
                    cv2.putText(
                        left_view,
                        f"{label}:{point_idx}",
                        (int(x) + 6, int(y) - 6),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (255, 255, 255),
                        1,
                        cv2.LINE_AA,
                    )

        # ---------------------------------------------------------------
        # Render the 3D panel and letterbox the 2D panel to match sizes.
        # ---------------------------------------------------------------
        right_view = self._render_3d_view(
            target_shape=(panel_h, panel_w),
            points_3d=tracked_points_3d,
            joint_edges=visible_edges,
            extra_points_3d=extra_points_3d,
            suppress_joint_indices=suppress_joint_indices_set,
            units_to_meters=units_to_meters,
            torso_index=torso_index,
            distance_m=distance_m,
        )
        # Letterbox the camera frame to the panel size so both panels
        # have identical dimensions for side-by-side composition.
        left_view = _letterbox(left_view, panel_w, panel_h)

        # ---------------------------------------------------------------
        # Compose the final output: title bar on top, panels below.
        # ---------------------------------------------------------------
        body_view = np.hstack((left_view, right_view))
        # Dark grey title bar spanning the full width.
        title_bar = np.ones((self.title_height, body_view.shape[1], 3), dtype=np.uint8) * 20
        # --- Title bar text: source name ---
        cv2.putText(
            title_bar,
            f"Pose Visualization | Source: {self.source_name}",
            (20, 26),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.72,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        # --- Title bar text: visualization FPS ---
        viz_fps_text = f"Visualization FPS: {self._fps:.1f}"
        cv2.putText(
            title_bar,
            viz_fps_text,
            (20, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.62,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        # --- Title bar text: inference FPS (from the pose model) ---
        if inference_fps is not None and np.isfinite(float(inference_fps)) and float(inference_fps) > 0:
            inference_fps_text = f"Inference FPS: {float(inference_fps):.1f}"
        else:
            inference_fps_text = "Inference FPS: n/a"
        cv2.putText(
            title_bar,
            inference_fps_text,
            (20, self.title_height - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.62,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

        # Stack title bar on top of the two panels.
        output = np.vstack((title_bar, body_view))

        # If the composited image doesn't exactly match the window size
        # (e.g. due to rounding), letterbox it to avoid stretching.
        if output.shape[1] != self.window_size[0] or output.shape[0] != self.window_size[1]:
            output = _letterbox(output, self.window_size[0], self.window_size[1], fill_color=(20, 20, 20))

        # Display the final frame and capture any key press.
        cv2.imshow(self.window_name, output)
        key = cv2.waitKey(wait_key_delay) & 0xFF

        return VisualizationResult(
            tracked_index=tracked_index,
            distance_m=distance_m,
            fps=self._fps,
            key=key,
            frame=output,
        )

    def close(self):
        """Destroy the OpenCV window and reset initialization state.

        Safe to call multiple times or even if the window was never created.
        After calling ``close()``, the next ``update()`` call will re-create
        the window automatically via ``_ensure_window()``.
        """
        try:
            cv2.destroyWindow(self.window_name)
        except Exception:
            pass
        self._window_initialized = False
