import argparse
import contextlib
import io
import json
import math
import re
import sys
from pathlib import Path

import cv2
import numpy as np


CURRENT_FILE = Path(__file__).resolve()
DL_POSE_DIR = CURRENT_FILE.parents[1]
SERVER_DIR = CURRENT_FILE.parents[2]
REPO_ROOT = CURRENT_FILE.parents[3]

from server.common import settings  # noqa: E402
from server.ikpy import ikpy_utils as ikpyu  # noqa: E402
from server.common.angle_classifier_pepper import AngleClassifier  # noqa: E402
from server.proposed.angle_classifier_human import HumanArmClassifier  # noqa: E402
from server.common.hand_detection.EMA_smoothing import EMASmoothing  # noqa: E402
from server.common.hand_keypoints import _extract_hand_points, _open_hand_landmarker  # noqa: E402
from server.common import hand_orientation  # noqa: E402
import server.common.kinematics.forward_kinematics as fk  # noqa: E402
import server.common.kinematics.scaling_spherical as scaling  # noqa: E402
import server.common.kinematics.transformation_matrices as tm  # noqa: E402
from server.baseline.evaluation import original_solution_adapter as original_adapter  # noqa: E402
from server.proposed.pose_handling import PoseHandler  # noqa: E402


ARM_SIDES = ("left", "right")
ANGLE_KEYS = ("shoulder_pitch", "shoulder_roll", "elbow_yaw", "elbow_roll", "wrist_yaw")
POINT_KEYS = ("shoulder", "elbow", "wrist")
METHODS = ("ground_truth", "current_solution_raw", "current_solution_constrained", "ikpy", "original_solution")
METRIC_NAMES = ("EEAh", "EEAr", "SOAx", "HJL", "WOM", "HJAr", "TSE", "SYN")
ROBOT_HAND_NORMAL_AXIS = 2
ROBOT_HAND_NORMAL_SIGN = -1.0
RUNTIME_ORIENTATION_KEY = "_runtime_hand_orientation_labels"
VIDEO_FOLDER_PATTERN = re.compile(r"^video_(\d+)$")
HUMAN_ARM_CLASSIFIER = HumanArmClassifier()

PEPPER_SHOULDER = {
    "right": np.array([-57.0, -149.74, 86.82], dtype=np.float64),
    "left": np.array([-57.0, 149.74, 86.82], dtype=np.float64),
}

# Ranges are used to normalize HJAr. They follow Pepper-compatible arm ranges in radians.
JOINT_RANGES = {
    "shoulder_pitch": (-2.0857, 2.0857),
    "shoulder_roll": {
        "right": (-1.5620, -0.0087),
        "left": (0.0087, 1.5620),
    },
    "elbow_yaw": (-2.0857, 2.0857),
    "elbow_roll": {
        "right": (0.0087, 1.5620),
        "left": (-1.5620, -0.0087),
    },
    "wrist_yaw": (-1.8238, 1.8238),
}


def _load_json(path):
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def _save_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(data, file, indent=4)


def _json_safe(value):
    if isinstance(value, np.ndarray):
        return [_json_safe(item) for item in value.tolist()]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        value = float(value)
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return None
        return value
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def _as_vector(value):
    if isinstance(value, dict):
        values = [value.get("x"), value.get("y"), value.get("z")]
    elif isinstance(value, list) and len(value) >= 3:
        values = value[:3]
    else:
        return None

    if any(item is None for item in values):
        return None

    try:
        vector = np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError):
        return None

    if not np.isfinite(vector).all():
        return None
    return vector


def _safe_float(value):
    if value is None:
        return None
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(numeric) or math.isinf(numeric):
        return None
    return numeric


def _annotation_files(input_path):
    input_path = input_path.resolve()
    if input_path.is_file():
        return [input_path]
    if (input_path / "annotations_filled.json").exists():
        return [input_path / "annotations_filled.json"]
    return sorted(input_path.rglob("annotations_filled.json"))


def _filter_annotation_files_by_max_video_index(files, max_video_index):
    """Keep annotations whose video_XXX index is at or below the limit."""
    if max_video_index is None:
        return list(files)
    if max_video_index < 1:
        raise ValueError("--max-video-index must be >= 1")

    filtered = []
    for path in files:
        match = VIDEO_FOLDER_PATTERN.fullmatch(path.parent.name)
        if match is None:
            raise ValueError(
                f"Cannot apply --max-video-index: annotation is not inside "
                f"a video_XXX folder: {path}"
            )
        if int(match.group(1)) <= max_video_index:
            filtered.append(path)
    return filtered


def _frame_points(frame_data, side, pepper_coordinates=False):
    arm = frame_data.get(f"{side}_arm", {})
    points = {
        "shoulder": _as_vector(arm.get("shoulder")),
        "elbow": _as_vector(arm.get("elbow")),
        "wrist": _as_vector(arm.get("wrist")),
    }
    if pepper_coordinates:
        points = {
            key: tm.set_joint_in_peppers_coordinates(value) if value is not None else None
            for key, value in points.items()
        }
    return points


def _posture_descriptor(points):
    if points is None:
        return None
    shoulder = points.get("shoulder")
    elbow = points.get("elbow")
    wrist = points.get("wrist")
    if any(point is None for point in (shoulder, elbow, wrist)):
        return None
    upper_arm = _unit(elbow - shoulder)
    forearm = _unit(wrist - elbow)
    if upper_arm is None or forearm is None:
        return None
    descriptor = np.concatenate((upper_arm, forearm)).astype(np.float64)
    return descriptor if np.isfinite(descriptor).all() else None


def _build_synergy_model(annotation_files, variance_threshold=0.95):
    descriptors = []
    rejected_samples = 0
    for path in annotation_files:
        annotations = _load_json(path)
        for frame_data in annotations.get("frames", []):
            for side in ARM_SIDES:
                descriptor = _posture_descriptor(
                    _frame_points(frame_data, side, pepper_coordinates=True)
                )
                if descriptor is None:
                    rejected_samples += 1
                else:
                    descriptors.append(descriptor)

    if len(descriptors) < 2:
        raise RuntimeError(
            "At least two valid human arm descriptors are required to build the synergy space."
        )

    samples = np.asarray(descriptors, dtype=np.float64)
    mean = np.mean(samples, axis=0)
    centered = samples - mean
    _, singular_values, vt = np.linalg.svd(centered, full_matrices=False)
    variances = (singular_values ** 2) / max(samples.shape[0] - 1, 1)
    total_variance = float(np.sum(variances))
    if total_variance <= 1e-12:
        raise RuntimeError("The human synergy descriptors have no usable variance.")

    explained_variance_ratio = variances / total_variance
    cumulative = np.cumsum(explained_variance_ratio)
    component_count = int(np.searchsorted(cumulative, variance_threshold) + 1)
    component_count = max(1, min(component_count, samples.shape[1] - 1, vt.shape[0]))

    return {
        "mean": mean,
        "components": vt[:component_count],
        "explained_variance": variances[:component_count],
        "explained_variance_ratio": explained_variance_ratio[:component_count],
        "cumulative_explained_variance": float(cumulative[component_count - 1]),
        "component_count": component_count,
        "variance_threshold": float(variance_threshold),
        "valid_samples": int(samples.shape[0]),
        "rejected_samples": int(rejected_samples),
        "descriptor_order": [
            "upper_arm_x",
            "upper_arm_y",
            "upper_arm_z",
            "forearm_x",
            "forearm_y",
            "forearm_z",
        ],
    }


def _syn(points, synergy_model):
    if synergy_model is None:
        return None
    descriptor = _posture_descriptor(points)
    if descriptor is None:
        return None
    centered = descriptor - synergy_model["mean"]
    coefficients = synergy_model["components"] @ centered
    reconstruction = synergy_model["mean"] + coefficients @ synergy_model["components"]
    return float(np.linalg.norm(descriptor - reconstruction))


def _ground_truth_angles(frame_data, side):
    arm = frame_data.get(f"{side}_arm", {})
    joint_angles = arm.get("joint_angles", {}) if isinstance(arm, dict) else {}
    return {
        "shoulder_pitch": _safe_float(joint_angles.get("shoulder_pitch")),
        "shoulder_roll": _safe_float(joint_angles.get("shoulder_roll")),
        "elbow_yaw": _safe_float(joint_angles.get("elbow_yaw")),
        "elbow_roll": _safe_float(joint_angles.get("elbow_roll")),
        "wrist_yaw": 0.0,
    }


def _compute_current_solution(frame_data, side):
    other_side = "left" if side == "right" else "right"
    torso = _as_vector(frame_data.get("torso"))
    points = _frame_points(frame_data, side)
    other_points = _frame_points(frame_data, other_side)

    required = (torso, points["shoulder"], points["elbow"], points["wrist"], other_points["shoulder"])
    if any(item is None for item in required):
        raise ValueError("missing coordinates")

    t1, t2, t3, t4, _ = scaling.compute_arm_targets(
        torso,
        points["shoulder"],
        points["elbow"],
        points["wrist"],
        other_points["shoulder"],
        side,
        None,
        use_human_mode=False,
    )
    return {
        "shoulder_pitch": _safe_float(t1),
        "shoulder_roll": _safe_float(t2),
        "elbow_yaw": _safe_float(t3),
        "elbow_roll": _safe_float(t4),
        "wrist_yaw": 0.0,
    }


def _runtime_arm_keys(side):
    arm_label = side.capitalize()
    wrist_alias = "LWristYaw" if side == "left" else "RWristYaw"
    return {
        "shoulder_pitch": f"ShoulderPitch_{arm_label}",
        "shoulder_roll": f"ShoulderRoll_{arm_label}",
        "elbow_yaw": f"ElbowYaw_{arm_label}",
        "elbow_roll": f"ElbowRoll_{arm_label}",
        "wrist_yaw": f"WristYaw_{arm_label}",
        "wrist_alias": wrist_alias,
    }


def _compact_to_runtime_angles(angles, side):
    keys = _runtime_arm_keys(side)
    runtime = {
        keys["shoulder_pitch"]: angles.get("shoulder_pitch"),
        keys["shoulder_roll"]: angles.get("shoulder_roll"),
        keys["elbow_yaw"]: angles.get("elbow_yaw"),
        keys["elbow_roll"]: angles.get("elbow_roll"),
        keys["wrist_yaw"]: angles.get("wrist_yaw", 0.0),
        keys["wrist_alias"]: angles.get("wrist_yaw", 0.0),
    }
    return {key: value for key, value in runtime.items() if value is not None}


def _runtime_to_compact_angles(runtime_angles, side):
    keys = _runtime_arm_keys(side)
    return {
        "shoulder_pitch": _safe_float(runtime_angles.get(keys["shoulder_pitch"])),
        "shoulder_roll": _safe_float(runtime_angles.get(keys["shoulder_roll"])),
        "elbow_yaw": _safe_float(runtime_angles.get(keys["elbow_yaw"])),
        "elbow_roll": _safe_float(runtime_angles.get(keys["elbow_roll"])),
        "wrist_yaw": _safe_float(runtime_angles.get(keys["wrist_yaw"], runtime_angles.get(keys["wrist_alias"], 0.0))),
    }


def _compute_current_solution_runtime(frame_data):
    human_angles = {}
    pepper_angles = {}

    for side in ARM_SIDES:
        other_side = "left" if side == "right" else "right"
        torso = _as_vector(frame_data.get("torso"))
        points = _frame_points(frame_data, side)
        other_points = _frame_points(frame_data, other_side)

        required = (torso, points["shoulder"], points["elbow"], points["wrist"], other_points["shoulder"])
        if any(item is None for item in required):
            raise ValueError(f"missing {side} coordinates")

        human_values = scaling.compute_arm_targets(
            torso,
            points["shoulder"],
            points["elbow"],
            points["wrist"],
            other_points["shoulder"],
            side,
            None,
            use_human_mode=True,
        )
        pepper_values = scaling.compute_arm_targets(
            torso,
            points["shoulder"],
            points["elbow"],
            points["wrist"],
            other_points["shoulder"],
            side,
            None,
            use_human_mode=False,
        )

        human_compact = {
            "shoulder_pitch": _safe_float(human_values[0]),
            "shoulder_roll": _safe_float(human_values[1]),
            "elbow_yaw": _safe_float(human_values[2]),
            "elbow_roll": _safe_float(human_values[3]),
            "wrist_yaw": 0.0,
        }
        pepper_compact = {
            "shoulder_pitch": _safe_float(pepper_values[0]),
            "shoulder_roll": _safe_float(pepper_values[1]),
            "elbow_yaw": _safe_float(pepper_values[2]),
            "elbow_roll": _safe_float(pepper_values[3]),
            "wrist_yaw": 0.0,
        }

        human_angles.update(_compact_to_runtime_angles(human_compact, side))
        pepper_angles.update(_compact_to_runtime_angles(pepper_compact, side))

    return human_angles, pepper_angles


def _orientation_labels_for_frame(frame_data):
    labels = {}
    for side in ARM_SIDES:
        label = _wrist_orientation_label(frame_data, side)
        if label is None:
            continue
        labels[side.capitalize()] = {"primary": str(label).upper()}
    return labels


def _create_hand_orientation_smoothers(alpha=0.3, confidence_threshold=0.85):
    tracked_point_ids = (0, 1, 5, 17)
    return {
        hand: {
            point_id: EMASmoothing(
                alpha=alpha,
                confidence_threshold=confidence_threshold,
            )
            for point_id in tracked_point_ids
        }
        for hand in ("Right", "Left")
    }


def _smooth_hand_points_for_orientation(hand_points_3d, smoothers):
    smoothed_points = {}
    for hand in ("Right", "Left"):
        points = hand_points_3d.get(hand)
        if not isinstance(points, dict) or not points:
            continue

        smoothed_hand = {}
        for point_id in (0, 1, 5, 17):
            point = points.get(point_id)
            if point is None:
                continue
            point = np.asarray(point, dtype=np.float32).reshape(-1)
            if point.size < 3 or not np.isfinite(point[:3]).all():
                continue
            smoothed_hand[point_id] = np.asarray(
                smoothers[hand][point_id].smooth(point[:3], confidence=1.0),
                dtype=np.float32,
            )

        if smoothed_hand:
            smoothed_points[hand] = smoothed_hand
    return smoothed_points


def _runtime_orientation_labels_for_frame(frame_data):
    labels = frame_data.get(RUNTIME_ORIENTATION_KEY, {})
    return labels if isinstance(labels, dict) else {}


def _runtime_wrist_orientation_label(frame_data, side):
    hand_labels = _runtime_orientation_labels_for_frame(frame_data).get(side.capitalize(), {})
    if not isinstance(hand_labels, dict):
        return None
    label = hand_labels.get("primary")
    if label is None or str(label).upper() == "UNKNOWN":
        return None
    return str(label).upper()


def _resolve_current_wom_orientation(
    detected_label,
    method_angles,
    side,
    chains,
    previous_labels,
):
    """Resolve the live-style WOM label for one current-solution arm.

    Prefer the current MediaPipe label, carry the most recent valid MediaPipe
    label for the same hand when detection is missing, and use the same Pepper
    forward-kinematics approximation as the baseline methods when no previous
    MediaPipe label exists. The FK fallback intentionally fixes WristYaw at
    zero, matching the evaluator's existing Darja/original-solution convention.
    """
    normalized = None
    if detected_label is not None:
        candidate = str(detected_label).strip().upper()
        if candidate and candidate != "UNKNOWN":
            normalized = candidate

    if normalized is not None:
        previous_labels[side] = normalized
        return normalized, "mediapipe"

    previous = previous_labels.get(side)
    if previous is not None:
        return previous, "previous_mediapipe"

    if not isinstance(method_angles, dict):
        return None, "fk_zero_wrist_yaw"

    fk_angles = dict(method_angles)
    fk_angles["wrist_yaw"] = 0.0
    return (
        _robot_hand_orientation_label(fk_angles, side, chains),
        "fk_zero_wrist_yaw",
    )


def _populate_runtime_orientation_labels(frames, video_folder, fps):
    """Run the live MediaPipe/palm-normal label path over saved dataset frames."""
    smoothers = _create_hand_orientation_smoothers()
    landmarker = _open_hand_landmarker()
    try:
        previous_timestamp_ms = -1
        for frame_index, frame_data in enumerate(frames):
            frame_data[RUNTIME_ORIENTATION_KEY] = {}
            image_file = frame_data.get("image_file")
            if not image_file:
                continue

            image_path = video_folder / "frames" / image_file
            image = cv2.imread(str(image_path))
            if image is None:
                continue

            timestamp = frame_data.get("timestamp")
            if timestamp is None:
                timestamp_ms = int(round(frame_index * 1000.0 / max(float(fps), 1.0)))
            else:
                timestamp_ms = int(round(float(timestamp) * 1000.0))
            timestamp_ms = max(timestamp_ms, previous_timestamp_ms + 1)
            previous_timestamp_ms = timestamp_ms

            _, hand_points_3d = _extract_hand_points(image, landmarker, timestamp_ms)
            smoothed_points = _smooth_hand_points_for_orientation(hand_points_3d, smoothers)
            frame_data[RUNTIME_ORIENTATION_KEY] = (
                hand_orientation.compute_hand_orientation_labels(smoothed_points)
            )
    finally:
        landmarker.close()


def _compute_current_solution_constrained(frame_data, state):
    frame_key = id(frame_data)
    if state.get("frame_key") == frame_key:
        return state["constrained_by_side"]

    pose_handler = state.setdefault("pose_handler", PoseHandler())
    classifier = state.setdefault("classifier", AngleClassifier())

    human_angles, pepper_angles = _compute_current_solution_runtime(frame_data)
    orientation_labels = _runtime_orientation_labels_for_frame(frame_data)

    singularity_right = classifier.check_singularity_poses_fsm(human_angles, arm="right")
    singularity_left = classifier.check_singularity_poses_fsm(human_angles, arm="left")

    constrained_angles = pepper_angles.copy()
    constrained_angles = pose_handler.apply_singularity_constraints(
        singularity_right,
        constrained_angles,
        orientation_labels,
        arm="right",
    )
    constrained_angles = pose_handler.apply_singularity_constraints(
        singularity_left,
        constrained_angles,
        orientation_labels,
        arm="left",
    )

    state["frame_key"] = frame_key
    state["constrained_by_side"] = {
        side: _runtime_to_compact_angles(constrained_angles, side)
        for side in ARM_SIDES
    }
    state["singularity"] = {
        "right": singularity_right,
        "left": singularity_left,
    }
    return state["constrained_by_side"]


def _compute_ikpy_solution(frame_data, side, chains, joint_indices):
    torso = _as_vector(frame_data.get("torso"))
    wrist = _frame_points(frame_data, side)["wrist"]
    if torso is None or wrist is None:
        raise ValueError("missing coordinates")

    left_chain, right_chain = chains
    ik_result, chain = ikpyu.compute_ik_from_wrist_thor_coordinate_adjusted(
        wrist / 1000.0,
        torso / 1000.0,
        left_chain,
        right_chain,
        side,
        joint_indices=joint_indices,
    )
    angles, names, _ = ikpyu.get_controllable_joints(chain, ik_result)
    by_name = dict(zip(names, angles))

    prefix = "L" if side == "left" else "R"
    return {
        "shoulder_pitch": _safe_float(by_name.get(f"{prefix}ShoulderPitch")),
        "shoulder_roll": _safe_float(by_name.get(f"{prefix}ShoulderRoll")),
        "elbow_yaw": _safe_float(by_name.get(f"{prefix}ElbowYaw")),
        "elbow_roll": _safe_float(by_name.get(f"{prefix}ElbowRoll")),
        "wrist_yaw": 0.0,
    }


def _compute_original_solution(frame_data, side):
    other_side = "left" if side == "right" else "right"
    torso = _as_vector(frame_data.get("torso"))
    points = _frame_points(frame_data, side)
    other_points = _frame_points(frame_data, other_side)

    return original_adapter.compute_arm_angles(
        torso,
        points["shoulder"],
        points["elbow"],
        points["wrist"],
        other_points["shoulder"],
        side,
    )


def _fk_points(angles, side):
    if any(angles.get(key) is None for key in ("shoulder_pitch", "shoulder_roll", "elbow_yaw", "elbow_roll")):
        return None

    t1 = angles["shoulder_pitch"]
    t2 = angles["shoulder_roll"]
    t3 = angles["elbow_yaw"]
    t4 = angles["elbow_roll"]

    return {
        "shoulder": PEPPER_SHOULDER[side].copy(),
        "elbow": fk.get_elbow_position(t1, t2, side),
        "wrist": fk.get_wrist_position(t1, t2, t3, t4, side),
    }


def _chain_for_side(chains, side):
    if chains is None:
        return None
    left_chain, right_chain = chains
    return left_chain if side == "left" else right_chain


def _chain_joint_vector(chain, angles, side):
    prefix = "L" if side == "left" else "R"
    name_to_angle = {
        f"{prefix}ShoulderPitch": "shoulder_pitch",
        f"{prefix}ShoulderRoll": "shoulder_roll",
        f"{prefix}ElbowYaw": "elbow_yaw",
        f"{prefix}ElbowRoll": "elbow_roll",
        f"{prefix}WristYaw": "wrist_yaw",
    }
    q = np.zeros(len(chain.links), dtype=np.float64)
    for index, link in enumerate(chain.links):
        angle_key = name_to_angle.get(link.name)
        if angle_key is not None:
            q[index] = angles.get(angle_key) or 0.0
    return q


def _pepper_direction_label(vector):
    vector = _unit(vector)
    if vector is None:
        return None

    axis_index = int(np.argmax(np.abs(vector)))
    if axis_index == 0:
        return "FRONT" if vector[0] > 0 else "BACK"
    if axis_index == 1:
        return "LEFT" if vector[1] > 0 else "RIGHT"
    return "UP" if vector[2] > 0 else "DOWN"


def _robot_hand_orientation_label(angles, side, chains):
    chain = _chain_for_side(chains, side)
    if chain is None:
        return None
    if any(angles.get(key) is None for key in ("shoulder_pitch", "shoulder_roll", "elbow_yaw", "elbow_roll")):
        return None

    transform = chain.forward_kinematics(_chain_joint_vector(chain, angles, side))
    rotation = transform[:3, :3]
    # Approximation: Pepper has no MediaPipe-like palm keypoints in the IK result.
    # The negative local Z axis is used as the palm-facing normal of the hand frame.
    hand_normal = ROBOT_HAND_NORMAL_SIGN * rotation[:, ROBOT_HAND_NORMAL_AXIS]
    return _pepper_direction_label(hand_normal)


def _unit(vector):
    if vector is None:
        return None
    norm = np.linalg.norm(vector)
    if norm < 1e-8 or not np.isfinite(norm):
        return None
    return vector / norm


def _eea(points):
    if points is None or points.get("shoulder") is None or points.get("elbow") is None:
        return None
    vector = points["elbow"] - points["shoulder"]
    norm = np.linalg.norm(vector)
    if norm < 1e-8:
        return None
    return float(vector[2] / norm)


def _ee_ar(human_points, robot_points):
    human_eea = _eea(human_points)
    robot_eea = _eea(robot_points)
    if human_eea is None or robot_eea is None:
        return None
    return abs(human_eea - robot_eea)


def _so_ax(human_points, robot_points):
    if human_points is None or robot_points is None:
        return None
    human_vec = _unit(human_points["elbow"] - human_points["shoulder"])
    robot_vec = _unit(robot_points["elbow"] - robot_points["shoulder"])
    if human_vec is None or robot_vec is None:
        return None
    return float(np.linalg.norm(human_vec - robot_vec))


def _joint_range(angle_key, side):
    value = JOINT_RANGES[angle_key]
    if isinstance(value, dict):
        return value[side]
    return value


def _hj_ar(gt_angles, method_angles, side):
    errors = []
    for angle_key in ANGLE_KEYS:
        gt_value = gt_angles.get(angle_key)
        method_value = method_angles.get(angle_key)
        if gt_value is None or method_value is None:
            continue
        min_value, max_value = _joint_range(angle_key, side)
        denom = max_value - min_value
        if abs(denom) < 1e-8:
            continue
        errors.append(((gt_value - method_value) / denom) ** 2)
    if not errors:
        return None
    return float(np.mean(errors))


def _hjl_diagnostics(method_angles, side):
    required_keys = (
        "shoulder_pitch",
        "shoulder_roll",
        "elbow_yaw",
        "elbow_roll",
    )
    radians_by_key = {
        key: _safe_float(method_angles.get(key))
        for key in required_keys
    }
    missing = [key for key, value in radians_by_key.items() if value is None]
    if missing:
        return {
            "valid": False,
            "indicator": None,
            "invalid_reason": f"missing or invalid angles: {', '.join(missing)}",
            "angles_deg": None,
            "singularity": None,
            "classification": None,
            "violations": [],
        }

    runtime_angles = _compact_to_runtime_angles(radians_by_key, side)
    singularity = AngleClassifier().check_singularity_poses_fsm(
        runtime_angles,
        arm=side,
    )
    classification = HUMAN_ARM_CLASSIFIER.classify(
        sp=radians_by_key["shoulder_pitch"],
        ey=radians_by_key["elbow_yaw"],
        arm=side,
        singularity=singularity,
    )
    angles_deg = {
        key: float(math.degrees(value))
        for key, value in radians_by_key.items()
    }

    violations = []
    if classification == "NON_HUMAN_DOABLE":
        violations.append(
            {
                "arm": side,
                "joint": "shoulder_pitch/elbow_yaw",
                "motion": "non_human_doable_combination",
                "shoulder_pitch_deg": angles_deg["shoulder_pitch"],
                "elbow_yaw_deg": angles_deg["elbow_yaw"],
                "singularity_type": singularity.get("singularity_type"),
            }
        )

    return {
        "valid": True,
        "indicator": 1 if violations else 0,
        "invalid_reason": None,
        "angles_deg": angles_deg,
        "singularity": singularity,
        "classification": classification or "NOT_APPLICABLE",
        "violations": violations,
    }


def _wom(gt_orientation, method_orientation):
    if gt_orientation is None or method_orientation is None:
        return None
    return 1 if str(gt_orientation).upper() == str(method_orientation).upper() else 0


def _wrist_orientation_label(frame_data, side):
    arm_data = frame_data.get(f"{side}_arm", {})
    if not isinstance(arm_data, dict):
        return None
    return arm_data.get("wrist_orientation")


def _frame_metrics(
    frame_data,
    side,
    method_angles,
    method_orientation=None,
    synergy_model=None,
):
    human_points = _frame_points(frame_data, side, pepper_coordinates=True)
    robot_points = _fk_points(method_angles, side)
    gt_angles = _ground_truth_angles(frame_data, side)
    gt_orientation = frame_data.get(f"{side}_arm", {}).get("wrist_orientation")

    return {
        "EEAh": _eea(human_points),
        "EEAr": _ee_ar(human_points, robot_points),
        "SOAx": _so_ax(human_points, robot_points),
        "HJL": None,
        "HJL_diagnostics": _hjl_diagnostics(method_angles, side),
        "WOM": _wom(gt_orientation, method_orientation),
        "HJAr": _hj_ar(gt_angles, method_angles, side),
        "TSE": None,
        "SYN": _syn(robot_points, synergy_model),
    }


def _compute_tse_for_method(method_frames):
    by_side = {side: [] for side in ARM_SIDES}
    for frame_result in method_frames:
        for side in ARM_SIDES:
            angles = frame_result["arms"][side].get("angles")
            if angles is None:
                by_side[side].append(None)
            else:
                by_side[side].append(np.asarray([angles.get(key, np.nan) for key in ANGLE_KEYS], dtype=np.float64))

    for frame_index, frame_result in enumerate(method_frames):
        for side in ARM_SIDES:
            if frame_index < 2:
                frame_result["arms"][side]["metrics"]["TSE"] = None
                continue
            q0 = by_side[side][frame_index]
            q1 = by_side[side][frame_index - 1]
            q2 = by_side[side][frame_index - 2]
            if q0 is None or q1 is None or q2 is None:
                frame_result["arms"][side]["metrics"]["TSE"] = None
                continue
            if not (np.isfinite(q0).all() and np.isfinite(q1).all() and np.isfinite(q2).all()):
                frame_result["arms"][side]["metrics"]["TSE"] = None
                continue
            d = q0 - 2.0 * q1 + q2
            frame_result["arms"][side]["metrics"]["TSE"] = float(np.linalg.norm(d) ** 2)


def _mean(values):
    clean = [value for value in values if value is not None]
    if not clean:
        return None
    return float(np.mean(clean))


def _summarize(video_results):
    summary = {}
    for method_name, method_data in video_results["methods"].items():
        method_summary = {}
        for side in ARM_SIDES:
            metric_summary = {}
            for metric_name in METRIC_NAMES:
                values = [
                    frame["arms"][side]["metrics"].get(metric_name)
                    for frame in method_data["frames"]
                    if frame["arms"][side].get("metrics")
                ]
                metric_summary[metric_name] = _mean(values)
            method_summary[side] = metric_summary
        summary[method_name] = method_summary
    return summary


def _method_result(method_name, frame_data, side, chains=None, joint_indices=None, method_state=None):
    if method_name == "ground_truth":
        return _ground_truth_angles(frame_data, side), _wrist_orientation_label(frame_data, side), None
    if method_name == "current_solution_raw":
        angles = _compute_current_solution(frame_data, side)
        return angles, _runtime_wrist_orientation_label(frame_data, side), None
    if method_name == "current_solution_constrained":
        constrained_by_side = _compute_current_solution_constrained(frame_data, method_state)
        return constrained_by_side[side], _runtime_wrist_orientation_label(frame_data, side), None
    if method_name == "ikpy":
        angles = _compute_ikpy_solution(frame_data, side, chains, joint_indices)
        return angles, _robot_hand_orientation_label(angles, side, chains), None
    if method_name == "original_solution":
        angles = _compute_original_solution(frame_data, side)
        return angles, _robot_hand_orientation_label(angles, side, chains), None
    raise ValueError(method_name)


def evaluate_file(path, methods, chains=None, joint_indices=None, synergy_model=None):
    annotations = _load_json(path)
    frames = annotations.get("frames", [])
    if any(method in methods for method in ("current_solution_raw", "current_solution_constrained")):
        _populate_runtime_orientation_labels(
            frames,
            path.parent,
            annotations.get("fps", 8),
        )
    method_states = {
        "current_solution_constrained": {
            "pose_handler": PoseHandler(),
            "classifier": AngleClassifier(),
        }
    }
    current_wom_previous_labels = {
        method_name: {}
        for method_name in methods
        if method_name in ("current_solution_raw", "current_solution_constrained")
    }
    video_result = {
        "video_id": annotations.get("video_id", path.parent.name),
        "category": annotations.get("category", path.parent.parent.name),
        "annotation_file": str(path),
        "methods": {
            method_name: {
                "frames": [],
            }
            for method_name in methods
        },
    }

    for frame_data in frames:
        for method_name in methods:
            frame_result = {
                "frame_id": frame_data.get("frame_id"),
                "timestamp": frame_data.get("timestamp"),
                "image_file": frame_data.get("image_file"),
                "arms": {},
            }
            for side in ARM_SIDES:
                try:
                    angles, orientation, error = _method_result(
                        method_name,
                        frame_data,
                        side,
                        chains=chains,
                        joint_indices=joint_indices,
                        method_state=method_states.setdefault(method_name, {}),
                    )
                except Exception as exc:
                    angles, orientation, error = None, None, str(exc)

                detected_orientation = orientation
                wom_orientation_source = None
                if method_name in current_wom_previous_labels:
                    orientation, wom_orientation_source = _resolve_current_wom_orientation(
                        detected_orientation,
                        angles,
                        side,
                        chains,
                        current_wom_previous_labels[method_name],
                    )

                if angles is None:
                    arm_result = {
                        "angles": None,
                        "metrics": {
                            metric_name: None for metric_name in METRIC_NAMES
                        },
                        "error": error,
                    }
                    if method_name in current_wom_previous_labels:
                        arm_result.update({
                            "wom_orientation_detected_label": detected_orientation,
                            "wom_orientation_label": orientation,
                            "wom_orientation_source": wom_orientation_source,
                        })
                    frame_result["arms"][side] = arm_result
                    continue

                frame_result["arms"][side] = {
                    "angles": angles,
                    "wom_orientation_label": orientation,
                    "metrics": _frame_metrics(
                        frame_data,
                        side,
                        angles,
                        method_orientation=orientation,
                        synergy_model=synergy_model,
                    ),
                    "error": error,
                }
                if method_name in current_wom_previous_labels:
                    frame_result["arms"][side].update({
                        "wom_orientation_detected_label": detected_orientation,
                        "wom_orientation_source": wom_orientation_source,
                    })

            arm_diagnostics = {
                side: frame_result["arms"][side].get("metrics", {}).get("HJL_diagnostics")
                for side in ARM_SIDES
            }
            hjl_frame_valid = all(
                isinstance(diagnostics, dict) and diagnostics.get("valid")
                for diagnostics in arm_diagnostics.values()
            )
            hjl_violations = [
                violation
                for diagnostics in arm_diagnostics.values()
                if isinstance(diagnostics, dict)
                for violation in diagnostics.get("violations", [])
            ]
            hjl_indicator = (1 if hjl_violations else 0) if hjl_frame_valid else None
            for side in ARM_SIDES:
                frame_result["arms"][side]["metrics"]["HJL"] = hjl_indicator
            frame_result["hjl"] = {
                "valid": hjl_frame_valid,
                "indicator": hjl_indicator,
                "violations": hjl_violations,
                "invalid_arms": [
                    side
                    for side, diagnostics in arm_diagnostics.items()
                    if not isinstance(diagnostics, dict) or not diagnostics.get("valid")
                ],
            }

            video_result["methods"][method_name]["frames"].append(frame_result)

    for method_name in methods:
        _compute_tse_for_method(video_result["methods"][method_name]["frames"])

    video_result["summary"] = _summarize(video_result)
    video_result["hjl_sequences"] = {}
    for method_name, method_data in video_result["methods"].items():
        frame_indicators = [
            frame.get("hjl", {}).get("indicator")
            for frame in method_data["frames"]
        ]
        valid_indicators = [
            indicator for indicator in frame_indicators if indicator is not None
        ]
        cause_counts = {}
        for frame in method_data["frames"]:
            for violation in frame.get("hjl", {}).get("violations", []):
                cause = f"{violation['arm']}.{violation['motion']}"
                cause_counts[cause] = cause_counts.get(cause, 0) + 1
        video_result["hjl_sequences"][method_name] = {
            "score": _mean(valid_indicators),
            "valid_frames": len(valid_indicators),
            "invalid_frames": len(frame_indicators) - len(valid_indicators),
            "violating_frames": sum(valid_indicators),
            "violation_cause_counts": cause_counts,
        }

    video_result["syn_sequences"] = {}
    for method_name, method_data in video_result["methods"].items():
        method_sequences = {}
        for side in ARM_SIDES:
            values = [
                frame["arms"][side]["metrics"].get("SYN")
                for frame in method_data["frames"]
                if frame["arms"][side].get("metrics")
            ]
            valid_values = [value for value in values if value is not None]
            method_sequences[side] = {
                "score": _mean(valid_values),
                "valid_frames": len(valid_values),
                "rejected_frames": len(values) - len(valid_values),
            }
        video_result["syn_sequences"][method_name] = method_sequences
    return video_result


def _hjl_dataset_summary(videos, methods):
    result = {"overall": {}, "by_category": {}}
    for method_name in methods:
        overall_scores = []
        category_scores = {}
        overall_causes = {}
        for video in videos:
            category = video.get("category", "unknown")
            sequence = video.get("hjl_sequences", {}).get(method_name, {})
            score = sequence.get("score")
            if score is not None:
                overall_scores.append(score)
                category_scores.setdefault(category, []).append(score)
            for cause, count in sequence.get("violation_cause_counts", {}).items():
                overall_causes[cause] = overall_causes.get(cause, 0) + count

        result["overall"][method_name] = {
            "mean": _mean(overall_scores),
            "std": float(np.std(overall_scores)) if overall_scores else None,
            "sequence_count": len(overall_scores),
            "violation_cause_counts": overall_causes,
        }
        for category, scores in category_scores.items():
            result["by_category"].setdefault(category, {})[method_name] = {
                "mean": _mean(scores),
                "std": float(np.std(scores)) if scores else None,
                "sequence_count": len(scores),
            }
    return result


def _syn_dataset_summary(videos, methods):
    result = {"overall": {}, "by_category": {}}
    for method_name in methods:
        overall_scores = []
        category_scores = {}
        for video in videos:
            category = video.get("category", "unknown")
            sequences = video.get("syn_sequences", {}).get(method_name, {})
            for side in ARM_SIDES:
                score = sequences.get(side, {}).get("score")
                if score is None:
                    continue
                overall_scores.append(score)
                category_scores.setdefault(category, []).append(score)

        result["overall"][method_name] = {
            "mean": _mean(overall_scores),
            "std": float(np.std(overall_scores)) if overall_scores else None,
            "sequence_count": len(overall_scores),
        }
        for category, scores in category_scores.items():
            result["by_category"].setdefault(category, {})[method_name] = {
                "mean": _mean(scores),
                "std": float(np.std(scores)) if scores else None,
                "sequence_count": len(scores),
            }
    return result


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate IK methods against annotated dataset frames using Chapter 4-style metrics."
    )
    parser.add_argument(
        "input",
        nargs="?",
        type=Path,
        default=REPO_ROOT / "dataset" / "pepper_singularity_motions",
        help="annotations_filled.json, video folder, category folder, or dataset folder.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO_ROOT / "04-evaluation" / "results" / "ik_method_metrics.json",
        help="Output JSON path.",
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        choices=METHODS,
        default=["ground_truth", "current_solution_raw", "current_solution_constrained", "ikpy", "original_solution"],
        help="Methods to evaluate.",
    )
    parser.add_argument(
        "--synergy-reference",
        type=Path,
        default=REPO_ROOT / "dataset" / "synergy_reference_motions",
        help="Human annotations used only to fit the SYN PCA model.",
    )
    parser.add_argument(
        "--synergy-variance-threshold",
        type=float,
        default=0.95,
        help="Cumulative explained-variance threshold for retained synergies.",
    )
    parser.add_argument(
        "--max-video-index",
        type=int,
        default=None,
        help=(
            "Evaluate only video_XXX folders whose numeric index is at or below "
            "this value, independently in every category."
        ),
    )
    return parser.parse_args()


def main():
    args = parse_args()
    files = _annotation_files(args.input)
    if not files:
        raise RuntimeError(f"No annotations_filled.json files found under {args.input}")
    if not 0.0 < args.synergy_variance_threshold <= 1.0:
        raise ValueError("--synergy-variance-threshold must be in (0, 1].")

    synergy_files = _annotation_files(args.synergy_reference)
    if not synergy_files:
        raise RuntimeError(
            f"No annotations_filled.json files found under synergy reference "
            f"{args.synergy_reference}. Run Metrabs annotation and angle preparation first."
        )
    synergy_root = args.synergy_reference.resolve()
    files = [
        path
        for path in files
        if synergy_root not in path.resolve().parents and path.resolve() != synergy_root
    ]
    files = _filter_annotation_files_by_max_video_index(files, args.max_video_index)
    if not files:
        raise RuntimeError(
            "No evaluation files remain after excluding the synergy reference "
            "dataset and applying the video-index limit."
        )
    if args.max_video_index is not None:
        print(
            f"[info] evaluating video indices through video_{args.max_video_index:03d}: "
            f"{len(files)} annotation files"
        )

    print(f"[info] fitting human synergy space from {len(synergy_files)} annotation files")
    synergy_model = _build_synergy_model(
        synergy_files,
        variance_threshold=args.synergy_variance_threshold,
    )
    print(
        f"[info] retained {synergy_model['component_count']} components "
        f"({synergy_model['cumulative_explained_variance']:.2%} cumulative variance)"
    )

    chains = None
    joint_indices = {
        "rwri_smpl": 4,
        "lwri_smpl": 7,
        "thor_smpl": 8,
    }
    if any(
        method in args.methods
        for method in (
            "current_solution_raw",
            "current_solution_constrained",
            "ikpy",
            "original_solution",
        )
    ):
        with contextlib.redirect_stdout(io.StringIO()):
            chains = ikpyu.load_pepper_chains(str(DL_POSE_DIR))

    results = {
        "dataset": str(args.input),
        "output_file": str(args.output),
        "metrics": {
            "EEAh": "Normalized human elbow elevation reference value.",
            "EEAr": "Reference elbow elevation error, lower is better.",
            "SOAx": "Normalized shoulder-to-elbow direction error, lower is better.",
            "HJL": "Ratio of valid two-arm frames with a non-human-doable ShoulderPitch/ElbowYaw combination during a ShoulderRoll or dual singularity, lower is better.",
            "WOM": "Wrist orientation match between the annotated human hand label and the method's robot hand label.",
            "HJAr": "Normalized joint angle similarity to stored annotation angles, lower is better.",
            "TSE": "Second-order joint trajectory smoothness error, lower is better.",
            "SYN": "Human-synergy-space posture reconstruction error, lower is better.",
        },
        "synergy_model": {
            "reference_dataset": str(args.synergy_reference),
            **synergy_model,
        },
        "hjl_configuration": {
            "classifier": "HumanArmClassifier",
            "invalid_frame_policy": "exclude_and_report",
            "classification_angles": [
                "shoulder_pitch",
                "elbow_yaw",
            ],
            "singularity_gate_angles": [
                "shoulder_roll",
                "elbow_roll",
            ],
            "applicable_singularity_types": [
                "shoulder_roll",
                "dual",
            ],
            "not_directly_classified": [
                "wrist_yaw",
                "head_yaw",
                "head_pitch",
            ],
        },
        "notes": [
            "wrist_yaw is treated as 0.0 for all currently implemented methods.",
            "current_solution_raw is the current arm-scaling output before runtime singularity handling.",
            "current_solution_constrained applies the current runtime PoseHandler dual-singularity FSM in frame order.",
            "current_solution_raw and current_solution_constrained WOM use the current MediaPipe hand-orientation label when available, otherwise carry forward the most recent valid label for that hand within the video, and otherwise derive a Pepper hand label through forward kinematics with WristYaw fixed at 0.0.",
            "ikpy WOM derives a robot hand orientation label from Pepper hand-frame forward kinematics instead of copying the human hand annotation.",
            "The ikpy robot hand orientation label uses the negative local Z axis of the Pepper hand frame as a palm-normal approximation.",
            "original_solution uses an offline adapter around the cloned original kinematics math; the cloned original repository is not modified.",
            "HJAr uses stored annotation angles as reference; if those were generated by current_solution, current_solution HJAr is a sanity check, not independent validation.",
            "HJL uses HumanArmClassifier's side-specific piecewise-linear ShoulderPitch/ElbowYaw reachability envelope when a ShoulderRoll or dual singularity is present.",
            "HJL is one binary value for the complete two-arm frame. NON_HUMAN_DOABLE on either arm sets the frame to 1; missing required angles exclude the frame and are reported separately.",
            "Frames without a ShoulderRoll or dual singularity are classified as not applicable by HumanArmClassifier and contribute a valid HJL value of 0.",
            "SYN uses one PCA model fitted only on normalized human upper-arm and forearm direction descriptors from the synergy reference dataset.",
            "Each video-side arm is treated as one sequence for SYN aggregation; dataset summaries report the mean and standard deviation across those sequence scores.",
        ],
        "videos": [],
    }

    for path in files:
        print(f"[info] evaluating {path}")
        results["videos"].append(
            evaluate_file(
                path,
                args.methods,
                chains=chains,
                joint_indices=joint_indices,
                synergy_model=synergy_model,
            )
        )

    results["hjl_summary"] = _hjl_dataset_summary(results["videos"], args.methods)
    results["syn_summary"] = _syn_dataset_summary(results["videos"], args.methods)
    _save_json(args.output, _json_safe(results))
    print(f"[done] wrote {args.output}")


if __name__ == "__main__":
    main()
