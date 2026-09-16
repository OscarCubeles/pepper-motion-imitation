import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

CURRENT_FILE = Path(__file__).resolve()

DL_POSE_DIR = CURRENT_FILE.parents[1]   # 01-server/dl-pose
SERVER_DIR = CURRENT_FILE.parents[2]    # 01-server
REPO_ROOT = CURRENT_FILE.parents[3]     # Pepper-Imitation-System

from server.common import settings  # noqa: E402
from server.common import pose_mapping as pose_map  # noqa: E402

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}
ARM_SIDES = ("left", "right")
ARM_JOINTS = ("shoulder", "elbow", "wrist")
JOINT_ANGLE_KEYS = ("shoulder_pitch", "shoulder_roll", "elbow_yaw", "elbow_roll")


def _json_safe(value):
    if value is None:
        return None
    if isinstance(value, np.ndarray):
        return value.tolist()
    if torch.is_tensor(value):
        return value.detach().cpu().numpy().tolist()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def _to_numpy(tensor):
    if tensor is None:
        return None
    if torch.is_tensor(tensor):
        return tensor.detach().cpu().numpy()
    return np.asarray(tensor)


def _empty_arm_data():
    return {
        "shoulder": None,
        "elbow": None,
        "wrist": None,
        "joint_angles": {key: None for key in JOINT_ANGLE_KEYS},
        "wrist_orientation": None,
    }


def _load_base_annotations(video_folder, reset_from_template=False):
    filled_path = video_folder / "annotations_filled.json"
    template_path = video_folder / "annotations.json"

    if filled_path.exists() and not reset_from_template:
        with filled_path.open("r", encoding="utf-8") as file:
            return json.load(file)

    if not template_path.exists():
        return {
            "video_id": video_folder.name,
            "category": video_folder.parent.name,
            "frames": [],
        }

    with template_path.open("r", encoding="utf-8") as file:
        return json.load(file)


def _save_annotations(annotations_path, annotations):
    with annotations_path.open("w", encoding="utf-8") as file:
        json.dump(annotations, file, indent=4)


def _frame_files(frames_folder):
    return sorted(
        path for path in frames_folder.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def _find_video_folders(input_path):
    input_path = input_path.resolve()

    if input_path.is_file() and input_path.suffix.lower() in IMAGE_EXTENSIONS:
        return []

    if (input_path / "frames").is_dir():
        return [input_path]

    return sorted(
        path for path in input_path.rglob("frames")
        if path.is_dir()
    )


def _draw_metrabs_overlay(frame, poses2d, joint_edges, suppress_joint_indices):
    overlay = frame.copy()
    if poses2d is None or len(poses2d) == 0:
        cv2.putText(
            overlay,
            "No Metrabs pose detected",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0, 0, 255),
            2,
            cv2.LINE_AA,
        )
        return overlay

    suppressed = set(int(idx) for idx in suppress_joint_indices)
    colors = [
        (0, 255, 255),
        (255, 128, 0),
        (0, 200, 0),
        (255, 0, 255),
    ]

    for person_idx, person_poses2d in enumerate(poses2d):
        color = colors[person_idx % len(colors)]
        points = np.asarray(person_poses2d, dtype=np.float32)

        for edge in joint_edges:
            joint_a, joint_b = int(edge[0]), int(edge[1])
            if joint_a in suppressed or joint_b in suppressed:
                continue
            if joint_a >= len(points) or joint_b >= len(points):
                continue
            point_a = points[joint_a]
            point_b = points[joint_b]
            if not np.isfinite(point_a[:2]).all() or not np.isfinite(point_b[:2]).all():
                continue

            cv2.line(
                overlay,
                tuple(np.round(point_a[:2]).astype(int)),
                tuple(np.round(point_b[:2]).astype(int)),
                color,
                2,
                cv2.LINE_AA,
            )

        for joint_idx, point in enumerate(points):
            if joint_idx in suppressed or not np.isfinite(point[:2]).all():
                continue
            cv2.circle(
                overlay,
                tuple(np.round(point[:2]).astype(int)),
                3,
                (0, 0, 255),
                -1,
                cv2.LINE_AA,
            )

    return overlay


def _extract_metrabs_predictions(pred):
    poses2d = _to_numpy(pred.get("poses2d") if pred else None)
    poses3d = _to_numpy(pred.get("poses3d") if pred else None)

    return poses3d, poses2d


def _remove_unwanted_metrabs_fields(frame_data):
    frame_data.pop("metrabs", None)


def _ensure_arm_schema(frame_data):
    frame_data.setdefault("torso", None)

    for side in ARM_SIDES:
        arm_key = f"{side}_arm"
        arm_data = frame_data.get(arm_key)
        if not isinstance(arm_data, dict):
            arm_data = _empty_arm_data()
            frame_data[arm_key] = arm_data

        for joint in ARM_JOINTS:
            arm_data.setdefault(joint, None)

        joint_angles = arm_data.get("joint_angles")
        if not isinstance(joint_angles, dict):
            joint_angles = {}
            arm_data["joint_angles"] = joint_angles
        for key in JOINT_ANGLE_KEYS:
            joint_angles.setdefault(key, None)

        legacy_orientation = frame_data.get(f"wrist_orientation_{side}")
        if arm_data.get("wrist_orientation") is None and legacy_orientation is not None:
            arm_data["wrist_orientation"] = legacy_orientation
        else:
            arm_data.setdefault("wrist_orientation", None)


def _normalize_name(name):
    return "".join(ch for ch in str(name).lower() if ch.isalnum())


def _matches_side(name, side):
    if side == "left":
        return "left" in name or name.startswith("l")
    return "right" in name or name.startswith("r")


def _matches_joint(name, joint):
    aliases = {
        "shoulder": ("shoulder", "sho"),
        "elbow": ("elbow", "elb"),
        "wrist": ("wrist", "wri"),
    }
    return any(alias in name for alias in aliases[joint])


def _resolve_arm_joint_indices(joint_names):
    indices = {side: {} for side in ARM_SIDES}
    for joint_idx, joint_name in enumerate(joint_names):
        normalized = _normalize_name(joint_name)
        for side in ARM_SIDES:
            if not _matches_side(normalized, side):
                continue
            for joint in ARM_JOINTS:
                if _matches_joint(normalized, joint):
                    indices[side].setdefault(joint, joint_idx)
    return indices


def _resolve_torso_joint_index(joint_names, torso_joint=None):
    if torso_joint is not None:
        try:
            torso_idx = int(torso_joint)
            if 0 <= torso_idx < len(joint_names):
                return torso_idx
        except ValueError:
            requested_name = _normalize_name(torso_joint)
            for joint_idx, joint_name in enumerate(joint_names):
                if _normalize_name(joint_name) == requested_name:
                    return joint_idx

            for joint_idx, joint_name in enumerate(joint_names):
                if requested_name in _normalize_name(joint_name):
                    return joint_idx

        raise ValueError(f"Could not resolve torso joint: {torso_joint}")

    try:
        if getattr(settings, "SKELETON", None) == getattr(settings, "SKELETON_SMPL_HEAD_30", "smpl+head_30"):
            torso_idx = 9
        else:
            torso_idx = int(settings.SOURCE_BODY_IDXS[8])
        if 0 <= torso_idx < len(joint_names):
            return torso_idx
    except (AttributeError, IndexError, TypeError, ValueError):
        pass

    for preferred_name in ("spine", "pelvis", "spinechest", "thorax", "torso"):
        for joint_idx, joint_name in enumerate(joint_names):
            if _normalize_name(joint_name) == preferred_name:
                return joint_idx
    return None


def _set_points_from_metrabs(frame_data, poses3d, arm_joint_indices, torso_joint_index, overwrite=False):
    _remove_unwanted_metrabs_fields(frame_data)
    _ensure_arm_schema(frame_data)

    if poses3d is None or len(poses3d) == 0:
        return False

    first_pose = np.asarray(poses3d[0], dtype=np.float32)
    filled_any = False

    if (frame_data.get("torso") is None or overwrite) and torso_joint_index is not None:
        if torso_joint_index < len(first_pose):
            torso = first_pose[torso_joint_index, :3]
            if np.isfinite(torso).all():
                frame_data["torso"] = _json_safe(torso)
                filled_any = True

    for side in ARM_SIDES:
        arm_data = frame_data[f"{side}_arm"]
        for joint in ARM_JOINTS:
            if arm_data.get(joint) is not None and not overwrite:
                continue

            joint_idx = arm_joint_indices.get(side, {}).get(joint)
            if joint_idx is None or joint_idx >= len(first_pose):
                continue

            point = first_pose[joint_idx, :3]
            if not np.isfinite(point).all():
                continue

            arm_data[joint] = _json_safe(point)
            filled_any = True

    return filled_any


def _annotation_by_image_file(annotations):
    frames = annotations.setdefault("frames", [])
    return {
        frame.get("image_file"): frame
        for frame in frames
        if isinstance(frame, dict) and frame.get("image_file")
    }


def _ensure_frame_annotation(annotations, frame_path, frame_idx):
    by_image = _annotation_by_image_file(annotations)
    existing = by_image.get(frame_path.name)
    if existing is not None:
        return existing

    frame_data = {
        "frame_id": frame_idx,
        "timestamp": None,
        "image_file": frame_path.name,
    }
    annotations.setdefault("frames", []).append(frame_data)
    return frame_data


def process_video_folder(
    video_folder,
    model,
    joint_names,
    joint_edges,
    suppress_joint_indices,
    intrinsic_matrix,
    distortion_coeffs,
    skeleton,
    save_visualizations=True,
    overwrite=False,
    reset_from_template=False,
    torso_joint=None,
    max_frames=None,
):
    frames_folder = video_folder / "frames"
    frame_paths = _frame_files(frames_folder)
    if max_frames is not None:
        frame_paths = frame_paths[:max_frames]

    if not frame_paths:
        print(f"[skip] No frames found in {frames_folder}")
        return

    annotations = _load_base_annotations(video_folder, reset_from_template=reset_from_template)
    output_annotations_path = video_folder / "annotations_filled.json"
    visualization_folder = video_folder / "metrabs_visualizations"
    if save_visualizations:
        visualization_folder.mkdir(parents=True, exist_ok=True)

    print(f"[info] Processing {video_folder} ({len(frame_paths)} frames)")
    arm_joint_indices = _resolve_arm_joint_indices(joint_names)
    torso_joint_index = _resolve_torso_joint_index(joint_names, torso_joint=torso_joint)
    torso_joint_name = joint_names[torso_joint_index] if torso_joint_index is not None else None
    print(f"[info] torso joint index: {torso_joint_index}, name: {torso_joint_name}")

    for frame_idx, frame_path in enumerate(frame_paths):
        frame_data = _ensure_frame_annotation(annotations, frame_path, frame_idx)
        _remove_unwanted_metrabs_fields(frame_data)
        _ensure_arm_schema(frame_data)

        image = cv2.imread(str(frame_path))
        if image is None:
            frame_data["metrabs_error"] = f"Could not read image: {frame_path.name}"
            continue

        try:
            pred = pose_map._detect_poses(
                model,
                image,
                intrinsic_matrix,
                distortion_coeffs,
                skeleton=skeleton,
            )
            poses3d, poses2d = _extract_metrabs_predictions(pred)
            filled_any = _set_points_from_metrabs(
                frame_data,
                poses3d,
                arm_joint_indices,
                torso_joint_index,
                overwrite=overwrite,
            )
            frame_data.pop("metrabs_error", None)
        except Exception as exc:
            filled_any = False
            frame_data["metrabs_error"] = str(exc)
            poses2d = None

        if save_visualizations:
            output_name = f"{frame_path.stem}_metrabs.jpg"
            output_path = visualization_folder / output_name
            overlay = _draw_metrabs_overlay(
                image,
                poses2d,
                joint_edges,
                suppress_joint_indices,
            )
            cv2.imwrite(str(output_path), overlay)
            frame_data["metrabs_visualization"] = str(
                output_path.relative_to(video_folder).as_posix()
            )

        print(f"  {frame_path.name}: filled_points={filled_any}")

    _save_annotations(output_annotations_path, annotations)
    print(f"[done] Wrote {output_annotations_path}")


def process_single_image(
    image_path,
    model,
    joint_names,
    joint_edges,
    suppress_joint_indices,
    intrinsic_matrix,
    distortion_coeffs,
    skeleton,
):
    image = cv2.imread(str(image_path))
    if image is None:
        raise RuntimeError(f"Could not read image: {image_path}")

    pred = pose_map._detect_poses(
        model,
        image,
        intrinsic_matrix,
        distortion_coeffs,
        skeleton=skeleton,
    )
    poses3d, poses2d = _extract_metrabs_predictions(pred)

    output_json = image_path.with_name(f"{image_path.stem}_metrabs.json")
    output_image = image_path.with_name(f"{image_path.stem}_metrabs.jpg")

    with output_json.open("w", encoding="utf-8") as file:
        json.dump({"image_file": image_path.name, "poses3d": _json_safe(poses3d)}, file, indent=4)

    overlay = _draw_metrabs_overlay(image, poses2d, joint_edges, suppress_joint_indices)
    cv2.imwrite(str(output_image), overlay)

    print(f"[done] Wrote {output_json}")
    print(f"[done] Wrote {output_image}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run Metrabs on recorded dataset frames and save keypoints/visualizations."
    )
    parser.add_argument(
        "input",
        type=Path,
        help="A video_XXX folder, a dataset/category folder, or a single image.",
    )
    parser.add_argument(
        "--skeleton",
        default=settings.SKELETON,
        help=f"Metrabs skeleton name. Default: {settings.SKELETON}",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing shoulder/elbow/wrist values with Metrabs values.",
    )
    parser.add_argument(
        "--reset-from-template",
        action="store_true",
        help="Ignore annotations_filled.json and rebuild it from annotations.json.",
    )
    parser.add_argument(
        "--torso-joint",
        default=None,
        help="Override torso source joint by name or index, e.g. bell_smpl, spin_smpl, thor_smpl, or 3.",
    )
    parser.add_argument(
        "--no-visualizations",
        action="store_true",
        help="Only write keypoints into annotations_filled.json.",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Process only the first N frames, useful for testing.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    input_path = args.input

    if not input_path.exists():
        raise FileNotFoundError(input_path)

    torch.backends.cudnn.benchmark = True

    intrinsic_matrix, distortion_coeffs, model, joint_names, joint_edges = pose_map._load_metrabs_model(
        skeleton=args.skeleton
    )
    suppress_joint_indices, _ = pose_map._setup_metrabs_joints(joint_names)

    if input_path.is_file() and input_path.suffix.lower() in IMAGE_EXTENSIONS:
        process_single_image(
            input_path,
            model,
            joint_names,
            joint_edges,
            suppress_joint_indices,
            intrinsic_matrix,
            distortion_coeffs,
            args.skeleton,
        )
        return

    video_folders = _find_video_folders(input_path)
    if not video_folders:
        raise RuntimeError(
            f"No dataset video folders found. Expected a folder containing frames/: {input_path}"
        )

    for item in video_folders:
        video_folder = item.parent if item.name == "frames" else item
        process_video_folder(
            video_folder,
            model,
            joint_names,
            joint_edges,
            suppress_joint_indices,
            intrinsic_matrix,
            distortion_coeffs,
            args.skeleton,
            save_visualizations=not args.no_visualizations,
            overwrite=args.overwrite,
            reset_from_template=args.reset_from_template,
            torso_joint=args.torso_joint,
            max_frames=args.max_frames,
        )


if __name__ == "__main__":
    main()
