import argparse
import json
import math
import shutil
import sys
from pathlib import Path

import numpy as np


CURRENT_FILE = Path(__file__).resolve()
DL_POSE_DIR = CURRENT_FILE.parents[1]
SERVER_DIR = CURRENT_FILE.parents[2]
REPO_ROOT = CURRENT_FILE.parents[3]

for path in (DL_POSE_DIR, SERVER_DIR, REPO_ROOT):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

import settings  # noqa: E402
settings.add_server_dir_to_path()

import kinematics.scaling_spherical as scaling  # noqa: E402


ARM_SIDES = ("left", "right")
POINT_KEYS = ("shoulder", "elbow", "wrist")
JOINT_ANGLE_KEYS = ("shoulder_pitch", "shoulder_roll", "elbow_yaw", "elbow_roll")
ANGLE_KEY_MAP = dict(zip(JOINT_ANGLE_KEYS, range(4)))
# WristYaw is intentionally not computed for these dataset annotations.
# Treat it as 0.0 when consuming these angle labels unless a later script adds it.


def _empty_joint_angles():
    return {key: None for key in JOINT_ANGLE_KEYS}


def _load_json(path):
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def _save_json(path, data):
    with path.open("w", encoding="utf-8") as file:
        json.dump(data, file, indent=4)


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


def _safe_angle(value, degrees=False):
    if value is None:
        return None
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(numeric) or math.isinf(numeric):
        return None
    if degrees:
        numeric = math.degrees(numeric)
    return numeric


def _ensure_arm_schema(frame_data, side):
    arm_key = f"{side}_arm"
    arm_data = frame_data.get(arm_key)
    if not isinstance(arm_data, dict):
        arm_data = {}
        frame_data[arm_key] = arm_data

    for point_key in POINT_KEYS:
        arm_data.setdefault(point_key, None)

    joint_angles = arm_data.get("joint_angles")
    if not isinstance(joint_angles, dict):
        joint_angles = {}
        arm_data["joint_angles"] = joint_angles
    for angle_key in JOINT_ANGLE_KEYS:
        joint_angles.setdefault(angle_key, None)

    arm_data.setdefault("wrist_orientation", None)
    return arm_data


def _compute_side_angles(frame_data, side, degrees=False):
    other_side = "left" if side == "right" else "right"
    arm_data = _ensure_arm_schema(frame_data, side)
    other_arm_data = _ensure_arm_schema(frame_data, other_side)

    torso = _as_vector(frame_data.get("torso"))
    shoulder = _as_vector(arm_data.get("shoulder"))
    elbow = _as_vector(arm_data.get("elbow"))
    wrist = _as_vector(arm_data.get("wrist"))
    other_shoulder = _as_vector(other_arm_data.get("shoulder"))

    if any(value is None for value in (torso, shoulder, elbow, wrist, other_shoulder)):
        return None

    try:
        raw_angles = scaling.compute_arm_targets(
            torso,
            shoulder,
            elbow,
            wrist,
            other_shoulder,
            side,
            None,
            use_human_mode=False,
        )[:4]
    except Exception as exc:
        return {"error": str(exc)}

    return {
        angle_key: _safe_angle(raw_angles[index], degrees=degrees)
        for angle_key, index in ANGLE_KEY_MAP.items()
    }


def _fill_frame_angles(frame_data, overwrite=False, degrees=False):
    changed = 0
    errors = []

    for side in ARM_SIDES:
        arm_data = _ensure_arm_schema(frame_data, side)
        computed = _compute_side_angles(frame_data, side, degrees=degrees)

        if computed is None:
            errors.append(f"{side}: missing coordinates")
            continue
        if "error" in computed:
            errors.append(f"{side}: {computed['error']}")
            continue

        joint_angles = arm_data["joint_angles"]
        for angle_key, angle_value in computed.items():
            if angle_value is None:
                errors.append(f"{side}.{angle_key}: invalid result")
                continue
            if overwrite or joint_angles.get(angle_key) is None:
                if joint_angles.get(angle_key) != angle_value:
                    joint_angles[angle_key] = angle_value
                    changed += 1

    return changed, errors


def _annotation_files(input_path):
    input_path = input_path.resolve()

    if input_path.is_file():
        return [input_path]

    if (input_path / "annotations_filled.json").exists():
        return [input_path / "annotations_filled.json"]

    return sorted(input_path.rglob("annotations_filled.json"))


def _backup_file(path):
    backup_path = path.with_name(f"{path.stem}.before_angles{path.suffix}")
    if not backup_path.exists():
        shutil.copy2(path, backup_path)
    return backup_path


def process_file(path, overwrite=False, degrees=False, dry_run=False, backup=True):
    annotations = _load_json(path)
    frames = annotations.get("frames", [])
    changed = 0
    frame_errors = {}

    for frame_index, frame_data in enumerate(frames):
        if not isinstance(frame_data, dict):
            frame_errors[frame_index] = ["frame is not an object"]
            continue

        frame_changed, errors = _fill_frame_angles(
            frame_data,
            overwrite=overwrite,
            degrees=degrees,
        )
        changed += frame_changed
        if errors:
            frame_errors[frame_data.get("frame_id", frame_index)] = errors

    backup_path = None
    if changed and not dry_run:
        if backup:
            backup_path = _backup_file(path)
        _save_json(path, annotations)

    return {
        "path": path,
        "frames": len(frames),
        "changed": changed,
        "errors": frame_errors,
        "backup": backup_path,
    }


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compute Pepper arm joint angles from coordinates in annotations_filled.json."
    )
    parser.add_argument(
        "input",
        type=Path,
        help="annotations_filled.json, a video folder, a category folder, or the dataset folder.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing joint angle values. By default only missing values are filled.",
    )
    parser.add_argument(
        "--degrees",
        action="store_true",
        help="Store angles in degrees instead of radians.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Compute what would change without writing files.",
    )
    parser.add_argument(
        "--no-backup",
        action="store_true",
        help="Do not create annotations_filled.before_angles.json before writing.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if not args.input.exists():
        raise FileNotFoundError(args.input)

    files = _annotation_files(args.input)
    if not files:
        raise RuntimeError(f"No annotations_filled.json files found under {args.input}")

    total_changed = 0
    for path in files:
        result = process_file(
            path,
            overwrite=args.overwrite,
            degrees=args.degrees,
            dry_run=args.dry_run,
            backup=not args.no_backup,
        )
        total_changed += result["changed"]
        mode = "dry-run" if args.dry_run else "written"
        unit = "degrees" if args.degrees else "radians"
        print(
            f"[{mode}] {result['path']} "
            f"frames={result['frames']} changed_values={result['changed']} unit={unit}"
        )
        if result["backup"] is not None:
            print(f"  backup={result['backup']}")
        if result["errors"]:
            print(f"  frames_with_errors={len(result['errors'])}")

    print(f"[done] total_changed_values={total_changed}")


if __name__ == "__main__":
    main()
