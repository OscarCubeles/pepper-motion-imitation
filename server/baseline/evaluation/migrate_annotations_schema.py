import argparse
import json
from pathlib import Path


JOINT_ANGLE_KEYS = ("shoulder_pitch", "shoulder_roll", "elbow_yaw", "elbow_roll")
POINT_KEYS = ("shoulder", "elbow", "wrist")


def empty_arm_data():
    return {
        "shoulder": None,
        "elbow": None,
        "wrist": None,
        "joint_angles": {key: None for key in JOINT_ANGLE_KEYS},
        "wrist_orientation": None,
    }


def merge_missing(target, source):
    for key, value in source.items():
        if target.get(key) is None and value is not None:
            target[key] = value


def migrate_frame(frame):
    migrated = {
        "frame_id": frame.get("frame_id"),
        "timestamp": frame.get("timestamp"),
        "image_file": frame.get("image_file"),
        "torso": frame.get("torso"),
        "left_arm": empty_arm_data(),
        "right_arm": empty_arm_data(),
    }

    for optional_key in ("metrabs_visualization", "metrabs_error"):
        if optional_key in frame:
            migrated[optional_key] = frame[optional_key]

    for side in ("left", "right"):
        arm_key = f"{side}_arm"
        existing_arm = frame.get(arm_key)
        if isinstance(existing_arm, dict):
            for point_key in POINT_KEYS:
                migrated[arm_key][point_key] = existing_arm.get(point_key)

            existing_angles = existing_arm.get("joint_angles", {})
            if isinstance(existing_angles, dict):
                merge_missing(migrated[arm_key]["joint_angles"], existing_angles)

            migrated[arm_key]["wrist_orientation"] = existing_arm.get("wrist_orientation")

    legacy_angles = frame.get("joint_angles", {})
    if not isinstance(legacy_angles, dict):
        legacy_angles = {}

    for side in ("left", "right"):
        arm_key = f"{side}_arm"

        legacy_orientation = frame.get(f"wrist_orientation_{side}")
        if migrated[arm_key]["wrist_orientation"] is None:
            migrated[arm_key]["wrist_orientation"] = legacy_orientation

        for point_key in POINT_KEYS:
            if migrated[arm_key][point_key] is None:
                migrated[arm_key][point_key] = frame.get(point_key)

        merge_missing(migrated[arm_key]["joint_angles"], legacy_angles)

    return migrated


def migrate_file(path, backup=True):
    with path.open("r", encoding="utf-8") as file:
        annotations = json.load(file)

    frames = annotations.get("frames", [])
    annotations["frames"] = [migrate_frame(frame) for frame in frames]

    if backup:
        backup_path = path.with_suffix(".legacy.json")
        if not backup_path.exists():
            backup_path.write_text(path.read_text(encoding="utf-8"), encoding="utf-8")

    with path.open("w", encoding="utf-8") as file:
        json.dump(annotations, file, indent=4)

    return len(frames)


def annotation_files(root):
    if root.is_file():
        return [root]

    return sorted(
        path for path in root.rglob("annotations*.json")
        if ".legacy" not in path.name
    )


def main():
    parser = argparse.ArgumentParser(description="Migrate dataset annotations to left_arm/right_arm schema.")
    parser.add_argument(
        "path",
        type=Path,
        help="annotations_filled.json, a video folder, or a dataset/category folder.",
    )
    parser.add_argument(
        "--no-backup",
        action="store_true",
        help="Do not create a .legacy.json backup before writing.",
    )
    args = parser.parse_args()

    files = annotation_files(args.path)
    if not files:
        raise RuntimeError(f"No annotation JSON files found under {args.path}")

    for path in files:
        count = migrate_file(path, backup=not args.no_backup)
        print(f"Migrated {path} ({count} frames)")


if __name__ == "__main__":
    main()
