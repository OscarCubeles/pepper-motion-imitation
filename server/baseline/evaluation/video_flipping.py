"""Create a horizontally mirrored dataset video and matching annotations."""

import argparse
import copy
import json
import math
import re
import shutil
import tempfile
from pathlib import Path

import cv2


VIDEO_FOLDER_PATTERN = re.compile(r"^video_(\d+)$")
REQUIRED_ANNOTATIONS = ("annotations.json", "annotations_filled.json")
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}
POINT_KEYS = ("shoulder", "elbow", "wrist")
SIGNED_ANGLE_KEYS = ("shoulder_roll", "elbow_yaw", "elbow_roll")
ORIENTATION_MIRROR = {
    "LEFT": "RIGHT",
    "RIGHT": "LEFT",
    "FRONT": "FRONT",
    "BACK": "BACK",
    "UP": "UP",
    "DOWN": "DOWN",
}


def _load_json(path):
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def _save_json(path, data):
    with path.open("w", encoding="utf-8") as file:
        json.dump(data, file, indent=4)


def _negate_number(value, field_name):
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field_name} must be numeric or null, got {value!r}")
    if not math.isfinite(float(value)):
        raise ValueError(f"{field_name} must be finite, got {value!r}")
    return -value


def mirror_point(point, field_name="point", x_offset_mm=0.0):
    """Reflect a MetrAbs camera-space point across the optical vertical plane."""
    if point is None:
        return None

    if isinstance(x_offset_mm, bool) or not isinstance(x_offset_mm, (int, float)):
        raise ValueError(f"x_offset_mm must be numeric, got {x_offset_mm!r}")
    if not math.isfinite(float(x_offset_mm)):
        raise ValueError(f"x_offset_mm must be finite, got {x_offset_mm!r}")

    mirrored = copy.deepcopy(point)
    if isinstance(mirrored, list) and len(mirrored) >= 3:
        mirrored[0] = _negate_number(mirrored[0], f"{field_name}.x") + x_offset_mm
        return mirrored
    if isinstance(mirrored, dict) and all(axis in mirrored for axis in ("x", "y", "z")):
        mirrored["x"] = _negate_number(mirrored["x"], f"{field_name}.x") + x_offset_mm
        return mirrored
    raise ValueError(f"{field_name} must be null, [x, y, z], or an x/y/z object")


def _mirror_orientation(value):
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError(f"wrist_orientation must be a string or null, got {value!r}")
    normalized = value.strip().upper()
    if normalized not in ORIENTATION_MIRROR:
        raise ValueError(f"Unsupported wrist_orientation: {value!r}")
    return ORIENTATION_MIRROR[normalized]


def mirror_arm(arm, field_name="arm", x_offset_mm=0.0):
    if arm is None:
        return None
    if not isinstance(arm, dict):
        raise ValueError(f"{field_name} must be an object")

    mirrored = copy.deepcopy(arm)
    for point_key in POINT_KEYS:
        if point_key in mirrored:
            mirrored[point_key] = mirror_point(
                mirrored[point_key],
                f"{field_name}.{point_key}",
                x_offset_mm=x_offset_mm,
            )

    angles = mirrored.get("joint_angles")
    if angles is not None:
        if not isinstance(angles, dict):
            raise ValueError(f"{field_name}.joint_angles must be an object")
        for angle_key in SIGNED_ANGLE_KEYS:
            if angle_key in angles:
                angles[angle_key] = _negate_number(
                    angles[angle_key],
                    f"{field_name}.joint_angles.{angle_key}",
                )

    if "wrist_orientation" in mirrored:
        mirrored["wrist_orientation"] = _mirror_orientation(mirrored["wrist_orientation"])
    return mirrored


def mirror_frame_annotation(frame, x_offset_mm=0.0):
    if not isinstance(frame, dict):
        raise ValueError("Every annotation frame must be an object")

    mirrored = copy.deepcopy(frame)
    if "torso" in mirrored:
        mirrored["torso"] = mirror_point(
            mirrored["torso"],
            "torso",
            x_offset_mm=x_offset_mm,
        )

    old_left = frame.get("left_arm")
    old_right = frame.get("right_arm")
    if "left_arm" in frame or "right_arm" in frame:
        mirrored["left_arm"] = mirror_arm(
            old_right,
            "right_arm",
            x_offset_mm=x_offset_mm,
        )
        mirrored["right_arm"] = mirror_arm(
            old_left,
            "left_arm",
            x_offset_mm=x_offset_mm,
        )

    # Existing visualizations contain unmirrored labels and text.
    mirrored.pop("metrabs_visualization", None)
    return mirrored


def mirror_annotations(annotations, output_video_id, x_offset_mm=0.0):
    if not isinstance(annotations, dict):
        raise ValueError("Annotation root must be an object")
    frames = annotations.get("frames")
    if not isinstance(frames, list):
        raise ValueError("Annotation root must contain a frames list")

    mirrored = copy.deepcopy(annotations)
    mirrored["video_id"] = output_video_id
    mirrored["frames"] = [
        mirror_frame_annotation(frame, x_offset_mm=x_offset_mm)
        for frame in frames
    ]
    return mirrored


def _frame_names(annotations, annotation_name):
    names = []
    for index, frame in enumerate(annotations.get("frames", [])):
        if not isinstance(frame, dict) or not frame.get("image_file"):
            raise ValueError(f"{annotation_name} frame {index} has no image_file")
        names.append(frame["image_file"])
    if len(names) != len(set(names)):
        raise ValueError(f"{annotation_name} contains duplicate image_file values")
    return names


def _validate_source(source):
    source = source.resolve()
    match = VIDEO_FOLDER_PATTERN.match(source.name)
    if not source.is_dir() or match is None:
        raise ValueError(f"Input must be a video_XXX directory: {source}")

    video_path = source / "video.mp4"
    frames_path = source / "frames"
    if not video_path.is_file():
        raise FileNotFoundError(video_path)
    if not frames_path.is_dir():
        raise FileNotFoundError(frames_path)

    annotation_data = {}
    expected_names = None
    for annotation_name in REQUIRED_ANNOTATIONS:
        annotation_path = source / annotation_name
        if not annotation_path.is_file():
            raise FileNotFoundError(annotation_path)
        data = _load_json(annotation_path)
        if data.get("video_id") != source.name:
            raise ValueError(
                f"{annotation_path} video_id is {data.get('video_id')!r}, expected {source.name!r}"
            )
        if data.get("category") != source.parent.name:
            raise ValueError(
                f"{annotation_path} category is {data.get('category')!r}, "
                f"expected {source.parent.name!r}"
            )
        names = _frame_names(data, annotation_name)
        if expected_names is None:
            expected_names = names
        elif names != expected_names:
            raise ValueError("Canonical annotation files do not contain the same ordered frames")
        annotation_data[annotation_name] = data

    if not expected_names:
        raise ValueError(f"No annotated frames found in {source}")
    for frame_name in expected_names:
        frame_path = frames_path / frame_name
        if frame_path.suffix.lower() not in IMAGE_EXTENSIONS or not frame_path.is_file():
            raise FileNotFoundError(frame_path)

    stored_images = sorted(
        path.name for path in frames_path.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )
    if sorted(expected_names) != stored_images:
        raise ValueError("frames/ contents do not exactly match the canonical annotations")

    return source, annotation_data, expected_names


def _next_video_id(category_path):
    indices = []
    for path in category_path.iterdir():
        if not path.is_dir():
            continue
        match = VIDEO_FOLDER_PATTERN.match(path.name)
        if match:
            indices.append(int(match.group(1)))
    return f"video_{(max(indices) + 1 if indices else 1):03d}"


def _mirror_stored_frames(source_frames, destination_frames, frame_names):
    destination_frames.mkdir(parents=True, exist_ok=False)
    dimensions = None
    for frame_name in frame_names:
        source_path = source_frames / frame_name
        image = cv2.imread(str(source_path), cv2.IMREAD_UNCHANGED)
        if image is None:
            raise RuntimeError(f"Could not read frame: {source_path}")
        current_dimensions = (image.shape[1], image.shape[0])
        if dimensions is None:
            dimensions = current_dimensions
        elif current_dimensions != dimensions:
            raise ValueError(f"Frame dimensions changed at {source_path}")
        if not cv2.imwrite(str(destination_frames / frame_name), cv2.flip(image, 1)):
            raise RuntimeError(f"Could not write mirrored frame: {frame_name}")
    return dimensions


def _mirror_video(source_path, destination_path, expected_count, expected_dimensions):
    capture = cv2.VideoCapture(str(source_path))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open video: {source_path}")

    fps = float(capture.get(cv2.CAP_PROP_FPS))
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if fps <= 0 or (width, height) != expected_dimensions:
        capture.release()
        raise ValueError(
            f"Invalid video properties: fps={fps}, dimensions={width}x{height}, "
            f"expected dimensions={expected_dimensions[0]}x{expected_dimensions[1]}"
        )

    writer = cv2.VideoWriter(
        str(destination_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )
    if not writer.isOpened():
        capture.release()
        raise RuntimeError(f"Could not create video: {destination_path}")

    frame_count = 0
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            writer.write(cv2.flip(frame, 1))
            frame_count += 1
    finally:
        capture.release()
        writer.release()

    if frame_count != expected_count:
        raise ValueError(
            f"video.mp4 contains {frame_count} frames but annotations contain {expected_count}"
        )
    return fps


def augment_video(source, x_offset_mm=0.0):
    source, annotation_data, frame_names = _validate_source(Path(source))
    category_path = source.parent
    output_video_id = _next_video_id(category_path)
    destination = category_path / output_video_id
    if destination.exists():
        raise FileExistsError(destination)

    temporary = Path(tempfile.mkdtemp(prefix=f".{output_video_id}_flipping_", dir=str(category_path)))
    try:
        dimensions = _mirror_stored_frames(
            source / "frames",
            temporary / "frames",
            frame_names,
        )
        fps = _mirror_video(
            source / "video.mp4",
            temporary / "video.mp4",
            len(frame_names),
            dimensions,
        )

        for annotation_name, data in annotation_data.items():
            annotated_fps = data.get("fps")
            if annotated_fps is not None and not math.isclose(
                float(annotated_fps), fps, rel_tol=0.0, abs_tol=0.01
            ):
                raise ValueError(
                    f"{annotation_name} FPS {annotated_fps} does not match video FPS {fps}"
                )
            _save_json(
                temporary / annotation_name,
                mirror_annotations(
                    data,
                    output_video_id,
                    x_offset_mm=x_offset_mm,
                ),
            )

        # Rename only after every output has been validated and written.
        temporary.replace(destination)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise

    return destination


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Horizontally mirror a dataset video, swap its left/right annotations, "
            "and create the next video_XXX folder in the same category."
        )
    )
    parser.add_argument("input", type=Path, help="Source dataset video_XXX folder.")
    parser.add_argument(
        "--x-offset-mm",
        "--x-offset",
        dest="x_offset_mm",
        type=float,
        default=0.0,
        help=(
            "Camera-space X translation in millimetres after mirroring. "
            "Positive values move projected annotations right; default: 0."
        ),
    )
    return parser.parse_args()


def main():
    args = parse_args()
    destination = augment_video(args.input, x_offset_mm=args.x_offset_mm)
    print(f"[done] Created mirrored dataset video: {destination}")
    print(f"[info] Applied annotation X offset: {args.x_offset_mm:g} mm")
    print("[note] Source files were not modified.")
    print(
        "[note] Coordinates use camera-space x reflection. Projection with the original "
        "unmirrored camera calibration may show a small horizontal offset."
    )


if __name__ == "__main__":
    main()
