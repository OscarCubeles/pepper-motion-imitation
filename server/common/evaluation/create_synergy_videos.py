import argparse
import json
import shutil
import time
from pathlib import Path

import cv2


DEFAULT_CATEGORY = "synergy_reference_motions"


def _next_video_index(category_path):
    indices = _existing_video_indices(category_path)
    return indices[-1] + 1 if indices else 1


def _existing_video_indices(category_path):
    if not category_path.exists():
        return []

    indices = []
    for path in category_path.iterdir():
        if not path.is_dir() or not path.name.startswith("video_"):
            continue
        try:
            indices.append(int(path.name.split("_", 1)[1]))
        except (IndexError, ValueError):
            continue
    return sorted(indices)


def _empty_arm():
    return {
        "shoulder": None,
        "elbow": None,
        "wrist": None,
        "joint_angles": {
            "shoulder_pitch": None,
            "shoulder_roll": None,
            "elbow_yaw": None,
            "elbow_roll": None,
        },
        "wrist_orientation": None,
    }


def _empty_annotations(video_id, category, fps, duration_seconds, purpose):
    return {
        "video_id": video_id,
        "category": category,
        "fps": fps,
        "duration_seconds": duration_seconds,
        "purpose": purpose,
        "frames": [],
    }


def _frame_annotation(frame_idx, fps, frame_name):
    return {
        "frame_id": frame_idx,
        "timestamp": round(frame_idx / fps, 3),
        "image_file": frame_name,
        "torso": None,
        "left_arm": _empty_arm(),
        "right_arm": _empty_arm(),
    }


def _delete_last_video(category_path):
    indices = _existing_video_indices(category_path)
    if not indices:
        print("No previous synergy video to redo.")
        return None

    video_idx = indices[-1]
    video_folder = category_path / f"video_{video_idx:03d}"
    shutil.rmtree(video_folder)
    print(f"Deleted previous synergy recording: {video_folder}")
    return video_idx


def _countdown(seconds):
    for remaining in range(seconds, 0, -1):
        print(f"Starting recording in {remaining}...")
        time.sleep(1)


def record_video(cap, args, video_idx=None):
    category_path = args.dataset_root / args.category
    category_path.mkdir(parents=True, exist_ok=True)

    if video_idx is None:
        video_idx = _next_video_index(category_path)

    video_id = f"video_{video_idx:03d}"
    video_folder = category_path / video_id
    frames_folder = video_folder / "frames"
    frames_folder.mkdir(parents=True, exist_ok=True)

    video_path = video_folder / "video.mp4"
    annotations_path = video_folder / "annotations.json"

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if width <= 0 or height <= 0:
        raise RuntimeError("Could not read webcam frame size.")

    writer = cv2.VideoWriter(
        str(video_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        args.fps,
        (width, height),
    )

    annotations = _empty_annotations(
        video_id,
        args.category,
        args.fps,
        args.duration,
        "synergy_space_reference",
    )

    _countdown(args.countdown)
    print(f"Recording {video_id} for {args.duration}s at {args.fps} FPS...")

    frames_per_video = int(args.fps * args.duration)
    delay_ms = max(1, int(1000 / args.fps))

    for frame_idx in range(frames_per_video):
        ret, frame = cap.read()
        if not ret:
            print("Stopped early: webcam did not return a frame.")
            break

        writer.write(frame)

        frame_name = f"frame_{frame_idx:03d}.jpg"
        cv2.imwrite(str(frames_folder / frame_name), frame)
        annotations["frames"].append(_frame_annotation(frame_idx, args.fps, frame_name))

        if not args.no_preview:
            cv2.imshow("Synergy reference recording", frame)
            if cv2.waitKey(delay_ms) & 0xFF == 27:
                print("Stopped early: ESC pressed.")
                break
        else:
            time.sleep(1 / args.fps)

    writer.release()

    with annotations_path.open("w", encoding="utf-8") as file:
        json.dump(annotations, file, indent=4)

    print(f"Saved synergy video dataset folder: {video_folder}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Record webcam videos into the dataset structure for synergy-space reference motions."
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("dataset"),
        help="Dataset root folder.",
    )
    parser.add_argument(
        "--category",
        default=DEFAULT_CATEGORY,
        help=f"Dataset category to create. Default: {DEFAULT_CATEGORY}",
    )
    parser.add_argument("--camera", type=int, default=0, help="OpenCV camera index.")
    parser.add_argument("--fps", type=int, default=8, help="Recording FPS and extracted frame FPS.")
    parser.add_argument("--duration", type=int, default=15, help="Seconds per recorded video.")
    parser.add_argument("--countdown", type=int, default=3, help="Countdown seconds before each recording.")
    parser.add_argument("--no-preview", action="store_true", help="Do not show the OpenCV preview window.")
    return parser.parse_args()


def main():
    args = parse_args()
    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open webcam {args.camera}")

    category_path = args.dataset_root / args.category
    print("Synergy reference recorder")
    print(f"Output category: {category_path}")
    print("Press ENTER to record next video, type r + ENTER to redo last video, CTRL+C to exit.")

    try:
        while True:
            command = input("\nReady? Press ENTER, or type r to redo last video: ").strip().lower()
            if command in ("r", "redo"):
                video_idx = _delete_last_video(category_path)
                if video_idx is None:
                    continue
                record_video(cap, args, video_idx=video_idx)
            else:
                record_video(cap, args)
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
