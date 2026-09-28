"""Create face-blurred copies of the synergy reference motion frames."""

import argparse
import shutil
from pathlib import Path

import cv2

from .blur_dataset_faces import IMAGE_EXTENSIONS, blur_faces


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_SOURCE = REPO_ROOT / "dataset" / "synergy_reference_motions"
SOURCE_DIRECTORY_NAME = "frames"
OUTPUT_DIRECTORY_NAME = "frames_blurred"


def blur_synergy_reference_frames(source_root=DEFAULT_SOURCE):
    """Blur faces in each video folder without modifying its original frames."""
    source_root = Path(source_root).resolve()
    if not source_root.is_dir():
        raise FileNotFoundError(
            "Synergy reference motions folder not found: {}".format(source_root)
        )

    cascade_path = Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml"
    face_detector = cv2.CascadeClassifier(str(cascade_path))
    if face_detector.empty():
        raise RuntimeError("Could not load face detector: {}".format(cascade_path))

    if source_root.name.startswith("video_"):
        video_directories = [source_root]
        output_root = source_root.parent
    else:
        video_directories = sorted(
            path
            for path in source_root.iterdir()
            if path.is_dir() and path.name.startswith("video_")
        )
        output_root = source_root

    processed_videos = 0
    processed_frames = 0
    blurred_faces = 0

    for video_directory in video_directories:
        frames_directory = video_directory / SOURCE_DIRECTORY_NAME
        if not frames_directory.is_dir():
            loose_images = sorted(
                path
                for path in video_directory.iterdir()
                if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
            )
            if not loose_images:
                print("Skipping {}: no frames found.".format(video_directory.name))
                continue

            frames_directory.mkdir(parents=True)
            for source_path in loose_images:
                shutil.copy2(str(source_path), str(frames_directory / source_path.name))
            print(
                "Copied {} loose images into {}.".format(
                    len(loose_images),
                    frames_directory.relative_to(output_root),
                )
            )

        output_directory = video_directory / OUTPUT_DIRECTORY_NAME
        output_directory.mkdir(parents=True, exist_ok=True)

        for source_path in sorted(frames_directory.iterdir()):
            if not source_path.is_file():
                continue

            output_path = output_directory / source_path.name
            if source_path.suffix.lower() not in IMAGE_EXTENSIONS:
                shutil.copy2(str(source_path), str(output_path))
                continue

            image = cv2.imread(str(source_path))
            if image is None:
                raise RuntimeError("Could not read frame: {}".format(source_path))

            image, face_count = blur_faces(image, face_detector)
            if face_count:
                if not cv2.imwrite(str(output_path), image):
                    raise RuntimeError("Could not write frame: {}".format(output_path))
            else:
                shutil.copy2(str(source_path), str(output_path))

            processed_frames += 1
            blurred_faces += face_count

        processed_videos += 1
        print(
            "Processed {} -> {}".format(
                video_directory.name,
                output_directory.relative_to(output_root),
            )
        )

    print("Finished.")
    print("Source: {}".format(source_root))
    print("Videos processed: {}".format(processed_videos))
    print("Frames processed: {}".format(processed_frames))
    print("Faces blurred: {}".format(blurred_faces))


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Create a frames_blurred folder beside the frames folder in every "
            "synergy reference video. Original frames are left unchanged."
        )
    )
    parser.add_argument(
        "source",
        nargs="?",
        type=Path,
        default=DEFAULT_SOURCE,
        help=(
            "Synergy reference category or single video_XXX folder "
            "(default: dataset/synergy_reference_motions)."
        ),
    )
    args = parser.parse_args()
    blur_synergy_reference_frames(args.source)


if __name__ == "__main__":
    main()
