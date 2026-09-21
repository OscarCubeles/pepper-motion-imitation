"""Copy a dataset and blur detected faces in every saved frame."""

import argparse
import shutil
from pathlib import Path

import cv2


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_SOURCE = REPO_ROOT / "dataset"
DEFAULT_OUTPUT = REPO_ROOT / "processed_data"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}


def blur_faces(image, face_detector):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_detector.detectMultiScale(gray_image, scaleFactor=1.3, minNeighbors=5)

    for x, y, width, height in faces:
        face = image[y : y + height, x : x + width]
        image[y : y + height, x : x + width] = cv2.GaussianBlur(
            face,
            (23, 23),
            30,
        )

    return image, len(faces)


def process_dataset(source_root, output_root):
    source_root = Path(source_root).resolve()
    output_root = Path(output_root).resolve()

    if not source_root.is_dir():
        raise FileNotFoundError("Dataset folder not found: {}".format(source_root))
    if output_root == source_root or source_root in output_root.parents:
        raise ValueError("The output folder must be outside the source dataset folder.")

    cascade_path = Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml"
    face_detector = cv2.CascadeClassifier(str(cascade_path))
    if face_detector.empty():
        raise RuntimeError("Could not load face detector: {}".format(cascade_path))

    copied_files = 0
    processed_frames = 0
    blurred_faces = 0
    skipped_videos = 0

    for source_path in source_root.rglob("*"):
        relative_path = source_path.relative_to(source_root)
        output_path = output_root / relative_path

        if source_path.is_dir():
            output_path.mkdir(parents=True, exist_ok=True)
            continue

        if source_path.suffix.lower() == ".mp4":
            skipped_videos += 1
            continue

        output_path.parent.mkdir(parents=True, exist_ok=True)
        is_frame = (
            source_path.parent.name == "frames"
            and source_path.suffix.lower() in IMAGE_EXTENSIONS
        )

        if not is_frame:
            shutil.copy2(str(source_path), str(output_path))
            copied_files += 1
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

        if processed_frames % 100 == 0:
            print("Processed {} frames...".format(processed_frames))

    print("Finished.")
    print("Source: {}".format(source_root))
    print("Output: {}".format(output_root))
    print("Frames processed: {}".format(processed_frames))
    print("Faces blurred: {}".format(blurred_faces))
    print("Other files copied: {}".format(copied_files))
    print("MP4 videos skipped: {}".format(skipped_videos))


def main():
    parser = argparse.ArgumentParser(
        description="Copy the dataset and blur faces in every frames folder."
    )
    parser.add_argument(
        "--source",
        type=Path,
        default=DEFAULT_SOURCE,
        help="Source dataset folder (default: dataset).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Output folder (default: processed_data).",
    )
    args = parser.parse_args()
    process_dataset(args.source, args.output)


if __name__ == "__main__":
    main()
