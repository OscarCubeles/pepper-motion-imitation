"""Blur faces in every Metrabs visualization folder in a dataset."""

import argparse
import shutil
from pathlib import Path

import cv2

from .blur_dataset_faces import IMAGE_EXTENSIONS, blur_faces


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_SOURCE = REPO_ROOT / "dataset"
SOURCE_DIRECTORY_NAME = "metrabs_visualizations"
OUTPUT_DIRECTORY_NAME = "metrabs_visualizations_blurred"


def blur_metrabs_visualizations(source_root):
    source_root = Path(source_root).resolve()

    if not source_root.is_dir():
        raise FileNotFoundError("Dataset folder not found: {}".format(source_root))

    cascade_path = Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml"
    face_detector = cv2.CascadeClassifier(str(cascade_path))
    if face_detector.empty():
        raise RuntimeError("Could not load face detector: {}".format(cascade_path))

    visualization_directories = sorted(
        path
        for path in source_root.rglob(SOURCE_DIRECTORY_NAME)
        if path.is_dir()
    )
    processed_images = 0
    blurred_faces = 0
    copied_files = 0

    for source_directory in visualization_directories:
        output_directory = source_directory.with_name(OUTPUT_DIRECTORY_NAME)
        output_directory.mkdir(parents=True, exist_ok=True)

        for source_path in source_directory.rglob("*"):
            relative_path = source_path.relative_to(source_directory)
            output_path = output_directory / relative_path

            if source_path.is_dir():
                output_path.mkdir(parents=True, exist_ok=True)
                continue

            output_path.parent.mkdir(parents=True, exist_ok=True)
            if source_path.suffix.lower() not in IMAGE_EXTENSIONS:
                shutil.copy2(str(source_path), str(output_path))
                copied_files += 1
                continue

            image = cv2.imread(str(source_path))
            if image is None:
                raise RuntimeError("Could not read visualization: {}".format(source_path))

            image, face_count = blur_faces(image, face_detector)
            if face_count:
                if not cv2.imwrite(str(output_path), image):
                    raise RuntimeError(
                        "Could not write visualization: {}".format(output_path)
                    )
            else:
                shutil.copy2(str(source_path), str(output_path))

            processed_images += 1
            blurred_faces += face_count

            if processed_images % 100 == 0:
                print("Processed {} visualizations...".format(processed_images))

    print("Finished.")
    print("Source: {}".format(source_root))
    print("Visualization folders processed: {}".format(len(visualization_directories)))
    print("Images processed: {}".format(processed_images))
    print("Faces blurred: {}".format(blurred_faces))
    print("Other files copied: {}".format(copied_files))


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Blur faces in every metrabs_visualizations folder and write the "
            "results to sibling metrabs_visualizations_blurred folders."
        )
    )
    parser.add_argument(
        "source",
        nargs="?",
        type=Path,
        default=DEFAULT_SOURCE,
        help="Dataset or category folder (default: dataset).",
    )
    args = parser.parse_args()
    blur_metrabs_visualizations(args.source)


if __name__ == "__main__":
    main()
