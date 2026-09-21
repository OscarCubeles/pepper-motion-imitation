"""Delete copied MP4 videos from the processed dataset."""

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
PROCESSED_DATA = REPO_ROOT / "processed_data"


def main():
    if not PROCESSED_DATA.is_dir():
        raise FileNotFoundError(
            "Processed dataset folder not found: {}".format(PROCESSED_DATA)
        )

    video_files = list(PROCESSED_DATA.rglob("*.mp4"))
    for video_file in video_files:
        video_file.unlink()

    print("Deleted {} MP4 files from {}".format(len(video_files), PROCESSED_DATA))


if __name__ == "__main__":
    main()
