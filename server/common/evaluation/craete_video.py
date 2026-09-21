import cv2
import os
import json
import shutil
from pathlib import Path

# ==========================================================
# CONFIGURATION
# ==========================================================

FPS = 8
DURATION_SECONDS = 3
FRAMES_PER_VIDEO = FPS * DURATION_SECONDS

DATASET_ROOT = "dataset"
CATEGORY = "basketball_movements" #"pepper_singularity_motions" # "police_traffic", "upper_body_jogging", "celebration_movements", "basketball_movements"

#


# ==========================================================
# HELPERS
# ==========================================================

def get_next_video_index(category_path):
    existing = [
        d for d in os.listdir(category_path)
        if d.startswith("video_")
    ]

    if not existing:
        return 1

    indices = []
    for folder in existing:
        try:
            indices.append(int(folder.split("_")[1]))
        except:
            pass

    return max(indices) + 1


def get_existing_video_indices(category_path):
    if not os.path.isdir(category_path):
        return []

    indices = []
    for folder in os.listdir(category_path):
        if not folder.startswith("video_"):
            continue
        try:
            indices.append(int(folder.split("_")[1]))
        except:
            pass

    return sorted(indices)


def delete_last_video(category_path):
    indices = get_existing_video_indices(category_path)
    if not indices:
        print("No previous video to redo.")
        return None

    last_idx = indices[-1]
    last_video_folder = os.path.join(category_path, f"video_{last_idx:03d}")
    shutil.rmtree(last_video_folder)
    print(f"Deleted previous recording:\n{last_video_folder}")
    return last_idx


def create_json(video_id, category):
    return {
        "video_id": video_id,
        "category": category,
        "fps": FPS,
        "duration_seconds": DURATION_SECONDS,
        "frames": []
    }


def create_arm_data():
    return {
        "shoulder": None,
        "elbow": None,
        "wrist": None,
        "joint_angles": {
            "shoulder_pitch": None,
            "shoulder_roll": None,
            "elbow_yaw": None,
            "elbow_roll": None
        },
        "wrist_orientation": None
    }


# ==========================================================
# MAIN RECORD FUNCTION
# ==========================================================

def record_video(cap, video_idx=None):

    category_path = os.path.join(DATASET_ROOT, CATEGORY)
    Path(category_path).mkdir(parents=True, exist_ok=True)

    if video_idx is None:
        video_idx = get_next_video_index(category_path)

    video_id = f"video_{video_idx:03d}"

    video_folder = os.path.join(category_path, video_id)
    frames_folder = os.path.join(video_folder, "frames")

    Path(frames_folder).mkdir(parents=True, exist_ok=True)

    video_path = os.path.join(video_folder, "video.mp4")
    json_path = os.path.join(video_folder, "annotations.json")

    # -----------------------------
    # VIDEO WRITER
    # -----------------------------
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    writer = cv2.VideoWriter(
        video_path,
        cv2.VideoWriter_fourcc(*"mp4v"),
        FPS,
        (width, height)
    )

    # -----------------------------
    # CREATE JSON HERE (IMPORTANT)
    # -----------------------------
    annotations = create_json(video_id, CATEGORY)

    print("\nStarting recording in 3 seconds...")
    cv2.waitKey(1000)
    print("\nStarting recording in 2 seconds...")
    cv2.waitKey(1000)
    print("\nStarting recording in 1 second...")
    cv2.waitKey(1000)

    print(f"Recording {video_id}...")

    # ==========================================================
    # RECORD LOOP
    # ==========================================================
    for frame_idx in range(FRAMES_PER_VIDEO):

        ret, frame = cap.read()
        if not ret:
            break

        # save video
        writer.write(frame)

        # save frame image
        frame_name = f"frame_{frame_idx:03d}.jpg"
        frame_path = os.path.join(frames_folder, frame_name)
        cv2.imwrite(frame_path, frame)

        # -----------------------------
        # ADD FRAME TO JSON (HERE)
        # -----------------------------
        frame_data = {
            "frame_id": frame_idx,
            "timestamp": round(frame_idx / FPS, 3),
            "image_file": frame_name,
            "torso": None,
            "left_arm": create_arm_data(),
            "right_arm": create_arm_data()
        }

        annotations["frames"].append(frame_data)

        cv2.imshow("Recording", frame)

        if cv2.waitKey(int(1000 / FPS)) & 0xFF == 27:
            break

    # -----------------------------
    # SAVE EVERYTHING AT END
    # -----------------------------
    writer.release()

    with open(json_path, "w") as f:
        json.dump(annotations, f, indent=4)

    print(f"Saved:\n{video_folder}")


# ==========================================================
# MAIN
# ==========================================================

def main():

    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        raise RuntimeError("Cannot open webcam")

    print("Press ENTER to record next video, type r + ENTER to redo last video, CTRL+C to exit")

    while True:
        command = input("\nReady? Press ENTER, or type r to redo last video: ").strip().lower()

        if command in ("r", "redo"):
            category_path = os.path.join(DATASET_ROOT, CATEGORY)
            video_idx = delete_last_video(category_path)
            if video_idx is None:
                continue
            record_video(cap, video_idx=video_idx)
        else:
            record_video(cap)


if __name__ == "__main__":
    main()
