import argparse
import time
from pathlib import Path

import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

HAND_CONNECTIONS = [
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 4),
    (0, 5),
    (5, 6),
    (6, 7),
    (7, 8),
    (5, 9),
    (9, 10),
    (10, 11),
    (11, 12),
    (9, 13),
    (13, 14),
    (14, 15),
    (15, 16),
    (13, 17),
    (17, 18),
    (18, 19),
    (19, 20),
    (0, 17),
]


def _normalized_to_pixel(lm, width, height):
    x = min(max(int(lm.x * width), 0), width - 1)
    y = min(max(int(lm.y * height), 0), height - 1)
    return x, y


# Draws the hand links and joints on the frame, and also adds the handedness label if available.
def _draw_landmarks(frame, hand_landmarks, handedness):
    height, width = frame.shape[:2]
    for hand_index, landmarks in enumerate(hand_landmarks):
        points = [_normalized_to_pixel(lm, width, height) for lm in landmarks]
        for start_idx, end_idx in HAND_CONNECTIONS:
            cv2.line(frame, points[start_idx], points[end_idx], (0, 255, 0), 2)
        for point in points:
            cv2.circle(frame, point, 3, (0, 255, 255), -1)

        if handedness and hand_index < len(handedness):
            category = handedness[hand_index][0]
            label = f"{category.category_name} ({category.score:.2f})"
            x, y = _normalized_to_pixel(landmarks[0], width, height)
            y -= 10
            cv2.putText(
                frame,
                label,
                (max(0, x), max(20, y)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 200, 0),
                2,
            )


def run_hand_landmarks(model_path, camera_index=0, width=None, height=None):
    base_options = python.BaseOptions(model_asset_path=model_path)
    options = vision.HandLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.VIDEO,
        num_hands=2,
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)
    if width:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    if height:
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    if not cap.isOpened():
        raise RuntimeError(f"Unable to open camera index {camera_index}.")

    with vision.HandLandmarker.create_from_options(options) as landmarker:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
            timestamp_ms = int(time.monotonic() * 1000)
            result = landmarker.detect_for_video(mp_image, timestamp_ms)

            annotated = frame.copy()
            # Draw landmarks and handedness labels if available.
            if result.hand_landmarks:
                _draw_landmarks(annotated, result.hand_landmarks, result.handedness)

            cv2.imshow("MediaPipe Hands", annotated)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()
    cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(description="Run MediaPipe hand landmark detection.")
    default_model = Path(__file__).resolve().with_name("hand_landmarker.task")
    parser.add_argument("--model", default=str(default_model), help="Path to .task model")
    parser.add_argument("--camera", type=int, default=0, help="Camera index for OpenCV")
    parser.add_argument("--width", type=int, default=None, help="Capture width")
    parser.add_argument("--height", type=int, default=None, help="Capture height")
    args = parser.parse_args()

    run_hand_landmarks(args.model, args.camera, args.width, args.height)


if __name__ == "__main__":
    main()
