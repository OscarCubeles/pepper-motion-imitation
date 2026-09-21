import argparse

import cv2


def main():
    parser = argparse.ArgumentParser(description="Open a webcam preview window.")
    parser.add_argument("--camera", type=int, default=0, help="Camera index. Default: 0")
    args = parser.parse_args()

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open webcam with index {args.camera}")

    print("Webcam opened. Press ESC or q to close.")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                raise RuntimeError("Could not read frame from webcam")

            cv2.imshow("Webcam", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == 27 or key == ord("q"):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
