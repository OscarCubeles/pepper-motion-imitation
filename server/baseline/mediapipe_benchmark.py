"""
Simple MediaPipe Pose Estimation Benchmark
Opens camera, runs pose estimation, and measures performance metrics.
Mirrors metrabs_simple_benchmark.py but uses MediaPipe Pose.
"""

import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np
import time
import csv
import os
from pathlib import Path

os.environ["KMP_DUPLICATE_LIB_OK"] = "FALSE"

import settings
settings.add_server_dir_to_path()


# ==========================================
# CONFIGURATION
# ==========================================

NUM_MEASUREMENTS = 100
OUTPUT_CSV = "mediapipe_benchmark.csv"
POSE_MODEL_PATH = "01-server/hand_detection/pose_landmarker_full.task"

# MediaPipe Pose skeleton connections
POSE_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 7),       # face left
    (0, 4), (4, 5), (5, 6), (6, 8),       # face right
    (9, 10),                               # mouth
    (11, 12),                              # shoulders
    (11, 13), (13, 15),                    # left arm
    (12, 14), (14, 16),                    # right arm
    (15, 17), (17, 19), (19, 15),          # left hand
    (16, 18), (18, 20), (20, 16),          # right hand
    (15, 21), (16, 22),                    # wrist pinky
    (11, 23), (12, 24),                    # torso top
    (23, 24),                              # hips
    (23, 25), (25, 27), (27, 29), (29, 31), (31, 27),  # left leg
    (24, 26), (26, 28), (28, 30), (30, 32), (32, 28),  # right leg
]


# ==========================================
# GPU MONITORING
# ==========================================

def get_gpu_stats():
    """Get current GPU memory and utilization statistics."""
    try:
        from pynvml import nvmlInit, nvmlDeviceGetHandleByIndex, nvmlDeviceGetMemoryInfo, nvmlDeviceGetUtilizationRates
        nvmlInit()
        handle = nvmlDeviceGetHandleByIndex(0)
        mem_info = nvmlDeviceGetMemoryInfo(handle)
        util_rates = nvmlDeviceGetUtilizationRates(handle)
        
        gpu_memory_mb = mem_info.used / 1024 / 1024
        gpu_util_percent = util_rates.gpu
        
        return {
            "gpu_memory_mb": round(gpu_memory_mb, 2),
            "gpu_util_percent": round(gpu_util_percent, 1)
        }
    except Exception as e:
        return {"gpu_memory_mb": 0, "gpu_util_percent": 0}


# ==========================================
# CAMERA & MODEL SETUP
# ==========================================

def _open_camera():
    """Open camera with same settings as server."""
    cap = cv2.VideoCapture(settings.CAMERA_INDEX, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)
    if not cap.isOpened():
        raise RuntimeError(f"Unable to open camera index {settings.CAMERA_INDEX}.")
    return cap


def load_mediapipe_model():
    """Load MediaPipe Pose model."""
    print("Loading MediaPipe Pose model...")
    
    base_options = python.BaseOptions(model_asset_path=POSE_MODEL_PATH)
    options = vision.PoseLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.IMAGE,
        num_poses=1,
        min_pose_detection_confidence=0.5,
        min_pose_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    
    landmarker = vision.PoseLandmarker.create_from_options(options)
    print("MediaPipe Pose model loaded")
    return landmarker


def draw_keypoints(frame, pose_landmarks):
    """Draw pose keypoints and skeleton on frame."""
    if not pose_landmarks or len(pose_landmarks) == 0:
        return frame
    
    frame_h, frame_w = frame.shape[:2]
    
    # Draw skeleton lines
    for start_idx, end_idx in POSE_CONNECTIONS:
        if start_idx < len(pose_landmarks) and end_idx < len(pose_landmarks):
            start = pose_landmarks[start_idx]
            end = pose_landmarks[end_idx]
            
            start_pos = (int(start.x * frame_w), int(start.y * frame_h))
            end_pos = (int(end.x * frame_w), int(end.y * frame_h))
            
            # Check bounds
            if (0 <= start_pos[0] < frame_w and 0 <= start_pos[1] < frame_h and
                0 <= end_pos[0] < frame_w and 0 <= end_pos[1] < frame_h):
                cv2.line(frame, start_pos, end_pos, (0, 255, 0), 2)
    
    # Draw keypoint circles
    for landmark in pose_landmarks:
        cx, cy = int(landmark.x * frame_w), int(landmark.y * frame_h)
        if 0 <= cx < frame_w and 0 <= cy < frame_h:
            cv2.circle(frame, (cx, cy), 4, (0, 0, 255), -1)
    
    return frame


# ==========================================
# BENCHMARK
# ==========================================

def benchmark():
    """Run benchmark loop."""
    measurements = []
    
    print(f"\n{'='*60}")
    print("MEDIAPIPE POSE BENCHMARK")
    print(f"{'='*60}")
    print(f"Target measurements: {NUM_MEASUREMENTS}")
    print(f"Model: {POSE_MODEL_PATH}\n")
    
    # Setup
    cap = _open_camera()
    landmarker = load_mediapipe_model()
    
    frame_count = 0
    measurement_count = 0
    
    try:
        while measurement_count < NUM_MEASUREMENTS:
            ret, image = cap.read()
            if not ret:
                print("Failed to read frame, retrying...")
                continue
            
            frame_count += 1
            
            # Convert BGR to RGB for MediaPipe
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_image)
            
            # Record time before inference
            start_time = time.perf_counter()
            
            # Run pose estimation
            try:
                detection_result = landmarker.detect(mp_image)
            except Exception as e:
                print(f"Inference error on frame {frame_count}: {e}")
                continue
            
            # Record time after inference
            end_time = time.perf_counter()
            
            # Get GPU stats
            gpu_stats = get_gpu_stats()
            
            # Calculate metrics
            inference_time_ms = (end_time - start_time) * 1000
            fps = 1.0 / (end_time - start_time) if (end_time - start_time) > 0 else 0
            
            # Store measurement
            measurement = {
                "model": "mediapipe",
                "frame_number": measurement_count + 1,
                "inference_time_ms": round(inference_time_ms, 4),
                "fps": round(fps, 2),
                "gpu_memory_mb": gpu_stats["gpu_memory_mb"],
                "gpu_util_percent": gpu_stats["gpu_util_percent"],
            }
            measurements.append(measurement)
            measurement_count += 1
            
            # Print progress
            if measurement_count % 10 == 0:
                print(f"Progress: {measurement_count}/{NUM_MEASUREMENTS}")
                print(f"  Inference time: {inference_time_ms:.4f}ms, FPS: {fps:.2f}")
            
            # Display on frame
            frame_vis = image.copy()
            
            # Draw keypoints if available
            if detection_result and detection_result.pose_landmarks:
                frame_vis = draw_keypoints(frame_vis, detection_result.pose_landmarks[0])
            
            cv2.putText(
                frame_vis,
                f"Frame {measurement_count}/{NUM_MEASUREMENTS}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2
            )
            cv2.putText(
                frame_vis,
                f"Inference: {inference_time_ms:.2f}ms | FPS: {fps:.1f}",
                (10, 70),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2
            )
            
            cv2.imshow("MediaPipe Pose Benchmark", frame_vis)
            
            # Exit with ESC
            if cv2.waitKey(1) & 0xFF == 27:
                print("\nBenchmark interrupted by user")
                break
    
    finally:
        cap.release()
        cv2.destroyAllWindows()
    
    # Summary
    if measurements:
        inference_times = [m["inference_time_ms"] for m in measurements]
        fps_values = [m["fps"] for m in measurements]
        gpu_memory = [m["gpu_memory_mb"] for m in measurements]
        gpu_util = [m["gpu_util_percent"] for m in measurements]
        
        print(f"\nSummary Statistics:")
        print(f"  Total frames: {frame_count}")
        print(f"  Successful measurements: {measurement_count}")
        print(f"\n  Inference Time (ms):")
        print(f"    Min: {min(inference_times):.4f}")
        print(f"    Max: {max(inference_times):.4f}")
        print(f"    Mean: {np.mean(inference_times):.4f}")
        print(f"    Std: {np.std(inference_times):.4f}")
        print(f"\n  FPS:")
        print(f"    Mean: {np.mean(fps_values):.2f}")
        print(f"    Min: {min(fps_values):.2f}")
        print(f"    Max: {max(fps_values):.2f}")
        print(f"\n  GPU Memory (MB):")
        print(f"    Mean: {np.mean(gpu_memory):.2f}")
        print(f"    Min: {min(gpu_memory):.2f}")
        print(f"    Max: {max(gpu_memory):.2f}")
        print(f"\n  GPU Utilization (%):")
        print(f"    Mean: {np.mean(gpu_util):.1f}")
        print(f"    Min: {min(gpu_util):.1f}")
        print(f"    Max: {max(gpu_util):.1f}")
    
    return measurements


# ==========================================
# CSV STORAGE
# ==========================================

def save_to_csv(measurements, output_file):
    """Save measurements to CSV."""
    file_exists = Path(output_file).exists()
    
    try:
        with open(output_file, 'a', newline='') as csvfile:
            fieldnames = ["model", "frame_number", "inference_time_ms", "fps", "gpu_memory_mb", "gpu_util_percent"]
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            
            if not file_exists:
                writer.writeheader()
                print(f"\nCreated CSV file: {output_file}")
            
            writer.writerows(measurements)
            print(f"Saved {len(measurements)} measurements to {output_file}")
            
    except Exception as e:
        print(f"Error saving to CSV: {e}")


# ==========================================
# MAIN
# ==========================================

def main():
    """Main function."""
    print("\n" + "="*60)
    print("MEDIAPIPE POSE ESTIMATION BENCHMARK")
    print("="*60)
    print(f"Measurements: {NUM_MEASUREMENTS}")
    print(f"Output file: {OUTPUT_CSV}")
    
    measurements = benchmark()
    
    if measurements:
        save_to_csv(measurements, OUTPUT_CSV)
        print(f"\nBenchmark completed!")
    else:
        print("\nNo measurements collected.")


if __name__ == "__main__":
    main()