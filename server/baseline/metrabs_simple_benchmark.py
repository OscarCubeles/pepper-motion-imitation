"""
Simple Metrabs Pose Estimation Benchmark
Opens camera, runs pose estimation, and measures performance metrics.
Based on Pose_3D_metrabs_server_mediapipe_hand.py
"""

import cv2
import torch
import os
import time
import csv
from pathlib import Path

os.environ["KMP_DUPLICATE_LIB_OK"] = "FALSE"

import settings
settings.add_server_dir_to_path()

import numpy as np
import pose_mapping as pose_map


# ==========================================
# CONFIGURATION
# ==========================================

NUM_MEASUREMENTS = 100
OUTPUT_CSV = "metrabs_simple_benchmark.csv"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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


def load_metrabs_model():
    """Load Metrabs model and calibration."""
    print("Loading Metrabs model...")
    intrinsic_matrix, distortion_coeffs, model, joint_names, joint_edges = pose_map._load_metrabs_model()
    print(f"Model loaded on {DEVICE}")
    print(f"Skeleton: {settings.SKELETON}")
    return model, intrinsic_matrix, distortion_coeffs, joint_edges


def draw_keypoints(frame, poses2d, joint_edges, keypoint_radius=5, line_color=(0, 255, 0), point_color=(0, 0, 255)):
    """Draw 2D keypoints and skeleton on frame."""
    # Convert GPU tensor to numpy if needed
    if hasattr(poses2d, 'cpu'):
        poses2d = poses2d.cpu().numpy()
    elif not isinstance(poses2d, np.ndarray):
        poses2d = np.array(poses2d)
    
    frame_h, frame_w = frame.shape[:2]
    
    # Draw skeleton lines
    if joint_edges is not None:
        for edge in joint_edges:
            i, j = edge
            if i < len(poses2d[0]) and j < len(poses2d[0]):
                pt1 = poses2d[0, i].astype(int)
                pt2 = poses2d[0, j].astype(int)
                
                # Check if points are within frame
                if (0 <= pt1[0] < frame_w and 0 <= pt1[1] < frame_h and
                    0 <= pt2[0] < frame_w and 0 <= pt2[1] < frame_h):
                    cv2.line(frame, tuple(pt1), tuple(pt2), line_color, 2)
    
    # Draw keypoint circles
    for joint_idx in range(poses2d[0].shape[0]):
        pt = poses2d[0, joint_idx].astype(int)
        if 0 <= pt[0] < frame_w and 0 <= pt[1] < frame_h:
            cv2.circle(frame, tuple(pt), keypoint_radius, point_color, -1)
    
    return frame


# ==========================================
# BENCHMARK
# ==========================================

def benchmark():
    """Run benchmark loop."""
    measurements = []
    
    print(f"\n{'='*60}")
    print("METRABS SIMPLE BENCHMARK")
    print(f"{'='*60}")
    print(f"Target measurements: {NUM_MEASUREMENTS}")
    print(f"Device: {DEVICE}\n")
    
    # Setup
    cap = _open_camera()
    model, intrinsic_matrix, distortion_coeffs, joint_edges = load_metrabs_model()
    
    torch.backends.cudnn.benchmark = True
    
    frame_count = 0
    measurement_count = 0
    
    try:
        while measurement_count < NUM_MEASUREMENTS:
            ret, image = cap.read()
            if not ret:
                print("Failed to read frame, retrying...")
                continue
            
            frame_count += 1
            
            # Record time before inference
            start_time = time.perf_counter()
            
            # Run pose estimation (same way as server)
            try:
                pred = pose_map._detect_poses(model, image, intrinsic_matrix, distortion_coeffs)
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
                "model": "metrabs",
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
            if pred and "poses2d" in pred:
                frame_vis = draw_keypoints(frame_vis, pred["poses2d"], joint_edges)
            
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
            
            cv2.imshow("Metrabs Benchmark", frame_vis)
            
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
    print("METRABS POSE ESTIMATION BENCHMARK")
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
