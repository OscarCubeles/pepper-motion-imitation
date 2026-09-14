import cv2
import torch
import numpy as np
import os
import sys
import argparse
import time
from pathlib import Path
import settings
from ikpy.chain import Chain
import json
import ikpy_utils as ikpyu

# Add parent directory to path for importing perf_metrics
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pose_mapping as pose_map

# Add server directory to path for imports
settings.add_server_dir_to_path()



def draw_angles_on_frame(frame, angles_dict, font_scale=0.6):
    """Draw joint angles on frame (only controllable joints).
    
    Args:
        frame: OpenCV frame
        angles_dict: Dict with 'right' and 'left' keys containing (ik_result, chain) tuples
        font_scale: Font size
    """
    h, w = frame.shape[:2]
    y_offset = 30
    x_left = 10
    x_right = w // 2
    
    for side, (ik_result, chain) in angles_dict.items():
        # Filter to only controllable joints
        angles, names, indices = ikpyu.get_controllable_joints(chain, ik_result)
        print(f"Controllable joints for {side} arm: {list(zip(names, angles))}\n\n\n")
        x = x_right if side == 'right' else x_left
        y = y_offset
        
        # Arm header
        cv2.putText(frame, f"{side.upper()} ARM", (x, y),
                   cv2.FONT_HERSHEY_SIMPLEX, font_scale + 0.2, (0, 255, 0), 2)
        y += 25
        
        # Joint angles (controllable only)
        for idx, name, angle in zip(indices, names, angles):
            angle_deg = np.degrees(angle)
            text = f"[{idx}] {name}: {angle_deg:.1f}°"
            cv2.putText(frame, text, (x, y),
                       cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), 1)
            y += 20


def draw_end_effector_coords(frame, metrabs_pose_mm, arm='right', joint_indices=None):
    """Draw end effector absolute and relative coordinates on frame."""

    if joint_indices is None:
        raise ValueError("joint_indices must be provided")

    # Use actual SMPL joint names
    wrist_key = 'rwri_smpl' if arm == 'right' else 'lwri_smpl'

    wrist_idx = joint_indices[wrist_key]
    thor_idx = joint_indices['thor_smpl']

    if wrist_idx is None or thor_idx is None:
        return

    # Extract positions in mm
    wrist_mm = metrabs_pose_mm[wrist_idx]
    thor_mm = metrabs_pose_mm[thor_idx]

    # Convert to meters
    wrist_m = wrist_mm / 1000.0
    thor_m = thor_mm / 1000.0

    # Calculate relative position (wrist relative to thorax)
    relative_pos_raw = wrist_m - thor_m

    # Swap Y and Z to align Metrabs coordinates with ikpy convention
    relative_pos_ikpy = np.array([
        relative_pos_raw[0],
        relative_pos_raw[2],
        relative_pos_raw[1]
    ])

    h, w = frame.shape[:2]
    x = w // 2 if arm == 'right' else 10
    y = h - 200

    # Header
    cv2.putText(
        frame,
        f"{arm.upper()} END EFFECTOR",
        (x, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 255, 0),
        2
    )
    y += 20

    # Absolute wrist position
    cv2.putText(
        frame,
        "Absolute (Metrabs):",
        (x, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (200, 200, 200),
        1
    )
    y += 16

    cv2.putText(frame, f"  X: {wrist_m[0]:7.4f} m", (x, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    y += 15

    cv2.putText(frame, f"  Y: {wrist_m[1]:7.4f} m", (x, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    y += 15

    cv2.putText(frame, f"  Z: {wrist_m[2]:7.4f} m", (x, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    # Relative wrist position
    y += 18

    cv2.putText(
        frame,
        "(thor_smpl):",
        (x, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (200, 200, 200),
        1
    )
    y += 16

    cv2.putText(frame, f"  X: {thor_m[0]:7.4f} m", (x, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    y += 15

    cv2.putText(frame, f"  Y: {thor_m[1]:7.4f} m", (x, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    y += 15

    cv2.putText(frame, f"  Z: {thor_m[2]:7.4f} m", (x, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)




def main():
    calibrate = False
    
    parser = argparse.ArgumentParser(description="Real-time Metrabs + ikpy IK")
    parser.add_argument('--camera', type=int, default=0, help='Camera index (default: 0)')
    parser.add_argument('--fps', type=int, default=10, help='Target FPS (default: 10)')
    args = parser.parse_args()
    
    print("[*] Loading models...")
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Load chains
    left_arm_chain, right_arm_chain = ikpyu.load_pepper_chains(script_dir)
    print(f"  ✓ Loaded Pepper chains (left: {len(left_arm_chain)} joints, right: {len(right_arm_chain)} joints)")
    
    # Load Metrabs with pose_mapping utilities (handles CUDA device setup)
    metrabs_model, intrinsic_matrix, distortion_coeffs, joint_names, joint_edges = ikpyu.load_metrabs_model()
    # Get joint edges for SKELETON_SMPL_HEAD_30 to match the skeleton used in detection
    joint_edges = metrabs_model.per_skeleton_joint_edges[settings.SKELETON_SMPL_HEAD_30].cpu().numpy()
    # Get dynamic joint indices for the SMPL+Head 30 skeleton
    joint_indices = ikpyu.get_joint_indices(metrabs_model, settings.SKELETON_SMPL_HEAD_30)
    print("  ✓ Loaded Metrabs model")
    
    # Open camera
    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        print(f"Error: Cannot open camera {args.camera}")
        sys.exit(1)
    
    cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)
    print(f"  ✓ Opened camera {args.camera}")
    print("\n[*] Starting real-time loop (press 'q' to quit)...")
    
    frame_count = 0
    fps_time = time.time()
    target_dt = 1.0 / args.fps
    
    try:
        while True:
            loop_start = time.time()
            
            # Capture frame
            ret, frame = cap.read()
            if not ret:
                print("Error: Cannot read frame")
                break
            
            frame_count += 1
            
            # Detect poses with Metrabs using pose_mapping wrapper (handles CUDA device setup)
            #try:
            # Use pose_map._detect_poses which properly handles device placement
            try:
                pred = pose_map._detect_poses(metrabs_model, frame, intrinsic_matrix, distortion_coeffs, skeleton=settings.SKELETON_SMPL_HEAD_30)
            except Exception as e:
                print(f"Error during pose detection: {e}")
                pred = None
            if pred is None or len(pred.get('poses3d', [])) == 0:
                cv2.putText(frame, "No pose detected", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            else:
                # Get poses from prediction (already on CPU from pose_map._detect_poses)
                poses3d_list = pose_map._to_cpu_numpy(pred.get('poses3d'))
                poses2d_list = pose_map._to_cpu_numpy(pred.get('poses2d'))
                metrabs_pose_mm = poses3d_list[0]
                
                if calibrate:
                    ikpyu.update_metrabs_calibration(
                        metrabs_pose_mm,
                        joint_indices,
                        json_path="metrabs_workspace.json"
                    )
                    

                # Draw skeleton keypoints on frame
                ikpyu.draw_skeleton_keypoints(frame, poses2d_list, joint_edges, metrabs_model.per_skeleton_joint_names[settings.SKELETON_SMPL_HEAD_30])
                
                # Compute IK for both arms
                angles_dict = {}
                for arm in ['right', 'left']:
                    ik_result, chain = ikpyu.compute_ik_from_metrabs_coordinate_adjusted_2(
                        metrabs_pose_mm, left_arm_chain, right_arm_chain, arm, joint_indices
                    )
                    angles_dict[arm] = (ik_result, chain)

                #print(f"Angles_dict: {angles_dict}\n\n")
                
                # Draw joint angles and end effector coordinates on frame
                draw_angles_on_frame(frame, angles_dict)
                draw_end_effector_coords(frame, metrabs_pose_mm, arm='right', joint_indices=joint_indices)
                draw_end_effector_coords(frame, metrabs_pose_mm, arm='left', joint_indices=joint_indices)
            
            #except Exception as e:
            #    #print(f"Error in pose detection: {e}")
            #    cv2.putText(frame, f"Error: {str(e)[:40]}", (10, 30),
            #               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            # Display FPS
            if frame_count % 30 == 0:
                elapsed = time.time() - fps_time
                current_fps = 30 / elapsed
                fps_time = time.time()
                print(f"[Frame {frame_count}] FPS: {current_fps:.1f}")
            
            cv2.putText(frame, f"FPS: {args.fps}", (10, frame.shape[0] - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
            
            # Display
            cv2.imshow("Metrabs + ikpy IK", frame)
            
            # Regulate FPS
            loop_time = time.time() - loop_start
            sleep_time = target_dt - loop_time
            if sleep_time > 0:
                time.sleep(sleep_time)
            
            # Quit on 'q'
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    
    finally:
        cap.release()
        cv2.destroyAllWindows()
        print(f"\n[✓] Processed {frame_count} frames. Exiting.")


if __name__ == "__main__":
    main()