import cv2
import torch
import os
import threading
import time
import json
import settings
import mediapipe as mp
import numpy as np
import visualizations
import queue as thread_queue
import pose_mapping as pose_map
import websocket_client as clientws
from angle_calculator import AngleCalculator
from angle_classifier_pepper import AngleClassifier
from angle_classifier_human import HumanArmClassifier
from hand_detection.EMA_smoothing import EMASmoothing
import hand_orientation as ho
from pose_handling import PoseHandler


from multiprocessing import Event, Process, Queue
from perf_metrics import consume_send_metrics, make_send_metrics
from websocket_client import _put_latest, _encode_pose_base64, shutdown_server
#from singularity_fsm import SingularityFSM

# Perspose: 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
os.environ["KMP_DUPLICATE_LIB_OK"] = "FALSE"
settings.add_server_dir_to_path()



def _create_hand_orientation_smoothers(alpha=0.3, confidence_threshold=0.85):
    tracked_point_ids = (0, 1, 5, 17)
    return {
        "Right": {point_id: EMASmoothing(alpha=alpha, confidence_threshold=confidence_threshold) for point_id in tracked_point_ids},
        "Left": {point_id: EMASmoothing(alpha=alpha, confidence_threshold=confidence_threshold) for point_id in tracked_point_ids},
    }


def _smooth_hand_points_for_orientation(hand_points_3d_anchored, smoothers):
    smoothed_points = {}

    for hand_label in ("Right", "Left"):
        hand_points = hand_points_3d_anchored.get(hand_label)
        if not isinstance(hand_points, dict) or not hand_points:
            continue

        hand_smoothers = smoothers.get(hand_label, {})
        smoothed_hand_points = {}

        for point_id in (0, 1, 5, 17):
            point_value = hand_points.get(point_id)
            if point_value is None:
                continue

            point_array = np.asarray(point_value, dtype=np.float32).reshape(-1)
            if point_array.size < 3 or not np.isfinite(point_array[:3]).all():
                continue

            smoother = hand_smoothers.get(point_id)
            if smoother is None:
                smoothed_point = point_array[:3]
            else:
                smoothed_point = smoother.smooth(point_array[:3], confidence=1.0)

            smoothed_hand_points[int(point_id)] = np.asarray(smoothed_point, dtype=np.float32)

        if smoothed_hand_points:
            smoothed_points[hand_label] = smoothed_hand_points

    return smoothed_points

def _open_camera():
    cap = cv2.VideoCapture(settings.CAMERA_INDEX, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)
    if not cap.isOpened():
        raise RuntimeError(f"Unable to open camera index {settings.CAMERA_INDEX}.")
    return cap


def setup_sender_thread(server, stop_event):
    """Setup sender queue, metrics, and worker thread.
    
    Args:
        server: WebSocket server instance
        stop_event: Event to signal shutdown
        
    Returns:
        tuple: (send_queue, send_metrics, send_metrics_lock, sender_thread)
    """
    send_queue = thread_queue.Queue(maxsize=1)
    send_metrics = make_send_metrics()
    send_metrics_lock = threading.Lock()
    sender_thread = threading.Thread(
        target=clientws._sender_worker,
        args=(server, send_queue, stop_event, send_metrics, send_metrics_lock),
        daemon=True,
    )
    sender_thread.start()
    
    return send_queue, send_metrics, send_metrics_lock, sender_thread


def setup_visualizer(stop_event, joint_edges, suppress_joint_indices):
    """Setup visualization queue and process.
    
    Args:
        stop_event: Event to signal shutdown
        joint_edges: Joint edge connections for visualization
        suppress_joint_indices: Indices of joints to suppress in visualization
        
    Returns:
        tuple: (vis_queue, vis_process) or (None, None) if visualization disabled
    """
    if settings.ENABLE_VISUALIZATION:
        vis_queue = Queue(maxsize=1)
        vis_process = Process(
            target=visualizations._visualizer_worker,
            args=(vis_queue, stop_event, joint_edges, suppress_joint_indices),
            daemon=True,
        )
        vis_process.start()
        print("Starting detection loop. Press 'q' in the visualization window to exit.")
        return vis_queue, vis_process
    else:
        print("[perf] headless benchmark mode enabled (visualizer disabled).")
        print("Starting detection loop. Press Ctrl+C to exit.")
        return None, None


def send_angles_to_client(full_payload, human_angles, pepper_angles, pepper_angles_prev, send_queue, active_client, frame_ready_ts, infer_fps=30.0):
    """Send computed angles and speeds to the client via WebSocket.
    
    Adds human and pepper angles (in radians) and computed speeds to the payload,
    encodes as JSON, and sends through the queue to the sender worker thread.
    
    Args:
        full_payload: Dictionary containing pose data to be sent
        human_angles: Dict of human joint angles in radians
        pepper_angles: Dict of pepper robot angles in radians
        pepper_angles_prev: Dict of previous pepper angles for speed computation
        send_queue: Queue for outgoing messages
        active_client: The connected client to send to
        frame_ready_ts: Timestamp when frame was ready
        infer_fps: Inference FPS for speed computation (default 30.0)
    """
    
    # Compute speeds from pose_handling module (convert fps to dt in seconds)
    dt = 1.0 / infer_fps if infer_fps > 0 else 0.033
    pepper_speeds = ph.compute_joint_speeds(pepper_angles_prev, pepper_angles, dt=dt)
    
    # Add angles and speeds to the payload being sent to client
    full_payload["angles"] = {
        "human": human_angles,  # in radians
        "pepper": pepper_angles,  # in radians, ready for Pepper
        "speeds": pepper_speeds,  # rad/s computed from angle deltas
    }
    
    # Encode and send to client
    encoded = json.dumps(full_payload)
    _put_latest(send_queue, (active_client, encoded, frame_ready_ts))


def send_angles_to_client_constrained(pose_handler, full_payload, human_angles, pepper_angles, pepper_angles_prev, singularity_right, singularity_left, send_queue, active_client, frame_ready_ts, infer_fps=30.0):
    """Send computed angles with dual singularity constraints and speeds to the client via WebSocket.
    
    Applies dual singularity constraints to pepper_angles (modifies ElbowYaw and WristYaw when
    dual singularity is detected, with logic based on hand orientation), then adds angles and speeds to the payload.
    
    Args:
        pose_handler: PoseHandler instance managing FSMs
        full_payload: Dictionary containing pose data to be sent (includes orientation_labels)
        human_angles: Dict of human joint angles in radians
        pepper_angles: Dict of pepper robot angles in radians (will be modified by constraints)
        pepper_angles_prev: Dict of previous pepper angles for speed computation
        singularity_right: Singularity dict for right arm
        singularity_left: Singularity dict for left arm
        send_queue: Queue for outgoing messages
        active_client: The connected client to send to
        frame_ready_ts: Timestamp when frame was ready
        infer_fps: Inference FPS for speed computation (default 30.0)
    """
    
    # Extract orientation labels from full payload
    orientation_labels = full_payload.get("hand_orientation", {})
    
    # Apply constraints for each arm independently based on its singularity type
    pepper_angles = pose_handler.apply_singularity_constraints(singularity_right, pepper_angles, orientation_labels, arm='right')
    pepper_angles = pose_handler.apply_singularity_constraints(singularity_left, pepper_angles, orientation_labels, arm='left')
    
    # Compute speeds from pose_handler
    dt = 1.0 / infer_fps if infer_fps > 0 else 0.033
    pepper_speeds = pose_handler.compute_joint_speeds(pepper_angles_prev, pepper_angles, dt=dt)
    
    # Add angles and speeds to the payload being sent to client
    full_payload["angles"] = {
        "human": human_angles,  # in radians
        "pepper": pepper_angles,  # in radians, ready for Pepper with constraints applied
        "speeds": pepper_speeds,  # rad/s computed from angle deltas
    }
    
    # Encode and send to client
    encoded = json.dumps(full_payload)
    _put_latest(send_queue, (active_client, encoded, frame_ready_ts))


def main():
    stop_event = Event()
    clientws._set_server_stop_event(stop_event)

    torch.backends.cudnn.benchmark = True

    cap = None
    hand_landmarker = None
    hand_worker = None
    server = None
    server_thread = None
    send_queue = None
    send_metrics = None
    send_metrics_lock = None
    sender_thread = None
    vis_queue = None
    vis_process = None
    wrisyaw_right, wrisyaw_left = 0, 0

    try:
        cap = _open_camera()

        # Setup Metrabs model and calibration
        intrinsic_matrix, distortion_coeffs, model, joint_names, joint_edges = pose_map._load_metrabs_model(skeleton=settings.SKELETON_SMPL_HEAD_30)
        
        # Setup joint indices
        suppress_joint_indices, wrist_joint_indices = pose_map._setup_metrabs_joints(joint_names)
        
        # Setup hand detection and server
        hand_worker, hand_landmarker, server, server_thread = pose_map._setup_hand_and_server()
        
        # Setup sender thread for pose transmission
        send_queue, send_metrics, send_metrics_lock, sender_thread = setup_sender_thread(server, stop_event)
        
        # Setup angle calculator and classifier
        calculator = AngleCalculator()
        classifier = AngleClassifier()
        human_arm_classifier = HumanArmClassifier()
        pose_handler = PoseHandler()  # Initialize pose constraint handler with FSMs
        

        # Setup visualization worker
        vis_queue, vis_process = setup_visualizer(stop_event, joint_edges, suppress_joint_indices)

        vis_interval = 1.0 / max(settings.VISUALIZER_MAX_FPS, 1e-3)
        next_vis_push = 0.0
        infer_count = 0
        infer_time_sum = 0.0
        perf_t0 = time.perf_counter()
        latest_infer_fps = None
        pose_out = np.zeros((settings.OUT_JOINT_COUNT, settings.PARAMS_PER_JOINT), dtype=np.float32)
        copy_stream = torch.cuda.Stream()
        pending_pose_transfer = None
        frame_id = 0
        pepper_angles_prev = {}  # Track previous angles for speed computation

        while not stop_event.is_set():
            active_client = clientws._get_active_client()
            
            if settings.REQUIRE_CLIENT_MESSAGE_TO_START and not clientws._stream_ready.is_set():
                time.sleep(0.01)
                continue
            
            if active_client is None and not settings.ENABLE_VISUALIZATION:
                pending_pose_transfer = None
                time.sleep(0.005)
                continue

            ret, image = cap.read()
            if not ret:
                print("Can't receive frame (stream end?). Exiting ...")
                break
            frame_ready_ts = time.perf_counter()

            frame_id += 1
            current_frame_id = frame_id
            frame_timestamp_ms = int(time.monotonic() * 1000)
            need_hand_points = (active_client is not None) or settings.ENABLE_VISUALIZATION
            if need_hand_points:
                # Start CPU hand work as early as possible so it overlaps the
                # current frame's GPU body inference.
                hand_worker.submit_frame(image, current_frame_id, frame_timestamp_ms)

            pred = None
            current_pose_transfer = None
            infer_start = time.perf_counter()
            try:
                pred = pose_map._detect_poses(model, image, intrinsic_matrix, distortion_coeffs)
                poses3d_gpu = pred.get("poses3d") if pred is not None else None

                if active_client is not None and poses3d_gpu is not None and poses3d_gpu.shape[0] > 0:
                    current_pose_transfer = pose_map._start_async_pose_transfer(
                        poses3d_gpu, current_frame_id, frame_ready_ts, copy_stream
                    )
            except Exception as exc:
                print(f"Pose estimation failed: {exc}")
            finally:
                infer_count += 1
                infer_time_sum += time.perf_counter() - infer_start

            now = time.perf_counter()
            perf_elapsed = now - perf_t0
            if perf_elapsed > 0.0 and infer_count > 0:
                latest_infer_fps = infer_count / perf_elapsed
            
            human_angles = {}
            pepper_angles = {}
            missing_keypoints = []
            singularity_left = None
            singularity_right = None
            hand_points_3d_anchored = {}
            pose_feasibility = {"is_feasible": False, "violations": {"left": [], "right": []}}
            pepper_feasibility = {"is_feasible": False, "violations": {"left": [], "right": []}}
            hand_orientation_smoothers = _create_hand_orientation_smoothers()
            
            if active_client is not None and pending_pose_transfer is not None:
                pending_pose_transfer["done_event"].synchronize()
                pose3d_prev_cpu = pending_pose_transfer["cpu_tensor"].numpy()
                pending_frame_id = pending_pose_transfer["frame_id"]
                _, pending_hand_points_3d_world = hand_worker.get_result(pending_frame_id, remove=True)
                if pending_hand_points_3d_world:
                    hand_points_3d_anchored = pose_map._anchor_hand_points_to_metrabs_wrist(
                        pose3d_prev_cpu,
                        pending_hand_points_3d_world,
                        wrist_joint_indices,
                    )
                pose_map._map_pose_20(pose3d_prev_cpu, hand_points_3d_anchored, pose_out)
                smoothed_hand_points_3d_anchored = _smooth_hand_points_for_orientation(
                    hand_points_3d_anchored,
                    hand_orientation_smoothers,
                )
                # Get hand orientation labels and prepare full payload
                orientation_labels = pose_map._get_hand_orientation_labels(smoothed_hand_points_3d_anchored)
                full_payload = pose_map.prepare_full_pose_data(pose_out, orientation_labels)
                
                #print(f"Orientation labels: {orientation_labels}")
                #print(f"full_payload for client: {full_payload} \n\n")
                # Compute BOTH human and pepper angles from the SAME full payload
                # This also computes forearm directions, upper arm directions, and wristyaw angles
                both_angles = calculator.compute_both_angles_from_payload_ikpy(joint_edges, full_payload)
                human_angles = both_angles.get("human", {})
                pepper_angles = both_angles.get("pepper", {})
                missing_keypoints = both_angles.get("missing_keypoints", [])
                

                #print(f"Pepper angles (radians): {pepper_angles}")
                forearm_direction_right = both_angles.get("forearm_direction_right")
                forearm_direction_left = both_angles.get("forearm_direction_left")
                upper_arm_direction_right = both_angles.get("upper_arm_direction_right")
                upper_arm_direction_left = both_angles.get("upper_arm_direction_left")
                



                # Extract wristyaw from computed angles
                wrisyaw_right = pepper_angles.get("RWristYaw", 0.0)
                wrisyaw_left = pepper_angles.get("LWristYaw", 0.0)

                
                pose_feasibility = classifier.check_feasibility_fsm(
                    human_angles,
                    forearm_direction_right,
                    forearm_direction_left,
                    orientation_labels,
                )
                pepper_feasibility = classifier.check_feasibility_fsm(
                    pepper_angles,
                    forearm_direction_right,
                    forearm_direction_left,
                    orientation_labels,
                )

                #visualizations.draw_forearm_labels(
                #    image,
                #    forearm_direction_right,
                #    forearm_direction_left,
                #    upper_arm_direction_right,
                #    upper_arm_direction_left,
                #    wrisyaw_right,
                #    wrisyaw_left,
                #)

                # Send angles and speeds to client
                #send_angles_to_client(full_payload, human_angles, pepper_angles, pepper_angles_prev, send_queue, active_client, pending_pose_transfer["frame_ready_ts"], infer_fps=latest_infer_fps or 30.0)
                
                # Update previous angles for next frame's speed computation
                pepper_angles_prev = pepper_angles.copy() if pepper_angles else {}
                
                # Compute singularity checks for both arms (always compute if we have angles)
                if human_angles:
                    singularity_right = classifier.check_singularity_poses_fsm(human_angles, arm='right')
                    singularity_left = classifier.check_singularity_poses_fsm(human_angles, arm='left')
                    


                    # Classify check if pepper pose is doable for human
                    human_reachability_right = human_arm_classifier.classify(
                        sp=human_angles.get("ShoulderPitch_Right"),
                        ey=human_angles.get("ElbowYaw_Right"),
                        arm="right",
                        singularity=singularity_right
                    )
                    human_reachability_left = human_arm_classifier.classify(
                        sp=human_angles.get("ShoulderPitch_Left"),
                        ey=human_angles.get("ElbowYaw_Left"),
                        arm="left",
                        singularity=singularity_left
                    )
                    
                    # Optional: print results for debugging
                    #if human_reachability_right == "NON_HUMAN_DOABLE":
                    #    print(f"[Human Reachability] RIGHT: {human_reachability_right}, elbow_yaw={np.degrees(human_angles.get('ElbowYaw_Right')):.2f}, shoulder_pitch={np.degrees(human_angles.get('ShoulderPitch_Right')):.2f}")
                    #if human_reachability_left == "NON_HUMAN_DOABLE":
                    #    print(f"[Human Reachability] LEFT: {human_reachability_left}, elbow_yaw={np.degrees(human_angles.get('ElbowYaw_Left')):.2f}, shoulder_pitch={np.degrees(human_angles.get('ShoulderPitch_Left')):.2f}")
            

            
                # Send angles and speeds to client with constraint handling
                send_angles_to_client_constrained(pose_handler, full_payload, human_angles, pepper_angles, pepper_angles_prev, singularity_right, singularity_left, send_queue, active_client, pending_pose_transfer["frame_ready_ts"], infer_fps=latest_infer_fps or 30.0)
                    

            pending_pose_transfer = current_pose_transfer

            # Get poses for visualization and logging
            poses2d_vis = pose_map._to_cpu_numpy(pred.get("poses2d")) if pred is not None else None
            poses3d_vis = pose_map._to_cpu_numpy(pred.get("poses3d")) if pred is not None else None

            # If we haven't computed angles yet (no active client), compute them from visualization data
            # This ensures angles are always available for visualization
            if not human_angles and poses3d_vis is not None and poses3d_vis.shape[0] > 0:
                first_person_keypoints = poses3d_vis[0]
                calculator.set_keypoints(first_person_keypoints)
                pepper_angles = calculator.compute_joint_angles()
                human_angles = pepper_angles.copy()
                pose_feasibility = classifier.check_feasibility_fsm(human_angles, None, None, None)
                pepper_feasibility = classifier.check_feasibility_fsm(pepper_angles, None, None, None)
            

            if settings.ENABLE_VISUALIZATION and now >= next_vis_push:
               hand_points_2d, hand_points_3d_world = hand_worker.get_result(current_frame_id)
               
               # Create a copy of the frame for visualization
               vis_frame = image.copy()
               
               _put_latest(
                   vis_queue,
                   {
                       "frame_bgr": vis_frame,
                       "poses2d": poses2d_vis,
                       "poses3d": poses3d_vis,
                       "extra_points_2d": hand_points_2d or {},
                       "extra_points_3d": hand_points_3d_anchored or {},
                       "inference_fps": latest_infer_fps,
                       "hand_points_3d_world": hand_points_3d_world,
                       "human_angles": human_angles,
                       "pepper_angles": pepper_angles,
                       "pose_feasibility": pose_feasibility,
                       "pepper_feasibility": pepper_feasibility,
                       "singularity_right": singularity_right,
                       "singularity_left": singularity_left,
                       "missing_keypoints": missing_keypoints,
                   },
               )
               next_vis_push = now + vis_interval

            if perf_elapsed >= settings.PERF_LOG_INTERVAL_SEC:
                infer_fps = infer_count / perf_elapsed if perf_elapsed > 0 else 0.0
                if send_metrics is not None and send_metrics_lock is not None:
                    send_count, avg_total_latency_ms = consume_send_metrics(send_metrics, send_metrics_lock)
                else:
                    send_count = 0
                    avg_total_latency_ms = None
                send_fps = send_count / perf_elapsed if perf_elapsed > 0 else 0.0
                latest_infer_fps = infer_fps
                if not settings.ENABLE_VISUALIZATION:
                    latency_text = f"{avg_total_latency_ms:.1f}" if avg_total_latency_ms is not None else "n/a"
                    print(
                        f"[perf] infer_fps={infer_fps:.1f} "
                        f"send_fps={send_fps:.1f} "
                        f"total_latency_ms={latency_text}"
                    )
                infer_count = 0
                infer_time_sum = 0.0
                perf_t0 = now

    except KeyboardInterrupt:
        print("Stopping server...")
    finally:
        shutdown_server(
            stop_event,
            send_queue,
            sender_thread,
            vis_queue,
            vis_process,
            cap,
            hand_worker,
            hand_landmarker,
            server,
            server_thread,
        )


if __name__ == "__main__":
    main()
