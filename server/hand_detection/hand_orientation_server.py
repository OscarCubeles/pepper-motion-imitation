import cv2
import torch
import base64
import os
import sys
import threading
import time
import settings
import mediapipe as mp
import numpy as np
import queue as thread_queue

from collections import deque
from multiprocessing import Event, Process, Queue
from pathlib import Path
from queue import Empty, Full

from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from websocket_server import WebsocketServer

from calibration_utils import load_metrabs_calibration
from metrabs_pytorch.inference import metrabsInference

os.environ["KMP_DUPLICATE_LIB_OK"] = "FALSE"
settings.add_server_dir_to_path()

from visualization import PoseVisualizer
from perf_metrics import consume_send_metrics, make_send_metrics, record_send_metrics




# Mapping from Metrabs kinectv2_25 joints to outgoing 12x3 pose:
# 0:Nose, 1:Neck, 2:RShoulder, 3:RElbow, 4:RWrist, 5:LShoulder, 6:LElbow, 7:LWrist, 8:Torso, 9:SpineBase
# 10:RTip, 11:LTip
TARGET_BODY_IDXS = np.arange(10, dtype=np.int64)
SOURCE_BODY_IDXS = np.array([3, 2, 8, 9, 10, 4, 5, 6, 1, 0], dtype=np.int64)

HAND_LANDMARK_IDS = (0, 12)
HAND_JOINT_NAME_TOKENS = ("hand", "thumb", "index", "middle", "ring", "pinky", "finger", "tip")
HAND_JOINT_NAME_PREFIXES = ("lhan", "rhan", "lhnd", "rhnd", "lthu", "rthu", "lfin", "rfin", "lhtip", "rhtip")
METRABS_UNITS_TO_METERS = 0.001


def _load_calibration():
    return load_metrabs_calibration()


_active_client = None
_active_client_lock = threading.Lock()
_stream_ready = threading.Event()
_server_stop_event = None


def _set_active_client(client):
    global _active_client
    with _active_client_lock:
        _active_client = client


def _get_active_client():
    with _active_client_lock:
        return _active_client


def _clear_active_client(client):
    global _active_client
    with _active_client_lock:
        if _active_client is not None and _active_client.get("id") == client.get("id"):
            _active_client = None
            return True
    return False


def _set_server_stop_event(stop_event):
    global _server_stop_event
    _server_stop_event = stop_event


def _sender_worker(
    server: WebsocketServer,
    send_queue,
    stop_event: Event,
    send_metrics: dict,
    send_metrics_lock: threading.Lock,
):
    while not stop_event.is_set():
        try:
            payload = send_queue.get(timeout=0.1)
        except Empty:
            continue

        if payload is None:
            break

        client, encoded, frame_ready_ts = payload
        if client is None:
            continue

        active_client = _get_active_client()
        if active_client is None or active_client.get("id") != client.get("id"):
            continue

        try:
            server.send_message(client, encoded)
        except Exception:
            _clear_active_client(client)
        else:
            record_send_metrics(send_metrics, send_metrics_lock, frame_ready_ts)


def _open_camera():
    cap = cv2.VideoCapture(settings.CAMERA_INDEX, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)
    if not cap.isOpened():
        raise RuntimeError(f"Unable to open camera index {settings.CAMERA_INDEX}.")
    return cap


def _open_hand_landmarker():
    hand_model_path = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "hand_detection", "hand_landmarker.task")
    )
    hand_options = vision.HandLandmarkerOptions(
        base_options=python.BaseOptions(model_asset_path=hand_model_path),
        running_mode=vision.RunningMode.VIDEO,
        num_hands=2,
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    return vision.HandLandmarker.create_from_options(hand_options)


def _put_latest(queue_obj: Queue, payload):
    try:
        queue_obj.put_nowait(payload)
        return
    except Full:
        pass
    try:
        queue_obj.get_nowait()
    except Empty:
        pass
    try:
        queue_obj.put_nowait(payload)
    except Full:
        pass


def _to_cpu_numpy(tensor):
    if tensor is None:
        return None
    cpu_tensor = tensor.detach().to(device="cpu", non_blocking=True)
    torch.cuda.current_stream().synchronize()
    return cpu_tensor.numpy()


def _encode_pose_base64(pose_out: np.ndarray) -> str:
    payload_bytes = np.asarray(pose_out, dtype="<f4").tobytes(order="C")
    return base64.b64encode(payload_bytes).decode("ascii")


def _normalized_to_pixel(lm, width, height):
    x = min(max(int(lm.x * width), 0), width - 1)
    y = min(max(int(lm.y * height), 0), height - 1)
    return x, y


def _get_hand_joint_indices(joint_names):
    hand_joint_indices = []
    for joint_idx, joint_name in enumerate(joint_names):
        name = str(joint_name).lower()
        if "wrist" in name:
            continue
        if any(token in name for token in HAND_JOINT_NAME_TOKENS) or any(
            name.startswith(prefix) for prefix in HAND_JOINT_NAME_PREFIXES
        ):
            hand_joint_indices.append(joint_idx)
    return tuple(hand_joint_indices)


def _get_wrist_joint_indices(joint_names):
    wrist_joint_indices = {}
    for joint_idx, joint_name in enumerate(joint_names):
        name = str(joint_name).lower()
        if "wrist" not in name and "wri" not in name:
            continue
        if ("left" in name) or ("lwri" in name) or name.startswith("l"):
            wrist_joint_indices.setdefault("Left", int(joint_idx))
        elif ("right" in name) or ("rwri" in name) or name.startswith("r"):
            wrist_joint_indices.setdefault("Right", int(joint_idx))
    return wrist_joint_indices


def _assign_hands(hands):
    hand_map = {}
    unassigned = []

    for hand in hands:
        label = hand.get("label")
        if label in ("Left", "Right") and label not in hand_map:
            hand_map[label] = hand
        else:
            unassigned.append(hand)

    if unassigned:
        def mean_x(hand):
            points = hand.get("points_2d", {})
            if not points:
                return float("inf")
            return float(np.mean([p[0] for p in points.values()]))

        unassigned.sort(key=mean_x)
        for hand in unassigned:
            if "Left" not in hand_map:
                hand_map["Left"] = hand
            elif "Right" not in hand_map:
                hand_map["Right"] = hand

    return hand_map


def _extract_hand_points(image_bgr, landmarker, timestamp_ms):
    height, width = image_bgr.shape[:2]
    frame_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
    result = landmarker.detect_for_video(mp_image, timestamp_ms)
    if not result.hand_landmarks:
        return {}, {}
    world_landmarks_list = getattr(result, "hand_world_landmarks", None) or []

    hands = []
    for hand_index, hand_landmarks in enumerate(result.hand_landmarks):
        label = None
        if result.handedness and hand_index < len(result.handedness):
            handedness = result.handedness[hand_index]
            if handedness:
                label = handedness[0].category_name

        points_2d = {}
        points_3d = {}
        hand_world_landmarks = world_landmarks_list[hand_index] if hand_index < len(world_landmarks_list) else None
        for idx in HAND_LANDMARK_IDS:
            if idx < len(hand_landmarks):
                lm = hand_landmarks[idx]
                x_px, y_px = _normalized_to_pixel(lm, width, height)
                points_2d[idx] = (x_px, y_px)
                if hand_world_landmarks is not None and idx < len(hand_world_landmarks):
                    wlm = hand_world_landmarks[idx]
                    points_3d[idx] = np.array([float(wlm.x), float(wlm.y), float(wlm.z)], dtype=np.float32)

        if points_2d:
            hands.append({"label": label, "points_2d": points_2d, "points_3d": points_3d})

    hand_map = _assign_hands(hands)
    hand_points_2d = {label: data["points_2d"] for label, data in hand_map.items()}
    hand_points_3d = {label: data["points_3d"] for label, data in hand_map.items()}
    return hand_points_2d, hand_points_3d


class _HandDetectionWorker:
    """Runs MediaPipe hand detection on a background thread so the main
    loop can continue GPU inference without waiting for CPU-bound hand
    detection. Results are keyed by frame ID so pose fusion can match the
    correct hand result to the correct pose frame."""

    def __init__(self, landmarker):
        self._landmarker = landmarker
        self._lock = threading.Lock()
        self._completed_results = {}
        self._completed_order = deque()
        self._max_completed_results = 8
        self._pending_frame = None
        self._pending_frame_id = None
        self._pending_ts = None
        self._frame_event = threading.Event()
        self._stop = False
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def submit_frame(self, image_bgr, frame_id, timestamp_ms):
        """Queue the latest frame for hand detection (drops stale frames)."""
        with self._lock:
            self._pending_frame = image_bgr
            self._pending_frame_id = frame_id
            self._pending_ts = timestamp_ms
        self._frame_event.set()

    def get_result(self, frame_id, remove=False):
        """Return the exact hand result for a frame ID if it is available."""
        with self._lock:
            result = self._completed_results.get(frame_id)
            if result is None:
                return {}, {}
            if remove:
                self._completed_results.pop(frame_id, None)
                try:
                    self._completed_order.remove(frame_id)
                except ValueError:
                    pass
            return result

    def stop(self):
        self._stop = True
        self._frame_event.set()
        self._thread.join(timeout=1.0)

    def _store_result(self, frame_id, result):
        if frame_id in self._completed_results:
            try:
                self._completed_order.remove(frame_id)
            except ValueError:
                pass
        self._completed_results[frame_id] = result
        self._completed_order.append(frame_id)
        while len(self._completed_order) > self._max_completed_results:
            expired_frame_id = self._completed_order.popleft()
            self._completed_results.pop(expired_frame_id, None)

    def _run(self):
        while not self._stop:
            self._frame_event.wait(timeout=0.1)
            self._frame_event.clear()
            with self._lock:
                frame = self._pending_frame
                frame_id = self._pending_frame_id
                ts = self._pending_ts
                self._pending_frame = None
                self._pending_frame_id = None
            if frame is None or frame_id is None:
                continue
            result = _extract_hand_points(frame, self._landmarker, ts)
            with self._lock:
                self._store_result(frame_id, result)


def _anchor_hand_points_to_metrabs_wrist(
    pose3d_metrabs,
    hand_points_world_3d: dict[str, dict[int, np.ndarray]],
    wrist_joint_indices: dict[str, int],
    units_to_meters: float = METRABS_UNITS_TO_METERS,
) -> dict[str, dict[int, np.ndarray]]:
    if pose3d_metrabs is None:
        return {}
    pose3d_arr = np.asarray(pose3d_metrabs, dtype=np.float32)
    if pose3d_arr.ndim != 2 or pose3d_arr.shape[0] == 0:
        return {}
    if units_to_meters <= 0:
        return {}

    meters_to_pose_units = float(1.0 / units_to_meters)
    anchored_points: dict[str, dict[int, np.ndarray]] = {}
    for label, label_points in hand_points_world_3d.items():
        if not label_points or 0 not in label_points:
            continue

        wrist_joint_idx = wrist_joint_indices.get(label)
        if wrist_joint_idx is None or wrist_joint_idx >= pose3d_arr.shape[0]:
            continue

        metrabs_wrist = np.asarray(pose3d_arr[wrist_joint_idx, :3], dtype=np.float32)
        mp_wrist = np.asarray(label_points[0], dtype=np.float32).reshape(-1)
        if mp_wrist.size < 3 or not np.isfinite(metrabs_wrist).all() or not np.isfinite(mp_wrist[:3]).all():
            continue

        anchored_label_points: dict[int, np.ndarray] = {}
        for point_idx, point_world in label_points.items():
            point_world_arr = np.asarray(point_world, dtype=np.float32).reshape(-1)
            if point_world_arr.size < 3 or not np.isfinite(point_world_arr[:3]).all():
                continue
            delta_pose_units = (point_world_arr[:3] - mp_wrist[:3]) * meters_to_pose_units
            anchored_label_points[int(point_idx)] = (metrabs_wrist + delta_pose_units).astype(np.float32)

        if anchored_label_points:
            anchored_points[str(label)] = anchored_label_points

    return anchored_points


def _map_pose_12(pose3d_cpu: np.ndarray, hand_points_3d: dict[str, dict[int, np.ndarray]], pose_out: np.ndarray):
    pose_out.fill(0.0)
    pose_out[TARGET_BODY_IDXS] = pose3d_cpu[SOURCE_BODY_IDXS]
    pose_out *= 0.001

    # MediaPipe wrist (landmark 0) is used only for anchoring; only fingertip
    # positions are streamed to the client for hand open/close.
    right_hand = hand_points_3d.get("Right")
    left_hand = hand_points_3d.get("Left")
    if right_hand:
        if 12 in right_hand:
            pose_out[10, :] = right_hand[12] * 0.001
    if left_hand:
        if 12 in left_hand:
            pose_out[11, :] = left_hand[12] * 0.001


def _visualizer_worker(
    vis_queue: Queue,
    stop_event: Event,
    joint_edges: np.ndarray,
    suppress_joint_indices: tuple[int, ...],
):
    visualizer = PoseVisualizer(source_name="dl-pose")
    try:
        while not stop_event.is_set():
            try:
                payload = vis_queue.get(timeout=0.1)
            except Empty:
                continue
            if payload is None:
                break

            vis_result = visualizer.update(
                frame_bgr=payload["frame_bgr"],
                poses2d=payload["poses2d"],
                poses3d=payload["poses3d"],
                joint_edges=joint_edges,
                extra_points_2d=payload.get("extra_points_2d"),
                extra_points_3d=payload.get("extra_points_3d"),
                suppress_joint_indices=suppress_joint_indices,
                torso_index=1,
                units_to_meters=0.001,
                source_name="dl-pose",
                draw_bounding_box=settings.ENABLE_BOUNDING_BOX,
                inference_fps=payload.get("inference_fps"),
            )
            if vis_result.key == ord("q"):
                stop_event.set()
                break
    finally:
        visualizer.close()
        cv2.destroyAllWindows()


def _new_client(client, server):
    print(f"New client connected and was given id {client['id']}")


def _is_start_message(message) -> bool:
    if message is None:
        return False
    if isinstance(message, bytes):
        try:
            message = message.decode("utf-8")
        except Exception:
            return False
    return str(message).strip().lower() == settings.START_MESSAGE


def _message_received(client, server, message):
    if not _is_start_message(message):
        return
    _set_active_client(client)
    _stream_ready.set()


def _client_left(client, server):
    print(f"Client({client['id']}) disconnected")
    active_client_disconnected = _clear_active_client(client)
    if settings.SHUTDOWN_ON_CLIENT_DISCONNECT and active_client_disconnected and _server_stop_event is not None:
        print("Active client disconnected. Stopping server...")
        _server_stop_event.set()


def main():
    stop_event = Event()
    _set_server_stop_event(stop_event)

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

    try:
        cap = _open_camera()
        intrinsic_matrix, distortion_coeffs = _load_calibration()

        model_dir = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "metrabs_eff2l_384px_800k_28ds_pytorch",
        )
        metrabs_inference_model = metrabsInference.metrabs_inference(model_dir)
        model = metrabs_inference_model.load_model()
        joint_names = model.per_skeleton_joint_names[settings.SKELETON]
        joint_edges = model.per_skeleton_joint_edges[settings.SKELETON].cpu().numpy()
        suppress_joint_indices = _get_hand_joint_indices(joint_names)
        wrist_joint_indices = _get_wrist_joint_indices(joint_names)
        if "Left" not in wrist_joint_indices or "Right" not in wrist_joint_indices:
            print(f"[warn] Could not resolve both wrist joints for {settings.SKELETON}: {wrist_joint_indices}")
        hand_landmarker = _open_hand_landmarker()
        hand_worker = _HandDetectionWorker(hand_landmarker)

        server = WebsocketServer(host="127.0.0.1", port=8080)
        server.set_fn_new_client(_new_client)
        server.set_fn_message_received(_message_received)
        server.set_fn_client_left(_client_left)

        server_thread = threading.Thread(target=server.serve_forever, daemon=True)
        server_thread.start()
        print("WebSocket Server running on ws://127.0.0.1:8080")

        send_queue = thread_queue.Queue(maxsize=1)
        send_metrics = make_send_metrics()
        send_metrics_lock = threading.Lock()
        sender_thread = threading.Thread(
            target=_sender_worker,
            args=(server, send_queue, stop_event, send_metrics, send_metrics_lock),
            daemon=True,
        )
        sender_thread.start()

        if settings.ENABLE_VISUALIZATION:
            vis_queue = Queue(maxsize=1)
            vis_process = Process(
                target=_visualizer_worker,
                args=(vis_queue, stop_event, joint_edges, suppress_joint_indices),
                daemon=True,
            )
            vis_process.start()
            print("Starting detection loop. Press 'q' in the visualization window to exit.")
        else:
            print("[perf] headless benchmark mode enabled (visualizer disabled).")
            print("Starting detection loop. Press Ctrl+C to exit.")

        vis_interval = 1.0 / max(settings.VISUALIZER_MAX_FPS, 1e-3)
        next_vis_push = 0.0
        infer_count = 0
        infer_time_sum = 0.0
        perf_t0 = time.perf_counter()
        latest_infer_fps = None
        pose_out = np.zeros((settings.JOINT_COUNT, settings.PARAMS_PER_JOINT), dtype=np.float32)
        copy_stream = torch.cuda.Stream()
        pending_pose_transfer = None
        frame_id = 0

        while not stop_event.is_set():
            active_client = _get_active_client()
            if settings.REQUIRE_CLIENT_MESSAGE_TO_START and not _stream_ready.is_set():
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
                with torch.inference_mode(), torch.device("cuda"):
                    image_pt = torch.from_numpy(image).permute(2, 0, 1).cuda(non_blocking=True)
                    pred = model.detect_poses(
                        image_pt,
                        intrinsic_matrix=intrinsic_matrix,
                        distortion_coeffs=distortion_coeffs,
                        detector_threshold=settings.DETECTOR_THRESHOLD,
                        detector_nms_iou_threshold=settings.DETECTOR_NMS_IOU,
                        max_detections=settings.MAX_DETECTIONS,
                        skeleton=settings.SKELETON,
                        num_aug=settings.NUM_AUG,
                        antialias_factor=settings.ANTIALIAS_FACTOR,
                        internal_batch_size=settings.INTERNAL_BATCH_SIZE,
                        average_aug=settings.AVERAGE_AUG,
                        suppress_implausible_poses=settings.SUPPRESS_IMPLAUSIBLE_POSES,
                        detector_flip_aug=settings.DETECTOR_FLIP_AUG,
                    )
                poses3d_gpu = pred.get("poses3d") if pred is not None else None
                if active_client is not None and poses3d_gpu is not None and poses3d_gpu.shape[0] > 0:
                    # OP-2: start async GPU→CPU transfer (sync deferred below)
                    pose3d_first_gpu = poses3d_gpu[0].detach().contiguous()
                    pose3d_first_cpu = torch.empty(
                        pose3d_first_gpu.shape,
                        dtype=pose3d_first_gpu.dtype,
                        device="cpu",
                        pin_memory=True,
                    )
                    infer_done_event = torch.cuda.Event()
                    copy_done_event = torch.cuda.Event()
                    infer_done_event.record(torch.cuda.current_stream())
                    with torch.cuda.stream(copy_stream):
                        copy_stream.wait_event(infer_done_event)
                        pose3d_first_cpu.copy_(pose3d_first_gpu, non_blocking=True)
                        copy_done_event.record(copy_stream)
                    current_pose_transfer = {
                        "cpu_tensor": pose3d_first_cpu,
                        "done_event": copy_done_event,
                        "src_tensor": pose3d_first_gpu,
                        "frame_id": current_frame_id,
                        "frame_ready_ts": frame_ready_ts,
                    }
            except Exception as exc:
                print(f"Pose estimation failed: {exc}")
            finally:
                infer_count += 1
                infer_time_sum += time.perf_counter() - infer_start

            now = time.perf_counter()
            perf_elapsed = now - perf_t0
            if perf_elapsed > 0.0 and infer_count > 0:
                latest_infer_fps = infer_count / perf_elapsed
            hand_points_3d_anchored = {}
            if active_client is not None and pending_pose_transfer is not None:
                pending_pose_transfer["done_event"].synchronize()
                pose3d_prev_cpu = pending_pose_transfer["cpu_tensor"].numpy()
                pending_frame_id = pending_pose_transfer["frame_id"]
                _, pending_hand_points_3d_world = hand_worker.get_result(pending_frame_id, remove=True)
                if pending_hand_points_3d_world:
                    hand_points_3d_anchored = _anchor_hand_points_to_metrabs_wrist(
                        pose3d_prev_cpu,
                        pending_hand_points_3d_world,
                        wrist_joint_indices,
                    )
                _map_pose_12(pose3d_prev_cpu, hand_points_3d_anchored, pose_out)
                encoded = _encode_pose_base64(pose_out)
                _put_latest(send_queue, (active_client, encoded, pending_pose_transfer["frame_ready_ts"]))
            pending_pose_transfer = current_pose_transfer

            if settings.ENABLE_VISUALIZATION and now >= next_vis_push:
                hand_points_2d, _ = hand_worker.get_result(current_frame_id)
                poses2d_vis = _to_cpu_numpy(pred.get("poses2d")) if pred is not None else None
                poses3d_vis = _to_cpu_numpy(pred.get("poses3d")) if pred is not None else None
                _put_latest(
                    vis_queue,
                    {
                        "frame_bgr": image.copy(),
                        "poses2d": poses2d_vis,
                        "poses3d": poses3d_vis,
                        "extra_points_2d": hand_points_2d or {},
                        "extra_points_3d": hand_points_3d_anchored or {},
                        "inference_fps": latest_infer_fps,
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
        stop_event.set()
        _set_server_stop_event(None)
        if send_queue is not None:
            _put_latest(send_queue, None)
        if sender_thread is not None:
            sender_thread.join(timeout=1.0)
        if vis_queue is not None:
            _put_latest(vis_queue, None)
        if vis_process is not None:
            vis_process.join(timeout=2.0)
            if vis_process.is_alive():
                vis_process.terminate()
        if cap is not None:
            cap.release()
        if hand_worker is not None:
            hand_worker.stop()
        if hand_landmarker is not None:
            hand_landmarker.close()
        if server is not None:
            try:
                server.shutdown()
            except Exception:
                pass
            try:
                server.server_close()
            except Exception:
                pass
        if server_thread is not None:
            server_thread.join(timeout=1.0)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
