from __future__ import print_function
import os
import sys
import time
import base64
import json
import struct
import keyboard
import websocket
import numpy as np
from threading import Event, Thread, Lock

if __package__:
    from . import pepper_config as settings
    from . import client_visualization as client_vis
else:
    import pepper_config as settings
    import client_visualization as client_vis

if __package__:
    from .server_angle_payload import build_server_joint_targets
else:
    from server_angle_payload import build_server_joint_targets

try:
    from Queue import Empty, Full, Queue
except ImportError:
    from queue import Empty, Full, Queue

settings.set_pose_stream_runtime_dir()
import scaling_spherical as scale
import pepper_commands as commands


# TODO: Put this in the pepper_config at some point
PORT = 8080
SERVER_HOST = "localhost"
MIRRORING_IMITATION = False
JOINT_COUNT_BASE = 10
JOINT_COUNT_WITH_HAND_TIPS = 12
JOINT_COUNT_WITH_HAND_ORIENTATION = 20
JOINT_COUNT_WITH_HAND_PALM_VECTORS = 26
PARAMS_PER_JOINT = 3
FLOAT_BYTES = 4
JOINT_BYTES = PARAMS_PER_JOINT * FLOAT_BYTES
EXPECTED_BYTES_BASE = JOINT_COUNT_BASE * JOINT_BYTES
EXPECTED_BYTES_WITH_HAND_TIPS = JOINT_COUNT_WITH_HAND_TIPS * JOINT_BYTES
EXPECTED_BYTES_WITH_HAND_ORIENTATION = JOINT_COUNT_WITH_HAND_ORIENTATION * JOINT_BYTES
EXPECTED_BYTES_WITH_HAND_PALM_VECTORS = JOINT_COUNT_WITH_HAND_PALM_VECTORS * JOINT_BYTES
REQUEST_MESSAGE = "keypoints"
STOP_KEY = "q"
RUN_DURATION_SEC = None
RUN_DURATION_FROM_FIRST_PAYLOAD = False
FIRST_PAYLOAD_TIMEOUT_SEC = None
ENABLE_ROBOT_IMITATION = True
MANAGE_PEPPER_SECURITY = True
SEND_TO_STAND_ON_EXIT = True
MODE_LABEL = "Imitation"
BACKEND_PROFILE = settings.POSE_BACKEND_PROFILE
COMMAND_RATE_HZ = settings.get_command_rate_hz_for_backend(BACKEND_PROFILE)
COMMAND_RATE_HZ_FALLBACK = 15.0
COMMAND_MIN_HZ = 1.0
PERF_LOG_INTERVAL_SEC = 2.0
LATENCY_TRAILER_MAGIC = b"LAT1"
LATENCY_TRAILER_STRUCT = struct.Struct("<4sIQ")
LATENCY_TRAILER_SIZE = LATENCY_TRAILER_STRUCT.size
LATENCY_SYNC_REQ_PREFIX = "latency_sync_req:"
LATENCY_SYNC_RSP_PREFIX = "latency_sync_rsp:"
LATENCY_SYNC_PROBE_COUNT = 8
LATENCY_SYNC_PROBE_INTERVAL_SEC = 0.05
LATENCY_SYNC_WAIT_TIMEOUT_SEC = 0.5

# Reused buffer to avoid repeated allocations
floatArrayBase = np.zeros((JOINT_COUNT_BASE, PARAMS_PER_JOINT), dtype=np.float32)
floatArrayWithHandTips = np.zeros((JOINT_COUNT_WITH_HAND_TIPS, PARAMS_PER_JOINT), dtype=np.float32)
floatArrayWithHandOrientation = np.zeros((JOINT_COUNT_WITH_HAND_ORIENTATION, PARAMS_PER_JOINT), dtype=np.float32)
floatArrayWithHandPalmVectors = np.zeros((JOINT_COUNT_WITH_HAND_PALM_VECTORS, PARAMS_PER_JOINT), dtype=np.float32)
stop_event = Event()
_hand_tracking_enabled = None
_joint_observer = None
_cleanup_done = False
_stop_reason = None
_first_payload_ts = None
_latest_joint_targets = None
_latest_joint_targets_lock = Lock()
_latency_sync_lock = Lock()
_latency_sync_pending = {}
_latency_sync_samples = []
_latency_clock_offset_us = None
_latest_hand_orientation = None
_hand_orientation_lock = Lock()

rx_queue = Queue(maxsize=1)
rx_thread = None
command_thread = None
clock_sync_thread = None


def _reset_runtime_state():
    global stop_event
    global _hand_tracking_enabled, _joint_observer, _cleanup_done, _stop_reason, _first_payload_ts
    global _latest_joint_targets, _latency_sync_pending, _latency_sync_samples, _latency_clock_offset_us
    global _latest_hand_orientation
    global rx_queue, rx_thread, command_thread, clock_sync_thread
    stop_event = Event()
    _hand_tracking_enabled = None
    _joint_observer = None
    _cleanup_done = False
    _stop_reason = None
    _first_payload_ts = None
    _latest_joint_targets = None
    _latest_hand_orientation = None
    rx_queue = Queue(maxsize=1)
    rx_thread = None
    command_thread = None
    clock_sync_thread = None
    with _latency_sync_lock:
        _latency_sync_pending = {}
        _latency_sync_samples = []
        _latency_clock_offset_us = None
    scale.reset_ik_continuity_state()


def _put_latest(queue_obj, payload):
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


def _set_latest_joint_targets(names, angles, speeds, frame_id=None, capture_ts_client_us=None):
    global _latest_joint_targets
    with _latest_joint_targets_lock:
        _latest_joint_targets = (
            list(names) if names is not None else [],
            list(angles) if angles is not None else [],
            list(speeds) if speeds is not None else None,
            None if frame_id is None else int(frame_id),
            None if capture_ts_client_us is None else int(capture_ts_client_us),
        )


def _get_latest_joint_targets():
    with _latest_joint_targets_lock:
        if _latest_joint_targets is None:
            return None
        names, angles, speeds, frame_id, capture_ts_client_us = _latest_joint_targets
        return (
            list(names),
            list(angles),
            list(speeds) if speeds is not None else None,
            frame_id,
            capture_ts_client_us,
        )


def _parse_latency_sync_response(message):
    if message is None:
        return None
    if isinstance(message, bytes):
        try:
            message = message.decode("utf-8")
        except Exception:
            return None
    text = str(message).strip()
    if not text.startswith(LATENCY_SYNC_RSP_PREFIX):
        return None
    parts = text.split(":")
    if len(parts) != 4:
        return None
    try:
        client_send_us = int(parts[1])
        server_recv_us = int(parts[2])
        server_send_us = int(parts[3])
    except Exception:
        return None
    return client_send_us, server_recv_us, server_send_us


def _record_latency_sync_response(client_send_us, server_recv_us, server_send_us):
    client_recv_us = int(time.time() * 1000000.0)
    with _latency_sync_lock:
        if client_send_us not in _latency_sync_pending:
            return False
        request_send_us = int(_latency_sync_pending.pop(client_send_us))
        rtt_us = (
            float(client_recv_us - request_send_us) -
            float(server_send_us - server_recv_us)
        )
        if rtt_us < 0.0:
            rtt_us = 0.0
        offset_us = (
            (float(server_recv_us) - float(request_send_us)) +
            (float(server_send_us) - float(client_recv_us))
        ) / 2.0
        _latency_sync_samples.append((rtt_us, offset_us))
    return True


def _get_latency_clock_offset_us():
    with _latency_sync_lock:
        return _latency_clock_offset_us


def _set_best_latency_clock_offset():
    global _latency_clock_offset_us
    with _latency_sync_lock:
        if not _latency_sync_samples:
            _latency_clock_offset_us = None
            return None, 0
        best_rtt_us, best_offset_us = min(_latency_sync_samples, key=lambda sample: sample[0])
        _latency_clock_offset_us = int(round(best_offset_us))
        sample_count = len(_latency_sync_samples)
    return float(best_rtt_us), sample_count


def _run_latency_clock_sync(ws):
    with _latency_sync_lock:
        _latency_sync_pending.clear()
        _latency_sync_samples[:] = []

    for _ in range(LATENCY_SYNC_PROBE_COUNT):
        if stop_event.is_set():
            break
        client_send_us = int(time.time() * 1000000.0)
        with _latency_sync_lock:
            _latency_sync_pending[client_send_us] = client_send_us
        try:
            ws.send("%s%d" % (LATENCY_SYNC_REQ_PREFIX, client_send_us))
        except Exception as exc:
            if not _is_expected_disconnect_error(exc):
                print("latency sync send failed: " + str(exc))
            break
        time.sleep(LATENCY_SYNC_PROBE_INTERVAL_SEC)

    wait_deadline = time.time() + LATENCY_SYNC_WAIT_TIMEOUT_SEC
    while (time.time() < wait_deadline) and (not stop_event.is_set()):
        with _latency_sync_lock:
            pending_count = len(_latency_sync_pending)
        if pending_count == 0:
            break
        time.sleep(0.01)

    with _latency_sync_lock:
        _latency_sync_pending.clear()
    best_rtt_us, sample_count = _set_best_latency_clock_offset()
    offset_us = _get_latency_clock_offset_us()
    if offset_us is None:
        print("[perf] latency_sync unavailable (total_latency_ms will be n/a).")
        return
    print(
        "[perf] latency_sync samples=%d best_rtt_ms=%.2f clock_offset_us=%d" % (
            sample_count,
            (best_rtt_us / 1000.0) if best_rtt_us is not None else 0.0,
            offset_us,
        )
    )


def _split_pose_payload(decoded_bytes):
    frame_meta = None
    payload_bytes = decoded_bytes
    if decoded_bytes is None:
        return payload_bytes, frame_meta
    if len(decoded_bytes) >= LATENCY_TRAILER_SIZE:
        trailer = decoded_bytes[-LATENCY_TRAILER_SIZE:]
        try:
            magic, frame_id, capture_ts_us = LATENCY_TRAILER_STRUCT.unpack(trailer)
        except Exception:
            magic = None
        if magic == LATENCY_TRAILER_MAGIC:
            payload_bytes = decoded_bytes[:-LATENCY_TRAILER_SIZE]
            frame_meta = {
                "frame_id": int(frame_id),
                "capture_ts_us": int(capture_ts_us),
            }
    return payload_bytes, frame_meta


def _resolve_command_rate_hz():
    try:
        rate_hz = float(COMMAND_RATE_HZ)
    except Exception:
        rate_hz = COMMAND_RATE_HZ_FALLBACK
    if rate_hz < COMMAND_MIN_HZ:
        rate_hz = COMMAND_MIN_HZ
    return rate_hz


def _command_dispatch_loop():
    tick_sec = 1.0 / _resolve_command_rate_hz()
    next_tick = time.time()
    perf_t0 = time.time()
    while not stop_event.is_set():
        targets = _get_latest_joint_targets()
        if targets is not None:
            names, angles, speeds, frame_id, capture_ts_client_us = targets
            if angles:
                commands.send_joint_targets_speed(
                    names,
                    angles,
                    speeds=speeds,
                    frame_id=frame_id,
                    capture_ts_client_us=capture_ts_client_us,
                )

        now_perf = time.time()
        perf_elapsed = now_perf - perf_t0
        if perf_elapsed >= PERF_LOG_INTERVAL_SEC:
            command_count, avg_total_latency_ms = commands.consume_total_latency_metrics()
            command_fps = command_count / perf_elapsed if perf_elapsed > 0 else 0.0
            latency_text = "%.1f" % avg_total_latency_ms if avg_total_latency_ms is not None else "n/a"
            #print(
            #    "[perf] command_fps=%.1f total_latency_ms=%s" % (
            #        command_fps,
            #        latency_text,
            #    )
            #)
            perf_t0 = now_perf

        next_tick += tick_sec
        sleep_sec = next_tick - time.time()
        if sleep_sec <= 0.0:
            # Keep fixed-rate behavior without queueing old ticks.
            missed_ticks = int(abs(sleep_sec) / tick_sec) + 1
            next_tick += (missed_ticks * tick_sec)
            continue
        time.sleep(sleep_sec)


def _get_latest_hand_orientation():
    """Get the latest hand orientation labels."""
    with _hand_orientation_lock:
        return _latest_hand_orientation


def _set_latest_hand_orientation(labels):
    """Set the latest hand orientation labels."""
    global _latest_hand_orientation
    with _hand_orientation_lock:
        _latest_hand_orientation = labels


def _decode_dispatch_loop():
    while not stop_event.is_set():
        try:
            message = rx_queue.get(timeout=0.1)
        except Empty:
            continue
        if message is None:
            break
        try:
            # JSON payloads may also contain server-computed Pepper angles.
            payload_bytes = None
            frame_meta = None
            hand_orientation = None
            angles_data = None
            
            try:
                json_data = json.loads(message)
                if isinstance(json_data, dict) and "pose_keypoints" in json_data:
                    # New JSON format with hand orientation labels
                    pose_base64 = json_data.get("pose_keypoints")
                    hand_orientation = json_data.get("hand_orientation", {})
                    angles_data = json_data.get("angles")
                    decoded = base64.b64decode(pose_base64)
                    payload_bytes, frame_meta = _split_pose_payload(decoded)
                    _set_latest_hand_orientation(hand_orientation)
            except (json.JSONDecodeError, ValueError, TypeError):
                # Fall back to old base64 format for backward compatibility
                decoded = base64.b64decode(message)
                payload_bytes, frame_meta = _split_pose_payload(decoded)
            
            jointArray = storeDataInArray(payload_bytes)
            if jointArray is None:
                continue
            frame_id = None
            capture_ts_client_us = None
            if frame_meta is not None:
                frame_id = frame_meta.get("frame_id")
                capture_ts_server_us = frame_meta.get("capture_ts_us")
                offset_us = _get_latency_clock_offset_us()
                if (capture_ts_server_us is not None) and (offset_us is not None):
                    capture_ts_client_us = int(capture_ts_server_us - offset_us)
            recieve_joints(
                jointArray,
                frame_id=frame_id,
                capture_ts_client_us=capture_ts_client_us,
                hand_orientation=hand_orientation,
                angles=angles_data,
            )
        except Exception as exc:
            if not stop_event.is_set():
                print("decode/dispatch error: " + str(exc))


def _start_workers():
    global rx_thread, command_thread
    commands.reset_joint_command_shaper()
    commands.reset_total_latency_metrics()
    rx_thread = Thread(target=_decode_dispatch_loop)
    rx_thread.daemon = True
    rx_thread.start()
    if ENABLE_ROBOT_IMITATION:
        command_thread = Thread(target=_command_dispatch_loop)
        command_thread.daemon = True
        command_thread.start()


def _stop_workers():
    global clock_sync_thread
    stop_event.set()
    _put_latest(rx_queue, None)
    if rx_thread is not None:
        try:
            rx_thread.join(0.5)
        except Exception:
            pass
    if command_thread is not None:
        try:
            command_thread.join(0.5)
        except Exception:
            pass
    if clock_sync_thread is not None:
        try:
            clock_sync_thread.join(0.5)
        except Exception:
            pass


def _set_hand_tracking(enabled):
    global _hand_tracking_enabled
    enabled = bool(enabled)
    if _hand_tracking_enabled == enabled:
        return
    scale.set_hand_tracking_enabled(enabled)
    _hand_tracking_enabled = enabled


def _set_stop_reason(reason):
    global _stop_reason
    if _stop_reason is None:
        _stop_reason = reason


def _emit_joint_observer(payload):
    if _joint_observer is None:
        return
    try:
        _joint_observer(payload)
    except Exception as exc:
        if not stop_event.is_set():
            print("joint observer error: " + str(exc))


def _cleanup_session():
    global _cleanup_done
    if _cleanup_done:
        return
    _cleanup_done = True
    _stop_workers()
    if MANAGE_PEPPER_SECURITY:
        try:
            commands.enable_security_setting()
        except Exception as exc:
            print("cleanup warning (enable security): " + str(exc))
    if SEND_TO_STAND_ON_EXIT:
        try:
            commands.send_to_stand()
        except Exception as exc:
            print("cleanup warning (send to stand): " + str(exc))


def _is_expected_disconnect_error(error):
    if stop_event.is_set():
        return True
    if error is None:
        return False
    message = str(error)
    expected_tokens = (
        "10054",
        "10053",
        "10057",
        "forcibly closed by the remote host",
        "connection reset by peer",
        "connection is already closed",
        "socket is already closed",
    )
    message_lower = message.lower()
    for token in expected_tokens:
        if token in message_lower:
            return True
    return False


# Store and receive joint 3D positions from the server.
def recieve_joints(jointMatrix, frame_num=0, frame_id=None, capture_ts_client_us=None, hand_orientation=None, angles=None):
    # check if matrix is empty
    if not np.all(jointMatrix == 0.0):
        # Copy once because the decode buffers are reused across frames.
        jointMatrix = jointMatrix.copy()

        # Proposed and IKPy servers provide angles. Baseline payloads omit them
        # and continue through the existing pose-based IK path below.
        used_server_angles = False
        if ENABLE_ROBOT_IMITATION:
            names, server_angles, speeds = build_server_joint_targets(angles)
            if server_angles:
                _set_latest_joint_targets(
                    names,
                    server_angles,
                    speeds,
                    frame_id=frame_id,
                    capture_ts_client_us=capture_ts_client_us,
                )
                used_server_angles = True
        
        # Show hand orientation if received
        #if hand_orientation:
        #    print("[JOINTS] Hand Orientation available: {}".format(hand_orientation))

        # Head
        Nose = jointMatrix[0]
        Neck = jointMatrix[1]
        # Right arm
        RShoulder = jointMatrix[2]
        RElbow = jointMatrix[3]
        RWrist = jointMatrix[4]

        # Left arm
        LShoulder = jointMatrix[5]
        LElbow = jointMatrix[6]
        LWrist = jointMatrix[7]

        # Torso
        Torso = jointMatrix[8]
        SpineBase = jointMatrix[9]
        # MediaPipe hand tracking now streams only fingertip positions.
        if jointMatrix.shape[0] >= JOINT_COUNT_WITH_HAND_TIPS:
            _set_hand_tracking(True)
            RTip = jointMatrix[10]
            LTip = jointMatrix[11]
        else:
            _set_hand_tracking(False)
            RTip = None
            LTip = None

        RHandWrist = None
        RThumbCMC = None
        RIndexMCP = None
        RPinkyMCP = None
        LHandWrist = None
        LThumbCMC = None
        LIndexMCP = None
        LPinkyMCP = None
        RServerPalmX = None
        RServerPalmY = None
        RServerPalmZ = None
        LServerPalmX = None
        LServerPalmY = None
        LServerPalmZ = None
        if jointMatrix.shape[0] >= JOINT_COUNT_WITH_HAND_ORIENTATION:
            
            RHandWrist = jointMatrix[12]
            RThumbCMC = jointMatrix[13]
            RIndexMCP = jointMatrix[14]
            RPinkyMCP = jointMatrix[15]
            LHandWrist = jointMatrix[16]
            LThumbCMC = jointMatrix[17]
            LIndexMCP = jointMatrix[18]
            LPinkyMCP = jointMatrix[19]
        
        # Extract server-computed palm vectors if available (indices 20-25)
        if jointMatrix.shape[0] >= JOINT_COUNT_WITH_HAND_PALM_VECTORS:
            RServerPalmX = jointMatrix[20]
            RServerPalmY = jointMatrix[21]
            RServerPalmZ = jointMatrix[22]
            LServerPalmX = jointMatrix[23]
            LServerPalmY = jointMatrix[24]
            LServerPalmZ = jointMatrix[25]
            
            # Print all keypoint values for understanding the coordinate system
            #print(u"\n" + u"="*100)
            #print(u"ALL KEYPOINT VALUES (CLIENT RECEIVED) - Units in METERS")
            #print(u"="*100)
            #
            #print(u"\n--- BODY KEYPOINTS ---")
            #print(u"Nose (0):           X={0:.4f}, Y={1:.4f}, Z={2:.4f}".format(Nose[0], Nose[1], Nose[2]))
            #print(u"Neck (1):           X={0:.4f}, Y={1:.4f}, Z={2:.4f}".format(Neck[0], Neck[1], Neck[2]))
            #print(u"Torso (8):          X={0:.4f}, Y={1:.4f}, Z={2:.4f}".format(Torso[0], Torso[1], Torso[2]))
            #print(u"SpineBase (9):      X={0:.4f}, Y={1:.4f}, Z={2:.4f}".format(SpineBase[0], SpineBase[1], SpineBase[2]))
            #
            #print(u"\n--- RIGHT ARM ---")
            #print(u"R Shoulder (2):     X={0:.4f}, Y={1:.4f}, Z={2:.4f}".format(RShoulder[0], RShoulder[1], RShoulder[2]))
            #print(u"R Elbow (3):        X={0:.4f}, Y={1:.4f}, Z={2:.4f}".format(RElbow[0], RElbow[1], RElbow[2]))
            #print(u"R Wrist (4):        X={0:.4f}, Y={1:.4f}, Z={2:.4f}".format(RWrist[0], RWrist[1], RWrist[2]))
            #if RTip is not None:
            #    print(u"R Tip (10):         X={0:.4f}, Y={1:.4f}, Z={2:.4f}".format(RTip[0], RTip[1], RTip[2]))
            #
            #print(u"\n--- LEFT ARM ---")
            #print(u"L Shoulder (5):     X={0:.4f}, Y={1:.4f}, Z={2:.4f}".format(LShoulder[0], LShoulder[1], LShoulder[2]))
            #print(u"L Elbow (6):        X={0:.4f}, Y={1:.4f}, Z={2:.4f}".format(LElbow[0], LElbow[1], LElbow[2]))
            #print(u"L Wrist (7):        X={0:.4f}, Y={1:.4f}, Z={2:.4f}".format(LWrist[0], LWrist[1], LWrist[2]))
            #if LTip is not None:
            #    print(u"L Tip (11):         X={0:.4f}, Y={1:.4f}, Z={2:.4f}".format(LTip[0], LTip[1], LTip[2]))
            #
            #print(u"\n--- RIGHT HAND ORIENTATION KEYPOINTS ---")
            #print(u"R Hand Wrist (12):  X={0:.4f}, Y={1:.4f}, Z={2:.4f}".format(RHandWrist[0], RHandWrist[1], RHandWrist[2]))
            #print(u"R Thumb CMC (13):   X={0:.4f}, Y={1:.4f}, Z={2:.4f}".format(RThumbCMC[0], RThumbCMC[1], RThumbCMC[2]))
            #print(u"R Index MCP (14):   X={0:.4f}, Y={1:.4f}, Z={2:.4f}".format(RIndexMCP[0], RIndexMCP[1], RIndexMCP[2]))
            #print(u"R Pinky MCP (15):   X={0:.4f}, Y={1:.4f}, Z={2:.4f}".format(RPinkyMCP[0], RPinkyMCP[1], RPinkyMCP[2]))
            #
            #print(u"\n--- LEFT HAND ORIENTATION KEYPOINTS ---")
            #print(u"L Hand Wrist (16):  X={0:.4f}, Y={1:.4f}, Z={2:.4f}".format(LHandWrist[0], LHandWrist[1], LHandWrist[2]))
            #print(u"L Thumb CMC (17):   X={0:.4f}, Y={1:.4f}, Z={2:.4f}".format(LThumbCMC[0], LThumbCMC[1], LThumbCMC[2]))
            #print(u"L Index MCP (18):   X={0:.4f}, Y={1:.4f}, Z={2:.4f}".format(LIndexMCP[0], LIndexMCP[1], LIndexMCP[2]))
            #print(u"L Pinky MCP (19):   X={0:.4f}, Y={1:.4f}, Z={2:.4f}".format(LPinkyMCP[0], LPinkyMCP[1], LPinkyMCP[2]))
            #print(u"="*100 + u"\n")

        _emit_joint_observer({
            "frame_num": frame_num,
            "frame_id": frame_id,
            "capture_ts_client_us": capture_ts_client_us,
            "hand_orientation": hand_orientation,
            "nose": Nose,
            "neck": Neck,
            "torso": Torso,
            "spine_base": SpineBase,
            "left_shoulder": LShoulder,
            "left_elbow": LElbow,
            "left_wrist": LWrist,
            "right_shoulder": RShoulder,
            "right_elbow": RElbow,
            "right_wrist": RWrist,
            "right_tip": RTip,
            "left_tip": LTip,
            "right_hand_wrist": RHandWrist,
            "right_thumb_cmc": RThumbCMC,
            "right_index_mcp": RIndexMCP,
            "right_pinky_mcp": RPinkyMCP,
            "left_hand_wrist": LHandWrist,
            "left_thumb_cmc": LThumbCMC,
            "left_index_mcp": LIndexMCP,
            "left_pinky_mcp": LPinkyMCP,
            "right_server_palm_x": RServerPalmX,
            "right_server_palm_y": RServerPalmY,
            "right_server_palm_z": RServerPalmZ,
            "left_server_palm_x": LServerPalmX,
            "left_server_palm_y": LServerPalmY,
            "left_server_palm_z": LServerPalmZ,
        })
        if ENABLE_ROBOT_IMITATION and not used_server_angles:
            # Get discrete orientation labels if available
            hand_orientation_data = _get_latest_hand_orientation()
            right_discrete_orientation = None
            left_discrete_orientation = None
            if hand_orientation_data is not None:
                # Server sends 'Right' and 'Left' keys with nested dicts containing 'primary' label
                if 'Right' in hand_orientation_data:
                    right_discrete_orientation = hand_orientation_data['Right'].get('primary')
                if 'Left' in hand_orientation_data:
                    left_discrete_orientation = hand_orientation_data['Left'].get('primary')
            
            names, angles, speeds = scale.build_frame_joint_targets_hand(
                Torso,
                Neck,
                Nose,
                RShoulder,
                RElbow,
                RWrist,
                RTip,
                LShoulder,
                LElbow,
                LWrist,
                LTip,
                SpineBase,
                RHandWrist,
                RThumbCMC,
                RIndexMCP,
                RPinkyMCP,
                LHandWrist,
                LThumbCMC,
                LIndexMCP,
                LPinkyMCP,
                RServerPalmX,
                RServerPalmY,
                RServerPalmZ,
                LServerPalmX,
                LServerPalmY,
                LServerPalmZ,
                right_discrete_orientation,
                left_discrete_orientation,

            )
            if angles:
                _set_latest_joint_targets(
                    names,
                    angles,
                    speeds,
                    frame_id=frame_id,
                    capture_ts_client_us=capture_ts_client_us,
                )


# Function to convert the array of bytes in array of floats
def storeDataInArray(byte_arr):
    global floatArrayBase, floatArrayWithHandTips, floatArrayWithHandOrientation, floatArrayWithHandPalmVectors
    if len(byte_arr) < EXPECTED_BYTES_BASE:
        return None

    if len(byte_arr) >= EXPECTED_BYTES_WITH_HAND_PALM_VECTORS:
        values = np.frombuffer(
            byte_arr,
            dtype=np.float32,
            count=JOINT_COUNT_WITH_HAND_PALM_VECTORS * PARAMS_PER_JOINT,
        )
        floatArrayWithHandPalmVectors[:] = values.reshape((JOINT_COUNT_WITH_HAND_PALM_VECTORS, PARAMS_PER_JOINT))
        return floatArrayWithHandPalmVectors

    if len(byte_arr) >= EXPECTED_BYTES_WITH_HAND_ORIENTATION:
        values = np.frombuffer(
            byte_arr,
            dtype=np.float32,
            count=JOINT_COUNT_WITH_HAND_ORIENTATION * PARAMS_PER_JOINT,
        )
        floatArrayWithHandOrientation[:] = values.reshape((JOINT_COUNT_WITH_HAND_ORIENTATION, PARAMS_PER_JOINT))
        return floatArrayWithHandOrientation

    if len(byte_arr) >= EXPECTED_BYTES_WITH_HAND_TIPS:
        values = np.frombuffer(
            byte_arr,
            dtype=np.float32,
            count=JOINT_COUNT_WITH_HAND_TIPS * PARAMS_PER_JOINT,
        )
        floatArrayWithHandTips[:] = values.reshape((JOINT_COUNT_WITH_HAND_TIPS, PARAMS_PER_JOINT))
        return floatArrayWithHandTips

    values = np.frombuffer(
        byte_arr,
        dtype=np.float32,
        count=JOINT_COUNT_BASE * PARAMS_PER_JOINT,
    )
    floatArrayBase[:] = values.reshape((JOINT_COUNT_BASE, PARAMS_PER_JOINT))
    return floatArrayBase

def on_message(ws, message):
    global _first_payload_ts
    parsed_sync = _parse_latency_sync_response(message)
    if parsed_sync is not None:
        client_send_us, server_recv_us, server_send_us = parsed_sync
        _record_latency_sync_response(client_send_us, server_recv_us, server_send_us)
        return
    if _first_payload_ts is None:
        _first_payload_ts = time.time()
    _put_latest(rx_queue, message)


def on_error(ws, error):
    if _is_expected_disconnect_error(error):
        return
    _set_stop_reason("websocket_error")
    print("### error ###")
    print(error)

def on_close(ws, *args):
    _set_stop_reason("connection_closed")
    _cleanup_session()
    print("### closed ###")

def on_open(ws):
    global clock_sync_thread

    def run_clock_sync_and_start():
        _run_latency_clock_sync(ws)
        if stop_event.is_set():
            return
        try:
            ws.send(REQUEST_MESSAGE)
        except Exception as e:
            if not _is_expected_disconnect_error(e):
                print(e)
                print("failed to send initial keypoints request")
            return

    clock_sync_thread = Thread(target=run_clock_sync_and_start)
    clock_sync_thread.daemon = True
    clock_sync_thread.start()

    def run(*args):
        start_ts = time.time()
        while not stop_event.is_set():
            if keyboard.is_pressed(STOP_KEY):
                _set_stop_reason("keyboard_stop")
                stop_event.set()
                ws.close()
                print("closing connection")
                break
            if RUN_DURATION_SEC is not None:
                now = time.time()
                if RUN_DURATION_FROM_FIRST_PAYLOAD:
                    if _first_payload_ts is not None:
                        if (now - _first_payload_ts) >= RUN_DURATION_SEC:
                            _set_stop_reason("duration_elapsed")
                            stop_event.set()
                            ws.close()
                            print("capture duration reached (from first payload), closing connection")
                            break
                    elif FIRST_PAYLOAD_TIMEOUT_SEC is not None and (now - start_ts) >= FIRST_PAYLOAD_TIMEOUT_SEC:
                        _set_stop_reason("first_payload_timeout")
                        stop_event.set()
                        ws.close()
                        print("no payload received before startup timeout, closing connection")
                        break
                elif (now - start_ts) >= RUN_DURATION_SEC:
                    _set_stop_reason("duration_elapsed")
                    stop_event.set()
                    ws.close()
                    print("capture duration reached, closing connection")
                    break
            time.sleep(0.02)
    control_thread = Thread(target=run)
    control_thread.daemon = True
    control_thread.start()


def print_usage_mode():
    if ENABLE_ROBOT_IMITATION:
        print(
            "%s mode: " % MODE_LABEL +
            ("mirroring" if MIRRORING_IMITATION else "non-mirroring (anatomical)")
        )
        print(
            "Robot command loop: %.1f FPS fixed-rate (%s backend profile). "
            "Decode loop runs as fast as possible; command loop uses latest buffered pose." % (
                _resolve_command_rate_hz(),
                BACKEND_PROFILE,
            )
        )
    else:
        print("%s mode: logging-only (Pepper imitation disabled)" % MODE_LABEL)
    if RUN_DURATION_SEC is None:
        print("Press %s to stop the session!" % STOP_KEY)
    else:
        if RUN_DURATION_FROM_FIRST_PAYLOAD:
            if FIRST_PAYLOAD_TIMEOUT_SEC is None:
                print("Press %s to stop early (auto-stop %.1f s after first payload)." % (STOP_KEY, RUN_DURATION_SEC))
            else:
                print(
                    "Press %s to stop early (auto-stop %.1f s after first payload, startup timeout %.1f s)." % (
                        STOP_KEY,
                        RUN_DURATION_SEC,
                        FIRST_PAYLOAD_TIMEOUT_SEC,
                    )
                )
        else:
            print("Press %s to stop early (auto-stop after %.1f s)." % (STOP_KEY, RUN_DURATION_SEC))


def run_pose_stream_client(
    server_host="localhost",
    port=8080,
    mirroring_imitation=False,
    request_message="keypoints",
    stop_key="q",
    run_duration_sec=None,
    run_duration_from_first_payload=False,
    first_payload_timeout_sec=None,
    joint_observer=None,
    enable_robot_imitation=True,
    manage_pepper_security=True,
    send_to_stand_on_exit=True,
    mode_label="Imitation",
    backend_profile=settings.POSE_BACKEND_PROFILE,
    command_rate_hz=None,
):
    """Run the imitation streaming client.

    `joint_observer` receives a dict with torso/arm/head keypoints for each decoded frame.
    """
    global PORT, SERVER_HOST, MIRRORING_IMITATION, REQUEST_MESSAGE, STOP_KEY, RUN_DURATION_SEC
    global RUN_DURATION_FROM_FIRST_PAYLOAD, FIRST_PAYLOAD_TIMEOUT_SEC, _joint_observer
    global ENABLE_ROBOT_IMITATION, MANAGE_PEPPER_SECURITY, SEND_TO_STAND_ON_EXIT, MODE_LABEL
    global BACKEND_PROFILE, COMMAND_RATE_HZ

    _reset_runtime_state()
    SERVER_HOST = str(server_host) if server_host else "localhost"
    PORT = int(port)
    MIRRORING_IMITATION = bool(mirroring_imitation)
    REQUEST_MESSAGE = str(request_message)
    STOP_KEY = str(stop_key)
    RUN_DURATION_SEC = None if run_duration_sec is None else float(run_duration_sec)
    RUN_DURATION_FROM_FIRST_PAYLOAD = bool(run_duration_from_first_payload)
    FIRST_PAYLOAD_TIMEOUT_SEC = None if first_payload_timeout_sec is None else float(first_payload_timeout_sec)
    _joint_observer = joint_observer
    ENABLE_ROBOT_IMITATION = bool(enable_robot_imitation)
    MANAGE_PEPPER_SECURITY = bool(manage_pepper_security)
    SEND_TO_STAND_ON_EXIT = bool(send_to_stand_on_exit)
    MODE_LABEL = str(mode_label) if mode_label else "Pose stream"
    BACKEND_PROFILE = str(backend_profile or settings.POSE_BACKEND_PROFILE).strip().lower()
    if command_rate_hz is None:
        COMMAND_RATE_HZ = settings.get_command_rate_hz_for_backend(BACKEND_PROFILE)
    else:
        try:
            COMMAND_RATE_HZ = float(command_rate_hz)
        except Exception:
            COMMAND_RATE_HZ = settings.get_command_rate_hz_for_backend(BACKEND_PROFILE)

    if MANAGE_PEPPER_SECURITY:
        commands.disable_security_settings()
    scale.set_mirroring_imitation_enabled(MIRRORING_IMITATION)
    _set_hand_tracking(False)
    _start_workers()

    print_usage_mode()

    websocket.enableTrace(False)
    ws_url = "ws://{}:{}/PepperCommands".format(SERVER_HOST, PORT)
    print("Pose server endpoint: %s" % ws_url)
    ws = websocket.WebSocketApp(
        ws_url,
        on_message=on_message,
        on_error=on_error,
        on_close=on_close,
    )
    ws.on_open = on_open

    try:
        ws.run_forever()
    finally:
        _cleanup_session()

    return {
        "stop_reason": _stop_reason,
        "server_host": SERVER_HOST,
        "port": PORT,
        "mirroring_imitation": MIRRORING_IMITATION,
        "run_duration_sec": RUN_DURATION_SEC,
        "run_duration_from_first_payload": RUN_DURATION_FROM_FIRST_PAYLOAD,
        "first_payload_timeout_sec": FIRST_PAYLOAD_TIMEOUT_SEC,
        "first_payload_received": (_first_payload_ts is not None),
        "enable_robot_imitation": ENABLE_ROBOT_IMITATION,
        "backend_profile": BACKEND_PROFILE,
        "command_rate_hz": _resolve_command_rate_hz(),
    }


def run_imitation_client(*args, **kwargs):
    # Backward-compatible alias for older callers.
    return run_pose_stream_client(*args, **kwargs)


def main():
    return run_pose_stream_client(
        server_host=SERVER_HOST,
        port=PORT,
        mirroring_imitation=MIRRORING_IMITATION,
        request_message=REQUEST_MESSAGE,
    )


if __name__ == "__main__":
    main()
