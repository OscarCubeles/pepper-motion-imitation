"""
WebSocket client management module for handling client connections and messages.
Extracted from Pose_3D_metrabs_server_mediapipe_hand.py
"""

import threading
import os
import sys
import settings
import base64
import numpy as np
import cv2

from queue import Empty, Full
from websocket_server import WebsocketServer
from multiprocessing import Queue
from perf_metrics import record_send_metrics

# Add parent directory to path for importing perf_metrics
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _put_latest(queue_obj: Queue, payload):
    """Put payload into queue, dropping old items if queue is full.
    
    Implements a non-blocking put strategy: if queue is full, drop oldest item
    and try again. Useful for real-time data streams where latest is more
    important than complete history.
    
    Args:
        queue_obj: Queue to put payload into
        payload: Data to queue
    """
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


def _encode_pose_base64(pose_out: np.ndarray) -> str:
    """Encode pose data to base64 for WebSocket transmission.
    
    Converts a 3D pose array to little-endian float32 bytes and encodes
    as base64 ASCII string for efficient WebSocket transmission.
    
    Args:
        pose_out: Pose array with shape [num_joints, 3]
        
    Returns:
        str: Base64-encoded pose data
    """
    payload_bytes = np.asarray(pose_out, dtype="<f4").tobytes(order="C")
    return base64.b64encode(payload_bytes).decode("ascii")


# Global state for active client management
_active_client = None
_active_client_lock = threading.Lock()
_stream_ready = threading.Event()
_server_stop_event = None


def _set_active_client(client):
    """Set the current active client."""
    global _active_client
    with _active_client_lock:
        _active_client = client


def _get_active_client():
    """Get the current active client."""
    with _active_client_lock:
        return _active_client


def _clear_active_client(client):
    """Clear the active client if it matches the provided client ID."""
    global _active_client
    with _active_client_lock:
        if _active_client is not None and _active_client.get("id") == client.get("id"):
            _active_client = None
            return True
    return False


def _set_server_stop_event(stop_event):
    """Set the server stop event."""
    global _server_stop_event
    _server_stop_event = stop_event


def _sender_worker(
    server: WebsocketServer,
    send_queue,
    stop_event,
    send_metrics: dict,
    send_metrics_lock: threading.Lock,
):
    """Worker thread that sends pose data to connected clients."""
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


def _new_client(client, server):
    """Callback when a new client connects to the WebSocket server."""
    print(f"New client connected and was given id {client['id']}")


def _is_start_message(message) -> bool:
    """Check if a message is the START message defined in settings."""
    if message is None:
        return False
    if isinstance(message, bytes):
        try:
            message = message.decode("utf-8")
        except Exception:
            return False
    return str(message).strip().lower() == settings.START_MESSAGE


def _message_received(client, server, message):
    """Callback when a message is received from a client."""
    if not _is_start_message(message):
        return
    _set_active_client(client)
    _stream_ready.set()


def _client_left(client, server):
    """Callback when a client disconnects from the WebSocket server."""
    print(f"Client({client['id']}) disconnected")
    active_client_disconnected = _clear_active_client(client)
    if settings.SHUTDOWN_ON_CLIENT_DISCONNECT and active_client_disconnected and _server_stop_event is not None:
        print("Active client disconnected. Stopping server...")
        _server_stop_event.set()


def _create_server(host: str = "127.0.0.1", port: int = 8080) -> tuple[WebsocketServer, threading.Thread]:
    """Create and start a WebSocket server with callback handlers.
    
    Initializes a WebSocket server with client connection callbacks and
    spawns a daemon thread to run the server.
    
    Args:
        host: Server host address (default: "127.0.0.1")
        port: Server port (default: 8080)
        
    Returns:
        tuple: (WebsocketServer instance, server_thread)
    """
    server = WebsocketServer(host=host, port=port)
    server.set_fn_new_client(_new_client)
    server.set_fn_message_received(_message_received)
    server.set_fn_client_left(_client_left)
    
    server_thread = threading.Thread(target=server.serve_forever, daemon=True)
    server_thread.start()
    print(f"WebSocket Server running on ws://{host}:{port}")
    
    return server, server_thread


def shutdown_server(
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
):
    """Clean up and shutdown all resources.
    
    Performs graceful shutdown of all server infrastructure, worker threads,
    hardware, and background processes.
    
    Args:
        stop_event: Multiprocessing event to signal shutdown
        send_queue: Queue for sender worker thread
        sender_thread: Sender worker thread
        vis_queue: Queue for visualization process
        vis_process: Visualization process
        cap: OpenCV camera capture object
        hand_worker: Hand detection worker
        hand_landmarker: MediaPipe hand landmarker
        server: WebSocket server
        server_thread: Server thread
    """
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
