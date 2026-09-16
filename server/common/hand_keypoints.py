import os
import threading
import numpy as np
import cv2
import mediapipe as mp
from server.common import settings

from collections import deque
from queue import Empty, Full
from multiprocessing import Queue
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


def _open_hand_landmarker():
    """Initialize and return MediaPipe HandLandmarker for hand detection.
    
    Returns:
        HandLandmarker: Configured MediaPipe hand detection model
    """
    hand_model_path = str(settings.HAND_LANDMARKER_PATH)
    hand_options = vision.HandLandmarkerOptions(
        base_options=python.BaseOptions(model_asset_path=hand_model_path),
        running_mode=vision.RunningMode.VIDEO,
        num_hands=2,
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    return vision.HandLandmarker.create_from_options(hand_options)


def _get_hand_joint_indices(joint_names):
    """Extract hand-related joint indices from skeleton.
    
    Filters skeleton joints to identify those related to hands by checking
    against HAND_JOINT_NAME_TOKENS and HAND_JOINT_NAME_PREFIXES settings.
    Excludes wrist joints.
    
    Args:
        joint_names: List of joint name strings from the skeleton model
        
    Returns:
        tuple: Indices of hand-related joints (excluding wrists)
    """
    hand_joint_indices = []
    for joint_idx, joint_name in enumerate(joint_names):
        name = str(joint_name).lower()
        if "wrist" in name:
            continue
        if any(token in name for token in settings.HAND_JOINT_NAME_TOKENS) or any(
            name.startswith(prefix) for prefix in settings.HAND_JOINT_NAME_PREFIXES
        ):
            hand_joint_indices.append(joint_idx)
    return tuple(hand_joint_indices)


def _extract_hand_points(image_bgr, landmarker, timestamp_ms):
    """
    Extract hand landmarks from a video frame using MediaPipe.
    
    Returns 2D pixel coordinates and 3D world coordinates for each hand.
    Only extracts specific landmark indices defined in settings.HAND_LANDMARK_IDS.
    
    Args:
        image_bgr: OpenCV BGR image frame
        landmarker: MediaPipe HandLandmarker instance
        timestamp_ms: Frame timestamp in milliseconds
        
    Returns:
        tuple: (hand_points_2d, hand_points_3d)
            - hand_points_2d: dict {"Left"/"Right": {landmark_id: (x_px, y_px)}}
            - hand_points_3d: dict {"Left"/"Right": {landmark_id: np.array([x, y, z])}}
    """
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
        
        for idx in settings.HAND_LANDMARK_IDS:
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


def _normalized_to_pixel(lm, width, height):
    """
    Convert normalized landmark coordinates to pixel coordinates.
    
    Args:
        lm: MediaPipe landmark with x, y normalized coordinates [0, 1]
        width: Image width in pixels
        height: Image height in pixels
        
    Returns:
        tuple: (x_pixel, y_pixel) clamped to image bounds
    """
    x = min(max(int(lm.x * width), 0), width - 1)
    y = min(max(int(lm.y * height), 0), height - 1)
    return x, y


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


class HandDetectionWorker:
    """
    Background worker thread for MediaPipe hand detection.
    
    Decouples hand detection (CPU-bound) from pose inference (GPU-bound).
    Results are keyed by frame ID to match with corresponding pose frames.
    Maintains a sliding window of completed results to handle async processing.
    """

    def __init__(self, landmarker):
        """
        Initialize hand detection worker.
        
        Args:
            landmarker: MediaPipe HandLandmarker instance
        """
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
        """
        Queue the latest frame for hand detection (drops stale frames).
        
        Args:
            image_bgr: OpenCV BGR image frame
            frame_id: Unique frame identifier for result matching
            timestamp_ms: Frame timestamp in milliseconds
        """
        with self._lock:
            self._pending_frame = image_bgr
            self._pending_frame_id = frame_id
            self._pending_ts = timestamp_ms
        self._frame_event.set()

    def get_result(self, frame_id, remove=False):
        """
        Return the hand detection result for a specific frame ID if available.
        
        Args:
            frame_id: Frame ID to retrieve result for
            remove: If True, remove result from cache after retrieval
            
        Returns:
            tuple: (hand_points_2d, hand_points_3d) or ({}, {}) if not ready
        """
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
        """Signal worker thread to stop and wait for completion."""
        self._stop = True
        self._frame_event.set()
        self._thread.join(timeout=1.0)

    def _store_result(self, frame_id, result):
        """
        Store detection result and maintain sliding window of recent results.
        
        Args:
            frame_id: Frame ID for result
            result: (hand_points_2d, hand_points_3d) tuple
        """
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
        """Main worker thread loop: process frames from queue."""
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

