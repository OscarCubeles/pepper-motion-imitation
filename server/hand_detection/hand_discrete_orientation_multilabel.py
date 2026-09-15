import argparse
import time
from pathlib import Path
import math
import numpy as np
import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from draw import euler_to_rotation_matrix, draw_orientation_cuboid
from EMA_smoothing_visualization import EMAVisualizationTracker


class EMASmoothing:
    """
    Exponential Moving Average smoothing for orientation vectors.
    Helps reduce jitter and occlusion artifacts.
    Discards values with low confidence to prevent corrupting the smoothed estimate.
    """
    def __init__(self, alpha=0.2, confidence_threshold=0.9):
        """
        Args:
            alpha: smoothing factor (0-1). Higher = more responsive, lower = smoother
            confidence_threshold: minimum confidence to apply smoothing (0-1)
        """
        self.alpha = alpha
        self.confidence_threshold = confidence_threshold
        self.previous_value = None
    
    def smooth(self, current_value, confidence=1.0):
        """
        Apply EMA smoothing to current value if confidence is high enough.
        
        Args:
            current_value: numpy array of current measurement
            confidence: confidence score (0-1). If below threshold, keeps previous value
        
        Returns:
            smoothed value
        """
        if self.previous_value is None:
            self.previous_value = current_value.copy()
            return current_value
        
        # If confidence is too low, don't update - keep previous smoothed value
        if confidence < self.confidence_threshold:
            return self.previous_value
        
        # EMA formula: smoothed = alpha * current + (1 - alpha) * previous
        smoothed = self.alpha * current_value + (1 - self.alpha) * self.previous_value
        self.previous_value = smoothed.copy()
        return smoothed
    
    def reset(self):
        """Reset the smoothing state."""
        self.previous_value = None


def is_back_of_hand(palm_z):
    return False


def orientation_to_euler(palm_x, palm_y, palm_z):
    """
    Convert palm axes to Euler angles (yaw, pitch, roll)
    """
    yaw = math.degrees(math.atan2(palm_x[1], palm_x[0]))
    pitch = math.degrees(math.atan2(-palm_z[0], np.sqrt(palm_z[1]**2 + palm_z[2]**2)))
    roll = math.degrees(math.atan2(palm_z[1], palm_z[2]))
    return yaw, pitch, roll


def euler_to_rotation_matrix(yaw, pitch, roll):
    yaw = np.radians(yaw)
    pitch = np.radians(pitch)
    roll = np.radians(roll)

    Rz = np.array([
        [np.cos(yaw), -np.sin(yaw), 0],
        [np.sin(yaw),  np.cos(yaw), 0],
        [0, 0, 1]
    ])
    Ry = np.array([
        [np.cos(pitch), 0, np.sin(pitch)],
        [0, 1, 0],
        [-np.sin(pitch), 0, np.cos(pitch)]
    ])
    Rx = np.array([
        [1, 0, 0],
        [0, np.cos(roll), -np.sin(roll)],
        [0, np.sin(roll),  np.cos(roll)]
    ])
    return Rz @ Ry @ Rx


def compute_hand_orientation_4(landmarks, is_left_hand=False):
    """
    Compute orientation using 4 keypoints: wrist, index MCP, pinky MCP, thumb CMC
    Returns:
        palm_x, palm_y, palm_z, back_of_hand (bool)
    
    Args:
        landmarks: MediaPipe hand landmarks
        is_left_hand: bool, whether this is a left hand
    """
    wrist = np.array([landmarks[0].x, landmarks[0].y, landmarks[0].z])
    index_mcp = np.array([landmarks[5].x, landmarks[5].y, landmarks[5].z])
    pinky_mcp = np.array([landmarks[17].x, landmarks[17].y, landmarks[17].z])
    thumb_cmc = np.array([landmarks[1].x, landmarks[1].y, landmarks[1].z])

    # Vectors along the palm plane
    v1 = index_mcp - wrist
    v2 = pinky_mcp - wrist
    
    # Compute palm normal (pointing outward from palm)
    # Due to MediaPipe's mirrored coordinates for left/right hands, use different cross product orders
    if is_left_hand:
        # For LEFT hand: v2 × v1 gives outward normal
        palm_z = np.cross(v2, v1)
    else:
        # For RIGHT hand: v1 × v2 gives outward normal (opposite order due to chirality)
        palm_z = np.cross(v1, v2)
    
    palm_z /= np.linalg.norm(palm_z)

    # X axis: use thumb as consistent anatomical reference (always on same side relative to palm)
    v_thumb = thumb_cmc - wrist
    # Project thumb vector onto palm plane to get consistent X direction
    palm_x = v_thumb - np.dot(v_thumb, palm_z) * palm_z
    palm_x /= np.linalg.norm(palm_x)

    # Y axis along palm (perpendicular to both X and Z)
    palm_y = np.cross(palm_z, palm_x)
    palm_y /= np.linalg.norm(palm_y)

    back_of_hand = is_back_of_hand(palm_z)

    return palm_x, palm_y, palm_z, back_of_hand

def compute_palm_orientation(landmarks, is_left_hand=False):
    """
    Compute palm axes using 4 keypoints: wrist(0), thumbCMC(1), indexMCP(5), pinkyMCP(17)
    Returns:
        palm_x, palm_y, palm_z: orthonormal axes
        palm_back: bool (True if back of hand facing camera)
    
    Args:
        landmarks: MediaPipe hand landmarks
        is_left_hand: bool, whether this is a left hand
    """
    # Keypoints as numpy arrays (NORMALIZED VIDEO COORDINATES from MediaPipe video detection)
    wrist = np.array([landmarks[0].x, landmarks[0].y, landmarks[0].z])
    thumb = np.array([landmarks[1].x, landmarks[1].y, landmarks[1].z])
    index = np.array([landmarks[5].x, landmarks[5].y, landmarks[5].z])
    pinky = np.array([landmarks[17].x, landmarks[17].y, landmarks[17].z])

    # DEBUG: Print raw keypoints
    hand_label = "Left" if is_left_hand else "Right"
    print(f"\n[MULTILABEL] {hand_label} Hand - NORMALIZED VIDEO keypoints:")
    print(f"  wrist: {wrist}")
    print(f"  thumb: {thumb}")
    print(f"  index: {index}")
    print(f"  pinky: {pinky}")

    # Vectors along the palm plane
    v1 = index - wrist
    v2 = pinky - wrist
    
    # DEBUG: Print vectors
    print(f"  v1 (index-wrist): {v1}")
    print(f"  v2 (pinky-wrist): {v2}")
    
    # Compute palm normal with correct cross product order for each hand chirality
    if is_left_hand:
        palm_z = np.cross(v2, v1)
    else:
        palm_z = np.cross(v1, v2)
    
    # DEBUG: Print before normalize
    print(f"  palm_z (before norm): {palm_z}")
    
    palm_z /= np.linalg.norm(palm_z)
    
    # DEBUG: Print after normalize
    print(f"  palm_z (after norm): {palm_z}")

    # X axis: use thumb as anatomical reference (consistent for both hands)
    v_thumb = thumb - wrist
    # Project thumb onto palm plane
    palm_x = v_thumb - np.dot(v_thumb, palm_z) * palm_z
    palm_x /= np.linalg.norm(palm_x)

    # DEBUG: Print palm_x
    print(f"  palm_x: {palm_x}")

    # Y axis along palm
    palm_y = np.cross(palm_z, palm_x)
    palm_y /= np.linalg.norm(palm_y)

    # DEBUG: Print all final vectors
    print(f"  palm_y: {palm_y}")
    print(f"[MULTILABEL] FINAL: palm_x={palm_x}, palm_y={palm_y}, palm_z={palm_z}\n")

    # Determine if back of hand is facing camera
    palm_back = palm_z[2] < 0  # assuming camera looks along +Z

    return palm_x, palm_y, palm_z, palm_back


def classify_palm_orientation(palm_z, threshold=0.5):
    """
    Classify palm orientation into 6 discrete directions based on palm normal vector.
    
    The palm_z vector points from inside the hand (wrist) to outside (palm direction).
    
    Coordinate system:
    - X: right
    - Y: down 
    - Z: forward (towards camera)
    
    Args:
        palm_z: unit normal vector [x, y, z]
        threshold: minimum component magnitude to classify (default 0.5)
    
    Returns:
        orientation_label: str - one of ['UP', 'DOWN', 'LEFT', 'RIGHT', 'FRONT', 'BACK']
    """
    # Normalize to ensure unit vector
    palm_z = np.array(palm_z, dtype=float)
    palm_z = palm_z / (np.linalg.norm(palm_z) + 1e-6)
    
    x, y, z = palm_z[0], palm_z[1], palm_z[2]
    
    # Get absolute values and find dominant axis
    abs_x, abs_y, abs_z = abs(x), abs(y), abs(z)
    
    # Find which axis is most dominant
    if abs_z > abs_x and abs_z > abs_y:
        # Z-axis dominant (front/back)
        return "FRONT" if z < 0 else "BACK"
    elif abs_y > abs_x and abs_y > abs_z:
        # Y-axis dominant (up/down)
        return "DOWN" if y > 0 else "UP"
    elif abs_x > abs_y and abs_x > abs_z:
        # X-axis dominant (left/right)
        return "RIGHT" if x < 0 else "LEFT"
    else:
        # Ambiguous case, use secondary priority
        if abs_z > threshold:
            return "FRONT" if z < 0 else "BACK"
        elif abs_y > threshold:
            return "DOWN" if y > 0 else "UP"
        else:
            return "RIGHT" if x < 0 else "LEFT"


def classify_palm_orientation_multilabel(palm_z):
    """
    Classify palm orientation using multilabel approach.
    Returns both the primary discrete label AND three axis-based sublabels.
    
    Coordinate system:
    - X: left (-) / right (+)
    - Y: down (-) / up (+)
    - Z: back (-) / front (+) (toward camera)
    
    Args:
        palm_z: unit normal vector [x, y, z]
    
    Returns:
        tuple: (primary_label, x_label, y_label, z_label)
        Example: ("FRONT", "right", "down", "front")
    """
    # Normalize to ensure unit vector
    palm_z = np.array(palm_z, dtype=float)
    palm_z = palm_z / (np.linalg.norm(palm_z) + 1e-6)
    
    x, y, z = palm_z[0], palm_z[1], palm_z[2]
    
    # Classify each axis independently
    x_label = "left" if x > 0 else "right"  # Match primary: x < 0 → right, x > 0 → left
    y_label = "down" if y > 0 else "up"  # Note: y > 0 means down in our coordinate system
    z_label = "front" if z < 0 else "back"  # Note: z < 0 means front (toward camera)
    
    # Determine primary label (which axis is most dominant)
    abs_x, abs_y, abs_z = abs(x), abs(y), abs(z)
    
    if abs_z > abs_x and abs_z > abs_y:
        primary_label = "FRONT" if z < 0 else "BACK"
    elif abs_y > abs_x and abs_y > abs_z:
        primary_label = "DOWN" if y > 0 else "UP"
    elif abs_x > abs_y and abs_x > abs_z:
        primary_label = "RIGHT" if x < 0 else "LEFT"
    else:
        # Fallback to z if ambiguous
        primary_label = "FRONT" if z < 0 else "BACK"
    
    return primary_label, x_label, y_label, z_label


def get_handedness_label(handedness_result):
    """
    Extract handedness label (Left or Right) from MediaPipe result.
    
    Args:
        handedness_result: MediaPipe handedness object
    
    Returns:
        str: "LEFT" or "RIGHT"
    """
    if handedness_result and len(handedness_result) > 0:
        return handedness_result[0].category_name.upper()
    return "UNKNOWN"


def debug_hand_vectors(landmarks, hand_label):
    """
    Debug function: Print raw vectors from MediaPipe landmarks.
    Shows what we're actually getting before any transformations.
    """
    wrist = np.array([landmarks[0].x, landmarks[0].y, landmarks[0].z])
    index_mcp = np.array([landmarks[5].x, landmarks[5].y, landmarks[5].z])
    pinky_mcp = np.array([landmarks[17].x, landmarks[17].y, landmarks[17].z])
    thumb_cmc = np.array([landmarks[1].x, landmarks[1].y, landmarks[1].z])
    
    # RAW vectors
    v1 = index_mcp - wrist
    v2 = pinky_mcp - wrist
    v_thumb = thumb_cmc - wrist
    
    print(f"\n{'='*80}")
    print(f"{hand_label} HAND - RAW VECTORS")
    print(f"{'='*80}")
    print(f"Wrist: {wrist}")
    print(f"Index MCP: {index_mcp}")
    print(f"Pinky MCP: {pinky_mcp}")
    print(f"Thumb CMC: {thumb_cmc}")
    print(f"\nv1 (index - wrist):   {v1} | norm: {np.linalg.norm(v1):.4f}")
    print(f"v2 (pinky - wrist):   {v2} | norm: {np.linalg.norm(v2):.4f}")
    print(f"v_thumb (thumb - wrist): {v_thumb} | norm: {np.linalg.norm(v_thumb):.4f}")
    
    # Cross products
    cross_v1_v2 = np.cross(v1, v2)
    cross_v2_v1 = np.cross(v2, v1)
    
    print(f"\nCross products:")
    print(f"v1 × v2: {cross_v1_v2} | norm: {np.linalg.norm(cross_v1_v2):.4f}")
    print(f"v2 × v1: {cross_v2_v1} | norm: {np.linalg.norm(cross_v2_v1):.4f}")
    
    # Palmx axis options
    palm_x_option1 = index_mcp - pinky_mcp  # index towards pinky
    palm_x_option2 = pinky_mcp - index_mcp  # pinky towards index
    
    print(f"\nPalm X axis options:")
    print(f"index - pinky: {palm_x_option1} | norm: {np.linalg.norm(palm_x_option1):.4f}")
    print(f"pinky - index: {palm_x_option2} | norm: {np.linalg.norm(palm_x_option2):.4f}")
    print(f"{'='*80}\n")


def draw_orientation_cuboid_from_ypr(frame, centroid, yaw, pitch, roll, size=(0.06, 0.12, 0.02)):
    """
    Draw cuboid using yaw, pitch, roll instead of a rotation matrix.

    centroid : 3D point (normalized mediapipe coords)
    yaw, pitch, roll : Euler angles in degrees
    size : cuboid dimensions (X,Y,Z)
    """
    
    h, w, _ = frame.shape

    lx, ly, lz = size

    # Convert to rotation matrix
    R = euler_to_rotation_matrix(yaw, pitch, roll)

    # cuboid vertices in local coordinate frame
    corners = np.array([
        [-lx, -ly, -lz],
        [ lx, -ly, -lz],
        [ lx,  ly, -lz],
        [-lx,  ly, -lz],
        [-lx, -ly,  lz],
        [ lx, -ly,  lz],
        [ lx,  ly,  lz],
        [-lx,  ly,  lz],
    ])

    # rotate cuboid
    rotated = (R @ corners.T).T

    # translate to centroid
    points = rotated + centroid

    # convert to pixels
    pts_px = [(int(p[0]*w), int(p[1]*h)) for p in points]

    edges = [
        (0,1),(1,2),(2,3),(3,0),
        (4,5),(5,6),(6,7),(7,4),
        (0,4),(1,5),(2,6),(3,7)
    ]

    for e in edges:
        cv2.line(frame, pts_px[e[0]], pts_px[e[1]], (0,200,255), 2)

def draw_orientation_cuboid_2(frame, centroid, R, size=(0.06, 0.12, 0.02)):
    """
    Draw a cuboid aligned with the hand coordinate frame.

    centroid : 3D point (normalized mediapipe coords)
    R        : 3x3 rotation matrix (columns = X,Y,Z axes)
    size     : cuboid size (length along X,Y,Z)
    """
    return
    h, w, _ = frame.shape

    lx, ly, lz = size

    # cuboid vertices in local coordinate frame
    corners = np.array([
        [-lx, -ly, -lz],
        [ lx, -ly, -lz],
        [ lx,  ly, -lz],
        [-lx,  ly, -lz],
        [-lx, -ly,  lz],
        [ lx, -ly,  lz],
        [ lx,  ly,  lz],
        [-lx,  ly,  lz],
    ])

    # rotate cuboid
    rotated = (R @ corners.T).T

    # translate to centroid
    points = rotated + centroid

    # convert to pixels
    pts_px = [(int(p[0]*w), int(p[1]*h)) for p in points]

    edges = [
        (0,1),(1,2),(2,3),(3,0),
        (4,5),(5,6),(6,7),(7,4),
        (0,4),(1,5),(2,6),(3,7)
    ]

    for e in edges:
        cv2.line(frame, pts_px[e[0]], pts_px[e[1]], (0,200,255), 2)

def draw_palm_square(frame, landmarks, yaw, pitch, roll, back_of_hand):
    """
    Draw a quadrilateral representing the palm using 4 keypoints:
    wrist (0), thumb CMC (1), index MCP (5), pinky MCP (17)
    Also overlay yaw/pitch/roll and palm/back label.
    """
    h, w, _ = frame.shape
    # Keypoints
    wrist = landmarks[0]
    thumb_cmc = landmarks[1]
    index_mcp = landmarks[5]
    pinky_mcp = landmarks[17]

    # Convert to pixels
    pts_px = {
        "wrist": (int(wrist.x * w), int(wrist.y * h)),
        "thumb": (int(thumb_cmc.x * w), int(thumb_cmc.y * h)),
        "index": (int(index_mcp.x * w), int(index_mcp.y * h)),
        "pinky": (int(pinky_mcp.x * w), int(pinky_mcp.y * h)),
    }

    # Draw edges in a loop: 0→1→5→17→0
    cv2.line(frame, pts_px["wrist"], pts_px["thumb"], (255, 255, 0), 2)
    cv2.line(frame, pts_px["thumb"], pts_px["index"], (255, 255, 0), 2)
    cv2.line(frame, pts_px["index"], pts_px["pinky"], (255, 255, 0), 2)
    cv2.line(frame, pts_px["pinky"], pts_px["wrist"], (255, 255, 0), 2)

    # Draw joints in the same colors as camera visualization
    cv2.circle(frame, pts_px["wrist"], 6, (0, 255, 0), -1)   # green
    cv2.circle(frame, pts_px["index"], 6, (255, 0, 0), -1)   # blue
    cv2.circle(frame, pts_px["pinky"], 6, (0, 0, 255), -1)   # red
    cv2.circle(frame, pts_px["thumb"], 6, (0, 255, 255), -1) # yellow

    # Show angles
    cv2.putText(frame, f"Yaw:{yaw:.1f} Pitch:{pitch:.1f} Roll:{roll:.1f}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 2)

    # Palm/back label
    label = "Back" if back_of_hand else "Palm"
    color = (0,0,255) if back_of_hand else (0,255,0)
    cv2.putText(frame, label, (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)








# -------------------------
# Drawing helpers
# -------------------------
HAND_CONNECTIONS = [
    (0, 1),(1, 2),(2, 3),(3, 4),
    (0, 5),(5, 6),(6, 7),(7, 8),
    (5, 9),(9,10),(10,11),(11,12),
    (9,13),(13,14),(14,15),(15,16),
    (13,17),(17,18),(18,19),(19,20),
    (0,17)
]

def _normalized_to_pixel(lm, width, height):
    x = min(max(int(lm.x * width), 0), width-1)
    y = min(max(int(lm.y * height), 0), height-1)
    return x, y


def draw_landmarks(frame, hand_landmarks, handedness):
    height, width = frame.shape[:2]
    for hand_index, landmarks in enumerate(hand_landmarks):
        points = [_normalized_to_pixel(lm, width, height) for lm in landmarks]
        for start_idx, end_idx in HAND_CONNECTIONS:
            cv2.line(frame, points[start_idx], points[end_idx], (255, 255, 255), 2)
        for point in points:
            cv2.circle(frame, point, 3, (0, 255, 255), -1)

        if handedness and hand_index < len(handedness):
            category = handedness[hand_index][0]
            label = f"{category.category_name} ({category.score:.2f})"
            x, y = _normalized_to_pixel(landmarks[0], width, height)
            y -= 10
            cv2.putText(frame,
                        label,
                        (max(0,x), max(20,y)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0,200,0),
                        2)


def draw_keypoints_from_coords(frame, keypoint_coords, label="RAW"):
    """
    Draw hand keypoints from normalized coordinates.
    Smoothed keypoints are drawn as smaller circles overlaid on top of raw keypoints.
    
    Args:
        frame: image to draw on
        keypoint_coords: dict with keys ["wrist", "thumb", "index", "pinky"], each is [x, y, z] normalized
        label: "RAW" or "SMOOTHED" for display label (not shown if empty)
    """
    height, width = frame.shape[:2]
    
    # Convert all keypoints to pixel coordinates
    pts_px = {}
    for key in ["wrist", "thumb", "index", "pinky"]:
        if key in keypoint_coords:
            coord = keypoint_coords[key]
            x = int(coord[0] * width)
            y = int(coord[1] * height)
            x = min(max(x, 0), width - 1)
            y = min(max(y, 0), height - 1)
            pts_px[key] = (x, y)
    
    # Draw keypoints with different colors - smaller circles for smoothed
    colors = {
        "wrist": (0, 255, 0),    # green
        "thumb": (0, 255, 255),  # yellow
        "index": (255, 0, 0),    # blue
        "pinky": (0, 0, 255),    # red
    }
    
    for key, color in colors.items():
        if key in pts_px:
            # Draw smaller circle for smoothed points (slightly offset to show difference)
            cv2.circle(frame, pts_px[key], 3, color, -1)  # Smaller radius than raw (5)


def draw_axes_vectors(frame, centroid, x_axis, y_axis, z_axis, scale=1.0):
    """
    Draw the X, Y, Z axes vectors as numeric components on screen
    frame: image
    centroid: np.array([x, y, z])
    x_axis, y_axis, z_axis: unit vectors in camera coordinates
    scale: optional scaling for display
    """
    h, w, _ = frame.shape

    # Scale and move to screen coordinates
    def to_px(vec):
        return int(centroid[0]*w + vec[0]*scale*w), int(centroid[1]*h - vec[1]*scale*h)

    # X axis
    end_x = to_px(x_axis)
    end_y = to_px(y_axis)
    end_z = to_px(z_axis)
    centroid_px = (int(centroid[0]*w), int(centroid[1]*h))

    #cv2.line(frame, centroid_px, end_x, (0,255,0), 2)
    #cv2.line(frame, centroid_px, end_y, (255,0,0), 2)
    #cv2.line(frame, centroid_px, end_z, (0,255,255), 2)

    # Overlay numeric vector components (rounded)
    #cv2.putText(frame, f"X: [{x_axis[0]:.2f},{x_axis[1]:.2f},{x_axis[2]:.2f}]",
    #            (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)
    #cv2.putText(frame, f"Y: [{y_axis[0]:.2f},{y_axis[1]:.2f},{y_axis[2]:.2f}]",
    #            (10,110), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,0), 1)
    #cv2.putText(frame, f"Z: [{z_axis[0]:.2f},{z_axis[1]:.2f},{z_axis[2]:.2f}]",
    #            (10,130), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,255), 1)

# -------------------------
# Main loop
# -------------------------
def run_hand_landmarks(model_path, camera_index=0, width=None, height=None):
    base_options = python.BaseOptions(model_asset_path=model_path)
    options = vision.HandLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.VIDEO,
        num_hands=2,
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5
    )

    cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)
    if width:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    if height:
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    if not cap.isOpened():
        raise RuntimeError(f"Unable to open camera index {camera_index}.")
    
    # Initialize EMA smoothers for keypoints (wrist, thumb, index_mcp, pinky_mcp) for each hand
    ema_smoothers = {
        "LEFT": {
            "wrist": EMASmoothing(alpha=0.3, confidence_threshold=0.85),
            "thumb": EMASmoothing(alpha=0.3, confidence_threshold=0.85),
            "index": EMASmoothing(alpha=0.3, confidence_threshold=0.85),
            "pinky": EMASmoothing(alpha=0.3, confidence_threshold=0.85),
            "middle": EMASmoothing(alpha=0.3, confidence_threshold=0.85),
            "ring": EMASmoothing(alpha=0.3, confidence_threshold=0.85),
            "confidence": EMASmoothing(alpha=0.3, confidence_threshold=0.0),  # Always smooth confidence
        },
        "RIGHT": {
            "wrist": EMASmoothing(alpha=0.3, confidence_threshold=0.85),
            "thumb": EMASmoothing(alpha=0.3, confidence_threshold=0.85),
            "index": EMASmoothing(alpha=0.3, confidence_threshold=0.85),
            "pinky": EMASmoothing(alpha=0.3, confidence_threshold=0.85),
            "middle": EMASmoothing(alpha=0.3, confidence_threshold=0.85),
            "ring": EMASmoothing(alpha=0.3, confidence_threshold=0.85),
            "confidence": EMASmoothing(alpha=0.3, confidence_threshold=0.0),  # Always smooth confidence
        }
    }
    
    # Initialize visualization tracker for EMA smoothing analysis
    ema_tracker = EMAVisualizationTracker(max_history=300)

    with vision.HandLandmarker.create_from_options(options) as landmarker:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            h, w_img, _ = frame.shape

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
            timestamp_ms = int(time.monotonic() * 1000)
            result = landmarker.detect_for_video(mp_image, timestamp_ms)

            annotated = frame.copy()

            if result.hand_landmarks:
                # Draw raw landmarks on the frame
                draw_landmarks(annotated, result.hand_landmarks, result.handedness)

                for hand_idx, hand in enumerate(result.hand_landmarks):
                    # Get handedness (LEFT or RIGHT)
                    hand_label = get_handedness_label(result.handedness[hand_idx] if hand_idx < len(result.handedness) else None)
                    is_left = (hand_label == "LEFT")
                    
                    # Get detection confidence from handedness result
                    hand_confidence = 0.0
                    if hand_idx < len(result.handedness) and result.handedness[hand_idx]:
                        hand_confidence = result.handedness[hand_idx][0].score
                    
                    # DEBUG: Print raw vectors
                    debug_hand_vectors(hand, hand_label)
                    
                    # -------------------------
                    # Extract and smooth keypoints
                    # -------------------------
                    # Get raw keypoints
                    wrist_raw = np.array([hand[0].x, hand[0].y, hand[0].z])
                    thumb_raw = np.array([hand[1].x, hand[1].y, hand[1].z])
                    index_mcp_raw = np.array([hand[5].x, hand[5].y, hand[5].z])
                    pinky_mcp_raw = np.array([hand[17].x, hand[17].y, hand[17].z])
                    
                    # Apply EMA smoothing to keypoint positions based on hand confidence
                    wrist = ema_smoothers[hand_label]["wrist"].smooth(wrist_raw, confidence=hand_confidence)
                    thumb = ema_smoothers[hand_label]["thumb"].smooth(thumb_raw, confidence=hand_confidence)
                    index_mcp = ema_smoothers[hand_label]["index"].smooth(index_mcp_raw, confidence=hand_confidence)
                    pinky_mcp = ema_smoothers[hand_label]["pinky"].smooth(pinky_mcp_raw, confidence=hand_confidence)
                    
                    # Track raw and smoothed keypoints for visualization
                    raw_kpts = {
                        "wrist": wrist_raw,
                        "thumb": thumb_raw,
                        "index": index_mcp_raw,
                        "pinky": pinky_mcp_raw,
                    }
                    smoothed_kpts = {
                        "wrist": wrist,
                        "thumb": thumb,
                        "index": index_mcp,
                        "pinky": pinky_mcp,
                    }
                    ema_tracker.add_frame(hand_label, raw_kpts, smoothed_kpts)
                    
                    # Draw smoothed keypoints on the same frame (in different colors/style)
                    draw_keypoints_from_coords(annotated, smoothed_kpts, label="")
                    
                    # -------------------------
                    # Compute orientation from smoothed keypoints
                    # -------------------------
                    # Vectors along the palm plane
                    v1 = index_mcp - wrist
                    v2 = pinky_mcp - wrist
                    
                    # Compute palm normal with correct cross product order per hand chirality
                    if is_left:
                        palm_z = np.cross(v2, v1)
                    else:
                        palm_z = np.cross(v1, v2)
                    palm_z /= np.linalg.norm(palm_z)
                    
                    # X axis: use thumb as consistent anatomical reference
                    v_thumb = thumb - wrist
                    palm_x = v_thumb - np.dot(v_thumb, palm_z) * palm_z
                    palm_x /= np.linalg.norm(palm_x)
                    
                    # Y axis along palm
                    palm_y = np.cross(palm_z, palm_x)
                    palm_y /= np.linalg.norm(palm_y)
                    
                    # Compute Euler angles from smoothed axes
                    yaw_old, pitch_old, roll_old = orientation_to_euler(palm_x, palm_y, palm_z)
                    
                    # Get DISCRETE orientation (single label)
                    discrete_orientation = classify_palm_orientation(palm_z)
                    
                    # Get MULTILABEL orientation (primary + 3 axis labels)
                    primary_label, x_label, y_label, z_label = classify_palm_orientation_multilabel(palm_z)
                    multilabel_str = f"{x_label} {z_label} {y_label}"  # e.g., "right front down"
                    
                    # Print discrete result to console
                    print(f"{hand_label} Hand: {discrete_orientation} | Multilabel: {multilabel_str} | Palm Normal: X={palm_z[0]:.3f}, Y={palm_z[1]:.3f}, Z={palm_z[2]:.3f}")

                    # -------------------------
                    # Extract additional keypoints for visualization
                    # -------------------------
                    # Get middle and ring finger bases (keypoints 9 and 13)
                    middle_mcp_raw = np.array([hand[9].x, hand[9].y, hand[9].z])
                    ring_mcp_raw = np.array([hand[13].x, hand[13].y, hand[13].z])
                    
                    # Apply EMA smoothing to these keypoints
                    middle_mcp = ema_smoothers[hand_label]["middle"].smooth(middle_mcp_raw, confidence=hand_confidence)
                    ring_mcp = ema_smoothers[hand_label]["ring"].smooth(ring_mcp_raw, confidence=hand_confidence)

                    # -------------------------
                    # Compute hand-centered visualization axes (using smoothed keypoints)
                    # -------------------------
                    # Note: palm_x, palm_y, palm_z are already computed and smoothed from above

                    # Centroid of the palm
                    centroid = (wrist + thumb + index_mcp + pinky_mcp) / 4
                    line_length = 0.1

                    # Compute end points for each axis
                    end_x = centroid + palm_x * line_length
                    end_z = centroid + palm_z * line_length
                    
                    # Y-axis: from centroid to midpoint between keypoint 9 (middle) and 13 (ring)
                    mid_9_13 = (middle_mcp + ring_mcp) / 2
                    palm_y_vis = mid_9_13 - centroid
                    palm_y_vis /= np.linalg.norm(palm_y_vis)
                    end_y = centroid + palm_y_vis * line_length

                    # -------------------------
                    # Compute yaw/pitch/roll from hand axes
                    # -------------------------
                    R_hand = np.column_stack((palm_x, palm_y_vis, palm_z))  # X,Y,Z columns using visualization Y-axis
                    yaw_new   = math.degrees(math.atan2(R_hand[1,0], R_hand[0,0]))
                    pitch_new = math.degrees(math.atan2(-R_hand[2,0], np.sqrt(R_hand[2,1]**2 + R_hand[2,2]**2)))
                    roll_new  = math.degrees(math.atan2(R_hand[2,1], R_hand[2,2]))

                    draw_orientation_cuboid_2(
                        annotated,
                        centroid,
                        R_hand
                    )

                    # -------------------------
                    # Map axes to webcam pixels
                    # -------------------------
                    def to_cam_px(pt):
                        return int(pt[0] * w_img), int(pt[1] * h)

                    centroid_px = to_cam_px(centroid)
                    x_px = to_cam_px(end_x)
                    y_px = to_cam_px(end_y)
                    z_px = to_cam_px(end_z)

                    # Draw the axes on the frame
                    #cv2.line(annotated, centroid_px, x_px, (0, 255, 0), 2)      # green (X-axis: palm_x, thumb direction)
                    #cv2.line(annotated, centroid_px, y_px, (255, 0, 0), 2)      # blue (Y-axis: palm_y, along palm)
                    cv2.line(annotated, centroid_px, z_px, (0, 0, 255), 2)    # yellow (Z-axis: palm_z, palm normal)
                    cv2.circle(annotated, centroid_px, 4, (0, 0, 255), -1)      # red (centroid)

                    # -------------------------
                    # Display discrete orientation on screen
                    # -------------------------
                    discrete_color = (0, 255, 0)  # green
                    cv2.putText(annotated,
                                f"{hand_label} Hand: {discrete_orientation}",
                                (10, 30 + hand_idx * 120), cv2.FONT_HERSHEY_SIMPLEX, 1.2, discrete_color, 3)
                    
                    # Display multilabel orientation (x, y, z)
                    multilabel_color = (255, 200, 0)  # cyan-ish
                    cv2.putText(annotated,
                                f"Multilabel: {multilabel_str}",
                                (10, 55 + hand_idx * 120), cv2.FONT_HERSHEY_SIMPLEX, 0.9, multilabel_color, 2)
                    
                    # Display detection confidence (real-time)
                    confidence_color = (0, 255, 255) if hand_confidence > 0.7 else (0, 165, 255)
                    #cv2.putText(annotated,
                    #            f"Confidence: {hand_confidence:.3f}",
                    #            (10, 80 + hand_idx * 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, confidence_color, 2)

            cv2.imshow("MediaPipe Hands", annotated)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()
    cv2.destroyAllWindows()
    
    # Display EMA smoothing visualization
    print("\nGenerating EMA smoothing visualization...")
    import matplotlib.pyplot as plt
    fig = ema_tracker.plot_smoothing()
    plt.show()


# -------------------------
# Main entry
# -------------------------
def main():
    parser = argparse.ArgumentParser(description="MediaPipe Hands with orientation cuboid.")
    default_model = Path(__file__).resolve().with_name("hand_landmarker.task")
    parser.add_argument("--model", default=str(default_model), help="Path to .task model")
    parser.add_argument("--camera", type=int, default=0, help="Camera index")
    parser.add_argument("--width", type=int, default=None, help="Capture width")
    parser.add_argument("--height", type=int, default=None, help="Capture height")
    args = parser.parse_args()
    run_hand_landmarks(args.model, args.camera, args.width, args.height)


if __name__ == "__main__":
    main()