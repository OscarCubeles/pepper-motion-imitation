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


def compute_hand_orientation_4(landmarks):
    """
    Compute orientation using 4 keypoints: wrist, index MCP, pinky MCP, thumb CMC
    Returns:
        palm_x, palm_y, palm_z, back_of_hand (bool)
    """
    wrist = np.array([landmarks[0].x, landmarks[0].y, landmarks[0].z])
    index_mcp = np.array([landmarks[5].x, landmarks[5].y, landmarks[5].z])
    pinky_mcp = np.array([landmarks[17].x, landmarks[17].y, landmarks[17].z])
    thumb_cmc = np.array([landmarks[1].x, landmarks[1].y, landmarks[1].z])

    # Vectors along the palm plane
    v1 = index_mcp - wrist
    v2 = pinky_mcp - wrist
    palm_z = np.cross(v1, v2)
    palm_z /= np.linalg.norm(palm_z)

    # X axis across palm (pinky → index)
    palm_x = index_mcp - pinky_mcp
    palm_x /= np.linalg.norm(palm_x)

    # Y axis along palm
    palm_y = np.cross(palm_z, palm_x)
    palm_y /= np.linalg.norm(palm_y)

    back_of_hand = is_back_of_hand(palm_z)

    return palm_x, palm_y, palm_z, back_of_hand

def compute_palm_orientation(landmarks):
    """
    Compute palm axes using 4 keypoints: wrist(0), thumbCMC(1), indexMCP(5), pinkyMCP(17)
    Returns:
        palm_x, palm_y, palm_z: orthonormal axes
        palm_back: bool (True if back of hand facing camera)
    """
    # Keypoints as numpy arrays
    wrist = np.array([landmarks[0].x, landmarks[0].y, landmarks[0].z])
    thumb = np.array([landmarks[1].x, landmarks[1].y, landmarks[1].z])
    index = np.array([landmarks[5].x, landmarks[5].y, landmarks[5].z])
    pinky = np.array([landmarks[17].x, landmarks[17].y, landmarks[17].z])

    # Option A: normal from 3 points
    v1 = index - wrist
    v2 = pinky - wrist
    palm_z = np.cross(v1, v2)
    palm_z /= np.linalg.norm(palm_z)

    # X axis across palm
    palm_x = index - pinky
    palm_x /= np.linalg.norm(palm_x)

    # Y axis along palm
    palm_y = np.cross(palm_z, palm_x)
    palm_y /= np.linalg.norm(palm_y)

    # Determine if back of hand is facing camera
    palm_back = palm_z[2] < 0  # assuming camera looks along +Z

    return palm_x, palm_y, palm_z, palm_back

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
    cv2.putText(frame, f"X: [{x_axis[0]:.2f},{x_axis[1]:.2f},{x_axis[2]:.2f}]",
                (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)
    cv2.putText(frame, f"Y: [{y_axis[0]:.2f},{y_axis[1]:.2f},{y_axis[2]:.2f}]",
                (10,110), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,0), 1)
    cv2.putText(frame, f"Z: [{z_axis[0]:.2f},{z_axis[1]:.2f},{z_axis[2]:.2f}]",
                (10,130), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,255), 1)

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

    with vision.HandLandmarker.create_from_options(options) as landmarker:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            h, w_img, _ = frame.shape
            annotated = frame.copy()

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
            timestamp_ms = int(time.monotonic() * 1000)
            result = landmarker.detect_for_video(mp_image, timestamp_ms)

            if result.hand_landmarks:
                draw_landmarks(annotated, result.hand_landmarks, result.handedness)

                for hand in result.hand_landmarks:
                    # -------------------------
                    # Compute old orientation (traditional yaw/pitch/roll)
                    # -------------------------
                    palm_x, palm_y, palm_z, palm_back = compute_hand_orientation_4(hand)
                    yaw_old, pitch_old, roll_old = orientation_to_euler(palm_x, palm_y, palm_z)

                    # -------------------------
                    # Compute new hand-centered axes
                    # -------------------------
                    wrist = np.array([hand[0].x, hand[0].y, hand[0].z])
                    thumb = np.array([hand[1].x, hand[1].y, hand[1].z])
                    index_mcp = np.array([hand[5].x, hand[5].y, hand[5].z])
                    pinky_mcp = np.array([hand[17].x, hand[17].y, hand[17].z])

                    # Centroid of the palm
                    centroid = (wrist + thumb + index_mcp + pinky_mcp) / 4
                    line_length = 0.1

                    # Palm normal
                    palm_normal = palm_z
                    end_normal = centroid + palm_normal * line_length

                    # Vertical axis (midpoint wrist/thumb → midpoint index/pinky)
                    mid_top = (wrist + thumb) / 2
                    mid_bottom = (index_mcp + pinky_mcp) / 2
                    vertical = mid_bottom - mid_top
                    vertical /= np.linalg.norm(vertical)
                    end_vertical = centroid + vertical * line_length

                    # Third axis perpendicular to palm_normal and vertical
                    third_axis = np.cross(palm_normal, vertical)
                    third_axis /= np.linalg.norm(third_axis)
                    end_third = centroid + third_axis * line_length

                    # -------------------------
                    # Compute new yaw/pitch/roll from hand axes
                    # -------------------------
                    R_hand = np.column_stack((third_axis, vertical, palm_normal))  # X,Y,Z
                    yaw_new   = math.degrees(math.atan2(R_hand[1,0], R_hand[0,0]))
                    pitch_new = math.degrees(math.atan2(-R_hand[2,0], np.sqrt(R_hand[2,1]**2 + R_hand[2,2]**2)))
                    roll_new  = math.degrees(math.atan2(R_hand[2,1], R_hand[2,2]))

                    draw_orientation_cuboid_2( # TODO: this works well but not the "from ypr version", need to check the math there
                        annotated,
                        centroid,
                        R_hand
                    )

                    # Draws hand cuboid fron yaw/pitch/roll
                    #draw_orientation_cuboid(annotated,centroid, yaw_new, pitch_new, roll_new)

                    #draw_orientation_cuboid_from_ypr(
                    #    annotated,
                    #    centroid,
                    #    yaw_new,
                    #    pitch_new,
                    #    roll_new
                    #)

                    # -------------------------
                    # Map axes to webcam pixels
                    # -------------------------
                    def to_cam_px(pt):
                        return int(pt[0] * w_img), int(pt[1] * h)

                    centroid_px = to_cam_px(centroid)
                    normal_px = to_cam_px(end_normal)
                    vertical_px = to_cam_px(end_vertical)
                    third_px = to_cam_px(end_third)

                    # Draw the axes on webcam
                    cv2.line(annotated, centroid_px, normal_px, (0, 255, 255), 2)   # cyan
                    cv2.line(annotated, centroid_px, vertical_px, (255, 0, 0), 2)   # blue
                    cv2.line(annotated, centroid_px, third_px, (0, 255, 0), 2)      # green
                    cv2.circle(annotated, centroid_px, 4, (0, 0, 255), -1)          # red

                    # Draw the axes vectors numerically
                    draw_axes_vectors(
                        annotated,
                        centroid,
                        third_axis,   # X axis
                        vertical,     # Y axis
                        palm_normal,  # Z axis
                        scale=0.1
                    )

                    # -------------------------
                    # Display both old and new yaw/pitch/roll
                    # -------------------------
                    cv2.putText(annotated,
                                f"OLD YPR: Y:{yaw_old:.1f} P:{pitch_old:.1f} R:{roll_old:.1f}",
                                (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255),2)
                    cv2.putText(annotated,
                                f"NEW YPR: Y:{yaw_new:.1f} P:{pitch_new:.1f} R:{roll_new:.1f}",
                                (10,60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0),2)

            cv2.imshow("MediaPipe Hands", annotated)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()
    cv2.destroyAllWindows()


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