import cv2
import numpy as np


HAND_CONNECTIONS = [
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 4),
    (0, 5),
    (5, 6),
    (6, 7),
    (7, 8),
    (5, 9),
    (9, 10),
    (10, 11),
    (11, 12),
    (9, 13),
    (13, 14),
    (14, 15),
    (15, 16),
    (13, 17),
    (17, 18),
    (18, 19),
    (19, 20),
    (0, 17),
]

def _normalized_to_pixel(lm, width, height):
    x = min(max(int(lm.x * width), 0), width - 1)
    y = min(max(int(lm.y * height), 0), height - 1)
    return x, y


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


# Draws a cuboid with the same orientation as the hand, to visualize the hand orientation in a more intuitive way.
def render_orientation_cuboid(yaw, pitch, roll, size=400, length=120, width=40, depth=20):
    """
    Returns an OpenCV image of a cuboid representing hand orientation.

    The cuboid is centered in the image, and the long axis corresponds to fingers.
    """
    frame = np.zeros((size, size, 3), dtype=np.uint8)
    center = np.array([size // 2, size // 2])

    # Rotation matrix from yaw/pitch/roll
    R = euler_to_rotation_matrix(yaw, pitch, roll)

    # Cuboid corners (local coordinates)
    l, w, d = length/2, width/2, depth/2
    cuboid = np.array([
        [-l, -w, -d],
        [ l, -w, -d],
        [ l,  w, -d],
        [-l,  w, -d],
        [-l, -w,  d],
        [ l, -w,  d],
        [ l,  w,  d],
        [-l,  w,  d],
    ])

    # Rotate cuboid
    rotated = cuboid @ R.T

    # Project to 2D
    points = []
    for p in rotated:
        x = int(center[0] + p[0])
        y = int(center[1] + p[1])
        points.append((x, y))

    # Draw cuboid edges
    edges = [
        (0,1),(1,2),(2,3),(3,0),  # bottom
        (4,5),(5,6),(6,7),(7,4),  # top
        (0,4),(1,5),(2,6),(3,7)   # sides
    ]
    for e in edges:
        cv2.line(frame, points[e[0]], points[e[1]], (255,255,255), 2)

    # Display angles
    cv2.putText(frame,
                f"Yaw:{yaw:.1f} Pitch:{pitch:.1f} Roll:{roll:.1f}",
                (20, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 255),
                2)
    return frame




def draw_orientation_cuboid(frame, center, yaw, pitch, roll, length=120, width=40, depth=20):
    """
    Draw a 3D cuboid to visualize hand orientation.

    Args:
        frame: OpenCV image
        center: (cx, cy) center of cuboid
        yaw, pitch, roll: angles in degrees
        length: along fingers (long)
        width: across palm
        depth: thickness
    """
    R = euler_to_rotation_matrix(yaw, pitch, roll)

    # Cuboid corners relative to center
    l, w, d = length/2, width/2, depth/2
    cuboid = np.array([
        [-l, -w, -d],
        [ l, -w, -d],
        [ l,  w, -d],
        [-l,  w, -d],
        [-l, -w,  d],
        [ l, -w,  d],
        [ l,  w,  d],
        [-l,  w,  d],
    ])

    # Rotate cuboid
    rotated = cuboid @ R.T

    # Project to 2D
    points = []
    cx, cy = center
    for p in rotated:
        x = int(cx + p[0])
        y = int(cy + p[1])
        points.append((x, y))

    # Draw edges
    edges = [
        (0,1),(1,2),(2,3),(3,0),
        (4,5),(5,6),(6,7),(7,4),
        (0,4),(1,5),(2,6),(3,7)
    ]
    for e in edges:
        cv2.line(frame, points[e[0]], points[e[1]], (0,0,255), 2)

    # Optional: draw axis labels
    cv2.putText(frame,
                f"Yaw:{yaw:.1f} Pitch:{pitch:.1f} Roll:{roll:.1f}",
                (20,30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0,255,255),
                2)



# Draws the hand links and joints on the frame, and also adds the handedness label if available.
def draw_landmarks(frame, hand_landmarks, handedness):
    height, width = frame.shape[:2]
    for hand_index, landmarks in enumerate(hand_landmarks):
        points = [_normalized_to_pixel(lm, width, height) for lm in landmarks]
        for start_idx, end_idx in HAND_CONNECTIONS:
            cv2.line(frame, points[start_idx], points[end_idx], (0, 255, 0), 2)
        for point in points:
            cv2.circle(frame, point, 3, (0, 255, 255), -1)

        if handedness and hand_index < len(handedness):
            category = handedness[hand_index][0]
            label = f"{category.category_name} ({category.score:.2f})"
            x, y = _normalized_to_pixel(landmarks[0], width, height)
            y -= 10
            cv2.putText(
                frame,
                label,
                (max(0, x), max(20, y)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 200, 0),
                2,
            )

