import numpy as np
import settings
import wrist_yaw_db


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
    x_label = "left" if x > 0 else "right"
    y_label = "down" if y > 0 else "up"
    z_label = "front" if z < 0 else "back"
    
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


def compute_hand_palm_vectors(hand_points_world_3d):
    """
    Compute palm_x, palm_y, palm_z vectors for each hand from keypoints.
    Uses same algorithm as client-side hand_orientation.py
    
    Args:
        hand_points_world_3d: dict with "Left"/"Right" keys containing keypoint dicts
        
    Returns:
        dict with "Left"/"Right" keys containing (palm_x, palm_y, palm_z) tuples
    """
    palm_vectors = {}
    
    for hand_label, keypoints in hand_points_world_3d.items():
        if 0 in keypoints and 1 in keypoints and 5 in keypoints and 17 in keypoints:
            try:
                is_left = (hand_label == "Left")
                wrist = np.array(keypoints[0], dtype=np.float32)
                thumb = np.array(keypoints[1], dtype=np.float32)
                index = np.array(keypoints[5], dtype=np.float32)
                pinky = np.array(keypoints[17], dtype=np.float32)
                
                # DEBUG: Print raw keypoints
                
                
                # Compute palm normal with correct cross product order per hand chirality
                v1 = index - wrist
                v2 = pinky - wrist
                        
                if is_left:
                    palm_z = np.cross(v2, v1)
                else:
                    palm_z = np.cross(v1, v2)
                
                # DEBUG: Print before normalize
                
                norm_z = np.linalg.norm(palm_z)
                if norm_z > 1e-6:
                    palm_z = palm_z / norm_z
                else:
                    palm_z = np.array([0.0, 0.0, 1.0], dtype=np.float32)
                
                # DEBUG: Print after normalize
                
                # X axis: thumb projection onto palm plane
                v_thumb = thumb - wrist
                palm_x = v_thumb - np.dot(v_thumb, palm_z) * palm_z
                norm_x = np.linalg.norm(palm_x)
                if norm_x > 1e-6:
                    palm_x = palm_x / norm_x
                else:
                    palm_x = np.array([1.0, 0.0, 0.0], dtype=np.float32)
                
                # DEBUG: Print palm_x
                
                # Y axis: perpendicular to both X and Z
                palm_y = np.cross(palm_z, palm_x)
                norm_y = np.linalg.norm(palm_y)
                if norm_y > 1e-6:
                    palm_y = palm_y / norm_y
                else:
                    palm_y = np.array([0.0, 1.0, 0.0], dtype=np.float32)
                
 
                palm_vectors[hand_label] = (palm_x, palm_y, palm_z)
            except Exception as e:
                print("Error computing hand palm vectors for {0}: {1}".format(hand_label, e))
    
    return palm_vectors


def compute_hand_orientation_labels(hand_points_world_3d):
    """
    Compute hand orientation multilabel classifications for both hands.
    Includes primary label and three axis-based sublabels.
    
    Args:
        hand_points_world_3d: dict with "Left"/"Right" keys containing keypoint dicts
        
    Returns:
        dict with "Left"/"Right" keys containing dicts with:
            - 'primary': primary orientation label
            - 'x_label': X-axis label
            - 'y_label': Y-axis label
            - 'z_label': Z-axis label
            - 'multilabel': combined string like "RIGHT FRONT DOWN"
    """
    orientation_labels = {}
    
    for hand_label, keypoints in hand_points_world_3d.items():
        if 0 in keypoints and 1 in keypoints and 5 in keypoints and 17 in keypoints:
            try:
                is_left = (hand_label == "Left")
                wrist = np.array(keypoints[0], dtype=np.float32)
                index = np.array(keypoints[5], dtype=np.float32)
                pinky = np.array(keypoints[17], dtype=np.float32)
                
                # Compute palm_z with chirality
                v1 = index - wrist
                v2 = pinky - wrist
                
                if is_left:
                    palm_z = np.cross(v2, v1)
                else:
                    palm_z = np.cross(v1, v2)
                
                norm_z = np.linalg.norm(palm_z)
                if norm_z > 1e-6:
                    palm_z = palm_z / norm_z
                else:
                    continue
                
                # Classify
                primary_label, x_label, y_label, z_label = classify_palm_orientation_multilabel(palm_z)
                multilabel_str = u"{0} {1} {2}".format(x_label.upper(), z_label.upper(), y_label.upper())
                
                orientation_labels[hand_label] = {
                    'primary': primary_label,
                    'x_label': x_label,
                    'y_label': y_label,
                    'z_label': z_label,
                    'multilabel': multilabel_str
                }
            except Exception as e:
                print("Error computing orientation labels for {0}: {1}".format(hand_label, e))
    
    return orientation_labels

def get_wrist_yaws(U_left, F_left, U_right, F_right, hand_orientation_labels, prev_yaw_R, prev_yaw_L):
    """
    Get wrist yaw classifications for both hands.
    
    Args:
        U_left, F_left: upper arm and forearm directions for left hand
        U_right, F_right: upper arm and forearm directions for right hand
        hand_orientation_labels: dict with "Left"/"Right" keys containing orientation label dicts
        prev_yaw_R, prev_yaw_L: previous wrist yaw values for right and left hands (for degenerate cases)
    
    Returns:
        tuple: (wrist_yaw_right, wrist_yaw_left)
    """
    #yaw_right = get_wrist_yaw(U_right, F_right, hand_orientation_labels.get("Right", {}).get('primary', None), prev_yaw_R)
    yaw_right = get_wrist_yaw2(U_right, F_right, hand_orientation_labels.get("Right", {}).get('primary', None), prev_yaw_R, 'right')
    #print(f"Computed right wrist yaw: {yaw_right}")
    #yaw_left = get_wrist_yaw(U_left, F_left, hand_orientation_labels.get("Left", {}).get('primary', None), prev_yaw_L)
    yaw_left = get_wrist_yaw2(U_left, F_left, hand_orientation_labels.get("Left", {}).get('primary', None), prev_yaw_L, 'left')
    
    return yaw_right, yaw_left


# TODO: At some point this should be changed to having the same names everywhere to avoid normalizing back and forth. 
def normalize_orientation(label: str) -> str:
    """
    Convert all orientation formats to canonical lowercase form.
    """
    if label is None:
        return None

    label = label.strip().lower()

    mapping = {
        "front": "forward",
        "forward": "forward",

        "back": "backward",
        "backward": "backward",

        "left": "left",
        "right": "right",

        "up": "up",
        "down": "down",
    }

    return mapping.get(label, label)


def get_wrist_yaw(U, F, H, prev_yaw):

    U = normalize_orientation(U)
    F = normalize_orientation(F)
    H = normalize_orientation(H)
    print("Computing wrist yaw for hand with U={0}, F={1}, H={2}".format(U, F, H))
    # Degenerate case
    if U == F or U == settings.opposite_wrist.get(F):
        # TODO: Handle this as this is a singularity position case :)
        print("Degenerate case for wrist yaw: U={0}, F={1}. Returning previous yaw: {2}".format(U, F, prev_yaw))
        return prev_yaw

    # Missing mapping safety
    if (F, U) not in settings.cross_wrist:
        print("Warning: (F, U) pair ({0}, {1}) not in cross_wrist mapping. Returning previous yaw: {2}".format(F, U, prev_yaw))
        return prev_yaw

    R = settings.cross_wrist[(F, U)]

    if (F, R) not in settings.cross_wrist:
        print("Warning: (F, R) pair ({0}, {1}) not in cross_wrist mapping. Returning previous yaw: {2}".format(F, R, prev_yaw))
        return prev_yaw

    T = settings.cross_wrist[(F, R)]

    if H == R:
        return 0
    elif H == T:
        return 90
    elif H == settings.opposite_wrist.get(R):
        return 180
    elif H == settings.opposite_wrist.get(T):
        return -90
    else:
        return prev_yaw
    

def normalize_label(label):
    if not label:
        return ''
    
    label = label.strip().upper()
    
    # Normalize synonyms
    if label == "FORWARD":
        return "FRONT"
    
    return label


def label_to_angle(orientation_label, arm):
    """
    Convert hand orientation label to wrist yaw angle in degrees.
    
    Args:
        orientation_label: One of "OUTWARD", "INWARD", "360", "CENTERED", or containing "NAN"
        arm: Hand side - 'left' or 'right'
    
    Returns:
        Angle in degrees. Mapping:
        - "CENTERED": 0°
        - "INWARD": 60° (left), -60° (right)
        - "OUTWARD": -60° (left), 60° (right)
        - "360": 0°
        - Contains "NAN": 0°
        - None or unrecognized: 0°
    """
    if orientation_label is None:
        return 0
    
    label_str = str(orientation_label).strip().upper()
    
    # Check for NAN
    if "NAN" in label_str:
        return 0
    
    # Check for CENTERED and 360
    if label_str == "CENTERED" or label_str == "360":
        return 0
    
    # Normalize arm to lowercase for comparison
    arm_side = str(arm).strip().lower() if arm else 'right'
    
    # INWARD: left hand 60°, right hand -60°
    if label_str == "INWARD":
        return 60 if arm_side == 'left' else -60
    
    # OUTWARD: opposite of INWARD (left hand -60°, right hand 60°)
    elif label_str == "OUTWARD":
        return -60 if arm_side == 'left' else 60
    
    elif label_str == "SUPER OUTWARD":
        return -90 if arm_side == 'left' else 90

    # Default fallback for unrecognized labels
    return 0 
    

def get_wrist_yaw2(U, F, H, prev_yaw, arm):
    """Lookup wrist yaw from the CSV-based table.

    Uses `wrist_yaw_db` singleton to query by (UpperArm, Forearm, HandPalm).
    Returns a numeric value when possible, otherwise falls back to `prev_yaw`.
    """
    # Normalize to CSV keys (uppercase)
    upper = normalize_label(U)
    fore = normalize_label(F)
    palm = normalize_label(H)
    hand_flag = (arm or '').strip().lower()
    #print(upper, fore, palm, hand_flag)

    val = wrist_yaw_db.query_wrist_twist(upper, fore, palm, hand_flag)
    #print(f"Retrieved wrist yaw value: {val}")
    if val is None:
        val = prev_yaw
    
    return label_to_angle(val, arm)

    #v = val.strip()
    # Treat obvious NAN markers as missing
    #if 'NAN' in v.upper():
    #    return prev_yaw

    # Common token mappings
    #if v.upper() == 'CENTERED':
    #    return 0

    # Try numeric conversion
    #try:
    #    return float(v)
    #except Exception:
    #    # Try to extract a number from string (e.g. '360', '360;')
    #    try:
    #        import re
    #        m = re.search(r'-?\d+(?:\.\d+)?', v)
    #        if m:
    #            return float(m.group(0))
    #    except Exception:
    #        pass

    # Fallback to previous yaw if we can't interpret the CSV cell
    