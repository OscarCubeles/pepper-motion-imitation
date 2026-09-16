import numpy as np
import server.common.kinematics.forward_kinematics as fk
import server.common.kinematics.constants as constants
from numpy import pi, arctan, arctan2, arcsin, arccos, sqrt, sin, cos, transpose, radians, real, isnan, nan, linalg, array


def _new_hand_tracking_state():
    return {
        "calibration_samples": [],
        "is_calibrated": False,
        "reference_distance_m": None,
        "d_min_m": None,
        "d_max_m": None,
        "last_command": constants.DEFAULT_HAND_COMMAND,
    }


RIGHT_HAND_STATE = _new_hand_tracking_state()
LEFT_HAND_STATE = _new_hand_tracking_state()


def get_arm_partial_angles_impl(ex, ey, ez, arm, apply_pepper_limits=True):
    """Get ShoulderPitch (t1) and ShoulderRoll (t2) from elbow position."""
    l1 = -57.0
    l3 = 86.82
    l4 = 181.2
    l6 = 0.13
    l2_right = -149.74
    l2_left = 149.74
    l5_right = -15.0
    l5_left = 15.0
    
    T1_MIN = -2.1
    T1_MAX = 2.1
    T2_RIGHT_MIN = -2.617
    T2_RIGHT_MAX = -0.8727
    T2_LEFT_MIN = 0.8727
    T2_LEFT_MAX = 2.617
    
    l2 = l2_right if arm == 'right' else l2_left
    l5 = l5_right if arm == 'right' else l5_left
    
    t2_temp = arcsin((ey - l2) / sqrt(l4**2 + l5**2)) - arctan(l5/l4)
    
    if arm == 'right':
        if t2_temp + arctan(l5/l4) > -pi/2 - arctan(l5/l4):
            shoulderRoll = t2_temp
        else:
            shoulderRoll = pi - t2_temp - 2*arctan(l5/l4)
        
        if apply_pepper_limits:
            shoulderRoll = max(T2_RIGHT_MIN, min(T2_RIGHT_MAX, shoulderRoll))
    
    elif arm == 'left':
        if t2_temp + arctan(l5/l4) < pi/2 - arctan(l5/l4):
            shoulderRoll = t2_temp
        else:
            shoulderRoll = pi - t2_temp - 2*arctan(l5/l4)
        
        if apply_pepper_limits:
            shoulderRoll = max(T2_LEFT_MIN, min(T2_LEFT_MAX, shoulderRoll))
    
    n = l4*cos(shoulderRoll) - l5*sin(shoulderRoll)
    t1_1 = arctan2(ex-l1, ez-l3) - arctan2(sqrt((ex-l1)**2 + (ez-l3)**2 - l6**2), l6)
    t1_2 = arctan2(l6*(ex-l1) - n*(ez-l3), l6*(ez-l3) + n*(ex-l1))
    
    if apply_pepper_limits:
        if np.iscomplex(t1_1):
            t1_1 = nan
        elif t1_1 < T1_MIN:
            t1_1 = nan
        elif t1_1 > T1_MAX:
            t1_1 = nan
        
        if np.iscomplex(t1_2):
            t1_2 = nan
        elif t1_2 < T1_MIN:
            t1_2 = nan
        elif t1_2 > T1_MAX:
            t1_2 = nan
    
    sol1 = fk.get_elbow_position(t1_1, shoulderRoll, arm)
    sol2 = fk.get_elbow_position(t1_2, shoulderRoll, arm)
    
    dist1 = sqrt((sol1[0]-ex)**2 + (sol1[1]-ey)**2 + (sol1[2]-ez)**2)
    dist2 = sqrt((sol2[0]-ex)**2 + (sol2[1]-ey)**2 + (sol2[2]-ez)**2)
    
    if dist1 <= dist2 or isnan(dist2):
        shoulderPitch = t1_1
    elif dist1 > dist2 or isnan(dist1):
        shoulderPitch = t1_2
    else:
        shoulderPitch = nan
    
    return shoulderPitch, shoulderRoll


def get_arm_partial_angles2(ex, ey, ez, arm):
    l1 = -57.0
    l3 = 86.82
    l4 = 181.2
    l6 = 0.13 

    if arm == 'right':
        l2 = -149.74
        l5 = -15.0 
    elif arm == 'left':
        l2 = 149.74
        l5 = 15.0 

    # theta2
    t2_temp = arcsin((ey - l2) / sqrt(l4**2 + l5**2)) - arctan(l5/l4)

    if arm == 'right':
        if t2_temp + arctan(l5/l4) > -pi/2 - arctan(l5/l4):
            shoulderRoll = t2_temp
        else:
            shoulderRoll = -pi - arcsin((ey - l2) / sqrt(l4**2 + l5**2)) - arctan(l5/l4)
            if shoulderRoll < -1.5630:
                shoulderRoll = t2_temp

        if shoulderRoll.imag != 0 or shoulderRoll.real < -1.58 or shoulderRoll.real >= -0.0087:
            shoulderRoll = -0.0087

    elif arm == 'left':
        if t2_temp + arctan(l5/l4) < pi/2 - arctan(l5/l4):
            shoulderRoll = t2_temp
        else:
            shoulderRoll = pi - arcsin((ey - l2) / sqrt(l4**2 + l5**2)) - arctan(l5/l4)
            if shoulderRoll > 1.5630:
                shoulderRoll = t2_temp

        if shoulderRoll.imag != 0 or shoulderRoll.real > 1.58 or shoulderRoll.real <= 0.0087:
            shoulderRoll = 0.0087

    # theta1
    n = l4*cos(shoulderRoll) - l5*sin(shoulderRoll)
    t1_1 = arctan2(ex-l1, ez-l3) - arctan2(sqrt((ex-l1)**2 + (ez-l3)**2 - l6**2), l6)
    t1_2 = arctan2(l6*(ex-l1) - n*(ez-l3), l6*(ez-l3) + n*(ex-l1))

    # print("t1: " + str(t1_1) + " " + str(t1_2))

    if t1_1.imag != 0 or t1_1.real < -2.1 or t1_1.real > 2.1:
        t1_1 = nan

    if t1_2.imag != 0 or t1_2.real < -2.1 or t1_2.real > 2.1:
        t1_2 = nan

    sol1 = fk.get_elbow_position(t1_1, shoulderRoll, arm)
    sol2 = fk.get_elbow_position(t1_2, shoulderRoll, arm)

    dist1 = sqrt((sol1[0]-ex)**2 + (sol1[1]-ey)**2 + (sol1[2]-ez)**2)
    dist2 = sqrt((sol2[0]-ex)**2 + (sol2[1]-ey)**2 + (sol2[2]-ez)**2)

    if dist1 < dist2 or isnan(dist2): 
        shoulderPitch = t1_1
    elif dist1 > dist2 or isnan(dist1):
        shoulderPitch = t1_2

    return shoulderPitch, shoulderRoll


def get_arm_partial_angles(ex, ey, ez, arm):
    """Get ShoulderPitch and ShoulderRoll for Pepper (with limits)."""
    return get_arm_partial_angles2(ex, ey, ez, arm)

def get_arm_partial_angles_human(ex, ey, ez, arm):
    """Get ShoulderPitch and ShoulderRoll for human motion (no limits)."""
    return get_arm_partial_angles_impl(ex, ey, ez, arm, apply_pepper_limits=False)

def calculate_theta4_impl( t1, t2, px, py, pz, arm, apply_pepper_limits=True):
    """Calculate ElbowRoll (t4) from wrist position."""
    l1 = -57.0
    l3 = 86.82
    l4 = 150
    d3 = 181.2
    z3 = 0.13
    l2_right = -149.74
    l2_left = 149.74
    alpha_deg = 9
    
    T4_RIGHT_MIN = 0.0087
    T4_RIGHT_MAX = 2.617
    T4_LEFT_MIN = -2.617
    T4_LEFT_MAX = -0.0087
    
    l2 = l2_right if arm == 'right' else l2_left
    
    t2_adj = t2 - pi/2
    alpha = radians(alpha_deg)
    
    term3 = (px - l1)*sin(t1) + (pz - l3)*cos(t1) - z3
    term4 = ((pz-l3)*sin(t1)*sin(t2_adj) + (py-l2)*cos(t2_adj) - d3 - 
                (px-l1)*sin(t2_adj)*cos(t1))
    term2 = sin(alpha)*term3 + cos(alpha)*term4
    term1 = (1.0/l4)*term2
    
    if term1 > 1.0:
        theta4 = nan
    else:
        if arm == 'right':
            theta4 = arccos(term1)
        elif arm == 'left':
            theta4 = -arccos(term1)
    
    if apply_pepper_limits:
        if arm == 'right':
            if not isnan(theta4) and (theta4 < T4_RIGHT_MIN or theta4 > T4_RIGHT_MAX):
                theta4 = nan
        elif arm == 'left':
            if not isnan(theta4) and (theta4 < T4_LEFT_MIN or theta4 > T4_LEFT_MAX):
                theta4 = nan
    
    return theta4

def calculate_theta4(t1, t2, px, py, pz, arm):
    """Calculate ElbowRoll (t4) for Pepper (with limits)."""
    return calculate_theta4_impl(t1, t2, px, py, pz, arm, apply_pepper_limits=True)

def calculate_theta4_human( t1, t2, px, py, pz, arm):
    """Calculate ElbowRoll (t4) for human motion (no limits)."""
    return calculate_theta4_impl(t1, t2, px, py, pz, arm, apply_pepper_limits=False)

def calculate_theta3( t1, t2, t4, px, py, pz, arm):
    """Calculate ElbowYaw (t3) from wrist position."""
    l1 = -57.0
    l3 = 86.82
    d3 = 181.2
    d5 = 150
    
    if arm == 'right':
        l2 = -149.74
        a3 = 15.0
    elif arm == 'left':
        l2 = 149.74
        a3 = -15.0
    
    t2_adj = t2 - pi/2
    alpha = radians(9)
    
    aterm = ((d3 + d5*cos(alpha)*cos(t4) + cos(t1)*sin(t2_adj)*(px-l1) - 
                sin(t1)*sin(t2_adj)*(pz-l3) - cos(t2_adj)*(py-l2)) / (d5*sin(t4)*sin(alpha)))
    bterm = ((cos(t2_adj)*sin(t1)*(pz-l3) + a3 - cos(t1)*cos(t2_adj)*(px-l1) - 
                sin(t2_adj)*(py-l2)) / (d5*sin(t4)))
    
    if not np.iscomplex(aterm) and not np.iscomplex(bterm):
        theta3 = arctan2(aterm, bterm)
        if theta3 < -2.1:
            theta3 = nan
        elif theta3 > 2.1:
            theta3 = nan
    else:
        theta3 = nan
    
    return theta3

def get_arm_t3t4_angles(t1, t2, px, py, pz, arm):
    """Get ElbowYaw (t3) and ElbowRoll (t4) from wrist position."""
    elbowRoll = calculate_theta4(t1, t2, px, py, pz, arm)
    elbowYaw = calculate_theta3(t1, t2, elbowRoll, px, py, pz, arm)
    return elbowYaw, elbowRoll


# Function that returns the head angles given the head end-effector position (BottomCamera)
def get_head_angles(px, py, pz):
    l1 = -38.0
    l2 = 169.9
    # bottom camera
    l3 = 93.6
    l4 = 61.6

    # theta1
    headYaw  = arctan2(py, (px - l1))
    # theta2
    headPitch = arcsin((l2 - pz) / sqrt(l4**2 + l3**2)) + arctan(l4/l3)
    # print("Head Pitch: " + str(headPitch))

    # Check if the calculated angles fall within Pepper's range
    # if not set the angles to 0
    if headYaw.imag != 0.0:  
        headYaw = nan
    elif headYaw.real < -2.1:
        headYaw = -2.1
    elif headYaw.real > 2.1:
        headYaw = 2.1

    if headPitch.imag != 0.0:
        headPitch = nan
    elif headPitch.real < -0.71:
        headPitch = -0.71
    elif headPitch.real > 0.45:
        headPitch = 0.45

    return headYaw, headPitch


# Function that returns the HipPitch (t2), HipRoll (t3) and KneePitch (t4) given the position of the torso (px, py, pz)
def get_torso_angles_new(px, py, pz):
    global prev_hipPitch_new

    l1 = 0.02
    l2 = 139.0
    a2 = 268.0
    a3 = 79.0

    # theta3
    hipRoll = arcsin(py/l2)
    A = l2*cos(hipRoll) + a3
    # theta2
    term1 = (px**2 + pz**2 - A**2 - l1**2 - a2**2) / (2*a2 * sqrt(l1**2 + A**2))

    t2_1 = arcsin(term1) - arctan2(A, l1)
    t2_2 = pi - arcsin(term1) - arctan2(A, l1)

    # Clamping t2 values to be within Pepper's range
    if t2_1.imag != 0: 
        t2_1 = nan
    elif t2_1.real < -1.04:
        t2_1 = -1.04
    elif t2_1.real > 1.04:
        t2_1 = 1.04

    if t2_2.imag != 0: 
        t2_2 = nan
    elif t2_2.real < -1.04:
        t2_2 = -1.04
    elif t2_2.real > 1.04:
        t2_2 = 1.04

    # Forward kinematics solutions
    sol1 = fk.get_torso_position(t2_1, hipRoll)
    sol2 = fk.get_torso_position(t2_2, hipRoll)

    # Calculate the distance between the forward kinematics solutions and the desired position
    dist1 = sqrt((sol1[0]-px)**2 + (sol1[1]-py)**2 + (sol1[2]-pz)**2)
    dist2 = sqrt((sol2[0]-px)**2 + (sol2[1]-py)**2 + (sol2[2]-pz)**2)

    # Choose the t2 value that results in the smaller distance to the desired position
    if isnan(dist1) and isnan(dist2):
        hipPitch = nan
        kneePitch = nan
    elif dist1 < dist2 or isnan(dist2): 
        hipPitch = t2_1
        t2_rounded = round(t2_1,2)
        kneePitch = constants.pairs[t2_rounded]
    elif dist1 > dist2 or isnan(dist1):
        hipPitch = t2_2
        t2_rounded = round(t2_2,2)
        kneePitch = constants.pairs[t2_rounded]

    # Update the previous hipPitch value if the current hipPitch is valid, otherwise use the previous value
    if not isnan(hipPitch):
        prev_hipPitch_new = round(hipPitch, 2)
    else:
        hipPitch = prev_hipPitch_new
        kneePitch = constants.pairs.get(round(prev_hipPitch_new, 2), nan)

    return kneePitch, hipPitch, hipRoll


# Returns a continuous Pepper hand command in [0, 1] using wrist-tip distance.
def hand_open_close_right(wrist, tip, prev_dist=None):
    return update_hand_command(RIGHT_HAND_STATE, wrist, tip)


# Returns a continuous Pepper hand command in [0, 1] using wrist-tip distance.
def hand_open_close_left(wrist, tip, prev_dist=None):
    return update_hand_command(LEFT_HAND_STATE, wrist, tip)


def _is_finite_number(value):
    try:
        number = float(value)
    except (TypeError, ValueError):
        return False
    if number != number:
        return False
    return number != float("inf") and number != float("-inf")


def _is_valid_3d_point(point):
    if point is None:
        return False
    try:
        if len(point) < 3:
            return False
    except TypeError:
        return False
    return _is_finite_number(point[0]) and _is_finite_number(point[1]) and _is_finite_number(point[2])



def compute_wrist_tip_distance_m(wrist, tip):
    # Use the 3D Euclidean distance in meters between wrist and hand tip.
    if not _is_valid_3d_point(wrist) or not _is_valid_3d_point(tip):
        return None

    dx = float(wrist[0]) - float(tip[0])
    dy = float(wrist[1]) - float(tip[1])
    dz = float(wrist[2]) - float(tip[2])
    distance_m = sqrt(dx * dx + dy * dy + dz * dz)

    if not _is_finite_number(distance_m):
        return None
    if distance_m < constants.HAND_MIN_VALID_DISTANCE_M or distance_m > constants.HAND_MAX_VALID_DISTANCE_M:
        # Ignore clearly unrealistic measurements.
        return None
    return float(distance_m)


def _clamp(value, lower, upper):
    if value < lower:
        return lower
    if value > upper:
        return upper
    return value


def _initialize_hand_range(state, reference_distance_m):
    d_min = max(constants.HAND_MIN_VALID_DISTANCE_M, reference_distance_m - constants.HAND_REFERENCE_MARGIN_M)
    d_max = min(constants.HAND_MAX_VALID_DISTANCE_M, reference_distance_m + constants.HAND_REFERENCE_MARGIN_M)
    if d_max - d_min <= constants.HAND_DISTANCE_EPSILON:
        d_max = d_min + constants.HAND_DISTANCE_EPSILON

    state["reference_distance_m"] = reference_distance_m
    state["d_min_m"] = d_min
    state["d_max_m"] = d_max
    state["is_calibrated"] = True
    state["last_command"] = constants.DEFAULT_HAND_COMMAND


def update_hand_command(state, wrist, tip):
    distance_m = compute_wrist_tip_distance_m(wrist, tip)
    if distance_m is None:
        return state["last_command"]

    # Initial calibration: collect N valid samples and center command at 0.5.
    if not state["is_calibrated"]:
        state["calibration_samples"].append(distance_m)
        if len(state["calibration_samples"]) >= constants.HAND_CALIBRATION_FRAME_COUNT:
            reference_distance_m = sum(state["calibration_samples"]) / float(len(state["calibration_samples"]))
            _initialize_hand_range(state, reference_distance_m)
        return constants.DEFAULT_HAND_COMMAND

    if distance_m > state["d_max_m"]:
        state["d_max_m"] = distance_m
    elif distance_m < state["d_min_m"]:
        state["d_min_m"] = distance_m

    span = state["d_max_m"] - state["d_min_m"]
    if span <= constants.HAND_DISTANCE_EPSILON:
        state["last_command"] = constants.DEFAULT_HAND_COMMAND
        return constants.DEFAULT_HAND_COMMAND

    command = (distance_m - state["d_min_m"]) / span
    command = _clamp(command, 0.0, 1.0)
    state["last_command"] = command
    return command