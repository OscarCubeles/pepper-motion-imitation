from numpy import isnan, nan, arctan, arctan2, arcsin, arccos, sqrt, radians, pi, real, zeros, cos, sin
from forward_kinematics import get_torso_position, get_elbow_position
from hip_knee_pairs import pairs

prev_hipPitch_new = 0.0
IK_BRANCH_SWITCH_MARGIN_MM = 10.0

_prev_solution_branch = {
    "arm_right_shoulder_pitch": None,
    "arm_left_shoulder_pitch": None,
    "torso_hip_pitch": None,
}

HAND_CALIBRATION_FRAME_COUNT = 10
HAND_REFERENCE_MARGIN_M = 0.01
HAND_MIN_VALID_DISTANCE_M = 0.02
HAND_MAX_VALID_DISTANCE_M = 0.25
HAND_DISTANCE_EPSILON = 1e-6
DEFAULT_HAND_COMMAND = 0.5


def _new_hand_tracking_state():
    return {
        "calibration_samples": [],
        "is_calibrated": False,
        "reference_distance_m": None,
        "d_min_m": None,
        "d_max_m": None,
        "last_command": DEFAULT_HAND_COMMAND,
    }


RIGHT_HAND_STATE = _new_hand_tracking_state()
LEFT_HAND_STATE = _new_hand_tracking_state()


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


def _clamp(value, lower, upper):
    if value < lower:
        return lower
    if value > upper:
        return upper
    return value


def _choose_solution_branch(prev_choice, dist1, dist2):
    if not _is_finite_number(dist1) and not _is_finite_number(dist2):
        if prev_choice in (1, 2):
            return prev_choice
        return 1
    if not _is_finite_number(dist1):
        return 2
    if not _is_finite_number(dist2):
        return 1

    if prev_choice == 1:
        if dist2 + IK_BRANCH_SWITCH_MARGIN_MM < dist1:
            return 2
        return 1
    if prev_choice == 2:
        if dist1 + IK_BRANCH_SWITCH_MARGIN_MM < dist2:
            return 1
        return 2

    if dist1 <= dist2:
        return 1
    return 2


def reset_branch_selection_state():
    _prev_solution_branch["arm_right_shoulder_pitch"] = None
    _prev_solution_branch["arm_left_shoulder_pitch"] = None
    _prev_solution_branch["torso_hip_pitch"] = None


def _compute_wrist_tip_distance_m(wrist, tip):
    # Use the 3D Euclidean distance in meters between wrist and hand tip.
    if not _is_valid_3d_point(wrist) or not _is_valid_3d_point(tip):
        return None

    dx = float(wrist[0]) - float(tip[0])
    dy = float(wrist[1]) - float(tip[1])
    dz = float(wrist[2]) - float(tip[2])
    distance_m = sqrt(dx * dx + dy * dy + dz * dz)

    if not _is_finite_number(distance_m):
        return None
    if distance_m < HAND_MIN_VALID_DISTANCE_M or distance_m > HAND_MAX_VALID_DISTANCE_M:
        # Ignore clearly unrealistic measurements.
        return None
    return float(distance_m)


def _initialize_hand_range(state, reference_distance_m):
    d_min = max(HAND_MIN_VALID_DISTANCE_M, reference_distance_m - HAND_REFERENCE_MARGIN_M)
    d_max = min(HAND_MAX_VALID_DISTANCE_M, reference_distance_m + HAND_REFERENCE_MARGIN_M)
    if d_max - d_min <= HAND_DISTANCE_EPSILON:
        d_max = d_min + HAND_DISTANCE_EPSILON

    state["reference_distance_m"] = reference_distance_m
    state["d_min_m"] = d_min
    state["d_max_m"] = d_max
    state["is_calibrated"] = True
    state["last_command"] = DEFAULT_HAND_COMMAND


def _update_hand_command(state, wrist, tip):
    distance_m = _compute_wrist_tip_distance_m(wrist, tip)
    if distance_m is None:
        return state["last_command"]

    # Initial calibration: collect N valid samples and center command at 0.5.
    if not state["is_calibrated"]:
        state["calibration_samples"].append(distance_m)
        if len(state["calibration_samples"]) >= HAND_CALIBRATION_FRAME_COUNT:
            reference_distance_m = sum(state["calibration_samples"]) / float(len(state["calibration_samples"]))
            _initialize_hand_range(state, reference_distance_m)
        return DEFAULT_HAND_COMMAND

    if distance_m > state["d_max_m"]:
        state["d_max_m"] = distance_m
    elif distance_m < state["d_min_m"]:
        state["d_min_m"] = distance_m

    span = state["d_max_m"] - state["d_min_m"]
    if span <= HAND_DISTANCE_EPSILON:
        state["last_command"] = DEFAULT_HAND_COMMAND
        return DEFAULT_HAND_COMMAND

    command = (distance_m - state["d_min_m"]) / span
    command = _clamp(command, 0.0, 1.0)
    state["last_command"] = command
    return command

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

# Function that returns all 4 angles of the arm chain given the elbow positon (ex, ey, ez) and wrist position (px, py, pz)
def get_arm_all_angles(ex, ey, ez, px, py, pz, arm):
    [shoulderPitch, shoulderRoll] = get_arm_partial_angles(ex, ey, ez, arm)
    elbowRoll = calculate_theta4(shoulderPitch, shoulderRoll, px, py, pz, arm)
    elbowYaw = calculate_theta3(shoulderPitch, shoulderRoll, elbowRoll, px, py, pz, arm)

    return shoulderPitch, shoulderRoll, elbowYaw, elbowRoll

# Function that returns the ElbowYaw (t3) and ElbowRoll (t4) given the partial angles (t1, t2) and the wrist position (px, py, pz)
def get_arm_t3t4_angles(t1, t2, px, py, pz, arm):
    elbowRoll = calculate_theta4(t1, t2, px, py, pz, arm)
    elbowYaw = calculate_theta3(t1, t2, elbowRoll, px, py, pz, arm)

    return elbowYaw, elbowRoll

# Function that returns the partial angles ShoulderPitch (t1) and ShoulderRoll (t2) given the elbow position (ex, ey, ez)
def get_arm_partial_angles(ex, ey, ez, arm):
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

        if shoulderRoll.imag != 0:
            shoulderRoll = nan
        elif shoulderRoll.real < -1.57:  
            shoulderRoll = -1.5630
        elif shoulderRoll.real >= -0.0087:
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
        if shoulderRoll.imag != 0:
            shoulderRoll = nan
        elif shoulderRoll.real <= 0.0087:  
            shoulderRoll = 0.0087
        elif shoulderRoll.real > 1.57:
            shoulderRoll = 1.5630

    # theta1
    n = l4*cos(shoulderRoll) - l5*sin(shoulderRoll)
    t1_1 = arctan2(ex-l1, ez-l3) - arctan2(sqrt((ex-l1)**2 + (ez-l3)**2 - l6**2), l6)
    t1_2 = arctan2(l6*(ex-l1) - n*(ez-l3), l6*(ez-l3) + n*(ex-l1))

    if t1_1.imag != 0: 
        t1_1 = nan
    elif t1_1.real < -2.1: 
        t1_1 = -2.1
    elif t1_1.real > 2.1:
        t1_1 = 2.1

    if t1_2.imag != 0: 
        t1_2 = nan
    elif t1_2.real < -2.1: 
        t1_2 = -2.1
    elif t1_2.real > 2.1:
        t1_2 = 2.1

    sol1 = get_elbow_position(t1_1, shoulderRoll, arm)
    sol2 = get_elbow_position(t1_2, shoulderRoll, arm)

    dist1 = sqrt((sol1[0]-ex)**2 + (sol1[1]-ey)**2 + (sol1[2]-ez)**2)
    dist2 = sqrt((sol2[0]-ex)**2 + (sol2[1]-ey)**2 + (sol2[2]-ez)**2)

    branch_key = "arm_right_shoulder_pitch"
    if arm == "left":
        branch_key = "arm_left_shoulder_pitch"
    prev_choice = _prev_solution_branch.get(branch_key)
    branch_choice = _choose_solution_branch(prev_choice, dist1, dist2)
    _prev_solution_branch[branch_key] = branch_choice
    if branch_choice == 1:
        shoulderPitch = t1_1
    else:
        shoulderPitch = t1_2

    return shoulderPitch, shoulderRoll
        
# Function that returns the ElbowRoll (t4) given the angles t1, t2 and the wrist position (px, py, pz)
def calculate_theta4(t1, t2, px, py, pz, arm):
    l1 = -57.0
    l3 = 86.82
    l4 = 150
    d3 = 181.2
    z3 = 0.13

    if arm == 'right':
        l2 = -149.74
    elif arm == 'left':
        l2 = 149.74

    t2 = t2 - pi/2
    alpha = radians(9)

    term3 = (px - l1)*sin(t1) + (pz - l3)*cos(t1) - z3
    term4 = (pz-l3)*sin(t1)*sin(t2) + (py-l2)*cos(t2) - d3 - (px-l1)*sin(t2)*cos(t1)
    term2 = sin(alpha)*term3 + cos(alpha)*term4
    term1 = (1.0/l4)*term2      #IT IS IMPORTANT TO PUT .0 otherwise it rounds to 0

    if term1 > 1.0:
        theta4 = nan
    else:
        if arm == 'right':
            theta4 = arccos(term1)
        elif arm == 'left':
            theta4 = -arccos(term1)

    if arm == 'right':
        if theta4.imag != 0.0:
            theta4 = nan  
        elif theta4 > 1.58:
            theta4 = 1.57
        elif theta4 < 0.0087:
            theta4 = 0.0087
    elif arm == 'left':
        if theta4.imag != 0.0:
            theta4 = nan 
        elif theta4 > -0.0087: 
            theta4 = -0.0087
        elif theta4 < -1.58:
            theta4 = -1.57

    return theta4

# Function that returns the ElbowYaw (t3) given the angles t1, t2, t4 and the wrist position (px, py, pz)
def calculate_theta3(t1, t2, t4, px, py, pz, arm):
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

    t2 = t2 - pi/2
    alpha = radians(9)

    aterm = (d3 + d5*cos(alpha)*cos(t4) + cos(t1)*sin(t2)*(px-l1) - sin(t1)*sin(t2)*(pz-l3) - cos(t2)*(py-l2)) / (d5*sin(t4)*sin(alpha))
    bterm = (cos(t2)*sin(t1)*(pz-l3) + a3 - cos(t1)*cos(t2)*(px-l1) - sin(t2)*(py-l2)) / (d5*sin(t4))

    if aterm.imag == 0 and bterm.imag == 0:
        theta3 = arctan2(aterm, bterm)
        if theta3 < -2.1: 
            theta3 = -2.1
        elif theta3 > 2.1:
            theta3 = 2.1
    else:
        theta3 = nan

    return theta3

# Returns a continuous Pepper hand command in [0, 1] using wrist-tip distance.
def hand_open_close_right(wrist, tip, prev_dist=None):
    return _update_hand_command(RIGHT_HAND_STATE, wrist, tip)


# Returns a continuous Pepper hand command in [0, 1] using wrist-tip distance.
def hand_open_close_left(wrist, tip, prev_dist=None):
    return _update_hand_command(LEFT_HAND_STATE, wrist, tip)

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
    sol1 = get_torso_position(t2_1, hipRoll)
    sol2 = get_torso_position(t2_2, hipRoll)

    # Calculate the distance between the forward kinematics solutions and the desired position
    dist1 = sqrt((sol1[0]-px)**2 + (sol1[1]-py)**2 + (sol1[2]-pz)**2)
    dist2 = sqrt((sol2[0]-px)**2 + (sol2[1]-py)**2 + (sol2[2]-pz)**2)

    # Choose the t2 value that results in the smaller distance to the desired position
    prev_choice = _prev_solution_branch.get("torso_hip_pitch")
    branch_choice = _choose_solution_branch(prev_choice, dist1, dist2)
    _prev_solution_branch["torso_hip_pitch"] = branch_choice
    if branch_choice == 1:
        hipPitch = t2_1
    else:
        hipPitch = t2_2

    if isnan(hipPitch):
        kneePitch = nan
    else:
        t2_rounded = round(hipPitch, 2)
        kneePitch = pairs.get(t2_rounded, nan)

    # Update the previous hipPitch value if the current hipPitch is valid, otherwise use the previous value
    if not isnan(hipPitch):
        prev_hipPitch_new = round(hipPitch, 2)
    else:
        hipPitch = prev_hipPitch_new
        kneePitch = pairs.get(round(prev_hipPitch_new, 2), nan)

    return kneePitch, hipPitch, hipRoll
