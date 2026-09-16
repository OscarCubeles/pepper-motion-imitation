import server.common.kinematics.transformation_matrices as tm
import server.common.kinematics.inverse_kinematics as ik
import server.common.kinematics.workspace_fitting as workspace    
from numpy import pi, arctan, arctan2, arcsin, arccos, sqrt, sin, cos, transpose, radians, real, isnan, nan, linalg, array
import server.common.kinematics.constants as constants
from server.common.kinematics.kinematics_singleton import get_kinematics_config

ENABLE_MIRRORING_IMITATION = False
_prev_arm_solution_branch = {"right": None, "left": None}

def to_spherical(position):
    """Convert cartesian to spherical coordinates."""
    r = linalg.norm(position)
    if r < 1e-6:
        return 0.0, 0.0, 0.0
    phi = arctan2(position[1], position[0])
    cos_arg = position[2] / r
    if cos_arg > 1.0:
        cos_arg = 1.0
    elif cos_arg < -1.0:
        cos_arg = -1.0
    theta = arccos(cos_arg)
    return r, phi, theta

def to_cartesian(r, phi, theta):
    """Convert spherical to cartesian coordinates."""
    position = array([
        r * sin(theta) * cos(phi),
        r * sin(theta) * sin(phi),
        r * cos(theta)
    ])
    return position


def _get_head_angles_mode_aware(head_yaw, head_pitch):
	if ENABLE_MIRRORING_IMITATION:
		return -head_yaw, head_pitch
	return head_yaw, head_pitch

def _torso_hip_roll_command(hip_roll):
	# Anatomical mode uses the current sign convention (flip).
	# Mirroring mode intentionally inverts left/right torso motion.
	if ENABLE_MIRRORING_IMITATION:
		return hip_roll
	return -hip_roll


def compute_arm_targets(torso, shoulder, elbow, wrist, other_shoulder, arm, hand_tip, use_human_mode=False):
    """
    Compute arm joint angles (t1, t2, t3, t4).
    
    Args:
        use_human_mode: If True, compute human motion angles (no limits). If False, compute Pepper angles (with limits).
    """
    hand = None
    if arm == 'right':
        pepper_shoulder = array([-57, -149.74, 86.82])
    elif arm == 'left':
        pepper_shoulder = array([-57, 149.74, 86.82])
    
    # Step 1: Transform to Pepper's coordinate system
    torso = tm.set_joint_in_peppers_coordinates(torso)
    shoulder_new = tm.set_joint_in_peppers_coordinates(shoulder) - torso
    elbow_new = tm.set_joint_in_peppers_coordinates(elbow) - torso
    wrist_new = tm.set_joint_in_peppers_coordinates(wrist) - torso
    other_shoulder_new = tm.set_joint_in_peppers_coordinates(other_shoulder) - torso
    
    # Step 2: Create rotation matrix and rotate joints
    if arm == 'right':
        rot_Mat = tm.rotation_mat_arms(shoulder_new, other_shoulder_new)
    else:
        rot_Mat = tm.rotation_mat_arms(other_shoulder_new, shoulder_new)
    
    shoulder_new = rot_Mat.dot(shoulder_new)
    elbow_new = rot_Mat.dot(elbow_new) - shoulder_new
    wrist_new = rot_Mat.dot(wrist_new) - shoulder_new - elbow_new
    
    # Step 3: Scale segments
    elbow_scaled = 181.2 * elbow_new/linalg.norm(elbow_new)
    wrist_scaled = 150 * wrist_new/linalg.norm(wrist_new) + elbow_scaled + pepper_shoulder
    
    # Step 4: Fit to workspace
    t1, t2, elbow_fitted, wrist_fitted = workspace.get_elbow_wrist_fitted(elbow_scaled, wrist_scaled, arm, use_human_mode)
    
    # Step 5: Get t3 and t4
    elbowRoll = ik.calculate_theta4(t1, t2, wrist_fitted[0], wrist_fitted[1], wrist_fitted[2], arm) if not use_human_mode else ik.calculate_theta4_human(t1, t2, wrist_fitted[0], wrist_fitted[1], wrist_fitted[2], arm)
    elbowYaw = ik.calculate_theta3(t1, t2, elbowRoll, wrist_fitted[0], wrist_fitted[1], wrist_fitted[2], arm)
    
    t3 = elbowYaw
    t4 = elbowRoll

    if constants.ENABLE_HAND_TRACKING and hand_tip is not None:
        wrist = wrist / 1000 # undo scaling
        hand = ik.hand_open_close_left(wrist, hand_tip) # TODO: Crec que es indiferent posar left o right aquí, però cal revisar
        
    
    return t1, t2, t3, t4, hand


def compute_head_targets(torso, neck, nose, rshoulder, lshoulder):
	global prev_t1_head, prev_t2_head
	pepper_head_pos = array([-38.0, 0, 169.9])
	if nose[0] == 0.0 and nose[1] == 0.0 and nose[2] == 0.0:
		head_yaw, head_pitch = _get_head_angles_mode_aware(prev_t1_head, prev_t2_head)
		return list(constants.HEAD_JOINTS), [head_yaw, head_pitch]

	torso = tm.set_joint_in_peppers_coordinates(torso)
	rshoulder_new = tm.set_joint_in_peppers_coordinates(rshoulder) - torso
	lshoulder_new = tm.set_joint_in_peppers_coordinates(lshoulder) - torso
	neck_new = tm.set_joint_in_peppers_coordinates(neck) - torso
	nose_new = tm.set_joint_in_peppers_coordinates(nose) - torso

	rot_Mat = tm.rotation_mat(rshoulder_new, lshoulder_new)
	neck_new = rot_Mat.dot(neck_new)
	nose_new = rot_Mat.dot(nose_new) - neck_new

	norm = linalg.norm(nose_new)
	if norm < 1e-6:
		head_yaw, head_pitch = _get_head_angles_mode_aware(prev_t1_head, prev_t2_head)
		return list(constants.HEAD_JOINTS), [head_yaw, head_pitch]

	nose_scaled = 112.05 * nose_new/norm 
	r_eq, phi_eq, theta_eq = to_spherical(nose_scaled)
	nose_fitted = pepper_head_pos + to_cartesian(r_eq, phi_eq, theta_eq)
	[t1, t2] = ik.get_head_angles(nose_fitted[0], nose_fitted[1], nose_fitted[2])
	prev_t1_head = t1
	prev_t2_head = t2
	head_yaw, head_pitch = _get_head_angles_mode_aware(t1, t2)
	return list(constants.HEAD_JOINTS), [head_yaw, head_pitch]


def compute_torso_targets(spine_base, torso):
	spine_base = spine_base.copy()
	torso = torso.copy()
	spine_base[1] *= -1
	torso[1] *= -1

	torso = tm.set_joint_in_peppers_coordinates(torso)
	spine_base = tm.set_joint_in_peppers_coordinates(spine_base)
	torso_new = torso - spine_base
	hip_scaled = array([0, 0, 347])
	torso_scaled = 139 * torso_new/linalg.norm(torso_new) + hip_scaled
	[t1, t2, t3] = ik.get_torso_angles_new(torso_scaled[0], torso_scaled[1], torso_scaled[2])
	return list(constants.TORSO_JOINTS), [t1, t2, _torso_hip_roll_command(t3)]


# Higher margin for switching branches when both are valid, to avoid jitter from nearly-equal solutions
def choose_arm_solution_branch(arm, dist1, dist2):
    kinematics_config = get_kinematics_config()
    margin = kinematics_config.get_ik_branch_switch_margin()
    
    prev_choice = _prev_arm_solution_branch.get(arm)
    if not ik._is_finite_number(dist1) and not ik._is_finite_number(dist2):
        return prev_choice if prev_choice in (1, 2) else 1
    if not ik._is_finite_number(dist1):
        return 2
    if not ik._is_finite_number(dist2):
        return 1

    if prev_choice == 1:
        if dist2 + margin < dist1:
            return 2
        return 1
    if prev_choice == 2:
        if dist1 + margin < dist2:
            return 1
        return 2

    if dist1 <= dist2:
        return 1
    return 2


def set_arm_solution_branch(arm, branch_choice):
    """
    Set the previous arm solution branch for hysteresis.
    
    Args:
        arm: "right" or "left"
        branch_choice: The branch index (1 or 2) to store
    """
    if arm in _prev_arm_solution_branch:
        _prev_arm_solution_branch[arm] = branch_choice