from numpy import savetxt, dot, pi, linalg, array, transpose, zeros, arctan2, arccos, sin, cos, sqrt
import sys
# sys.path.append('../kinematics/')
import transformation_matrices as tfm
import inverse_kinematics_rangeupdate as ik
import forward_kinematics as fk
import hand_orientation as ho
import math_functions as mf
import pepper_commands as pc
from joint_constraints import (
	apply_nullspace_constraints,
	reset_nullspace_constraints_state,
)
import joint_constraints as jc

from datetime import datetime

try:
	import sys
	import os
	sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
	import client_visualization as client_vis
	CLIENT_VIS_AVAILABLE = True
except Exception:
	CLIENT_VIS_AVAILABLE = False

prev_t1_head = 0.0
prev_t2_head = 0.0

ENABLE_HAND_TRACKING = False
ENABLE_MIRRORING_IMITATION = False
ENABLE_HAND_ORIENTATION_TRACKING = True # Set to false for circular wrist rotation instead of hand orientation tracking
SINGULARITY_NORM_EPS = 1e-6
IK_BRANCH_SWITCH_MARGIN_MM = 10.0 # Margin to switch IK branches when both are valid, to avoid jitter from nearly-equal solutions


_right_wrist_yaw_phase = 0.0
_left_wrist_yaw_phase = 0.0
_WRIST_YAW_STEP_RAD = 0.08
_WRIST_YAW_MIN_RAD = -(104.5 * pi / 180.0)
_WRIST_YAW_MAX_RAD = (104.5 * pi / 180.0)
_last_right_wrist_yaw = 0.0
_last_left_wrist_yaw = 0.0


_prev_right_arm_targets = None
_prev_left_arm_targets = None
_prev_torso_targets = None
_prev_arm_solution_branch = {"right": None, "left": None}


def _is_finite_number(value):
	try:
		number = float(value)
	except Exception:
		return False
	if number != number:
		return False
	return number != float("inf") and number != float("-inf")


def _angles_are_valid(angles):
	for value in angles:
		if not _is_finite_number(value):
			return False
	return True


def _copy_arm_targets(targets):
	if targets is None:
		return [], [], []
	names, angles, speeds = targets
	return list(names), list(angles), list(speeds)


def _copy_torso_targets(targets):
	if targets is None:
		return list(pc.TORSO_JOINTS), [0.0, 0.0, 0.0]
	names, angles = targets
	return list(names), list(angles)


# Higher margin for switching branches when both are valid, to avoid jitter from nearly-equal solutions
def _choose_arm_solution_branch(arm, dist1, dist2):
	prev_choice = _prev_arm_solution_branch.get(arm)
	if not _is_finite_number(dist1) and not _is_finite_number(dist2):
		return prev_choice if prev_choice in (1, 2) else 1
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


def set_hand_tracking_enabled(enabled):
	global ENABLE_HAND_TRACKING
	ENABLE_HAND_TRACKING = bool(enabled)


def set_mirroring_imitation_enabled(enabled):
	global ENABLE_MIRRORING_IMITATION
	ENABLE_MIRRORING_IMITATION = bool(enabled)


def reset_ik_continuity_state():
	global prev_t1_head, prev_t2_head
	global _prev_right_arm_targets, _prev_left_arm_targets, _prev_torso_targets
	prev_t1_head = 0.0
	prev_t2_head = 0.0
	_prev_right_arm_targets = None
	_prev_left_arm_targets = None
	_prev_torso_targets = None
	_prev_arm_solution_branch["right"] = None
	_prev_arm_solution_branch["left"] = None
	try:
		ik.reset_branch_selection_state()
	except Exception:
		pass
	reset_nullspace_constraints_state()


def _get_head_angles_mode_aware(head_yaw, head_pitch):
	if ENABLE_MIRRORING_IMITATION:
		return -head_yaw, head_pitch
	return head_yaw, head_pitch


def _get_target_arm_side_and_angles(human_arm_side, t1, t2, t3, t4):
	if not ENABLE_MIRRORING_IMITATION:
		return human_arm_side, (t1, t2, t3, t4)

	# Mirroring maps human right->Pepper left and human left->Pepper right,
	# with the same sign convention used in the legacy mirroring pipeline.
	if human_arm_side == "right":
		return "left", (t1, -t2, -t3, -t4)
	return "right", (t1, -t2, -t3, -t4)


def _get_arm_joint_names_for_side(arm_side):
	if arm_side == "right":
		return list(pc.RIGHT_ARM_JOINTS)
	return list(pc.LEFT_ARM_JOINTS)


def _get_hand_joint_name_for_side(arm_side):
	if arm_side == "right":
		return "RHand"
	return "LHand"


def _extend_joint_targets(names_out, angles_out, names_in, angles_in, speeds_out=None, speed_fraction=None):
	names_out.extend(names_in)
	angles_out.extend(angles_in)
	if speeds_out is not None and speed_fraction is not None:
		speeds_out.extend([speed_fraction] * len(names_in))


def _torso_hip_roll_command(hip_roll):
	# Anatomical mode uses the current sign convention (flip).
	# Mirroring mode intentionally inverts left/right torso motion.
	if ENABLE_MIRRORING_IMITATION:
		return hip_roll
	return -hip_roll


# Function for scaling the joints used for calculating the angles (IK) for the head chain
def _compute_head_targets(torso, neck, nose, rshoulder, lshoulder):
	global prev_t1_head, prev_t2_head
	pepper_head_pos = array([0, 0, 169.9])
	if nose[0] == 0.0 and nose[1] == 0.0 and nose[2] == 0.0:
		head_yaw, head_pitch = _get_head_angles_mode_aware(prev_t1_head, prev_t2_head)
		return list(pc.HEAD_JOINTS), [head_yaw, head_pitch]

	torso = tfm.set_joint_in_peppers_coordinates(torso)
	rshoulder_new = tfm.set_joint_in_peppers_coordinates(rshoulder) - torso
	lshoulder_new = tfm.set_joint_in_peppers_coordinates(lshoulder) - torso
	neck_new = tfm.set_joint_in_peppers_coordinates(neck) - torso
	nose_new = tfm.set_joint_in_peppers_coordinates(nose) - torso

	rot_Mat = tfm.rotation_mat(rshoulder_new, lshoulder_new)
	neck_new = rot_Mat.dot(neck_new)
	nose_new = rot_Mat.dot(nose_new) - neck_new

	norm = linalg.norm(nose_new)
	if norm < 1e-6:
		head_yaw, head_pitch = _get_head_angles_mode_aware(prev_t1_head, prev_t2_head)
		return list(pc.HEAD_JOINTS), [head_yaw, head_pitch]

	nose_scaled = 112.05 * nose_new/norm 
	r_eq, phi_eq, theta_eq = to_spherical(nose_scaled)
	nose_fitted = pepper_head_pos + to_cartesian(r_eq, phi_eq, theta_eq)
	[t1, t2] = ik.get_head_angles(nose_fitted[0], nose_fitted[1], nose_fitted[2])
	prev_t1_head = t1
	prev_t2_head = t2
	head_yaw, head_pitch = _get_head_angles_mode_aware(t1, t2)
	return list(pc.HEAD_JOINTS), [head_yaw, head_pitch]


def head_scaling(torso, neck, nose, rshoulder, lshoulder):
	names, angles = _compute_head_targets(torso, neck, nose, rshoulder, lshoulder)
	pc.send_joint_targets_speed(names, angles)

# Function for scaling the joints used for calculating the angles (IK) for the right arm chain
def _compute_right_arm_targets(torso, shoulder, elbow, wrist, rtip, lshoulder):
	global _prev_right_arm_targets
	pepper_shoulder = array([-57, -149.74, 86.82])
	try:
		torso = tfm.set_joint_in_peppers_coordinates(torso)
		shoulder_new = tfm.set_joint_in_peppers_coordinates(shoulder) - torso
		elbow_new = tfm.set_joint_in_peppers_coordinates(elbow) - torso
		wrist_new = tfm.set_joint_in_peppers_coordinates(wrist) - torso
		lshoulder_new = tfm.set_joint_in_peppers_coordinates(lshoulder) - torso

		rot_Mat = tfm.rotation_mat_arms(shoulder_new, lshoulder_new)
		shoulder_new = rot_Mat.dot(shoulder_new)
		elbow_new = rot_Mat.dot(elbow_new) - shoulder_new
		wrist_new = rot_Mat.dot(wrist_new) - shoulder_new - elbow_new

		elbow_norm = linalg.norm(elbow_new)
		wrist_norm = linalg.norm(wrist_new)
		if (
			not _is_finite_number(elbow_norm) or
			not _is_finite_number(wrist_norm) or
			elbow_norm <= SINGULARITY_NORM_EPS or
			wrist_norm <= SINGULARITY_NORM_EPS
		):
			return _copy_arm_targets(_prev_right_arm_targets)

		elbow_scaled = 181.2 * elbow_new / elbow_norm
		wrist_scaled = 150 * wrist_new / wrist_norm + elbow_scaled + pepper_shoulder

		[t1, t2, elbow_fitted, wrist_fitted] = get_elbow_wrist_fitted(elbow_scaled, wrist_scaled, 'right')
		[t3, t4] = ik.get_arm_t3t4_angles(t1, t2, wrist_fitted[0], wrist_fitted[1], wrist_fitted[2], 'right')
	except Exception:
		return _copy_arm_targets(_prev_right_arm_targets)

	if not _angles_are_valid([t1, t2, t3, t4]):
		return _copy_arm_targets(_prev_right_arm_targets)

	target_arm_side, (cmd_t1, cmd_t2, cmd_t3, cmd_t4) = _get_target_arm_side_and_angles(
		"right", t1, t2, t3, t4
	)
	names = []
	angles = []
	speeds = []
	if (
		ENABLE_HAND_TRACKING
		and rtip is not None
		and linalg.norm(rtip) > 1e-6
	):
		hand = ik.hand_open_close_right(wrist, rtip)
		names.append(_get_hand_joint_name_for_side(target_arm_side))
		angles.append(hand)
		speeds.append(pc.HAND_SPEED_FRACTION)
	if not (t1 == 0.0 and t2 == 0.0 and t3 == 0.0 and t4 == 0.0):
		_extend_joint_targets(
			names,
			angles,
			_get_arm_joint_names_for_side(target_arm_side),
			[cmd_t1, cmd_t2, cmd_t3, cmd_t4],
			speeds,
			pc.ARM_SPEED_FRACTION,
		)
	if angles:
		_prev_right_arm_targets = (list(names), list(angles), list(speeds))
	return names, angles, speeds


def rightArm_scaling(torso, shoulder, elbow, wrist, rtip, lshoulder):
	names, angles, _speeds = _compute_right_arm_targets(torso, shoulder, elbow, wrist, rtip, lshoulder)
	pc.send_joint_targets_speed(names, angles)


# Function for scaling the joints used for calculating the angles (IK) for the left arm chain
def _compute_left_arm_targets(torso, shoulder, elbow, wrist, ltip, rshoulder):
	global _prev_left_arm_targets
	pepper_shoulder = array([-57, 149.74, 86.82])
	try:
		torso = tfm.set_joint_in_peppers_coordinates(torso)
		shoulder_new = tfm.set_joint_in_peppers_coordinates(shoulder) - torso
		elbow_new = tfm.set_joint_in_peppers_coordinates(elbow) - torso
		wrist_new = tfm.set_joint_in_peppers_coordinates(wrist) - torso
		rshoulder_new = tfm.set_joint_in_peppers_coordinates(rshoulder) - torso
		
		rot_Mat = tfm.rotation_mat_arms(rshoulder_new, shoulder_new)
		shoulder_new = rot_Mat.dot(shoulder_new)
		elbow_new = rot_Mat.dot(elbow_new) - shoulder_new
		wrist_new = rot_Mat.dot(wrist_new) - shoulder_new - elbow_new

		elbow_norm = linalg.norm(elbow_new)
		wrist_norm = linalg.norm(wrist_new)
		if (
			not _is_finite_number(elbow_norm) or
			not _is_finite_number(wrist_norm) or
			elbow_norm <= SINGULARITY_NORM_EPS or
			wrist_norm <= SINGULARITY_NORM_EPS
		):
			return _copy_arm_targets(_prev_left_arm_targets)

		elbow_scaled = 181.2 * elbow_new / elbow_norm
		wrist_scaled = 150 * wrist_new / wrist_norm + elbow_scaled + pepper_shoulder

		[t1, t2, elbow_fitted, wrist_fitted] = get_elbow_wrist_fitted(elbow_scaled, wrist_scaled, 'left')
		[t3, t4] = ik.get_arm_t3t4_angles(t1, t2, wrist_fitted[0], wrist_fitted[1], wrist_fitted[2], 'left')
	except Exception:
		return _copy_arm_targets(_prev_left_arm_targets)

	if not _angles_are_valid([t1, t2, t3, t4]):
		return _copy_arm_targets(_prev_left_arm_targets)

	target_arm_side, (cmd_t1, cmd_t2, cmd_t3, cmd_t4) = _get_target_arm_side_and_angles(
		"left", t1, t2, t3, t4
	)
	names = []
	angles = []
	speeds = []
	if (
		ENABLE_HAND_TRACKING
		and ltip is not None
		and linalg.norm(ltip) > 1e-6
	):
		hand = ik.hand_open_close_left(wrist, ltip)
		

		names.append(_get_hand_joint_name_for_side(target_arm_side))
		angles.append(hand)
		speeds.append(pc.HAND_SPEED_FRACTION)
	if not (t1 == 0.0 and t2 == 0.0 and t3 == 0.0 and t4 == 0.0):
		_extend_joint_targets(
			names,
			angles,
			_get_arm_joint_names_for_side(target_arm_side),
			[cmd_t1, cmd_t2, cmd_t3, cmd_t4],
			speeds,
			pc.ARM_SPEED_FRACTION,
		)
	if angles:
		_prev_left_arm_targets = (list(names), list(angles), list(speeds))
	return names, angles, speeds


def leftArm_scaling(torso, shoulder, elbow, wrist, ltip, rshoulder):
	names, angles, _speeds = _compute_left_arm_targets(torso, shoulder, elbow, wrist, ltip, rshoulder)
	pc.send_joint_targets_speed(names, angles)
	   
def left_hand_values(lwrist, ltip):
	hand = ik.hand_open_close_left(lwrist, ltip)
	pc.send_leftHand_values(hand)

def right_hand_values(rwrist, rtip):
	hand = ik.hand_open_close_right(rwrist, rtip)
	pc.send_rightHand_values(hand)

# Function for scaling the joints used for calculating the angles (IK) for the torso chain
def _compute_torso_targets(spine_base, torso):
	global _prev_torso_targets
	spine_base = spine_base.copy()
	torso = torso.copy()
	spine_base[1] *= -1
	torso[1] *= -1

	try:
		torso = tfm.set_joint_in_peppers_coordinates(torso)
		spine_base = tfm.set_joint_in_peppers_coordinates(spine_base)
		torso_new = torso - spine_base
		torso_norm = linalg.norm(torso_new)
		if (not _is_finite_number(torso_norm)) or torso_norm <= SINGULARITY_NORM_EPS:
			return _copy_torso_targets(_prev_torso_targets)

		hip_scaled = array([0, 0, 347])
		torso_scaled = 139 * torso_new / torso_norm + hip_scaled
		[t1, t2, t3] = ik.get_torso_angles_new(torso_scaled[0], torso_scaled[1], torso_scaled[2])
	except Exception:
		return _copy_torso_targets(_prev_torso_targets)

	angles = [t1, t2, _torso_hip_roll_command(t3)]
	if not _angles_are_valid(angles):
		return _copy_torso_targets(_prev_torso_targets)

	_prev_torso_targets = (list(pc.TORSO_JOINTS), list(angles))
	return list(pc.TORSO_JOINTS), angles


def torso_scaling(spine_base, torso):
	names, angles = _compute_torso_targets(spine_base, torso)
	pc.send_joint_targets_speed(names, angles)


def build_frame_joint_targets(torso, neck, nose, rshoulder, relbow, rwrist, rtip,
							  lshoulder, lelbow, lwrist, ltip, spine_base):
	names = []
	angles = []
	speeds = []
	head_names, head_angles = _compute_head_targets(torso, neck, nose, rshoulder, lshoulder)
	_extend_joint_targets(names, angles, head_names, head_angles, speeds, pc.HEAD_SPEED_FRACTION)
	right_names, right_angles, right_speeds = _compute_right_arm_targets(torso, rshoulder, relbow, rwrist, rtip, lshoulder)
	names.extend(right_names)
	angles.extend(right_angles)
	speeds.extend(right_speeds)
	left_names, left_angles, left_speeds = _compute_left_arm_targets(torso, lshoulder, lelbow, lwrist, ltip, rshoulder)
	names.extend(left_names)
	angles.extend(left_angles)
	speeds.extend(left_speeds)
	torso_names, torso_angles = _compute_torso_targets(spine_base, torso)
	_extend_joint_targets(names, angles, torso_names, torso_angles, speeds, pc.TORSO_SPEED_FRACTION)
	apply_nullspace_constraints(names, angles)
	return names, angles, speeds



# Function fitting the spherical coordinates for the elbow and wrist end-effector
def get_elbow_wrist_fitted(elbow, wrist, arm):
	# Check the spherical coordinates
	# Check which arm for the rotation of the sphere
	if arm == 'right':
		rot_x = mf.rx_calc(pi/2, 3)
	elif arm == 'left':
		rot_x = mf.rx_calc(-pi/2, 3)

	# This rotation is necessary fot setting the origin and direction of the sphere
	elbow_spherical = transpose(rot_x).dot(elbow)
	# Formulas for conversion from cartesian to spherical coordinates
	er_eq, ephi_eq, etheta_eq = to_spherical(elbow_spherical)
	# Checking whether the sphercial coordinates are within Pepper's
	# print("BEFORE " + arm + " elbow: phi " + str(ephi_eq) + " theta " + str(etheta_eq))
	# workspace represented as sphere
	if etheta_eq < 0.0:	 
		etheta_eq = 0.0079	 
	elif etheta_eq > 1.48:	
		etheta_eq = 1.48

	if etheta_eq > 0.06 and etheta_eq < 0.165:
		if ephi_eq < -3.15:
			ephi_eq = -3.14
		elif ephi_eq > 3.15:
			ephi_eq = 3.14

		# Go back to cartesian coordinates
		elbow_fitted = rot_x.dot(to_cartesian(er_eq, ephi_eq, etheta_eq))
		# Get the fitted wrist position and the t1 and t2
		t1, t2, wrist_fitted = get_wrist_fitted(elbow_fitted, wrist, elbow, arm)

	# else:
	if ephi_eq < -2.1 or ephi_eq > 2.1:	
		# print(arm + " phi elbow for both values")  
		ephi_eq1 = -2.09
		# Go back to cartesian coordinates
		elbow_fitted1 = rot_x.dot(to_cartesian(er_eq, ephi_eq1, etheta_eq))

		ephi_eq2 = 2.09
		# Go back to cartesian coordinates
		elbow_fitted2 = rot_x.dot(to_cartesian(er_eq, ephi_eq2, etheta_eq))
					
		# Check the wrist spherical coordinates
		t1_1, t2_1, wrist_fitted1 = get_wrist_fitted(elbow_fitted1, wrist, elbow, arm)
		t1_2, t2_2, wrist_fitted2 = get_wrist_fitted(elbow_fitted2, wrist, elbow, arm)
		# Caluclate the distance from the new fitted wrist position to the scaled position
		dist1 = sqrt((wrist[0]-wrist_fitted1[0])**2 + (wrist[1]-wrist_fitted1[1])**2 + (wrist[2]-wrist_fitted1[2])**2)
		dist2 = sqrt((wrist[0]-wrist_fitted2[0])**2 + (wrist[1]-wrist_fitted2[1])**2 + (wrist[2]-wrist_fitted2[2])**2)
		# Keep switching conservative near nearly-equal solutions.
		branch_choice = _choose_arm_solution_branch(arm, dist1, dist2)
		_prev_arm_solution_branch[arm] = branch_choice
		if branch_choice == 1: 
			wrist_fitted = wrist_fitted1
			elbow_fitted = elbow_fitted1
			t1 = t1_1
			t2 = t2_1
		else:
			wrist_fitted = wrist_fitted2
			elbow_fitted = elbow_fitted2
			t1 = t1_2
			t2 = t2_2

	else:
		# Go back to cartesian coordinates
		elbow_fitted = rot_x.dot(to_cartesian(er_eq, ephi_eq, etheta_eq))
		# Get the fitted wrist position and the t1 and t2
		t1, t2, wrist_fitted = get_wrist_fitted(elbow_fitted, wrist, elbow, arm)

	# print("AFTER " + arm + " elbow: phi " + str(ephi_eq) + " theta " + str(etheta_eq))

	return t1, t2, elbow_fitted, wrist_fitted

# Function fitting the spherical coordinates for the wrist end-effector
def get_wrist_fitted(elbow_fitted, wrist, elbow, arm):
	# Check which arm for the shoulder position
	if arm == 'right':
		pepper_shoulder = array([-57, -149.74, 86.82])
	elif arm == 'left':
		pepper_shoulder = array([-57, 149.74, 86.82])
	# Attach elbow to shoulder to calculate the t1 and t2 below
	elbow_new = pepper_shoulder + elbow_fitted
	# Calculate the partial angles using the fitted elbow positon
	[t1, t2] = ik.get_arm_partial_angles(elbow_new[0], elbow_new[1], elbow_new[2], arm)

	# Calculate the transformation matrix for the spherical coordinates
	Mat = tfm.trans_elbow_workspace(t1, t2, elbow_fitted, pepper_shoulder, arm)
	rot_x = mf.rx_calc(pi/2, 3)
	rot_y = mf.ry_calc(pi/2, 3)
	wrist = wrist - elbow - pepper_shoulder
	# This rotation is necessary fot setting the origin and direction of the sphere
	wrist_spherical = transpose(rot_y).dot(transpose(rot_x).dot(transpose(Mat).dot(wrist)))

	# Formulas for conversion from cartesian to spherical coordinates
	wr_eq, wphi_eq, wtheta_eq = to_spherical(wrist_spherical)
	# print("BEFORE " + arm + " wrist: phi " + str(wphi_eq) + " theta " + str(wtheta_eq))
	# Checking whether the sphercial coordinates are within Pepper's
	# workspace represented as sphere
	if wtheta_eq < 0.008:
		wtheta_eq = 0.0087
	elif wtheta_eq > 1.56:
		wtheta_eq = 1.55

	if  wphi_eq < -2.09:
		wphi_eq = -2.08
	elif  wphi_eq > 2.06:
		wphi_eq = 2.05

	wrist_fitted = Mat.dot(rot_x.dot(rot_y.dot(to_cartesian(wr_eq, wphi_eq, wtheta_eq)))) + elbow_new

	return t1, t2, wrist_fitted

# Function to convert from cartesian to spherical
def to_spherical(position):
	
	r = linalg.norm(position)
	if r <= SINGULARITY_NORM_EPS:
		return 0.0, 0.0, 0.0
	phi = arctan2(position[1], position[0])
	cos_arg = position[2] / r
	if cos_arg > 1.0:
		cos_arg = 1.0
	elif cos_arg < -1.0:
		cos_arg = -1.0
	theta = arccos(cos_arg)

	return r, phi, theta

# Function to convert from spherical to cartesian
def to_cartesian(r, phi, theta):

	position = array([r*sin(theta)*cos(phi), \
				r*sin(theta)*sin(phi), \
				r*cos(theta)])

	return position

# print(get_time())

def _compute_right_arm_targets_hand(torso, shoulder, elbow, wrist, rtip, lshoulder,
								rhandwrist, rthumbcmc, rindexmcp, rpinkymcp,
								rserverpalmx=None, rserverpalmy=None, rserverpalmz=None,
								right_discrete_orientation=None):
	"""
	Compute right arm targets with hand orientation tracking support.
	
	Args:
		Standard arm parameters plus right hand keypoints for orientation tracking.
		rserverpalmx, rserverpalmy, rserverpalmz: Server-computed right palm vectors (optional).
		right_discrete_orientation: Discrete orientation label (optional) for discrete mode.
	"""
	names, angles, speeds = _compute_right_arm_targets(torso, shoulder, elbow, wrist, rtip, lshoulder)
	if angles:
		target_arm_side, _ = _get_target_arm_side_and_angles("right", 0.0, 0.0, 0.0, 0.0)
		wrist_joint = "RWristYaw" if target_arm_side == "right" else "LWristYaw"
		
		if ENABLE_HAND_ORIENTATION_TRACKING:
			# Compute wrist yaw from hand orientation tracking
			wrist_angle = ho.compute_wristyaw_from_angles(
				names, angles,
				rhandwrist, rthumbcmc, rindexmcp, rpinkymcp,
				"right",
				server_palm_x=rserverpalmx, server_palm_y=rserverpalmy, server_palm_z=rserverpalmz,
				enable_validation_log=True,
				discrete_orientation=right_discrete_orientation
			)
		else:
			# Use circular rotation (original behavior)
			wrist_angle = _advance_wrist_yaw_phase(target_arm_side)
		
		# Send it to pepper
		names.append(wrist_joint)
		angles.append(wrist_angle)
		speeds.append(pc.ARM_SPEED_FRACTION)
	return names, angles, speeds


def _advance_wrist_yaw_phase(side):
	global _right_wrist_yaw_phase, _left_wrist_yaw_phase
	if side == "right":
		_right_wrist_yaw_phase += _WRIST_YAW_STEP_RAD
		if _right_wrist_yaw_phase > _WRIST_YAW_MAX_RAD:
			_right_wrist_yaw_phase = _WRIST_YAW_MIN_RAD
		return _right_wrist_yaw_phase
	_left_wrist_yaw_phase += _WRIST_YAW_STEP_RAD
	if _left_wrist_yaw_phase > _WRIST_YAW_MAX_RAD:
		_left_wrist_yaw_phase = _WRIST_YAW_MIN_RAD
	return _left_wrist_yaw_phase

def build_frame_joint_targets_hand(torso, neck, nose, rshoulder, relbow, rwrist, rtip,
								  lshoulder, lelbow, lwrist, ltip, spine_base, rhandwrist, 
								  rthumbcmc, rindexmcp, rpinkymcp, lhandwrist, lthumbcmc, lindexmcp, lpinkymcp,
								  rserverpalmx=None, rserverpalmy=None, rserverpalmz=None,
								  lserverpalmx=None, lserverpalmy=None, lserverpalmz=None,
								  right_discrete_orientation=None, left_discrete_orientation=None):
	"""
	Build joint targets for complete frame including hand orientation tracking.
	
	Args:
		right_discrete_orientation: Discrete label for right hand (optional)
		left_discrete_orientation: Discrete label for left hand (optional)
	"""
	names = []
	angles = []
	speeds = []

	head_names, head_angles = _compute_head_targets(torso, neck, nose, rshoulder, lshoulder)
	_extend_joint_targets(names, angles, head_names, head_angles, speeds, pc.HEAD_SPEED_FRACTION)
	right_names, right_angles, right_speeds = _compute_right_arm_targets_hand(
		torso,
		rshoulder,
		relbow,
		rwrist,
		rtip,
		lshoulder,
		rhandwrist,
		rthumbcmc,
		rindexmcp,
		rpinkymcp,
		rserverpalmx, rserverpalmy, rserverpalmz,
		right_discrete_orientation
	)
	names.extend(right_names)
	angles.extend(right_angles)
	speeds.extend(right_speeds)
	left_names, left_angles, left_speeds = _compute_left_arm_targets_hand(
		torso,
		lshoulder,
		lelbow,
		lwrist,
		ltip,
		rshoulder,
		lhandwrist,
		lthumbcmc,
		lindexmcp,
		lpinkymcp,
		lserverpalmx, lserverpalmy, lserverpalmz,
		left_discrete_orientation
	)
	names.extend(left_names)
	angles.extend(left_angles)
	speeds.extend(left_speeds)
	torso_names, torso_angles = _compute_torso_targets(spine_base, torso)
	_extend_joint_targets(names, angles, torso_names, torso_angles, speeds, pc.TORSO_SPEED_FRACTION)
	jc.apply_nullspace_constraints(names, angles)
	
	# Extract wrist yaw angles for visualization
	right_wrist_yaw = _extract_wrist_yaw_from_targets(names, angles, 'right')
	left_wrist_yaw = _extract_wrist_yaw_from_targets(names, angles, 'left')
	
	# Build and display visualization if available
	#if CLIENT_VIS_AVAILABLE:
	#	try:
	#		vis_payload = _build_hand_vis_payload(
	#			rhandwrist, rthumbcmc, rindexmcp, rpinkymcp,
	#			lhandwrist, lthumbcmc, lindexmcp, lpinkymcp,
	#			right_wrist_yaw, left_wrist_yaw
	#		)
	#		client_vis.update_hand_orientation_debug(vis_payload)
	#	except Exception as e:
	#		print("Warning: Hand visualization update failed: {0}".format(e))
	
	return names, angles, speeds


def _compute_left_arm_targets_hand(torso, shoulder, elbow, wrist, ltip, rshoulder,
								lhandwrist, lthumbcmc, lindexmcp, lpinkymcp,
								lserverpalmx=None, lserverpalmy=None, lserverpalmz=None,
								left_discrete_orientation=None):
	"""
	Compute left arm targets with hand orientation tracking support.
	
	Args:
		Standard arm parameters plus left hand keypoints for orientation tracking.
		lserverpalmx, lserverpalmy, lserverpalmz: Server-computed left palm vectors (optional).
		left_discrete_orientation: Discrete orientation label (optional) for discrete mode.
	"""
	names, angles, speeds = _compute_left_arm_targets(torso, shoulder, elbow, wrist, ltip, rshoulder)
	if angles:
		target_arm_side, _ = _get_target_arm_side_and_angles("left", 0.0, 0.0, 0.0, 0.0)
		wrist_joint = "RWristYaw" if target_arm_side == "right" else "LWristYaw"
		
		if ENABLE_HAND_ORIENTATION_TRACKING:
			# Compute wrist yaw from hand orientation tracking
			wrist_angle = ho.compute_wristyaw_from_angles(
				names, angles,
				lhandwrist, lthumbcmc, lindexmcp, lpinkymcp,
				"left",
				server_palm_x=lserverpalmx, server_palm_y=lserverpalmy, server_palm_z=lserverpalmz,
				enable_validation_log=True,
				discrete_orientation=left_discrete_orientation
			)
		else:
			# Use circular rotation (original behavior)
			wrist_angle = _advance_wrist_yaw_phase(target_arm_side)
		
		names.append(wrist_joint)
		angles.append(wrist_angle)
		speeds.append(pc.ARM_SPEED_FRACTION)
	return names, angles, speeds
	   
def left_hand_values(lwrist, ltip):
	hand = ik.hand_open_close_left(lwrist, ltip)
	pc.send_leftHand_values(hand)

def right_hand_values(rwrist, rtip):
	hand = ik.hand_open_close_right(rwrist, rtip)
	pc.send_rightHand_values(hand)


def _extract_wrist_yaw_from_targets(names, angles, arm_side):
	"""
	Extract WristYaw angle from joint names and angles lists for the given arm.
	
	Args:
		names: List of joint names
		angles: List of corresponding angles
		arm_side: 'right' or 'left'
	
	Returns:
		wrist_yaw_angle (float) or None if not found
	"""
	try:
		wrist_joint_name = "RWristYaw" if arm_side == "right" else "LWristYaw"
		for i, name in enumerate(names):
			if wrist_joint_name in str(name):
				return angles[i]
	except Exception:
		pass
	return None


def _build_hand_vis_payload(rhandwrist, rthumbcmc, rindexmcp, rpinkymcp,
							 lhandwrist, lthumbcmc, lindexmcp, lpinkymcp,
							 right_wrist_yaw, left_wrist_yaw):
	"""
	Build visualization payload for client hand orientation debug display.
	
	Args:
		Hand keypoints for right and left (8 points total)
		Wrist yaw angles (in radians) sent to robot
	
	Returns:
		dict with 'right' and 'left' hand blocks for visualization
	"""
	try:
		payload = {'right': {}, 'left': {}}
		
		# Process right hand
		rhand_keypoints = {
			0: rhandwrist,
			1: rthumbcmc,
			5: rindexmcp,
			17: rpinkymcp,
		}
		
		# Check if any valid keypoint exists
		r_valid = any(kp is not None and sum(abs(x) for x in kp) > 1e-6 for kp in rhand_keypoints.values())
		
		if r_valid:
			# Compute orientation from received keypoints
			r_yaw, r_pitch, r_roll, r_orient_valid = client_vis._compute_orientation_from_keypoints(rhand_keypoints)
			
			payload['right'] = {
				'valid': r_orient_valid,
				'keypoints': rhand_keypoints,
				'received_ypr': (r_yaw, r_pitch, r_roll),
				'wrist_yaw_cmd': right_wrist_yaw if right_wrist_yaw is not None else 0.0,
			}
		else:
			payload['right'] = {'valid': False}
		
		# Process left hand
		lhand_keypoints = {
			0: lhandwrist,
			1: lthumbcmc,
			5: lindexmcp,
			17: lpinkymcp,
		}
		
		l_valid = any(kp is not None and sum(abs(x) for x in kp) > 1e-6 for kp in lhand_keypoints.values())
		
		if l_valid:
			# Compute orientation from received keypoints
			l_yaw, l_pitch, l_roll, l_orient_valid = client_vis._compute_orientation_from_keypoints(lhand_keypoints)
			
			payload['left'] = {
				'valid': l_orient_valid,
				'keypoints': lhand_keypoints,
				'received_ypr': (l_yaw, l_pitch, l_roll),
				'wrist_yaw_cmd': left_wrist_yaw if left_wrist_yaw is not None else 0.0,
			}
		else:
			payload['left'] = {'valid': False}
		
		return payload
		
	except Exception as e:
		print("Error building hand visualization payload: {0}".format(e))
		return {'right': {'valid': False}, 'left': {'valid': False}}