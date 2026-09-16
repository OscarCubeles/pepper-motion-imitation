from numpy import savetxt, dot, pi, linalg, array, transpose, zeros, arctan2, arccos, sin, cos, sqrt
import sys
sys.path.append('../kinematics_baseline/')
import transformation_matrices as tfm
import inverse_kinematics_rangeupdate as ik
import forward_kinematics as fk
import math_functions as mf
import pepper_commands as pc
from datetime import datetime

prev_dist_right = 0.0
prev_dist_left = 0.0

prev_t1_head = 0.0
prev_t2_head = 0.0


# Function for scaling the joints used for calculating the angles (IK) for the head chain
def head_scaling(torso, neck, nose, rshoulder, lshoulder):
	global prev_t1_head, prev_t2_head
	# Pepper bottom camera pos in the torso frame
	pepper_head_pos = array([-38.0, 0, 169.9])

	if nose[0] == 0.0 and nose[1] == 0.0 and nose[2] == 0.0:
		pc.send_head_values_speed(prev_t1_head, prev_t2_head)
	else:
		# Setting the orgin to be the torso joint
		torso = tfm.set_joint_in_peppers_coordinates(torso)
		rshoulder_new = tfm.set_joint_in_peppers_coordinates(rshoulder) - torso
		lshoulder_new = tfm.set_joint_in_peppers_coordinates(lshoulder) - torso
		neck_new = tfm.set_joint_in_peppers_coordinates(neck) - torso
		nose_new = tfm.set_joint_in_peppers_coordinates(nose) - torso #- neck_new

		# take into consideration the rotation of the human body with respect to the camera 
		rot_Mat = tfm.rotation_mat(rshoulder_new, lshoulder_new)
		neck_new = rot_Mat.dot(neck_new) #- torso 
		nose_new = rot_Mat.dot(nose_new) - neck_new # - torso

		# Scale the nose position to fit pepper
		# 112.05 is the length from pepper head joints to the bottom camera
		nose_scaled = 112.05 * nose_new/linalg.norm(nose_new) 
		# calculate the spherical cooridnates of the nose positon
		r_eq, phi_eq, theta_eq = to_spherical(nose_scaled)
		# check if the spherical coordinates are within the workspace if not 
		# map them to the closest point
		if theta_eq < 0.3:	
			theta_eq = 0.3	
		elif theta_eq > 1.45:  
			theta_eq = 1.45	

		if phi_eq < -2.1:
			phi_eq = -2.1	
		elif phi_eq > 2.1:
			phi_eq = 2.1

		# converted fitted nose spherical coordinates in cartesian position
		nose_fitted = pepper_head_pos + to_cartesian(r_eq, phi_eq, theta_eq)
		# use the scaled and fitted nose position to calculate the head angles
		[t1, t2] = ik.get_head_angles(nose_fitted[0], nose_fitted[1], nose_fitted[2])
		# send the head angles as motion commands
		pc.send_head_values_speed(-t1, t2)

		prev_t1_head = t1
		prev_t2_head = t2

# Function for scaling the joints used for calculating the angles (IK) for the right arm chain
def rightArm_scaling(torso, shoulder, elbow, wrist, rthumb, rtip, lshoulder):
	global prev_dist_right

	pepper_shoulder = array([-57, -149.74, 86.82])
	# Setting the joints in Pepper cooridnates
	torso = tfm.set_joint_in_peppers_coordinates(torso)
	shoulder_new = tfm.set_joint_in_peppers_coordinates(shoulder) - torso
	elbow_new = tfm.set_joint_in_peppers_coordinates(elbow) - torso #- shoulder_new
	wrist_new = tfm.set_joint_in_peppers_coordinates(wrist) - torso #- shoulder_new - elbow_new
	lshoulder_new = tfm.set_joint_in_peppers_coordinates(lshoulder) - torso

	# Orientation with respect to the Kinect camera
	rot_Mat = tfm.rotation_mat_arms(shoulder_new, lshoulder_new)
	shoulder_new = rot_Mat.dot(shoulder_new)
	elbow_new = rot_Mat.dot(elbow_new) - shoulder_new
	wrist_new = rot_Mat.dot(wrist_new) - shoulder_new - elbow_new

	# Scale the elbow joint to fit Pepper shoulder-elbow length 
	elbow_scaled = 181.2 * elbow_new/linalg.norm(elbow_new)
	# Scale the wrist joint to fit Pepper elbow-wrist length
	wrist_scaled = 150 * wrist_new/linalg.norm(wrist_new) + elbow_scaled + pepper_shoulder

	# Fit the elbow and wrist position in Pepper's workspace
	[t1, t2, elbow_fitted, wrist_fitted] = get_elbow_wrist_fitted(elbow_scaled, wrist_scaled, 'right')	

	# Caluclate the other two angles using the fitted wrist position
	[t3, t4] = ik.get_arm_t3t4_angles(t1, t2, wrist_fitted[0], wrist_fitted[1], wrist_fitted[2], 'right')

	# Calculate the distance between the thumb and the tip of the hand 
	hand = ik.hand_open_close_right(rthumb, rtip, prev_dist_right)
	prev_dist_right = hand
	# Send the calculated angles as motion commands
	if t1 == 0.0 and t2 == 0.0 and t3 == 0.0 and t4 == 0.0:
	   	pc.send_to_stand()
	else:
		pc.send_leftArm_values_speed(t1, -t2, -t3, -t4, hand)
		# pos_sensed[1, :], pos_sensed[2, :], angles_sensed[2:7,0], angles_speed[2:7,0]  = pc.send_leftArm_values_speed(t1, -t2, -t3, -t4, hand)
		# pc.send_rightArm_values(t1, t2, t3, t4, hand)
		# pc.send_rightArm_values_speed(t1, t2, t3, t4)


# Function for scaling the joints used for calculating the angles (IK) for the left arm chain
def leftArm_scaling(torso, shoulder, elbow, wrist, lthumb, ltip, rshoulder):
	global prev_dist_left

	pepper_shoulder = array([-57, 149.74, 86.82])
	# Setting the joints in Pepper cooridnates
	torso = tfm.set_joint_in_peppers_coordinates(torso)
	shoulder_new = tfm.set_joint_in_peppers_coordinates(shoulder) - torso
	elbow_new = tfm.set_joint_in_peppers_coordinates(elbow) - torso #- shoulder_new
	wrist_new = tfm.set_joint_in_peppers_coordinates(wrist) - torso #- shoulder_new - elbow_new
	rshoulder_new = tfm.set_joint_in_peppers_coordinates(rshoulder) - torso
	
	# Orientation with respect to the Kinect camera
	rot_Mat = tfm.rotation_mat_arms(rshoulder_new, shoulder_new)
	shoulder_new = rot_Mat.dot(shoulder_new)
	elbow_new = rot_Mat.dot(elbow_new) - shoulder_new
	wrist_new = rot_Mat.dot(wrist_new) - shoulder_new - elbow_new

	# Scale the elbow joint to fit Pepper shoulder-elbow length 
	elbow_scaled = 181.2 * elbow_new/linalg.norm(elbow_new)
	# Scale the wrist joint to fit Pepper elbow-wrist length
	wrist_scaled = 150 * wrist_new/linalg.norm(wrist_new) + elbow_scaled + pepper_shoulder

	# Fit the elbow and wrist position in Pepper's workspace
	[t1, t2, elbow_fitted, wrist_fitted] = get_elbow_wrist_fitted(elbow_scaled, wrist_scaled, 'left')

	# Caluclate the other two angles using the fitted wrist position
	[t3, t4] = ik.get_arm_t3t4_angles(t1, t2, wrist_fitted[0], wrist_fitted[1], wrist_fitted[2], 'left')

	# Calculate the distance between the thumb and the tip of the hand 
	hand = ik.hand_open_close_left(lthumb, ltip, prev_dist_left)
	prev_dist_left = hand

	# Send the calculated angles as motion commands
	if t1 == 0.0 and t2 == 0.0 and t3 == 0.0 and t4 == 0.0:
		pc.send_to_stand()
	else:
		pc.send_rightArm_values_speed(t1, -t2, -t3, -t4, hand)
		# pos_sensed[3, :], pos_sensed[4, :], angles_sensed[7:12,0], angles_speed[7:12,0]  = pc.send_rightArm_values_speed(t1, -t2, -t3, -t4, hand)
		# pc.send_leftArm_values(t1, t2, t3, t4, hand)
		# pc.send_leftArm_values_speed(t1, t2, t3, t4)
	   
def left_hand_values(lthumb, ltip):
	global prev_dist_left
	hand, new_dist = hand_open_close_left(lthumb, ltip, prev_dist_left)
	prev_dist_left = new_dist
	# print(hand)
	pc.send_leftHand_values(hand)

def right_hand_values(rthumb, rtip):
	global prev_dist_right
	hand, new_dist = hand_open_close_right(rthumb, rtip, prev_dist_right)
	prev_dist_right = new_dist

	pc.send_rightHand_values(hand)

# Function for scaling the joints used for calculating the angles (IK) for the torso chain
def torso_scaling(spine_base, torso):
	# Setting the joints in Pepper cooridnates
	torso = tfm.set_joint_in_peppers_coordinates(torso)
	spine_base = tfm.set_joint_in_peppers_coordinates(spine_base)

	# make the spine_base origin
	torso_new = torso - spine_base

	# assign the hip vaule to be 347 mm from the knee, which is the origin for Pepper
	hip_scaled = array([0, 0, 347])

	# Sclaing the torso position to fit Pepper hipRoll-to-Torso length
	torso_scaled = 139 * torso_new/linalg.norm(torso_new) + hip_scaled

	# Calculate the IK or the torso chain angles using the scaled torso position
	[t1, t2, t3] = ik.get_torso_angles_constrained(torso_scaled[0], torso_scaled[1], torso_scaled[2])
	# Send calculated angles to Pepper
	pc.send_torso_values_speed(t1, t2, -t3)
	# pc.send_torso_values_speed_t2t3(t2, t3)



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
		# Pick the position and angles with the lowest distance
		if dist1 < dist2: 
			wrist_fitted = wrist_fitted1
			elbow_fitted = elbow_fitted1
			t1 = t1_1
			t2 = t2_1
		elif dist1 > dist2:
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
	phi = arctan2(position[1], position[0])
	theta = arccos(position[2]/r)

	return r, phi, theta

# Function to convert from spherical to cartesian
def to_cartesian(r, phi, theta):

	position = array([r*sin(theta)*cos(phi), \
				r*sin(theta)*sin(phi), \
				r*cos(theta)])

	return position

# print(get_time())
