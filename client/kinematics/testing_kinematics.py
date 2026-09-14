import csv 
import numpy as np
import time
from inverse_kinematics import get_arm_all_angles, get_head_angles, get_torso_angles, get_arm_partial_angles
from forward_kinematics import get_wrist_position, get_elbow_position, get_head_position, get_torso_position


# HEAD EVALUATION ======================================================================================================

# floatArray = np.zeros((10, 5067))
# i = 0
# # Read the angle values used for the physical robot to be used for choregraphe
# with open('actual-pepper-head-values-within-range.csv', 'r') as csvfile:
# # with open('virtual-pepper-head-values-sent-within-range.csv', 'r') as csvfile:
# 	reader = csv.reader(csvfile)
# 	for row in reader:
# 		floatArray[i, :] = row
# 		i+=1


# # head yaw
# pt1_sent = floatArray[0,:]
# pt1_sensed = floatArray[1,:]

# # head pitch
# pt2_sent = floatArray[2,:]
# pt2_sensed = floatArray[3,:]

# # top camera 
# ptop_x = floatArray[4,:]
# ptop_y = floatArray[5,:]
# ptop_z = floatArray[6,:]

# # bottom camera
# pbottom_x = floatArray[7,:]
# pbottom_y = floatArray[8,:]
# pbottom_z = floatArray[9,:]


# cam_pos = np.zeros((5067, 3))

# angles = np.zeros((5067, 2))

# fk_times = np.zeros((5067, 1))
# ik_times = np.zeros((5067, 1))


# for i in range(0, floatArray.shape[1]):
# 	# forward kinematics
# 	start = time.time()
# 	cam_pos[i, :] = get_head_position(pt1_sensed[i], pt2_sensed[i])
# 	end = time.time()
# 	fk_times[i] = end-start

# 	# inverse kinematics 
# 	start = time.time()
# 	angles[i, :] = get_head_angles(pbottom_x[i], pbottom_y[i], pbottom_z[i]) 
# 	end = time.time()
# 	ik_times[i] = end-start


# # HEAD IK =====================================================================================================
# print("Mean t1: " + str(np.mean(abs(pt1_sensed - angles[:, 0]))))
# print("STD t1: " + str(np.std(abs(pt1_sensed - angles[:, 0]))))

# print("Mean t2: " + str(np.mean(abs(pt2_sensed - angles[:, 1]))))
# print("STD t2: " + str(np.std(abs(pt2_sensed - angles[:, 1]))))

# print("Mean IK comp time: " + str(np.mean(ik_times) * 1000))
# print("STD  IK comp time: " + str(np.std(ik_times) * 1000))


# for i in range(124, 129):

# 	# inverse kinematics 
# 	angles[i, :] = get_arm_all_angles(pe_x[i], pe_y[i], pe_z[i], pw_x[i], pw_y[i], pw_z[i], 'right') 

# print("Angles: ")
# print(angles[124:128,:])

# all_angles = np.c_[pt1_sensed, angles[:, 0], pt2_sensed, angles[:, 1],  pt3_sensed, angles[:, 2], pt4_sensed, angles[:, 3], angles[:, 4]]
# print(all_angles.shape)

# np.savetxt("calculated_arm_angles_IK_eval.csv", all_angles, delimiter=",", fmt='%f')


# HEAD FK ===========================================================================================================
# euc_dist_physical = np.sqrt((pbottom_x - cam_pos[:, 0])**2 + (pbottom_y - cam_pos[:, 1])**2 + (pbottom_z - cam_pos[:, 2])**2);

# print("Mean distance FK full: " + str(np.mean(euc_dist_physical)))

# print("STD FK full: " + str(np.std(euc_dist_physical)))

# print("Mean FK comp time: " + str(np.mean(fk_times) * 1000))
# print("STD  FK comp time: " + str(np.std(fk_times) * 1000))



# # ARMS EVALUATION ======================================================================================================
# print("RIGHT")
# floatArray = np.zeros((14, 12912))
# i = 0
# # Read the angle values used for the physical robot to be used for choregraphe
# # with open('actual-right-arm-values-part-all-within-range.csv', 'r') as csvfile:
# with open('virtual-right-arm-values-within-range.csv', 'r') as csvfile:
# 	reader = csv.reader(csvfile)
# 	for row in reader:
# 		floatArray[i, :] = row
# 		i+=1


# # shoulder pitch
# pt1_sent = floatArray[0,:]
# pt1_sensed = floatArray[1,:]

# # shoulder roll
# pt2_sent = floatArray[2,:]
# pt2_sensed = floatArray[3,:]

# # elbow yaw
# pt3_sent = floatArray[4,:]
# pt3_sensed = floatArray[5,:]

# # elbow roll
# pt4_sent = floatArray[6,:]
# pt4_sensed = floatArray[7,:]

# # wrist position
# pw_x = floatArray[8,:]
# pw_y = floatArray[9,:]
# pw_z = floatArray[10,:]


# # elbow position
# pe_x = floatArray[11,:]
# pe_y = floatArray[12,:]
# pe_z = floatArray[13,:]


# elbow_pos = np.zeros((12912, 3))
# wrist_pos = np.zeros((12912, 3))

# angles = np.zeros((12912, 4))
# angles2 = np.zeros((12912, 2))

# fk_times_partial = np.zeros((12912, 1))
# fk_times_full = np.zeros((12912, 1))
# ik_times = np.zeros((12912, 1))
# ik_times2 = np.zeros((12912, 1))

# term1 = np.zeros((12912, 1))

# # sensed_ang = np.c_[pt1_sensed, pt2_sensed, pt3_sensed, pt4_sensed]

# for i in range(0, floatArray.shape[1]):
# 	# forward kinematics
# 	start = time.time()
# 	wrist_pos[i, :] = get_wrist_position(pt1_sensed[i], pt2_sensed[i], pt3_sensed[i], pt4_sensed[i], 'right')
# 	end = time.time()
# 	fk_times_full[i] = end-start

# 	start = time.time()
# 	elbow_pos[i, :]  = get_elbow_position(pt1_sensed[i], pt2_sensed[i], 'right')
# 	end = time.time()
# 	fk_times_partial[i] = end-start
	 

# 	# inverse kinematics 
# 	start = time.time()
# 	angles[i, :] = get_arm_all_angles(pe_x[i], pe_y[i], pe_z[i], pw_x[i], pw_y[i], pw_z[i], 'right') 
# 	end = time.time()
# 	ik_times[i] = end-start

# 	start = time.time()
# 	angles2[i, :] = get_arm_partial_angles(pe_x[i], pe_y[i], pe_z[i], 'right') 
# 	end = time.time()
# 	ik_times2[i] = end-start

# # print(ik_times)
# # ARMS IK =====================================================================================================
# print("Mean t1: " + str(np.mean(abs(pt1_sensed - angles[:, 0]))))
# print("STD t1: " + str(np.std(abs(pt1_sensed - angles[:, 0]))))

# print("Mean t2: " + str(np.mean(abs(pt2_sensed - angles[:, 1]))))
# print("STD t2: " + str(np.std(abs(pt2_sensed - angles[:, 1]))))

# print("Mean t3: " + str(np.mean(abs(pt3_sensed - angles[:, 2]))))
# print("STD t3: " + str(np.std(abs(pt3_sensed - angles[:, 2]))))

# print("Mean t4: " + str(np.mean(abs(pt4_sensed - angles[:, 3]))))
# print("STD t4: " + str(np.std(abs(pt4_sensed - angles[:, 3]))))

# print("Mean IK comp time: " + str(np.mean(ik_times) * 1000))
# print("STD  IK comp time: " + str(np.std(ik_times) * 1000))

# print("Mean IK partial comp time: " + str(np.mean(ik_times2) * 1000))
# print("STD  IK partial comp time: " + str(np.std(ik_times2) * 1000))

# # ARMS FK ===========================================================================================================
# euc_dist_physical = np.sqrt((pw_x - wrist_pos[:, 0])**2 + (pw_y - wrist_pos[:, 1])**2 + (pw_z - wrist_pos[:, 2])**2);

# euc_dist_partial = np.sqrt((pe_x - elbow_pos[:, 0])**2 + (pe_y - elbow_pos[:, 1])**2 + (pe_z - elbow_pos[:, 2])**2);

# print("Mean distance FK full: " + str(np.mean(euc_dist_physical)))

# print("STD FK full: " + str(np.std(euc_dist_physical)))

# print("Mean distance FK partial: " + str(np.mean(euc_dist_partial)))

# print("STD FK partial: " + str(np.std(euc_dist_partial)))

# print("Mean FK partial comp time: " + str(np.mean(fk_times_partial)* 1000))
# print("STD  FK comp partil time: " + str(np.std(fk_times_partial)* 1000))

# print("Mean FK full comp time: " + str(np.mean(fk_times_full)* 1000))
# print("STD  FK comp full time: " + str(np.std(fk_times_full)* 1000))

# # Left ----------------------------------------------------------------------------

# print("LEFT")
# floatArray = np.zeros((14, 12912))
# i = 0
# # Read the angle values used for the physical robot to be used for choregraphe
# # with open('actual-right-arm-values-part-all-within-range.csv', 'r') as csvfile:
# with open('virtual-left-arm-values-within-range.csv', 'r') as csvfile:
# 	reader = csv.reader(csvfile)
# 	for row in reader:
# 		floatArray[i, :] = row
# 		i+=1


# # shoulder pitch
# pt1_sent = floatArray[0,:]
# pt1_sensed = floatArray[1,:]

# # shoulder roll
# pt2_sent = floatArray[2,:]
# pt2_sensed = floatArray[3,:]

# # elbow yaw
# pt3_sent = floatArray[4,:]
# pt3_sensed = floatArray[5,:]

# # elbow roll
# pt4_sent = floatArray[6,:]
# pt4_sensed = floatArray[7,:]

# # wrist position
# pw_x = floatArray[8,:]
# pw_y = floatArray[9,:]
# pw_z = floatArray[10,:]


# # elbow position
# pe_x = floatArray[11,:]
# pe_y = floatArray[12,:]
# pe_z = floatArray[13,:]


# elbow_pos = np.zeros((12912, 3))
# wrist_pos = np.zeros((12912, 3))

# angles = np.zeros((12912, 4))
# term1 = np.zeros((12912, 1))

# # sensed_ang = np.c_[pt1_sensed, pt2_sensed, pt3_sensed, pt4_sensed]

# for i in range(0, floatArray.shape[1]):
# 	# forward kinematics
# 	start = time.time()
# 	wrist_pos[i, :] = get_wrist_position(pt1_sensed[i], pt2_sensed[i], pt3_sensed[i], pt4_sensed[i], 'left')
# 	end = time.time()
# 	fk_times_full[i] = end-start

# 	start = time.time()
# 	elbow_pos[i, :]  = get_elbow_position(pt1_sensed[i], pt2_sensed[i], 'left')
# 	end = time.time()
# 	fk_times_partial[i] = end-start
	 

# 	# inverse kinematics 
# 	start = time.time()
# 	angles[i, :] = get_arm_all_angles(pe_x[i], pe_y[i], pe_z[i], pw_x[i], pw_y[i], pw_z[i], 'left') 
# 	end = time.time()
# 	ik_times[i] = end-start


# # ARMS IK =====================================================================================================
# print("Mean t1: " + str(np.mean(abs(pt1_sensed - angles[:, 0]))))
# print("STD t1: " + str(np.std(abs(pt1_sensed - angles[:, 0]))))

# print("Mean t2: " + str(np.mean(abs(pt2_sensed - angles[:, 1]))))
# print("STD t2: " + str(np.std(abs(pt2_sensed - angles[:, 1]))))

# print("Mean t3: " + str(np.mean(abs(pt3_sensed - angles[:, 2]))))
# print("STD t3: " + str(np.std(abs(pt3_sensed - angles[:, 2]))))

# print("Mean t4: " + str(np.mean(abs(pt4_sensed - angles[:, 3]))))
# print("STD t4: " + str(np.std(abs(pt4_sensed - angles[:, 3]))))

# print("Mean IK comp time: " + str(np.mean(ik_times)* 1000))
# print("STD  IK comp time: " + str(np.std(ik_times)* 1000))

# # ARMS FK ===========================================================================================================
# euc_dist_physical = np.sqrt((pw_x - wrist_pos[:, 0])**2 + (pw_y - wrist_pos[:, 1])**2 + (pw_z - wrist_pos[:, 2])**2);

# euc_dist_partial = np.sqrt((pe_x - elbow_pos[:, 0])**2 + (pe_y - elbow_pos[:, 1])**2 + (pe_z - elbow_pos[:, 2])**2);

# print("Mean distance FK full: " + str(np.mean(euc_dist_physical)))

# print("STD FK full: " + str(np.std(euc_dist_physical)))

# print("Mean distance FK partial: " + str(np.mean(euc_dist_partial)))

# print("STD FK partial: " + str(np.std(euc_dist_partial)))

# print("Mean FK partial comp time: " + str(np.mean(fk_times_partial)* 1000))
# print("STD  FK comp partil time: " + str(np.std(fk_times_partial)* 1000))

# print("Mean FK full comp time: " + str(np.mean(fk_times_full)* 1000))
# print("STD  FK comp full time: " + str(np.std(fk_times_full)* 1000))



# TORSO EVALUATION ======================================================================================================

floatArray = np.zeros((8, 1387))
i = 0
# Read the angle values used for the physical robot to be used for choregraphe
with open('actual-pepper-torso-values-within-range.csv', 'r') as csvfile:
# with open('virtual-pepper-torso-values-within-range.csv', 'r') as csvfile:
	reader = csv.reader(csvfile)
	for row in reader:
		floatArray[i, :] = row
		i+=1


# knee pitch
pt1_sensed = floatArray[0,:]

# hip pitch
pt2_sent = floatArray[1,:]
pt2_sensed = floatArray[2,:]

# hip roll 
pt3_sent = floatArray[3,:]
pt3_sensed = floatArray[4,:]

# torso position
ptop_x = floatArray[5,:]
ptop_y = floatArray[6,:]
ptop_z = floatArray[7,:]


cam_pos = np.zeros((1387, 3))

angles = np.zeros((1387, 3))

fk_times = np.zeros((1387, 1))
ik_times = np.zeros((1387, 1))


for i in range(0, floatArray.shape[1]):
	# forward kinematics
	start = time.time()
	cam_pos[i, :] = get_torso_position(pt2_sensed[i], pt3_sensed[i], pt1_sensed[i])
	end = time.time()
	fk_times[i] = end-start

	# inverse kinematics 
	start = time.time()
	angles[i, :] = get_torso_angles(ptop_x[i], ptop_y[i], ptop_z[i]) 
	end = time.time()
	ik_times[i] = end-start


# TORSO IK =====================================================================================================
print("Mean t1: " + str(np.mean(abs(pt1_sensed - angles[:, 0]))))
print("STD t1: " + str(np.std(abs(pt1_sensed - angles[:, 0]))))

print("Mean t2: " + str(np.mean(abs(pt2_sensed - angles[:, 1]))))
print("STD t2: " + str(np.std(abs(pt2_sensed - angles[:, 1]))))

print("Mean t3: " + str(np.mean(abs(pt3_sensed - angles[:, 2]))))
print("STD t3: " + str(np.std(abs(pt3_sensed - angles[:, 2]))))

print("Mean IK comp time: " + str(np.mean(ik_times) * 1000))
print("STD  IK comp time: " + str(np.std(ik_times) * 1000))

# for i in range(124, 129):

# 	# inverse kinematics 
# 	angles[i, :] = get_arm_all_angles(pe_x[i], pe_y[i], pe_z[i], pw_x[i], pw_y[i], pw_z[i], 'right') 

# print("Angles: ")
# print(angles[124:128,:])

# all_angles = np.c_[pt1_sensed, angles[:, 0], pt2_sensed, angles[:, 1],  pt3_sensed, angles[:, 2], pt4_sensed, angles[:, 3], angles[:, 4]]
# print(all_angles.shape)

# np.savetxt("calculated_arm_angles_IK_eval.csv", all_angles, delimiter=",", fmt='%f')


# TORSO FK ===========================================================================================================
euc_dist_physical = np.sqrt((ptop_x - cam_pos[:, 0])**2 + (ptop_y - cam_pos[:, 1])**2 + (ptop_z - cam_pos[:, 2])**2);

print("Mean distance FK full: " + str(np.mean(euc_dist_physical)))

print("STD FK full: " + str(np.std(euc_dist_physical)))

print("Mean FK comp time: " + str(np.mean(fk_times) * 1000))
print("STD  FK comp time: " + str(np.std(fk_times) * 1000))




# ========================================================================================================================

# for i in range(124, 129):

# 	# inverse kinematics 
# 	angles[i, :] = get_arm_all_angles(pe_x[i], pe_y[i], pe_z[i], pw_x[i], pw_y[i], pw_z[i], 'right') 

# print("Angles: ")
# print(angles[124:128,:])

# all_angles = np.c_[pt1_sensed, angles[:, 0], pt2_sensed, angles[:, 1],  pt3_sensed, angles[:, 2], pt4_sensed, angles[:, 3], angles[:, 4]]
# print(all_angles.shape)

# np.savetxt("calculated_arm_angles_IK_eval.csv", all_angles, delimiter=",", fmt='%f')