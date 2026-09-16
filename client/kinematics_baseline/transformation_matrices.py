from forward_kinematics import get_wrist_position
import numpy as np
import math_functions as mf
import time

#-------------------------------------------------
# Transformation matrices

def set_joint_in_peppers_coordinates(joint):
    return mf.rz_calc(-np.pi/2, 3).dot(mf.rx_calc(np.pi/2, 3).dot(joint * 1000))


def set_joint_in_peppers_coordinates_mirror(joint):
    mat = mf.rz_calc(-np.pi/2, 3).dot(mf.rx_calc(np.pi/2, 3).dot(joint * 1000))
    return np.array([1, 1, 1]) * (mf.rz_calc(-np.pi/2, 3).dot(mf.rx_calc(np.pi/2, 3).dot(joint * 1000)))
    


# Function for computing the transformation matrix with unit vectors
# for the workspace of the wrist
def trans_elbow_workspace(t1, t2, elbow, pepper_shoulder, arm):
    # Get the vectors when the elbow angles are 0 and when it is 90 degrees 
    # with the 0 position of the elbows
    if arm == 'right':
        # old version
        # pos90 = get_wrist_position(t1, t2, 1.5620, 1.5620, arm) - elbow - pepper_shoulder
        pos90 = get_wrist_position(t1, t2, 0.0, 1.5620, arm) - elbow - pepper_shoulder
        pos0 = get_wrist_position(t1, t2, 0.0, 0.0087, arm) - elbow - pepper_shoulder
    elif arm == 'left':
        # old version
        # pos90 = get_wrist_position(t1, t2, -1.5620, -1.5620, arm) - elbow - pepper_shoulder
        pos90 = get_wrist_position(t1, t2, 0.0, -1.5620, arm) - elbow - pepper_shoulder
        pos0 = get_wrist_position(t1, t2, 0.0, -0.0087, arm) - elbow - pepper_shoulder
    # Assign the axises
    x_axis = (pos0) / np.linalg.norm(pos0)

    y_axis = (pos90) / np.linalg.norm(pos90)
    # Compute the normal to x-z axis
    z_axis = np.cross(x_axis, y_axis)

    return np.transpose([x_axis, y_axis, z_axis])

# Function for computing the transformation matrix with unit vectors
def rotation_mat_arms(rshoulder, lshoulder):

    # Calculate the midpoint of the shoulders
    mid_shoulder = (rshoulder + lshoulder) / 2

    # Calculating the normalized direction vector that points in the direction
    # of the left shoulder as seen from the right shoulder
    vec_LR = lshoulder - rshoulder
    vec_LR_norm = vec_LR / np.linalg.norm(vec_LR)

    # Calculate the normalized direction vector which starts at the torso
    # and points in the direction of the mid shoulder point
    mh_dir = - mid_shoulder
    mh_dir_norm = mh_dir / np.linalg.norm(mh_dir)

    # Calculate the cross product of vec_LR (the direction vector in the direction of
    # the left shoulder) and dirMHn (the direction vector in the direction of midhip).
    # Then you get the normal vector to the plane made up by VS and dirMHn,
    # which points outward

    vec1 = np.cross(mh_dir_norm, vec_LR_norm)
    vec1_norm = vec1 / np.linalg.norm(vec1)

    # Calculate the cross product of vLRn (direction vector in direction of left
    # shoulder, as seen from the right shoulder) and vec1n (the vector that is
    # normal to vLRn and dirMHn) to get the normal vector to the plane made up
    # by vLRn and vec1n

    vec2 = np.cross(vec1_norm, vec_LR_norm)
    vec2_norm = vec2 / np.linalg.norm(vec2)


    trans_mat = np.transpose(np.c_[vec1_norm, vec_LR_norm, vec2_norm])
    # The transpose is returned because the vectors are in 1 x 3 instead of 3 x 1

    return trans_mat

def rotation_mat(right_point, left_point):
    # Calculate the midpoint of the points
    mid_point = (right_point + left_point) / 2

    # Calculating the normalized direction vector that points in the direction
    # of the right point as seen from the left point
    vec_LR = left_point - right_point
    vec_LR_norm = vec_LR / np.linalg.norm(vec_LR)

    # Calculate the normalized direction vector which starts at the origin
    # and points in the direction of the mid shoulder point
    mh_dir = - mid_point
    mh_dir_norm = mh_dir / np.linalg.norm(mh_dir)

    # Calculate the cross product of vec_LR_norm and mh_dir_norm
    # to get the normal vector pointing outward
    vec1 = np.cross(mh_dir_norm, vec_LR_norm)
    vec1_norm = vec1 / np.linalg.norm(vec1)

    # Calculate the cross product of vec_LR_norm and vec1_norm to get 
    # the normal vector to the plane made up by vec_LR_norm and vec1_norm
    vec2 = np.cross(vec1_norm, vec_LR_norm)
    vec2_norm = vec2 / np.linalg.norm(vec2)

    rot_mat = np.transpose(np.c_[vec1_norm, vec_LR_norm, vec2_norm])

    return rot_mat
    

def rotation_mat_torso(mid_hip, hip_right, hip_left):
    # Take the midknee to midhip as the Z-axis
    # mh_dir = -mid_hip
    # mh_dir_norm = mh_dir/np.linalg.norm(mh_dir)

    # # Calculate the vector that will be used to get the X axis
    # vec_lr =  knee_left - knee_right
    # vec_lr_norm = vec_lr/np.linalg.norm(vec_lr)

    # # Calculate the X axis using a cross product
    # # vec1 = np.cross(vec_lr_norm, mh_dir_norm)
    # vec1 = np.cross(mh_dir_norm, vec_lr_norm)
    # vec1_norm = vec1/np.linalg.norm(vec1)
    
    # # Calculate the Y axis using a cross product
    # # vec2 = np.cross(mh_dir_norm, vec1_norm)
    # vec2 = np.cross(vec1_norm, vec_lr_norm)
    # vec2_norm = vec2/np.linalg.norm(vec2)

    mh_dir = -mid_hip
    mh_dir_norm = mh_dir/np.linalg.norm(mh_dir)

    # Calculate the vector that will be used to get the X axis
    vec_lr =  hip_left - hip_right
    vec_lr_norm = vec_lr/np.linalg.norm(vec_lr)

    # Calculate the X axis using a cross product
    # vec1 = np.cross(vec_lr_norm, mh_dir_norm)
    vec1 = np.cross(mh_dir_norm, vec_lr_norm)
    vec1_norm = vec1/np.linalg.norm(vec1)
    
    # Calculate the Y axis using a cross product
    # vec2 = np.cross(mh_dir_norm, vec1_norm)
    vec2 = np.cross(vec1_norm, vec_lr_norm)
    vec2_norm = vec2/np.linalg.norm(vec2)

    Mat = np.transpose(np.c_[vec1_norm, vec_lr_norm, vec2_norm])

    return Mat

