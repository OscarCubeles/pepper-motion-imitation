import server.common.kinematics.math_functions as mathik
import server.common.kinematics.forward_kinematics as fk
from numpy import pi, arctan, arctan2, arcsin, arccos, sqrt, sin, cos, transpose, radians, real, isnan, nan, linalg, array
import numpy as np

def set_joint_in_peppers_coordinates(joint):
    """Transform joint to Pepper's coordinate system."""
    return mathik.rz_calc(-pi/2, 3).dot(mathik.rx_calc(pi/2, 3).dot(joint * 1000))

def rotation_mat_arms( rshoulder, lshoulder):
    """Create rotation matrix from shoulder positions."""
    mid_shoulder = (rshoulder + lshoulder) / 2
    vec_LR = lshoulder - rshoulder
    vec_LR_norm = vec_LR / linalg.norm(vec_LR)
    
    mh_dir = -mid_shoulder
    mh_dir_norm = mh_dir / linalg.norm(mh_dir)
    
    vec1 = np.cross(mh_dir_norm, vec_LR_norm)
    vec1_norm = vec1 / linalg.norm(vec1)
    
    vec2 = np.cross(vec1_norm, vec_LR_norm)
    vec2_norm = vec2 / linalg.norm(vec2)
    
    trans_mat = transpose(np.c_[vec1_norm, vec_LR_norm, vec2_norm])
    return trans_mat

def trans_elbow_workspace( t1, t2, elbow, pepper_shoulder, arm, use_human_mode=False):
    """Create transformation matrix for wrist workspace."""
    if arm == 'right':
        t4_ref = 0.0 if use_human_mode else 0.0087
        pos90 = fk.get_wrist_position(t1, t2, 0.0, 1.5620, arm) - elbow - pepper_shoulder
        pos0 = fk.get_wrist_position(t1, t2, 0.0, t4_ref, arm) - elbow - pepper_shoulder
    elif arm =='left':
        t4_ref = 0.0 if use_human_mode else -0.0087
        pos90 = fk.get_wrist_position(t1, t2, 0.0, -1.5620, arm) - elbow - pepper_shoulder
        pos0 = fk.get_wrist_position(t1, t2, 0.0, t4_ref, arm) - elbow - pepper_shoulder
    
    x_axis = pos0 / linalg.norm(pos0)
    y_axis = pos90 / linalg.norm(pos90)
    z_axis = np.cross(x_axis, y_axis)
    
    return transpose([x_axis, y_axis, z_axis])

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