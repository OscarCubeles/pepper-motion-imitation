from server.common.kinematics import inverse_kinematics as ik
from server.common.kinematics import transformation_matrices as tm
from server.common.kinematics import math_functions as mathik
from server.common.kinematics import scaling_spherical as scaling 
from numpy import pi, arctan, arctan2, arcsin, arccos, sqrt, sin, cos, transpose, radians, real, isnan, nan, linalg, array

def get_wrist_fitted( elbow_fitted, wrist, elbow, arm, use_human_mode=False):
    """Fit wrist position to workspace constraints."""
    if arm == 'right':
        pepper_shoulder = array([-57, -149.74, 86.82])
    elif arm == 'left':
        pepper_shoulder = array([-57, 149.74, 86.82])
    
    elbow_new = pepper_shoulder + elbow_fitted
    
    if use_human_mode:
        t1, t2 = ik.get_arm_partial_angles_human(elbow_new[0], elbow_new[1], elbow_new[2], arm)
    else:
        t1, t2 = ik.get_arm_partial_angles(elbow_new[0], elbow_new[1], elbow_new[2], arm)
    
    Mat = tm.trans_elbow_workspace(t1, t2, elbow_fitted, pepper_shoulder, arm, use_human_mode)
    rot_x = mathik.rx_calc(pi/2, 3)
    rot_y = mathik.ry_calc(pi/2, 3)
    wrist = wrist - elbow - pepper_shoulder
    
    wrist_spherical = transpose(rot_y).dot(transpose(rot_x).dot(transpose(Mat).dot(wrist)))
    
    wr_eq, wphi_eq, wtheta_eq = scaling.to_spherical(wrist_spherical)
    
    if not use_human_mode:
        if wtheta_eq < 0.008:
            wtheta_eq = 0.0087
        elif wtheta_eq > 1.56:
            wtheta_eq = 1.55
        
        if wphi_eq < -2.09:
            wphi_eq = -2.08
        elif wphi_eq > 2.06:
            wphi_eq = 2.05
    
    wrist_fitted = Mat.dot(rot_x.dot(rot_y.dot(scaling.to_cartesian(wr_eq, wphi_eq, wtheta_eq)))) + elbow_new
    
    return t1, t2, wrist_fitted

def get_elbow_wrist_fitted(elbow, wrist, arm, use_human_mode=False):
    """Fit both elbow and wrist to workspace constraints."""
    if arm == 'right':
        rot_x = mathik.rx_calc(pi/2, 3)
    elif arm == 'left':
        rot_x = mathik.rx_calc(-pi/2, 3)
    
    elbow_spherical = transpose(rot_x).dot(elbow)
    er_eq, ephi_eq, etheta_eq = scaling.to_spherical(elbow_spherical)
    
    if not use_human_mode:
        if etheta_eq < 0.0:
            etheta_eq = 0.0079
        elif etheta_eq > 1.48:
            etheta_eq = 1.48
        
        if ephi_eq < -3.15:
            ephi_eq = -3.14
        elif ephi_eq > 3.15:
            ephi_eq = 3.14
    
    if etheta_eq > 0.06 and etheta_eq < 0.165:
        elbow_fitted = rot_x.dot(scaling.to_cartesian(er_eq, ephi_eq, etheta_eq))
        t1, t2, wrist_fitted = get_wrist_fitted(elbow_fitted, wrist, elbow, arm, use_human_mode)
    elif not use_human_mode and (ephi_eq < -2.1 or ephi_eq > 2.1):
        ephi_eq1 = -2.09
        elbow_fitted1 = rot_x.dot(scaling.to_cartesian(er_eq, ephi_eq1, etheta_eq))
        ephi_eq2 = 2.09
        elbow_fitted2 = rot_x.dot(scaling.to_cartesian(er_eq, ephi_eq2, etheta_eq))
        
        t1_1, t2_1, wrist_fitted1 = get_wrist_fitted(elbow_fitted1, wrist, elbow, arm, use_human_mode)
        t1_2, t2_2, wrist_fitted2 = get_wrist_fitted(elbow_fitted2, wrist, elbow, arm, use_human_mode)
        
        dist1 = sqrt((wrist[0]-wrist_fitted1[0])**2 + (wrist[1]-wrist_fitted1[1])**2 + (wrist[2]-wrist_fitted1[2])**2)
        dist2 = sqrt((wrist[0]-wrist_fitted2[0])**2 + (wrist[1]-wrist_fitted2[1])**2 + (wrist[2]-wrist_fitted2[2])**2)
        # TODO: Add here the distance thing for the hysteresis and distance solution

        branch_choice = scaling.choose_arm_solution_branch(arm, dist1, dist2)
        scaling.set_arm_solution_branch(arm, branch_choice)  
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


        #if dist1 <= dist2:
        #    wrist_fitted = wrist_fitted1
        #    elbow_fitted = elbow_fitted1
        #    t1 = t1_1
        #    t2 = t2_1
        #else:
        #    wrist_fitted = wrist_fitted2
        #    elbow_fitted = elbow_fitted2
        #    t1 = t1_2
        #    t2 = t2_2
    else:
        elbow_fitted = rot_x.dot(scaling.to_cartesian(er_eq, ephi_eq, etheta_eq))
        t1, t2, wrist_fitted = get_wrist_fitted(elbow_fitted, wrist, elbow, arm, use_human_mode)
    
    return t1, t2, elbow_fitted, wrist_fitted
