import transformation_matrices as tf
from numpy import dot, pi, array, transpose
import math_functions as mf

# torso = array([0.004180649, 0.03427865, 1.428411]) 

# lshoulder = array([-0.1410899, 0.2276636, 1.401736]) - torso
# lelbow = array([-0.231998, 0.431197, 1.324431]) - torso
# lwrist = array([-0.219993, 0.6298395, 1.30153]) - torso

# rshoulder = array([0.1720636, 0.2483574, 1.467632]) - torso
# relbow = [0.2938134, 0.4558499, 1.466948] - torso
# rwrist = [0.3320537, 0.692629, 1.460736] - torso

# torso = array([-0.0439507, -0.1121121, 1.498769]) 

# lshoulder = array([-0.2606537, -0.01699516, 1.513237]) - torso
# lelbow = array([ -0.2687752, -0.2436426, 1.526361]) - torso
# lwrist = array([-0.482917, -0.1994639, 1.457398]) - torso

# rshoulder = array([0.05810986, 0.1185088, 1.430159]) - torso
# relbow = [0.2735359, 0.188331, 1.441096] - torso
# rwrist = [0.2666711, 0.2756205, 1.324982] - torso



# torso_new = mf.rz_calc(-pi/2, 3).dot(mf.rx_calc(pi/2, 3).dot(torso * 1000))
# rshoulder_new = mf.rz_calc(-pi/2, 3).dot(mf.rx_calc(pi/2, 3).dot(rshoulder * 1000))
# relbow_new = mf.rz_calc(-pi/2, 3).dot(mf.rx_calc(pi/2, 3).dot(relbow * 1000))
# rwrist_new = mf.rz_calc(-pi/2, 3).dot(mf.rx_calc(pi/2, 3).dot(rwrist * 1000))

# lshoulder_new = mf.rz_calc(-pi/2, 3).dot(mf.rx_calc(pi/2, 3).dot(lshoulder * 1000)) 
# lelbow_new = mf.rz_calc(-pi/2, 3).dot(mf.rx_calc(pi/2, 3).dot(lelbow * 1000)) 
# lwrist_new = mf.rz_calc(-pi/2, 3).dot(mf.rx_calc(pi/2, 3).dot(lwrist * 1000)) 

# Mat = tf.rotation_mat_arms(rshoulder_new, lshoulder_new)
# print(Mat)

# print(Mat.dot(lshoulder_new))
# print(Mat.dot(rshoulder_new))

knee_right = array([0.1719148, -0.6525004, 1.473885])
knee_left = array([0.01066385, -0.6494913, 1.457184])
hip_right = array([0.1703661, -0.3647335, 1.455506])
hip_left = array([0.0120683, -0.3569806, 1.460568]) 
torso = array([0.1011772, -0.04750534, 1.492721])

knee_right = mf.rz_calc(-pi/2, 3).dot(mf.rx_calc(pi/2, 3).dot(knee_right * 1000))
knee_left = mf.rz_calc(-pi/2, 3).dot(mf.rx_calc(pi/2, 3).dot(knee_left * 1000))
hip_right = mf.rz_calc(-pi/2, 3).dot(mf.rx_calc(pi/2, 3).dot(hip_right * 1000))
hip_left = mf.rz_calc(-pi/2, 3).dot(mf.rx_calc(pi/2, 3).dot(hip_left * 1000))
torso = mf.rz_calc(-pi/2, 3).dot(mf.rx_calc(pi/2, 3).dot(torso * 1000))


mid_knee = (knee_right + knee_left)/2;

mid_knee = (knee_right + knee_left)/2

hip_right = hip_right - mid_knee
hip_left = hip_left - mid_knee

knee_right = knee_right - mid_knee
knee_left = knee_left - mid_knee

mid_hip = (hip_right + hip_left)/2 
torso = torso - mid_knee - mid_hip

Mat = tf.rotation_mat_torso(mid_hip, hip_right, hip_left)
# print(Mat)


t1 = -2.0866
t2 =  1.2765
elbow = array([-74.4349, 327.5005, 117.3270])
pepper_shoulder = array([-57, 149.74, 86.82])

print(tf.trans_elbow_workspace(t1, t2, elbow, pepper_shoulder, 'left'))











