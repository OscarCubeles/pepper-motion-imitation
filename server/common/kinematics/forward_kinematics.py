from numpy import pi, arctan, arctan2, arcsin, arccos, sqrt, sin, cos, transpose, radians, real, isnan, nan, linalg, array
import server.common.kinematics.constants as constants

def get_elbow_position(t1, t2, arm):
    """Get elbow position from joint angles."""
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
    
    px = l1 + l6*sin(t1) + cos(t1)*(l4*cos(t2) - l5*sin(t2))
    py = l2 + l5*cos(t2) + l4*sin(t2)
    pz = l3 + l6*cos(t1) - sin(t1)*(l4*cos(t2) - l5*sin(t2))
    
    return array([px, py, pz])

def get_wrist_position(t1, t2, t3, t4, arm):
    """Get wrist position from all 4 joint angles."""
    l1 = -57.0
    l3 = 86.82
    l4 = 150
    d3 = 181.2
    z3 = 0.13
    
    if arm == 'right':
        l2 = -149.74
        a3 = 15.0
    elif arm == 'left':
        l2 = 149.74
        a3 = -15.0
    
    t2_adj = t2 - pi/2
    a = radians(9)
    
    px = (l1 - cos(t1)*(sin(t2_adj)*(d3 + l4*cos(a)*cos(t4) - l4*sin(a)*sin(t3)*sin(t4)) - 
            cos(t2_adj)*(a3 - l4*cos(t3)*sin(t4))) + sin(t1)*(z3 + l4*sin(a)*cos(t4) + l4*cos(a)*sin(t3)*sin(t4)))
    
    py = (l2 + cos(t2_adj)*(d3 + l4*cos(a)*cos(t4) - l4*sin(a)*sin(t3)*sin(t4)) + 
            sin(t2_adj)*(a3 - l4*cos(t3)*sin(t4)))
    
    pz = (l3 + sin(t1)*(sin(t2_adj)*(d3 + l4*cos(a)*cos(t4) - l4*sin(a)*sin(t3)*sin(t4)) - 
            cos(t2_adj)*(a3 - l4*cos(t3)*sin(t4))) + cos(t1)*(z3 + l4*sin(a)*cos(t4) + l4*cos(a)*sin(t3)*sin(t4)))
    
    return array([px, py, pz])



# Function to get the position of the torso end-effector (torso)
def get_torso_position(t2, t3, t1=None):
    l3 = 0.02
    l4 = 139.0

    a2 = 268.0
    a3 = 79.0

    if t1 is None:
        t1 = 0.0
    else:
        t2_rounded = round(t2,2)
        t1 = constants.pairs[t2_rounded]

    t1 = t1 + pi/2

    px = cos(t1)*(cos(t2)*(l4*cos(t3) + a3) + l3*sin(t2) + a2) - sin(t1)*(sin(t2)*(l4*cos(t3) + a3) - l3*cos(t2))
    py = l4*sin(t3)
    pz = cos(t1)*(sin(t2)*(l4*cos(t3) + a3) - l3*cos(t2)) + sin(t1)*(cos(t2)*(l4*cos(t3) + a3) + l3*sin(t2) + a2)


    return array([px, py, pz])
       