"""IKPy-specific angle calculator."""

import base64
import os
from typing import Dict

import numpy as np

from server.common import pose_mapping as pose_map
from server.common.angle_calculator import AngleCalculator as CommonAngleCalculator
from server.common.kinematics import scaling_spherical as scaling
from server.ikpy import ikpy_utils as ikpyu


class AngleCalculator(CommonAngleCalculator):
    """Shared angle calculations plus IKPy arm-chain solving."""

    def __init__(self):
        super().__init__()
        script_dir = os.path.dirname(os.path.abspath(__file__))
        self.left_arm_chain, self.right_arm_chain = ikpyu.load_pepper_chains(script_dir)

    def compute_both_angles_from_payload_ikpy(self, joint_edges, payload: Dict) -> Dict[str, Dict]:
        """
        Compute BOTH human and Pepper angles from payload using IKPy.
        
        This is an alternative method that uses IKPy for inverse kinematics calculations.
        It can be used for comparison or as a fallback if the main method fails.
        
        Args:
            payload: Dictionary with "pose_keypoints" (base64-encoded array) and "hand_orientation"
        
        Returns:
             Dictionary with structure:
            {
                "human": {"ShoulderPitch_Right": ... (radians), "RWristYaw": ..., ...},
                "pepper": {"ShoulderPitch_Right": ... (radians), "RWristYaw": ..., ...},
                "forearm_direction_right": "forward"|"back"|"up"|"down"|"left"|"right"|None,
                "forearm_direction_left": "forward"|"back"|"up"|"down"|"left"|"right"|None,
                "upper_arm_direction_right": "forward"|"back"|"up"|"down"|"left"|"right"|None,
                "upper_arm_direction_left": "forward"|"back"|"up"|"down"|"left"|"right"|None,
                "missing_keypoints": ["Nose(idx:0)", "WristLeft(idx:7)", ...],
            }
        """

        joint_indices = ikpyu.get_joint_indices_from_edges(joint_edges)
        pose_b64 = payload.get("pose_keypoints")

        if pose_b64 is None:
            raise ValueError("payload missing 'pose_keypoints' field")
        
        pose_bytes = base64.b64decode(pose_b64)
        pose_array = np.frombuffer(pose_bytes, dtype=np.float32).reshape(26, 3)
        r_wrist = pose_array[4, :] 
        l_wrist = pose_array[7, :] 
        torso = pose_array[8, :] 

        missing_keypoints = self.detect_missing_keypoints(pose_array)


        # Compute IK for both arms
        angles_dict = {}
        for arm in ['right', 'left']:
            ik_result, chain = ikpyu.compute_ik_from_wrist_thor_coordinate_adjusted(
                r_wrist if arm == 'right' else l_wrist,
                torso,
                self.left_arm_chain,
                self.right_arm_chain,
                arm,
                joint_indices
            )
            angles_dict[arm] = (ik_result, chain)
        

        human_angles = {}  #TODO: Note that human angles here are the same so no infeasibilities will be detected when using ikpy, this needs to be considered at some point
        pepper_angles = {}

        for side, (ik_result, chain) in angles_dict.items():
            # Filter to only controllable joints    
            angles, names, indices = ikpyu.get_controllable_joints(chain, ik_result)   

            joint_dict = dict(zip(names, angles))

            if side == "left":
                pepper_angles["ShoulderPitch_Left"] = joint_dict["LShoulderPitch"]
                pepper_angles["ShoulderRoll_Left"]  = joint_dict["LShoulderRoll"]
                pepper_angles["ElbowYaw_Left"]      = joint_dict["LElbowYaw"]
                pepper_angles["ElbowRoll_Left"]     = joint_dict["LElbowRoll"]

            elif side == "right":
                pepper_angles["ShoulderPitch_Right"] = joint_dict["RShoulderPitch"]
                pepper_angles["ShoulderRoll_Right"]  = joint_dict["RShoulderRoll"]
                pepper_angles["ElbowYaw_Right"]      = joint_dict["RElbowYaw"]
                pepper_angles["ElbowRoll_Right"]     = joint_dict["RElbowRoll"] 
            
 
        scale_to_mm = 1000.0
            
        # Extract all keypoints needed for head, torso, and arm angles
        nose = pose_array[0, :] * scale_to_mm           # Index 0
        neck = pose_array[1, :] * scale_to_mm           # Index 1
        r_shoulder = pose_array[2, :] * scale_to_mm     # Index 2
        l_shoulder = pose_array[5, :] * scale_to_mm     # Index 5
        l_wrist = pose_array[7, :] * scale_to_mm        # Index 7
        spine_base = pose_array[9, :] * scale_to_mm     # Index 9 (Pelvis)
        r_elbow = pose_array[3, :] * scale_to_mm        # Index 3
        l_elbow = pose_array[6, :] * scale_to_mm        # Index 6


        if all(x is not None for x in [torso, neck, nose, r_shoulder, l_shoulder]):
            head_names, head_angles = scaling.compute_head_targets(
                torso, neck, nose, r_shoulder, l_shoulder
            )
            pepper_angles["HeadYaw"] = head_angles[0]
            pepper_angles["HeadPitch"] = head_angles[1]
            human_angles["HeadYaw"] = head_angles[0]
            human_angles["HeadPitch"] = head_angles[1]

        if all(x is not None for x in [spine_base, torso]):
            torso_names, torso_angles = scaling.compute_torso_targets(spine_base, torso)
            pepper_angles["TorsoPitch"] =0 #TODO: review later if this can be adjusted, for now put all torso values to 0
            pepper_angles["TorsoRoll"] = 0
            human_angles["TorsoPitch"] = 0
            human_angles["TorsoRoll"] = 0

        # Compute forearm and upper arm directions
        forearm_direction_right = pose_map._compute_discrete_direction_label(r_wrist - r_elbow)
        forearm_direction_left = pose_map._compute_discrete_direction_label(l_wrist - l_elbow)
        upper_arm_direction_right = pose_map._compute_discrete_direction_label(r_elbow - r_shoulder)
        upper_arm_direction_left = pose_map._compute_discrete_direction_label(l_elbow - l_shoulder)
            

        return {
                "human": human_angles,
                "pepper": pepper_angles,
                "forearm_direction_right": forearm_direction_right,
                "forearm_direction_left": forearm_direction_left,
                "upper_arm_direction_right": upper_arm_direction_right,
                "upper_arm_direction_left": upper_arm_direction_left,
                "missing_keypoints": missing_keypoints,
            }


