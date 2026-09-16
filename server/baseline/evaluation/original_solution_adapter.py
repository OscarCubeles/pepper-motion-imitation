import importlib
import math
import sys
from pathlib import Path

import numpy as np


CURRENT_FILE = Path(__file__).resolve()
REPO_ROOT = CURRENT_FILE.parents[3]
ORIGINAL_KINEMATICS_DIR = (
    REPO_ROOT
    / "06-original_solution"
    / "Pepper-HumanImitation-System"
    / "client"
    / "kinematics-1.3-clean"
)


def _load_original_modules():
    if not ORIGINAL_KINEMATICS_DIR.exists():
        raise RuntimeError(f"Original kinematics folder not found: {ORIGINAL_KINEMATICS_DIR}")

    original_path = str(ORIGINAL_KINEMATICS_DIR)
    if original_path not in sys.path:
        sys.path.insert(0, original_path)

    # Do not import original scaling_spherical.py here: it imports pepper_commands,
    # which is live Pepper/NAOqi Python 2 code. The math below mirrors the original
    # arm path while keeping the original repository untouched.
    tfm = importlib.import_module("transformation_matrices")
    ik = importlib.import_module("inverse_kinematics_rangeupdate")
    mf = importlib.import_module("math_functions")
    return tfm, ik, mf


_TFM = None
_IK = None
_MF = None


def _ensure_original_modules_loaded():
    """Load the optional historical implementation only when it is evaluated."""
    global _TFM, _IK, _MF
    if _TFM is None or _IK is None or _MF is None:
        _TFM, _IK, _MF = _load_original_modules()


def _safe_vector(value):
    if value is None:
        return None
    vector = np.asarray(value, dtype=np.float64)
    if vector.shape[0] < 3 or not np.isfinite(vector[:3]).all():
        return None
    return vector[:3]


def _to_original_input_units(vector_mm):
    # The original dl_to_pepper_torch.py sends pose coordinates divided by 1000,
    # and original transformation_matrices.set_joint_in_peppers_coordinates()
    # multiplies by 1000. The annotations are stored in millimeters.
    return vector_mm / 1000.0


def _unit_scale(vector, length):
    norm = np.linalg.norm(vector)
    if norm < 1e-8 or not np.isfinite(norm):
        raise ValueError("invalid zero-length vector")
    return length * vector / norm


def _to_spherical(position):
    radius = np.linalg.norm(position)
    if radius < 1e-8 or not np.isfinite(radius):
        raise ValueError("invalid spherical radius")
    phi = np.arctan2(position[1], position[0])
    theta = np.arccos(position[2] / radius)
    return radius, phi, theta


def _to_cartesian(radius, phi, theta):
    return np.array(
        [
            radius * np.sin(theta) * np.cos(phi),
            radius * np.sin(theta) * np.sin(phi),
            radius * np.cos(theta),
        ],
        dtype=np.float64,
    )


def _get_wrist_fitted(elbow_fitted, wrist, elbow, arm):
    if arm == "right":
        pepper_shoulder = np.array([-57.0, -149.74, 86.82], dtype=np.float64)
    elif arm == "left":
        pepper_shoulder = np.array([-57.0, 149.74, 86.82], dtype=np.float64)
    else:
        raise ValueError(arm)

    elbow_new = pepper_shoulder + elbow_fitted
    t1, t2 = _IK.get_arm_partial_angles(elbow_new[0], elbow_new[1], elbow_new[2], arm)

    matrix = _TFM.trans_elbow_workspace(t1, t2, elbow_fitted, pepper_shoulder, arm)
    rot_x = _MF.rx_calc(np.pi / 2, 3)
    rot_y = _MF.ry_calc(np.pi / 2, 3)
    wrist = wrist - elbow - pepper_shoulder
    wrist_spherical = np.transpose(rot_y).dot(np.transpose(rot_x).dot(np.transpose(matrix).dot(wrist)))

    wr_eq, wphi_eq, wtheta_eq = _to_spherical(wrist_spherical)
    if wtheta_eq < 0.008:
        wtheta_eq = 0.0087
    elif wtheta_eq > 1.56:
        wtheta_eq = 1.55

    if wphi_eq < -2.09:
        wphi_eq = -2.08
    elif wphi_eq > 2.06:
        wphi_eq = 2.05

    wrist_fitted = matrix.dot(rot_x.dot(rot_y.dot(_to_cartesian(wr_eq, wphi_eq, wtheta_eq)))) + elbow_new
    return t1, t2, wrist_fitted


def _get_elbow_wrist_fitted(elbow, wrist, arm):
    if arm == "right":
        rot_x = _MF.rx_calc(np.pi / 2, 3)
    elif arm == "left":
        rot_x = _MF.rx_calc(-np.pi / 2, 3)
    else:
        raise ValueError(arm)

    elbow_spherical = np.transpose(rot_x).dot(elbow)
    er_eq, ephi_eq, etheta_eq = _to_spherical(elbow_spherical)

    if etheta_eq < 0.0:
        etheta_eq = 0.0079
    elif etheta_eq > 1.48:
        etheta_eq = 1.48

    if etheta_eq > 0.06 and etheta_eq < 0.165:
        if ephi_eq < -3.15:
            ephi_eq = -3.14
        elif ephi_eq > 3.15:
            ephi_eq = 3.14

    if ephi_eq < -2.1 or ephi_eq > 2.1:
        ephi_eq1 = -2.09
        elbow_fitted1 = rot_x.dot(_to_cartesian(er_eq, ephi_eq1, etheta_eq))
        ephi_eq2 = 2.09
        elbow_fitted2 = rot_x.dot(_to_cartesian(er_eq, ephi_eq2, etheta_eq))

        t1_1, t2_1, wrist_fitted1 = _get_wrist_fitted(elbow_fitted1, wrist, elbow, arm)
        t1_2, t2_2, wrist_fitted2 = _get_wrist_fitted(elbow_fitted2, wrist, elbow, arm)
        dist1 = np.linalg.norm(wrist - wrist_fitted1)
        dist2 = np.linalg.norm(wrist - wrist_fitted2)
        if dist1 <= dist2:
            return t1_1, t2_1, elbow_fitted1, wrist_fitted1
        return t1_2, t2_2, elbow_fitted2, wrist_fitted2

    elbow_fitted = rot_x.dot(_to_cartesian(er_eq, ephi_eq, etheta_eq))
    t1, t2, wrist_fitted = _get_wrist_fitted(elbow_fitted, wrist, elbow, arm)
    return t1, t2, elbow_fitted, wrist_fitted


def compute_arm_angles(torso, shoulder, elbow, wrist, other_shoulder, side):
    _ensure_original_modules_loaded()
    torso = _safe_vector(torso)
    shoulder = _safe_vector(shoulder)
    elbow = _safe_vector(elbow)
    wrist = _safe_vector(wrist)
    other_shoulder = _safe_vector(other_shoulder)
    if any(item is None for item in (torso, shoulder, elbow, wrist, other_shoulder)):
        raise ValueError("missing coordinates")

    torso = _TFM.set_joint_in_peppers_coordinates(_to_original_input_units(torso))
    shoulder_new = _TFM.set_joint_in_peppers_coordinates(_to_original_input_units(shoulder)) - torso
    elbow_new = _TFM.set_joint_in_peppers_coordinates(_to_original_input_units(elbow)) - torso
    wrist_new = _TFM.set_joint_in_peppers_coordinates(_to_original_input_units(wrist)) - torso
    other_shoulder_new = _TFM.set_joint_in_peppers_coordinates(_to_original_input_units(other_shoulder)) - torso

    if side == "right":
        pepper_shoulder = np.array([-57.0, -149.74, 86.82], dtype=np.float64)
        rot_mat = _TFM.rotation_mat_arms(shoulder_new, other_shoulder_new)
    elif side == "left":
        pepper_shoulder = np.array([-57.0, 149.74, 86.82], dtype=np.float64)
        rot_mat = _TFM.rotation_mat_arms(other_shoulder_new, shoulder_new)
    else:
        raise ValueError(side)

    shoulder_new = rot_mat.dot(shoulder_new)
    elbow_new = rot_mat.dot(elbow_new) - shoulder_new
    wrist_new = rot_mat.dot(wrist_new) - shoulder_new - elbow_new

    elbow_scaled = _unit_scale(elbow_new, 181.2)
    wrist_scaled = _unit_scale(wrist_new, 150.0) + elbow_scaled + pepper_shoulder

    t1, t2, _, wrist_fitted = _get_elbow_wrist_fitted(elbow_scaled, wrist_scaled, side)
    t3, t4 = _IK.get_arm_t3t4_angles(t1, t2, wrist_fitted[0], wrist_fitted[1], wrist_fitted[2], side)

    angles = {
        "shoulder_pitch": _clean_angle(t1),
        "shoulder_roll": _clean_angle(t2),
        "elbow_yaw": _clean_angle(t3),
        "elbow_roll": _clean_angle(t4),
        "wrist_yaw": 0.0,
    }
    if any(value is None for value in angles.values()):
        raise ValueError("original solution produced invalid angle")
    return angles


def _clean_angle(value):
    try:
        numeric = float(np.real(value))
    except (TypeError, ValueError):
        return None
    if math.isnan(numeric) or math.isinf(numeric):
        return None
    return numeric
