"""Validation and mapping for optional server-computed Pepper angles."""

import math


SERVER_ANGLE_MAPPING = (
    ("ShoulderPitch_Right", "RShoulderPitch"),
    ("ShoulderRoll_Right", "RShoulderRoll"),
    ("ElbowYaw_Right", "RElbowYaw"),
    ("ElbowRoll_Right", "RElbowRoll"),
    ("RWristYaw", "RWristYaw"),
    ("RHand", "RHand"),
    ("ShoulderPitch_Left", "LShoulderPitch"),
    ("ShoulderRoll_Left", "LShoulderRoll"),
    ("ElbowYaw_Left", "LElbowYaw"),
    ("ElbowRoll_Left", "LElbowRoll"),
    ("LWristYaw", "LWristYaw"),
    ("LHand", "LHand"),
    ("HeadYaw", "HeadYaw"),
    ("HeadPitch", "HeadPitch"),
    ("TorsoPitch", "HipPitch"),
    ("TorsoRoll", "HipRoll"),
)

REQUIRED_SERVER_ANGLE_KEYS = (
    "ShoulderPitch_Right",
    "ShoulderRoll_Right",
    "ElbowYaw_Right",
    "ElbowRoll_Right",
    "ShoulderPitch_Left",
    "ShoulderRoll_Left",
    "ElbowYaw_Left",
    "ElbowRoll_Left",
)


def _finite_float(value):
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(number) or math.isinf(number):
        return None
    return number


def build_server_joint_targets(angles):
    """Return validated NAOqi targets, or empty lists for an unusable payload."""
    if not isinstance(angles, dict):
        return [], [], []
    pepper_angles = angles.get("pepper")
    if not isinstance(pepper_angles, dict):
        return [], [], []

    # A partial arm command must not disable the established local-IK path.
    # Proposed and IKPy servers both produce these eight core joints when their
    # server-side calculation succeeds.
    for angle_key in REQUIRED_SERVER_ANGLE_KEYS:
        if angle_key not in pepper_angles:
            return [], [], []
        if _finite_float(pepper_angles[angle_key]) is None:
            return [], [], []

    names = []
    values = []
    speeds = []
    for angle_key, naoqi_name in SERVER_ANGLE_MAPPING:
        if angle_key not in pepper_angles or pepper_angles[angle_key] is None:
            continue
        value = _finite_float(pepper_angles[angle_key])
        if value is None:
            continue
        names.append(naoqi_name)
        values.append(value)
        speeds.append(0.5)
    return names, values, speeds
