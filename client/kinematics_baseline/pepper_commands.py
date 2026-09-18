from naoqi import ALProxy
import os
import sys
import numpy as np
import threading
import time

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
CLIENT_DIR = os.path.normpath(os.path.join(THIS_DIR, ".."))
if THIS_DIR not in sys.path:
    sys.path.insert(0, THIS_DIR)
if CLIENT_DIR not in sys.path:
    sys.path.insert(0, CLIENT_DIR)
EXERCISES_DIR = os.path.normpath(os.path.join(THIS_DIR, "..", "exercises"))
if EXERCISES_DIR not in sys.path:
    sys.path.insert(0, EXERCISES_DIR)

from torso_imitation_logger import log_torso_command
from joint_limits import clamp_joint_target
from pepper_config import (
    PEPPER_IP,
    PEPPER_PORT,
    HEAD_CHAIN_SPEED_FRACTION,
    TORSO_CHAIN_SPEED_FRACTION,
    ARM_CHAIN_SPEED_FRACTION,
    HAND_CHAIN_SPEED_FRACTION,
    POSE_SMOOTHING_DEADBAND_RAD,
    POSE_SMOOTHING_MAX_DELTA_RAD_PER_TICK,
    POSE_SMOOTHING_ALPHA_SLOW,
    POSE_SMOOTHING_ALPHA_FAST,
    POSE_SMOOTHING_FAST_DELTA_RAD,
)

IP = PEPPER_IP
PORT = PEPPER_PORT

motion = ALProxy("ALMotion", IP, PORT)
tracker = ALProxy("ALTracker", IP, PORT)
memory = ALProxy("ALMemory", IP, PORT)

HEAD_JOINTS = ["HeadYaw", "HeadPitch"]
RIGHT_ARM_JOINTS = ["RShoulderPitch", "RShoulderRoll", "RElbowYaw", "RElbowRoll"]
LEFT_ARM_JOINTS = ["LShoulderPitch", "LShoulderRoll", "LElbowYaw", "LElbowRoll"]
TORSO_JOINTS = ["KneePitch", "HipPitch", "HipRoll"]

HEAD_SPEED_FRACTION = float(HEAD_CHAIN_SPEED_FRACTION)
ARM_SPEED_FRACTION = float(ARM_CHAIN_SPEED_FRACTION)
TORSO_SPEED_FRACTION = float(TORSO_CHAIN_SPEED_FRACTION)
HAND_SPEED_FRACTION = float(HAND_CHAIN_SPEED_FRACTION)

_SMOOTHING_DEADBAND_RAD = max(0.0, float(POSE_SMOOTHING_DEADBAND_RAD))
_SMOOTHING_MAX_DELTA_RAD_PER_TICK = max(0.0, float(POSE_SMOOTHING_MAX_DELTA_RAD_PER_TICK))
_SMOOTHING_ALPHA_SLOW = min(max(float(POSE_SMOOTHING_ALPHA_SLOW), 0.0), 1.0)
_SMOOTHING_ALPHA_FAST = min(max(float(POSE_SMOOTHING_ALPHA_FAST), 0.0), 1.0)
if _SMOOTHING_ALPHA_FAST < _SMOOTHING_ALPHA_SLOW:
    _SMOOTHING_ALPHA_FAST = _SMOOTHING_ALPHA_SLOW
_SMOOTHING_FAST_DELTA_RAD = max(float(POSE_SMOOTHING_FAST_DELTA_RAD), 1e-6)
_SHAPER_EPS = 1e-9

_command_shape_lock = threading.Lock()
_filtered_angles_by_joint = {}
_last_sent_angles_by_joint = {}
_latency_metrics_lock = threading.Lock()
_latency_metrics = {
    "command_count": 0,
    "total_latency_ms_sum": 0.0,
    "total_latency_samples": 0,
}


def _filter_valid_joint_targets(names, angles, speeds=None):
    valid_names = []
    valid_angles = []
    valid_speeds = [] if speeds is not None else None
    for i, (name, angle) in enumerate(zip(names, angles)):
        try:
            angle_value = float(angle)
        except Exception:
            continue
        if np.isnan(angle_value):
            continue
        angle_value = clamp_joint_target(name, angle_value)
        valid_names.append(name)
        valid_angles.append(angle_value)
        if valid_speeds is not None:
            valid_speeds.append(float(speeds[i]) if i < len(speeds) else ARM_SPEED_FRACTION)
    return valid_names, valid_angles, valid_speeds


def reset_joint_command_shaper():
    with _command_shape_lock:
        _filtered_angles_by_joint.clear()
        _last_sent_angles_by_joint.clear()


def reset_total_latency_metrics():
    with _latency_metrics_lock:
        _latency_metrics["command_count"] = 0
        _latency_metrics["total_latency_ms_sum"] = 0.0
        _latency_metrics["total_latency_samples"] = 0


def _record_total_latency_metrics(capture_ts_client_us, setangles_call_us):
    with _latency_metrics_lock:
        _latency_metrics["command_count"] += 1
        if capture_ts_client_us is None:
            return
        try:
            capture_us = float(capture_ts_client_us)
        except Exception:
            return
        latency_ms = (float(setangles_call_us) - capture_us) / 1000.0
        if latency_ms < 0.0:
            return
        _latency_metrics["total_latency_ms_sum"] += latency_ms
        _latency_metrics["total_latency_samples"] += 1


def consume_total_latency_metrics():
    with _latency_metrics_lock:
        command_count = int(_latency_metrics["command_count"])
        total_latency_ms_sum = float(_latency_metrics["total_latency_ms_sum"])
        total_latency_samples = int(_latency_metrics["total_latency_samples"])
        _latency_metrics["command_count"] = 0
        _latency_metrics["total_latency_ms_sum"] = 0.0
        _latency_metrics["total_latency_samples"] = 0

    avg_total_latency_ms = None
    if total_latency_samples > 0:
        avg_total_latency_ms = total_latency_ms_sum / float(total_latency_samples)
    return command_count, avg_total_latency_ms


def _clamp(value, lower, upper):
    if value < lower:
        return lower
    if value > upper:
        return upper
    return value


def _adaptive_alpha(abs_delta):
    if abs_delta <= 0.0:
        return _SMOOTHING_ALPHA_SLOW
    ratio = _clamp(abs_delta / _SMOOTHING_FAST_DELTA_RAD, 0.0, 1.0)
    return _SMOOTHING_ALPHA_SLOW + ((_SMOOTHING_ALPHA_FAST - _SMOOTHING_ALPHA_SLOW) * ratio)


def _shape_joint_targets(names, angles):
    with _command_shape_lock:
        shaped_angles = []
        has_change = False
        for name, raw in zip(names, angles):
            raw_value = float(raw)
            prev_filtered = _filtered_angles_by_joint.get(name)
            prev_sent = _last_sent_angles_by_joint.get(name)

            if prev_filtered is None:
                filtered = raw_value
            else:
                alpha = _adaptive_alpha(abs(raw_value - prev_filtered))
                filtered = prev_filtered + (alpha * (raw_value - prev_filtered))

            if prev_sent is None:
                shaped = filtered
            else:
                step = filtered - prev_sent
                if _SMOOTHING_MAX_DELTA_RAD_PER_TICK > 0.0:
                    step = _clamp(
                        step,
                        -_SMOOTHING_MAX_DELTA_RAD_PER_TICK,
                        _SMOOTHING_MAX_DELTA_RAD_PER_TICK,
                    )
                shaped = prev_sent + step
                if abs(shaped - prev_sent) < _SMOOTHING_DEADBAND_RAD:
                    shaped = prev_sent

            # Reapply hard limits after filtering and step limiting so the
            # value passed to ALMotion can never leave the permitted range.
            shaped = clamp_joint_target(name, shaped)
            _filtered_angles_by_joint[name] = shaped
            shaped_angles.append(shaped)

            if prev_sent is None or abs(shaped - prev_sent) > _SHAPER_EPS:
                has_change = True

        if not has_change:
            return None

        for name, shaped in zip(names, shaped_angles):
            _last_sent_angles_by_joint[name] = shaped

    return shaped_angles


def send_joint_targets_speed(
    names,
    angles,
    speed_fraction=ARM_SPEED_FRACTION,
    speeds=None,
    frame_id=None,
    capture_ts_client_us=None,
):
    """Send joint targets to Pepper.

    If *speeds* is a list of per-joint speed fractions it is forwarded directly
    to ``motion.setAngles`` so that each chain can run at its own speed.  When
    *speeds* is ``None`` the scalar *speed_fraction* is used for every joint
    (legacy behaviour).
    """
    names, angles, filtered_speeds = _filter_valid_joint_targets(names, angles, speeds)
    if not angles:
        return
    shaped_angles = _shape_joint_targets(names, angles)
    if shaped_angles is None:
        return
    speed_arg = filtered_speeds if filtered_speeds is not None else speed_fraction
    try:
        setangles_call_us = int(time.time() * 1000000.0)
        motion.setAngles(names, shaped_angles, speed_arg)
        _record_total_latency_metrics(capture_ts_client_us, setangles_call_us)
        command_map = dict(zip(names, shaped_angles))
        if "KneePitch" in command_map and "HipPitch" in command_map:
            log_torso_command(command_map["KneePitch"], command_map["HipPitch"])
    except BaseException, err:
        print("Error in joint targets: " + str(shaped_angles))
        print err


def disable_security_settings():
    motion.setExternalCollisionProtectionEnabled("Arms", False)
    motion.setOrthogonalSecurityDistance(0.4)
    motion.setTangentialSecurityDistance(0.1)
    print("Security settings are disabled!")

def enable_security_setting():
    motion.setExternalCollisionProtectionEnabled("Arms", True)
    motion.setOrthogonalSecurityDistance(0.05)
    motion.setTangentialSecurityDistance(0.05)
    print("Security settings are enabled!")

def send_to_stand():
    try: 
        posture = ALProxy("ALRobotPosture", IP, PORT)
        posture.goToPosture("Stand", 0.4)
    except BaseException, err:
        print("Frame of error: " + str(frame_num))

def get_pos_sensed():
    global pos_sensed
    return pos_sensed

def get_pos_inverse(): 
    global pos_inverse
    return pos_inverse

def get_angles_calc():
    global angles_calc
    return angles_calc

def send_head_values(HeadYaw, HeadPitch):

    names = ["HeadYaw", "HeadPitch"]
    angles = [HeadYaw, HeadPitch]

    try: 
        motion.setAngles(names, angles, fractionMaxSpeed)
    except BaseException, err:
        print err

def send_rightArm_values(t1, t2, t3, t4, hand=0.5, frame_num=0):

    names = ["RShoulderPitch", "RShoulderRoll", "RElbowYaw", "RElbowRoll", "RHand"] 
    angles = [t1, t2, t3, t4, hand]

    try: 
        motion.setAngles(names, angles, fractionMaxSpeed)
    except BaseException, err:
        print err

def send_rightHand_values(hand):

    names = ["RHand"]
    angles = [hand]
    names, angles, _ = _filter_valid_joint_targets(names, angles)
    if not angles:
        return

    try:
        motion.setAngles(names, angles, HAND_SPEED_FRACTION)
    except BaseException, err:
        print err

def send_leftHand_values(hand):

    names = ["LHand"]
    angles = [hand]
    names, angles, _ = _filter_valid_joint_targets(names, angles)
    if not angles:
        return

    try:
        motion.setAngles(names, angles, HAND_SPEED_FRACTION)
    except BaseException, err:
        print err

def send_leftArm_values(t1, t2, t3, t4, hand=0.5, frame_num=0):

    names = ["LShoulderPitch", "LShoulderRoll", "LElbowYaw", "LElbowRoll", "LHand"] 
    angles = [t1, t2, t3, t4, hand]

    try: 
        motion.setAngles(names, angles, ARM_SPEED_FRACTION)
    except BaseException, err:
        print err

def send_torso_values(t1, t2, t3):

    names = ["KneePitch", "HipPitch", "HipRoll"]
    angles = [t1, t2, t3]
    names, angles, _ = _filter_valid_joint_targets(names, angles)
    if not angles:
        return
   
    try: 
        motion.setAngles(names, angles, TORSO_SPEED_FRACTION)
        log_torso_command(t1, t2)
    except BaseException, err:
        #print("Frame of error: " + str(frame_num))
        print err

def send_head_values_speed(HeadYaw, HeadPitch):
    names, angles, _ = _filter_valid_joint_targets(HEAD_JOINTS, [HeadYaw, HeadPitch])
    if not angles:
        return
    try:
        motion.setAngles(names, angles, HEAD_SPEED_FRACTION)
    except BaseException, err:
        print("Error in head joint values:" + str(angles))
        print err



def send_rightArm_values_speed(t1, t2, t3, t4, hand=0.5, frame_num=0):
    # For low-latency mode, hands are intentionally ignored.
    names, angles, _ = _filter_valid_joint_targets(RIGHT_ARM_JOINTS, [t1, t2, t3, t4])
    if not angles:
        return
    try:
        motion.setAngles(names, angles, ARM_SPEED_FRACTION)
    except BaseException, err:
        print("Error in right arm joints: " + str(angles))
        print err
    

def send_leftArm_values_speed(t1, t2, t3, t4, hand=0.5, frame_num=0):
    # For low-latency mode, hands are intentionally ignored.
    names, angles, _ = _filter_valid_joint_targets(LEFT_ARM_JOINTS, [t1, t2, t3, t4])
    if not angles:
        return
    try:
        motion.setAngles(names, angles, ARM_SPEED_FRACTION)
    except BaseException, err:
        print("Error in left arm joints: " + str(angles))
        print err


def send_torso_values_speed(t1, t2, t3):
    names, angles, _ = _filter_valid_joint_targets(TORSO_JOINTS, [t1, t2, t3])
    if not angles:
        return
    try:
        motion.setAngles(names, angles, TORSO_SPEED_FRACTION)
        command_map = dict(zip(names, angles))
        if "KneePitch" in command_map and "HipPitch" in command_map:
            log_torso_command(command_map["KneePitch"], command_map["HipPitch"])
    except BaseException, err:
        print("Error in torso joints: " + str(angles))
        print err


def send_torso_values_speed_t2t3(t2, t3):
    names = ["HipPitch", "HipRoll"]
    angles = [t2, t3]
    names, angles, _ = _filter_valid_joint_targets(names, angles)
    if not angles:
        return

    try:
        motion.setAngles(names, angles, TORSO_SPEED_FRACTION)
    except BaseException, err:
        #print("Frame of error: " + str(frame_num))
        print err

def send_torso_t3_speed(t3):
    names = ["HipRoll"]
    angles = [t3]
    names, angles, _ = _filter_valid_joint_targets(names, angles)
    if not angles:
        return

    try:
        motion.setAngles(names, angles, TORSO_SPEED_FRACTION)
    except BaseException, err:
        #print("Frame of error: " + str(frame_num))
        print err


def get_current_angle(name):
    return motion.getAngles(name, True)

def get_current_position(name, frame):
    return motion.getPosition(name, frame, True)

def look_at(x, y, z, frame):
    useWholeBody = False
    tracker.lookAt([x, y, z], frame, 0.5, useWholeBody)
