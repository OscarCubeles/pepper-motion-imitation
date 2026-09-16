import os
import sys
import math

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
CLIENT_DIR = os.path.normpath(os.path.join(THIS_DIR, ".."))
if CLIENT_DIR not in sys.path:
    sys.path.insert(0, CLIENT_DIR)

from pepper_config import (
    POSE_NULLSPACE_HYSTERESIS_RAD,
    POSE_NULLSPACE_R_SHOULDER_ROLL_TRIGGER_RAD,
    POSE_NULLSPACE_L_SHOULDER_ROLL_TRIGGER_RAD,
    POSE_NULLSPACE_R_ELBOW_ROLL_TRIGGER_RAD,
    POSE_NULLSPACE_L_ELBOW_ROLL_TRIGGER_RAD,
)

_HYSTERESIS_MARGIN_RAD = max(0.0, float(POSE_NULLSPACE_HYSTERESIS_RAD))
_ZERO_RAD = 0.0
_HALF_PI_RAD = float(math.pi / 2.0)
_LOCK_CANDIDATES_SHOULDER_PITCH = (-_HALF_PI_RAD, _ZERO_RAD, _HALF_PI_RAD)
_LOCK_CANDIDATES_ELBOW_YAW = (-_HALF_PI_RAD, _ZERO_RAD, _HALF_PI_RAD)

_GUARD_STATE = {
    "right_shoulder_lock": {"active": False, "selected_lock_rad": None},
    "left_shoulder_lock": {"active": False, "selected_lock_rad": None},
    "right_elbow_lock": {"active": False, "selected_lock_rad": None},
    "left_elbow_lock": {"active": False, "selected_lock_rad": None},
}


def _is_finite_number(value):
    try:
        number = float(value)
    except Exception:
        return False
    if number != number:
        return False
    return number != float("inf") and number != float("-inf")


def _build_joint_index(names):
    index_by_name = {}
    for i, name in enumerate(names):
        index_by_name[str(name)] = i
    return index_by_name


def _get_joint_value(index_by_name, angles, joint_name):
    idx = index_by_name.get(joint_name)
    if idx is None or idx >= len(angles):
        return None
    value = angles[idx]
    if not _is_finite_number(value):
        return None
    return float(value)


def _set_joint_value(index_by_name, angles, joint_name, value):
    idx = index_by_name.get(joint_name)
    if idx is None or idx >= len(angles):
        return False
    angles[idx] = float(value)
    return True


def _choose_nearest_lock(current_angle, candidates):
    if not candidates:
        return None
    if current_angle is None or (not _is_finite_number(current_angle)):
        return _ZERO_RAD

    current = float(current_angle)
    best_candidate = None
    best_distance = None
    for candidate in candidates:
        candidate_value = float(candidate)
        distance = abs(current - candidate_value)
        if best_distance is None or distance < best_distance:
            best_distance = distance
            best_candidate = candidate_value
            continue
        if distance == best_distance:
            # Tie rule: prefer neutral lock.
            if abs(candidate_value - _ZERO_RAD) < abs(best_candidate - _ZERO_RAD):
                best_candidate = candidate_value
    return best_candidate


def _update_leq_guard(is_active, observed_value, trigger_value, margin):
    if observed_value is None:
        return bool(is_active)
    if is_active:
        # Release only above trigger + margin.
        return bool(observed_value <= (trigger_value + margin))
    return bool(observed_value <= trigger_value)


def _update_geq_guard(is_active, observed_value, trigger_value, margin):
    if observed_value is None:
        return bool(is_active)
    if is_active:
        # Release only below trigger - margin.
        return bool(observed_value >= (trigger_value - margin))
    return bool(observed_value >= trigger_value)


def _print_lock_activation(guard_label, trigger_joint, comparator, trigger_value, observed_value, target_joint, lock_value):
    observed_text = "n/a"
    if _is_finite_number(observed_value):
        observed_text = "%.4f rad (%.1f deg)" % (float(observed_value), math.degrees(float(observed_value)))
    lock_text = "n/a"
    if _is_finite_number(lock_value):
        lock_text = "%.4f rad (%.1f deg)" % (float(lock_value), math.degrees(float(lock_value)))
    print(
        "[nullspace] lock reached: %s | trigger: %s %s %.4f rad | observed: %s | %s -> %s" % (
            str(guard_label),
            str(trigger_joint),
            str(comparator),
            float(trigger_value),
            observed_text,
            str(target_joint),
            lock_text,
        )
    )


def reset_nullspace_constraints_state():
    _GUARD_STATE["right_shoulder_lock"]["active"] = False
    _GUARD_STATE["right_shoulder_lock"]["selected_lock_rad"] = None
    _GUARD_STATE["left_shoulder_lock"]["active"] = False
    _GUARD_STATE["left_shoulder_lock"]["selected_lock_rad"] = None
    _GUARD_STATE["right_elbow_lock"]["active"] = False
    _GUARD_STATE["right_elbow_lock"]["selected_lock_rad"] = None
    _GUARD_STATE["left_elbow_lock"]["active"] = False
    _GUARD_STATE["left_elbow_lock"]["selected_lock_rad"] = None


def apply_nullspace_constraints(names, angles):
    if names is None or angles is None:
        return names, angles
    if not names or not angles:
        return names, angles

    index_by_name = _build_joint_index(names)

    right_shoulder_state = _GUARD_STATE["right_shoulder_lock"]
    right_shoulder_roll = _get_joint_value(index_by_name, angles, "RShoulderRoll")
    was_active = bool(right_shoulder_state["active"])
    is_active = _update_leq_guard(
        was_active,
        right_shoulder_roll,
        float(POSE_NULLSPACE_R_SHOULDER_ROLL_TRIGGER_RAD),
        _HYSTERESIS_MARGIN_RAD,
    )
    right_shoulder_state["active"] = bool(is_active)
    if (not was_active) and is_active:
        current_pitch = _get_joint_value(index_by_name, angles, "RShoulderPitch")
        right_shoulder_state["selected_lock_rad"] = _choose_nearest_lock(
            current_pitch,
            _LOCK_CANDIDATES_SHOULDER_PITCH,
        )
        _print_lock_activation(
            "right_shoulder",
            "RShoulderRoll",
            "<=",
            float(POSE_NULLSPACE_R_SHOULDER_ROLL_TRIGGER_RAD),
            right_shoulder_roll,
            "RShoulderPitch",
            right_shoulder_state.get("selected_lock_rad"),
        )
    elif was_active and (not is_active):
        right_shoulder_state["selected_lock_rad"] = None

    if right_shoulder_state["active"]:
        lock_value = right_shoulder_state.get("selected_lock_rad")
        if lock_value is None:
            lock_value = _choose_nearest_lock(None, _LOCK_CANDIDATES_SHOULDER_PITCH)
            right_shoulder_state["selected_lock_rad"] = lock_value
        _set_joint_value(
            index_by_name,
            angles,
            "RShoulderPitch", # Hardcoding the pitch when to lock is triggered.
            float(lock_value), 
        )

    left_shoulder_state = _GUARD_STATE["left_shoulder_lock"]
    left_shoulder_roll = _get_joint_value(index_by_name, angles, "LShoulderRoll")
    was_active = bool(left_shoulder_state["active"])
    is_active = _update_geq_guard(
        was_active,
        left_shoulder_roll,
        float(POSE_NULLSPACE_L_SHOULDER_ROLL_TRIGGER_RAD),
        _HYSTERESIS_MARGIN_RAD,
    )
    left_shoulder_state["active"] = bool(is_active)
    if (not was_active) and is_active:
        current_pitch = _get_joint_value(index_by_name, angles, "LShoulderPitch")
        left_shoulder_state["selected_lock_rad"] = _choose_nearest_lock(
            current_pitch,
            _LOCK_CANDIDATES_SHOULDER_PITCH,
        )
        _print_lock_activation(
            "left_shoulder",
            "LShoulderRoll",
            ">=",
            float(POSE_NULLSPACE_L_SHOULDER_ROLL_TRIGGER_RAD),
            left_shoulder_roll,
            "LShoulderPitch",
            left_shoulder_state.get("selected_lock_rad"),
        )
    elif was_active and (not is_active):
        left_shoulder_state["selected_lock_rad"] = None

    if left_shoulder_state["active"]:
        lock_value = left_shoulder_state.get("selected_lock_rad")
        if lock_value is None:
            lock_value = _choose_nearest_lock(None, _LOCK_CANDIDATES_SHOULDER_PITCH)
            left_shoulder_state["selected_lock_rad"] = lock_value
        _set_joint_value(
            index_by_name,
            angles,
            "LShoulderPitch",
            float(lock_value),
        )

    right_elbow_state = _GUARD_STATE["right_elbow_lock"]
    right_elbow_roll = _get_joint_value(index_by_name, angles, "RElbowRoll")
    was_active = bool(right_elbow_state["active"])
    is_active = _update_leq_guard(
        was_active,
        right_elbow_roll,
        float(POSE_NULLSPACE_R_ELBOW_ROLL_TRIGGER_RAD),
        _HYSTERESIS_MARGIN_RAD,
    )
    right_elbow_state["active"] = bool(is_active)
    if (not was_active) and is_active:
        current_yaw = _get_joint_value(index_by_name, angles, "RElbowYaw")
        right_elbow_state["selected_lock_rad"] = _choose_nearest_lock(
            current_yaw,
            _LOCK_CANDIDATES_ELBOW_YAW,
        )
        _print_lock_activation(
            "right_elbow",
            "RElbowRoll",
            "<=",
            float(POSE_NULLSPACE_R_ELBOW_ROLL_TRIGGER_RAD),
            right_elbow_roll,
            "RElbowYaw",
            right_elbow_state.get("selected_lock_rad"),
        )
    elif was_active and (not is_active):
        right_elbow_state["selected_lock_rad"] = None

    if right_elbow_state["active"]:
        lock_value = right_elbow_state.get("selected_lock_rad")
        if lock_value is None:
            lock_value = _choose_nearest_lock(None, _LOCK_CANDIDATES_ELBOW_YAW)
            right_elbow_state["selected_lock_rad"] = lock_value
        _set_joint_value(
            index_by_name,
            angles,
            "RElbowYaw",
            float(lock_value),
        )

    left_elbow_state = _GUARD_STATE["left_elbow_lock"]
    left_elbow_roll = _get_joint_value(index_by_name, angles, "LElbowRoll")
    was_active = bool(left_elbow_state["active"])
    is_active = _update_geq_guard(
        was_active,
        left_elbow_roll,
        float(POSE_NULLSPACE_L_ELBOW_ROLL_TRIGGER_RAD),
        _HYSTERESIS_MARGIN_RAD,
    )
    left_elbow_state["active"] = bool(is_active)
    if (not was_active) and is_active:
        current_yaw = _get_joint_value(index_by_name, angles, "LElbowYaw")
        left_elbow_state["selected_lock_rad"] = _choose_nearest_lock(
            current_yaw,
            _LOCK_CANDIDATES_ELBOW_YAW,
        )
        _print_lock_activation(
            "left_elbow",
            "LElbowRoll",
            ">=",
            float(POSE_NULLSPACE_L_ELBOW_ROLL_TRIGGER_RAD),
            left_elbow_roll,
            "LElbowYaw",
            left_elbow_state.get("selected_lock_rad"),
        )
    elif was_active and (not is_active):
        left_elbow_state["selected_lock_rad"] = None

    if left_elbow_state["active"]:
        lock_value = left_elbow_state.get("selected_lock_rad")
        if lock_value is None:
            lock_value = _choose_nearest_lock(None, _LOCK_CANDIDATES_ELBOW_YAW)
            left_elbow_state["selected_lock_rad"] = lock_value
        _set_joint_value(
            index_by_name,
            angles,
            "LElbowYaw",
            float(lock_value),
        )

    return names, angles