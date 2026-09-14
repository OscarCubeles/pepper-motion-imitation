import os
import sys


# Imitation Client variables
SERVER_HOST = "localhost"
PORT = 8080
MIRRORING_IMITATION = False
REQUEST_MESSAGE = "keypoints"

# Feedback Client variables
DEFAULT_FEEDBACK_SERVER_URL = "http://localhost:8091"
DEFAULT_TIMEOUT_SEC = 8.0
DEFAULT_MAX_RETRIES = 1
DEFAULT_RETRY_DELAY_SEC = 0.2

# Pose Demonstration variables
DEFAULT_EXERCISE_NAME = "arms_rise"

# Pose Stream Runtime
JOINT_COUNT_BASE = 10
JOINT_COUNT_WITH_HAND_TIPS = 12
PARAMS_PER_JOINT = 3
FLOAT_BYTES = 4
JOINT_BYTES = PARAMS_PER_JOINT * FLOAT_BYTES
EXPECTED_BYTES_BASE = JOINT_COUNT_BASE * JOINT_BYTES
EXPECTED_BYTES_WITH_HAND_TIPS = JOINT_COUNT_WITH_HAND_TIPS * JOINT_BYTES
STOP_KEY = "q"
RUN_DURATION_SEC = None
ENABLE_ROBOT_IMITATION = True
MANAGE_PEPPER_SECURITY = True
SEND_TO_STAND_ON_EXIT = True
MODE_LABEL = "Imitation"


# Pepper connection settings
PEPPER_PORT = 9559
# Set to "real" to connect to physical Pepper on LAN; "sim" for Choregraphe/local simulator.
PEPPER_MODE = "sim" 
PEPPER_IP_BY_MODE = {
    "real": "192.168.0.102",
    "sim": "127.0.0.1",
}

# Nullspace settings functions
def _get_env_float(name, default):
    value = os.environ.get(name, None)
    if value is None:
        return float(default)
    value = str(value).strip()
    if not value:
        return float(default)
    try:
        return float(value)
    except Exception:
        raise ValueError("Invalid float for %s: %r" % (name, value))
    

def _get_env_choice(name, default, choices):
    value = os.environ.get(name, None)
    if value is None:
        return str(default)
    value = str(value).strip().lower()
    if not value:
        return str(default)
    if value not in choices:
        raise ValueError(
            "Invalid value for %s: %r (expected one of %s)" % (
                name,
                value,
                ", ".join(sorted(choices)),
            )
        )
    return value
    
    
# Low-latency command shaping parameters.
POSE_SMOOTHING_DEADBAND_RAD = _get_env_float("POSE_SMOOTHING_DEADBAND_RAD", 0.01)
POSE_SMOOTHING_MAX_DELTA_RAD_PER_TICK = _get_env_float("POSE_SMOOTHING_MAX_DELTA_RAD_PER_TICK", 0.06)
POSE_SMOOTHING_ALPHA_SLOW = _get_env_float("POSE_SMOOTHING_ALPHA_SLOW", 0.40)
POSE_SMOOTHING_ALPHA_FAST = _get_env_float("POSE_SMOOTHING_ALPHA_FAST", 0.80)
POSE_SMOOTHING_FAST_DELTA_RAD = _get_env_float("POSE_SMOOTHING_FAST_DELTA_RAD", 0.06)

# Null-space guardrails for arm singular/near-singular regions, lock candidates are fixed in joint_constraints.py: (-pi/2, 0, +pi/2).
POSE_NULLSPACE_HYSTERESIS_RAD = _get_env_float("POSE_NULLSPACE_HYSTERESIS_RAD", 0.06)  # rad
POSE_NULLSPACE_R_SHOULDER_ROLL_TRIGGER_RAD = _get_env_float("POSE_NULLSPACE_R_SHOULDER_ROLL_TRIGGER_RAD", -1.4)  # rad that triggers the lock
POSE_NULLSPACE_L_SHOULDER_ROLL_TRIGGER_RAD = _get_env_float("POSE_NULLSPACE_L_SHOULDER_ROLL_TRIGGER_RAD", 1.4)   # rad
POSE_NULLSPACE_R_ELBOW_ROLL_TRIGGER_RAD    = _get_env_float("POSE_NULLSPACE_R_ELBOW_ROLL_TRIGGER_RAD", 0.1)
POSE_NULLSPACE_L_ELBOW_ROLL_TRIGGER_RAD    = _get_env_float("POSE_NULLSPACE_L_ELBOW_ROLL_TRIGGER_RAD", -0.1)

# Low-latency command shaping parameters.
POSE_SMOOTHING_DEADBAND_RAD = _get_env_float("POSE_SMOOTHING_DEADBAND_RAD", 0.01)
POSE_SMOOTHING_MAX_DELTA_RAD_PER_TICK = _get_env_float("POSE_SMOOTHING_MAX_DELTA_RAD_PER_TICK", 0.06)
POSE_SMOOTHING_ALPHA_SLOW = _get_env_float("POSE_SMOOTHING_ALPHA_SLOW", 0.40)
POSE_SMOOTHING_ALPHA_FAST = _get_env_float("POSE_SMOOTHING_ALPHA_FAST", 0.80)
POSE_SMOOTHING_FAST_DELTA_RAD = _get_env_float("POSE_SMOOTHING_FAST_DELTA_RAD", 0.06)

# Backend-dependent command-rate profile used by imitation runtime.
POSE_BACKEND_PROFILE = _get_env_choice("POSE_BACKEND_PROFILE", "metrabs", ("metrabs", "zed"))
POSE_COMMAND_RATE_HZ_BY_BACKEND = {
    "metrabs": _get_env_float("POSE_COMMAND_RATE_HZ_METRABS", 15.0),
    "zed": _get_env_float("POSE_COMMAND_RATE_HZ_ZED", 25.0),
}

# Feedcbak Server Settings
FEEDBACK_SERVER_URL = os.environ.get("FEEDBACK_SERVER_URL", "http://localhost:8091").strip()
FEEDBACK_SERVER_TIMEOUT_SEC = _get_env_float("FEEDBACK_SERVER_TIMEOUT_SEC", 20.0)

# Chain speed fractions used for command shaping and command pacing.
CHAIN_SPEED_FRACTIONS = {
    "torso": 0.25,
    "head": 0.25,
    "others": 0.25,
}
HEAD_CHAIN_SPEED_FRACTION = CHAIN_SPEED_FRACTIONS["head"]
TORSO_CHAIN_SPEED_FRACTION = CHAIN_SPEED_FRACTIONS["torso"]
OTHER_CHAIN_SPEED_FRACTION = CHAIN_SPEED_FRACTIONS["others"]
ARM_CHAIN_SPEED_FRACTION = OTHER_CHAIN_SPEED_FRACTION
HAND_CHAIN_SPEED_FRACTION = OTHER_CHAIN_SPEED_FRACTION

# Pepper mode & IP based on mode
if PEPPER_MODE not in PEPPER_IP_BY_MODE:
    raise ValueError("Unsupported PEPPER_MODE: %s" % str(PEPPER_MODE))

PEPPER_IP = PEPPER_IP_BY_MODE[PEPPER_MODE]


def get_pepper_endpoint():
    return PEPPER_IP, PEPPER_PORT

def set_pose_stream_runtime_dir():
    CLIENT_DIR = os.path.dirname(os.path.abspath(__file__))
    KINEMATICS_DIR = os.path.join(CLIENT_DIR, "kinematics")
    EXERCISES_DIR = os.path.join(CLIENT_DIR, "exercises")

    # Keep legacy flat imports working after moving modules into subfolders.
    for path in (KINEMATICS_DIR, EXERCISES_DIR):
        if path not in sys.path:
            sys.path.insert(0, path)

def set_client_dir():
    CLIENT_DIR = os.path.dirname(os.path.abspath(__file__))
    if CLIENT_DIR not in sys.path:
        sys.path.insert(0, CLIENT_DIR)


def set_demonstration_client_dir():
    CLIENT_DIR = os.path.dirname(os.path.abspath(__file__))
    KINEMATICS_DIR = os.path.join(CLIENT_DIR, "kinematics")
    EXERCISES_DIR = os.path.join(CLIENT_DIR, "exercises")

    # Keep flat imports consistent with the existing client scripts.
    for path in (CLIENT_DIR, KINEMATICS_DIR, EXERCISES_DIR):
        if path not in sys.path:
            sys.path.insert(0, path)

def get_command_rate_hz_for_backend(backend_profile):
    profile = str(backend_profile or "").strip().lower()
    if profile not in POSE_COMMAND_RATE_HZ_BY_BACKEND:
        profile = "metrabs"
    return float(POSE_COMMAND_RATE_HZ_BY_BACKEND[profile])
