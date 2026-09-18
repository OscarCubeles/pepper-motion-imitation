from pathlib import Path
import numpy as np


COMMON_DIR = Path(__file__).resolve().parent
SERVER_DIR = COMMON_DIR.parent
REPO_ROOT = SERVER_DIR.parent
CALIBRATION_DIR = COMMON_DIR / "calibration_images"
MODELS_DIR = COMMON_DIR / "models"
METRABS_MODEL_DIR = MODELS_DIR / "metrabs_eff2l_384px_800k_28ds_pytorch"
YOLO_MODEL_PATH = MODELS_DIR / "yolov8m.pt"
HAND_LANDMARKER_PATH = COMMON_DIR / "hand_detection" / "hand_landmarker.task"
POSE_LANDMARKER_PATH = COMMON_DIR / "hand_detection" / "pose_landmarker_full.task"
PEPPER_RESOURCES_DIR = SERVER_DIR / "pepper_ik_resources"
IKPY_WORKSPACE_PATH = SERVER_DIR / "ikpy" / "metrabs_workspace.json"

# ================================================
# ----------Calibrations----------
# ================================================
CHECKERBOARD = (9, 6)        # inner corners
SQUARE_SIZE = 0.025          # meters (25 mm)
DIAGONAL_FOV_DEG = 78.0      # your webcam spec
CAMERA_INDEX = 0             # usually 0
OUTPUT_DIR = CALIBRATION_DIR


# ================================================
# ----------Metrabs----------
# ================================================
CAMERA_INDEX = 0
SKELETON = "smpl+head_30"  # "smpl+head_30" Make sure to change the mapping if the skeleton is changed, otherwise the joints will be mixed up.
SKELETON_SMPL_HEAD_30 = "smpl+head_30"
SKELETON_KINECT = "kinectv2_25"

ENABLE_VISUALIZATION = True
ENABLE_CAMERA_TEXT_OVERLAYS = False  # Keep the 2D camera panel free of text labels.
ENABLE_HAND_ORIENTATION_VISUALIZATION = False
ENABLE_BOUNDING_BOX = False
ENABLE_ANGLES_VISUALIZATION = True
SHOW_ANGLE_VALUES = True  # If True: show angles + feasibility + singularity. If False: show only feasibility + singularity
VISUALIZER_MAX_FPS = 15.0
PERF_LOG_INTERVAL_SEC = 2.0

KINECTV2_25_JOINTS = {
    "Pelvis": 0,
    "Spine": 1,
    "SpineChest": 2,
    "Neck": 3,
    "Head": 4,
    "ShoulderLeft": 5,
    "ElbowLeft": 6,
    "WristLeft": 7,
    "HandLeft": 8,
    "ShoulderRight": 9,
    "ElbowRight": 10,
    "WristRight": 11,
    "HandRight": 12,
    "HipLeft": 13,
    "KneeLeft": 14,
    "AnkleLeft": 15,
    "FootLeft": 16,
    "HipRight": 17,
    "KneeRight": 18,
    "AnkleRight": 19,
    "FootRight": 20,
}


SMPL_HEAD_30_JOINTS = {
    "pelv_smpl": 0,
    "lhip_smpl": 1,
    "rhip_smpl": 2,
    "bell_smpl": 3,
    "lkne_smpl": 4,
    "rkne_smpl": 5,
    "spin_smpl": 6,
    "lank_smpl": 7,
    "rank_smpl": 8,
    "thor_smpl": 9,
    "ltoe_smpl": 10,
    "rtoe_smpl": 11,
    "neck_smpl": 12,
    "lcla_smpl": 13,
    "rcla_smpl": 14,
    "head_smpl": 15,
    "lsho_smpl": 16,
    "rsho_smpl": 17,
    "lelb_smpl": 18,
    "relb_smpl": 19,
    "lwri_smpl": 20,
    "rwri_smpl": 21,
    "lhan_smpl": 22,
    "rhan_smpl": 23,
    "nose_coco": 24,
    "leye_coco": 25,
    "lear_coco": 26,
    "reye_coco": 27,
    "rear_coco": 28,
    "htop_mpi_inf_3dhp": 29,
}

# ================================================
# ----------Metrabs server----------
# ================================================
REQUIRE_CLIENT_MESSAGE_TO_START = True
START_MESSAGE = "keypoints"
SHUTDOWN_ON_CLIENT_DISCONNECT = True

DETECTOR_THRESHOLD = 0.5
DETECTOR_NMS_IOU = 0.7
MAX_DETECTIONS = 1
NUM_AUG = 1
ANTIALIAS_FACTOR = 1
INTERNAL_BATCH_SIZE = 128
AVERAGE_AUG = True
SUPPRESS_IMPLAUSIBLE_POSES = True
DETECTOR_FLIP_AUG = False

JOINT_COUNT = 10
PARAMS_PER_JOINT = 3


# ================================================
# ----------Metrabs server mediapipe----------
# ================================================
JOINT_COUNT = 12
PARAMS_PER_JOINT = 3


# ================================================
# ----------Metrabs mediapipe----------
# ================================================
#SKELETON = "smpl+head_30"  # "kinectv2_25"

# Mapping from Metrabs kinectv2_25 joints to outgoing 20x3 pose:
# 0:Nose, 1:Neck, 2:RShoulder, 3:RElbow, 4:RWrist, 5:LShoulder, 6:LElbow, 7:LWrist, 8:Torso, 9:SpineBase
# 10:RTip, 11:LTip
# 12:RWrist(mp), 13:RThumbCMC, 14:RIndexMCP, 15:RPinkyMCP
# 16:LWrist(mp), 17:LThumbCMC, 18:LIndexMCP, 19:LPinkyMCP
TARGET_BODY_IDXS = np.arange(10, dtype=np.int64)
SOURCE__KINECT_BODY_IDXS = np.array([3, 2, 8, 9, 10, 4, 5, 6, 1, 0], dtype=np.int64)
SOURCE_BODY_IDXS_SMPL_HEAD_30 = np.array([15, 12, 17, 19, 21, 16, 18, 20, 6, 0], dtype=np.int64)

if SKELETON == "kinectv2_25":
    SOURCE_BODY_IDXS = np.array([3, 2, 8, 9, 10, 4, 5, 6, 1, 0], dtype=np.int64)
if SKELETON == "smpl+head_30":
    SOURCE_BODY_IDXS = np.array([15, 12, 17, 19, 21, 16, 18, 20, 6, 0], dtype=np.int64)


RIGHT_TIP_OUT_IDX = 10
LEFT_TIP_OUT_IDX = 11
RIGHT_ORIENTATION_OUT_IDXS = (12, 13, 14, 15)
LEFT_ORIENTATION_OUT_IDXS = (16, 17, 18, 19)

HAND_TIP_ID = 12
HAND_ORIENTATION_IDS = (0, 1, 5, 17)
HAND_LANDMARK_IDS = HAND_ORIENTATION_IDS + (HAND_TIP_ID,)
HAND_JOINT_NAME_TOKENS = ("hand", "thumb", "index", "middle", "ring", "pinky", "finger", "tip")
HAND_JOINT_NAME_PREFIXES = ("lhan", "rhan", "lhnd", "rhnd", "lthu", "rthu", "lfin", "rfin", "lhtip", "rhtip")
METRABS_UNITS_TO_METERS = 0.001
OUT_JOINT_COUNT = 26




# ================================================
# ----------Wrist Orientation ----------
# ================================================
opposite_wrist = {
    "up": "down",
    "down": "up",
    "left": "right",
    "right": "left",
    "forward": "backward",
    "backward": "forward"
}

FRONT_LABEL = "forward"
BACK_LABEL = "backward"


cross_wrist = {
    ("forward","up"): "left",
    ("forward","down"): "right",
    ("forward","left"): "down",
    ("forward","right"): "up",

    ("backward","up"): "right",
    ("backward","down"): "left",
    ("backward","left"): "up",
    ("backward","right"): "down",

    ("up","forward"): "right",
    ("up","backward"): "left",
    ("up","left"): "forward",
    ("up","right"): "backward",

    ("down","forward"): "left",
    ("down","backward"): "right",
    ("down","left"): "backward",
    ("down","right"): "forward",

    ("left","up"): "forward",
    ("left","down"): "backward",
    ("left","forward"): "down",
    ("left","backward"): "up",

    ("right","up"): "backward",
    ("right","down"): "forward",
    ("right","forward"): "up",
    ("right","backward"): "down",
}

WRIST_AXIS_MAP = {
    "left":  ("x", -1),
    "right": ("x",  1),
    "up":    ("y",  1),
    "down":  ("y", -1),
    "forward":  ("z",  1),
    "backward": ("z", -1),
}


# ================================================
# ----------Hand Orientation Constraints ----------
# ================================================
# Exact orientation values from hand_orientation.py classification
HAND_ORIENTATION_FRONT = "FRONT"      # Palm facing toward camera
HAND_ORIENTATION_BACK = "BACK"        # Palm facing away from camera
HAND_ORIENTATION_UP = "UP"            # Palm facing upward
HAND_ORIENTATION_DOWN = "DOWN"        # Palm facing downward
HAND_ORIENTATION_LEFT = "LEFT"        # Palm facing left
HAND_ORIENTATION_RIGHT = "RIGHT"      # Palm facing right

# Mapping from hand orientation to ShoulderPitch angle (in radians)
# Used for singularity constraint application
HAND_ORIENTATION_TO_SHOULDER_PITCH = {
    HAND_ORIENTATION_DOWN: 0.0,           # 0 degrees - arm down
    HAND_ORIENTATION_FRONT: -1.3090,      # -75 degrees - arm forward
    HAND_ORIENTATION_BACK: 1.3090,        # 75 degrees - arm backward
}

# Safe angle for ElbowRoll to escape singularity (in radians)
ELBOW_ROLL_SAFE_ANGLE = 0.0785  # 4.5 degrees

# ShoulderRoll angles by arm during singularity constraint (in radians)
SHOULDER_ROLL_BY_ARM = {
    'right': -1.5621,  # -89.5 degrees
    'left': 1.5621,    # 89.5 degrees
}
