import numpy as np
from server.common import settings


CALIBRATION_DIR = settings.CALIBRATION_DIR
PARAMS_NPZ = CALIBRATION_DIR / "metrabs_camera_params.npz"
CAMERA_MATRIX_NPY = CALIBRATION_DIR / "camera_matrix.npy"
DIST_COEFFS_NPY = CALIBRATION_DIR / "dist_coeffs.npy"


def load_metrabs_calibration():
    if PARAMS_NPZ.exists():
        params = np.load(PARAMS_NPZ)
        intrinsic_matrix = params["intrinsic_matrix"]
        distortion_coeffs = params["distortion_coeffs"]
    elif CAMERA_MATRIX_NPY.exists() and DIST_COEFFS_NPY.exists():
        intrinsic_matrix = np.load(CAMERA_MATRIX_NPY)
        distortion_coeffs = np.load(DIST_COEFFS_NPY)
    else:
        raise FileNotFoundError(
            "Calibration files not found. Expected one of:\n"
            f" - {PARAMS_NPZ}\n"
            f" - {CAMERA_MATRIX_NPY} and {DIST_COEFFS_NPY}\n"
            "Run `python caliberate.py` in server/dl-pose to generate them."
        )

    intrinsic_matrix = np.asarray(intrinsic_matrix, dtype=np.float32)
    distortion_coeffs = np.asarray(distortion_coeffs, dtype=np.float32).reshape(-1)
    return intrinsic_matrix, distortion_coeffs


