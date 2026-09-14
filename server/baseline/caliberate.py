import cv2
import numpy as np
import math
from pathlib import Path
import settings


# ==========================
# INITIALIZE
# ==========================
criteria = (
    cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
    30,
    1e-6
)

objp = np.zeros((settings.CHECKERBOARD[0] * settings.CHECKERBOARD[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:settings.CHECKERBOARD[0], 0:settings.CHECKERBOARD[1]].T.reshape(-1, 2)
objp *= settings.SQUARE_SIZE

objpoints = []
imgpoints = []

cap = cv2.VideoCapture(settings.CAMERA_INDEX, cv2.CAP_DSHOW)
cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)
assert cap.isOpened(), "Could not open webcam"

ret, frame = cap.read()
h, w = frame.shape[:2]

# ==========================
# INITIAL INTRINSIC GUESS
# ==========================
diag_px = math.sqrt(w**2 + h**2)
f_init = diag_px / (2 * math.tan(math.radians(settings.DIAGONAL_FOV_DEG) / 2))

K_init = np.array([
    [f_init, 0, w / 2],
    [0, f_init, h / 2],
    [0, 0, 1]
], dtype=np.float64)

dist_init = np.zeros((5, 1))

print("Press SPACE to capture a view")
print("Press ENTER to calibrate")
print("Press ESC to quit")

# ==========================
# CAPTURE LOOP
# ==========================
while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    found, corners = cv2.findChessboardCorners(gray, settings.CHECKERBOARD)

    vis = frame.copy()
    if found:
        corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        cv2.drawChessboardCorners(vis, settings.CHECKERBOARD, corners, found)

    cv2.putText(
        vis,
        f"Captured views: {len(objpoints)}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2
    )

    cv2.imshow("Calibration", vis)
    key = cv2.waitKey(1)

    if key == 27 or len(objpoints) >= 50:  # ESC
        break

    if key == 32 and found:  # SPACE
        objpoints.append(objp.copy())
        imgpoints.append(corners.copy())
        print(f"Captured view {len(objpoints)}")

    if key == 13 and len(objpoints) >= 10:  # ENTER
        break

cap.release()
cv2.destroyAllWindows()

assert len(objpoints) >= 10, "Not enough calibration views"

# ==========================
# CALIBRATION
# ==========================
flags = (
    cv2.CALIB_USE_INTRINSIC_GUESS |
    cv2.CALIB_RATIONAL_MODEL
)

ret, K, dist, rvecs, tvecs = cv2.calibrateCamera(
    objpoints,
    imgpoints,
    (w, h),
    K_init,
    dist_init,
    flags=flags,
    criteria=criteria
)

# ==========================
# RESULTS
# ==========================
print("\n===== CAMERA INTRINSICS =====")
print("Intrinsic matrix K:")
print(K)

print("\nDistortion coefficients:")
print(dist.ravel())

fx, fy = K[0, 0], K[1, 1]
cx, cy = K[0, 2], K[1, 2]

fov_x = 2 * math.degrees(math.atan(w / (2 * fx)))
fov_y = 2 * math.degrees(math.atan(h / (2 * fy)))
fov_diag = 2 * math.degrees(math.atan(
    math.sqrt(w**2 + h**2) / (2 * fx)
))

print("\n===== DERIVED FOVs =====")
print(f"Horizontal FoV: {fov_x:.2f}°")
print(f"Vertical FoV:   {fov_y:.2f}°")
print(f"Diagonal FoV:   {fov_diag:.2f}°")

print("\n===== READY FOR METRABS =====")
print("intrinsic_matrix = K")
print("distortion_coeffs = dist")

# ==========================
# SAVE RESULTS
# ==========================
settings.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

intrinsic_matrix = K.astype(np.float32)
distortion_coeffs_opencv = dist.ravel().astype(np.float32)
# Metrabs accepts OpenCV-style coefficients in this order:
# (k1, k2, p1, p2, k3, k4, k5, k6, s1, s2, s3, s4)
distortion_coeffs_metrabs = distortion_coeffs_opencv[:12]

camera_matrix_path = settings.OUTPUT_DIR / "camera_matrix.npy"
dist_coeffs_path = settings.OUTPUT_DIR / "dist_coeffs.npy"
metrabs_params_path = settings.OUTPUT_DIR / "metrabs_camera_params.npz"

np.save(camera_matrix_path, intrinsic_matrix)
np.save(dist_coeffs_path, distortion_coeffs_metrabs)
np.savez(
    metrabs_params_path,
    intrinsic_matrix=intrinsic_matrix,
    distortion_coeffs=distortion_coeffs_metrabs,
    distortion_coeffs_opencv=distortion_coeffs_opencv,
    image_width=np.int32(w),
    image_height=np.int32(h),
    rms_reprojection_error=np.float32(ret),
)

print("\n===== SAVED FILES =====")
print(f"camera_matrix: {camera_matrix_path}")
print(f"dist_coeffs:   {dist_coeffs_path}")
print(f"metrabs npz:   {metrabs_params_path}")
