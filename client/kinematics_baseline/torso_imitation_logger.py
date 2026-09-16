import csv
import os
import threading
import time
import math
from datetime import datetime
import numpy as np

import transformation_matrices as tfm
import inverse_kinematics_rangeupdate as ik


class TorsoImitationLogger(object):
    def __init__(self, interval_seconds=0.1):
        self.interval_seconds = interval_seconds
        self._lock = threading.Lock()
        self._enabled = False
        self._last_write_ts = 0.0
        self._spine_keypoint = None
        self._torso_keypoint = None
        self._csv_path = None
        self._csv_file = None
        self._csv_writer = None
        self._start_ts = 0.0
        self._plot_samples = []
        self._spine_x_m = None
        self._torso_x_m = None

    def start(self, csv_path=None):
        with self._lock:
            if self._enabled:
                return self._csv_path

            if csv_path is None:
                stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                csv_path = os.path.join(
                    os.path.dirname(__file__),
                    "logs",
                    "torso_imitation_log_%s.csv" % stamp
                )

            output_dir = os.path.dirname(csv_path)
            if output_dir and not os.path.isdir(output_dir):
                os.makedirs(output_dir)

            self._csv_file = open(csv_path, "wb")
            self._csv_writer = csv.writer(self._csv_file)
            self._csv_writer.writerow([
                "timestamp",
                "knee_pitch",
                "hip_pitch",
                "spine_keypoint_xyz",
                "torso_keypoint_xyz",
            ])
            self._csv_file.flush()

            self._csv_path = csv_path
            self._start_ts = time.time()
            self._plot_samples = []
            self._spine_x_m = None
            self._torso_x_m = None
            self._last_write_ts = 0.0
            self._enabled = True
            return self._csv_path

    def stop(self):
        with self._lock:
            csv_path = self._csv_path
            plot_samples = list(self._plot_samples)
            self._enabled = False
            self._spine_keypoint = None
            self._torso_keypoint = None
            self._spine_x_m = None
            self._torso_x_m = None
            self._plot_samples = []
            self._start_ts = 0.0
            self._last_write_ts = 0.0
            if self._csv_file is not None:
                self._csv_file.close()
            self._csv_file = None
            self._csv_writer = None

        self._plot_spine_torso_x(csv_path, plot_samples)

    def get_csv_path(self):
        with self._lock:
            return self._csv_path

    def update_keypoints(self, spine_keypoint, torso_keypoint):
        with self._lock:
            self._spine_keypoint = _format_xyz(spine_keypoint)
            self._torso_keypoint = _format_xyz(torso_keypoint)
            self._spine_x_m = _extract_x(spine_keypoint)
            self._torso_x_m = _extract_x(torso_keypoint)

    def maybe_log(self, knee_pitch, hip_pitch):
        now = time.time()
        with self._lock:
            if not self._enabled or self._csv_writer is None:
                return
            if self._spine_keypoint is None or self._torso_keypoint is None:
                return
            if (now - self._last_write_ts) < self.interval_seconds:
                return

            self._csv_writer.writerow([
                datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f"),
                _safe_float_str(knee_pitch),
                _safe_float_str(hip_pitch),
                self._spine_keypoint,
                self._torso_keypoint,
            ])
            self._csv_file.flush()
            if self._spine_x_m is not None and self._torso_x_m is not None:
                elapsed_s = now - self._start_ts
                self._plot_samples.append((elapsed_s, self._spine_x_m, self._torso_x_m))
            self._last_write_ts = now

    def _plot_spine_torso_x(self, csv_path, samples):
        if not csv_path or not samples:
            return
        try:
            import matplotlib.pyplot as plt
        except Exception as err:
            print("matplotlib unavailable, skipping torso plot: %s" % str(err))
            return

        try:
            times = [row[0] for row in samples]
            spine_x_vals = [row[1] for row in samples]
            torso_x_vals = [row[2] for row in samples]

            fig = plt.figure(figsize=(10, 5))
            ax = fig.add_subplot(111)
            ax.plot(times, spine_x_vals, label="Spine x", linewidth=2.0)
            ax.plot(times, torso_x_vals, label="Torso x", linewidth=2.0)
            x_ticks = _build_ticks(min(times), max(times), 1.0)
            y_ticks = _build_ticks(min(min(spine_x_vals), min(torso_x_vals)),
                                   max(max(spine_x_vals), max(torso_x_vals)),
                                   0.05)
            if x_ticks:
                ax.set_xticks(x_ticks)
            if y_ticks:
                ax.set_yticks(y_ticks)
            ax.set_xlabel("Time (s)")
            ax.set_ylabel("Meters")
            ax.set_title("Spine/Torso x-direction over time")
            ax.grid(True, alpha=0.3)
            ax.legend(loc="best")
            fig.tight_layout()

            plot_path = os.path.splitext(csv_path)[0] + "_x_plot.png"
            fig.savefig(plot_path, dpi=150)
            print("Torso imitation x plot saved: %s" % plot_path)
            plt.show()
        except Exception as err:
            print("Failed to generate torso x plot: %s" % str(err))
        finally:
            try:
                plt.close("all")
            except Exception:
                pass



def _safe_float_str(value):
    try:
        return "%.6f" % float(value)
    except Exception:
        return ""


def _format_xyz(value):
    if value is None:
        return ""
    try:
        x = float(value[0])
        y = float(value[1])
        z = float(value[2])
        return "%.6f %.6f %.6f" % (x, y, z)
    except Exception:
        return ""


def _extract_x(value):
    try:
        return float(value[0])
    except Exception:
        return None


def _build_ticks(min_val, max_val, step):
    if step <= 0.0:
        return []
    if max_val < min_val:
        min_val, max_val = max_val, min_val

    start = math.floor(min_val / step) * step
    end = math.ceil(max_val / step) * step
    ticks = []
    current = start
    # Add an epsilon so floating point rounding does not drop the end tick.
    while current <= end + (step * 0.001):
        ticks.append(round(current, 6))
        current += step
    return ticks


_LOGGER = TorsoImitationLogger(interval_seconds=0.1)


def start_logging(csv_path=None):
    return _LOGGER.start(csv_path)


def stop_logging():
    _LOGGER.stop()


def get_logging_path():
    return _LOGGER.get_csv_path()


def update_torso_keypoints(spine_keypoint, torso_keypoint):
    _LOGGER.update_keypoints(spine_keypoint, torso_keypoint)


def log_torso_command(knee_pitch, hip_pitch):
    _LOGGER.maybe_log(knee_pitch, hip_pitch)


class ArmImitationLogger(object):
    def __init__(self, interval_seconds=0.1, default_show_plots=True):
        self.interval_seconds = interval_seconds
        self.default_show_plots = bool(default_show_plots)
        self._lock = threading.Lock()
        self._enabled = False
        self._last_write_ts = 0.0
        self._csv_path = None
        self._angles_csv_path = None
        self._csv_file = None
        self._angles_csv_file = None
        self._csv_writer = None
        self._angles_writer = None
        self._start_ts = 0.0
        self._keypoint_samples = []
        self._angle_samples = []
        self._torso = None
        self._neck = None
        self._l_shoulder = None
        self._l_elbow = None
        self._l_wrist = None
        self._r_shoulder = None
        self._r_elbow = None
        self._r_wrist = None
        self._show_plots = bool(default_show_plots)

    def start(self, csv_path=None, angles_csv_path=None, show_plots=None):
        with self._lock:
            if self._enabled:
                return self._csv_path, self._angles_csv_path

            stamp = None
            if csv_path is None or angles_csv_path is None:
                stamp = datetime.now().strftime("%Y%m%d_%H%M%S")

            if csv_path is None:
                csv_path = os.path.join(
                    os.path.dirname(__file__),
                    "logs",
                    "arm_keypoints_log_%s.csv" % stamp
                )

            if angles_csv_path is None:
                angles_csv_path = os.path.join(
                    os.path.dirname(__file__),
                    "logs",
                    "arm_angles_log_%s.csv" % stamp
                )

            output_dir = os.path.dirname(csv_path)
            if output_dir and not os.path.isdir(output_dir):
                os.makedirs(output_dir)

            output_dir = os.path.dirname(angles_csv_path)
            if output_dir and not os.path.isdir(output_dir):
                os.makedirs(output_dir)

            self._csv_file = open(csv_path, "wb")
            self._csv_writer = csv.writer(self._csv_file)
            self._csv_writer.writerow([
                "timestamp",
                "elapsed_s",
                "LShoulder_x", "LShoulder_y", "LShoulder_z",
                "LElbow_x", "LElbow_y", "LElbow_z",
                "LWrist_x", "LWrist_y", "LWrist_z",
                "RShoulder_x", "RShoulder_y", "RShoulder_z",
                "RElbow_x", "RElbow_y", "RElbow_z",
                "RWrist_x", "RWrist_y", "RWrist_z",
            ])
            self._csv_file.flush()

            self._angles_csv_file = open(angles_csv_path, "wb")
            self._angles_writer = csv.writer(self._angles_csv_file)
            self._angles_writer.writerow([
                "timestamp",
                "elapsed_s",
                "LShoulderPitch_rad",
                "LShoulderRoll_rad",
                "LElbowFlex_rad",
                "RShoulderPitch_rad",
                "RShoulderRoll_rad",
                "RElbowFlex_rad",
            ])
            self._angles_csv_file.flush()

            self._csv_path = csv_path
            self._angles_csv_path = angles_csv_path
            self._start_ts = time.time()
            self._keypoint_samples = []
            self._angle_samples = []
            if show_plots is None:
                self._show_plots = self.default_show_plots
            else:
                self._show_plots = bool(show_plots)
            self._last_write_ts = 0.0
            self._enabled = True
            return self._csv_path, self._angles_csv_path

    def stop(self):
        with self._lock:
            csv_path = self._csv_path
            angles_csv_path = self._angles_csv_path
            keypoint_samples = list(self._keypoint_samples)
            angle_samples = list(self._angle_samples)
            show_plots = bool(self._show_plots)
            self._enabled = False
            self._last_write_ts = 0.0
            self._start_ts = 0.0
            self._keypoint_samples = []
            self._angle_samples = []
            self._torso = None
            self._neck = None
            self._l_shoulder = None
            self._l_elbow = None
            self._l_wrist = None
            self._r_shoulder = None
            self._r_elbow = None
            self._r_wrist = None
            if self._csv_file is not None:
                self._csv_file.close()
            if self._angles_csv_file is not None:
                self._angles_csv_file.close()
            self._csv_file = None
            self._angles_csv_file = None
            self._csv_writer = None
            self._angles_writer = None
            self._show_plots = self.default_show_plots

        self._plot_keypoints(csv_path, keypoint_samples, show_plot=show_plots)
        self._plot_angles(angles_csv_path, angle_samples, show_plot=show_plots)

    def get_csv_paths(self):
        with self._lock:
            return self._csv_path, self._angles_csv_path

    def update_keypoints(self, torso, neck, l_shoulder, l_elbow, l_wrist,
                         r_shoulder, r_elbow, r_wrist):
        with self._lock:
            self._torso = _format_vec(torso)
            self._neck = _format_vec(neck)
            self._l_shoulder = _format_vec(l_shoulder)
            self._l_elbow = _format_vec(l_elbow)
            self._l_wrist = _format_vec(l_wrist)
            self._r_shoulder = _format_vec(r_shoulder)
            self._r_elbow = _format_vec(r_elbow)
            self._r_wrist = _format_vec(r_wrist)

        self._maybe_log()

    def _maybe_log(self):
        now = time.time()
        with self._lock:
            if not self._enabled or self._csv_writer is None or self._angles_writer is None:
                return
            if (now - self._last_write_ts) < self.interval_seconds:
                return

            if not self._has_all_keypoints():
                return

            elapsed_s = now - self._start_ts
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")

            self._csv_writer.writerow([
                timestamp,
                "%.6f" % elapsed_s,
                _safe_float_str(self._l_shoulder[0]), _safe_float_str(self._l_shoulder[1]), _safe_float_str(self._l_shoulder[2]),
                _safe_float_str(self._l_elbow[0]), _safe_float_str(self._l_elbow[1]), _safe_float_str(self._l_elbow[2]),
                _safe_float_str(self._l_wrist[0]), _safe_float_str(self._l_wrist[1]), _safe_float_str(self._l_wrist[2]),
                _safe_float_str(self._r_shoulder[0]), _safe_float_str(self._r_shoulder[1]), _safe_float_str(self._r_shoulder[2]),
                _safe_float_str(self._r_elbow[0]), _safe_float_str(self._r_elbow[1]), _safe_float_str(self._r_elbow[2]),
                _safe_float_str(self._r_wrist[0]), _safe_float_str(self._r_wrist[1]), _safe_float_str(self._r_wrist[2]),
            ])
            self._csv_file.flush()

            angles = self._compute_angles()
            if angles is None:
                angle_row = ["", "", "", "", "", "", ""]
            else:
                angle_row = [
                    _safe_float_str(angles["LShoulderPitch"]),
                    _safe_float_str(angles["LShoulderRoll"]),
                    _safe_float_str(angles["LElbowFlex"]),
                    _safe_float_str(angles["RShoulderPitch"]),
                    _safe_float_str(angles["RShoulderRoll"]),
                    _safe_float_str(angles["RElbowFlex"]),
                ]

            self._angles_writer.writerow([timestamp, "%.6f" % elapsed_s] + angle_row)
            self._angles_csv_file.flush()

            self._keypoint_samples.append((
                elapsed_s,
                self._l_shoulder, self._l_elbow, self._l_wrist,
                self._r_shoulder, self._r_elbow, self._r_wrist
            ))
            if angles is not None:
                self._angle_samples.append((
                    elapsed_s,
                    angles["LShoulderPitch"],
                    angles["LShoulderRoll"],
                    angles["LElbowFlex"],
                    angles["RShoulderPitch"],
                    angles["RShoulderRoll"],
                    angles["RElbowFlex"],
                ))

            self._last_write_ts = now

    def _has_all_keypoints(self):
        return (
            self._torso is not None and
            self._neck is not None and
            self._l_shoulder is not None and
            self._l_elbow is not None and
            self._l_wrist is not None and
            self._r_shoulder is not None and
            self._r_elbow is not None and
            self._r_wrist is not None
        )

    def _compute_angles(self):
        try:
            torso = tfm.set_joint_in_peppers_coordinates(np.array(self._torso, dtype=np.float32))
            l_sh = tfm.set_joint_in_peppers_coordinates(np.array(self._l_shoulder, dtype=np.float32)) - torso
            l_el = tfm.set_joint_in_peppers_coordinates(np.array(self._l_elbow, dtype=np.float32)) - torso
            l_wr = tfm.set_joint_in_peppers_coordinates(np.array(self._l_wrist, dtype=np.float32)) - torso
            r_sh = tfm.set_joint_in_peppers_coordinates(np.array(self._r_shoulder, dtype=np.float32)) - torso
            r_el = tfm.set_joint_in_peppers_coordinates(np.array(self._r_elbow, dtype=np.float32)) - torso
            r_wr = tfm.set_joint_in_peppers_coordinates(np.array(self._r_wrist, dtype=np.float32)) - torso
        except Exception:
            return None

        rot_mat = tfm.rotation_mat_arms(r_sh, l_sh)

        l_sh_r = rot_mat.dot(l_sh)
        r_sh_r = rot_mat.dot(r_sh)
        l_el_r = rot_mat.dot(l_el) - l_sh_r
        r_el_r = rot_mat.dot(r_el) - r_sh_r
        l_wr_r = rot_mat.dot(l_wr) - l_sh_r - l_el_r
        r_wr_r = rot_mat.dot(r_wr) - r_sh_r - r_el_r

        left = self._compute_side_angles_pepper(l_el_r, l_wr_r, "left")
        right = self._compute_side_angles_pepper(r_el_r, r_wr_r, "right")
        if left is None or right is None:
            return None

        return {
            "LShoulderPitch": left[0],
            "LShoulderRoll": left[1],
            "LElbowFlex": left[2],
            "RShoulderPitch": right[0],
            "RShoulderRoll": right[1],
            "RElbowFlex": right[2],
        }

    def _compute_side_angles_pepper(self, elbow_vec, wrist_vec, arm):
        elbow_norm = np.linalg.norm(elbow_vec)
        wrist_norm = np.linalg.norm(wrist_vec)
        if elbow_norm < 1e-6 or wrist_norm < 1e-6:
            return None

        elbow_flex = _angle_between(elbow_vec.tolist(), wrist_vec.tolist())
        if elbow_flex is None:
            return None

        elbow_scaled = 181.2 * elbow_vec / elbow_norm
        if arm == "right":
            pepper_shoulder = np.array([-57.0, -149.74, 86.82], dtype=np.float32)
        else:
            pepper_shoulder = np.array([-57.0, 149.74, 86.82], dtype=np.float32)
        elbow_pos = elbow_scaled + pepper_shoulder

        shoulder_pitch, shoulder_roll = ik.get_arm_partial_angles(
            elbow_pos[0], elbow_pos[1], elbow_pos[2], arm
        )
        try:
            if math.isnan(float(shoulder_pitch)) or math.isnan(float(shoulder_roll)):
                return None
        except Exception:
            return None

        return shoulder_pitch, shoulder_roll, elbow_flex

    def _plot_keypoints(self, csv_path, samples, show_plot=True):
        if not csv_path or not samples:
            return
        try:
            import matplotlib.pyplot as plt
        except Exception as err:
            print("matplotlib unavailable, skipping keypoint plot: %s" % str(err))
            return

        try:
            times = [row[0] for row in samples]
            labels = [
                "LShoulder", "LElbow", "LWrist",
                "RShoulder", "RElbow", "RWrist",
            ]
            series = {label: [] for label in labels}
            for row in samples:
                _, l_sh, l_el, l_wr, r_sh, r_el, r_wr = row
                series["LShoulder"].append(l_sh)
                series["LElbow"].append(l_el)
                series["LWrist"].append(l_wr)
                series["RShoulder"].append(r_sh)
                series["RElbow"].append(r_el)
                series["RWrist"].append(r_wr)

            fig, axes = plt.subplots(3, 1, figsize=(11, 8), sharex=True)
            coord_labels = ["X (m)", "Y (m)", "Z (m)"]
            for axis_idx in range(3):
                ax = axes[axis_idx]
                for label in labels:
                    ax.plot(times, [pt[axis_idx] for pt in series[label]], label=label, linewidth=1.5)
                ax.set_ylabel(coord_labels[axis_idx])
                ax.grid(True, alpha=0.3)

            axes[0].set_title("Arm keypoints over time")
            axes[-1].set_xlabel("Time (s)")
            axes[0].legend(loc="best", ncol=3, fontsize=8)
            fig.tight_layout()

            plot_path = os.path.splitext(csv_path)[0] + "_plot.png"
            fig.savefig(plot_path, dpi=150)
            print("Arm keypoint plot saved: %s" % plot_path)
            if show_plot:
                plt.show()
        except Exception as err:
            print("Failed to generate arm keypoint plot: %s" % str(err))
        finally:
            try:
                plt.close("all")
            except Exception:
                pass

    def _plot_angles(self, csv_path, samples, show_plot=True):
        if not csv_path or not samples:
            return
        try:
            import matplotlib.pyplot as plt
        except Exception as err:
            print("matplotlib unavailable, skipping angle plot: %s" % str(err))
            return

        try:
            times = [row[0] for row in samples]
            l_pitch = [row[1] for row in samples]
            l_roll = [row[2] for row in samples]
            l_elbow = [row[3] for row in samples]
            r_pitch = [row[4] for row in samples]
            r_roll = [row[5] for row in samples]
            r_elbow = [row[6] for row in samples]

            fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
            axes[0].plot(times, l_pitch, label="LShoulderPitch", linewidth=1.6)
            axes[0].plot(times, l_roll, label="LShoulderRoll", linewidth=1.6)
            axes[0].plot(times, l_elbow, label="LElbowFlex", linewidth=1.6)
            axes[0].set_ylabel("Radians")
            axes[0].set_title("Left arm angles")
            axes[0].grid(True, alpha=0.3)
            axes[0].legend(loc="best")

            axes[1].plot(times, r_pitch, label="RShoulderPitch", linewidth=1.6)
            axes[1].plot(times, r_roll, label="RShoulderRoll", linewidth=1.6)
            axes[1].plot(times, r_elbow, label="RElbowFlex", linewidth=1.6)
            axes[1].set_ylabel("Radians")
            axes[1].set_title("Right arm angles")
            axes[1].grid(True, alpha=0.3)
            axes[1].legend(loc="best")
            axes[1].set_xlabel("Time (s)")

            fig.tight_layout()
            plot_path = os.path.splitext(csv_path)[0] + "_plot.png"
            fig.savefig(plot_path, dpi=150)
            print("Arm angle plot saved: %s" % plot_path)
            if show_plot:
                plt.show()
        except Exception as err:
            print("Failed to generate arm angle plot: %s" % str(err))
        finally:
            try:
                plt.close("all")
            except Exception:
                pass


def _format_vec(value):
    if value is None:
        return None
    try:
        return [float(value[0]), float(value[1]), float(value[2])]
    except Exception:
        return None


def _vec_sub(a, b):
    return [a[0] - b[0], a[1] - b[1], a[2] - b[2]]


def _vec_scale(a, s):
    return [a[0] * s, a[1] * s, a[2] * s]


def _vec_dot(a, b):
    return (a[0] * b[0]) + (a[1] * b[1]) + (a[2] * b[2])


def _vec_cross(a, b):
    return [
        (a[1] * b[2]) - (a[2] * b[1]),
        (a[2] * b[0]) - (a[0] * b[2]),
        (a[0] * b[1]) - (a[1] * b[0]),
    ]


def _vec_norm(a):
    return math.sqrt(_vec_dot(a, a))


def _vec_normalize(a):
    if a is None:
        return None
    norm = _vec_norm(a)
    if norm < 1e-6:
        return None
    return [a[0] / norm, a[1] / norm, a[2] / norm]


def _clamp(value, min_val, max_val):
    if value < min_val:
        return min_val
    if value > max_val:
        return max_val
    return value


def _angle_between(a, b):
    dot = _vec_dot(a, b)
    dot = _clamp(dot, -1.0, 1.0)
    return math.acos(dot)


def _sign_with_zero(value, eps=1e-6):
    if value > eps:
        return 1.0
    if value < -eps:
        return -1.0
    return 0.0


_ARM_LOGGER = ArmImitationLogger(interval_seconds=0.1)


def start_arm_logging(csv_path=None, angles_csv_path=None, show_plots=None):
    return _ARM_LOGGER.start(csv_path, angles_csv_path, show_plots=show_plots)


def stop_arm_logging():
    _ARM_LOGGER.stop()


def get_arm_logging_paths():
    return _ARM_LOGGER.get_csv_paths()


def update_arm_keypoints(torso, neck, l_shoulder, l_elbow, l_wrist,
                         r_shoulder, r_elbow, r_wrist):
    _ARM_LOGGER.update_keypoints(torso, neck, l_shoulder, l_elbow, l_wrist,
                                 r_shoulder, r_elbow, r_wrist)
