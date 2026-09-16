import copy
import json
import math
import sys
from pathlib import Path

import cv2
import numpy as np
import streamlit as st


CURRENT_FILE = Path(__file__).resolve()

DL_POSE_DIR = CURRENT_FILE.parents[1]
REPO_ROOT = CURRENT_FILE.parents[2]

if str(DL_POSE_DIR) not in sys.path:
    sys.path.insert(0, str(DL_POSE_DIR))

DEFAULT_DATASET_ROOT = REPO_ROOT / "dataset"


ORIENTATION_LABELS = ["FRONT", "BACK", "UP", "DOWN", "LEFT", "RIGHT"]
ARM_SIDES = ["left", "right"]
JOINT_ANGLE_KEYS = ["shoulder_pitch", "shoulder_roll", "elbow_yaw", "elbow_roll"]
JOINT_ANGLE_DISPLAY_ORDER = ["elbow_roll", "elbow_yaw", "shoulder_pitch", "shoulder_roll"]
ANGLE_UNITS = ["Radians", "Degrees"]
POINT_KEYS = ["shoulder", "elbow", "wrist"]
POINT_COLORS = {
    "torso": (255, 255, 255),
    "left_shoulder": (0, 255, 255),
    "left_elbow": (0, 165, 255),
    "left_wrist": (0, 0, 255),
    "right_shoulder": (255, 255, 0),
    "right_elbow": (255, 128, 0),
    "right_wrist": (255, 0, 255),
}


def _load_json(path):
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def _save_json(path, data):
    with path.open("w", encoding="utf-8") as file:
        json.dump(data, file, indent=4)


def _video_folders(category_path):
    return sorted(
        path for path in category_path.iterdir()
        if path.is_dir() and path.name.startswith("video_")
    )


def _annotation_paths(video_folder):
    return {
        "template": video_folder / "annotations.json",
        "filled": video_folder / "annotations_filled.json",
    }


def _load_annotations(video_folder):
    paths = _annotation_paths(video_folder)
    if paths["filled"].exists():
        return _load_json(paths["filled"]), paths["filled"], "annotations_filled.json"
    if paths["template"].exists():
        return _load_json(paths["template"]), paths["filled"], "annotations.json"
    return None, paths["filled"], None


def _state_key(video_folder):
    return str(video_folder.resolve())


def _ensure_loaded(video_folder):
    key = _state_key(video_folder)
    if st.session_state.get("loaded_video") == key:
        return

    annotations, save_path, source_name = _load_annotations(video_folder)
    st.session_state.loaded_video = key
    st.session_state.video_folder = video_folder
    st.session_state.annotations = annotations
    st.session_state.save_path = save_path
    st.session_state.source_name = source_name
    st.session_state.frame_index = 0


def _orientation_select(label, value, key):
    options = [None] + ORIENTATION_LABELS
    if value not in options:
        options.append(value)
    return st.selectbox(
        label,
        options,
        index=options.index(value),
        format_func=lambda item: "missing" if item is None else str(item),
        key=key,
    )


def _number_or_missing(label, value, key, step=0.01, fmt="%.6f"):
    missing = value is None
    missing = st.checkbox(f"{label} missing", value=missing, key=f"{key}_missing")
    if missing:
        st.number_input(label, value=0.0, step=step, format=fmt, disabled=True, key=f"{key}_disabled")
        return None

    numeric_value = 0.0 if value is None else float(value)
    return st.number_input(label, value=numeric_value, step=step, format=fmt, key=key)


def _current_number_from_state(value, key):
    if st.session_state.get(f"{key}_missing", value is None):
        return None
    return st.session_state.get(key, value)


def _angle_unit_suffix(angle_unit):
    return "deg" if angle_unit == "Degrees" else "rad"


def _angle_from_storage(value, angle_unit):
    if value is None:
        return None
    if angle_unit == "Degrees":
        return math.degrees(float(value))
    return float(value)


def _angle_to_storage(value, angle_unit):
    if value is None:
        return None
    if angle_unit == "Degrees":
        return math.radians(float(value))
    return float(value)


def _point_editor(point_name, value, frame_key):
    if isinstance(value, dict):
        current = [
            value.get("x"),
            value.get("y"),
            value.get("z"),
        ]
        output_as_dict = True
    elif isinstance(value, list) and len(value) >= 3:
        current = value[:3]
        output_as_dict = False
    else:
        current = [None, None, None]
        output_as_dict = False

    st.markdown(f"**{point_name.capitalize()}**")
    cols = st.columns(3)
    edited = []
    axes = ["x", "y", "z"]
    for axis, col, axis_value in zip(axes, cols, current):
        with col:
            edited.append(
                _number_or_missing(
                    axis,
                    axis_value,
                    key=f"{frame_key}_{point_name}_{axis}",
                    step=2.5,
                )
            )

    if all(item is None for item in edited):
        return None
    if output_as_dict:
        return {"x": edited[0], "y": edited[1], "z": edited[2]}
    return edited


def _empty_arm_data():
    return {
        "shoulder": None,
        "elbow": None,
        "wrist": None,
        "joint_angles": {key: None for key in JOINT_ANGLE_KEYS},
        "wrist_orientation": None,
    }


def _ensure_frame_arm_schema(frame_data):
    frame_data.setdefault("torso", None)

    for side in ARM_SIDES:
        arm_key = f"{side}_arm"
        arm_data = frame_data.get(arm_key)
        if not isinstance(arm_data, dict):
            arm_data = _empty_arm_data()
            frame_data[arm_key] = arm_data

        for point_name in POINT_KEYS:
            arm_data.setdefault(point_name, None)

        joint_angles = arm_data.get("joint_angles")
        if not isinstance(joint_angles, dict):
            joint_angles = {}
            arm_data["joint_angles"] = joint_angles
        for angle_name in JOINT_ANGLE_KEYS:
            joint_angles.setdefault(angle_name, None)

        legacy_orientation = frame_data.get(f"wrist_orientation_{side}")
        if arm_data.get("wrist_orientation") is None and legacy_orientation is not None:
            arm_data["wrist_orientation"] = legacy_orientation
        else:
            arm_data.setdefault("wrist_orientation", None)

    return frame_data


def _draft_frame_from_widget_state(video_folder, frame_data, frame_index, angle_unit="Radians"):
    frame_key = f"{video_folder.name}_{frame_index}"
    draft = _ensure_frame_arm_schema(copy.deepcopy(frame_data))

    current_torso = draft.get("torso")
    if isinstance(current_torso, dict):
        values = [current_torso.get("x"), current_torso.get("y"), current_torso.get("z")]
        output_torso_as_dict = True
    elif isinstance(current_torso, list) and len(current_torso) >= 3:
        values = current_torso[:3]
        output_torso_as_dict = False
    else:
        values = [None, None, None]
        output_torso_as_dict = False

    edited_torso = []
    for axis, current_value in zip(("x", "y", "z"), values):
        key = f"{frame_key}_torso_{axis}"
        edited_torso.append(_current_number_from_state(current_value, key))

    if all(item is None for item in edited_torso):
        draft["torso"] = None
    elif output_torso_as_dict:
        draft["torso"] = {"x": edited_torso[0], "y": edited_torso[1], "z": edited_torso[2]}
    else:
        draft["torso"] = edited_torso

    for side in ARM_SIDES:
        arm_data = draft[f"{side}_arm"]
        for point_name in POINT_KEYS:
            current = arm_data.get(point_name)
            if isinstance(current, dict):
                values = [current.get("x"), current.get("y"), current.get("z")]
                output_as_dict = True
            elif isinstance(current, list) and len(current) >= 3:
                values = current[:3]
                output_as_dict = False
            else:
                values = [None, None, None]
                output_as_dict = False

            edited = []
            for axis, current_value in zip(("x", "y", "z"), values):
                key = f"{frame_key}_{side}_{point_name}_{axis}"
                edited.append(_current_number_from_state(current_value, key))

            if all(item is None for item in edited):
                arm_data[point_name] = None
            elif output_as_dict:
                arm_data[point_name] = {"x": edited[0], "y": edited[1], "z": edited[2]}
            else:
                arm_data[point_name] = edited

        joint_angles = arm_data.setdefault("joint_angles", {})
        for angle_name in JOINT_ANGLE_KEYS:
            unit_key = f"{frame_key}_{side}_{angle_name}_{_angle_unit_suffix(angle_unit)}"
            stored_value = joint_angles.get(angle_name)
            display_value = _angle_from_storage(stored_value, angle_unit)
            edited_value = _current_number_from_state(display_value, unit_key)
            joint_angles[angle_name] = _angle_to_storage(edited_value, angle_unit)

        orientation_key = f"{frame_key}_{side}_wrist_orientation"
        if orientation_key in st.session_state:
            arm_data["wrist_orientation"] = st.session_state[orientation_key]

    return draft


def _frame_image_path(video_folder, frame_data, show_metrabs_visualization):
    if show_metrabs_visualization and frame_data.get("metrabs_visualization"):
        candidate = video_folder / frame_data["metrabs_visualization"]
        if candidate.exists():
            return candidate

    image_file = frame_data.get("image_file")
    if not image_file:
        return None

    candidate = video_folder / "frames" / image_file
    if candidate.exists():
        return candidate
    return None


def _raw_frame_path(video_folder, frame_data):
    image_file = frame_data.get("image_file")
    if not image_file:
        return None

    candidate = video_folder / "frames" / image_file
    if candidate.exists():
        return candidate
    return None


def _metrabs_visualization_path(video_folder, frame_data):
    if not frame_data.get("metrabs_visualization"):
        return None

    candidate = video_folder / frame_data["metrabs_visualization"]
    if candidate.exists():
        return candidate
    return None


def _extract_xyz(value):
    if isinstance(value, dict):
        coords = [value.get("x"), value.get("y"), value.get("z")]
    elif isinstance(value, list) and len(value) >= 3:
        coords = value[:3]
    else:
        return None

    if any(item is None for item in coords):
        return None

    try:
        return np.asarray(coords, dtype=np.float32)
    except (TypeError, ValueError):
        return None


def _project_corrected_points(points_3d):
    valid_points = {
        name: point for name, point in points_3d.items()
        if point is not None and np.isfinite(point).all()
    }
    if not valid_points:
        return {}

    point_names = list(valid_points.keys())
    object_points = np.asarray([valid_points[name] for name in point_names], dtype=np.float32)

    projected = {}
    if np.all(np.abs(object_points[:, 2]) > 1e-6):
        try:
            from calibration_utils import load_metrabs_calibration

            camera_matrix, distortion_coeffs = load_metrabs_calibration()
            image_points, _ = cv2.projectPoints(
                object_points.reshape(-1, 1, 3),
                np.zeros((3, 1), dtype=np.float32),
                np.zeros((3, 1), dtype=np.float32),
                camera_matrix,
                distortion_coeffs,
            )
            for name, point in zip(point_names, image_points.reshape(-1, 2)):
                projected[name] = point
            return projected
        except Exception:
            pass

    for name, point in valid_points.items():
        projected[name] = point[:2]
    return projected


def _corrected_overlay_image(video_folder, frame_data):
    raw_path = _raw_frame_path(video_folder, frame_data)
    if raw_path is None:
        return None

    frame = cv2.imread(str(raw_path))
    if frame is None:
        return None

    frame_data = _ensure_frame_arm_schema(copy.deepcopy(frame_data))
    points_3d = {}
    points_3d["torso"] = _extract_xyz(frame_data.get("torso"))
    for side in ARM_SIDES:
        arm_data = frame_data[f"{side}_arm"]
        for point_name in POINT_KEYS:
            points_3d[f"{side}_{point_name}"] = _extract_xyz(arm_data.get(point_name))

    projected = _project_corrected_points(points_3d)

    for side in ARM_SIDES:
        line_color = (0, 255, 0) if side == "left" else (255, 0, 255)
        for start_name, end_name in (
            (f"{side}_shoulder", f"{side}_elbow"),
            (f"{side}_elbow", f"{side}_wrist"),
        ):
            if start_name not in projected or end_name not in projected:
                continue
            start = tuple(np.round(projected[start_name]).astype(int))
            end = tuple(np.round(projected[end_name]).astype(int))
            cv2.line(frame, start, end, line_color, 3, cv2.LINE_AA)

        torso_name = "torso"
        shoulder_name = f"{side}_shoulder"
        if torso_name in projected and shoulder_name in projected:
            start = tuple(np.round(projected[torso_name]).astype(int))
            end = tuple(np.round(projected[shoulder_name]).astype(int))
            cv2.line(frame, start, end, line_color, 2, cv2.LINE_AA)

    for point_name, point in projected.items():
        center = tuple(np.round(point).astype(int))
        cv2.circle(frame, center, 7, POINT_COLORS.get(point_name, (255, 255, 255)), -1, cv2.LINE_AA)

    if not projected:
        cv2.putText(
            frame,
            "No corrected left/right arm coordinates",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 255),
            2,
            cv2.LINE_AA,
        )

    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


def _show_image_path(image_path, video_folder, title):
    if image_path is None:
        st.warning(f"{title} not found.")
        return

    st.image(
        str(image_path),
        caption=f"{title}: {image_path.relative_to(video_folder)}",
        use_container_width=True,
    )


def _render_frame_images(video_folder, frame_data):
    display_mode = st.radio(
        "Image view",
        ["Raw frame", "Metrabs visualization", "Corrected labels", "All three"],
        horizontal=True,
    )

    raw_path = _raw_frame_path(video_folder, frame_data)
    metrabs_path = _metrabs_visualization_path(video_folder, frame_data)

    if display_mode == "Raw frame":
        _show_image_path(raw_path, video_folder, "Raw frame")
        return

    if display_mode == "Metrabs visualization":
        _show_image_path(metrabs_path, video_folder, "Metrabs visualization")
        return

    if display_mode == "Corrected labels":
        corrected = _corrected_overlay_image(video_folder, frame_data)
        if corrected is None:
            st.warning("Corrected labels view could not be rendered.")
        else:
            st.image(corrected, caption="Corrected labels", use_container_width=True)
        return

    raw_col, metrabs_col, corrected_col = st.columns(3)
    with raw_col:
        _show_image_path(raw_path, video_folder, "Raw frame")
    with metrabs_col:
        _show_image_path(metrabs_path, video_folder, "Metrabs visualization")
    with corrected_col:
        corrected = _corrected_overlay_image(video_folder, frame_data)
        if corrected is None:
            st.warning("Corrected labels view could not be rendered.")
        else:
            st.image(corrected, caption="Corrected labels", use_container_width=True)


def _save_current_frame(frame_index, updated_frame):
    annotations = copy.deepcopy(st.session_state.annotations)
    annotations["frames"][frame_index] = updated_frame
    _save_json(st.session_state.save_path, annotations)
    st.session_state.annotations = annotations


def _point_is_missing(value):
    if value is None:
        return True
    if isinstance(value, dict):
        return any(value.get(axis) is None for axis in ("x", "y", "z"))
    if isinstance(value, list):
        return len(value) < 3 or any(item is None for item in value[:3])
    return True


def _missing_fields(frame_data):
    missing = []
    frame_data = _ensure_frame_arm_schema(copy.deepcopy(frame_data))

    if _point_is_missing(frame_data.get("torso")):
        missing.append("torso")

    for side in ARM_SIDES:
        arm_data = frame_data[f"{side}_arm"]
        for point_name in POINT_KEYS:
            if _point_is_missing(arm_data.get(point_name)):
                missing.append(f"{side}_arm.{point_name}")

        joint_angles = arm_data.get("joint_angles", {})
        for angle_name in JOINT_ANGLE_KEYS:
            if joint_angles.get(angle_name) is None:
                missing.append(f"{side}_arm.joint_angles.{angle_name}")

        if arm_data.get("wrist_orientation") is None:
            missing.append(f"{side}_arm.wrist_orientation")

    return missing


def _render_missing_overview(frames):
    rows = []
    missing_frame_options = []

    for index, frame_data in enumerate(frames):
        missing = _missing_fields(frame_data)
        frame_id = frame_data.get("frame_id", index)
        rows.append(
            {
                "frame": frame_id,
                "image": frame_data.get("image_file", ""),
                "missing_count": len(missing),
                "missing_fields": ", ".join(missing) if missing else "complete",
            }
        )
        if missing:
            missing_frame_options.append((index, frame_id, len(missing)))

    complete_count = sum(1 for row in rows if row["missing_count"] == 0)
    st.sidebar.subheader("Missing Data")
    st.sidebar.progress(complete_count / len(frames))
    st.sidebar.caption(f"{complete_count}/{len(frames)} frames complete")

    if missing_frame_options:
        selected = st.sidebar.selectbox(
            "Jump to incomplete frame",
            missing_frame_options,
            format_func=lambda item: f"Frame {item[1]} ({item[2]} missing)",
        )
        if st.sidebar.button("Go to selected frame"):
            st.session_state.frame_index = selected[0]
            st.rerun()
    else:
        st.sidebar.success("All frames are complete.")

    with st.expander("Frame Completion Overview", expanded=False):
        st.dataframe(rows, use_container_width=True, hide_index=True)


def _render_arm_coordinates(frame_key, side, arm_data):
    st.subheader(f"{side.capitalize()} Arm Coordinates")
    for point_name in POINT_KEYS:
        arm_data[point_name] = _point_editor(
            point_name,
            arm_data.get(point_name),
            f"{frame_key}_{side}",
        )


def _render_arm_angles(frame_key, side, arm_data, angle_unit):
    unit_label = "deg" if angle_unit == "Degrees" else "rad"
    unit_suffix = _angle_unit_suffix(angle_unit)

    st.subheader(f"{side.capitalize()} Arm Angles")
    joint_angles = arm_data.setdefault("joint_angles", {})
    for angle_name in JOINT_ANGLE_DISPLAY_ORDER:
        stored_value = joint_angles.get(angle_name)
        display_value = _angle_from_storage(stored_value, angle_unit)
        edited_value = _number_or_missing(
            f"{angle_name} ({unit_label})",
            display_value,
            key=f"{frame_key}_{side}_{angle_name}_{unit_suffix}",
            step=1.0 if angle_unit == "Degrees" else 0.01,
            fmt="%.3f" if angle_unit == "Degrees" else "%.6f",
        )
        joint_angles[angle_name] = _angle_to_storage(edited_value, angle_unit)

    arm_data["wrist_orientation"] = _orientation_select(
        f"{side.capitalize()} wrist orientation",
        arm_data.get("wrist_orientation"),
        key=f"{frame_key}_{side}_wrist_orientation",
    )


def _render_annotation_form(video_folder, frame_data, frame_index):
    frame_key = f"{video_folder.name}_{frame_index}"

    edited_frame = copy.deepcopy(frame_data)
    _ensure_frame_arm_schema(edited_frame)

    st.subheader("Torso")
    edited_frame["torso"] = _point_editor(
        "torso",
        edited_frame.get("torso"),
        frame_key,
    )

    editor_mode = st.radio(
        "Arm data to edit",
        ["Angles", "Coordinates"],
        horizontal=True,
        help="Both arms stay visible side by side. Switch this view to edit angles or 3-D coordinates.",
    )
    angle_unit = st.radio("Angle unit", ANGLE_UNITS, horizontal=True) if editor_mode == "Angles" else ANGLE_UNITS[0]

    left_arm_col, right_arm_col = st.columns(2)
    with left_arm_col:
        if editor_mode == "Angles":
            _render_arm_angles(frame_key, "left", edited_frame["left_arm"], angle_unit)
        else:
            _render_arm_coordinates(frame_key, "left", edited_frame["left_arm"])
    with right_arm_col:
        if editor_mode == "Angles":
            _render_arm_angles(frame_key, "right", edited_frame["right_arm"], angle_unit)
        else:
            _render_arm_coordinates(frame_key, "right", edited_frame["right_arm"])

    save_prev_col, save_col, save_next_col = st.columns(3)
    save_prev_clicked = save_prev_col.button("Save and previous")
    save_clicked = save_col.button("Save frame")
    save_next_clicked = save_next_col.button("Save and next")

    if save_prev_clicked or save_clicked or save_next_clicked:
        edited_frame = _draft_frame_from_widget_state(video_folder, edited_frame, frame_index, angle_unit)
        _save_current_frame(frame_index, edited_frame)
        st.success(f"Saved to {st.session_state.save_path.name}")
        frame_count = len(st.session_state.annotations.get("frames", []))
        if save_prev_clicked:
            st.session_state.frame_index = max(frame_index - 1, 0)
            st.rerun()
        if save_next_clicked:
            st.session_state.frame_index = min(frame_index + 1, frame_count - 1)
            st.rerun()


def main():
    st.set_page_config(page_title="Dataset Annotation", layout="wide")
    st.title("Dataset Annotation")

    dataset_root = Path(
        st.sidebar.text_input("Dataset folder", value=str(DEFAULT_DATASET_ROOT))
    )
    if not dataset_root.exists():
        st.error(f"Dataset folder does not exist: {dataset_root}")
        return

    categories = sorted(path for path in dataset_root.iterdir() if path.is_dir())
    if not categories:
        st.info("No dataset categories found.")
        return

    category = st.sidebar.selectbox("Dataset category", categories, format_func=lambda path: path.name)
    videos = _video_folders(category)
    if not videos:
        st.info(f"No video folders found in {category}")
        return

    video_folder = st.sidebar.selectbox("Video", videos, format_func=lambda path: path.name)
    if st.sidebar.button("Reload annotations"):
        st.session_state.pop("loaded_video", None)
        st.rerun()

    _ensure_loaded(video_folder)

    annotations = st.session_state.annotations
    if annotations is None:
        st.error(f"No annotations.json found for {video_folder}")
        return

    frames = annotations.get("frames", [])
    if not frames:
        st.info("No frames found in the annotation file.")
        return

    st.sidebar.caption(f"Loaded from: {st.session_state.source_name}")
    st.sidebar.caption(f"Saving to: {st.session_state.save_path.name}")
    _render_missing_overview(frames)

    frame_count = len(frames)
    st.session_state.frame_index = max(0, min(st.session_state.frame_index, frame_count - 1))

    nav_prev, nav_current, nav_next = st.columns([1, 2, 1])
    with nav_prev:
        if st.button("Previous", disabled=st.session_state.frame_index <= 0):
            st.session_state.frame_index -= 1
            st.rerun()
    with nav_current:
        selected_number = st.number_input(
            "Frame",
            min_value=0,
            max_value=frame_count - 1,
            value=st.session_state.frame_index,
            step=1,
        )
        if selected_number != st.session_state.frame_index:
            st.session_state.frame_index = int(selected_number)
            st.rerun()
    with nav_next:
        if st.button("Next", disabled=st.session_state.frame_index >= frame_count - 1):
            st.session_state.frame_index += 1
            st.rerun()

    frame_index = st.session_state.frame_index
    frame_data = frames[frame_index]
    preview_frame_data = _draft_frame_from_widget_state(video_folder, frame_data, frame_index)

    left_panel, right_panel = st.columns([3, 2])
    with left_panel:
        st.subheader(
            f"Frame {frame_data.get('frame_id', frame_index)} "
            f"at {frame_data.get('timestamp', 'n/a')}s"
        )
        _render_frame_images(video_folder, preview_frame_data)

    with right_panel:
        _render_annotation_form(video_folder, frame_data, frame_index)

    bottom_prev, bottom_next = st.columns(2)
    with bottom_prev:
        if st.button("Previous frame", disabled=st.session_state.frame_index <= 0, use_container_width=True):
            st.session_state.frame_index -= 1
            st.rerun()
    with bottom_next:
        if st.button("Next frame", disabled=st.session_state.frame_index >= frame_count - 1, use_container_width=True):
            st.session_state.frame_index += 1
            st.rerun()


if __name__ == "__main__":
    main()
