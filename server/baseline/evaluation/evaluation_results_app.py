import json
from functools import lru_cache
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st


CURRENT_FILE = Path(__file__).resolve()
REPO_ROOT = CURRENT_FILE.parents[3]
DEFAULT_RESULTS_PATH = REPO_ROOT / "04-evaluation" / "results" / "ik_method_metrics.json"
DEFAULT_PERFORMANCE_RESULTS_PATH = REPO_ROOT / "04-evaluation" / "results" / "performance_metrics.json"
METRIC_ORDER = ["EEAh", "EEAr", "SOAx", "HJL", "WOM", "HJAr", "TSE", "SYN"]
ANGLE_ORDER = ["shoulder_pitch", "shoulder_roll", "elbow_yaw", "elbow_roll", "wrist_yaw"]
GENERAL_IK_METRIC_ORDER = list(METRIC_ORDER)
GENERAL_EXCLUDED_METHODS = {"ground_truth"}
CATEGORY_DISPLAY_LABELS = {
    "pepper_singularity_motions": "singularity_motions",
}
COLOR_PALETTES = {
    "Blue": ["#0B3C6F", "#145DA0", "#1E81B0", "#4FA3D1", "#82C0E0", "#B8DDF0"],
    "Green": ["#0B4F3A", "#137A54", "#1F9D68", "#52B788", "#86CFA5", "#B7E4C7"],
    "Purple": ["#3C1361", "#5A189A", "#7B2CBF", "#9D4EDD", "#C77DFF", "#E0AAFF"],
    "Orange": ["#7F2704", "#B33B00", "#D95F02", "#F28E2B", "#F6B26B", "#FAD7A0"],
    "Red": ["#67000D", "#99000D", "#CB181D", "#EF3B2C", "#FB6A4A", "#FCAE91"],
    "Multicolor": ["#0B6EBD", "#6CB4EE", "#E53935", "#FF9E9E", "#2A9D8F", "#74E39A"],
}
METHOD_DISPLAY_LABELS = {
    "current_solution_raw": "Unconstrained",
    "current_solution_constrained": "Constrained",
    "ikpy": "IKPy",
    "original_solution": "Darja's IK",
    "ground_truth": "Ground truth",
}
METHOD_DISPLAY_ORDER = ["Darja's IK", "IKPy", "Unconstrained", "Constrained"]
METHOD_ORDER = ["original_solution", "ikpy", "current_solution_raw", "current_solution_constrained"]
PERFORMANCE_METRIC_ORDER = [
    "latency_ms",
    "latency_jitter_ms",
    "cpu_percent",
    "gpu_percent",
    "memory_percent",
    "memory_mb",
]
PERFORMANCE_METRIC_LABELS = {
    "latency_ms": "Latency (ms)",
    "latency_jitter_ms": "Latency jitter (ms)",
    "cpu_percent": "CPU %",
    "memory_percent": "Memory %",
    "memory_mb": "Memory (MiB)",
}
LOWER_IS_BETTER = {"EEAr", "SOAx", "HJL", "HJAr", "TSE", "SYN"}
PERFORMANCE_LOWER_IS_BETTER = set(PERFORMANCE_METRIC_ORDER)
REFERENCE_METRICS = {"EEAh"}

# Explicit directions used only by the matrix visualizations. Values are
# normalized so that 1 always means the best observed result for that metric.
IK_METRIC_DIRECTIONS = {
    "EEAr": "lower_is_better",
    "SOAx": "lower_is_better",
    "HJL": "lower_is_better",
    "WOM": "higher_is_better",
    "HJAr": "lower_is_better",
    "TSE": "lower_is_better",
    "SYN": "lower_is_better",
}


def _load_results(path):
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def _category_display_name(category):
    return CATEGORY_DISPLAY_LABELS.get(category, category)


def _metric_direction(metric_name):
    if metric_name in PERFORMANCE_LOWER_IS_BETTER:
        return "lower is better"
    if metric_name in REFERENCE_METRICS:
        return "reference value"
    if metric_name in LOWER_IS_BETTER:
        return "lower is better"
    return "higher is better"


def _summary_rows(results):
    rows = []
    for video in results.get("videos", []):
        video_id = video.get("video_id")
        category = _category_display_name(video.get("category"))
        for method_name, method_data in video.get("methods", {}).items():
            summary = video.get("summary", {}).get(method_name, {})
            frame_count = len(method_data.get("frames", []))
            for side, metrics in summary.items():
                row = {
                    "video_id": video_id,
                    "category": category,
                    "method": method_name,
                    "side": side,
                    "frames": frame_count,
                }
                for metric_name in METRIC_ORDER:
                    row[metric_name] = metrics.get(metric_name)
                rows.append(row)
    return pd.DataFrame(rows)


def _frame_rows(results):
    rows = []
    for video in results.get("videos", []):
        video_id = video.get("video_id")
        category = _category_display_name(video.get("category"))
        annotation_file = video.get("annotation_file")
        for method_name, method_data in video.get("methods", {}).items():
            for frame in method_data.get("frames", []):
                for side, arm_data in frame.get("arms", {}).items():
                    row = {
                        "video_id": video_id,
                        "category": category,
                        "annotation_file": annotation_file,
                        "method": method_name,
                        "side": side,
                        "frame_id": frame.get("frame_id"),
                        "timestamp": frame.get("timestamp"),
                        "image_file": frame.get("image_file"),
                        "error": arm_data.get("error"),
                        "wom_orientation_label": arm_data.get("wom_orientation_label"),
                        "wom_orientation_detected_label": arm_data.get(
                            "wom_orientation_detected_label"
                        ),
                        "wom_orientation_source": arm_data.get("wom_orientation_source"),
                    }
                    metrics = arm_data.get("metrics") or {}
                    angles = arm_data.get("angles") or {}
                    for metric_name in METRIC_ORDER:
                        row[metric_name] = metrics.get(metric_name)
                    for angle_name, angle_value in angles.items():
                        row[f"angle_{angle_name}"] = angle_value
                    points = arm_data.get("coordinates") or arm_data.get("points") or {}
                    if method_name == "ground_truth" and annotation_file:
                        points = _annotation_points(annotation_file, frame.get("frame_id"), side) or points
                    for joint_name in ("shoulder", "elbow", "wrist"):
                        point = points.get(joint_name) if isinstance(points, dict) else None
                        if isinstance(point, dict):
                            point = [point.get(axis) for axis in ("x", "y", "z")]
                        if isinstance(point, (list, tuple)) and len(point) >= 3:
                            for axis, value in zip(("x", "y", "z"), point[:3]):
                                row[f"{joint_name}_{axis}"] = value
                    rows.append(row)
    return pd.DataFrame(rows)


@lru_cache(maxsize=128)
def _load_annotation_frames(annotation_file):
    try:
        with Path(annotation_file).open("r", encoding="utf-8") as file:
            return json.load(file).get("frames", [])
    except (OSError, json.JSONDecodeError):
        return []


def _annotation_points(annotation_file, frame_id, side):
    for frame in _load_annotation_frames(str(annotation_file)):
        if frame.get("frame_id") != frame_id:
            continue
        arm = frame.get(f"{side}_arm") or {}
        return {joint: arm.get(joint) for joint in ("shoulder", "elbow", "wrist")}
    return None


def _filter_frame_rows(frame_df, videos, methods, sides):
    filtered = frame_df.copy()
    if videos:
        filtered = filtered[filtered["video_id"].isin(videos)]
    if methods:
        filtered = filtered[filtered["method"].isin(methods)]
    if sides:
        filtered = filtered[filtered["side"].isin(sides)]
    return filtered


def _adjacent_frame(frame_ids, current_frame, step):
    """Return the previous/next available frame without crossing the bounds."""
    frames = list(frame_ids)
    if not frames:
        return None
    if current_frame not in frames:
        return frames[0]
    current_index = frames.index(current_frame)
    target_index = max(0, min(current_index + step, len(frames) - 1))
    return frames[target_index]


def _set_comparison_frame(frame_ids, step):
    current_frame = st.session_state.get("comparison_frame")
    st.session_state["comparison_frame"] = _adjacent_frame(
        frame_ids,
        current_frame,
        step,
    )


def _metric_chart(frame_df, metric_name):
    plot_df = frame_df.dropna(subset=[metric_name]).copy()
    if plot_df.empty:
        st.info(f"No values available for {metric_name}.")
        return

    plot_df["series"] = plot_df["method"] + " / " + plot_df["side"]
    chart = (
        alt.Chart(plot_df)
        .mark_line(point=True)
        .encode(
            x=alt.X("frame_id:Q", title="Frame"),
            y=alt.Y(f"{metric_name}:Q", title=f"{metric_name} ({_metric_direction(metric_name)})"),
            color=alt.Color("series:N", title="Method / side"),
            tooltip=[
                "video_id:N",
                "frame_id:Q",
                "method:N",
                "side:N",
                alt.Tooltip(f"{metric_name}:Q", format=".6f"),
            ],
        )
        .properties(height=360)
    )
    st.altair_chart(chart, use_container_width=True)


def _show_frame_image(frame_df, video_id, frame_id, category=None):
    candidates = frame_df[
        (frame_df["video_id"] == video_id)
        & (frame_df["frame_id"] == frame_id)
        & frame_df["annotation_file"].notna()
        & frame_df["image_file"].notna()
    ]
    if category is not None:
        candidates = candidates[candidates["category"] == category]
    if candidates.empty:
        return

    first = candidates.iloc[0]
    annotation_path = Path(first["annotation_file"])
    image_path = annotation_path.parent / "frames" / str(first["image_file"])
    if image_path.exists():
        caption = f"{video_id} frame {frame_id}"
        if category is not None:
            caption = f"{category} / {caption}"
        st.image(str(image_path), caption=caption, use_container_width=True)


def _is_missing(value):
    return value is None or (not isinstance(value, (list, dict)) and pd.isna(value))


def _wom_comparison_rows(frame_rows):
    ground_truth_labels = {
        row["side"]: row.get("wom_orientation_label")
        for _, row in frame_rows[frame_rows["method"] == "ground_truth"].iterrows()
    }
    rows = []
    for _, row in frame_rows.iterrows():
        method_name = row["method"]
        side = row["side"]
        gt_label = ground_truth_labels.get(side)
        effective_label = row.get("wom_orientation_label")
        detected_label = row.get("wom_orientation_detected_label")
        source = row.get("wom_orientation_source")
        wom = row.get("WOM")

        if method_name == "ground_truth":
            status = "Reference"
            reason = "Manual ground-truth wrist orientation"
        elif _is_missing(wom):
            status = "Unavailable"
            reason = "WOM could not be computed"
        elif wom == 1:
            status = "Match"
            reason = "Effective method orientation matches ground truth"
        elif source == "previous_mediapipe":
            status = "Mismatch"
            reason = "MediaPipe was missing; retained previous orientation differs from ground truth"
        elif source == "fk_zero_wrist_yaw":
            status = "Mismatch"
            reason = "No current/previous MediaPipe label; zero-WristYaw FK label differs from ground truth"
        elif source == "mediapipe":
            status = "Mismatch"
            reason = "Current MediaPipe orientation differs from ground truth"
        else:
            status = "Mismatch"
            reason = "Method orientation differs from ground truth"

        rows.append({
            "Method": _method_display_name(method_name),
            "Arm": side,
            "Ground truth": gt_label,
            "Detected orientation": detected_label,
            "Effective orientation": effective_label,
            "Orientation source": source,
            "WOM": wom,
            "Status": status,
            "Explanation": reason,
        })
    return pd.DataFrame(rows)


def _angle_comparison_rows(frame_rows, angle_unit):
    angle_columns = [
        f"angle_{angle_name}"
        for angle_name in ANGLE_ORDER
        if f"angle_{angle_name}" in frame_rows.columns
    ]
    ground_truth_angles = {}
    for _, row in frame_rows[frame_rows["method"] == "ground_truth"].iterrows():
        ground_truth_angles[row["side"]] = {
            column: row.get(column) for column in angle_columns
        }

    factor = 180.0 / 3.141592653589793 if angle_unit == "Degrees" else 1.0
    suffix = "deg" if angle_unit == "Degrees" else "rad"
    rows = []
    for _, row in frame_rows.iterrows():
        output = {
            "Method": _method_display_name(row["method"]),
            "Arm": row["side"],
        }
        gt_angles = ground_truth_angles.get(row["side"], {})
        for column in angle_columns:
            label = column.removeprefix("angle_")
            value = row.get(column)
            gt_value = gt_angles.get(column)
            output[f"{label} ({suffix})"] = (
                None if _is_missing(value) else float(value) * factor
            )
            output[f"{label} delta ({suffix})"] = (
                None
                if _is_missing(value) or _is_missing(gt_value)
                else (float(value) - float(gt_value)) * factor
            )
        rows.append(output)
    return pd.DataFrame(rows)


def _style_method_angle_mismatches(angle_df):
    """Highlight constrained/Darja angle values when they differ on an arm."""
    styles = pd.DataFrame("", index=angle_df.index, columns=angle_df.columns)
    value_columns = [
        column for column in angle_df.columns
        if not "delta" in column and column not in {"Method", "Arm"}
    ]
    for side in angle_df["Arm"].dropna().unique():
        side_rows = angle_df[angle_df["Arm"] == side]
        constrained = side_rows[side_rows["Method"] == "Constrained"]
        darja = side_rows[side_rows["Method"] == "Darja's IK"]
        if constrained.empty or darja.empty:
            continue
        constrained_index = constrained.index[0]
        darja_index = darja.index[0]
        for column in value_columns:
            constrained_value = constrained.iloc[0][column]
            darja_value = darja.iloc[0][column]
            if _is_missing(constrained_value) or _is_missing(darja_value):
                continue
            if abs(float(constrained_value) - float(darja_value)) > 1e-9:
                styles.loc[[constrained_index, darja_index], column] = (
                    "background-color: #f8d7da; color: #9b1c1c; font-weight: 700;"
                )
    return styles


def _xyz_comparison_rows(frame_rows):
    rows = []
    for _, row in frame_rows.iterrows():
        output = {
            "Method": _method_display_name(row["method"]),
            "Arm": row["side"],
        }
        for joint_name in ("shoulder", "elbow", "wrist"):
            for axis in ("x", "y", "z"):
                column = f"{joint_name}_{axis}"
                output[f"{joint_name.capitalize()} {axis.upper()}"] = row.get(column)
        rows.append(output)
    return pd.DataFrame(rows)


def _style_xyz_mismatches(xyz_df):
    styles = pd.DataFrame("", index=xyz_df.index, columns=xyz_df.columns)
    value_columns = [column for column in xyz_df.columns if column not in {"Method", "Arm"}]
    for side in xyz_df["Arm"].dropna().unique():
        side_rows = xyz_df[xyz_df["Arm"] == side]
        constrained = side_rows[side_rows["Method"] == "Constrained"]
        darja = side_rows[side_rows["Method"] == "Darja's IK"]
        if constrained.empty or darja.empty:
            continue
        constrained_index = constrained.index[0]
        darja_index = darja.index[0]
        for column in value_columns:
            constrained_value = constrained.iloc[0][column]
            darja_value = darja.iloc[0][column]
            if _is_missing(constrained_value) or _is_missing(darja_value):
                continue
            if abs(float(constrained_value) - float(darja_value)) > 1e-9:
                styles.loc[[constrained_index, darja_index], column] = (
                    "background-color: #f8d7da; color: #9b1c1c; font-weight: 700;"
                )
    return styles


def _render_frame_comparison(frame_df):
    st.subheader("Ground Truth vs Methods, Frame by Frame")
    st.caption(
        "Choose one recorded frame to compare every method with its ground-truth "
        "joint angles and wrist-orientation annotation. Angle deltas are method minus ground truth."
    )

    categories = sorted(frame_df["category"].dropna().unique())
    selected_category = st.selectbox(
        "Category",
        categories,
        key="comparison_category",
    )
    category_rows = frame_df[frame_df["category"] == selected_category]
    video_ids = sorted(category_rows["video_id"].dropna().unique())
    selected_video = st.selectbox(
        "Video",
        video_ids,
        key="comparison_video",
    )
    video_rows = category_rows[category_rows["video_id"] == selected_video]
    frame_ids = sorted(video_rows["frame_id"].dropna().unique())
    if st.session_state.get("comparison_frame") not in frame_ids:
        st.session_state["comparison_frame"] = frame_ids[0]

    previous_col, slider_col, next_col = st.columns([1, 6, 1])
    with previous_col:
        st.button(
            "← Previous frame",
            key="comparison_previous_frame",
            disabled=st.session_state["comparison_frame"] == frame_ids[0],
            on_click=_set_comparison_frame,
            args=(frame_ids, -1),
            use_container_width=True,
        )
    with slider_col:
        selected_frame = st.select_slider(
            "Frame",
            options=frame_ids,
            key="comparison_frame",
        )
    with next_col:
        st.button(
            "Next frame →",
            key="comparison_next_frame",
            disabled=st.session_state["comparison_frame"] == frame_ids[-1],
            on_click=_set_comparison_frame,
            args=(frame_ids, 1),
            use_container_width=True,
        )

    available_methods = sorted(
        video_rows["method"].dropna().unique(),
        key=_method_sort_key,
    )
    comparison_methods = [
        method for method in available_methods if method != "ground_truth"
    ]
    available_sides = sorted(video_rows["side"].dropna().unique())
    control_left, control_right, control_unit = st.columns(3)
    with control_left:
        selected_methods = st.multiselect(
            "Methods to compare",
            comparison_methods,
            default=comparison_methods,
            key="comparison_methods",
            format_func=_method_display_name,
        )
    with control_right:
        selected_sides = st.multiselect(
            "Arms",
            available_sides,
            default=available_sides,
            key="comparison_sides",
        )
    with control_unit:
        angle_unit = st.radio(
            "Angle unit",
            ["Degrees", "Radians"],
            horizontal=True,
            key="comparison_angle_unit",
        )

    frame_rows = video_rows[
        (video_rows["frame_id"] == selected_frame)
        & video_rows["method"].isin(["ground_truth", *selected_methods])
        & video_rows["side"].isin(selected_sides)
    ].copy()
    if frame_rows.empty:
        st.info("Select at least one method and arm.")
        return

    method_order = {method: index for index, method in enumerate(available_methods)}
    side_order = {side: index for index, side in enumerate(available_sides)}
    frame_rows["_method_order"] = frame_rows["method"].map(method_order)
    frame_rows["_side_order"] = frame_rows["side"].map(side_order)
    frame_rows = frame_rows.sort_values(["_side_order", "_method_order"])

    image_col, values_col = st.columns([1, 2])
    with image_col:
        _show_frame_image(
            frame_df,
            selected_video,
            selected_frame,
            category=selected_category,
        )
    with values_col:
        st.markdown("#### Wrist orientation / WOM")
        wom_rows = _wom_comparison_rows(frame_rows)
        st.dataframe(wom_rows, use_container_width=True, hide_index=True)

    st.markdown("#### Joint angles and differences from ground truth")
    angle_rows = _angle_comparison_rows(frame_rows, angle_unit)
    st.dataframe(
        angle_rows.style.apply(_style_method_angle_mismatches, axis=None),
        use_container_width=True,
        hide_index=True,
    )
    st.caption("Red cells mark joint-angle values where Constrained and Darja's IK differ for the same arm.")

    st.markdown("#### Joint XYZ coordinates")
    xyz_rows = _xyz_comparison_rows(frame_rows)
    st.dataframe(
        xyz_rows.style.apply(_style_xyz_mismatches, axis=None),
        use_container_width=True,
        hide_index=True,
    )

    st.markdown("#### All frame-level metrics")
    metric_rows = frame_rows[["method", "side", *METRIC_ORDER, "error"]].copy()
    metric_rows["method"] = metric_rows["method"].map(_method_display_name)
    metric_rows = metric_rows.rename(columns={"method": "Method", "side": "Arm"})
    st.dataframe(metric_rows, use_container_width=True, hide_index=True)


def _render_ik_results(default_path):
    results_path = Path(
        st.sidebar.text_input("IK Results JSON", value=str(default_path))
    )
    if not results_path.exists():
        st.error(f"Results JSON not found: {results_path}")
        return

    results = _load_results(results_path)
    summary_df = _summary_rows(results)
    frame_df = _frame_rows(results)

    if summary_df.empty or frame_df.empty:
        st.info("No evaluation rows found in the selected results file.")
        return

    st.sidebar.caption(f"Loaded: {results_path}")
    available_videos = sorted(frame_df["video_id"].dropna().unique())
    available_methods = sorted(
        frame_df["method"].dropna().unique(),
        key=_method_sort_key,
    )
    available_sides = sorted(frame_df["side"].dropna().unique())

    videos = st.sidebar.multiselect("Videos", available_videos, default=available_videos)
    methods = st.sidebar.multiselect(
        "Methods",
        available_methods,
        default=available_methods,
    )
    sides = st.sidebar.multiselect("Sides", available_sides, default=available_sides)

    filtered_frames = _filter_frame_rows(frame_df, videos, methods, sides)
    filtered_summary = summary_df[
        summary_df["video_id"].isin(videos)
        & summary_df["method"].isin(methods)
        & summary_df["side"].isin(sides)
    ]

    metric_notes = results.get("metrics", {})
    with st.expander("Metric Notes", expanded=False):
        for metric_name in METRIC_ORDER:
            if metric_name in metric_notes:
                st.markdown(f"**{metric_name}**: {metric_notes[metric_name]}")
        for note in results.get("notes", []):
            st.caption(note)

    summary_tab, plots_tab, frames_tab, comparison_tab, errors_tab = st.tabs(
        ["Summary", "Frame Plots", "Frame Details", "Ground Truth Comparison", "Errors"]
    )

    with summary_tab:
        st.subheader("Per-video Summary")
        st.dataframe(
            filtered_summary.sort_values(["video_id", "method", "side"]),
            use_container_width=True,
            hide_index=True,
        )

        metric = st.selectbox("Compare metric", METRIC_ORDER, index=0, key="summary_metric")
        chart_df = filtered_summary.dropna(subset=[metric]).copy()
        if not chart_df.empty:
            chart_df["series"] = chart_df["method"] + " / " + chart_df["side"]
            chart = (
                alt.Chart(chart_df)
                .mark_bar()
                .encode(
                    x=alt.X("video_id:N", title="Video"),
                    y=alt.Y(f"{metric}:Q", title=f"{metric} ({_metric_direction(metric)})"),
                    color=alt.Color("series:N", title="Method / side"),
                    xOffset="series:N",
                    tooltip=[
                        "video_id:N",
                        "method:N",
                        "side:N",
                        alt.Tooltip(f"{metric}:Q", format=".6f"),
                    ],
                )
                .properties(height=360)
            )
            st.altair_chart(chart, use_container_width=True)

    with plots_tab:
        st.subheader("Frame-level Metric Curves")
        selected_video = st.selectbox("Video", videos or available_videos)
        selected_metric = st.selectbox("Metric", METRIC_ORDER, index=0)
        plot_rows = filtered_frames[filtered_frames["video_id"] == selected_video]
        _metric_chart(plot_rows, selected_metric)

    with frames_tab:
        st.subheader("Frame Details")
        selected_video = st.selectbox("Video", videos or available_videos, key="frame_video")
        video_frames = filtered_frames[filtered_frames["video_id"] == selected_video]
        frame_ids = sorted(video_frames["frame_id"].dropna().unique())
        if frame_ids:
            selected_frame = st.selectbox("Frame", frame_ids)
            left_col, right_col = st.columns([1, 2])
            with left_col:
                _show_frame_image(frame_df, selected_video, selected_frame)
            with right_col:
                frame_rows = video_frames[video_frames["frame_id"] == selected_frame]
                columns = ["method", "side", *METRIC_ORDER, "error"]
                st.dataframe(frame_rows[columns], use_container_width=True, hide_index=True)

                angle_columns = [column for column in frame_rows.columns if column.startswith("angle_")]
                st.dataframe(
                    frame_rows[["method", "side", *angle_columns]],
                    use_container_width=True,
                    hide_index=True,
                )

    with comparison_tab:
        _render_frame_comparison(frame_df)

    with errors_tab:
        st.subheader("Method Errors")
        error_rows = filtered_frames[filtered_frames["error"].notna()]
        if error_rows.empty:
            st.success("No method errors in the current selection.")
        else:
            st.dataframe(
                error_rows[["video_id", "frame_id", "method", "side", "error"]],
                use_container_width=True,
                hide_index=True,
            )


def _performance_summary_rows(results):
    rows = []
    for video in results.get("videos", []):
        video_id = video.get("video_id")
        category = _category_display_name(video.get("category"))
        for method_name, method_data in video.get("methods", {}).items():
            summary = method_data.get("summary", {})
            row = {
                "video_id": video_id,
                "category": category,
                "method": method_name,
                "frames": summary.get("frame_count"),
                "errors": summary.get("error_count"),
            }
            for metric_name in PERFORMANCE_METRIC_ORDER:
                row[metric_name] = summary.get(metric_name)
            rows.append(row)
    return pd.DataFrame(rows)


def _performance_frame_rows(results):
    rows = []
    for video in results.get("videos", []):
        video_id = video.get("video_id")
        category = _category_display_name(video.get("category"))
        annotation_file = video.get("annotation_file")
        for method_name, method_data in video.get("methods", {}).items():
            for frame in method_data.get("frames", []):
                row = {
                    "video_id": video_id,
                    "category": category,
                    "annotation_file": annotation_file,
                    "method": method_name,
                    "repeat_index": frame.get("repeat_index"),
                    "frame_id": frame.get("frame_id"),
                    "timestamp": frame.get("timestamp"),
                    "image_file": frame.get("image_file"),
                    "error": frame.get("error"),
                }
                for metric_name in PERFORMANCE_METRIC_ORDER:
                    row[metric_name] = frame.get(metric_name)
                rows.append(row)
    return pd.DataFrame(rows)


def _filter_performance_rows(frame_df, videos, methods):
    filtered = frame_df.copy()
    if videos:
        filtered = filtered[filtered["video_id"].isin(videos)]
    if methods:
        filtered = filtered[filtered["method"].isin(methods)]
    return filtered


def _selected_category_label(categories):
    if not categories:
        return "All selected categories"
    if len(categories) == 1:
        return categories[0]
    return "Selected categories"


def _add_method_display_labels(df):
    df = df.copy()
    df["method_label"] = df["method"].map(METHOD_DISPLAY_LABELS).fillna(df["method"])
    return df


def _method_display_name(method_name):
    return METHOD_DISPLAY_LABELS.get(method_name, method_name)


def _method_sort_key(method_name):
    if method_name in METHOD_ORDER:
        return METHOD_ORDER.index(method_name)
    return len(METHOD_ORDER)


def _aggregate_ik_for_general(summary_df, categories, methods, group_categories):
    filtered = summary_df.copy()
    if categories:
        filtered = filtered[filtered["category"].isin(categories)]
    filtered = filtered[~filtered["method"].isin(GENERAL_EXCLUDED_METHODS)]
    if methods:
        filtered = filtered[filtered["method"].isin(methods)]
    if filtered.empty:
        return filtered

    filtered = filtered.copy()
    if group_categories:
        filtered["category_group"] = _selected_category_label(categories)
        grouped_source = filtered
    else:
        filtered["category_group"] = filtered["category"].fillna("Uncategorized")
        grouped_source = filtered
        selected_count = filtered["category"].dropna().nunique()
        if selected_count > 1:
            aggregate_rows = filtered.copy()
            aggregate_rows["category_group"] = _selected_category_label(categories)
            grouped_source = pd.concat([filtered, aggregate_rows], ignore_index=True)

    grouped = (
        grouped_source.groupby(["category_group", "method"], dropna=False)[GENERAL_IK_METRIC_ORDER]
        .mean(numeric_only=True)
        .reset_index()
    )
    return _add_method_display_labels(grouped).sort_values(
        ["category_group", "method"],
        key=lambda series: series.map(_method_sort_key) if series.name == "method" else series,
    )


def _aggregate_performance_for_general(summary_df, categories, methods, group_categories):
    filtered = summary_df.copy()
    if categories:
        filtered = filtered[filtered["category"].isin(categories)]
    if methods:
        filtered = filtered[filtered["method"].isin(methods)]
    if filtered.empty:
        return filtered

    filtered = filtered.copy()
    if group_categories:
        filtered["category_group"] = _selected_category_label(categories)
        grouped_source = filtered
    else:
        filtered["category_group"] = filtered["category"].fillna("Uncategorized")
        grouped_source = filtered
        selected_count = filtered["category"].dropna().nunique()
        if selected_count > 1:
            aggregate_rows = filtered.copy()
            aggregate_rows["category_group"] = _selected_category_label(categories)
            grouped_source = pd.concat([filtered, aggregate_rows], ignore_index=True)

    grouped = (
        grouped_source.groupby(["category_group", "method"], dropna=False)[PERFORMANCE_METRIC_ORDER]
        .mean(numeric_only=True)
        .reset_index()
    )
    return _add_method_display_labels(grouped).sort_values(
        ["category_group", "method"],
        key=lambda series: series.map(_method_sort_key) if series.name == "method" else series,
    )


def _bar_group_encoding(grouping, palette_name="Multicolor"):
    if grouping == "Category":
        return {
            "x": alt.X(
                "category_group:N",
                title="Category",
                sort=None,
                axis=alt.Axis(labelAngle=0, labelLimit=260),
            ),
            "color": alt.Color(
                "method_label:N",
                title="Method",
                sort=METHOD_DISPLAY_ORDER,
                scale=alt.Scale(range=color_range),
            ),
            "x_offset": alt.XOffset("method_label:N", sort=METHOD_DISPLAY_ORDER),
        }
    return {
        "x": alt.X(
            "method_label:N",
            title="Method",
            sort=METHOD_DISPLAY_ORDER,
            axis=alt.Axis(labelAngle=0, labelLimit=260),
        ),
        "color": alt.Color(
            "category_group:N",
            title="Category",
            scale=alt.Scale(range=color_range),
        ),
        "x_offset": "category_group:N",
    }


def _general_metric_bar_chart(plot_df, metric_name, grouping, palette_name="Multicolor"):
    metric_df = plot_df.dropna(subset=[metric_name]).copy()
    if metric_df.empty:
        st.info(f"No values available for {metric_name}.")
        return

    encoding = _bar_group_encoding(grouping, palette_name)
    base = alt.Chart(metric_df).encode(
        x=encoding["x"],
        y=alt.Y(f"{metric_name}:Q", title=f"{metric_name} ({_metric_direction(metric_name)})"),
        color=encoding["color"],
        xOffset=encoding["x_offset"],
        tooltip=[
            "category_group:N",
            "method:N",
            "method_label:N",
            alt.Tooltip(f"{metric_name}:Q", format=".6f"),
        ],
    )
    bars = base.mark_bar()
    labels = base.mark_text(
        align="center",
        baseline="bottom",
        dy=-4,
        fontSize=11,
    ).encode(
        text=alt.Text(f"{metric_name}:Q", format=".3f"),
    )

    chart = (
        (bars + labels)
        .resolve_scale(color="shared")
        .properties(height=320)
    )
    st.altair_chart(chart, use_container_width=True)


def _bar_chart_with_value_labels(
    metric_df,
    metric_name,
    title,
    grouping,
    label_format=".3f",
    label_font_size=11,
    palette_name="Multicolor",
):
    encoding = _bar_group_encoding(grouping, palette_name)
    base = alt.Chart(metric_df).encode(
        x=encoding["x"],
        y=alt.Y(
            f"{metric_name}:Q",
            title=title,
        ),
        color=encoding["color"],
        xOffset=encoding["x_offset"],
        tooltip=[
            "category_group:N",
            "method:N",
            "method_label:N",
            alt.Tooltip(f"{metric_name}:Q", format=".6f"),
        ],
    )
    bars = base.mark_bar()
    labels = base.mark_text(
        align="center",
        baseline="bottom",
        dy=-4,
        fontSize=label_font_size,
    ).encode(
        text=alt.Text(f"{metric_name}:Q", format=label_format),
    )
    return bars + labels


def _latency_subplot(plot_df, grouping, palette_name="Multicolor"):
    available_metrics = ["latency_ms", "latency_jitter_ms"]
    clean_df = plot_df.dropna(subset=available_metrics, how="all").copy()
    if clean_df.empty:
        st.info("No latency values available for the selected categories.")
        return

    for metric_name in available_metrics:
        metric_df = clean_df.dropna(subset=[metric_name]).copy()
        chart = (
            _bar_chart_with_value_labels(
                metric_df,
                metric_name,
                f"{PERFORMANCE_METRIC_LABELS.get(metric_name, metric_name)} (lower is better)",
                grouping,
                label_format=".2f",
                label_font_size=10,
                palette_name=palette_name,
            )
            .properties(
                title=PERFORMANCE_METRIC_LABELS.get(metric_name, metric_name),
                height=380,
                width=900,
            )
        )
        st.altair_chart(chart, use_container_width=True)


def _ik_metric_long(summary_df, categories, methods, metrics):
    """Convert the existing wide IK summary dataframe to metric-long rows."""
    id_columns = ["video_id", "category", "method", "side"]
    available_metrics = [metric for metric in metrics if metric in summary_df.columns]
    if not available_metrics:
        return pd.DataFrame()
    filtered = summary_df[
        summary_df["category"].isin(categories)
        & summary_df["method"].isin(methods)
    ]
    if filtered.empty:
        return pd.DataFrame()
    return (
        filtered[id_columns + available_metrics]
        .melt(id_vars=id_columns, var_name="metric", value_name="value")
        .dropna(subset=["value"])
    )


def _aggregate_ik_metric_long(summary_df, categories, methods, metrics):
    long_df = _ik_metric_long(summary_df, categories, methods, metrics)
    if long_df.empty:
        return long_df
    grouped = (
        long_df.groupby(["method", "category", "metric"], as_index=False)
        .agg(
            value=("value", "mean"),
            observation_count=("value", "count"),
            standard_deviation=("value", "std"),
        )
    )
    grouped["method_label"] = grouped["method"].map(_method_display_name)
    grouped["category_label"] = grouped["category"]
    grouped["ci_low"] = grouped["value"]
    grouped["ci_high"] = grouped["value"]
    multi_observation = grouped["observation_count"] > 1
    standard_error = grouped["standard_deviation"] / grouped["observation_count"].pow(0.5)
    grouped.loc[multi_observation, "ci_low"] = (
        grouped.loc[multi_observation, "value"]
        - 1.96 * standard_error.loc[multi_observation]
    )
    grouped.loc[multi_observation, "ci_high"] = (
        grouped.loc[multi_observation, "value"]
        + 1.96 * standard_error.loc[multi_observation]
    )
    normalized_parts = []
    for metric_name, metric_rows in grouped.groupby("metric", sort=False):
        values = metric_rows["value"]
        minimum = values.min()
        maximum = values.max()
        if minimum == maximum:
            normalized = pd.Series(0.5, index=metric_rows.index)
        elif IK_METRIC_DIRECTIONS.get(metric_name) == "higher_is_better":
            normalized = (values - minimum) / (maximum - minimum)
        else:
            normalized = (maximum - values) / (maximum - minimum)
        normalized_parts.append(normalized)
    grouped["normalized_value"] = pd.concat(normalized_parts).sort_index()
    grouped["direction"] = grouped["metric"].map(IK_METRIC_DIRECTIONS).fillna("lower_is_better")
    grouped["metric_label"] = grouped.apply(
        lambda row: f"{row['metric']} ({'↑' if row['direction'] == 'higher_is_better' else '↓'})",
        axis=1,
    )
    return grouped


def _render_faceted_ik_heatmap(summary_df, categories, methods, metrics, palette_name):
    aggregated = _aggregate_ik_metric_long(summary_df, categories, methods, metrics)
    if aggregated.empty:
        st.info("No aggregated IK values are available for the selected heatmap population.")
        return

    metric_order = [metric for metric in metrics if metric in aggregated["metric"].unique()]
    method_order = [method for method in METHOD_DISPLAY_ORDER if method in aggregated["method_label"].unique()]
    color_range = COLOR_PALETTES.get(palette_name, COLOR_PALETTES["Multicolor"])
    rect_base = alt.Chart(aggregated).encode(
        x=alt.X("metric:N", title="Metric", sort=metric_order, axis=alt.Axis(labelAngle=0, labelLimit=100)),
        y=alt.Y("category_label:N", title="Video category", sort=categories, axis=alt.Axis(labelLimit=180)),
        color=alt.Color(
            "normalized_value:Q",
            title="Normalized performance",
            scale=alt.Scale(domain=[0, 1], scheme="viridis"),
            legend=alt.Legend(format=".1f"),
        ),
        tooltip=[
            alt.Tooltip("method_label:N", title="Method"),
            alt.Tooltip("category_label:N", title="Category"),
            alt.Tooltip("metric:N", title="Metric"),
            alt.Tooltip("value:Q", title="Original value", format=".3f"),
            alt.Tooltip("normalized_value:Q", title="Normalized performance", format=".3f"),
            alt.Tooltip("observation_count:Q", title="Observations"),
        ],
    )
    rects = rect_base.mark_rect(stroke="#ffffff", strokeWidth=1)
    labels = rect_base.mark_text(fontSize=11, color="#172c2b").encode(
        text=alt.Text("value:Q", format=".3f"),
        color=alt.value("#172c2b"),
    )
    chart = (
        alt.layer(rects, labels)
        .facet(
            column=alt.Column("method_label:N", title=None, sort=[_method_display_name(method) for method in METHOD_ORDER]),
            columns=2,
        )
        .resolve_scale(color="shared")
        .properties(
            width=300,
            height=max(180, 34 * len(categories)),
        )
    )
    st.altair_chart(chart, use_container_width=True)
    st.caption(
        "Colours show metric-wise normalized performance (0 = worst observed, 1 = best observed). "
        "Cell annotations show the original aggregated metric values; missing combinations remain blank."
    )


def _render_grouped_ik_dot_plots(summary_df, categories, methods, metrics, palette_name, show_error_bars):
    aggregated = _aggregate_ik_metric_long(summary_df, categories, methods, metrics)
    if aggregated.empty:
        st.info("No aggregated IK values are available for the selected dot-plot population.")
        return

    metric_order = [metric for metric in metrics if metric in aggregated["metric"].unique()]
    method_order = [method for method in METHOD_DISPLAY_ORDER if method in aggregated["method_label"].unique()]
    color_range = COLOR_PALETTES.get(palette_name, COLOR_PALETTES["Multicolor"])
    base = alt.Chart(aggregated).encode(
        y=alt.Y(
            "category_label:N",
            title="Video category",
            sort=categories,
            axis=alt.Axis(grid=True, labelLimit=180),
        ),
        yOffset=alt.YOffset("method_label:N", sort=[_method_display_name(method) for method in METHOD_ORDER]),
        color=alt.Color(
            "method_label:N",
            title="Method",
            sort=[_method_display_name(method) for method in METHOD_ORDER],
            scale=alt.Scale(range=color_range),
        ),
        shape=alt.Shape(
            "method_label:N",
            title="Method",
            sort=[_method_display_name(method) for method in METHOD_ORDER],
        ),
        tooltip=[
            alt.Tooltip("method_label:N", title="Method"),
            alt.Tooltip("category_label:N", title="Category"),
            alt.Tooltip("metric:N", title="Metric"),
            alt.Tooltip("value:Q", title="Original value", format=".3f"),
            alt.Tooltip("observation_count:Q", title="Observations"),
        ],
    )
    points = base.mark_point(filled=True, size=90).encode(
        x=alt.X("value:Q", title="Original metric value", scale=alt.Scale(zero=False), axis=alt.Axis(grid=False)),
    )
    layers = [points]
    if show_error_bars:
        error_bars = base.mark_rule(strokeWidth=2).encode(
            x=alt.X("ci_low:Q", title="Original metric value", scale=alt.Scale(zero=False)),
            x2="ci_high:Q",
        )
        layers.insert(0, error_bars)
    chart = (
        alt.layer(*layers)
        .facet(
            column=alt.Column(
                "metric_label:N",
                title="Metric (direction)",
                sort=[
                    f"{metric} ({'↑' if IK_METRIC_DIRECTIONS.get(metric) == 'higher_is_better' else '↓'})"
                    for metric in metric_order
                ],
            ),
            columns=2,
        )
        .resolve_scale(x="independent", y="shared", color="shared", shape="shared")
        .properties(
            width=320,
            height=max(180, 34 * len(categories)),
        )
    )
    st.altair_chart(chart, use_container_width=True)
    st.caption(
        "Points use original metric values. Facet titles show direction (↑ higher is better, ↓ lower is better); "
        "error bars are 95% normal-approximation intervals only when multiple observations are available."
    )


def _render_general_plots(default_ik_path, default_performance_path):
    ik_results_path = Path(
        st.sidebar.text_input("IK Results JSON", value=str(default_ik_path), key="general_ik_path")
    )
    performance_results_path = Path(
        st.sidebar.text_input(
            "Performance JSON",
            value=str(default_performance_path),
            key="general_performance_path",
        )
    )

    missing_paths = [
        str(path)
        for path in (ik_results_path, performance_results_path)
        if not path.exists()
    ]
    if missing_paths:
        st.error("Missing results file(s):\n\n" + "\n".join(missing_paths))
        return

    ik_results = _load_results(ik_results_path)
    performance_results = _load_results(performance_results_path)
    ik_summary_df = _summary_rows(ik_results)
    performance_summary_df = _performance_summary_rows(performance_results)

    if ik_summary_df.empty and performance_summary_df.empty:
        st.info("No rows found in the selected result files.")
        return

    categories = sorted(
        set(ik_summary_df.get("category", pd.Series(dtype=str)).dropna().unique())
        | set(performance_summary_df.get("category", pd.Series(dtype=str)).dropna().unique())
    )
    selected_categories = st.sidebar.multiselect(
        "Categories",
        categories,
        default=categories,
        key="general_categories",
    )
    group_categories = st.sidebar.checkbox(
        "Group selected categories",
        value=False,
        key="general_group_categories",
    )
    if categories and not selected_categories:
        st.info("Select at least one category to display the general plots.")
        return

    available_methods = sorted(
        (
            set(ik_summary_df.get("method", pd.Series(dtype=str)).dropna().unique())
            | set(performance_summary_df.get("method", pd.Series(dtype=str)).dropna().unique())
        )
        - GENERAL_EXCLUDED_METHODS,
        key=_method_sort_key,
    )
    selected_methods = st.sidebar.multiselect(
        "Methods",
        available_methods,
        default=available_methods,
        key="general_methods",
        format_func=_method_display_name,
    )
    if available_methods and not selected_methods:
        st.info("Select at least one method to display the general plots.")
        return

    selected_metrics = st.multiselect(
        "IK metrics to display",
        GENERAL_IK_METRIC_ORDER,
        default=GENERAL_IK_METRIC_ORDER,
        key="general_ik_metrics",
        help="Select or deselect metrics to update the general IK bar plots.",
    )
    bar_grouping = st.selectbox(
        "Bar grouping",
        ["Method", "Category"],
        index=0,
        key="general_bar_grouping",
        help=(
            "Method: each method is a group and categories are the bars. "
            "Category: each category is a group and methods are the bars."
        ),
    )
    color_palette = st.selectbox(
        "Color palette",
        list(COLOR_PALETTES),
        index=list(COLOR_PALETTES).index("Multicolor"),
        key="general_color_palette",
        help="Monochromatic palettes use different tonalities of the same base color.",
    )

    st.caption(f"IK results: {ik_results_path}")
    st.caption(f"Performance results: {performance_results_path}")

    ik_plot_df = _aggregate_ik_for_general(
        ik_summary_df,
        selected_categories,
        selected_methods,
        group_categories,
    )
    performance_plot_df = _aggregate_performance_for_general(
        performance_summary_df,
        selected_categories,
        selected_methods,
        group_categories,
    )

    ik_tab, performance_tab, matrix_tab, data_tab = st.tabs(
        ["IK Metric Bars", "Latency Subplot", "Matrix Visualizations", "Aggregated Data"]
    )

    with ik_tab:
        st.subheader(f"IK Metrics by {bar_grouping}")
        if ik_plot_df.empty:
            st.info("No IK values available for the selected categories.")
        elif not selected_metrics:
            st.info("Select at least one IK metric to display.")
        else:
            for row_start in range(0, len(selected_metrics), 2):
                columns = st.columns(2)
                for column, metric_name in zip(columns, selected_metrics[row_start:row_start + 2]):
                    with column:
                        st.markdown(f"**{metric_name}**")
                        _general_metric_bar_chart(
                            ik_plot_df,
                            metric_name,
                            bar_grouping,
                            color_palette,
                        )

    with performance_tab:
        st.subheader("Latency and Latency Jitter")
        _latency_subplot(performance_plot_df, bar_grouping, color_palette)

    with matrix_tab:
        st.subheader("Faceted Heatmap: Metrics by Video Category and Method")
        st.caption(
            "Each method has its own facet. Rows are video categories and columns are metrics; "
            "colours are normalized independently per metric while annotations retain original values."
        )
        _render_faceted_ik_heatmap(
            ik_summary_df,
            selected_categories,
            selected_methods,
            selected_metrics,
            color_palette,
        )

        st.subheader("Grouped Dot Plots: Method Comparison by Metric and Video Category")
        show_error_bars = st.checkbox(
            "Show 95% error bars when multiple observations are available",
            value=False,
            key="general_matrix_error_bars",
        )
        _render_grouped_ik_dot_plots(
            ik_summary_df,
            selected_categories,
            selected_methods,
            selected_metrics,
            color_palette,
            show_error_bars,
        )

    with data_tab:
        st.subheader("Aggregated IK Data")
        if ik_plot_df.empty:
            st.info("No IK data for the selected categories.")
        else:
            st.dataframe(
                ik_plot_df.sort_values(["category_group", "method"]),
                use_container_width=True,
                hide_index=True,
            )

        st.subheader("Aggregated Performance Data")
        if performance_plot_df.empty:
            st.info("No performance data for the selected categories.")
        else:
            st.dataframe(
                performance_plot_df.sort_values(["category_group", "method"]),
                use_container_width=True,
                hide_index=True,
            )


def _performance_metric_chart(frame_df, metric_name):
    plot_df = frame_df.dropna(subset=[metric_name]).copy()
    if plot_df.empty:
        st.info(f"No values available for {metric_name}.")
        return

    chart = (
        alt.Chart(plot_df)
        .mark_line(point=True)
        .encode(
            x=alt.X("frame_id:Q", title="Frame"),
            y=alt.Y(
                f"{metric_name}:Q",
                title=f"{PERFORMANCE_METRIC_LABELS.get(metric_name, metric_name)} ({_metric_direction(metric_name)})",
            ),
            color=alt.Color("method:N", title="Method"),
            tooltip=[
                "video_id:N",
                "frame_id:Q",
                "method:N",
                alt.Tooltip(f"{metric_name}:Q", format=".6f"),
            ],
        )
        .properties(height=360)
    )
    st.altair_chart(chart, use_container_width=True)


def _render_performance_results(default_path):
    results_path = Path(
        st.sidebar.text_input("Performance JSON", value=str(default_path))
    )
    if not results_path.exists():
        st.error(f"Performance JSON not found: {results_path}")
        return

    results = _load_results(results_path)
    summary_df = _performance_summary_rows(results)
    frame_df = _performance_frame_rows(results)

    if summary_df.empty or frame_df.empty:
        st.info("No performance rows found in the selected results file.")
        return

    st.sidebar.caption(f"Loaded: {results_path}")
    available_categories = sorted(frame_df["category"].dropna().unique())
    available_videos = sorted(frame_df["video_id"].dropna().unique())
    available_methods = sorted(frame_df["method"].dropna().unique())

    categories = st.sidebar.multiselect(
        "Categories",
        available_categories,
        default=available_categories,
        key="perf_categories",
    )
    videos = st.sidebar.multiselect("Videos", available_videos, default=available_videos)
    methods = st.sidebar.multiselect(
        "Performance methods",
        available_methods,
        default=available_methods,
        key="perf_methods",
        format_func=_method_display_name,
        help="Select or deselect methods to update every performance table and plot.",
    )
    performance_palette = st.sidebar.selectbox(
        "Performance color palette",
        list(COLOR_PALETTES),
        index=list(COLOR_PALETTES).index("Multicolor"),
        key="perf_color_palette",
        help="Choose one base color to display categories as different tonalities.",
    )

    filtered_frames = _filter_performance_rows(frame_df, videos, methods)
    filtered_frames = filtered_frames[filtered_frames["category"].isin(categories)]
    filtered_summary = summary_df[
        summary_df["category"].isin(categories)
        & summary_df["video_id"].isin(videos)
        & summary_df["method"].isin(methods)
    ]

    metric_notes = results.get("metrics", {})
    with st.expander("Performance Metric Notes", expanded=False):
        for metric_name in PERFORMANCE_METRIC_ORDER:
            if metric_name in metric_notes:
                st.markdown(f"**{metric_name}**: {metric_notes[metric_name]}")
        for note in results.get("notes", []):
            st.caption(note)

    summary_tab, plots_tab, frames_tab, errors_tab = st.tabs(
        ["Summary", "Frame Plots", "Frame Details", "Errors"]
    )

    with summary_tab:
        st.subheader("Per-video Performance Summary")
        st.dataframe(
            filtered_summary.sort_values(["video_id", "method"]),
            use_container_width=True,
            hide_index=True,
        )

        st.subheader("Performance Metrics by Method")
        performance_plot_df = _aggregate_performance_for_general(
            filtered_summary,
            categories,
            methods,
            group_categories=False,
        )
        available_metrics = [
            metric_name
            for metric_name in PERFORMANCE_METRIC_ORDER
            if metric_name in performance_plot_df.columns
            and performance_plot_df[metric_name].notna().any()
        ]
        if performance_plot_df.empty or not available_metrics:
            st.info("No performance values available for the selected categories.")
        else:
            for row_start in range(0, len(available_metrics), 2):
                columns = st.columns(2)
                for column, metric_name in zip(
                    columns,
                    available_metrics[row_start:row_start + 2],
                ):
                    with column:
                        st.markdown(
                            f"**{PERFORMANCE_METRIC_LABELS.get(metric_name, metric_name)}**"
                        )
                        _general_metric_bar_chart(
                            performance_plot_df,
                            metric_name,
                            grouping="Method",
                            palette_name=performance_palette,
                        )

        dataset_summary = results.get("summary", {})
        if dataset_summary:
            st.subheader("Dataset Summary")
            rows = []
            for method_name, summary in dataset_summary.items():
                row = {"method": method_name}
                row.update(summary)
                rows.append(row)
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    with plots_tab:
        st.subheader("Frame-level Performance Curves")
        selected_video = st.selectbox("Video", videos or available_videos, key="perf_plot_video")
        selected_metric = st.selectbox(
            "Metric",
            PERFORMANCE_METRIC_ORDER,
            index=0,
            key="perf_plot_metric",
            format_func=lambda metric_name: PERFORMANCE_METRIC_LABELS.get(metric_name, metric_name),
        )
        plot_rows = filtered_frames[filtered_frames["video_id"] == selected_video]
        _performance_metric_chart(plot_rows, selected_metric)

    with frames_tab:
        st.subheader("Frame Details")
        columns = ["video_id", "frame_id", "method", *PERFORMANCE_METRIC_ORDER, "error"]
        st.dataframe(
            filtered_frames[columns].sort_values(["video_id", "frame_id", "method"]),
            use_container_width=True,
            hide_index=True,
        )

    with errors_tab:
        st.subheader("Method Errors")
        error_rows = filtered_frames[filtered_frames["error"].notna()]
        if error_rows.empty:
            st.success("No method errors in the current selection.")
        else:
            st.dataframe(
                error_rows[["video_id", "frame_id", "method", "error"]],
                use_container_width=True,
                hide_index=True,
            )


def main():
    st.set_page_config(page_title="Evaluation Results", layout="wide")
    st.title("Evaluation Results")

    result_type = st.sidebar.radio("Result type", ["IK Metrics", "Performance Metrics", "General Plots"])
    if result_type == "IK Metrics":
        _render_ik_results(DEFAULT_RESULTS_PATH)
    elif result_type == "Performance Metrics":
        _render_performance_results(DEFAULT_PERFORMANCE_RESULTS_PATH)
    else:
        _render_general_plots(DEFAULT_RESULTS_PATH, DEFAULT_PERFORMANCE_RESULTS_PATH)


if __name__ == "__main__":
    main()
