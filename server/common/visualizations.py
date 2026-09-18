from queue import Empty, Full
from server.common.visualization import PoseVisualizer, _render_angles_panel
import numpy as np
from server.common import hand_orientation
import cv2
from multiprocessing import Event, Process, Queue
from server.common import settings


def _draw_hand_orientation_overlay(frame, hand_points_2d, hand_points_3d_world):
    """
    Draw hand orientation multilabel information as overlay text on the frame.
    
    Args:
        frame: OpenCV BGR frame to draw on
        hand_points_2d: dict with "Left"/"Right" keys containing 2D pixel keypoints
        hand_points_3d_world: dict with "Left"/"Right" keys containing 3D world keypoints
    """
    if frame is None or hand_points_3d_world is None:
        return
    
    height, width = frame.shape[:2]
    
    # Y position for drawing text (top-left area)
    y_pos = 30
    x_pos = 10
    y_offset = 60
    
    for hand_label in ["Right", "Left"]:
        if hand_label not in hand_points_3d_world:
            continue
        
        keypoints_3d = hand_points_3d_world[hand_label]
        if 0 not in keypoints_3d or 1 not in keypoints_3d or 5 not in keypoints_3d or 17 not in keypoints_3d:
            continue
        
        try:
            wrist = np.array(keypoints_3d[0], dtype=np.float32)
            thumb = np.array(keypoints_3d[1], dtype=np.float32)
            index = np.array(keypoints_3d[5], dtype=np.float32)
            pinky = np.array(keypoints_3d[17], dtype=np.float32)
            
            # Compute palm_z vector with chirality
            is_left = (hand_label == "Left")
            v1 = index - wrist
            v2 = pinky - wrist
            
            if is_left:
                palm_z = np.cross(v2, v1)
            else:
                palm_z = np.cross(v1, v2)
            
            norm_z = np.linalg.norm(palm_z)
            if norm_z > 1e-6:
                palm_z = palm_z / norm_z
            else:
                continue
            
            # Get multilabel classification
            primary_label, x_label, y_label, z_label = hand_orientation.classify_palm_orientation_multilabel(palm_z)
            multilabel_str = u"{0} {1} {2}".format(x_label.upper(), z_label.upper(), y_label.upper())
            
            # Color for text (different for each hand)
            color = (0, 255, 0) if hand_label == "Left" else (255, 0, 0)  # Green for left, Blue for right
            
            # Draw hand label and multilabel classification
            cv2.putText(frame, u"{0} Hand: {1}".format(hand_label, multilabel_str),
                       (x_pos, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            
            # Draw primary label
            cv2.putText(frame, u"Primary: {0}".format(primary_label),
                       (x_pos, y_pos + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            
            y_pos += y_offset  # Move down for next hand
        
        except Exception as e:
            pass  # Skip if error


def _draw_diagnostics_overlay(frame, missing_keypoints, pose_feasibility, pepper_feasibility, 
                              singularity_right, singularity_left):
    """
    Draw diagnostics information (missing keypoints, feasibility, singularities) on the frame.
    
    Args:
        frame: OpenCV BGR frame to draw on
        missing_keypoints: list of keypoint names that are missing
        pose_feasibility: dict with feasibility info for human pose
        pepper_feasibility: dict with feasibility info for pepper pose
        singularity_right: dict with singularity warnings for right arm
        singularity_left: dict with singularity warnings for left arm
    """
    if frame is None:
        return
    
    height, width = frame.shape[:2]
    y_pos = height - 320  # Start from bottom area
    x_pos = 10
    line_height = 20
    
    # Draw missing keypoints
    #if missing_keypoints:
    #    cv2.putText(frame, "MISSING KEYPOINTS:", (x_pos, y_pos), 
    #               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    #    y_pos += line_height
    #    
    #    # Join keypoints with comma and wrap if too long
    #    keypoints_str = ", ".join(missing_keypoints)
    #    if len(keypoints_str) > 80:
    #        # Split into multiple lines
    #        parts = keypoints_str.split(", ")
    #        current_line = ""
    #        for part in parts:
    #            if len(current_line) + len(part) + 2 > 80:
    #                cv2.putText(frame, current_line, (x_pos + 20, y_pos), 
    #                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
    #                y_pos += line_height
    #                current_line = part
    #            else:
    #                current_line += (", " if current_line else "") + part
    #        if current_line:
    #            cv2.putText(frame, current_line, (x_pos + 20, y_pos), 
    #                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
    #            y_pos += line_height
    #    else:
    #        cv2.putText(frame, keypoints_str, (x_pos + 20, y_pos), 
    #                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
    #        y_pos += line_height
    #else:
    #    cv2.putText(frame, "All keypoints detected", (x_pos, y_pos), 
    #               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    #    y_pos += line_height
    
    y_pos += 10  # Add spacing
    
    # Draw feasibility status
    human_feasible = pose_feasibility.get("is_feasible", False) if pose_feasibility else False
    pepper_feasible = pepper_feasibility.get("is_feasible", False) if pepper_feasibility else False
    
    human_color = (0, 255, 0) if human_feasible else (0, 0, 255)
    pepper_color = (0, 255, 0) if pepper_feasible else (0, 0, 255)
    
    cv2.putText(frame, "Human Pose: {}".format("FEASIBLE" if human_feasible else "INFEASIBLE"), 
               (x_pos, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.6, human_color, 2)
    y_pos += line_height
    
    cv2.putText(frame, "Pepper Pose: {}".format("FEASIBLE" if pepper_feasible else "INFEASIBLE"), 
               (x_pos, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.6, pepper_color, 2)
    y_pos += line_height
    
    y_pos += 10  # Add spacing
    
    # Draw singularity warnings
    if singularity_right and singularity_right.get("warnings"):
        cv2.putText(frame, "RIGHT ARM SINGULARITY:", (x_pos, y_pos), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)
        y_pos += line_height
        for warning in singularity_right.get("warnings", [])[:2]:  # Show max 2 warnings
            cv2.putText(frame, "  - " + str(warning)[:60], (x_pos, y_pos), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 1)
            y_pos += line_height
    
    if singularity_left and singularity_left.get("warnings"):
        cv2.putText(frame, "LEFT ARM SINGULARITY:", (x_pos, y_pos), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)
        y_pos += line_height
        for warning in singularity_left.get("warnings", [])[:2]:  # Show max 2 warnings
            cv2.putText(frame, "  - " + str(warning)[:60], (x_pos, y_pos), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 1)
            y_pos += line_height




def _visualizer_worker(
    vis_queue: Queue,
    stop_event: Event,
    joint_edges: np.ndarray,
    suppress_joint_indices: tuple[int, ...],
):
    visualizer = PoseVisualizer(source_name="dl-pose")
    try:
        while not stop_event.is_set():
            try:
                payload = vis_queue.get(timeout=0.1)
            except Empty:
                continue
            if payload is None:
                break

            # Add hand orientation overlay to frame BEFORE visualization
            hand_points_3d_world = payload.get("hand_points_3d_world")
            hand_points_2d = payload.get("extra_points_2d")
            frame_bgr = payload.get("frame_bgr")
            if (
                settings.ENABLE_CAMERA_TEXT_OVERLAYS
                and hand_points_3d_world
                and frame_bgr is not None
            ):
                _draw_hand_orientation_overlay(frame_bgr, hand_points_2d, hand_points_3d_world)
            
            # Add diagnostics overlay (missing keypoints, feasibility, singularities)
            missing_keypoints = payload.get("missing_keypoints", [])
            pose_feasibility = payload.get("pose_feasibility", {})
            pepper_feasibility = payload.get("pepper_feasibility", {})
            singularity_right = payload.get("singularity_right")
            singularity_left = payload.get("singularity_left")
            #if frame_bgr is not None:
            #    _draw_diagnostics_overlay(frame_bgr, missing_keypoints, pose_feasibility, 
            #                             pepper_feasibility, singularity_right, singularity_left)
            
            vis_result = visualizer.update(
                frame_bgr=payload["frame_bgr"],
                poses2d=payload["poses2d"],
                poses3d=payload["poses3d"],
                joint_edges=joint_edges,
                extra_points_2d=payload.get("extra_points_2d"),
                extra_points_3d=payload.get("extra_points_3d"),
                suppress_joint_indices=suppress_joint_indices,
                torso_index=1,
                units_to_meters=0.001,
                source_name="dl-pose",
                draw_bounding_box=settings.ENABLE_BOUNDING_BOX,
                draw_camera_text=settings.ENABLE_CAMERA_TEXT_OVERLAYS,
                inference_fps=payload.get("inference_fps"),
            )
            
            # Display angles in a separate window if available and enabled
            if settings.ENABLE_ANGLES_VISUALIZATION:
                human_angles = payload.get("human_angles")
                pose_feasibility = payload.get("pose_feasibility")
                singularity_right = payload.get("singularity_right")
                singularity_left = payload.get("singularity_left")
                if human_angles and isinstance(human_angles, dict) and len(human_angles) > 0:
                    angles_panel = _render_angles_panel(
                        human_angles, 
                        width=900, 
                        height=1000, 
                        pose_feasibility=pose_feasibility,
                        singularity_right=singularity_right,
                        singularity_left=singularity_left,
                        show_angle_values=settings.SHOW_ANGLE_VALUES, 
                        missing_keypoints=missing_keypoints,
                    )
                    cv2.imshow("Joint Angles", angles_panel)
            
            # Render hand palm vectors in separate window (if enabled)
            if hand_points_3d_world and settings.ENABLE_HAND_ORIENTATION_VISUALIZATION:
                _visualize_hand_palm_vectors(hand_points_3d_world)
            
            if vis_result.key == ord("q"):
                stop_event.set()
                break
    finally:
        visualizer.close()
        cv2.destroyAllWindows()



def _visualize_hand_palm_vectors(hand_points_3d_world):
    """
    Visualize hand palm vectors (specifically palm_z) in a separate matplotlib window.
    Similar to client-side _draw_orientation_subplot but showing only palm_z.
    
    Args:
        hand_points_3d_world: dict with "Left"/"Right" keys containing keypoint dicts
    """
    global _hand_palm_fig, _hand_palm_axes
    
    try:
        import matplotlib.pyplot as plt
        from matplotlib.figure import Figure
        
        # Create figure only once
        if _hand_palm_fig is None:
            _hand_palm_fig = plt.figure(figsize=(12, 5))
            _hand_palm_fig.suptitle(u'Hand Palm Vectors - Server View', fontsize=12, fontweight='bold')
        else:
            # Clear previous axes
            for ax in _hand_palm_fig.get_axes():
                ax.clear()
        
        # Create left and right subplots
        ax_right = plt.subplot(1, 2, 1)
        ax_left = plt.subplot(1, 2, 2)
        
        # Draw right hand
        if "Right" in hand_points_3d_world:
            _draw_palm_z_subplot(ax_right, hand_points_3d_world["Right"], u"Right Hand - Palm Z", is_left_hand=False)
        else:
            ax_right.text(0.5, 0.5, u'No data', ha='center', va='center', 
                         fontsize=12, transform=ax_right.transAxes)
            ax_right.set_xlim(-1.2, 1.2)
            ax_right.set_ylim(-1.2, 1.2)
            ax_right.set_aspect('equal')
        
        # Draw left hand
        if "Left" in hand_points_3d_world:
            _draw_palm_z_subplot(ax_left, hand_points_3d_world["Left"], u"Left Hand - Palm Z", is_left_hand=True)
        else:
            ax_left.text(0.5, 0.5, u'No data', ha='center', va='center', 
                        fontsize=12, transform=ax_left.transAxes)
            ax_left.set_xlim(-1.2, 1.2)
            ax_left.set_ylim(-1.2, 1.2)
            ax_left.set_aspect('equal')
        
        plt.tight_layout()
        plt.draw()
        plt.pause(0.001)
        
    except Exception as e:
        pass  # Matplotlib might not be available or display issues



# Global matplotlib figure for hand palm vectors visualization
_hand_palm_fig = None
_hand_palm_axes = None

def _draw_palm_z_subplot(ax, keypoints_dict, title, is_left_hand=False):
    """
    Draw only the palm_z vector on a subplot (similar to client visualization).
    
    Args:
        ax: matplotlib axes
        keypoints_dict: dict with indices 0, 1, 5, 17 for wrist, thumb, index, pinky
        title: subplot title
        is_left_hand: bool, whether this is a left hand (affects chirality of cross product)
    """
    try:
        ax.clear()
        ax.set_xlim(-1.2, 1.2)
        ax.set_ylim(-1.2, 1.2)
        ax.set_aspect('equal')
        ax.set_title(title, fontweight='bold')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.grid(True, alpha=0.3)
        
        # Check if we have all required keypoints
        if 0 not in keypoints_dict or 1 not in keypoints_dict or 5 not in keypoints_dict or 17 not in keypoints_dict:
            ax.text(0.5, 0.5, u'Missing keypoints', ha='center', va='center', 
                   fontsize=12, color='red', transform=ax.transAxes)
            return
        
        wrist = np.array(keypoints_dict[0], dtype=np.float32)
        thumb = np.array(keypoints_dict[1], dtype=np.float32)
        index = np.array(keypoints_dict[5], dtype=np.float32)
        pinky = np.array(keypoints_dict[17], dtype=np.float32)
        
        # Compute palm_z vector (normal to palm) with hand chirality consideration
        v1 = index - wrist
        v2 = pinky - wrist
        
        # Use correct cross product order based on hand chirality (same as _compute_hand_palm_vectors)
        if is_left_hand:
            palm_z = np.cross(v2, v1)  # Reversed order for left hand
        else:
            palm_z = np.cross(v1, v2)  # Normal order for right hand
        
        palm_z_norm = np.linalg.norm(palm_z)
        if palm_z_norm > 1e-6:
            palm_z = palm_z / palm_z_norm
        else:
            ax.text(0.5, 0.5, u'Degenerate palm', ha='center', va='center', 
                   fontsize=12, color='red', transform=ax.transAxes)
            return
        
        # Draw origin
        ax.plot(0, 0, 'ko', markersize=8, label=u'Palm origin')
        
        # Draw palm_z vector (blue)
        ax.arrow(0, 0, palm_z[0], palm_z[1], head_width=0.1, head_length=0.1, 
                fc='blue', ec='blue', linewidth=3, label=u'Z-axis (palm_z)')
        ax.text(palm_z[0] * 1.2, palm_z[1] * 1.2, u'Z', fontsize=12, 
               color='blue', fontweight='bold')
        
        # Compute multilabel classification
        primary_label, x_label, y_label, z_label = hand_orientation.classify_palm_orientation_multilabel(palm_z)
        multilabel_str = u"{0} {1} {2}".format(x_label.upper(), z_label.upper(), y_label.upper())
        
        # Display keypoints, multilabel, and vector info
        keypoint_text = (
            u"Multilabel: {0}\nPrimary: {1}\n\n"
            u"Keypoints (XYZ):\n"
            u"Wrist:  [{2: .4f}, {3: .4f}, {4: .4f}]\n"
            u"Thumb:  [{5: .4f}, {6: .4f}, {7: .4f}]\n"
            u"Index:  [{8: .4f}, {9: .4f}, {10: .4f}]\n"
            u"Pinky:  [{11: .4f}, {12: .4f}, {13: .4f}]\n"
            u"\n"
            u"Palm Z Vector:\n"
            u"[{14:.4f}, {15:.4f}, {16:.4f}]"
        ).format(
            multilabel_str, primary_label,
            wrist[0], wrist[1], wrist[2],
            thumb[0], thumb[1], thumb[2],
            index[0], index[1], index[2],
            pinky[0], pinky[1], pinky[2],
            palm_z[0], palm_z[1], palm_z[2]
        )
        ax.text(0.02, 0.98, keypoint_text, transform=ax.transAxes, fontsize=8,
               verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9),
               family='monospace')
        
        ax.legend(loc='lower right', fontsize=9)
        
    except Exception as e:
        ax.text(0.5, 0.5, u'Error: {0}'.format(str(e)[:30]), ha='center', va='center', 
               fontsize=10, color='red', transform=ax.transAxes)



def draw_forearm_labels(
    frame,
    right_forearm_label,
    left_forearm_label,
    right_upper_arm_label=None,
    left_upper_arm_label=None,
    right_wrist_yaw=None,
    left_wrist_yaw=None,
):
    """
    Draw discrete upper-arm and forearm direction labels onto the frame.
    """

    y_pos = 250
    line_gap = 25
    color = (0, 255, 0)

    entries = [
        ("Right upper arm", right_upper_arm_label),
        ("Right forearm", right_forearm_label),
        ("Right wrist yaw", right_wrist_yaw),
        ("Left upper arm", left_upper_arm_label),
        ("Left forearm", left_forearm_label),
        ("Left wrist yaw", left_wrist_yaw),
    ]

    for title, label in entries:
        if label is None:
            label_text = "n/a"
        else:
            label_text = str(label)

        #cv2.putText(
        #    frame,
        #    f"{title}: {label_text}",
        #    (10, y_pos),
        #    cv2.FONT_HERSHEY_SIMPLEX,
        #    0.6,
        #    color,
        #    2,
        #)
        y_pos += line_gap

    return frame
