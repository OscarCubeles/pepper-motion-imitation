"""
EMA Smoothing visualization for hand keypoints.
Tracks and plots raw vs smoothed keypoint values across X, Y, Z axes for both hands.
"""

import numpy as np
import matplotlib.pyplot as plt
from collections import deque
from matplotlib.animation import FuncAnimation


class EMAVisualizationTracker:
    """
    Tracks raw and smoothed keypoint values over time for visualization.
    Stores only the last N frames to avoid memory issues.
    """
    def __init__(self, max_history=300):
        """
        Args:
            max_history: maximum number of frames to store in history
        """
        self.max_history = max_history
        self.history = {
            "LEFT": {
                "raw": deque(maxlen=max_history),
                "smoothed": deque(maxlen=max_history),
            },
            "RIGHT": {
                "raw": deque(maxlen=max_history),
                "smoothed": deque(maxlen=max_history),
            }
        }
        self.frame_count = 0
    
    def add_frame(self, hand_label, raw_keypoints, smoothed_keypoints):
        """
        Add a frame of raw and smoothed keypoint data.
        
        Args:
            hand_label: "LEFT" or "RIGHT"
            raw_keypoints: dict with keys ["wrist", "thumb", "index", "pinky"], each is [x, y, z]
            smoothed_keypoints: dict with same structure as raw_keypoints
        """
        self.history[hand_label]["raw"].append(raw_keypoints.copy())
        self.history[hand_label]["smoothed"].append(smoothed_keypoints.copy())
        self.frame_count += 1
    
    def plot_smoothing(self):
        """
        Create a 2x3 subplot visualization showing EMA smoothing effect.
        
        Layout:
        - Rows: LEFT hand, RIGHT hand (2 rows)
        - Columns: X-axis, Y-axis, Z-axis (3 columns)
        
        Each subplot shows raw vs smoothed values for all 4 keypoints.
        """
        fig, axes = plt.subplots(2, 3, figsize=(16, 10))
        fig.suptitle("EMA Smoothing Effect on Hand Keypoints", fontsize=16, fontweight='bold')
        
        axis_names = ['X', 'Y', 'Z']
        axis_indices = [0, 1, 2]
        hand_labels = ['LEFT', 'RIGHT']
        keypoint_names = ['wrist', 'thumb', 'index', 'pinky']
        colors_raw = ['red', 'blue', 'green', 'orange']
        colors_smoothed = ['darkred', 'darkblue', 'darkgreen', 'darkorange']
        
        for hand_idx, hand_label in enumerate(hand_labels):
            for axis_idx, axis_name in enumerate(axis_names):
                ax = axes[hand_idx, axis_idx]
                
                raw_data = self.history[hand_label]["raw"]
                smoothed_data = self.history[hand_label]["smoothed"]
                
                if len(raw_data) == 0:
                    ax.text(0.5, 0.5, "No data", ha='center', va='center', transform=ax.transAxes)
                    ax.set_title(f"{hand_label} - {axis_name}-axis")
                    continue
                
                # Plot each keypoint
                for kp_idx, kp_name in enumerate(keypoint_names):
                    raw_values = [frame[kp_name][axis_indices[axis_idx]] for frame in raw_data]
                    smoothed_values = [frame[kp_name][axis_indices[axis_idx]] for frame in smoothed_data]
                    
                    # Raw data (lighter, more transparent)
                    ax.plot(raw_values, color=colors_raw[kp_idx], linestyle='--', 
                           alpha=0.5, linewidth=1, label=f"{kp_name} (raw)")
                    
                    # Smoothed data (darker, solid)
                    ax.plot(smoothed_values, color=colors_smoothed[kp_idx], linestyle='-', 
                           alpha=0.8, linewidth=2, label=f"{kp_name} (smoothed)")
                
                ax.set_title(f"{hand_label} Hand - {axis_name}-axis (normalized)", fontweight='bold')
                ax.set_xlabel("Frame")
                ax.set_ylabel(f"{axis_name} Position")
                ax.grid(True, alpha=0.3)
                ax.legend(fontsize=8, loc='best')
                # Z-axis has smaller range (depth), X and Y cover full frame
                if axis_idx == 2:  # Z-axis
                    ax.set_ylim([-0.15, 0.15])
                else:  # X and Y axes
                    ax.set_ylim([0, 1])
        
        plt.tight_layout()
        return fig


def create_live_visualization():
    """
    Create a live updating figure for EMA smoothing visualization.
    Returns figure and axes for updating during the main loop.
    """
    fig, axes = plt.subplots(3, 2, figsize=(14, 12))
    fig.suptitle("EMA Smoothing - Live Visualization", fontsize=16, fontweight='bold')
    
    # Set up axis labels
    axis_names = ['X-axis', 'Y-axis', 'Z-axis']
    for i, ax_name in enumerate(axis_names):
        axes[i, 0].set_ylabel(ax_name)
        axes[i, 0].set_title(f"LEFT Hand - {ax_name}")
        axes[i, 1].set_title(f"RIGHT Hand - {ax_name}")
        axes[i, 0].set_ylim([0, 1])
        axes[i, 1].set_ylim([0, 1])
        axes[i, 0].grid(True, alpha=0.3)
        axes[i, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig, axes
