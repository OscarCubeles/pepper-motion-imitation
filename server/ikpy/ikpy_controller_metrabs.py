# Some necessary imports
import numpy as np
import os
import json

from ikpy.chain import Chain
from ikpy.utils import plot

import matplotlib.pyplot as plt

from server.common import settings
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.widgets import Slider
from matplotlib.gridspec import GridSpec

from ipywidgets import interact, FloatSlider
import matplotlib.pyplot as plt


def load_chains(script_dir=None):
    # First, let's import the pepper chains
    resources_dir = settings.PEPPER_RESOURCES_DIR
    pepper_left_arm_chain = Chain.from_json_file(str(resources_dir / "pepper_left_arm.json"))
    pepper_right_arm_chain = Chain.from_json_file(str(resources_dir / "pepper_right_arm.json"))
    pepper_legs_chain = Chain.from_json_file(str(resources_dir / "pepper_legs.json"))
    pepper_head_chain = Chain.from_json_file(str(resources_dir / "pepper_head.json"))
    return pepper_left_arm_chain, pepper_right_arm_chain, pepper_legs_chain, pepper_head_chain


def compute_workspace_bounds(chain, samples_per_joint=6, verbose=False):
    """Compute reachable workspace bounds for a kinematic chain.
    
    Uses grid-based sampling across joint space to find min/max positions.
    
    Algorithm:
    1. For each joint, collect its angle bounds
    2. Create uniform samples across each joint range
    3. Generate all combinations of samples (grid)
    4. Compute forward kinematics for each combination
    5. Track min/max X, Y, Z coordinates
    6. Return workspace bounding box
    
    Args:
        chain: ikpy Chain object
        samples_per_joint: Number of samples per joint (default 6)
                          Higher = more accurate but slower
        verbose: Print progress and results
        
    Returns:
        dict: {
            'x_bounds': (min_x, max_x),
            'y_bounds': (min_y, max_y),
            'z_bounds': (min_z, max_z),
            'center': (center_x, center_y, center_z),
            'ranges': (range_x, range_y, range_z),
            'samples_tested': total_configurations
        }
    """
    # Extract joint bounds (filter out fixed joints)
    joint_bounds = []
    joint_names = []
    
    for idx, link in enumerate(chain.links):
        if link.bounds is not None and \
           link.bounds[0] != float('-inf') and \
           link.bounds[1] != float('inf'):
            joint_bounds.append(link.bounds)
            joint_names.append(link.name)
    
    if verbose:
        print(f"\nComputing workspace for chain with {len(joint_bounds)} controllable joints:")
        for name, bounds in zip(joint_names, joint_bounds):
            print(f"  {name:20s}: [{bounds[0]:7.4f}, {bounds[1]:7.4f}] rad")
    
    # Create sample points for each joint
    joint_samples = []
    for min_angle, max_angle in joint_bounds:
        samples = np.linspace(min_angle, max_angle, samples_per_joint)
        joint_samples.append(samples)
    
    # Initialize bounds
    bounds = {
        'x_min': float('inf'),  'x_max': float('-inf'),
        'y_min': float('inf'),  'y_max': float('-inf'),
        'z_min': float('inf'),  'z_max': float('-inf'),
    }
    
    # Generate all combinations and compute forward kinematics
    total_samples = np.prod([len(s) for s in joint_samples])
    
    if verbose:
        print(f"\nTesting {total_samples} configurations...")
    
    for config_idx, config in enumerate(np.ndindex(*[len(s) for s in joint_samples])):
        # Build joint angles: start with all zeros
        q = np.zeros(len(chain))
        
        # Fill in the controllable joint angles
        for j_idx, sample_idx in enumerate(config):
            # Find actual link index for this controllable joint
            link_idx = 0
            controllable_count = 0
            for idx, link in enumerate(chain.links):
                if link.bounds is not None and \
                   link.bounds[0] != float('-inf') and \
                   link.bounds[1] != float('inf'):
                    if controllable_count == j_idx:
                        link_idx = idx
                        break
                    controllable_count += 1
            
            q[link_idx] = joint_samples[j_idx][sample_idx]
        
        # Compute forward kinematics
        try:
            fk = chain.forward_kinematics(q)
            x, y, z = fk[:3, 3]
            
            # Update bounds
            bounds['x_min'] = min(bounds['x_min'], x)
            bounds['x_max'] = max(bounds['x_max'], x)
            bounds['y_min'] = min(bounds['y_min'], y)
            bounds['y_max'] = max(bounds['y_max'], y)
            bounds['z_min'] = min(bounds['z_min'], z)
            bounds['z_max'] = max(bounds['z_max'], z)
        except Exception as e:
            if verbose and config_idx < 5:
                print(f"  Config {config}: FK failed - {e}")
    
    # Calculate derived metrics
    x_range = bounds['x_max'] - bounds['x_min']
    y_range = bounds['y_max'] - bounds['y_min']
    z_range = bounds['z_max'] - bounds['z_min']
    
    center = (
        (bounds['x_min'] + bounds['x_max']) / 2,
        (bounds['y_min'] + bounds['y_max']) / 2,
        (bounds['z_min'] + bounds['z_max']) / 2,
    )
    
    result = {
        'x_bounds': (bounds['x_min'], bounds['x_max']),
        'y_bounds': (bounds['y_min'], bounds['y_max']),
        'z_bounds': (bounds['z_min'], bounds['z_max']),
        'center': center,
        'ranges': (x_range, y_range, z_range),
        'samples_tested': total_samples,
    }
    
    if verbose:
        print(f"\n{'='*60}")
        print(f"WORKSPACE BOUNDS (from {total_samples} samples):")
        print(f"{'='*60}")
        print(f"X: [{bounds['x_min']:7.4f}, {bounds['x_max']:7.4f}]  range: {x_range:.4f} m")
        print(f"Y: [{bounds['y_min']:7.4f}, {bounds['y_max']:7.4f}]  range: {y_range:.4f} m")
        print(f"Z: [{bounds['z_min']:7.4f}, {bounds['z_max']:7.4f}]  range: {z_range:.4f} m")
        print(f"\nCenter: ({center[0]:.4f}, {center[1]:.4f}, {center[2]:.4f})")
        print(f"{'='*60}\n")
    
    return result


def save_workspace_bounds(workspace, arm, script_dir):
    """Save computed workspace bounds to a JSON file for quick loading.
    
    Args:
        workspace: Dict returned by compute_workspace_bounds()
        arm: 'left' or 'right'
        script_dir: Directory to save the file in
    """
    filename = os.path.join(script_dir, f"workspace_bounds_{arm}_arm.json")
    
    # Convert tuple bounds to lists for JSON serialization
    data = {
        'x_bounds': list(workspace['x_bounds']),
        'y_bounds': list(workspace['y_bounds']),
        'z_bounds': list(workspace['z_bounds']),
        'center': list(workspace['center']),
        'ranges': list(workspace['ranges']),
        'samples_tested': int(workspace['samples_tested']),
    }
    
    with open(filename, 'w') as f:
        json.dump(data, f, indent=2)
    
    print(f"Workspace bounds saved to: {filename}")
    return filename


def load_workspace_bounds(arm, script_dir):
    """Load previously computed workspace bounds from file.
    
    Args:
        arm: 'left' or 'right'
        script_dir: Directory to load the file from
        
    Returns:
        dict: Workspace bounds or None if file doesn't exist
    """
    filename = os.path.join(script_dir, f"workspace_bounds_{arm}_arm.json")
    
    if not os.path.exists(filename):
        return None
    
    try:
        with open(filename, 'r') as f:
            data = json.load(f)
        
        # Convert lists back to tuples
        workspace = {
            'x_bounds': tuple(data['x_bounds']),
            'y_bounds': tuple(data['y_bounds']),
            'z_bounds': tuple(data['z_bounds']),
            'center': tuple(data['center']),
            'ranges': tuple(data['ranges']),
            'samples_tested': data['samples_tested'],
        }
        
        print(f"Workspace bounds loaded from: {filename}")
        return workspace
    except Exception as e:
        print(f"Failed to load workspace bounds: {e}")
        return None


def plot_chains(left_arm_chain, right_arm_chain, legs_chain, head_chain):
    fig, ax = plot.init_3d_figure()
    left_arm_chain.plot([0] * (len(left_arm_chain)), ax)
    right_arm_chain.plot([0] * (len(right_arm_chain)), ax)
    legs_chain.plot([0] * (len(legs_chain)), ax)
    head_chain.plot([0] * (len(head_chain)), ax)
    ax.legend()
    plt.show()


def interactive_ik(left_arm_chain, right_arm_chain, legs_chain, head_chain):
    """Interactive IK solver with position sliders."""
    fig, ax = plot.init_3d_figure()
    
    # Get initial position from forward kinematics
    x, y, z = right_arm_chain.forward_kinematics([0] * (len(left_arm_chain)))[:3, 3]
    size = 0.5
    
    xlim = ax.get_xlim3d()
    ylim = ax.get_ylim3d()
    zlim = ax.get_zlim3d()
    
    def goto(xl, yl, zl, xr, yr, zr):
        ax.clear()
        ax.set_xlim3d(xlim)
        ax.set_ylim3d(ylim)
        ax.set_zlim3d(zlim)
        
        # Right arm IK
        frame_target = np.eye(4)
        frame_target[:3, 3] = [xr, yr, zr]
        ik_right = right_arm_chain.inverse_kinematics_frame(frame_target, optimizer="scalar")
        print("Right arm joint angles (radians):", ik_right)
        right_arm_chain.plot(ik_right, ax, target=[xr, yr, zr])
        
        # Left arm IK
        frame_target[:3, 3] = [xl, yl, zl]
        ik_left = left_arm_chain.inverse_kinematics_frame(frame_target, optimizer="scalar")
        print("Left arm joint angles (radians):", ik_left)
        left_arm_chain.plot(ik_left, ax, target=[xl, yl, zl])
        
        # Plot legs and head
        legs_chain.plot([0] * (len(legs_chain)), ax)
        head_chain.plot([0] * (len(head_chain)), ax)
        ax.legend()
    
    interact(goto,
             xl=FloatSlider(min=x-size, max=x+size, value=x, step=0.01),
             yl=FloatSlider(min=y-size, max=y+size, value=y, step=0.01),
             zl=FloatSlider(min=z-size, max=z+size, value=z, step=0.01),
             xr=FloatSlider(min=x-size, max=x+size, value=x, step=0.01),
             yr=FloatSlider(min=y-size, max=y+size, value=y, step=0.01),
             zr=FloatSlider(min=z-size, max=z+size, value=z, step=0.01))


def interactive_ik_with_sliders(left_arm_chain, right_arm_chain, legs_chain, head_chain, arm="right", script_dir=None, use_workspace_bounds=False):
    """Interactive IK solver with real-time XYZ sliders and joint angle visualization.
    
    Uses matplotlib sliders to adjust target position and displays computed joint angles
    on the plot in real-time.
    
    Args:
        left_arm_chain: ikpy Chain for left arm
        right_arm_chain: ikpy Chain for right arm
        legs_chain: ikpy Chain for legs
        head_chain: ikpy Chain for head
        arm: "right" or "left" - which arm to control
        script_dir: Directory for saving/loading workspace bounds (optional)
        use_workspace_bounds: If True, use saved workspace bounds. 
                             If False, use ±0.5m around initial position (default: True)
    """
    # Get script directory if not provided
    if script_dir is None:
        script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Select chain based on arm
    chain = right_arm_chain if arm == "right" else left_arm_chain
    other_chain = left_arm_chain if arm == "right" else right_arm_chain
    
    # Get initial position from forward kinematics
    q0 = [0] * len(chain)
    initial_pos = chain.forward_kinematics(q0)[:3, 3]
    x_init, y_init, z_init = initial_pos
    
    # Determine slider ranges
    if use_workspace_bounds:
        # Try to load workspace bounds from file, otherwise compute and save
        print(f"\nLoading workspace bounds for {arm} arm...")
        workspace = load_workspace_bounds(arm, script_dir)
        
        if workspace is None:
            print(f"Computing workspace bounds (will be saved for future use)...")
            workspace = compute_workspace_bounds(chain, samples_per_joint=6, verbose=True)
            save_workspace_bounds(workspace, arm, script_dir)
        
        x_init, y_init, z_init = workspace['center']
        x_bounds, y_bounds, z_bounds = workspace['x_bounds'], workspace['y_bounds'], workspace['z_bounds']
        print(f"Using computed workspace bounds")
    else:
        # Use ±0.5m around initial position (old method)
        size = 0.5
        x_bounds = (x_init - size, x_init + size)
        y_bounds = (y_init - size, y_init + size)
        z_bounds = (z_init - size, z_init + size)
        print(f"Using ±{size}m around initial position: ({x_init:.4f}, {y_init:.4f}, {z_init:.4f})")
    
    # Create figure with GridSpec for proper layout
    fig = plt.figure(figsize=(16, 10))
    gs = GridSpec(4, 2, figure=fig, hspace=0.4, wspace=0.3)
    
    # Main 3D plot (left side, spans all rows)
    ax_3d = fig.add_subplot(gs[:, 0], projection='3d')
    
    # Sliders (right side, top)
    ax_x = fig.add_subplot(gs[0, 1])
    ax_y = fig.add_subplot(gs[1, 1])
    ax_z = fig.add_subplot(gs[2, 1])
    
    # Info text area (right side, bottom)
    ax_info = fig.add_subplot(gs[3, 1])
    ax_info.axis('off')
    
    # Create sliders using actual workspace bounds
    slider_x = Slider(ax_x, 'X (m)', x_bounds[0], x_bounds[1], valinit=x_init, color='red', alpha=0.7)
    slider_y = Slider(ax_y, 'Y (m)', y_bounds[0], y_bounds[1], valinit=y_init, color='green', alpha=0.7)
    slider_z = Slider(ax_z, 'Z (m)', z_bounds[0], z_bounds[1], valinit=z_init, color='blue', alpha=0.7)
    
    # Set fixed axis limits (centered around origin)
    ax_3d.set_xlim([-1.0, 1.0])
    ax_3d.set_ylim([-1.0, 1.0])
    ax_3d.set_zlim([-0.5, 1.5])
    ax_3d.set_xlabel('X (m)', fontsize=10)
    ax_3d.set_ylabel('Y (m)', fontsize=10)
    ax_3d.set_zlabel('Z (m)', fontsize=10)
    ax_3d.set_title(f'IK Solver - {arm.upper()} Arm', fontsize=12, fontweight='bold')
    
    # Text object for joint angles display
    text_obj = ax_info.text(0.05, 0.95, '', transform=ax_info.transAxes, fontsize=9,
                            verticalalignment='top', horizontalalignment='left', family='monospace',
                            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.85, 
                                    edgecolor='black', linewidth=1.5, pad=0.8))
    
    def update(val=None):
        """Update IK solution and plot based on slider values."""
        x, y, z = slider_x.val, slider_y.val, slider_z.val
        
        # Compute IK
        frame_target = np.eye(4)
        frame_target[:3, 3] = [x, y, z]
        
        try:
            ik_solution = chain.inverse_kinematics_frame(frame_target, optimizer="scalar", max_iter=1000)
        except Exception as e:
            print(f"IK failed: {e}")
            ik_solution = q0
        
        # Get controllable joints
        angles, names, indices = get_controllable_joints(chain, ik_solution)
        
        # Update the 3D plot (clear and redraw, but keep axis limits fixed)
        ax_3d.clear()
        
        # Plot the target arm
        if arm == "right":
            right_arm_chain.plot(ik_solution, ax_3d, target=[x, y, z])
            left_arm_chain.plot([0] * len(left_arm_chain), ax_3d)
        else:
            left_arm_chain.plot(ik_solution, ax_3d, target=[x, y, z])
            right_arm_chain.plot([0] * len(right_arm_chain), ax_3d)
        
        # Plot legs and head
        legs_chain.plot([0] * len(legs_chain), ax_3d)
        head_chain.plot([0] * len(head_chain), ax_3d)
        
        # Reapply fixed axis limits
        ax_3d.set_xlim([-1.0, 1.0])
        ax_3d.set_ylim([-1.0, 1.0])
        ax_3d.set_zlim([-0.5, 1.5])
        ax_3d.set_xlabel('X (m)', fontsize=10)
        ax_3d.set_ylabel('Y (m)', fontsize=10)
        ax_3d.set_zlabel('Z (m)', fontsize=10)
        ax_3d.set_title(f'IK Solver - {arm.upper()} Arm', fontsize=12, fontweight='bold')
        ax_3d.legend(loc='upper right', fontsize=8)
        
        # Update joint angles text display
        joint_info = f"TARGET POSITION:\n"
        joint_info += f"  X = {x:.4f} m\n"
        joint_info += f"  Y = {y:.4f} m\n"
        joint_info += f"  Z = {z:.4f} m\n"
        joint_info += f"\n{'='*45}\n"
        joint_info += f"{'JOINT ANGLES':<45}\n"
        joint_info += f"{'='*45}\n"
        
        for name, angle in zip(names, angles):
            joint_info += f"{name:25s}: {angle:8.4f} rad\n"
            joint_info += f"{'':25s}  ({np.degrees(angle):7.2f}°)\n"
        
        text_obj.set_text(joint_info)
        
        fig.canvas.draw_idle()
    
    # Connect sliders to update function
    slider_x.on_changed(update)
    slider_y.on_changed(update)
    slider_z.on_changed(update)
    
    # Initial plot
    update()
    
    plt.show()
    
def test_ik(left_arm_chain,
            right_arm_chain,
            legs_chain,
            head_chain,
            delta,
            arm="left",
            output_file="ik_test.png"):

    # Select arm
    chain = left_arm_chain if arm == "left" else right_arm_chain

    q0 = [0] * len(chain)

    current_pos = chain.forward_kinematics(q0)[:3, 3]

    print("Current end effector:")
    print(current_pos)

    target = current_pos + np.array(delta)

    print("Delta:")
    print(delta)

    print("Target:")
    print(target)

    frame_target = np.eye(4)
    frame_target[:3, 3] = target

    ik_solution = chain.inverse_kinematics_frame(
        frame_target,
        optimizer="scalar"
    )

    print("IK solution:")
    print(ik_solution)

    # Get controllable joints only
    angles, names, indices = get_controllable_joints(chain, ik_solution)

    print("\nJoint changes:")
    joint_info_text = "JOINT ANGLES:\n" + "="*35 + "\n"
    for name, angle in zip(names, angles):
        print(f"{name:20s}: {angle: .4f} rad ({np.degrees(angle): .2f} deg)")
        joint_info_text += f"{name:20s}: {angle: .4f} rad\n"
        joint_info_text += f"{'':20s}  ({np.degrees(angle): .2f}°)\n"

    reached = chain.forward_kinematics(ik_solution)[:3, 3]

    print("Reached:")
    print(reached)

    fig, ax = plot.init_3d_figure()

    # Plot moved arm
    if arm == "left":
        left_arm_chain.plot(ik_solution, ax, target=target)
        right_arm_chain.plot([0] * len(right_arm_chain), ax)
    else:
        right_arm_chain.plot(ik_solution, ax, target=target)
        left_arm_chain.plot([0] * len(left_arm_chain), ax)

    # Plot remaining chains
    legs_chain.plot([0] * len(legs_chain), ax)
    head_chain.plot([0] * len(head_chain), ax)

    ax.legend()

    # Add joint angles text box to the figure
    fig.text(
        0.02, 0.98,
        joint_info_text,
        transform=fig.transFigure,
        fontsize=8,
        verticalalignment='top',
        horizontalalignment='left',
        family='monospace',
        bbox=dict(
            boxstyle='round',
            facecolor='wheat',
            alpha=0.8,
            edgecolor='black',
            linewidth=1.5,
            pad=0.8
        )
    )

    plt.savefig(output_file, dpi=300, bbox_inches="tight")

    print(f"Saved figure to: {output_file}")

    plt.show()


def get_controllable_joints(chain, ik_result):
    """Extract only controllable joints from IK result.
    
    Filters out fixed joints (no bounds or infinite bounds).
    
    Args:
        chain: ikpy Chain
        ik_result: Full IK result array
        
    Returns:
        tuple: (filtered_angles, filtered_names, filtered_indices)
    """
    controllable_angles = []
    controllable_names = []
    controllable_indices = []
    
    for idx, link in enumerate(chain.links):
        if idx >= len(ik_result):
            continue
        
        # Check if this is a controllable joint (has bounds and not infinity)
        is_controllable = (
            link.bounds is not None and 
            link.bounds[0] != float('-inf') and 
            link.bounds[1] != float('inf')
        )
        
        if is_controllable:
            controllable_angles.append(ik_result[idx])
            controllable_names.append(link.name)
            controllable_indices.append(idx)
    
    return controllable_angles, controllable_names, controllable_indices

def main():
    # Get the directory where this script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Load the pepper chains
    left_arm_chain, right_arm_chain, legs_chain, head_chain = load_chains(script_dir)

    # Choose mode: 'plot', 'ik', or 'sliders'
    mode = 'sliders'  # Change to: 'plot' (static), 'ik' (ipywidgets), or 'sliders' (matplotlib)
    
    if mode == 'plot':
        plot_chains(left_arm_chain, right_arm_chain, legs_chain, head_chain)
    elif mode == 'ik':
        interactive_ik(left_arm_chain, right_arm_chain, legs_chain, head_chain)
        plot_chains(left_arm_chain, right_arm_chain, legs_chain, head_chain)
    elif mode == 'sliders':
        interactive_ik_with_sliders(left_arm_chain, right_arm_chain, legs_chain, head_chain, arm="right")


def main2():
    # Get the directory where this script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Load Pepper chains
    left_arm_chain, right_arm_chain, legs_chain, head_chain = load_chains(script_dir)

    # 1. Show Pepper in zero configuration
    #print("\n=== INITIAL POSE ===")
    #plot_chains(left_arm_chain, right_arm_chain, legs_chain, head_chain)
#
    ## 2. Test IK on left arm with visualization
    #print("\n=== LEFT ARM IK TEST ===")
    #test_ik(
    #    left_arm_chain,
    #    right_arm_chain,
    #    legs_chain,
    #    head_chain,
    #    delta=[0.0, 0.10, 0.0],
    #    arm="left",
    #    output_file="left_arm_move.png"
    #)
#
    ## 3. Test IK on right arm (optional)
    #print("\n=== RIGHT ARM IK TEST ===")
    #test_ik(
    #    left_arm_chain,
    #    right_arm_chain,
    #    legs_chain,
    #    head_chain,
    #    delta=[0.0, -0.10, 0.5],
    #    arm="right",
    #    output_file="right_arm_move.png"
    #)
    
    # 4. Interactive slider mode
    print("\n=== INTERACTIVE SLIDER MODE ===")
    interactive_ik_with_sliders(left_arm_chain, right_arm_chain, legs_chain, head_chain, arm="right")

if __name__ == "__main__":
    main2()
