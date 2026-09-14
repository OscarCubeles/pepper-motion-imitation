import cv2
import torch
import time
import sys
import settings
import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path
from calibration_utils import load_metrabs_calibration
from metrabs_pytorch.inference import metrabsInference
from matplotlib.patches import Rectangle
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

settings.set_demonstration_client_dir()

from visualization import PoseVisualizer


def plot_results(image, pred, joint_names, joint_edges):
    fig = plt.figure(figsize=(10, 5.2))
    canvas = FigureCanvas(fig)
    image_ax = fig.add_subplot(1, 2, 1)
    image_ax.imshow(image)
    for x, y, w, h, c in pred['boxes'].numpy():
        image_ax.add_patch(Rectangle((x, y), w, h, fill=False))

    pose_ax = fig.add_subplot(1, 2, 2, projection='3d')
    pose_ax.view_init(5, -75)
    pose_ax.set_xlim3d(-1500, 1500)
    pose_ax.set_zlim3d(-1500, 1500)
    pose_ax.set_ylim3d(2000, 5000)
    poses3d = pred['poses3d'].numpy()
    poses3d[..., 1], poses3d[..., 2] = poses3d[..., 2], -poses3d[..., 1]
    for pose3d, pose2d in zip(poses3d, pred['poses2d'].numpy()):
        for i_start, i_end in joint_edges:
            image_ax.plot(*zip(pose2d[i_start], pose2d[i_end]), marker='o', markersize=2)
            pose_ax.plot(*zip(pose3d[i_start], pose3d[i_end]), marker='o', markersize=2)
        image_ax.scatter(*pose2d.T, s=2)
        pose_ax.scatter(*pose3d.T, s=2)
    canvas.draw()  # Draw the canvas using Agg backend
    img = np.frombuffer(canvas.buffer_rgba(), dtype=np.uint8)
    img = img.reshape(canvas.get_width_height()[::-1] + (4,))[:, :, :3]

    plt.imshow(img)
    plt.show()

SCRIPT_DIR = Path(__file__).resolve().parent
model_dir = SCRIPT_DIR / "metrabs_eff2l_384px_800k_28ds_pytorch"
MeterabsInferenceModel = metrabsInference.metrabs_inference(str(model_dir))
model = MeterabsInferenceModel.load_model()
intrinsic_matrix, distortion_coeffs = load_metrabs_calibration()

skeleton = 'smpl+head_30'

joint_names = model.per_skeleton_joint_names[skeleton]
joint_edges = model.per_skeleton_joint_edges[skeleton].cpu().numpy()
visualizer = PoseVisualizer(source_name="dl-pose")

with torch.inference_mode(), torch.device('cuda'):
    image_filepath = SCRIPT_DIR / "calibration_images" / "demo-pose.jpg"
    image = cv2.imread(str(image_filepath))
    image_pt = torch.from_numpy(image).permute(2, 0, 1).cuda()
    print(image_pt.shape, image_pt.device)
    time_start = time.time()
    pred = model.detect_poses(
        image_pt,
        intrinsic_matrix=intrinsic_matrix,
        distortion_coeffs=distortion_coeffs,
        detector_threshold=0.01,
        suppress_implausible_poses=False,
        max_detections=1,
        skeleton=skeleton,
        num_aug=1,
    )

    pred = {k: v.cpu() for k, v in pred.items()}
    print(f"Time taken: {time.time() - time_start:.2f}s")

    # Legacy renderer kept for reference:
    # plot_results(cv2.cvtColor(image,cv2.COLOR_RGB2BGR), pred, joint_names, joint_edges)

    while True:
        vis_result = visualizer.update(
            frame_bgr=image,
            poses2d=pred.get("poses2d"),
            poses3d=pred.get("poses3d"),
            joint_edges=joint_edges,
            boxes=pred.get("boxes"),
            torso_index=1,
            units_to_meters=0.001,
            source_name="dl-pose-demo",
            wait_key_delay=30,
        )
        if vis_result.key == ord('q'):
            break
    visualizer.close()
