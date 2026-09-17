# Pepper Motion Imitation

Real-time human motion imitation for the SoftBank Pepper robot. The system transfers a person's upper-body motion to Pepper so that the robot reproduces the same movement. A Python server estimates the person's 3D pose from a webcam using MetrAbs and MediaPipe hand landmarks, then sends the resulting pose information to a Python 2.7 NAOqi client that commands Pepper.

The repository contains three server methods:

- **Baseline**: implements the human-to-Pepper motion imitation approach from Stoeva et al. (2021), *Analytical Solution of Pepper’s Inverse Kinematics for a Pose Matching Imitation System*. The detected human pose is streamed to the client, where Pepper joint targets are computed using the original analytical inverse kinematics formulation.

- **Proposed**: extends the baseline approach with the improvements introduced in this thesis. Pepper joint angles are computed on the server while incorporating wrist orientation estimation and singularity-handling constraints to improve motion quality and reduce abrupt or unnatural joint behavior.

- **IKPy**: computes Pepper joint angles using a generic inverse kinematics formulation implemented with IKPy. This method is included as a numerical IK baseline for comparison with both the original and proposed motion imitation approaches.

## Repository layout

```text
pepper-motion-imitation/
├── client/                           # Pepper/NAOqi client (Python 2.7)
├── dashboards/
│   ├── annotation-dashboard/         # Streamlit annotation interface
│   ├── streamlit-results-dashboard/  # Streamlit evaluation-results interface
│   └── results/                      # Evaluation results JSON files
├── dataset/                          # Recorded and annotated motion dataset
└── server/
    ├── common/                       # Shared code among different IK methods
    ├── baseline/                     # Baseline server and evaluation tools
    ├── proposed/                     # Proposed constrained method
    ├── ikpy/                         # IKPy method and workspace tools
    └── pepper_ik_resources/          # Pepper URDF and IK chain definitions
```

Run the commands in this README from the repository root unless a section says otherwise.


## Server setup

### Requirements

- Windows with a webcam
- Python 3 (Python 3.11 is a practical choice for the pinned packages)
- An NVIDIA GPU and a CUDA-compatible PyTorch installation for the real-time
  MetrAbs servers

Create and activate a dedicated environment, then install the pinned server
dependencies:

```powershell
python -m pip install -r server/requirements.txt
```

`server/requirements.txt` does not currently include every optional/runtime
package. Install PyTorch and torchvision for the CUDA version available on the
machine, then ensure these imports are also available:

```powershell
python -m pip install mediapipe ikpy ipywidgets
```

The Streamlit dashboards need additional packages:

```powershell
python -m pip install streamlit pandas altair
```

GPU-utilization reporting in the performance evaluator is optional. Install
`pynvml` if that measurement is needed.

### Required runtime assets

The server resolves assets relative to `server/common/settings.py`; it does not
depend on the current working directory.

Check that these files exist:

```text
server/common/models/metrabs_eff2l_384px_800k_28ds_pytorch/ckpt.pt
server/common/models/metrabs_eff2l_384px_800k_28ds_pytorch/config.yaml
server/common/models/metrabs_eff2l_384px_800k_28ds_pytorch/joint_info.npz
server/common/models/metrabs_eff2l_384px_800k_28ds_pytorch/joint_transform_matrix.npy
server/common/models/metrabs_eff2l_384px_800k_28ds_pytorch/skeleton_infos.pkl
server/common/models/yolov8m.pt
server/common/hand_detection/hand_landmarker.task
server/common/calibration_images/metrabs_camera_params.npz
```

The large MetrAbs checkpoint is intentionally excluded from Git and must be
provided locally.

The IKPy method additionally uses:

```text
server/ikpy/metrabs_workspace.json
server/pepper_ik_resources/pepper.urdf
server/pepper_ik_resources/pepper_left_arm.json
server/pepper_ik_resources/pepper_right_arm.json
```

### Camera calibration

Calibration files are read from `server/common/calibration_images/`. To create
new calibration data, run:

```powershell
python -m server.baseline.caliberate
```

The calibration program uses a 9 x 6 inner-corner checkerboard with 25 mm
squares. Capture at least ten views.

## Run a server

Start exactly one server:

```powershell
# Baseline method
python -m server.baseline.run_motion_imitation_server

# Proposed constrained method
python -m server.proposed.run_motion_imitation_server

# IKPy method
python -m server.ikpy.run_motion_imitation_server
```

The servers listen on `ws://127.0.0.1:8080`. They wait for a client to connect
and send `keypoints` before streaming. With visualization enabled in
`server/common/settings.py`, press `q` in the visualization window to stop;
otherwise use `Ctrl+C`.

The IKPy workspace can be generated or inspected with:

```powershell
python -m server.ikpy.run_ikpy_no_client
```

## Pose pipeline and payload

The current server pipeline is:

1. OpenCV captures a webcam frame.
2. MetrAbs detects one person and estimates a 3D `smpl+head_30` pose.
3. MediaPipe detects hand landmarks in a background worker.
4. Hand landmarks are anchored to the corresponding MetrAbs wrist.
5. The data is converted to metres and sent as JSON over WebSocket.

The `pose_keypoints` field is a base64-encoded `float32` array with shape
`26 x 3`:

- indices 0-9: nose, neck, shoulders, elbows, wrists, torso, and pelvis
- indices 10-11: right and left middle-finger tips
- indices 12-19: wrist, thumb, index, and pinky landmarks for both hands
- indices 20-25: right and left palm-axis vectors

The JSON also contains discrete hand-orientation labels. The proposed and IKPy
servers add server-computed human/Pepper angles and joint speeds; the baseline
server does not require an `angles` field.

Important runtime values are centralized in
`server/common/settings.py`. Current defaults include one detection,
`detector_threshold = 0.5`, one test-time augmentation, and visualization
enabled.

## Pepper client

The Pepper client is separate from the Python 3 server environment because the
NAOqi SDK requires 32-bit Python 2.7.

### Client requirements

- Python 2.7, 32-bit
- NAOqi Python SDK 2.5.5 for Windows, 32-bit
- `numpy`, `websocket-client`, and `keyboard`

Example environment setup:

```powershell
conda create -n pepper
conda activate pepper
conda config --env --set subdir win-32
conda install python=2.7
pip install numpy websocket-client keyboard
```

Make the NAOqi SDK importable in this environment according to its installation
instructions.

Select the robot endpoint in `client/pepper_config.py`:

```python
PEPPER_MODE = "sim"     # 127.0.0.1
# PEPPER_MODE = "real"  # 192.168.0.102 by default
```

After starting one of the servers, run:

```powershell
conda activate pepper
python client/run_imitation_client.py
```

The client connects to `localhost:8080`, sends `keypoints`, accepts the current
26-joint payload as well as older 10-, 12-, and 20-joint payloads, and drops
stale frames rather than building a queue. Press `q` to stop.

Because the server binds to `127.0.0.1`, the client must run on the same
computer unless the bind address and client host are changed in the code.

## Dashboards

### Evaluation results

Run:

```powershell
streamlit run dashboards/streamlit-results-dashboard/evaluation_results_app.py
```

The app reads these files by default:

```text
dashboards/results/ik_method_metrics.json
dashboards/results/performance_metrics.json
```

It provides IK metrics, performance metrics, general plots, per-frame views,
method comparisons, and error tables.

### Dataset annotation

Run:

```powershell
streamlit run dashboards/annotation-dashboard/dataset_annotation_app.py
```

The app reads categories below `dataset/`. For a selected video it loads
`annotations_filled.json` when available, otherwise `annotations.json`, and
saves edits to `annotations_filled.json`.

## Evaluation tools

Evaluation scripts live in `server/baseline/evaluation/`. The dataset layout is
expected to follow:

```text
dataset/<category>/video_XXX/
  video.mp4
  frames/
  annotations.json
  annotations_filled.json
```

Treat `annotations_filled.json` as manual ground truth and back it up before
bulk changes.

Common commands include:

```powershell
# Add MetrAbs coordinates to annotation files
python -m server.baseline.evaluation.annotate_metrabs_dataset dataset/pepper_singularity_motions

# Compute annotation joint angles
python -m server.baseline.evaluation.compute_annotation_joint_angles dataset/pepper_singularity_motions

# Evaluate all IK methods and refresh the dashboard input
python -m server.baseline.evaluation.evaluate_ik_methods dataset --output dashboards/results/ik_method_metrics.json

# Benchmark method computation and refresh the dashboard input
python -m server.baseline.evaluation.evaluate_performance_metrics dataset --output dashboards/results/performance_metrics.json
```

The evaluators still have legacy default output paths under
`04-evaluation/results/`. Pass `--output` as shown above when the generated data
should appear in the current dashboard.

The IK evaluator compares:

- `ground_truth`
- `current_solution_raw`
- `current_solution_constrained`
- `ikpy`
- `original_solution`

Its result file contains the metrics `EEAh`, `EEAr`, `SOAx`, `HJL`, `WOM`,
`HJAr`, `TSE`, and `SYN`. The performance evaluator measures the offline method
computation, not webcam capture, MetrAbs inference, network transfer, or robot
actuation.

Use each script's `--help` option for its complete arguments.

## Tests

Run the server tests from the repository root in the Python 3 environment:

```powershell
python -m unittest discover -s server/tests -p "test_*.py"
python -m unittest discover -s server/baseline/evaluation/tests -p "test_*.py"
```

The client tests belong to the Python 2.7 client environment:

```powershell
python -m unittest discover -s client/tests -p "test_*.py"
```

## Troubleshooting

### MetrAbs checkpoint not found

Place `ckpt.pt` in:

```text
server/common/models/metrabs_eff2l_384px_800k_28ds_pytorch/
```

### Camera cannot be opened

Close other applications using the webcam and check `CAMERA_INDEX` in
`server/common/settings.py`.

### Calibration files not found

Run `python -m server.baseline.caliberate` or restore the calibration files in
`server/common/calibration_images/`.

### MediaPipe model not found

Check that `server/common/hand_detection/hand_landmarker.task` exists.

### IKPy workspace not found

Run `python -m server.ikpy.run_ikpy_no_client` and create
`server/ikpy/metrabs_workspace.json` before starting the IKPy streaming server.

### No stream starts

Start a server first, then the client. The server deliberately waits until the
client sends the `keypoints` request.

### Streamlit command not found

Activate the Python 3 server environment and run:

```powershell
python -m pip install streamlit pandas altair
python -m streamlit run dashboards/streamlit-results-dashboard/evaluation_results_app.py
```
