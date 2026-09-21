# Pepper Motion Imitation

Real-time upper-body motion imitation for the SoftBank Pepper robot. A Python
3 server estimates a person's pose with MetrAbs and MediaPipe, then a Python
2.7 NAOqi client sends the resulting joint targets to Pepper.

The repository contains three motion-imitation methods:

- **[Baseline](server/baseline/README.md):** analytical pose-matching method
  with client-side inverse kinematics.
- **[Proposed Constrained Method](server/proposed/README.md):** server-side joint angles,
  wrist orientation, and singularity handling.
- **[IKPy](server/ikpy/README.md):** numerical inverse-kinematics method.

## Contents

#### [1. Methods and demo videos](#1-methods-and-demo-videos)

#### [2. Setup](#2-setup)

#### [3. Execute the pipeline](#3-execute-the-pipeline)

#### [4. Detailed documentation](#4-detailed-documentation)

## 1. Methods and demo videos

### 1.1 [Baseline](server/baseline/README.md)

Implements the analytical human-to-Pepper imitation approach. The server sends
the detected pose, and the client computes Pepper joint targets.

- [Baseline documentation](server/baseline/README.md)

<video src="dashboards/results/baseline_demo.mp4" controls width="720">
  Your browser does not support embedded video.
</video>


### 1.2 [Proposed Constrained Method](server/proposed/README.md)

Computes Pepper angles on the server and applies the thesis method's wrist
orientation and singularity-handling logic.

- [Proposed Constrained Method documentation](server/proposed/README.md)

<video src="dashboards/results/proposed_demo.mp4" controls width="720">
  Your browser does not support embedded video.
</video>


### 1.3 [IKPy](server/ikpy/README.md)

Uses IKPy as a numerical inverse-kinematics comparison method.

- [IKPy documentation](server/ikpy/README.md)

<video src="dashboards/results/ikpy_demo.mp4" controls width="720">
  Your browser does not support embedded video.
</video>


## 2. Setup

Run all commands from the repository root.

### 2.1 Server environment

Requirements:

- Windows and a webcam
- Miniconda or Anaconda
- Python 3.11, 64-bit
- NVIDIA GPU with a CUDA-compatible driver

Create the environment:

```powershell
conda create -n media python=3.11
conda activate media
conda config --env --set subdir win-64
```

Install CUDA-enabled PyTorch and the project dependencies:

```powershell
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu130
python -m pip install -r server/requirements.txt
python -m pip install mediapipe ikpy ipywidgets
```

Verify CUDA:

```powershell
python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'No CUDA GPU')"
```

The first value must be `True` for the maintained MetrAbs runtime.

### 2.2 Server assets

The following assets must exist:

```text
server/common/models/metrabs_eff2l_384px_800k_28ds_pytorch/
  config.yaml
  ckpt.pt
  joint_info.npz
  joint_transform_matrix.npy
  skeleton_infos.pkl

server/common/models/yolov8m.pt
server/common/hand_detection/hand_landmarker.task
```

Download and extract the
[MetrAbs Large PyTorch model](https://bit.ly/metrabs_l_pt) into the model
directory shown above. The large `ckpt.pt` file is intentionally excluded from
Git.

The IKPy method also requires:

```text
server/ikpy/metrabs_workspace.json
server/pepper_ik_resources/pepper.urdf
server/pepper_ik_resources/pepper_left_arm.json
server/pepper_ik_resources/pepper_right_arm.json
```

### 2.3 Camera calibration

The server requires camera calibration. Generate it with a 9 x 6 inner-corner
checkerboard with 25 mm squares:

```powershell
conda activate media
python -m server.baseline.caliberate
```

Use `Space` to capture a view and `Enter` after collecting at least ten valid
views. The resulting files are stored in
`server/common/calibration_images/`.

### 2.4 Pepper client environment

The NAOqi client requires a separate 32-bit Python 2.7 environment:

```powershell
conda create -n pepper
conda activate pepper
conda config --env --set subdir win-32
conda install python=2.7
pip install numpy websocket-client keyboard
```

Install the 32-bit NAOqi Python SDK 2.5.5 and make its `lib` directory
importable from the `pepper` environment. Verify it with:

```powershell
python -c "import platform; print(platform.architecture())"
python -c "from naoqi import qi; print(qi.__version__)"
```

Select the simulator or physical robot in `client/pepper_config.py`:

```python
PEPPER_MODE = "sim"     # 127.0.0.1
# PEPPER_MODE = "real"  # 192.168.0.102 by default
```

See the [client documentation](client/README.md) for the detailed SDK and
runtime configuration.

## 3. Execute the pipeline

Use two terminals. Start exactly one server method before starting the client.

### 3.1 Terminal 1: start a server

```powershell
conda activate media

# Choose one command:
python -m server.baseline.run_motion_imitation_server
python -m server.proposed.run_motion_imitation_server
python -m server.ikpy.run_motion_imitation_server
```

### 3.2 Terminal 2: start the shared client

```powershell
conda activate pepper
python client/run_imitation_client.py
```

The server listens on `ws://127.0.0.1:8080`. The client sends `keypoints` to
start streaming. Both processes must run on the same computer unless the bind
address and client host are changed.

Press `q` to stop the visualization/client, or use `Ctrl+C` in a headless
server. All three imitation methods are silent and do not send audio to Pepper.
The shared client also limits every `HipRoll` command to the inclusive range
from -15° to +15°.

## 4. Detailed documentation

- [Client](client/README.md): NAOqi setup, payload handling, command shaping,
  configuration, and troubleshooting.
- [Shared server code](server/common/README.md): MetrAbs, MediaPipe, payload,
  WebSocket, visualization, evaluation, settings, and shared assets.
- [Baseline](server/baseline/README.md): analytical baseline, calibration,
  and retained benchmark tools.
- [Proposed Constrained Method](server/proposed/README.md): server-side angle calculation
  and singularity handling.
- [IKPy](server/ikpy/README.md): IKPy chains, workspace calibration, runtime,
  and notebooks.
- [Tests](server/tests/README.md): available suites, commands, and scope.
- [Dashboards](dashboards/README.md): annotation and evaluation result apps.
- [Server overview](server/README.md): server-package navigation.
