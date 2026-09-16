# Pepper Motion Imitation

Real-time human pose estimation and motion imitation for the SoftBank Pepper
robot. The server estimates a 3D pose with MetrAbs and MediaPipe, then streams
pose data and optional Pepper joint angles to a Python 2.7 NAOqi client.

## Repository layout

- `server/common/`: shared pose estimation, transport, kinematics, models, and assets.
- `server/baseline/`: baseline server entry point and retained offline/legacy tools.
- `server/proposed/`: proposed constrained-motion method.
- `server/ikpy/`: IKPy server, local IKPy runner, controllers, and notebooks.
- `client/`: the shared Pepper client and robot-side kinematics.

## Server commands

Run commands from the repository root in the server environment:

```powershell
python -m server.baseline.run_motion_imitation_server
python -m server.proposed.run_motion_imitation_server
python -m server.ikpy.run_motion_imitation_server
python -m server.ikpy.run_ikpy_no_client
```

Install server dependencies from `server/requirements.txt`. The MetrAbs
checkpoint is intentionally untracked and must exist at:

```text
server/common/models/metrabs_eff2l_384px_800k_28ds_pytorch/ckpt.pt
```

## Pepper client

The canonical client entry point works with all three streaming servers:

```powershell
conda activate pepper
python client/run_imitation_client.py
```

The client uses server-computed angles when supplied by the proposed or IKPy
server and falls back to pose-based client IK for baseline payloads.
