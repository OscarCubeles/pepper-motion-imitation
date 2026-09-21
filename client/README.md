# Pepper Client

The client receives pose data from any of the three server methods and sends
joint commands to Pepper through NAOqi. The maintained entry point is
`client/run_imitation_client.py`.

## 1. Responsibilities

The client:

1. connects to the pose server over WebSocket;
2. sends the `keypoints` startup request;
3. decodes the latest pose payload;
4. uses server-computed angles when they are available;
5. otherwise computes analytical inverse kinematics from the pose;
6. shapes and dispatches commands to Pepper at a fixed rate; and
7. restores the configured robot safety state during cleanup.

The imitation client does not create an `ALTextToSpeech` or `ALAudioDevice`
proxy and does not send audio to Pepper.

## 2. Environment setup

Pepper's NAOqi Python SDK requires 32-bit Python 2.7.

```powershell
conda create -n pepper
conda activate pepper
conda config --env --set subdir win-32
conda install python=2.7
pip install numpy websocket-client keyboard
```

Verify the interpreter:

```powershell
python --version
python -c "import platform; print(platform.architecture())"
```

Expected values are Python `2.7.x` and `32bit`.

### 2.1 NAOqi SDK

Install the Windows 32-bit NAOqi Python SDK 2.5.5. One supported approach is
to add its `lib` directory to a `conda.pth` file inside the environment's
`Lib/site-packages` directory.

Example SDK path:

```text
C:\pynaoqi-python2.7-2.5.5.5-win32-vs2013\lib
```

Verify the SDK:

```powershell
python -c "from naoqi import qi; print(qi.__version__)"
```

Expected version: `2.5.5.5`.

## 3. Configuration

Client configuration is centralized in `client/pepper_config.py`.

### 3.1 Robot endpoint

Set `PEPPER_MODE`:

```python
PEPPER_MODE = "sim"     # Choregraphe/local simulator at 127.0.0.1
# PEPPER_MODE = "real"  # Physical Pepper at 192.168.0.102 by default
```

The default NAOqi port is `9559`.

### 3.2 Pose server

Default connection:

```python
SERVER_HOST = "localhost"
PORT = 8080
REQUEST_MESSAGE = "keypoints"
```

The server currently binds to `127.0.0.1`, so the client normally runs on the
same computer.

### 3.3 Command rate and smoothing

The runtime keeps only the newest pose and newest target command. It does not
queue stale motion frames.

Current configuration includes:

- backend-dependent command rate (`15 Hz` for the MetrAbs profile);
- a final `HipRoll` command limit of -15° to +15° for every method;
- per-joint deadband;
- per-tick delta limiting;
- adaptive exponential smoothing; and
- chain-specific Pepper speed fractions.

The corresponding values and environment-variable overrides are documented in
`client/pepper_config.py`.

## 4. Run the client

Start one server first, then run:

```powershell
conda activate pepper
python client/run_imitation_client.py
```

Press `q` to stop.

The same client is used for:

- `server.baseline.run_motion_imitation_server`;
- `server.proposed.run_motion_imitation_server`; and
- `server.ikpy.run_motion_imitation_server`.

## 5. Payload handling

The current server sends a JSON object containing a base64-encoded `float32`
pose under `pose_keypoints` and hand-orientation metadata under
`hand_orientation`.

The decoder accepts four layouts for backward compatibility:

| Joint count | Content |
|---:|---|
| 10 | Body joints only |
| 12 | Body joints and two fingertips |
| 20 | Body, fingertips, and hand-orientation landmarks |
| 26 | Current layout, including palm-axis vectors |

Proposed and IKPy payloads may also include an `angles` object with Pepper
angles and speeds. The baseline payload omits it, so the client uses its local
analytical kinematics path.

## 6. Main files

| File or folder | Purpose |
|---|---|
| `run_imitation_client.py` | Canonical, audio-free entry point |
| `pose_stream_runtime.py` | WebSocket receive, decode, buffering, and command loop |
| `server_angle_payload.py` | Validation and conversion of server angle payloads |
| `joint_limits.py` | Final shared joint limits, including HipRoll ±15° |
| `pepper_config.py` | Robot, server, timing, and smoothing configuration |
| `kinematics_baseline/` | Analytical transforms, fitting, IK, and Pepper commands |
| `client_visualization.py` | Client-side visualization helpers |
| `imitaiton_client*.py` | Legacy compatibility entry points; also audio-free |
| `demonstration_client.py` | Separate exercise-demonstration workflow |

The demonstration workflow is not one of the three imitation methods and has
its own external assets and audio behavior.

## 7. Tests

Run client tests in the Python 2.7 environment:

```powershell
python -m unittest discover -s client/tests -p "test_*.py"
```

See [the central testing guide](../server/tests/README.md) for the complete test
inventory.

## 8. Troubleshooting

### NAOqi cannot be imported

Confirm that the environment is 32-bit and that the SDK `lib` directory is in
`Lib/site-packages/conda.pth`.

### The client cannot connect

Start exactly one server, confirm it reports port `8080`, and check
`SERVER_HOST` in `pepper_config.py`.

### Pepper does not move

Check `PEPPER_MODE`, the robot IP, NAOqi connectivity, and
`ENABLE_ROBOT_IMITATION` in `pepper_config.py`.

### Motion is delayed

Check network latency and server inference FPS. Avoid increasing queue sizes;
the latest-only buffering is intentional.

## 9. Related documentation

- [Root setup and execution](../README.md)
- [Shared server and payload](../server/common/README.md)
- [Baseline method](../server/baseline/README.md)
- [Proposed Constrained Method](../server/proposed/README.md)
- [IKPy method](../server/ikpy/README.md)
