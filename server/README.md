# Client

## Overview

This directory contains the client-side half of the Exercise Motivation System.

The client runs on Python 2.7, receives streamed 3D pose data from a server, converts that pose into Pepper-compatible joint targets, and sends those commands to Pepper through the NAOqi SDK.

The maintained client entry points are:

- `imitation_client.py`
- `demonstration_client.py`
- `pose_stream_runtime.py`
- `kinematics_new/`
- `exercises/`

Latency measurement guide:

- `../LATENCY_README.md`

This README is the shared client overview. It explains the general working principle, installation, and the difference between imitation mode and demonstration mode. It does not go into detailed mode behavior.

## Working Principle

The client performs the same high-level steps regardless of which server backend is used:

1. connect to the pose server over WebSocket
2. request streamed keypoints
3. decode the incoming compact pose payload
4. convert the streamed keypoints into Pepper's coordinate system
5. apply workspace fitting and inverse kinematics
6. send joint targets to Pepper through NAOqi

In the maintained implementation:

- `pose_stream_runtime.py` owns the shared WebSocket runtime
- `kinematics_new/scaling_spherical.py` converts streamed pose frames into Pepper joint targets
- `kinematics_new/pepper_commands.py` sends those commands to Pepper

## Client Architecture

### Shared runtime

- `pose_stream_runtime.py`
  - shared stream receiver and decode path
  - handles WebSocket connection, stop behavior, and cleanup

### Kinematics and robot output

- `kinematics_new/`
  - coordinate transforms
  - workspace fitting
  - inverse kinematics
  - Pepper command output

### Exercise and scoring support

- `exercises/`
  - exercise templates
  - human-motion logging
  - template logging
  - DTW-based scoring utilities

## Modes

The maintained client supports two top-level modes.

### Imitation mode

Imitation mode is the live motion-following mode.

Pepper receives the streamed pose continuously and mirrors or imitates the detected person in real time.

Entry point:

- `client/imitation_client.py`

### Demonstration mode

Demonstration mode is the exercise-guided mode.

Pepper performs a predefined exercise template while the client records the human response and runs an assessment step afterward.

Current maintained demonstration feedback behavior:

- Start cues are prerecorded WAV files loaded from `ai_feedback_server/audio/`:
  - `Hallo.wav`
  - `Folgen.wav`
- After DTW scoring, feedback audio is requested from `FEEDBACK_SERVER_URL` and streamed to Pepper from RAM.
- End cue is a prerecorded WAV:
  - `Spaß.wav`
- No Pepper `ALTextToSpeech` fallback is used in this maintained path.
- Demonstration logging plots are saved to disk; interactive plot windows are disabled by default.
- Demonstration capture waits for the first pose payload before timing the capture window.
- If no payload arrives within startup timeout, demonstration capture aborts with stop reason `first_payload_timeout`.

Entry point:

- `client/demonstration_client.py`

## Installation

### Requirements

- Windows
- Miniconda or Anaconda
- Python 2.7, 32-bit
- NAOqi Python 2.7 SDK 2.5.5

The client must stay on Python 2.7 because Pepper communication depends on the NAOqi Python 2 SDK.

### 1. Create the conda environment

```powershell
conda create -n pepper
conda activate pepper
conda config --env --set subdir win-32
conda install python=2.7
```

Verify that the environment is really 32-bit:

```powershell
python -c "import platform; print(platform.architecture())"
python --version
```

Expected architecture output:

```text
('32bit', 'WindowsPE')
```

### 2. Install the NAOqi SDK

1. Download the NAOqi Python 2.7 SDK 2.5.5.
2. Extract it to a stable local path such as:
   - `C:\pynaoqi-python2.7-2.5.5.5-win32-vs2013\`
3. Create `conda.pth` in:
   - `C:\Users\<username>\miniconda3\envs\pepper\Lib\site-packages`
4. Add the SDK `lib` path to that file:
   - `C:\pynaoqi-python2.7-2.5.5.5-win32-vs2013\lib`

Verify the SDK import:

```powershell
python -c "from naoqi import qi; print(qi.__version__)"
```

Expected output:

```text
2.5.5.5
```

### 3. Install Python dependencies

```powershell
pip install numpy websocket-client keyboard
```

Optional, but useful for exercise logging plots:

```powershell
pip install matplotlib
```

## Configuration Notes

### Pose server host and port

The maintained client entry points use:

- host: `localhost`
- port: `8080`

These values are defined in:

- `client/imitation_client.py`
- `client/demonstration_client.py`
- `client/pose_stream_runtime.py`

Update them only if the server runs elsewhere.

### Pepper IP

Pepper endpoint config is centralized in:

- `client/pepper_config.py`

Change only `PEPPER_MODE`:

- `"sim"` uses `127.0.0.1`
- `"real"` uses `192.168.0.102`

All maintained client entry points and helper scripts now read from this shared config.

### Imitation command pacing and smoothing

Imitation mode now uses fixed-rate robot command dispatch plus lightweight command shaping:

- decode thread keeps consuming pose frames as fast as they arrive
- command thread sends at a fixed backend-dependent cadence
- each command tick uses the latest buffered targets (no command backlog)
- command shaping is applied before `ALMotion.setAngles` in `kinematics_new/pepper_commands.py`
  - per-joint deadband
  - per-tick delta clamp
  - adaptive EMA (`alpha_slow` to `alpha_fast`)

Configuration is centralized in `client/pepper_config.py`.

Current parameter examples from maintained defaults:

```python
POSE_BACKEND_PROFILE = "metrabs"
POSE_COMMAND_RATE_HZ_BY_BACKEND = {
    "metrabs": 15.0,
    "zed": 25.0,
}

POSE_SMOOTHING_DEADBAND_RAD = 0.01
POSE_SMOOTHING_MAX_DELTA_RAD_PER_TICK = 0.06
POSE_SMOOTHING_ALPHA_SLOW = 0.40
POSE_SMOOTHING_ALPHA_FAST = 0.80
POSE_SMOOTHING_FAST_DELTA_RAD = 0.06

CHAIN_SPEED_FRACTIONS = {
    "torso": 0.25,
    "head": 0.25,
    "others": 0.25,
}
```

### Feedback server URL and timeout

Demonstration-mode feedback audio calls read from environment via `client/pepper_config.py`:

- `FEEDBACK_SERVER_URL` (default: `http://localhost:8091`)
- `FEEDBACK_SERVER_TIMEOUT_SEC` (default: `20.0`)

Example (PowerShell):

```powershell
$env:FEEDBACK_SERVER_URL = "http://localhost:8091"
$env:FEEDBACK_SERVER_TIMEOUT_SEC = "20"
```

Playback path in demonstration mode:

- Feedback WAV bytes are streamed directly from RAM to Pepper using `ALAudioDevice.sendRemoteBufferToOutput`.
- No feedback file is written to Pepper storage.
- No temporary feedback WAV file is written on the laptop client.
- Prerecorded cue WAVs must exist in `ai_feedback_server/audio/`.
- For robust startup on cold pose servers, demonstration mode waits for first payload (default timeout: `12.0s`).

Quick manual audio-stream test:

```powershell
python client\test_play_laptop_audio_stream.py
```

## Run Commands

### Imitation mode

```powershell
conda activate pepper
python client\imitation_client.py
```

### Demonstration mode

```powershell
conda activate pepper
python client\demonstration_client.py
```

## Runtime Behavior

The shared client runtime in `pose_stream_runtime.py` currently:

- connects to `ws://<host>:<port>/PepperCommands`
- sends the startup message `keypoints`
- decodes either a `10 x 3` or `12 x 3` `float32` pose payload
- drops stale frames instead of building up a queue
- runs fixed-rate command dispatch in imitation mode:
  - `15 FPS` for `metrabs` backend profile
  - `25 FPS` for `zed` backend profile
- uses latest-target buffering between decode and command threads
- applies command shaping (deadband, delta clamp, adaptive EMA) before robot output
- supports manual stop with the `q` key
- supports auto-stop timers either from connection start or from first received payload
- supports optional startup timeout when no first payload arrives
- restores Pepper safety settings on cleanup in normal imitation mode

## Related Files

- `kinematics_new/README.md`
  - kinematics and Pepper-control internals

- `../server/README.md`
  - shared server architecture and transport contract

- `../server/dl-pose/README.md`
  - MetrAbs webcam backend

- `../server/zed_cam/README.md`
  - ZED backend
