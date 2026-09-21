# Baseline Method

The baseline method implements the analytical human-to-Pepper imitation path.
The server estimates and streams the human pose; the shared Pepper client
computes the robot joint targets locally with the analytical kinematics code in
`client/kinematics_baseline`.

## 1. Run the method

From the repository root:

```powershell
conda activate media
python -m server.baseline.run_motion_imitation_server
```

In a second terminal:

```powershell
conda activate pepper
python client/run_imitation_client.py
```

The imitation client is audio-free.

## 2. Method flow

1. The shared runtime captures the webcam image.
2. MetrAbs estimates the 3D body pose.
3. MediaPipe adds hand landmarks and orientation information.
4. The baseline server maps and sends the 26-point pose.
5. The client detects that no server `angles` object is present.
6. The client computes Pepper targets with its analytical kinematics path.
7. The client smooths and dispatches the targets to Pepper.

This separation is the main architecture distinction from the proposed and IKPy streaming
methods, which include server-computed angles.

## 3. Entry point

`run_motion_imitation_server.py` is the maintained baseline entry point. It
uses shared components from `server/common` for:

- camera opening and calibration;
- MetrAbs inference;
- MediaPipe hand tracking;
- pose mapping;
- visualization;
- WebSocket transport; and
- performance reporting.

The server listens on `127.0.0.1:8080` and waits for the client's `keypoints`
message.

## 4. Calibration

The calibration utility is located in this folder because it captures the same
webcam used by the pose server:

```powershell
conda activate media
python -m server.baseline.caliberate
```

Controls:

- `Space`: capture a valid checkerboard view;
- `Enter`: calibrate after at least ten captured views; and
- `Esc`: exit.

Output is saved under `server/common/calibration_images/`.

## 5. Benchmarks and retained tools

| File | Purpose |
|---|---|
| `metrabs_simple_benchmark.py` | Measures MetrAbs webcam inference behavior |
| `mediapipe_benchmark.py` | Measures MediaPipe hand detection behavior |
| `demo_multiplePeople.py` | Retained MetrAbs multi-person demonstration |
| `classifier.py` | Retained classification helper |
| `ikpy_utils.py` | Retained compatibility utility |
| `Pose_3D_metrabs_server_mediapipe_hand*.py` | Legacy server variants; not canonical entry points |

Use `run_motion_imitation_server.py` for the maintained baseline pipeline.

## 6. Expected payload

The baseline sends:

- `pose_keypoints`;
- `hand_orientation`; and
- transport timing metadata when applicable.

It does not require a server-computed `angles` object. See the
[shared runtime documentation](../common/README.md) for the 26-point layout.

## 7. Troubleshooting

### Calibration files are missing

Run `python -m server.baseline.caliberate`.

### No person is detected

Check lighting, keep the full upper body visible, and confirm that
`server/common/models/yolov8m.pt` exists.

### The server runs but Pepper does not move

Confirm that the shared client is connected and that its local baseline
kinematics path is enabled.

### Performance is poor

Disable visualization in `server/common/settings.py`, close other GPU-heavy
applications, and keep `NUM_AUG = 1` and `MAX_DETECTIONS = 1` unless measured
experiments justify changing them.

## 8. Related documentation

- [Root setup and execution](../../README.md)
- [Shared runtime](../common/README.md)
- [Client](../../client/README.md)
- [Tests](../tests/README.md)
