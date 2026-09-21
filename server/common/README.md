# Shared Server Runtime

`server/common` contains the pose-estimation, hand-tracking, transport,
visualization, configuration, shared kinematics, and evaluation code used by
all three server methods.

## 1. Architecture

The shared real-time path is:

1. OpenCV captures a webcam frame.
2. YOLOv8 detects the person.
3. MetrAbs estimates the `smpl+head_30` 3D body pose.
4. A background MediaPipe worker detects hand landmarks.
5. MediaPipe landmarks are anchored to the MetrAbs wrist positions.
6. The pose is mapped to the shared 26-point transport layout.
7. The selected method optionally adds Pepper angles and speeds.
8. The JSON payload is sent through the WebSocket server.

Only one person is retained by the maintained runtime.

## 2. Main modules

| Module | Responsibility |
|---|---|
| `settings.py` | Paths, MetrAbs parameters, joint maps, and visualization flags |
| `pose_mapping.py` | Model loading, inference, hand anchoring, and transport mapping |
| `hand_keypoints.py` | MediaPipe Tasks hand detection and background worker |
| `hand_orientation.py` | Palm axes and discrete orientation labels |
| `websocket_client.py` | Server connection, startup request, and latest-only sending |
| `angle_calculator.py` | Human and Pepper angle calculation from pose payloads |
| `perf_metrics.py` | Runtime send and latency measurements |
| `visualization.py` | Shared two-panel pose visualizer |
| `visualizations.py` | Visualization worker and diagnostic panels |
| `kinematics/` | Shared transforms, inverse/forward kinematics, and workspace fitting |
| `metrabs_pytorch/` | Local PyTorch implementation of MetrAbs |
| `evaluation/` | Dataset annotation and shared method-evaluation tools |

## 3. Runtime assets

All asset paths are resolved from `settings.py`; they do not depend on the
shell's working directory.

```text
server/common/
  calibration_images/
    metrabs_camera_params.npz
    camera_matrix.npy
    dist_coeffs.npy
  hand_detection/
    hand_landmarker.task
  models/
    yolov8m.pt
    metrabs_eff2l_384px_800k_28ds_pytorch/
      config.yaml
      ckpt.pt
      joint_info.npz
      joint_transform_matrix.npy
      skeleton_infos.pkl
```

The checkpoint is intentionally excluded from Git.

## 4. MetrAbs configuration

Important defaults in `settings.py`:

```python
SKELETON = "smpl+head_30"
DETECTOR_THRESHOLD = 0.5
DETECTOR_NMS_IOU = 0.7
MAX_DETECTIONS = 1
NUM_AUG = 1
ANTIALIAS_FACTOR = 1
INTERNAL_BATCH_SIZE = 128
AVERAGE_AUG = True
SUPPRESS_IMPLAUSIBLE_POSES = True
DETECTOR_FLIP_AUG = False
```

The model loader requires CUDA and moves the MetrAbs model to the GPU.

## 5. Pose and hand mapping

MetrAbs body coordinates are converted from millimetres to metres. MediaPipe
world landmarks are expressed relative to the MediaPipe wrist, scaled into the
MetrAbs units, and anchored to the matching MetrAbs wrist.

The current `float32` transport array has shape `26 x 3`:

| Indices | Content |
|---|---|
| 0-9 | Nose, neck, right shoulder/elbow/wrist, left shoulder/elbow/wrist, torso, pelvis |
| 10-11 | Right and left middle-finger tips |
| 12-15 | Right wrist, thumb, index, and pinky landmarks |
| 16-19 | Left wrist, thumb, index, and pinky landmarks |
| 20-22 | Right palm x/y/z axes |
| 23-25 | Left palm x/y/z axes |

Missing hand data remains zero-filled for that frame.

## 6. WebSocket contract

The server listens on:

```text
ws://127.0.0.1:8080
```

Streaming begins after the active client sends:

```text
keypoints
```

The current message is JSON:

```json
{
  "pose_keypoints": "<base64 float32 bytes>",
  "hand_orientation": {
    "Right": {"primary": "UP", "x_axis": "...", "y_axis": "...", "z_axis": "..."},
    "Left": {"primary": "UP", "x_axis": "...", "y_axis": "...", "z_axis": "..."}
  }
}
```

The proposed and IKPy methods add an `angles` object. The sender and receiver
both use latest-only queues so stale frames are discarded instead of executed
later.

## 7. Visualization

Visualization flags live in `settings.py`:

```python
ENABLE_VISUALIZATION = True
ENABLE_CAMERA_TEXT_OVERLAYS = False
ENABLE_HAND_ORIENTATION_VISUALIZATION = False
ENABLE_BOUNDING_BOX = False
ENABLE_ANGLES_VISUALIZATION = True
```

The camera panel deliberately contains no text labels. Skeleton lines, body
points, and hand points remain visible. The title bar, 3D plot, and separate
joint-angle window are unaffected.

Set `ENABLE_VISUALIZATION = False` for headless performance measurements.

## 8. Performance behavior

- Camera capture and MetrAbs inference run in the server process.
- MediaPipe hand detection uses a background worker.
- GPU-to-CPU pose transfer is deferred by one frame.
- Visualization uses a separate process and a queue of size one.
- WebSocket sending uses a latest-only queue.
- Performance summaries are printed periodically in headless mode.

These choices prioritize low latency over processing every captured frame.

## 9. Evaluation tools

Shared dataset annotation and method-comparison tools are documented in the
[evaluation README](evaluation/README.md). Dashboard usage is documented in
the [dashboards README](../../dashboards/README.md).

## 10. Extending the shared runtime

When changing the payload:

1. update `pose_mapping.py` and the constants in `settings.py`;
2. update the decoder in `client/pose_stream_runtime.py`;
3. preserve backward compatibility where practical;
4. update `server/tests/test_runtime_layout.py`; and
5. document the new layout here.

Avoid changing joint order silently because every method and the Pepper client
share this contract.

## 11. Related documentation

- [Root setup and execution](../../README.md)
- [Client](../../client/README.md)
- [Baseline](../baseline/README.md)
- [Proposed Constrained Method](../proposed/README.md)
- [IKPy](../ikpy/README.md)
- [Tests](../tests/README.md)
- [Dashboards](../../dashboards/README.md)
