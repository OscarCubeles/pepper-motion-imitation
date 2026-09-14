# DL-Pose: MetrAbs Webcam 3D Pose Server

## Overview

This directory contains the maintained webcam-based 3D human pose pipeline for the Exercise Motivation System.

Shared server architecture, transport behavior, visualization, and performance metrics are documented in [`../README.md`](../README.md).

It uses:

- `MetrAbs` for multi-person absolute 3D body pose estimation
- `YOLOv8` inside the MetrAbs multiperson pipeline for person detection
- `MediaPipe Hands` in the `*_mediapipe.py` variants to augment the body pose with fingertip information

There are two main runtime modes:

- `Pose_3D_metrabs_server.py`
  - body-only streaming
  - outputs a compact `10 x 3` pose for the Pepper client

- `Pose_3D_metrabs_server_mediapipe.py`
  - MetrAbs body pose plus MediaPipe fingertip augmentation
  - outputs a compact `12 x 3` pose for the Pepper client

This README describes the current code, not older scripts referenced elsewhere in the repo.

## How The Pipeline Works

### MetrAbs-only pipeline

1. OpenCV captures an RGB frame from the webcam.
2. Camera calibration is loaded from `calibration_images/`.
3. The frame is passed to `model.detect_poses(...)`.
4. MetrAbs runs person detection and 3D pose estimation.
5. The first detected body is mapped from the chosen skeleton convention to the compact Pepper streaming format.
6. The pose is converted to meters, serialized as `float32`, base64-encoded, and streamed over WebSocket.

### MetrAbs plus MediaPipe pipeline

The MediaPipe variant keeps the same body-pose path, then adds hand landmarks:

1. The webcam frame is submitted to a background MediaPipe hand worker.
2. MetrAbs estimates the 3D body pose.
3. MediaPipe returns wrist and fingertip landmarks for each hand.
4. The MediaPipe hand landmarks are anchored to the MetrAbs wrist positions in 3D.
5. Only fingertip positions are added to the outgoing pose.
6. The final output is streamed as a `12 x 3` pose:
   - the same 10 body joints as the body-only server
   - `RightTip`
   - `LeftTip`

This is used by the Pepper client for hand open/close estimation while keeping the transport format compact.

## Why We Use MetrAbs

MetrAbs gives us:

- absolute 3D pose from a monocular RGB webcam
- a configurable skeleton convention
- built-in person detection plus pose estimation
- test-time augmentation support
- output in metric 3D coordinates

In this project, the maintained server scripts use MetrAbs as the body-pose backbone and then remap the full output skeleton into the smaller Pepper-oriented joint set that the client expects.

## Why We Add MediaPipe

MetrAbs gives a strong body pose, but the Pepper client only needs a small amount of hand detail for hand-state control.

The MediaPipe variants add:

- explicit hand detection
- a wrist-to-fingertip measurement path
- a lightweight way to estimate hand openness without sending the full hand skeleton

The current implementation uses:

- landmark `0`: wrist
- landmark `12`: fingertip used for streaming

The fingertip is anchored to the MetrAbs wrist position before streaming so the hand points stay aligned with the MetrAbs body pose.

## Main Files

### Streaming servers

- `Pose_3D_metrabs_server.py`
  - webcam to WebSocket server
  - MetrAbs only
  - output shape: `10 x 3`

- `Pose_3D_metrabs_server_mediapipe.py`
  - webcam to WebSocket server
  - MetrAbs plus MediaPipe Hands
  - output shape: `12 x 3`

### Local visualization scripts

- `Pose_3D_metrabs.py`
  - local visualization without WebSocket streaming

- `Pose_3D_metrabs_mediapipe.py`
  - local visualization with MediaPipe hand augmentation

### Support files

- `calibration_utils.py`
  - loads the camera intrinsic matrix and distortion coefficients

- `caliberate.py`
  - generates calibration files from checkerboard captures

- `webcam.py`
  - simple webcam test utility

- `metrabs_pytorch/`
  - local PyTorch MetrAbs implementation used by the maintained scripts

## Current Runtime Configuration

The maintained server scripts use these settings:

```python
SKELETON = "kinectv2_25"
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

These values prioritize low latency and stable single-person streaming for Pepper.

## What `model.detect_poses(...)` Does In This Project

The maintained scripts call MetrAbs like this in practice:

```python
pred = model.detect_poses(
    image_pt,
    intrinsic_matrix=intrinsic_matrix,
    distortion_coeffs=distortion_coeffs,
    detector_threshold=DETECTOR_THRESHOLD,
    detector_nms_iou_threshold=DETECTOR_NMS_IOU,
    max_detections=MAX_DETECTIONS,
    skeleton=SKELETON,
    num_aug=NUM_AUG,
    antialias_factor=ANTIALIAS_FACTOR,
    internal_batch_size=INTERNAL_BATCH_SIZE,
    average_aug=AVERAGE_AUG,
    suppress_implausible_poses=SUPPRESS_IMPLAUSIBLE_POSES,
    detector_flip_aug=DETECTOR_FLIP_AUG,
)
```

Conceptually, `detect_poses(...)` does four things:

1. detect people in the image
2. crop the detected person regions
3. estimate 2D and 3D pose for each detected person
4. filter and post-process the results

The output dictionary contains:

- `boxes`
- `poses2d`
- `poses3d`

In this project, the Pepper streaming servers mainly use `poses3d`, and the visualization path may use both `poses2d` and `poses3d`.

## Meaning Of The Active MetrAbs Arguments

This section explains the arguments that matter in the maintained server scripts.

### `image`

- The input RGB image for pose estimation.
- In the maintained scripts, the frame is read from OpenCV and transferred to the GPU before inference.

### `intrinsic_matrix`

- The camera intrinsic matrix.
- This tells MetrAbs how the camera projects 3D points into the image plane.
- In this repo it is loaded from calibration files through `calibration_utils.py`.
- If calibration is missing, the maintained server will fail fast and ask you to run `python caliberate.py`.

Why it matters:

- better 3D scale and geometry
- more stable pose estimates than using an assumed field of view

### `distortion_coeffs`

- The OpenCV lens distortion coefficients.
- These compensate for webcam lens distortion before or during the pose estimation geometry.

Why it matters:

- improves accuracy, especially away from the image center
- reduces calibration mismatch in 3D reconstruction

### `skeleton`

- Selects which joint convention MetrAbs should output.
- The maintained server scripts use:

```python
SKELETON = "kinectv2_25"
```

Why this project uses it:

- it is easier to map to the Pepper client's compact joint layout
- the outgoing `10 x 3` and `12 x 3` formats are built from this convention

### `detector_threshold`

- Confidence threshold for the internal person detector.
- Lower values may detect more people but also increase false positives.
- Higher values reduce weak detections.

Current value:

```python
DETECTOR_THRESHOLD = 0.5
```

Why it is set this way:

- balanced for a single clearly visible person in front of a webcam

### `detector_nms_iou_threshold`

- Non-maximum suppression threshold for the internal detector.
- Controls how aggressively overlapping detections are merged or suppressed.

Current value:

```python
DETECTOR_NMS_IOU = 0.7
```

Why it is set this way:

- it avoids duplicates without being too aggressive around partial overlaps

### `max_detections`

- Maximum number of detected people to keep.

Current value:

```python
MAX_DETECTIONS = 1
```

Why it is set this way:

- Pepper only imitates one person
- limiting to one person reduces compute and ambiguity

### `num_aug`

- Number of test-time augmentations.
- Higher values can improve robustness but cost more latency.

Current value:

```python
NUM_AUG = 1
```

Why it is set this way:

- the maintained server is optimized for real-time streaming, not offline accuracy

### `average_aug`

- If multiple augmentations are used, average the results.

Current value:

```python
AVERAGE_AUG = True
```

In practice here:

- it keeps the output shape simple and predictable
- with `NUM_AUG = 1`, it has no real averaging cost but keeps the interface consistent

### `antialias_factor`

- Supersampling factor for internal crops before prediction.
- Higher values can reduce aliasing but cost more compute.

Current value:

```python
ANTIALIAS_FACTOR = 1
```

Why it is set this way:

- lowest-latency setting

### `internal_batch_size`

- Number of crops processed internally per chunk.
- This affects GPU memory usage and throughput.

Current value:

```python
INTERNAL_BATCH_SIZE = 128
```

Why it matters:

- too low can underuse the GPU
- too high can cause memory pressure

In this project:

- it is tuned for the current real-time single-person setup and available hardware assumptions

### `detector_flip_aug`

- Runs extra detector inference on a horizontally flipped image and aggregates detections.

Current value:

```python
DETECTOR_FLIP_AUG = False
```

Why it is disabled:

- extra cost for limited benefit in the current streaming use case

### `suppress_implausible_poses`

- Filters poses that look physically inconsistent or are likely to be bad predictions.

Current value:

```python
SUPPRESS_IMPLAUSIBLE_POSES = True
```

Why it is enabled:

- reduces unstable or obviously incorrect body outputs before they reach Pepper

## Output Skeleton And Pepper Mapping

Even though MetrAbs can output many skeleton conventions, the streaming server does not send the whole skeleton to the client.

Instead, the server remaps the selected MetrAbs skeleton to a compact Pepper-oriented output.

### Body-only output

The body-only server sends `10` joints in this order:

1. Nose
2. Neck
3. RightShoulder
4. RightElbow
5. RightWrist
6. LeftShoulder
7. LeftElbow
8. LeftWrist
9. Torso
10. SpineBase

### Body plus MediaPipe output

The MediaPipe server sends the same `10` joints plus:

11. RightTip
12. LeftTip

The Pepper client uses these fingertip points for hand open/close estimation.

## MediaPipe Fusion Details

The MediaPipe server variant does not replace MetrAbs body estimation. It augments it.

### What MediaPipe contributes

- hand detection on the latest frame
- wrist and fingertip landmarks
- per-hand `Left` or `Right` labeling when available

### How the fusion works

1. The body pose comes from MetrAbs.
2. MediaPipe detects hand landmarks in image space and world space.
3. The code resolves the MetrAbs left and right wrist joint indices for the selected skeleton.
4. Each MediaPipe hand is anchored to the corresponding MetrAbs wrist in 3D.
5. Only fingertip landmarks are placed into the outgoing transport payload.

### Why the fingertip is anchored

MediaPipe world coordinates and MetrAbs 3D coordinates are not identical coordinate systems or scales by default.

Anchoring the MediaPipe hand points to the MetrAbs wrist lets the project:

- keep MetrAbs as the authoritative body pose
- avoid a full coordinate-system merge
- recover a fingertip direction and distance suitable for Pepper hand control

## Coordinate Units

MetrAbs returns 3D body pose in millimeter-like metric coordinates.

In the maintained streaming servers:

- the remapped body pose is converted to meters before transport
- the client expects meters at the transport boundary

That is why the MetrAbs server multiplies or scales values by `0.001` before streaming.

## MetrAbs-Specific Performance Notes

The maintained MetrAbs servers are tuned for low latency with these backend-specific choices:

- `MAX_DETECTIONS = 1`
- `NUM_AUG = 1`
- `ANTIALIAS_FACTOR = 1`
- deferred GPU-to-CPU transfer handling in the body-only server
- background MediaPipe hand worker in the MediaPipe server

Shared server performance behavior and `perf_metrics.py` usage are documented in [`../README.md`](../README.md).

## Requirements

### Hardware

- Windows machine
- NVIDIA GPU recommended
- webcam

### Python environment

Recommended environment name:

```powershell
media
```

Create and install:

```powershell
conda create -n media python
conda activate media
conda config --env --set subdir win-64
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu130
pip install -r requirements.txt
pip install mediapipe
```

### Required assets

#### MetrAbs model

The following directory must exist:

```text
server/dl-pose/metrabs_eff2l_384px_800k_28ds_pytorch/
```

It must contain the model files expected by `metrabsInference.py`, including:

- `config.yaml`
- `ckpt.pt`
- `joint_info.npz`
- `joint_transform_matrix.npy`
- `skeleton_infos.pkl`

#### YOLO weights

The person detector inside `metrabs_pytorch/multiperson/person_detector.py` loads:

```python
ultralytics.YOLO('yolov8m.pt')
```

In the current repo layout, `yolov8m.pt` is stored at the repo root.

Keep it available when launching the server from the workspace root.

#### MediaPipe hand model

The MediaPipe variants expect:

```text
server/hand_detection/hand_landmarker.task
```

## Camera Calibration

Calibration is strongly recommended and is part of the maintained path.

Generate calibration files with:

```powershell
conda activate media
python caliberate.py
```

Expected output files are loaded from:

- `calibration_images/metrabs_camera_params.npz`
- or `calibration_images/camera_matrix.npy` and `calibration_images/dist_coeffs.npy`

If these files are missing, the maintained server scripts will stop with an error.

## Run Commands

Run from this directory or from the workspace root as appropriate.

### Body-only streaming server

```powershell
conda activate media
python server\dl-pose\Pose_3D_metrabs_server.py
```

### Body plus MediaPipe streaming server

```powershell
conda activate media
python server\dl-pose\Pose_3D_metrabs_server_mediapipe.py
```

### Local body-only visualization

```powershell
conda activate media
python server\dl-pose\Pose_3D_metrabs.py
```

### Local body plus MediaPipe visualization

```powershell
conda activate media
python server\dl-pose\Pose_3D_metrabs_mediapipe.py
```

## Shared Server Behavior

The shared WebSocket contract, startup message, joint transport layout, visualization path, and common performance reporting are documented in [`../README.md`](../README.md).

## Troubleshooting

### `Calibration files not found`

Cause:

- calibration data is missing

Fix:

```powershell
conda activate media
python caliberate.py
```

### `Unable to open camera index 0`

Cause:

- webcam not connected
- wrong camera index
- another app is using the webcam

Fix:

- close other camera apps
- test the webcam with `python server\dl-pose\webcam.py`
- adjust `CAMERA_INDEX` in the relevant script if needed

### YOLO weight file not found

Cause:

- `yolov8m.pt` is missing or the script is started from the wrong working directory

Fix:

- make sure `yolov8m.pt` is available
- prefer launching from the workspace root

### MediaPipe variant starts but no fingertip data is streamed

Cause:

- hand model missing
- MediaPipe cannot confidently detect hands
- hands not visible enough in the frame

Fix:

- verify `server/hand_detection/hand_landmarker.task` exists
- improve lighting
- keep hands visible and not too motion-blurred

### Low FPS or high latency

Cause:

- GPU not being used effectively
- another app is using the GPU
- visualization was enabled
- too much extra work was added in the hot path

Fix:

- run the server headless
- check the `[perf]` output
- close GPU-heavy apps
- keep `NUM_AUG = 1` and `MAX_DETECTIONS = 1` unless you have measured the impact

## Improvement Guidance

If you extend this module, prefer improving these areas:

- body-pose stability in `Pose_3D_metrabs_server.py`
- hand fusion robustness in `Pose_3D_metrabs_server_mediapipe.py`
- calibration handling in `calibration_utils.py`
- the MetrAbs-specific inference path and mapping logic

Avoid:

- increasing queue depth in the hot path
- switching the streaming payload to JSON
- enabling visualization by default
- changing the outgoing joint order without updating the client
