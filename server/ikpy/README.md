# IKPy Method

The IKPy method provides a numerical inverse-kinematics comparison for the
analytical baseline and proposed constrained method. It uses Pepper URDF chain definitions
and a calibrated MetrAbs workspace to convert detected wrist targets into robot
joint angles.

## 1. Additional setup

Install IKPy in the Python 3 server environment:

```powershell
conda activate media
python -m pip install ikpy ipywidgets
```

Required files:

```text
server/ikpy/metrabs_workspace.json
server/pepper_ik_resources/pepper.urdf
server/pepper_ik_resources/pepper_left_arm.json
server/pepper_ik_resources/pepper_right_arm.json
```

The chain JSON files refer to `pepper.urdf`, so keep these resources together
under `server/pepper_ik_resources/`.

## 2. Workspace calibration

`metrabs_workspace.json` defines the observed human/MetrAbs workspace used to
map wrist targets into Pepper's reachable workspace.

Generate or inspect it with:

```powershell
conda activate media
python -m server.ikpy.run_ikpy_no_client
```

This command opens the webcam and visualizes IKPy results without requiring the
Pepper client. Move both wrists throughout the intended capture workspace. The
tool continuously updates `server/ikpy/metrabs_workspace.json`; press `q` when
the coverage is complete.

The committed workspace file may be machine and camera specific. Recalibrate
it when the capture geometry changes substantially.

## 3. Run the streaming method

From the repository root:

```powershell
conda activate media
python -m server.ikpy.run_motion_imitation_server
```

In a second terminal:

```powershell
conda activate pepper
python client/run_imitation_client.py
```

The shared client consumes the server-computed Pepper angles. No audio is sent
to Pepper.

## 4. Method flow

1. The shared runtime creates the 26-point MetrAbs/MediaPipe pose.
2. `server/ikpy/angle_calculator.py` extracts target positions and shared pose
   information.
3. `ikpy_utils.py` maps targets through the calibrated workspace.
4. IKPy solves the configured Pepper arm chains.
5. Controllable Pepper joint values are extracted from the chain solution.
6. The maintained streaming path applies its shared feasibility and constraint
   processing.
7. Pepper angles and speeds are added to the payload and sent to the client.

## 5. Main files

| File | Purpose |
|---|---|
| `run_motion_imitation_server.py` | Real-time IKPy server |
| `run_ikpy_no_client.py` | Local webcam, workspace, and IK inspection tool |
| `angle_calculator.py` | Connects shared pose calculation to IKPy |
| `ikpy_utils.py` | Chain loading, workspace mapping, and numerical IK helpers |
| `ikpy_controller.py` | Retained interactive IKPy controller |
| `ikpy_controller_metrabs.py` | Retained MetrAbs-oriented controller |
| `metrabs_workspace.json` | Calibrated input workspace bounds |

The notebooks in this directory are exploratory resources and are not required
by the streaming entry point.

## 6. Chain configuration

The left and right arm chain definitions are stored outside this folder so all
methods can reference the same Pepper model:

```text
server/pepper_ik_resources/pepper_left_arm.json
server/pepper_ik_resources/pepper_right_arm.json
server/pepper_ik_resources/pepper.urdf
```

If the URDF or active-joint masks change, regenerate or validate the IKPy
solutions before collecting comparison data.

## 7. Output

Like the proposed constrained method, the IKPy server sends:

- the shared pose and hand-orientation fields;
- computed human/Pepper angle dictionaries; and
- per-joint speed estimates.

Angles use radians. The client validates server angles before dispatching them
to Pepper.

## 8. Evaluation

The offline evaluator exposes this method as `ikpy`. It uses the same Pepper
resources and workspace-mapping utilities as the runtime where applicable.

For a reproducible comparison:

- keep `metrabs_workspace.json` fixed;
- record the file revision with the results;
- use the same annotated videos as the other methods; and
- avoid adjusting the workspace for a single test video.

## 9. Troubleshooting

### Workspace calibration is missing

Run `python -m server.ikpy.run_ikpy_no_client` and create
`server/ikpy/metrabs_workspace.json`.

### IKPy cannot load a chain

Verify the JSON paths, `pepper.urdf`, and the `ikpy` installation in the
`media` environment.

### Targets appear compressed or unreachable

Inspect the calibrated workspace bounds and confirm that the camera position
matches the calibration setup.

### The client ignores the result

Inspect the outgoing `angles.pepper` dictionary and the validation logic in
`client/server_angle_payload.py`.

## 10. Related documentation

- [Root setup and execution](../../README.md)
- [Shared runtime](../common/README.md)
- [Client](../../client/README.md)
- [Baseline](../baseline/README.md)
- [Proposed Constrained Method](../proposed/README.md)
- [Tests](../tests/README.md)
