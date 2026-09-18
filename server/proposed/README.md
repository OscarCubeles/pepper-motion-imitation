# Proposed Method

The proposed method extends the analytical motion-imitation pipeline with
server-side Pepper angle calculation, wrist-orientation information, pose
feasibility checks, and stateful singularity handling.

## 1. Run the method

From the repository root:

```powershell
conda activate media
python -m server.proposed.run_motion_imitation_server
```

In a second terminal:

```powershell
conda activate pepper
python client/run_imitation_client.py
```

The imitation client uses the Pepper angles supplied by this server. No audio
is sent to Pepper.

## 2. Method flow

1. The shared MetrAbs and MediaPipe pipeline creates the 26-point pose.
2. `AngleCalculator` computes human and Pepper angles from that payload.
3. `AngleClassifier` evaluates feasibility and detects singularity conditions.
4. `HumanArmClassifier` checks whether relevant angle combinations are
   human-doable.
5. `PoseHandler` applies the maintained singularity constraints in frame order.
6. Joint speeds are calculated from the current and previous Pepper angles.
7. Pose data, hand orientation, angles, and speeds are sent to the client.

The current `PoseHandler` actively applies the dual-singularity branch. The
offline evaluator uses the same maintained behavior for the constrained method.

## 3. Main files

| File | Purpose |
|---|---|
| `run_motion_imitation_server.py` | Real-time proposed-method entry point |
| `pose_handling.py` | Applies singularity constraints and computes speeds |
| `dual_singularity_fsm.py` | Stateful transitions into and out of dual singularities |
| `angle_classifier_human.py` | Human reachability classification for arm-angle pairs |

Shared pose estimation, settings, transport, and visualization live in
`server/common`.

## 4. Singularity handling

The dual-singularity state machine contains three states:

- `NO_SINGULAR`;
- `TRANSITION_TO_SINGULAR`; and
- `STEADY_SINGULAR`.

It uses consecutive-frame thresholds rather than switching constraints from a
single detection. During the transition it gradually adjusts affected joints;
when the singular condition clears, it transitions back instead of changing
the pose abruptly.

The constraint path can use the detected hand-orientation labels when selecting
wrist behavior. Missing labels therefore affect the available orientation
information but do not prevent body-pose transport.

For the exact thresholds and target values, use
`server/proposed/dual_singularity_fsm.py` as the source of truth.

## 5. Angle payload

The proposed server adds this structure to the shared JSON payload:

```json
{
  "angles": {
    "human": {"...": 0.0},
    "pepper": {"...": 0.0},
    "speeds": {"...": 0.0}
  }
}
```

Angles are expressed in radians. The client validates the server angle payload
before converting it into Pepper command lists.

## 6. Visualization and diagnostics

The proposed server sends the following information to the shared visualizer:

- human and Pepper angles;
- feasibility results;
- left and right singularity results;
- missing keypoint information; and
- pose and hand points.

The 2D camera panel is text-free by default. Angle and feasibility information
can appear in the separate **Joint Angles** window when
`ENABLE_ANGLES_VISUALIZATION` is enabled.

## 7. Evaluation mapping

The offline evaluator exposes two related analytical variants:

- `current_solution_raw`: server-side analytical output before maintained
  runtime singularity handling; and
- `current_solution_constrained`: the output after applying the maintained
  `PoseHandler` sequence logic.

Use video-level summaries when comparing these variants; consecutive-frame
state matters for the constrained path.

## 8. Troubleshooting

### Constraints do not appear to activate

Keep the relevant pose visible for several consecutive frames. The state
machine intentionally requires temporal confirmation.

### Wrist behavior is unstable

Keep the full hand visible, improve lighting, and reduce motion blur so
MediaPipe can estimate the palm orientation consistently.

### The client ignores server angles

Confirm that `angles.pepper` is present in the received payload and check
`client/server_angle_payload.py` validation.

## 9. Related documentation

- [Root setup and execution](../../README.md)
- [Shared runtime and payload](../common/README.md)
- [Client](../../client/README.md)
- [Baseline](../baseline/README.md)
- [IKPy](../ikpy/README.md)
- [Tests](../tests/README.md)
