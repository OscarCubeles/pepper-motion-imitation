# Testing Guide

This document describes the automated tests currently stored in the repository
and the environment required to run them.

## 1. Test suites

### 1.1 Shared server tests

Location: `server/tests/`

| Test | Scope |
|---|---|
| `test_runtime_layout.py` | Shared asset paths, Pepper resources, and payload layout |
| `test_server_entrypoints.py` | Mocked startup/shutdown smoke tests for baseline, proposed, and IKPy servers |

The entry-point tests mock camera, model, WebSocket, and visualization
boundaries. They do not open hardware or command Pepper.

### 1.2 Evaluation tests

Location: `server/baseline/evaluation/tests/`

| Test | Scope |
|---|---|
| `test_evaluate_ik_wom.py` | Wrist-orientation metric behavior |
| `test_evaluation_results_app.py` | Evaluation dashboard data helpers |
| `test_video_flipping.py` | Video and annotation flipping behavior |

### 1.3 Client tests

Location: `client/tests/`

| Test | Scope |
|---|---|
| `test_server_angle_payload.py` | Validation and conversion of server-computed Pepper angles |
| `test_joint_limits.py` | Shared HipRoll ±15° command limit and pass-through behavior |

## 2. Run server tests

Use the Python 3 `media` environment from the repository root:

```powershell
conda activate media
python -m unittest discover -s server/tests -p "test_*.py"
python -m unittest discover -s server/baseline/evaluation/tests -p "test_*.py"
```

The shared server tests import runtime modules, so the server dependencies must
be installed. `test_runtime_layout.py` also expects the untracked MetrAbs
checkpoint to exist in its configured location.

## 3. Run client tests

Use the 32-bit Python 2.7 `pepper` environment:

```powershell
conda activate pepper
python -m unittest discover -s client/tests -p "test_*.py"
```

The client tests focus on pure payload logic and should not command Pepper.

## 4. What the automated tests do not cover

The current suites do not replace manual validation of:

- webcam access and camera calibration;
- CUDA availability and real-time MetrAbs inference;
- MediaPipe behavior under real lighting and occlusion;
- WebSocket timing under sustained streaming;
- Pepper network connectivity;
- physical joint motion and safety; or
- dashboard visual layout.

Test these areas manually before a live demonstration or data-collection run.

## 5. Recommended validation order

1. Run the shared server unit tests.
2. Run the evaluation tests.
3. Run the client payload tests.
4. Start the selected server without Pepper and inspect visualization.
5. Connect the client to the simulator.
6. Only then connect to the physical robot.

## 6. Adding tests

- Put shared runtime tests in `server/tests/`.
- Put evaluation-specific tests in `server/baseline/evaluation/tests/`.
- Put Python 2.7 client tests in `client/tests/`.
- Mock camera, CUDA, network, and robot boundaries in automated tests.
- Do not require physical Pepper hardware for a unit test.
- Preserve compatibility with the Python version of the target suite.

## 7. Related documentation

- [Root setup and execution](../../README.md)
- [Client](../../client/README.md)
- [Shared runtime](../common/README.md)
- [Baseline](../baseline/README.md)
- [Proposed Constrained Method](../proposed/README.md)
- [IKPy](../ikpy/README.md)
