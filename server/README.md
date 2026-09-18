# Motion-Imitation Server

The server is organized into one shared runtime and three method packages.

## Packages

- [`common/`](common/README.md): MetrAbs, MediaPipe, transport, visualization,
  settings, shared kinematics, and runtime assets.
- [`baseline/`](baseline/README.md): pose-streaming analytical baseline and
  evaluation tools.
- [`proposed/`](proposed/README.md): server-side analytical angles and
  singularity handling.
- [`ikpy/`](ikpy/README.md): numerical IKPy method, Pepper chains, and workspace
  calibration.
- [`tests/`](tests/README.md): server, evaluation, and client test guide.

## Entry points

Run from the repository root:

```powershell
python -m server.baseline.run_motion_imitation_server
python -m server.proposed.run_motion_imitation_server
python -m server.ikpy.run_motion_imitation_server
```

Start only one method at a time. All methods listen on
`ws://127.0.0.1:8080`, wait for the `keypoints` request, and use the shared
audio-free Pepper client.

See the [root README](../README.md) for installation and execution.
