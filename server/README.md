# Motion-imitation server

The runtime is organized as import-safe Python packages:

- `common`: shared MetrAbs, MediaPipe, kinematics, WebSocket, visualization, and assets.
- `baseline`: baseline streaming method.
- `proposed`: constrained proposed method and its state machines.
- `ikpy`: IKPy streaming method and local tools.

Run the canonical entry points from the repository root with `python -m`:

```powershell
python -m server.baseline.run_motion_imitation_server
python -m server.proposed.run_motion_imitation_server
python -m server.ikpy.run_motion_imitation_server
python -m server.ikpy.run_ikpy_no_client
```

All streaming servers listen on `ws://127.0.0.1:8080`, wait for the `keypoints`
request, and preserve the existing JSON/base64 payload contract.

Install dependencies with `pip install -r server/requirements.txt`. Runtime
asset paths are centralized in `server/common/settings.py` and do not depend on
the process working directory.

The IKPy streaming method additionally expects `server/ikpy/metrabs_workspace.json`.
Generate or refresh it with the no-client IKPy command before starting that server.
