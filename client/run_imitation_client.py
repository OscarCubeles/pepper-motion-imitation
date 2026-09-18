# -*- coding: utf-8 -*-
from __future__ import print_function

if __package__:
    from . import pepper_config as settings
else:
    import pepper_config as settings

settings.set_client_dir()

if __package__:
    from . import pose_stream_runtime
else:
    import pose_stream_runtime


def run_imitation_client(
    server_host=settings.SERVER_HOST,
    port=settings.PORT,
    mirroring_imitation=settings.MIRRORING_IMITATION,
    request_message=settings.REQUEST_MESSAGE,
    stop_key="q",
    run_duration_sec=None,
    joint_observer=None,
    backend_profile=settings.POSE_BACKEND_PROFILE,
):
    """Run the shared, audio-free imitation client."""
    return pose_stream_runtime.run_pose_stream_client(
        server_host=server_host,
        port=port,
        mirroring_imitation=mirroring_imitation,
        request_message=request_message,
        stop_key=stop_key,
        run_duration_sec=run_duration_sec,
        joint_observer=joint_observer,
        backend_profile=backend_profile,
    )


def main():
    return run_imitation_client(
        server_host=settings.SERVER_HOST,
        port=settings.PORT,
        backend_profile=settings.POSE_BACKEND_PROFILE,
    )


if __name__ == "__main__":
    main()
