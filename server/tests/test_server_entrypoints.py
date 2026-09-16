import importlib
import unittest
from contextlib import ExitStack
from unittest.mock import MagicMock, patch


ENTRYPOINTS = (
    "server.baseline.run_motion_imitation_server",
    "server.proposed.run_motion_imitation_server",
    "server.ikpy.run_motion_imitation_server",
)


class _StopBeforeFirstFrame:
    def __init__(self):
        self.check_count = 0

    def is_set(self):
        self.check_count += 1
        return True


class ServerEntrypointSmokeTests(unittest.TestCase):

    def test_servers_reach_streaming_loop_with_mocked_boundaries(self):
        for module_name in ENTRYPOINTS:
            with self.subTest(module=module_name):
                module = importlib.import_module(module_name)
                stop_event = _StopBeforeFirstFrame()

                with ExitStack() as stack:
                    stack.enter_context(patch.object(module, "Event", return_value=stop_event))
                    stack.enter_context(patch.object(module, "_open_camera", return_value=MagicMock()))
                    load_model = stack.enter_context(
                        patch.object(
                            module.pose_map,
                            "_load_metrabs_model",
                            return_value=(None, None, None, [], []),
                        )
                    )
                    stack.enter_context(
                        patch.object(
                            module.pose_map,
                            "_setup_metrabs_joints",
                            return_value=([], []),
                        )
                    )
                    stack.enter_context(
                        patch.object(
                            module.pose_map,
                            "_setup_hand_and_server",
                            return_value=(MagicMock(), MagicMock(), MagicMock(), MagicMock()),
                        )
                    )
                    stack.enter_context(
                        patch.object(
                            module,
                            "setup_sender_thread",
                            return_value=(None, None, None, None),
                        )
                    )
                    stack.enter_context(
                        patch.object(
                            module,
                            "setup_visualizer",
                            return_value=(None, None),
                        )
                    )
                    stack.enter_context(patch.object(module, "AngleCalculator", MagicMock))
                    stack.enter_context(patch.object(module, "AngleClassifier", MagicMock))
                    if hasattr(module, "HumanArmClassifier"):
                        stack.enter_context(patch.object(module, "HumanArmClassifier", MagicMock))
                    if hasattr(module, "PoseHandler"):
                        stack.enter_context(patch.object(module, "PoseHandler", MagicMock))
                    stack.enter_context(patch.object(module.torch.cuda, "Stream", return_value=MagicMock()))
                    shutdown = stack.enter_context(patch.object(module, "shutdown_server"))
                    stack.enter_context(patch.object(module.clientws, "_set_server_stop_event"))

                    module.main()

                load_model.assert_called_once()
                self.assertGreaterEqual(stop_event.check_count, 1)
                shutdown.assert_called_once()


if __name__ == "__main__":
    unittest.main()
