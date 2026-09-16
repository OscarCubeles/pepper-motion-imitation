import base64
import json
import unittest

import numpy as np

from server.common import pose_mapping, settings


class RuntimePathTests(unittest.TestCase):

    def test_shared_assets_resolve_independently_of_working_directory(self):
        self.assertTrue(settings.CALIBRATION_DIR.is_dir())
        self.assertTrue(settings.HAND_LANDMARKER_PATH.is_file())
        self.assertTrue(settings.YOLO_MODEL_PATH.is_file())
        self.assertTrue(settings.METRABS_MODEL_DIR.is_dir())
        self.assertTrue((settings.METRABS_MODEL_DIR / "config.yaml").is_file())
        self.assertTrue((settings.METRABS_MODEL_DIR / "ckpt.pt").is_file())

    def test_pepper_chain_files_keep_their_urdf_dependency(self):
        urdf_path = settings.PEPPER_RESOURCES_DIR / "pepper.urdf"
        self.assertTrue(urdf_path.is_file())
        for name in ("pepper_left_arm.json", "pepper_right_arm.json"):
            path = settings.PEPPER_RESOURCES_DIR / name
            with path.open("r", encoding="utf-8") as stream:
                config = json.load(stream)
            self.assertEqual(config["urdf_file"], "pepper.urdf")


class PayloadContractTests(unittest.TestCase):

    def test_baseline_pose_payload_does_not_require_angles(self):
        pose = np.arange(26 * 3, dtype=np.float32).reshape(26, 3)
        payload = pose_mapping.prepare_full_pose_data(pose, {"Right": {"primary": "UP"}})
        self.assertNotIn("angles", payload)
        self.assertEqual(payload["hand_orientation"]["Right"]["primary"], "UP")
        decoded = np.frombuffer(base64.b64decode(payload["pose_keypoints"]), dtype=np.float32)
        np.testing.assert_array_equal(decoded.reshape(26, 3), pose)


if __name__ == "__main__":
    unittest.main()
