from __future__ import print_function

import unittest

from client.server_angle_payload import build_server_joint_targets


class ServerAnglePayloadTests(unittest.TestCase):

    def _core_angles(self):
        return {
            "ShoulderPitch_Right": 0.1,
            "ShoulderRoll_Right": 0.2,
            "ElbowYaw_Right": 0.3,
            "ElbowRoll_Right": 0.4,
            "ShoulderPitch_Left": -0.1,
            "ShoulderRoll_Left": -0.2,
            "ElbowYaw_Left": -0.3,
            "ElbowRoll_Left": -0.4,
        }

    def test_maps_valid_server_angles(self):
        angles = self._core_angles()
        angles["LWristYaw"] = -0.5
        names, values, speeds = build_server_joint_targets({
            "pepper": angles
        })
        self.assertEqual(names[:4], [
            "RShoulderPitch", "RShoulderRoll", "RElbowYaw", "RElbowRoll"
        ])
        self.assertIn("LWristYaw", names)
        self.assertEqual(len(values), 9)
        self.assertEqual(speeds, [0.5] * 9)

    def test_rejects_invalid_or_missing_angles(self):
        for payload in (None, [], {}, {"pepper": None}, {"pepper": []}):
            self.assertEqual(build_server_joint_targets(payload), ([], [], []))

    def test_incomplete_or_invalid_core_angles_trigger_fallback(self):
        incomplete = self._core_angles()
        del incomplete["ElbowRoll_Left"]
        invalid = self._core_angles()
        invalid["ShoulderRoll_Right"] = float("nan")

        for pepper_angles in (incomplete, invalid):
            self.assertEqual(
                build_server_joint_targets({"pepper": pepper_angles}),
                ([], [], []),
            )


if __name__ == "__main__":
    unittest.main()
