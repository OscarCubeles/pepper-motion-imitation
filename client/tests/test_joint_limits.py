from __future__ import division

import math
import os
import sys
import unittest


CLIENT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if CLIENT_DIR not in sys.path:
    sys.path.insert(0, CLIENT_DIR)

from joint_limits import HIP_ROLL_LIMIT_RAD, clamp_joint_target


class JointLimitTests(unittest.TestCase):

    def test_hip_roll_is_clamped_to_plus_or_minus_fifteen_degrees(self):
        self.assertAlmostEqual(
            clamp_joint_target("HipRoll", math.radians(40.0)),
            HIP_ROLL_LIMIT_RAD,
        )
        self.assertAlmostEqual(
            clamp_joint_target("HipRoll", math.radians(-40.0)),
            -HIP_ROLL_LIMIT_RAD,
        )

    def test_hip_roll_inside_limit_is_unchanged(self):
        value = math.radians(8.0)
        self.assertAlmostEqual(clamp_joint_target("HipRoll", value), value)

    def test_other_joints_are_unchanged(self):
        value = math.radians(40.0)
        self.assertAlmostEqual(clamp_joint_target("HipPitch", value), value)


if __name__ == "__main__":
    unittest.main()
