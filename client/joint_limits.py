"""Shared Pepper joint limits applied immediately before command dispatch."""

from __future__ import division

import math


HIP_ROLL_LIMIT_DEG = 15.0
HIP_ROLL_LIMIT_RAD = math.radians(HIP_ROLL_LIMIT_DEG)


def clamp_joint_target(name, value):
    """Return a command target constrained by project-specific safety limits."""
    value = float(value)
    if name == "HipRoll":
        return max(-HIP_ROLL_LIMIT_RAD, min(HIP_ROLL_LIMIT_RAD, value))
    return value
