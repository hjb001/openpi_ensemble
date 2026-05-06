"""Test action clamping against joint limits."""

from __future__ import annotations

from examples.agibot_real import constants
from examples.agibot_real.env import AgibotRealEnvironment


def test_clamp_within_limits_passes_through():
    names = list(constants.ACTION_LAYOUT)
    target = [0.0] * len(names)
    out = AgibotRealEnvironment._clamp_action(names, target)
    assert out == target


def test_clamp_above_max_clamps_to_max():
    names = ["idx21_arm_l_joint1"]
    lo, hi = constants.JOINT_LIMITS[names[0]]
    out = AgibotRealEnvironment._clamp_action(names, [hi + 10.0])
    assert out == [hi]


def test_clamp_below_min_clamps_to_min():
    names = ["idx21_arm_l_joint1"]
    lo, _ = constants.JOINT_LIMITS[names[0]]
    out = AgibotRealEnvironment._clamp_action(names, [lo - 10.0])
    assert out == [lo]


def test_clamp_unknown_joint_passes_through():
    out = AgibotRealEnvironment._clamp_action(["unknown_joint"], [42.0])
    assert out == [42.0]


def test_clamp_omnipicker_gripper_range():
    # GDK_LIMITS for omnipicker grippers is (-0.785, 0). Sending 0.5 should
    # clamp to 0.
    out = AgibotRealEnvironment._clamp_action([constants.GRIPPER_L], [0.5])
    assert out == [0.0]
    out = AgibotRealEnvironment._clamp_action([constants.GRIPPER_L], [-2.0])
    assert out == [-0.785]
