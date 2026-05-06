"""Test state vector assembly."""

from __future__ import annotations

import numpy as np

from examples.agibot_real import constants
from examples.agibot_real.env import AgibotRealEnvironment


def _make_states():
    """Build (joint_states, end_state) dicts that mimic GDK responses with
    distinct positions per joint so we can verify ordering."""
    # Use float i+0.001 so off-by-one ordering bugs are detectable.
    body_positions = {
        name: float(i) + 0.001
        for i, name in enumerate(
            constants.ARM_L_JOINTS
            + constants.ARM_R_JOINTS
            + constants.WAIST_JOINTS
            + constants.HEAD_JOINTS
        )
    }
    left_ee = {constants.GRIPPER_L: -0.1}
    right_ee = {constants.GRIPPER_R: -0.2}

    joint_states = {
        "states": [{"name": n, "motor_position": p} for n, p in body_positions.items()],
    }
    end_state = {
        "left_end_state": {
            "names": list(left_ee.keys()),
            "end_states": [{"position": p} for p in left_ee.values()],
        },
        "right_end_state": {
            "names": list(right_ee.keys()),
            "end_states": [{"position": p} for p in right_ee.values()],
        },
    }
    return joint_states, end_state, body_positions, left_ee, right_ee


def test_pack_state_shape_and_dtype():
    joint_states, end_state, *_ = _make_states()
    out = AgibotRealEnvironment._pack_state(joint_states, end_state)
    assert out.shape == (constants.STATE_DIM,)
    assert out.dtype == np.float32


def test_pack_state_order_matches_layout():
    joint_states, end_state, body, left_ee, right_ee = _make_states()
    out = AgibotRealEnvironment._pack_state(joint_states, end_state)
    expected = []
    for name in constants.STATE_LAYOUT:
        if name in body:
            expected.append(body[name])
        elif name in left_ee:
            expected.append(left_ee[name])
        elif name in right_ee:
            expected.append(right_ee[name])
        else:
            expected.append(0.0)
    np.testing.assert_allclose(out[: len(expected)], expected, rtol=0, atol=1e-6)


def test_pack_state_padding_is_zero():
    joint_states, end_state, *_ = _make_states()
    out = AgibotRealEnvironment._pack_state(joint_states, end_state)
    np.testing.assert_array_equal(out[len(constants.STATE_LAYOUT) :], 0.0)


def test_pack_state_handles_missing_joints():
    # Empty dicts — every layout slot falls back to 0.
    out = AgibotRealEnvironment._pack_state(
        {"states": []},
        {"left_end_state": {"names": [], "end_states": []},
         "right_end_state": {"names": [], "end_states": []}},
    )
    assert out.shape == (constants.STATE_DIM,)
    np.testing.assert_array_equal(out, 0.0)
