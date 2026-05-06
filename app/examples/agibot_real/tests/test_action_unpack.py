"""Integration test: full apply_action path with the fake GDK."""

from __future__ import annotations

import numpy as np

from examples.agibot_real import constants
from examples.agibot_real.env import AgibotRealEnvironment
from examples.agibot_real.tests.fakes import agibot_gdk as fake_gdk


def _make_env(**kwargs) -> AgibotRealEnvironment:
    # Reset module-level state on the fake.
    fake_gdk._init_called = False
    fake_gdk._release_called = False
    return AgibotRealEnvironment(prompt="test prompt", gdk=fake_gdk, **kwargs)


def test_apply_action_dispatches_to_servo_with_layout_order():
    import pytest
    env = _make_env()
    action = np.zeros(len(constants.ACTION_LAYOUT), dtype=np.float32)
    action[0] = 0.123
    action[7] = -0.456
    env.apply_action({"actions": action})
    sent = env._robot.servo_calls[-1]
    assert sent.joint_names == list(constants.ACTION_LAYOUT)
    # float32 -> float64 introduces small precision drift; tolerate it.
    assert sent.joint_positions[0] == pytest.approx(0.123, abs=1e-5)
    assert sent.joint_positions[7] == pytest.approx(-0.456, abs=1e-5)


def test_apply_action_clamps_out_of_range():
    env = _make_env()
    a = np.zeros(len(constants.ACTION_LAYOUT), dtype=np.float32)
    a[0] = 100.0  # idx21_arm_l_joint1 limit is 3.071796
    env.apply_action({"actions": a})
    sent = env._robot.servo_calls[-1].joint_positions
    assert sent[0] == constants.JOINT_LIMITS["idx21_arm_l_joint1"][1]


def test_apply_action_dry_run_skips_servo():
    env = _make_env(dry_run=True)
    a = np.zeros(len(constants.ACTION_LAYOUT), dtype=np.float32)
    env.apply_action({"actions": a})
    assert env._robot.servo_calls == []


def test_apply_action_accepts_longer_action_with_truncation():
    """Server may return action_dim > 16 (padded). Client takes the first N."""
    env = _make_env()
    longer = np.zeros(32, dtype=np.float32)
    longer[0] = 0.5
    env.apply_action({"actions": longer})
    sent = env._robot.servo_calls[-1]
    assert len(sent.joint_positions) == len(constants.ACTION_LAYOUT)
    assert sent.joint_positions[0] == 0.5


def test_get_observation_returns_expected_keys_and_shapes():
    env = _make_env()
    # Pre-fill the fake with positions and frames.
    for name in (
        constants.ARM_L_JOINTS + constants.ARM_R_JOINTS
        + constants.WAIST_JOINTS + constants.HEAD_JOINTS
    ):
        env._robot.body_positions[name] = 0.1
    env._robot.left_ee[constants.GRIPPER_L] = -0.3
    env._robot.right_ee[constants.GRIPPER_R] = -0.4

    h, w = 480, 640
    arr = np.full((h, w, 3), 100, dtype=np.uint8)
    for cam_type in env._cam_types.values():
        env._camera.frames[cam_type] = fake_gdk.FakeImage(
            data=arr.tobytes(), width=w, height=h, encoding="rgb8"
        )

    obs = env.get_observation()
    assert set(obs) == {"images", "state", "prompt"}
    assert obs["prompt"] == "test prompt"
    assert obs["state"].shape == (constants.STATE_DIM,)
    assert obs["state"].dtype == np.float32
    for cam_key in ("top_head", "hand_left", "hand_right"):
        img = obs["images"][cam_key]
        assert img.shape == (3, 224, 224)
        assert img.dtype == np.uint8


def test_estop_engaged_raises_at_init():
    fake_gdk._init_called = False
    # Patch Robot to flip estop on first instantiation.
    real_robot_cls = fake_gdk.Robot

    class EstopRobot(real_robot_cls):
        def __init__(self):
            super().__init__()
            self.estop_left = True

    fake_gdk.Robot = EstopRobot
    try:
        import pytest
        with pytest.raises(RuntimeError, match="e-stop"):
            AgibotRealEnvironment(prompt="x", gdk=fake_gdk)
    finally:
        fake_gdk.Robot = real_robot_cls


def test_camera_timeout_raises():
    env = _make_env()
    # Leave env._camera.frames empty -> get_latest_image returns None.
    import pytest
    from examples.agibot_real.env import CameraTimeout
    with pytest.raises(CameraTimeout):
        env.get_observation()
