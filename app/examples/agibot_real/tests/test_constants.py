"""Sanity checks on the static constants. These don't touch GDK at all."""

from __future__ import annotations

from examples.agibot_real import constants


def test_state_layout_size():
    # 7 + 7 + 5 + 3 + 2 grippers = 24
    assert len(constants.STATE_LAYOUT) == 24
    assert constants.STATE_PAD == 8
    assert constants.STATE_DIM == 32


def test_state_layout_no_duplicates():
    assert len(set(constants.STATE_LAYOUT)) == len(constants.STATE_LAYOUT)


def test_action_layout_size():
    # left arm 7 + right arm 7 + grippers 2 = 16
    assert len(constants.ACTION_LAYOUT) == 16
    assert constants.ACTION_LAYOUT[:7] == constants.ARM_L_JOINTS
    assert constants.ACTION_LAYOUT[7:14] == constants.ARM_R_JOINTS


def test_every_action_joint_has_limits():
    for name in constants.ACTION_LAYOUT:
        assert name in constants.JOINT_LIMITS, f"missing limits for {name}"


def test_camera_keys_match_server_schema():
    # These must match LerobotACOTGo2DataConfig.repack_transforms keys.
    assert set(constants.CAMERA_TYPES) == {"top_head", "hand_left", "hand_right"}
