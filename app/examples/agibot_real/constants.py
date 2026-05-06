"""Constants for the Agibot G02 real-robot client.

Joint names, limits and ordering are taken from GDK v2.6.3.1 docs (Robot
section). The state layout is the canonical order used to assemble the
32-dim observation vector that the server-side `Go2ACOTInputs` /
`LerobotACOTGo2DataConfig` expects.

If `dump_state_layout.py` against the training dataset reveals a different
order, update STATE_LAYOUT here — nothing else needs to change.
"""

from __future__ import annotations

from typing import Final


ARM_L_JOINTS: Final[tuple[str, ...]] = (
    "idx21_arm_l_joint1",
    "idx22_arm_l_joint2",
    "idx23_arm_l_joint3",
    "idx24_arm_l_joint4",
    "idx25_arm_l_joint5",
    "idx26_arm_l_joint6",
    "idx27_arm_l_joint7",
)

ARM_R_JOINTS: Final[tuple[str, ...]] = (
    "idx61_arm_r_joint1",
    "idx62_arm_r_joint2",
    "idx63_arm_r_joint3",
    "idx64_arm_r_joint4",
    "idx65_arm_r_joint5",
    "idx66_arm_r_joint6",
    "idx67_arm_r_joint7",
)

WAIST_JOINTS: Final[tuple[str, ...]] = (
    "idx01_body_joint1",
    "idx02_body_joint2",
    "idx03_body_joint3",
    "idx04_body_joint4",
    "idx05_body_joint5",
)

HEAD_JOINTS: Final[tuple[str, ...]] = (
    "idx11_head_joint1",
    "idx12_head_joint2",
    "idx13_head_joint3",
)

# Omnipicker single-DOF gripper joints (default end-effector). Other end-effector
# types (dahuan, ctek90d, dexterous hands) have different ranges; see GDK docs
# joint-limit table. The runtime warns if get_end_state() reports a different
# type.
GRIPPER_L: Final[str] = "idx31_gripper_l_inner_joint1"
GRIPPER_R: Final[str] = "idx71_gripper_r_inner_joint1"

# Canonical 32-dim state layout (24 named slots + 8 zero pads).
# Server-side state_mask drops the 8 pads and selected slots; we must still
# supply the full 32-dim vector. See LerobotACOTGo2DataConfig.state_mask.
STATE_LAYOUT: Final[tuple[str, ...]] = (
    ARM_L_JOINTS
    + ARM_R_JOINTS
    + WAIST_JOINTS
    + HEAD_JOINTS
    + (GRIPPER_L, GRIPPER_R)
)

STATE_DIM: Final[int] = 32
STATE_PAD: Final[int] = STATE_DIM - len(STATE_LAYOUT)  # 8

# Mapping from server-side image key -> agibot_gdk.CameraType attribute name.
# We resolve the actual enum at runtime (the fake GDK used in tests doesn't
# need to mirror the real enum).
CAMERA_TYPES: Final[dict[str, str]] = {
    "top_head": "kHeadColor",
    "hand_left": "kHandLeftColor",
    "hand_right": "kHandRightColor",
}

# Training image resolution. The model accepts other sizes (resize_with_pad
# handles letterboxing) but matching the training spec is safest.
TARGET_HW: Final[tuple[int, int]] = (224, 224)

# Joint limits (radians) copied verbatim from GDK docs joint-limit tables.
# Used to clamp commands client-side before joint_servo_control. Out-of-range
# input causes GDK to raise; we never want to send such commands.
JOINT_LIMITS: Final[dict[str, tuple[float, float]]] = {
    # body / waist
    "idx01_body_joint1": (-1.082104, 0.000174),
    "idx02_body_joint2": (-0.000174, 2.652900),
    "idx03_body_joint3": (-1.919862, 1.570970),
    "idx04_body_joint4": (-0.436332, 0.436332),
    "idx05_body_joint5": (-3.045599, 3.045599),
    # head
    "idx11_head_joint1": (-1.570970, 1.570970),
    "idx12_head_joint2": (-0.349240, 0.349240),
    "idx13_head_joint3": (-0.534773, 0.534773),
    # left arm
    "idx21_arm_l_joint1": (-3.071796, 3.071796),
    "idx22_arm_l_joint2": (-2.059505, 2.059505),
    "idx23_arm_l_joint3": (-3.071796, 3.071796),
    "idx24_arm_l_joint4": (-2.495838, 1.012308),
    "idx25_arm_l_joint5": (-3.071796, 3.071796),
    "idx26_arm_l_joint6": (-1.012308, 1.012308),
    "idx27_arm_l_joint7": (-1.535907, 1.535907),
    # right arm
    "idx61_arm_r_joint1": (-3.071796, 3.071796),
    "idx62_arm_r_joint2": (-2.059505, 2.059505),
    "idx63_arm_r_joint3": (-3.071796, 3.071796),
    "idx64_arm_r_joint4": (-2.495838, 1.012308),
    "idx65_arm_r_joint5": (-3.071796, 3.071796),
    "idx66_arm_r_joint6": (-1.012308, 1.012308),
    "idx67_arm_r_joint7": (-1.535907, 1.535907),
    # omnipicker grippers (default)
    "idx31_gripper_l_inner_joint1": (-0.785, 0.0),
    "idx71_gripper_r_inner_joint1": (-0.785, 0.0),
}

# Maximum allowed change between consecutive command steps (per joint, rad).
# Server may emit large jumps at chunk boundaries; this is a soft guard against
# whip-cracking. Tuned empirically for 30 Hz control on G02 — adjust if real-
# robot trials show steady-state lag.
MAX_DELTA_PER_STEP: Final[float] = 0.10

# Action mapping: which dims of the (already-unmasked) action vector returned
# by the server correspond to which physical joints. Keep aligned with the
# server-side action_mask in LerobotACOTGo2DataConfig. Update together with
# STATE_LAYOUT if dump_state_layout.py reveals a different order.
ACTION_LAYOUT: Final[tuple[str, ...]] = (
    ARM_L_JOINTS
    + ARM_R_JOINTS
    + (GRIPPER_L, GRIPPER_R)
)
