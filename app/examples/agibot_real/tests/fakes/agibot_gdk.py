"""In-process fake of the `agibot_gdk` Python module.

Surface mirrors GDK v2.6.3.1 closely enough to drive `AgibotRealEnvironment`
unit tests. No threads, no I/O, no hardware. Tests inject this module via
`AgibotRealEnvironment(gdk=...)`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class GDKRes(Enum):
    kSuccess = 0
    kFailure = 1


class CameraType(Enum):
    kHeadColor = "kHeadColor"
    kHandLeftColor = "kHandLeftColor"
    kHandRightColor = "kHandRightColor"
    kHeadDepth = "kHeadDepth"
    kHandLeftDepth = "kHandLeftDepth"
    kHandRightDepth = "kHandRightDepth"
    kHeadStereoLeft = "kHeadStereoLeft"
    kHeadStereoRight = "kHeadStereoRight"


@dataclass
class FakeImage:
    data: bytes
    width: int
    height: int
    encoding: str = "rgb8"
    color_format: str = "RGB"
    bit_depth: int = 8
    timestamp_ns: int = 0


@dataclass
class JointServoControlReq:
    control_period: float = 0.0
    joint_names: list = field(default_factory=list)
    joint_positions: list = field(default_factory=list)
    joint_velocities: list = field(default_factory=list)


@dataclass
class JointControlReq:
    life_time: float = 0.0
    joint_names: list = field(default_factory=list)
    joint_positions: list = field(default_factory=list)
    joint_velocities: list = field(default_factory=list)
    detail: str = ""


# Module-level state so a test can inspect what was sent.
_init_called = False
_release_called = False


def gdk_init() -> GDKRes:
    global _init_called
    _init_called = True
    return GDKRes.kSuccess


def gdk_release() -> GDKRes:
    global _release_called
    _release_called = True
    return GDKRes.kSuccess


@dataclass
class _WholeBodyStatus(dict):
    pass


class Robot:
    """Programmable fake — tests poke at the public attrs to set what the
    next read returns and inspect what got sent."""

    def __init__(self) -> None:
        # name -> motor_position (rad). Tests pre-fill this.
        self.body_positions: dict[str, float] = {}
        # left/right end_state — names + positions.
        self.left_ee: dict[str, float] = {}
        self.right_ee: dict[str, float] = {}

        self.estop_left: bool = False
        self.estop_right: bool = False

        # Capture sent commands for assertions.
        self.servo_calls: list[JointServoControlReq] = []
        self.plan_calls: list[JointControlReq] = []

    def get_joint_states(self) -> dict:
        return {
            "timestamp": 0,
            "nums": len(self.body_positions),
            "states": [
                {
                    "name": n,
                    "mode": 5,
                    "position": p,
                    "velocity": 0.0,
                    "effort": 0.0,
                    "motor_position": p,
                    "motor_velocity": 0.0,
                    "motor_current": 0.0,
                    "error_code": 0,
                }
                for n, p in self.body_positions.items()
            ],
        }

    def get_end_state(self) -> dict:
        def pack(side_pos: dict[str, float]) -> dict:
            return {
                "controlled": True,
                "type": 1,
                "names": list(side_pos.keys()),
                "end_states": [
                    {"id": i, "enable": True, "position": p, "velocity": 0.0,
                     "effort": 0.0, "current": 0.0, "voltage": 0.0,
                     "temperature": 25.0, "status": 0, "err_code": 0}
                    for i, p in enumerate(side_pos.values())
                ],
            }

        return {
            "left_end_state": pack(self.left_ee),
            "right_end_state": pack(self.right_ee),
        }

    def get_whole_body_status(self) -> dict:
        return {
            "timestamp": 0,
            "right_arm_error": 0,
            "left_arm_error": 0,
            "right_arm_control": True,
            "left_arm_control": True,
            "right_arm_estop": self.estop_right,
            "left_arm_estop": self.estop_left,
            "right_end_error": 0,
            "left_end_error": 0,
            "right_end_model": "omnipicker",
            "left_end_model": "omnipicker",
            "waist_error": 0,
            "lift_error": 0,
            "neck_error": 0,
            "chassis_error": 0,
        }

    def joint_servo_control(self, req: JointServoControlReq) -> int:
        self.servo_calls.append(req)
        return 0

    def joint_control_request(self, req: JointControlReq) -> int:
        self.plan_calls.append(req)
        return 0


class Camera:
    """Returns a programmable image per CameraType. Tests fill `frames`."""

    def __init__(self, types: list[CameraType] | None = None) -> None:
        self.opened_types = list(types) if types else []
        self.frames: dict[CameraType, FakeImage | None] = {}
        self.closed: bool = False

    def get_latest_image(self, cam_type: CameraType, timeout_ms: float) -> FakeImage | None:  # noqa: ARG002
        return self.frames.get(cam_type)

    def close_camera(self) -> GDKRes:
        self.closed = True
        return GDKRes.kSuccess
