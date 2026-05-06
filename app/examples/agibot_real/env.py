"""Agibot G02 real-robot environment adapter for the openpi WS client runtime.

Wraps `agibot_gdk` (Python bindings for GDK v2.6.3.1) into the
`openpi_client.runtime.environment.Environment` contract. The packed
observation matches what the server-side `Go2ACOTInputs` /
`LerobotACOTGo2DataConfig` expects under `EnvMode.G2SIM`, so the existing
`scripts/serve_policy.py --env=g2sim` server consumes it without any change.

Pure I/O. No threads. Control rate is whatever the runtime drives us at; GDK
does the per-joint interpolation in `joint_servo_control`. If real-robot
trials show jitter, upgrade to a 100 Hz repeater thread (separate change).
"""

from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING, Any

import einops
import numpy as np
from openpi_client import image_tools
from openpi_client.runtime import environment as _environment
from typing_extensions import override

from . import constants

if TYPE_CHECKING:  # only for type hints — the real module is imported lazily
    import agibot_gdk

log = logging.getLogger(__name__)


class CameraTimeout(RuntimeError):
    """Raised when a single camera fails to deliver a frame within tolerance."""


class AgibotRealEnvironment(_environment.Environment):
    """Drives a physical Agibot G02 via GDK Python.

    Args:
        prompt: task description string sent with every observation.
        gdk: the `agibot_gdk` module (real one in production, fake in tests).
            Inject it so unit tests can run without a robot.
        render_hw: target (H, W) for resized camera frames. Server training
            used 224×224.
        reset_pose: optional 16-dim list (left arm 7 + right arm 7 + grippers 2)
            applied at `reset()`. None means do nothing on reset.
        image_timeout_ms: per-camera deadline for `get_latest_image`.
        speed_scale: in (0, 1]; scales the delta between current pose and
            commanded pose. <1 is for cautious first-time deployment.
        dry_run: if True, skip `joint_servo_control` calls (for staging).
    """

    def __init__(
        self,
        *,
        prompt: str,
        gdk: Any,
        render_hw: tuple[int, int] = constants.TARGET_HW,
        reset_pose: list[float] | None = None,
        image_timeout_ms: float = 200.0,
        speed_scale: float = 1.0,
        dry_run: bool = False,
    ) -> None:
        if not 0.0 < speed_scale <= 1.0:
            raise ValueError(f"speed_scale must be in (0, 1], got {speed_scale}")

        self._prompt = prompt
        self._gdk = gdk
        self._render_h, self._render_w = render_hw
        self._image_timeout_ms = image_timeout_ms
        self._speed_scale = speed_scale
        self._dry_run = dry_run
        self._reset_pose = reset_pose

        # Resolve CameraType enum values from the injected module. Doing this
        # lazily means tests can use a fake module with the same attribute
        # surface but different semantics.
        self._cam_types: dict[str, Any] = {
            key: getattr(gdk.CameraType, attr)
            for key, attr in constants.CAMERA_TYPES.items()
        }

        if gdk.gdk_init() != gdk.GDKRes.kSuccess:
            raise RuntimeError("gdk_init failed; check 10.42.1.101 connectivity")

        self._robot = gdk.Robot()
        self._camera = gdk.Camera(list(self._cam_types.values()))
        # GDK docs recommend sleeping ~2-3s for hardware to settle.
        time.sleep(2.0)

        self._check_estop()
        self._warn_if_unexpected_gripper()

    # ------------------------------------------------------------------
    # Environment interface

    @override
    def reset(self) -> None:
        if self._reset_pose is None:
            log.info("reset(): no reset_pose configured, skipping")
            return
        if len(self._reset_pose) != len(constants.ACTION_LAYOUT):
            raise ValueError(
                f"reset_pose length {len(self._reset_pose)} != "
                f"ACTION_LAYOUT length {len(constants.ACTION_LAYOUT)}"
            )
        log.info("reset(): moving to reset_pose")
        # Use planning move (joint_control_request) — slower but safer than
        # servo for the initial positioning.
        req = self._gdk.JointControlReq()
        req.joint_names = list(constants.ACTION_LAYOUT)
        req.joint_positions = list(self._reset_pose)
        req.joint_velocities = [0.3] * len(self._reset_pose)
        req.life_time = 8.0
        req.detail = "agibot_real reset"
        if not self._dry_run:
            self._robot.joint_control_request(req)

    @override
    def is_episode_complete(self) -> bool:
        # Real robot has no env-driven termination; rely on max_episode_steps
        # or external SIGINT.
        return False

    @override
    def get_observation(self) -> dict:
        images = {
            key: self._grab_image(cam_type)
            for key, cam_type in self._cam_types.items()
        }
        state = self._pack_state(
            self._robot.get_joint_states(),
            self._robot.get_end_state(),
        )
        return {
            "images": images,
            "state": state,
            "prompt": self._prompt,
        }

    @override
    def apply_action(self, action: dict) -> None:
        # The server returns a chunk; ActionChunkBroker hands us a single step.
        # Either {"actions": ndarray[D]} or just ndarray[D] depending on broker.
        a = action["actions"] if isinstance(action, dict) else action
        a = np.asarray(a, dtype=np.float32)
        if a.ndim != 1:
            raise ValueError(f"expected 1-D action, got shape {a.shape}")

        joint_names = list(constants.ACTION_LAYOUT)
        if a.shape[0] < len(joint_names):
            raise ValueError(
                f"action has {a.shape[0]} dims, need at least {len(joint_names)}"
            )
        target = a[: len(joint_names)].astype(np.float64).tolist()

        if self._speed_scale < 1.0:
            current = self._read_layout_positions(
                joint_names, self._robot.get_joint_states(), self._robot.get_end_state()
            )
            target = [c + self._speed_scale * (t - c) for c, t in zip(current, target)]

        target = self._clamp_action(joint_names, target)

        if not np.all(np.isfinite(target)):
            raise ValueError("action contained non-finite values; aborting")

        if self._dry_run:
            log.info("[dry-run] would send: %s", list(zip(joint_names, target))[:4])
            return

        req = self._gdk.JointServoControlReq()
        req.control_period = 0.05  # 50 ms; loop runs ~30 Hz, give some slack
        req.joint_names = joint_names
        req.joint_positions = target
        self._robot.joint_servo_control(req)

    # ------------------------------------------------------------------
    # Helpers (also re-used by unit tests)

    def _grab_image(self, cam_type: Any) -> np.ndarray:
        img = self._camera.get_latest_image(cam_type, self._image_timeout_ms)
        if img is None:
            raise CameraTimeout(f"no frame from camera {cam_type} within {self._image_timeout_ms} ms")
        return self._image_to_chw(img, self._render_h, self._render_w)

    @staticmethod
    def _image_to_chw(img: Any, target_h: int, target_w: int) -> np.ndarray:
        """Convert a GDK Image (or compatible duck-type) to CHW uint8."""
        encoding = (img.encoding or "").lower()
        if encoding not in {"rgb8", "bgr8"}:
            raise ValueError(f"unsupported image encoding {encoding!r}")
        arr = np.frombuffer(img.data, dtype=np.uint8).reshape(img.height, img.width, 3)
        if encoding == "bgr8":
            arr = arr[:, :, ::-1]
        arr = image_tools.convert_to_uint8(image_tools.resize_with_pad(arr, target_h, target_w))
        return einops.rearrange(arr, "h w c -> c h w")

    @staticmethod
    def _pack_state(joint_states: dict, end_state: dict) -> np.ndarray:
        positions = AgibotRealEnvironment._read_layout_positions(
            list(constants.STATE_LAYOUT), joint_states, end_state
        )
        out = np.zeros(constants.STATE_DIM, dtype=np.float32)
        out[: len(positions)] = positions
        return out

    @staticmethod
    def _read_layout_positions(
        layout: list[str],
        joint_states: dict,
        end_state: dict,
    ) -> list[float]:
        """Pull positions in `layout` order from GDK responses.

        joint_states comes from `Robot.get_joint_states()`; end_state from
        `Robot.get_end_state()`. We try the body/arm/head joints first, then
        fall back to end_state for end-effector joints (grippers / dexterous
        hands).
        """
        body_pos = {s["name"]: float(s["motor_position"]) for s in joint_states.get("states", [])}
        ee_pos: dict[str, float] = {}
        for side_key in ("left_end_state", "right_end_state"):
            side = end_state.get(side_key, {}) or {}
            for name, st in zip(side.get("names", []), side.get("end_states", [])):
                ee_pos[name] = float(st["position"])

        out: list[float] = []
        missing: list[str] = []
        for name in layout:
            if name in body_pos:
                out.append(body_pos[name])
            elif name in ee_pos:
                out.append(ee_pos[name])
            else:
                missing.append(name)
                out.append(0.0)
        if missing:
            log.warning("joints missing from GDK response, using 0.0: %s", missing)
        return out

    @staticmethod
    def _clamp_action(joint_names: list[str], target: list[float]) -> list[float]:
        out: list[float] = []
        for name, value in zip(joint_names, target):
            lo, hi = constants.JOINT_LIMITS.get(name, (-np.inf, np.inf))
            v = float(np.clip(value, lo, hi))
            out.append(v)
        return out

    def _check_estop(self) -> None:
        status = self._robot.get_whole_body_status()
        if status.get("right_arm_estop") or status.get("left_arm_estop"):
            raise RuntimeError("e-stop is engaged; release it before starting client")

    def _warn_if_unexpected_gripper(self) -> None:
        try:
            end_state = self._robot.get_end_state()
            for side in ("left_end_state", "right_end_state"):
                names = (end_state.get(side, {}) or {}).get("names", [])
                if names and not any("gripper" in n for n in names):
                    log.warning(
                        "%s reports non-gripper joints %s; constants.JOINT_LIMITS "
                        "are tuned for omnipicker, action clamps may be wrong",
                        side, names,
                    )
        except Exception:  # noqa: BLE001
            log.warning("could not query end-effector type", exc_info=True)

    def close(self) -> None:
        try:
            self._camera.close_camera()
        except Exception:  # noqa: BLE001
            log.exception("camera close_camera failed")
        try:
            self._gdk.gdk_release()
        except Exception:  # noqa: BLE001
            log.exception("gdk_release failed")

    def __del__(self) -> None:
        # Best-effort cleanup; users should call close() explicitly.
        try:
            self.close()
        except Exception:  # noqa: BLE001
            pass
