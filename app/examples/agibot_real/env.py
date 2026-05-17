"""AgiBot G2 real-hardware environment for the openpi-acot stack.

Layout follows the ``acot_agibot_real`` train config (LerobotACOTGo2DataConfig +
Go2ACOTInputs/Outputs):

  state  (21 dims) = arm_joints(14: L7+R7) + effectors(2: L+R) + waist(5)
  action (21 dims) = same layout — server returns 21-D chunks per step.

Cameras returned to the policy use the Go2 dataset names:
  ``top_head`` (= AgiBot ``head`` color), ``hand_left``, ``hand_right``.

The policy ``prompt`` and ``task`` are *not* derived from the robot — set them
on the environment via ``set_prompt`` / ``set_task`` (or pass at construction).
"""

import logging
import time
from io import BytesIO
from typing import List, Optional  # noqa: UP035

import agibot_gdk
import numpy as np
from openpi_client.runtime import environment as _environment
from PIL import Image
from typing_extensions import override

_log = logging.getLogger(__name__)


# Output layout produced by the policy (Go2ACOTOutputs slices to 21 dims).
ARM_JOINTS_PER_SIDE = 7
NUM_ARM_JOINTS = 2 * ARM_JOINTS_PER_SIDE  # 14
NUM_EFFECTORS = 2
NUM_WAIST_JOINTS = 5
ACTION_DIM = NUM_ARM_JOINTS + NUM_EFFECTORS + NUM_WAIST_JOINTS  # 21

# Slice indices into the 21-D vector.
LEFT_ARM = slice(0, 7)
RIGHT_ARM = slice(7, 14)
LEFT_EFF = slice(14, 15)
RIGHT_EFF = slice(15, 16)
WAIST = slice(16, 21)

# Dataset/policy camera key → AgiBot GDK image key.
CAMERA_RENAME = {
    "top_head": "head",
    "hand_left": "hand_left",
    "hand_right": "hand_right",
}


def _decode_color(image_info: dict) -> np.ndarray:
    """Decode a JPEG returned by Env.get_observation into uint8 H,W,3 RGB."""
    encoding = image_info["encoding"]
    if encoding != "JPEG":
        raise ValueError(f"Expected JPEG color image, got encoding={encoding!r}")
    raw = image_info["image_data"]
    if not isinstance(raw, (bytes, bytearray)):
        raw = bytes(raw)
    img = np.asarray(Image.open(BytesIO(raw)).convert("RGB"), dtype=np.uint8)

    return img


def _to_list_of_lists(values) -> List[List[float]]:  # noqa: UP006
    arr = np.asarray(values, dtype=np.float64)
    if arr.ndim == 1:
        arr = arr[None, :]
    if arr.ndim != 2:
        raise ValueError(f"Expected 1D or 2D values, got shape {arr.shape}")
    return [[float(x) for x in row] for row in arr]



class AgibotRealEnvironment(_environment.Environment):
    """Real-world AgiBot G2 env for the ``acot_agibot_real`` policy."""

    def __init__(
        self,
        prompt: str = "",
        task_name: str = "",
        camera_types: Optional[List["agibot_gdk.CameraType"]] = None,  # noqa: UP006,UP007
        trajectory_reference_time: float = 1.0,
        init_wait_seconds: float = 2.0,
        gdk_init: bool = True,
    ) -> None:
        self._prompt = prompt
        # NOTE: field name is `task_name` (read by Policy.post_process) — NOT
        # `task`, which would trigger Go2ACOTInputs.random_inject_prompt's
        # training branch and require episode_index. They are different things.
        self._task_name = task_name
        self._trajectory_reference_time = trajectory_reference_time
        self._owns_gdk = gdk_init

        if gdk_init:
            res = agibot_gdk.gdk_init()
            if res != agibot_gdk.GDKRes.kSuccess:
                raise RuntimeError(f"GDK init failed: {res}")

        if camera_types:
            self._env = agibot_gdk.Env(camera_types=camera_types)
        else:
            self._env = agibot_gdk.Env(
                camera_types=[
                    agibot_gdk.CameraType.kHeadColor,
                    agibot_gdk.CameraType.kHandLeftColor,
                    agibot_gdk.CameraType.kHandRightColor,
                ]
            )

        # Robot instance for diagnostic state reads (not for control)
        self._robot = agibot_gdk.Robot()

        time.sleep(init_wait_seconds)

        self._last_obs: Optional[dict] = None  # noqa: UP007

    # ----- user-supplied inference inputs -----------------------------------
    def set_prompt(self, prompt: str) -> None:
        self._prompt = prompt

    def set_task_name(self, task_name: str) -> None:
        self._task_name = task_name

    @property
    def prompt(self) -> str:
        return self._prompt

    @property
    def task_name(self) -> str:
        return self._task_name

    # ----- lifecycle --------------------------------------------------------
    def close(self) -> None:
        if self._owns_gdk:
            agibot_gdk.gdk_release()
            self._owns_gdk = False

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

    # ----- Environment ABC --------------------------------------------------
    @override
    def reset(self) -> None:
        # GDK has no software reset; making this a no-op avoids a blocking
        # get_observation that would deadlock if the cameras / DDS topics
        # haven't fully published yet.
        return

    @override
    def is_episode_complete(self) -> bool:
        return False

    @override
    def get_observation(self) -> dict:
        t0 = time.perf_counter()
        raw = self._env.get_observation(include_images=True)
        t1 = time.perf_counter()
        _log.info("GDK get_observation done in %.1f ms", (t1 - t0) * 1000)
        self._last_obs = raw

        states = raw["states"]
        arm_q = np.asarray(states["arm_joint_states"], dtype=np.float32)
        eff_q = np.asarray(states["effector_states"], dtype=np.float32)
        waist_q = np.asarray(states["waist_joint_states"], dtype=np.float32)
        if arm_q.shape[0] != NUM_ARM_JOINTS:
            raise RuntimeError(f"arm_joint_states has {arm_q.shape[0]} dims, expected {NUM_ARM_JOINTS}")
        if eff_q.shape[0] != NUM_EFFECTORS:
            raise RuntimeError(f"effector_states has {eff_q.shape[0]} dims, expected {NUM_EFFECTORS}")
        if waist_q.shape[0] != NUM_WAIST_JOINTS:
            raise RuntimeError(f"waist_joint_states has {waist_q.shape[0]} dims, expected {NUM_WAIST_JOINTS}")
        state_vec = np.concatenate([arm_q, eff_q, waist_q], dtype=np.float32)

        # Send raw uint8 HWC frames; the server-side ModelTransformFactory
        # handles resize/normalization. Hard-fail on a missing camera so we
        # don't silently drift from training-time inputs.
        raw_images = raw.get("images") or {}
        images: dict = {}
        for policy_name, gdk_name in CAMERA_RENAME.items():
            entry = raw_images.get(gdk_name)
            if entry is None or not entry.get("image_data"):
                raise RuntimeError(f"Missing camera {gdk_name!r} in Env observation")
            images[policy_name] = _decode_color(entry)

        # `task_name` is consumed by Policy.post_process (waist policy):
        #   * unset                          → return full 21-D action
        #   * "sorting_packages"[_continuous] → freeze waist[0:4] to state, use waist[4]
        #   * any other string               → truncate action to 16-D (drop waist)
        # `task` (different field!) would trigger Go2ACOTInputs' training-time
        # random_inject_prompt path, so we never set it.
        out = {
            "state": state_vec,
            "images": images,
            "prompt": self._prompt,
        }
        if self._task_name:
            out["task_name"] = self._task_name

        return out


    def execute_chunk(
        self,
        actions: np.ndarray,
        trajectory_reference_time: Optional[float] = None,  # noqa: UP007
    ) -> None:
        """Send a full ``[T, action_dim]`` chunk to ``execute_trajectory``.

        ``execute_trajectory`` is async — calling it again overrides the
        previous trajectory. So we ship the whole chunk as ONE call (chunk
        size = T, total duration = trajectory_reference_time) and let the
        controller execute it before we re-plan.

        ``action_dim`` must be 21 (arm14+eff2+waist5) or 16 (arm14+eff2,
        waist dropped — server returns this when post_process truncates).
        """
        arr = np.asarray(actions, dtype=np.float64)

        if arr.ndim == 1:
            arr = arr[None, :]
        if arr.ndim != 2 or arr.shape[-1] not in (16, ACTION_DIM):
            raise ValueError(
                f"Expected actions of shape [chunk, 16] or [chunk, {ACTION_DIM}]; got {arr.shape}"
            )

        trt = float(
            trajectory_reference_time
            if trajectory_reference_time is not None
            else self._trajectory_reference_time
        )

        traj = {
            "timestamps": time.time_ns(),
            "trajectory_reference_time": trt,
            "left_arm": {"kind": "JOINT_ABS", "values": _to_list_of_lists(arr[:, LEFT_ARM])},
            "right_arm": {"kind": "JOINT_ABS", "values": _to_list_of_lists(arr[:, RIGHT_ARM])},
            "left_effector": _to_list_of_lists(arr[:, LEFT_EFF]),
            "right_effector": _to_list_of_lists(arr[:, RIGHT_EFF]),
        }

        if arr.shape[-1] == ACTION_DIM:
            traj["waist"] = {"kind": "JOINT_ABS", "values": _to_list_of_lists(arr[:, WAIST])}

        self._env.execute_trajectory(traj)


    @override
    def apply_action(self, action: dict) -> None:
        """openpi-runtime Environment hook.

        DO NOT pair this with ``ActionChunkBroker`` — the broker emits one
        action per step (chunk_size=1) and the async ``execute_trajectory``
        will override every previous call before motion completes, so the
        robot won't move. Use ``execute_chunk`` from a custom loop that
        sends the full chunk in one shot.
        """
        actions = action["actions"] if isinstance(action, dict) else action
        self.execute_chunk(actions)