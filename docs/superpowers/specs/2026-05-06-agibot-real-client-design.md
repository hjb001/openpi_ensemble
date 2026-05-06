# Agibot G02 Real-Robot Inference Client — Design

Date: 2026-05-06
Status: Approved (in-conversation), implementation starting.
Scope: Add a new client `app/examples/agibot_real/` that drives a physical
Agibot G02 robot via GDK Python (v2.6.3.1), using the existing WebSocket
policy server (`scripts/serve_policy.py --env=g2sim`). **Server is not
modified.**

## 1. Architecture

```
agibot_real/main.py
  ├─ openpi_client.WebsocketClientPolicy ── WS ──▶ serve_policy.py (g2sim, unchanged)
  └─ AgibotRealEnvironment
        ├─ agibot_gdk.Robot   (joint state / joint_servo_control)
        └─ agibot_gdk.Camera  (kHeadColor / kHandLeftColor / kHandRightColor)
```

Server entry: `EnvMode.G2SIM`, training config
`acot_icra_simulation_challenge_reasoning_to_action`, ckpt
`./sim_real_action` (i.e. `app/sim_real_action`). Server-side data class
`LerobotACOTGo2DataConfig` already expects exactly the schema below.

## 2. Interface contract

Outbound observation (client → server):

| field | shape / dtype | source |
| --- | --- | --- |
| `images.top_head` | uint8 (3, 224, 224) CHW | `Camera.get_latest_image(kHeadColor)` |
| `images.hand_left` | uint8 (3, 224, 224) CHW | `kHandLeftColor` |
| `images.hand_right` | uint8 (3, 224, 224) CHW | `kHandRightColor` |
| `state` | float32 [32] | `Robot.get_joint_states()` + `get_end_state()` packed by `STATE_LAYOUT` + 8 zero pads |
| `prompt` | str | CLI arg, default `"Pour the workpiece into the box"` |

Inbound action (server → client): `actions: float32 [horizon, action_dim]`,
already unpacked by `Go2ACOTOutputs`. Client maps the first 16 active dims back
to `(left_arm[7], right_arm[7], gripper_l, gripper_r)` and clamps to joint
limits before sending via `Robot.joint_servo_control`.

## 3. Components

- `constants.py` — joint name lists, camera enum mapping, `STATE_LAYOUT`,
  joint limits, target image (H, W).
- `env.py` — `AgibotRealEnvironment(Environment)` with `reset`,
  `is_episode_complete`, `get_observation`, `apply_action`. Internals:
  `_pack_state`, `_grab_image`, `_clamp_action`.
- `main.py` — `tyro` CLI: host, port, prompt, action_horizon (≤30),
  num_episodes, max_episode_steps, render_hw, reset_pose, dry_run,
  speed_scale.
- `dump_state_layout.py` — one-shot helper: load a sample from training
  dataset and print joint-name order to confirm `STATE_LAYOUT` alignment.
- `tests/` — pure unit tests with a fake `agibot_gdk` module; pytest.
- `Dockerfile`, `requirements.in`, `README.md`.

## 4. Frequencies

- Control loop / inference: `max_hz=30` via `openpi_client.runtime.Runtime`.
- Internal GDK servo: 30 Hz from client; relies on GDK-side interpolation. If
  jitter is observed in real-robot trials, upgrade to a 100 Hz repeater
  thread (deferred — separate change).

## 5. Safety

Fail-fast on init: GDK init, robot connect, camera open, e-stop status, WS
connect (3×2s retries).
Fail-safe in control loop: image timeouts allow one stale frame, then abort;
servo errors hold pose for 1 s; SIGINT/SIGTERM trigger graceful close.
Software clamps in client: per-joint limits from GDK docs;
`MAX_DELTA_PER_STEP = 0.10 rad`; `--speed-scale` linearly attenuates command
delta from current pose; `--dry-run` skips `joint_servo_control`.

## 6. STATE_LAYOUT alignment (TODO before deploy)

The 32-dim state vector layout must match the training dataset. Run
`dump_state_layout.py` against any of the configured `repo_id` lerobot
datasets to print the canonical joint-name order, then update
`constants.STATE_LAYOUT` if needed. Initial guess (24 named slots + 8 pad):
`ARM_L (7) + ARM_R (7) + WAIST (5) + HEAD (3) + GRIPPER_L + GRIPPER_R`.

## 7. Gripper type

Default assumes **omnipicker** (range `[-0.785, 0]`). At init, read
`Robot.get_end_state()['left_end_state']['type']` and emit a warning if it
disagrees with the assumption.

## 8. Test layers

A. Unit tests with fake `agibot_gdk` — packers, clamps, image conversion.
B. `--dry-run` integration on dev box (real GDK + real WS, no servo) —
   check 30 s loop without errors.
C. Real-robot trials: speed-scale 0.3 → 0.5 → 1.0, idle → loaded.

## 9. Out of scope (not this change)

- Server-side changes (any).
- 100 Hz dual-thread control architecture.
- ROS2 integration.
- Dexterous hand (o10_t2 / o12_t2) support — gripper-only first.
- Multi-episode reset automation beyond returning to `reset_pose`.
