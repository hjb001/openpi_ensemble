"""Agibot G02 real-robot client entry point.

Connects to a running `serve_policy.py --env=g2sim ...` server over WebSocket,
fetches action chunks, and dispatches them via `agibot_gdk` to the physical
robot. The schema sent to the server matches `LerobotACOTGo2DataConfig`
exactly (top_head / hand_left / hand_right images + 32-dim state + prompt).

Typical use:

    python -m examples.agibot_real.main --host=10.42.1.102 --port=8000 \
        --prompt="Pour the workpiece into the box" --speed-scale=0.3

First time on hardware: keep `--dry-run` to verify the pipeline without
moving the robot, then drop it.
"""

from __future__ import annotations

import dataclasses
import logging
import sys

from openpi_client import action_chunk_broker
from openpi_client import websocket_client_policy as _websocket_client_policy
from openpi_client.runtime import runtime as _runtime
from openpi_client.runtime.agents import policy_agent as _policy_agent
import tyro

from examples.agibot_real import env as _env
from examples.agibot_real import constants


@dataclasses.dataclass
class Args:
    # WS server (where serve_policy.py is running).
    host: str = "127.0.0.1"
    port: int = 8000

    # Task prompt; default matches a training task in the ICRA challenge set.
    prompt: str = "Pour the workpiece into the box"

    # Action chunk slicing (must be <= server chunk_horizon, default 30).
    action_horizon: int = 25

    # Episode bounds.
    num_episodes: int = 1
    max_episode_steps: int = 1000

    # Image resolution to send to the server.
    render_height: int = constants.TARGET_HW[0]
    render_width: int = constants.TARGET_HW[1]

    # Initial pose (16 floats: ARM_L 7 + ARM_R 7 + GRIPPER_L + GRIPPER_R).
    # If None, reset() is a no-op.
    reset_pose: list[float] | None = None

    # Per-camera frame deadline.
    image_timeout_ms: float = 200.0

    # Safety knobs.
    speed_scale: float = 1.0
    dry_run: bool = False

    # Control loop frequency. GDK doc recommends 100 Hz for servo, but at the
    # client side we drive policy infer + dispatch at this rate; GDK
    # interpolates internally. Bump to 100 Hz only after deploying the
    # repeater-thread upgrade.
    max_hz: float = 30.0


def main(args: Args) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        force=True,
    )
    log = logging.getLogger("agibot_real.main")

    # Lazy import so unit tests / --help don't need the real GDK installed.
    try:
        import agibot_gdk
    except ImportError as exc:
        log.error(
            "agibot_gdk is not installed. Source the GDK env first: "
            "`source ~/.cache/agibot/app/env.sh` and pip install the wheel. "
            "Original: %s", exc,
        )
        sys.exit(2)

    ws_client_policy = _websocket_client_policy.WebsocketClientPolicy(
        host=args.host, port=args.port,
    )
    log.info("server metadata: %s", ws_client_policy.get_server_metadata())

    environment = _env.AgibotRealEnvironment(
        prompt=args.prompt,
        gdk=agibot_gdk,
        render_hw=(args.render_height, args.render_width),
        reset_pose=args.reset_pose,
        image_timeout_ms=args.image_timeout_ms,
        speed_scale=args.speed_scale,
        dry_run=args.dry_run,
    )

    try:
        runtime = _runtime.Runtime(
            environment=environment,
            agent=_policy_agent.PolicyAgent(
                policy=action_chunk_broker.ActionChunkBroker(
                    policy=ws_client_policy,
                    action_horizon=args.action_horizon,
                ),
            ),
            subscribers=[],
            max_hz=args.max_hz,
            num_episodes=args.num_episodes,
            max_episode_steps=args.max_episode_steps,
        )
        runtime.run()
    finally:
        environment.close()


if __name__ == "__main__":
    tyro.cli(main)
