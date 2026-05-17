"""Client launcher for the AgiBot G2 real-hardware env.

Loop is *chunk-level*, not step-level: each iteration runs one inference
(returning a 30-step chunk) and ships the whole chunk to
``execute_trajectory`` as a single async call. We then sleep ``hold_seconds``
before re-planning. Step-level slicing (ActionChunkBroker) does NOT work
here — see the docstring on ``AgibotRealEnvironment.apply_action``.
"""

import dataclasses
import logging
import time

import tyro
from openpi_client import websocket_client_policy as _websocket_client_policy

from examples.agibot_real import env as _env


@dataclasses.dataclass
class Args:
    # Server connection.
    host: str = "0.0.0.0"
    port: int = 8000

    # Inference inputs (the env forwards these to the policy server).
    prompt: str = ""
    # `task_name` is read by Policy.post_process to decide waist handling:
    #   "" / unset                              → keep full 21-D action
    #   "sorting_packages"[_continuous]         → freeze waist[0:4] to state
    #   anything else                           → truncate to 16-D (no waist)
    task_name: str = ""

    # Total time the controller takes to traverse one chunk.
    # Per-point interval = trajectory_reference_time / chunk_size.
    trajectory_reference_time: float = 1.0

    # Fraction of trajectory_reference_time to wait before re-planning.
    # <1.0 lets the next chunk overlap the tail of the current one (smooth
    # blending); =1.0 waits for the chunk to finish; >1.0 leaves the robot
    # idle between chunks.
    replan_fraction: float = 0.8

    # Stopping conditions.
    max_chunks: int = 0  # 0 = run forever (until Ctrl+C)
    max_seconds: float = 0.0  # 0 = no time limit


def main(args: Args) -> None:
    ws = _websocket_client_policy.WebsocketClientPolicy(host=args.host, port=args.port)
    logging.info(f"Server metadata: {ws.get_server_metadata()}")

    environment = _env.AgibotRealEnvironment(
        prompt=args.prompt,
        task_name=args.task_name,
        trajectory_reference_time=args.trajectory_reference_time,
    )

    hold = max(0.0, args.replan_fraction * args.trajectory_reference_time)
    started = time.time()
    chunk_idx = 0

    try:
        while True:
            if args.max_chunks and chunk_idx >= args.max_chunks:
                break
            if args.max_seconds and (time.time() - started) >= args.max_seconds:
                break

            obs = environment.get_observation()

            result = ws.infer(obs)
            actions = result["actions"]

            environment.execute_chunk(
                actions,
                trajectory_reference_time=args.trajectory_reference_time,
            )

            # execute_trajectory is async — sleep here while the controller
            # plays out the chunk. replan_fraction < 1.0 starts the next
            # inference slightly before the current chunk finishes so we get
            # a fresh observation for stitching.
            if hold > 0:
                time.sleep(hold)

            chunk_idx += 1
    except KeyboardInterrupt:
        logging.info("Interrupted; stopping.")
    finally:
        environment.close()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    tyro.cli(main)