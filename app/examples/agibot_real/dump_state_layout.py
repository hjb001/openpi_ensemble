"""One-shot helper: print joint-name order from a training lerobot dataset.

Run this once before deploying the real-robot client to confirm that
`constants.STATE_LAYOUT` matches the order the model was trained on. If the
dataset stores joint names in metadata, we print them; otherwise we print
the first sample's `observation.state` shape so you can verify the dim.

Usage:
    python -m examples.agibot_real.dump_state_layout \
        --repo-id=/mnt/public/E6/lerobot/8169/2026021101/gripper/task_6167
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-id", required=True, help="lerobot dataset path or repo id")
    parser.add_argument("--episode", type=int, default=0)
    args = parser.parse_args()

    try:
        # Lerobot is a heavy dep; only import on demand.
        from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
    except ImportError as exc:
        print(f"ERROR: lerobot is required: {exc}", file=sys.stderr)
        return 2

    ds = LeRobotDataset(args.repo_id)

    info_path = Path(args.repo_id) / "meta" / "info.json"
    if info_path.exists():
        info = json.loads(info_path.read_text())
        # lerobot v2 stores feature metadata under "features".
        feats = info.get("features", {})
        for key in ("observation.state", "action", "observation.images.top_head"):
            if key in feats:
                print(f"=== {key} ===")
                print(json.dumps(feats[key], indent=2))

    sample = ds[args.episode]
    print("\n=== sample 0 keys ===")
    for k, v in sample.items():
        try:
            shape = tuple(v.shape) if hasattr(v, "shape") else len(v)
        except TypeError:
            shape = "scalar"
        print(f"  {k}: shape={shape}, dtype={getattr(v, 'dtype', type(v).__name__)}")

    state = sample.get("observation.state")
    if state is not None:
        print(f"\nobservation.state[0]: {state.tolist() if hasattr(state, 'tolist') else state}")
        print("If the corresponding joint-name list differs from "
              "examples.agibot_real.constants.STATE_LAYOUT, update the constant "
              "before running the real-robot client.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
