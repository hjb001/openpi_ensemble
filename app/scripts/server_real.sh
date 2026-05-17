#!/usr/bin/env bash
# Start the task-routed ensemble server for the real-robot 2-checkpoint setup
# (./checkpoints/real_base + ./checkpoints/desktop_sanitized_10000, per routes.json).
#
# Usage:
#   bash scripts/server_real.sh <gpu_id> <port>
#   bash scripts/server_real.sh 0 8999
#
# Run from the app/ directory (so the ./checkpoints and ./routes.json relative
# paths resolve correctly).
set -euo pipefail

cart_num=${1:-0}
port=${2:-8999}

export TF_NUM_INTRAOP_THREADS=16
export CUDA_VISIBLE_DEVICES=${cart_num}
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.7
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_ALLOCATOR=platform
export XLA_FLAGS="--xla_gpu_autotune_level=0"

export PYTHONPATH=/root/openpi/src:${PYTHONPATH:-/app:/app/src}

GIT_LFS_SKIP_SMUDGE=1 uv run python scripts/serve_policy.py \
  --env G2SIM \
  --port "${port}" \
  policy:task-routed-ensemble \
  --policy.routes-json ./routes.json \
  --policy.default.config acot_icra_simulation_challenge_reasoning_to_action \
  --policy.default.dir ./checkpoints/real_base
