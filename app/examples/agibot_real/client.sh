#!/usr/bin/env bash
# Start the AgiBot G2 real-hardware client (examples/agibot_real/main.py).
#
# IMPORTANT: source the GDK env FIRST in the shell you launch this from so the
# `agibot_gdk` Python module resolves:
#   source ~/.cache/agibot/app/env.sh
#
# Usage:
#   bash examples/agibot_real/client.sh <task_name> [host] [port] [prompt]
#   bash examples/agibot_real/client.sh sorting_packages_real
#   bash examples/agibot_real/client.sh clean_the_desktop_real 192.168.1.10 8999
#   bash examples/agibot_real/client.sh scoop_popcorn_real 127.0.0.1 8999 "scoop the popcorn"
set -euo pipefail

task_name=${1:?task_name required (e.g. sorting_packages_real, clean_the_desktop_real, stock_and_straighten_shelf_real, scoop_popcorn_real)}
host=${2:-127.0.0.1}
port=${3:-8999}
prompt=${4:-""}

# Resolve repo root (= the `app/` dir, parent of examples/).
APP_DIR="$(cd "$(dirname "$0")/../.." && pwd)"

# examples.agibot_real lives at app/examples/...; openpi_client lives at
# app/packages/openpi-client/src/openpi_client. Both need to be importable.
export PYTHONPATH="${APP_DIR}:${APP_DIR}/packages/openpi-client/src:${PYTHONPATH:-}"

cd "${APP_DIR}"
exec python -m examples.agibot_real.main \
  --host "${host}" \
  --port "${port}" \
  --task-name "${task_name}" \
  --prompt "${prompt}"
