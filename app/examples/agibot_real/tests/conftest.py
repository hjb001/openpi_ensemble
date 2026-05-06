"""pytest configuration: make `examples.agibot_real` importable when running
from the `app/` directory, plus pre-pend the workspace `openpi_client` source
so tests do not require `pip install -e packages/openpi-client`."""

from __future__ import annotations

import sys
from pathlib import Path

# Add app/ to sys.path so `from examples.agibot_real import ...` works.
APP_DIR = Path(__file__).resolve().parents[3]
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

# Add the workspace openpi-client source so the env.py import chain (which
# uses openpi_client.image_tools / runtime.environment) does not require
# installing the package separately.
OPENPI_CLIENT_SRC = APP_DIR / "packages" / "openpi-client" / "src"
if OPENPI_CLIENT_SRC.exists() and str(OPENPI_CLIENT_SRC) not in sys.path:
    sys.path.insert(0, str(OPENPI_CLIENT_SRC))
