"""pytest configuration: make `examples.agibot_real` importable when running
from the `app/` directory and ensure the fake gdk is available."""

from __future__ import annotations

import os
import sys
from pathlib import Path

# Add app/ to sys.path so `from examples.agibot_real import ...` works.
APP_DIR = Path(__file__).resolve().parents[3]
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))
