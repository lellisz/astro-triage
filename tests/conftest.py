"""Pytest configuration: make ``src`` importable (``from eval.reweight import ...``)."""

import sys
from pathlib import Path

SRC = str(Path(__file__).resolve().parents[1] / "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)
