"""Pytest configuration: make ``src`` importable (``from eval.reweight import ...``)."""

import sys
from pathlib import Path

SRC = str(Path(__file__).resolve().parents[1] / "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

import pytest


@pytest.fixture(autouse=True)
def _stub_val_purity(request, monkeypatch):
    """Synthetic train_dl runs have no real pixel map; val_purity_summary is covered in test_dl_val_constants."""
    if request.module.__name__.startswith("test_train_dl"):
        from astro_triage.models import dl_session

        monkeypatch.setattr(
            dl_session, "val_purity_summary", lambda o, p: {"n_val": 3, "purity": 0.9}
        )
