"""Shared fixtures for the dl_eval tests (S11): a frozen synthetic environment, no real test data."""

from __future__ import annotations

import json
from pathlib import Path

from test_dl_val_freeze import APPROVAL, make_env

from astro_triage.evaluation import dl_eval as E
from astro_triage.evaluation import dl_val as V


def write_freeze(dl: Path, doc: dict) -> Path:
    path = dl / E.FREEZE_NAME
    path.write_text(json.dumps(doc), encoding="utf-8")
    return path


def frozen_env(root: Path):
    """(out, dl, lg, doc): 40 synthetic systems, delta.json and a freeze written by build_freeze."""
    out, dl, lg = make_env(root)
    doc = V.build_freeze(out, dl, lg, APPROVAL)
    write_freeze(dl, doc)
    return out, dl, lg, doc


def read_freeze(dl: Path) -> dict:
    return json.loads((dl / E.FREEZE_NAME).read_text(encoding="utf-8"))


def edit_freeze(dl: Path, fn) -> None:
    doc = read_freeze(dl)
    fn(doc)
    write_freeze(dl, doc)
