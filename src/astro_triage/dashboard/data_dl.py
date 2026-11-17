"""Loader of the deep-learning evaluation (``dl/eval/results_dl.json``).

No Streamlit import. Hashes are dropped on load; nothing here is per object or a path.
"""

from __future__ import annotations

import json

import pandas as pd

from astro_triage.dashboard.data_headline import _strip_sha256
from astro_triage.dashboard.data_release import Release

_EVAL_DIR = ("dl", "eval")


def load_results_dl(rel: Release) -> dict | None:
    """Parsed ``dl/eval/results_dl.json`` without any ``*sha256`` key; None when absent."""
    path = rel.root.joinpath(*_EVAL_DIR, "results_dl.json")
    if not path.is_file():
        return None
    return _strip_sha256(json.loads(path.read_text(encoding="utf-8")))


def load_risk_coverage_dl(rel: Release) -> pd.DataFrame | None:
    """``risk_coverage_dl.csv`` (system, seed, tau, coverage, accuracy); None when absent."""
    path = rel.root.joinpath(*_EVAL_DIR, "risk_coverage_dl.csv")
    if not path.is_file():
        return None
    return pd.read_csv(path)
