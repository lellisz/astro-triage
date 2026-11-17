"""Cached wrappers of the heavy aggregates; same names and signatures as in ``data``.

Each result is cached per release name plus a stamp of the artifacts it depends on (their
modification times), so a rewritten ``results.json``, ``delta.json`` or ``eval_l2.json`` drops
the stale entry. The ``Release`` itself is passed as ``_rel`` and never hashed.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

from astro_triage.dashboard import data as D
from astro_triage.dashboard import data_dl

_ARTIFACTS = (
    "eval/results.json",
    "train_report.json",
    "lgbm_seeds/delta.json",
    "layer2/eval_l2.json",
    "dl/eval/results_dl.json",
    "dl/eval/risk_coverage_dl.csv",
)
_MAX = 16


def root_stamp(root: Path) -> int:
    """Newest modification time (ns) among the artifacts under ``root``; 0 when none."""
    times = [(root / p).stat().st_mtime_ns for p in _ARTIFACTS if (root / p).is_file()]
    return max(times, default=0)


def stamp(rel: D.Release) -> int:
    """Newest modification time (ns) among the artifacts the aggregates read; 0 when none."""
    return root_stamp(rel.root)


@st.cache_data(show_spinner=False, max_entries=_MAX)
def _headline(rel_name: str, rel_stamp: int, model: str, _rel: D.Release) -> D.Headline:
    return D.headline(_rel, model)


def headline(rel: D.Release, model: str = "lgbm_full") -> D.Headline:
    return _headline(rel.name, stamp(rel), model, rel)


@st.cache_data(show_spinner=False, max_entries=_MAX)
def _purity_points(
    rel_name: str, rel_stamp: int, model: str, prior: str, _rel: D.Release
) -> list[dict]:
    return D.purity_points(_rel, model, prior)


def purity_points(
    rel: D.Release, model: str = "lgbm_full", prior: str = "P2"
) -> list[dict]:
    return _purity_points(rel.name, stamp(rel), model, prior, rel)


@st.cache_data(show_spinner=False, max_entries=_MAX)
def _abstention_curve(
    rel_name: str,
    rel_stamp: int,
    model: str,
    thresholds: np.ndarray | None,
    _rel: D.Release,
) -> pd.DataFrame:
    return D.abstention_curve(_rel, model, thresholds)


def abstention_curve(
    rel: D.Release, model: str = "lgbm_full", thresholds: np.ndarray | None = None
) -> pd.DataFrame:
    return _abstention_curve(rel.name, stamp(rel), model, thresholds, rel)


@st.cache_data(show_spinner=False, max_entries=_MAX)
def _calibration_points(
    rel_name: str, rel_stamp: int, model: str, n_bins: int, _rel: D.Release
) -> dict[str, pd.DataFrame]:
    return D.calibration_points(_rel, model, n_bins)


def calibration_points(
    rel: D.Release, model: str = "lgbm_full", n_bins: int = 10
) -> dict[str, pd.DataFrame]:
    return _calibration_points(rel.name, stamp(rel), model, n_bins, rel)


@st.cache_data(show_spinner=False, max_entries=_MAX)
def _error_rate_table(
    rel_name: str,
    rel_stamp: int,
    model: str,
    true_class: str,
    by: str,
    edges: tuple[float, ...] | None,
    _rel: D.Release,
) -> pd.DataFrame:
    return D.error_rate_table(_rel, model, true_class, by, edges)


def error_rate_table(
    rel: D.Release,
    model: str,
    true_class: str,
    by: str,
    edges: tuple[float, ...] | None = None,
) -> pd.DataFrame:
    return _error_rate_table(rel.name, stamp(rel), model, true_class, by, edges, rel)


@st.cache_data(show_spinner=False, max_entries=_MAX)
def _error_rate_table_z(
    rel_name: str, rel_stamp: int, model: str, _rel: D.Release
) -> pd.DataFrame:
    return D.error_rate_table_z(_rel, model)


def error_rate_table_z(rel: D.Release, model: str) -> pd.DataFrame:
    return _error_rate_table_z(rel.name, stamp(rel), model, rel)


@st.cache_data(show_spinner=False, max_entries=_MAX)
def _galaxy_totals(
    rel_name: str, rel_stamp: int, model: str, _rel: D.Release
) -> D.GalaxyTotals:
    return D.galaxy_totals(_rel, model)


def galaxy_totals(rel: D.Release, model: str) -> D.GalaxyTotals:
    return _galaxy_totals(rel.name, stamp(rel), model, rel)


@st.cache_data(show_spinner=False, max_entries=_MAX)
def _z_low_summary(
    rel_name: str, rel_stamp: int, model: str, _rel: D.Release
) -> D.ZLowSummary:
    return D.z_low_summary(_rel, model)


def z_low_summary(rel: D.Release, model: str) -> D.ZLowSummary:
    return _z_low_summary(rel.name, stamp(rel), model, rel)


@st.cache_data(show_spinner=False, max_entries=_MAX)
def _galaxy_scatter(
    rel_name: str, rel_stamp: int, model: str, _rel: D.Release
) -> pd.DataFrame:
    return D.galaxy_scatter(_rel, model)


def galaxy_scatter(rel: D.Release, model: str = "lgbm_full") -> pd.DataFrame:
    return _galaxy_scatter(rel.name, stamp(rel), model, rel)


@st.cache_data(show_spinner=False, max_entries=_MAX)
def _load_delta(rel_name: str, rel_stamp: int, _rel: D.Release) -> dict | None:
    return D.load_delta(_rel)


def load_delta(rel: D.Release) -> dict | None:
    return _load_delta(rel.name, stamp(rel), rel)


@st.cache_data(show_spinner=False, max_entries=_MAX)
def _load_layer2(rel_name: str, rel_stamp: int, _rel: D.Release) -> dict | None:
    return D.load_layer2(_rel)


def load_layer2(rel: D.Release) -> dict | None:
    return _load_layer2(rel.name, stamp(rel), rel)


@st.cache_data(show_spinner=False, max_entries=_MAX)
def _load_results_dl(rel_name: str, rel_stamp: int, _rel: D.Release) -> dict | None:
    return data_dl.load_results_dl(_rel)


def load_results_dl(rel: D.Release) -> dict | None:
    return _load_results_dl(rel.name, stamp(rel), rel)


@st.cache_data(show_spinner=False, max_entries=_MAX)
def _load_risk_coverage_dl(
    rel_name: str, rel_stamp: int, _rel: D.Release
) -> pd.DataFrame | None:
    return data_dl.load_risk_coverage_dl(_rel)


def load_risk_coverage_dl(rel: D.Release) -> pd.DataFrame | None:
    return _load_risk_coverage_dl(rel.name, stamp(rel), rel)


@st.cache_data(show_spinner=False, max_entries=1)
def _test_pixel_stats(
    rel_name: str, rel_stamp: int, model: str, _rel: D.Release, _pixels: pd.DataFrame
) -> pd.DataFrame:
    return D.test_pixel_stats(_rel, model, _pixels)


def test_pixel_stats(rel: D.Release, model: str, pixels: pd.DataFrame) -> pd.DataFrame:
    return _test_pixel_stats(rel.name, stamp(rel), model, rel, pixels)
