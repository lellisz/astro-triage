"""Release loading pieces: constants, the ``Release`` container, path helpers and Wilson intervals."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import pandas as pd

REPO_ROOT: Final = Path(__file__).resolve().parents[3]
INTERIM_DIR: Final = REPO_ROOT / "data" / "interim"
DEFAULT_RELEASE: Final = "dataset_100k_v0"

CLASSES: Final[tuple[str, ...]] = ("STAR", "GALAXY", "QSO")
MODELS: Final[tuple[str, ...]] = ("lgbm_full", "lgbm_noind")
SCORE_COLUMN: Final = {"STAR": "star", "GALAXY": "galaxy", "QSO": "qso"}

SEED: Final = 20261025
PRIOR_REGIMES: Final = ("P0", "P1", "P2", "P3")
POPULATIONS: Final = ("a_all", "c_sdsc", "a_combmod")
VARIANTS: Final = ("prior_matched", "as_shipped")


@dataclass(frozen=True)
class Release:
    """Everything loaded from one ``data/interim/<name>/`` directory."""

    name: str
    root: Path
    results: dict
    train_report: dict
    pixel_counts: pd.DataFrame  # all folds: healpix_l5, fold, n (objects)
    test: pd.DataFrame  # test fold, joined with features and predictions
    n_unscored: int  # test rows without a calibrated score (dropped)

    @property
    def meta(self) -> dict:
        return self.results["meta"]

    @property
    def is_provisional(self) -> bool:
        return "PROVISIONAL" in str(self.meta.get("header", "")).upper()


def strip_sha256(node: object) -> object:
    """Copy of a parsed JSON tree without any key that contains ``sha256``."""
    if isinstance(node, dict):
        return {k: strip_sha256(v) for k, v in node.items() if "sha256" not in k}
    if isinstance(node, list):
        return [strip_sha256(v) for v in node]
    return node


def single_file(root: Path, pattern: str) -> Path:
    files = sorted(root.glob(pattern))
    if len(files) != 1:
        raise FileNotFoundError(
            f"expected exactly one {pattern} in {root}, found {len(files)}"
        )
    return files[0]


def classes_upper(series: pd.Series) -> pd.Series:
    return series.astype(str).str.upper()


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval for a proportion; (nan, nan) when n is 0."""
    if n <= 0:
        return (math.nan, math.nan)
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))
