"""Shared fixture helper: LightGBM seed prediction files carrying train_class_fractions (spec S6/S10 pi_src check)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

CLASSES = ("STAR", "GALAXY", "QSO")
FRACS = {"STAR": 0.5, "GALAXY": 0.3, "QSO": 0.2}


def fractions_of(labels: np.ndarray) -> dict[str, float]:
    counts = np.bincount(np.asarray(labels, dtype=np.int64), minlength=3)
    return {c: float(counts[i] / counts.sum()) for i, c in enumerate(CLASSES)}


def write_lgbm_ref(outdir: Path, feature_set: str, k: int, fractions: dict[str, float]) -> Path:
    path = outdir / "lgbm_seeds" / f"predictions_lgbm_{feature_set}_s{k}_v0.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    table = pa.table({"source_id": pa.array([1], pa.int64())})
    pq.write_table(table.replace_schema_metadata({"train_class_fractions": json.dumps(fractions)}), path)
    return path
