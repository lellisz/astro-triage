"""Test side of the DL comparison, part 1: freeze gate and system loading (ADR 0006 section 5.2).

Nothing here reads a test label or computes a test metric before ``val_freeze.json`` exists and
validates (labels, delta and every prediction file hash). ``open_test_set`` is the only place that
calls ``runner.load_test_set``: it needs a non-empty owner approval and runs once per process.
The only write is the test-opened marker (``write_marker``), which holds no metrics.
"""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Final

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from astro_triage.evaluation.dl_compare import (
    COMPLETENESS_LEVELS,
    FEATURE_SETS,
    N_SEEDS,
    PREDICTION_VERSION,
    PRIMARY_PRIOR,
    TARGET_CLASSES,
)
from astro_triage.evaluation.dl_val import (
    FRACTION_TOL,
    current_constants,
    lgbm_reference,
    load_val_labels,
    planned_systems,
    sha256_file,
)
from astro_triage.evaluation.runner import (
    SUM_TOLERANCE,
    EvalError,
    ModelData,
    TestSet,
    load_test_set,
)
from astro_triage.models.lgbm import _atomic
from eval import metrics as M

FREEZE_NAME: Final = "val_freeze.json"
FREEZE_SCHEMA: Final = "astro_triage.dl_val_freeze"
MARKER_NAME: Final = "test_opened.json"
MARKER_SCHEMA: Final = "astro_triage.dl_test_opened"
_TEST_OPENED = {"done": False}


def refuse(msg: str) -> EvalError:
    return EvalError("dl evaluation refused: " + " ".join(msg.split()))


def _paths(dl_dir: Path, lgbm_dir: Path) -> dict[str, Path]:
    return {
        s: (lgbm_dir if s.startswith("lgbm_") else dl_dir)
        / f"predictions_{s}_{PREDICTION_VERSION}.parquet"
        for s in planned_systems()
    }


def load_freeze(dl_dir: Path) -> dict[str, object]:
    if "pilot" in (part.lower() for part in dl_dir.resolve().parts):
        raise refuse(f"{dl_dir} is a pilot directory; pilot runs are never evaluated")
    path = dl_dir / FREEZE_NAME
    if not path.is_file():
        raise refuse(f"{FREEZE_NAME} missing in {dl_dir}")
    try:
        raw = path.read_bytes()
        doc = json.loads(raw.decode("utf-8"))
    except (OSError, ValueError) as exc:
        raise refuse(f"cannot read {path}: {exc}") from exc
    if (
        not isinstance(doc, dict)
        or doc.get("schema_name") != FREEZE_SCHEMA
        or doc.get("schema_version") != 1
    ):
        raise refuse(f"{path} is not a {FREEZE_SCHEMA} v1 document")
    if not isinstance(doc.get("approval"), str) or not doc["approval"].strip():
        raise refuse(f"{path} has no approval text")
    for key in ("prediction_sha256", "train_class_fractions"):
        if not isinstance(doc.get(key), dict):
            raise refuse(f"{path}: {key} is not an object")
    doc["_file_sha256"] = hashlib.sha256(raw).hexdigest()
    return doc


def verify_freeze(outdir: Path, dl_dir: Path, lgbm_dir: Path) -> dict[str, object]:
    """Load the freeze and check labels, delta and all planned prediction files by sha256."""
    freeze = load_freeze(dl_dir)
    frozen_consts = freeze.get("constants")
    current = json.loads(json.dumps(current_constants()))
    if not isinstance(frozen_consts, dict):
        frozen_consts = {}
    differ = sorted(
        k for k in set(current) | set(frozen_consts) if frozen_consts.get(k) != current.get(k)
    )
    if differ:
        raise refuse(f"freeze constants differ from current code: {', '.join(differ)}")
    paths = _paths(dl_dir, lgbm_dir)
    missing = [p.name for p in paths.values() if not p.is_file()]
    if missing:
        raise refuse(f"{len(missing)} planned files missing: {missing}")
    delta = lgbm_dir / "delta.json"
    pairs = [
        ("labels", outdir / "labels.parquet", freeze.get("labels_sha256")),
        ("delta.json", delta, freeze.get("delta_sha256")),
    ]
    for label, path, expected in pairs:
        if not path.is_file() or sha256_file(path) != expected:
            raise refuse(f"{label} sha256 differs from the freeze ({path})")
    frozen = freeze["prediction_sha256"]
    assert isinstance(frozen, dict)
    if set(frozen) != set(paths):
        raise refuse("freeze does not list exactly the planned systems")
    bad = [s for s, p in paths.items() if sha256_file(p) != frozen[s]]
    if bad:
        raise refuse(f"sha256 differs from the freeze for {bad}")
    _check_thresholds(freeze, delta)
    return freeze


def _check_thresholds(freeze: dict[str, object], delta_path: Path) -> None:
    """Freeze thresholds: exact structure, finite values in (0, 1], LightGBM ones equal delta.json's."""
    thr = freeze.get("thresholds")
    if not isinstance(thr, dict):
        raise refuse("freeze thresholds is not an object")
    systems = planned_systems()
    if set(thr) != set(systems):
        raise refuse("freeze thresholds do not cover exactly the planned systems")
    cstars = tuple(f"{c:.2f}" for c in COMPLETENESS_LEVELS)
    for s in systems:
        per_class = thr[s]
        if not isinstance(per_class, dict) or set(per_class) != set(TARGET_CLASSES):
            raise refuse(f"freeze thresholds {s}: classes are not {list(TARGET_CLASSES)}")
        for cls in TARGET_CLASSES:
            per_c = per_class[cls]
            if not isinstance(per_c, dict) or set(per_c) != set(cstars):
                raise refuse(f"freeze thresholds {s}/{cls}: levels are not {list(cstars)}")
            for c in cstars:
                v = per_c[c]
                ok = isinstance(v, float | int) and not isinstance(v, bool)
                if not ok or not np.isfinite(v) or not 0.0 < v <= 1.0:
                    raise refuse(f"freeze thresholds {s}/{cls}/{c} is not a float in (0, 1]: {v!r}")
    try:
        cells = json.loads(delta_path.read_text(encoding="utf-8"))["delta"]
        for fs in FEATURE_SETS:
            for cls in TARGET_CLASSES:
                for c in cstars:
                    ref = cells[fs][cls][c][PRIMARY_PRIOR]["thresholds"]
                    for k in range(N_SEEDS):
                        if thr[f"lgbm_{fs}_s{k}"][cls][c] != ref[k]:
                            raise refuse(
                                f"freeze thresholds lgbm_{fs}_s{k}/{cls}/{c} differ from delta.json"
                            )
    except (OSError, KeyError, IndexError, ValueError, TypeError) as exc:
        raise refuse(f"cannot compare freeze thresholds with {delta_path.name}: {exc!r}") from exc


def open_test_set(
    outdir: Path, dl_dir: Path, lgbm_dir: Path, owner_approval: str
) -> tuple[TestSet, dict[str, object]]:
    """The single test opening: freeze verified first, owner approval required, once per process."""
    if not owner_approval.strip():
        raise refuse("test opening needs the owner's explicit approval text (A6)")
    if (dl_dir / MARKER_NAME).exists():
        raise refuse("test already opened")
    freeze = verify_freeze(outdir, dl_dir, lgbm_dir)
    if _TEST_OPENED["done"]:
        raise refuse("test set was already opened in this process; it is opened once")
    _TEST_OPENED["done"] = True
    return load_test_set(outdir), freeze


def write_marker(dl_dir: Path, *, freeze_sha: str, git_sha: str) -> Path:
    """Record that the test set was opened (no metrics); refuses to overwrite an existing marker."""
    path = dl_dir / MARKER_NAME
    if path.exists():
        raise refuse(f"{MARKER_NAME} already exists in {dl_dir}")
    doc = {
        "schema_name": MARKER_SCHEMA,
        "schema_version": 1,
        "opened_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "git_sha": git_sha,
        "freeze_sha256": freeze_sha,
    }
    text = json.dumps(doc, indent=2, sort_keys=True) + "\n"
    _atomic(path, lambda tmp: Path(tmp).write_text(text, encoding="utf-8"))
    return path


def _read_proba(path: Path, expected_ids: np.ndarray, test_ids: np.ndarray) -> np.ndarray:
    cols = ["source_id", *(f"p_cal_{c.lower()}" for c in M.CLASSES)]
    try:
        pred = pq.read_table(path, columns=cols).to_pandas()
    except (OSError, KeyError, ValueError, pa.ArrowException) as exc:
        raise refuse(f"cannot read {path.name}: {exc}") from exc
    ids = pred["source_id"].to_numpy(dtype=np.int64)
    if not np.array_equal(np.sort(ids), expected_ids):
        raise refuse(f"{path.name} does not cover exactly the val + test ids")
    aligned: pd.DataFrame = pred.set_index("source_id").loc[test_ids]
    p = aligned[cols[1:]].to_numpy(dtype=np.float64)
    if not np.isfinite(p).all() or np.abs(p.sum(axis=1) - 1.0).max() > SUM_TOLERANCE:
        raise refuse(f"{path.name}: p_cal is not finite or does not sum to 1")
    return p


def load_dl_systems(
    outdir: Path, ts: TestSet, *, dl_dir: Path, lgbm_dir: Path
) -> dict[str, ModelData]:
    """The 40 planned systems aligned on ``ts.source_id``; the freeze is verified before any read."""
    freeze = verify_freeze(outdir, dl_dir, lgbm_dir)
    fractions = freeze["train_class_fractions"]
    assert isinstance(fractions, dict)
    val_ids = load_val_labels(outdir)["source_id"].to_numpy(dtype=np.int64)
    expected = np.union1d(val_ids, ts.source_id)
    if expected.size != val_ids.size + ts.source_id.size:
        raise refuse("val and test ids overlap")
    out: dict[str, ModelData] = {}
    for name, path in _paths(dl_dir, lgbm_dir).items():
        ref = lgbm_reference(name)
        try:
            pi = {c: float(fractions[name][c]) for c in M.CLASSES}
            pi_ref = {c: float(fractions[ref][c]) for c in M.CLASSES}
        except (KeyError, TypeError, ValueError) as exc:
            raise refuse(f"freeze has no train_class_fractions for {name}") from exc
        if any(abs(pi[c] - pi_ref[c]) > FRACTION_TOL for c in M.CLASSES):
            raise refuse(f"train_class_fractions of {name} differ from {ref}")
        proba = _read_proba(path, expected, ts.source_id)
        out[name] = ModelData(name, proba, np.ones(len(proba), dtype=bool), pi)
    return out
