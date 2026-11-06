"""build_freeze / planned_systems (ADR 0006 section 5, spec S9): refusals, schema, val-only invariant."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from astro_triage.evaluation import dl_val as V
from astro_triage.evaluation.runner import EvalError
from astro_triage.split import build_pixel_map

SHIFT = 2**49  # source_id // 2**35 // 4**7 == pixel
Y_VAL = ["STAR"] * 10 + ["GALAXY"] * 10 + ["QSO"] * 10
N_VAL, N_TEST = len(Y_VAL), 12
FRAC = {"STAR": 0.5, "GALAXY": 0.3, "QSO": 0.2}
APPROVAL = "owner approved the freeze on 2026-11-06"


def write_labels(root: Path) -> tuple[np.ndarray, np.ndarray]:
    pm = build_pixel_map()
    vp = pm[pm.fold == "val"].healpix_l5.to_numpy(dtype=np.int64)[:N_VAL]
    tp = pm[pm.fold == "test"].healpix_l5.to_numpy(dtype=np.int64)[:N_TEST]
    val_ids, test_ids = vp * SHIFT + np.arange(1, N_VAL + 1), tp * SHIFT + 7
    df = pd.DataFrame(
        {
            "source_id": np.concatenate([val_ids, test_ids]),
            "sdss_class": Y_VAL + ["STAR"] * N_TEST,
            "healpix_l5": np.concatenate([vp, tp]),
            "fold": ["val"] * N_VAL + ["test"] * N_TEST,
        }
    )
    root.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pandas(df, preserve_index=False), root / "labels.parquet")
    return val_ids, test_ids


def write_pred(path: Path, ids: np.ndarray, p: np.ndarray, frac: dict[str, float] = FRAC,
               comps: list[Path] | None = None) -> None:
    cols = {"source_id": ids.astype(np.int64)}
    cols.update({f"p_cal_{c}": p[:, i] for i, c in enumerate(("star", "galaxy", "qso"))})
    table = pa.Table.from_pandas(pd.DataFrame(cols).iloc[::-1], preserve_index=False)
    meta = {"train_class_fractions": json.dumps(frac)}
    if comps is not None:
        meta["component_sha256"] = json.dumps({c.name: V.sha256_file(c) for c in comps})
    pq.write_table(table.replace_schema_metadata(meta), path)


def val_p(i: int, val_seed: int) -> np.ndarray:
    return np.random.default_rng(1000 * i + val_seed).random((N_VAL, 3))


def row_p(i: int, test_seed: int | None) -> np.ndarray:
    if test_seed is None:
        return np.full((N_TEST, 3), np.nan)
    return np.random.default_rng(7000 + 13 * i + test_seed).random((N_TEST, 3))


def make_env(root: Path, val_seed=1, test_seed: int | None = 2, fracs: dict | None = None):
    """Writes labels, all 40 planned files and a matching delta.json; returns (out, dl, lg)."""
    out, dl, lg = root / "rel", root / "dl", root / "lg"
    root.mkdir(parents=True, exist_ok=True)
    dl.mkdir(exist_ok=True), lg.mkdir(exist_ok=True)
    val_ids, test_ids = write_labels(out)
    ids = np.concatenate([val_ids, test_ids])
    for i, s in enumerate(V.planned_systems()):
        d = lg if s.startswith("lgbm_") else dl
        p = np.vstack([val_p(i, val_seed), row_p(i, test_seed)])
        comps = None
        if s.startswith("ens_"):
            net, k = s.split("_")[1], s.rsplit("_s", 1)[1]
            comps = [lg / f"predictions_lgbm_full_s{k}_v0.parquet", dl / f"predictions_{net}_full_s{k}_v0.parquet"]
        write_pred(d / f"predictions_{s}_v0.parquet", ids, p, (fracs or {}).get(s, FRAC), comps)
    shas = {
        fs: {f"lgbm_{fs}_s{k}": V.sha256_file(lg / f"predictions_lgbm_{fs}_s{k}_v0.parquet") for k in range(5)}
        for fs in ("full", "noind")
    }
    (lg / "delta.json").write_text(json.dumps({"prediction_sha256": shas}), encoding="utf-8")
    return out, dl, lg


def keys(o):
    if isinstance(o, dict):
        for k, v in o.items():
            yield str(k)
            yield from keys(v)
    elif isinstance(o, list):
        for v in o:
            yield from keys(v)


def test_planned_systems_are_40_unique_ids():
    got = V.planned_systems()
    want = {f"{b}_s{k}" for b in ("lgbm_full", "lgbm_noind", "mlp_full", "mlp_noind", "ftt_full",
                                  "ftt_noind", "ens_mlp_full", "ens_ftt_full") for k in range(5)}
    assert len(got) == 40 and set(got) == want


def test_empty_approval_refused(tmp_path):
    out, dl, lg = make_env(tmp_path)
    for text in ("", "  \n\t"):
        with pytest.raises(EvalError, match="approval"):
            V.build_freeze(out, dl, lg, text)


def test_missing_planned_files_listed(tmp_path):
    out, dl, lg = make_env(tmp_path)
    a, b = dl / "predictions_ftt_noind_s2_v0.parquet", lg / "predictions_lgbm_full_s4_v0.parquet"
    a.unlink(), b.unlink()
    with pytest.raises(EvalError) as ei:
        V.build_freeze(out, dl, lg, APPROVAL)
    assert "2 planned" in str(ei.value) and a.name in str(ei.value) and b.name in str(ei.value)


def test_missing_delta_refused(tmp_path):
    out, dl, lg = make_env(tmp_path)
    (lg / "delta.json").unlink()
    with pytest.raises(EvalError, match="delta.json"):
        V.build_freeze(out, dl, lg, APPROVAL)


def test_seed_refitted_after_delta_refused(tmp_path):
    out, dl, lg = make_env(tmp_path)
    ids = np.concatenate(write_labels(out))
    write_pred(lg / "predictions_lgbm_full_s2_v0.parquet", ids, np.random.default_rng(99).random((ids.size, 3)))
    with pytest.raises(EvalError, match="sha mismatch") as ei:
        V.build_freeze(out, dl, lg, APPROVAL)
    assert "lgbm_full_s2" in str(ei.value)


@pytest.mark.parametrize("bad", ["mlp_noind_s3", "ens_ftt_full_s1", "lgbm_full_s0"])
def test_fraction_mismatch_refused(tmp_path, bad):
    other = {"STAR": 0.5, "GALAXY": 0.3 + 1e-8, "QSO": 0.2 - 1e-8}
    out, dl, lg = make_env(tmp_path, fracs={bad: other})
    with pytest.raises(EvalError, match="train_class_fractions"):
        V.build_freeze(out, dl, lg, APPROVAL)


def test_fraction_within_tolerance_accepted(tmp_path):
    near = {"STAR": 0.5, "GALAXY": 0.3 + 5e-10, "QSO": 0.2}
    out, dl, lg = make_env(tmp_path, fracs={"ens_mlp_full_s4": near})
    assert V.build_freeze(out, dl, lg, APPROVAL)["schema_version"] == 1


def kth(scores: np.ndarray, n_keep: int) -> float:
    return float(np.sort(scores)[::-1][n_keep - 1])


def test_document_schema_and_thresholds(tmp_path):
    out, dl, lg = make_env(tmp_path)
    doc = V.build_freeze(out, dl, lg, f"  {APPROVAL}  ")
    assert doc["schema_name"] == "astro_triage.dl_val_freeze" and doc["schema_version"] == 1
    assert doc["approval"] == APPROVAL
    assert doc["labels_sha256"] == V.sha256_file(out / "labels.parquet")
    assert doc["delta_sha256"] == V.sha256_file(lg / "delta.json")
    systems = V.planned_systems()
    assert set(doc["prediction_sha256"]) == set(doc["thresholds"]) == set(systems)
    for i, s in enumerate(systems):
        d = lg if s.startswith("lgbm_") else dl
        assert doc["prediction_sha256"][s] == V.sha256_file(d / f"predictions_{s}_v0.parquet")
        p = val_p(i, 1)
        for cls, col, lo in (("QSO", 2, 20), ("GALAXY", 1, 10)):
            for c_star, k in (("0.80", 8), ("0.90", 9)):  # 10 true rows: k = ceil(C* * 10)
                got = doc["thresholds"][s][cls][c_star]
                assert isinstance(got, float) and got == kth(p[lo : lo + 10, col], k)
    assert doc["constants"]["n_seeds"] == 5 and doc["constants"]["primary_prior"] == "P2"
    assert doc["constants"]["n_boot"] == 1000 and doc["constants"]["rng_seed"] == 20261025
    assert not any("test" in k.lower() for k in keys(doc))


def stable(doc):
    """Everything except the file hashes (which necessarily cover test rows)."""
    return {k: v for k, v in doc.items() if k not in ("prediction_sha256", "delta_sha256")}


def test_deterministic_and_test_rows_do_not_leak(tmp_path):
    a = V.build_freeze(*make_env(tmp_path / "a"), APPROVAL)
    assert a == V.build_freeze(*make_env(tmp_path / "a2"), APPROVAL)
    for ts in (3, None):  # other test probabilities, then NaN test probabilities
        b = V.build_freeze(*make_env(tmp_path / f"t{ts}", test_seed=ts), APPROVAL)
        assert stable(b) == stable(a) and b["prediction_sha256"] != a["prediction_sha256"]
    c = V.build_freeze(*make_env(tmp_path / "v", val_seed=2), APPROVAL)
    assert c["thresholds"] != a["thresholds"]
    assert all(math.isfinite(v) for t in c["thresholds"].values() for q in t.values() for v in q.values())
