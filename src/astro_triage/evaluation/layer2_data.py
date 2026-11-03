"""Load, validate and fit for the Layer 2 evaluation (ADR 0005 D7, D4.4-D4.5).

Everything fitted here (verbal calibration, tau2, tau*) uses the val fold only; the test fold
is read for evaluation. The evaluation is refused, never partial, when a routed object has no
terminal record or when the routing manifest does not describe the predictions on disk.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Final

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from astro_triage.abstention.policy import (
    ABSTAIN,
    CLASSES,
    apply_verbal_calibration,
    choose_threshold,
    combine,
    fit_verbal_calibration,
    l1_confidence,
    score_columns,
    sha256_file,
)
from astro_triage.evaluation import layer2 as L2
from astro_triage.evaluation.runner import G_BINS, EvalError
from astro_triage.llm.consolidate import output_path
from astro_triage.llm.prompt import PROMPT_VERSION
from astro_triage.llm.routing import MANIFEST_NAME, select_routed
from astro_triage.split import build_pixel_map, pixel_l5
from eval import metrics as M

R_MAX_TAU2: Final = 1.0  # ADR D4.5 gives no routed-fraction cap for tau2 (A_keep2 only)
A_KEEP2_DEFAULT: Final = 0.90
TAU2_FITTED: Final = "fitted"
TAU2_INFEASIBLE: Final = "infeasible: no grid value reaches A_keep2"
TAU2_UNCALIBRATED: Final = "not_applied: uncalibrated (fewer than 100 OK val answers)"
FORCED_REASON_INFEASIBLE: Final = (
    "tau2 infeasible on val: Layer 2 answers are never trusted, "
    "S3 = S1 and S2 = S0 by construction"
)
FAILURE_EXCLUDED: Final = frozenset({"OK", "UNKNOWN"})


@dataclass(frozen=True)
class FoldData:
    ids: L2.IntArr
    y: L2.StrArr
    pixel: L2.IntArr
    l1_label: L2.StrArr
    l1_conf: L2.FloatArr
    routed: pd.DataFrame  # select_routed output: ids, p_cal_*, l1_conf, features
    l2: pd.DataFrame  # Layer 2 parquet rows, sorted by source_id
    meta: dict[str, str]
    parquet_sha256: str = ""


def _read_meta(path: Path) -> dict[str, str]:
    raw = pq.read_schema(path).metadata or {}
    return {k.decode(): v.decode() for k, v in raw.items() if k != b"pandas"}


def load_fold(
    release_dir: Path,
    fold: str,
    manifest: Path,
    model: str,
    arm: str,
    enable_thinking: bool,
) -> FoldData:
    routed = select_routed(release_dir, fold, manifest, None)  # checks predictions sha
    pred_name = json.loads(manifest.read_text(encoding="utf-8"))["predictions_file"]
    labels = pd.read_parquet(
        release_dir / "labels.parquet",
        columns=["source_id", "sdss_class", "healpix_l5"],
        filters=[("fold", "==", fold)],
    ).sort_values("source_id", ignore_index=True)
    pred = pd.read_parquet(
        release_dir / Path(pred_name).name, columns=["source_id", *score_columns()]
    )
    df = labels.merge(pred, on="source_id", how="left", validate="1:1")
    if df[score_columns()].isna().any(axis=None):
        raise EvalError(f"predictions do not cover every {fold} object")
    y = df["sdss_class"].astype(str).str.upper().to_numpy()
    if not set(np.unique(y)) <= set(CLASSES):
        raise EvalError(
            f"unexpected classes in {fold}: {sorted(set(y) - set(CLASSES))}"
        )
    ids = df["source_id"].to_numpy(dtype=np.int64)
    l1_label = np.asarray(CLASSES)[df[score_columns()].to_numpy(dtype=float).argmax(1)]
    path = output_path(release_dir, fold, model, arm, enable_thinking=enable_thinking)
    if not path.exists():
        raise EvalError(
            f"{path.name} not found: run scripts/layer2_run.py for the {fold} fold "
            "and finish it (every routed object needs a terminal record)"
        )
    l2 = pd.read_parquet(path).sort_values("source_id", ignore_index=True)
    if not np.array_equal(l2["source_id"].to_numpy(), routed["source_id"].to_numpy()):
        raise EvalError(
            f"{path.name} does not hold exactly the {len(routed)} routed {fold} objects "
            f"({len(l2)} rows): no terminal record may be missing"
        )
    if l2["status"].isna().any():
        raise EvalError(f"{path.name} has routed objects without a status")
    meta = _read_meta(path)
    if meta.get("routing_manifest_sha256") != sha256_file(manifest):
        raise EvalError(
            f"{path.name} was consolidated under a different routing manifest"
        )
    for key, want in (
        ("model_requested", model),
        ("arm", arm),
        ("fold", fold),
        ("enable_thinking", str(enable_thinking).lower()),
    ):
        if meta.get(key) != want:
            raise EvalError(
                f"{path.name} metadata {key}={meta.get(key)!r}, expected {want!r}"
            )
    if meta.get("prompt_version") != PROMPT_VERSION:
        raise EvalError(f"{path.name} was produced with another prompt version")
    return FoldData(
        ids=ids,
        y=y,
        pixel=df["healpix_l5"].to_numpy(dtype=np.int64),
        l1_label=l1_label,
        l1_conf=l1_confidence(df).to_numpy(dtype=float),
        routed=routed,
        l2=l2,
        meta=meta,
        parquet_sha256=sha256_file(path),
    )


def align_l2(
    fd: FoldData, calib: dict[str, list[float | None]] | None
) -> tuple[L2.ObjArr, L2.ObjArr, L2.FloatArr, L2.FloatArr]:
    """Per-object status, label, calibrated s_l2 and raw verbal score (NOT_ROUTED / "" / NaN)."""
    n = len(fd.ids)
    pos = pd.Index(fd.ids).get_indexer(fd.l2["source_id"])
    status = np.full(n, "NOT_ROUTED", dtype=object)
    label = np.full(n, "", dtype=object)
    raw = np.full(n, np.nan)
    status[pos] = fd.l2["status"].astype(str).to_numpy()
    label[pos] = fd.l2["label_l2"].astype(object).fillna("").to_numpy()
    raw[pos] = pd.to_numeric(fd.l2["confidence_raw"]).to_numpy(dtype=float) / 100.0
    return status, label, apply_verbal_calibration(raw, calib), raw


def choose_tau_star(conf: L2.FloatArr, target_coverage: float) -> tuple[float, float]:
    """Layer 1 threshold whose coverage is closest to ``target_coverage`` (ties: smaller tau)."""
    s = np.sort(conf)
    cand = np.unique(s)
    cov = 1.0 - np.searchsorted(s, cand, side="left") / s.size
    best = int(np.argmin(np.abs(cov - target_coverage)))
    return float(cand[best]), float(cov[best])


@dataclass(frozen=True)
class Fit:
    tau: float
    tau2: float | None  # +inf iff infeasible, None iff uncalibrated; see tau2_status
    tau2_status: str
    a_keep2: float
    r_max2: float
    verbal_calibration: dict[str, list[float | None]] | None
    calibration_status: str
    n_val: int
    n_val_routed: int
    n_val_ok: int
    tau_star: float
    val_coverage_s3: float
    val_coverage_s1_star: float
    val_parquet_sha256: str
    routing_manifest_sha256: str


def fit_on_val(val: FoldData, tau: float, a_keep2: float) -> Fit:
    """Verbal calibration, tau2 and tau* from val only."""
    status, label, _, raw = align_l2(val, None)
    answered = status == "OK"
    correct = label[answered] == val.y[answered]
    calib = fit_verbal_calibration(raw[answered], correct)
    tau2: float | None = None
    tau2_status = TAU2_UNCALIBRATED
    if calib is not None:
        s_ok = apply_verbal_calibration(raw[answered], calib)
        try:
            tau2 = float(choose_threshold(s_ok, correct, a_keep2, R_MAX_TAU2)["tau"])
            tau2_status = TAU2_FITTED
        except ValueError:
            # With R_MAX_TAU2 = 1.0 the only failure is "no grid value reaches A_keep2".
            tau2 = float("inf")
            tau2_status = TAU2_INFEASIBLE
    s_val = apply_verbal_calibration(raw, calib)
    s3 = combine(
        val.l1_label, val.l1_conf, status, label, s_val, tau, tau2, "abstain_on_fail"
    )
    cov_s3 = float((s3 != ABSTAIN).mean())
    tau_star, cov_star = choose_tau_star(val.l1_conf, cov_s3)
    return Fit(
        tau=tau,
        tau2=tau2,
        tau2_status=tau2_status,
        a_keep2=a_keep2,
        r_max2=R_MAX_TAU2,
        verbal_calibration=calib,
        calibration_status="calibrated"
        if calib is not None
        else "uncalibrated (fewer than 100 OK val answers; tau2 not applied)",
        n_val=len(val.ids),
        n_val_routed=len(val.l2),
        n_val_ok=int(answered.sum()),
        tau_star=tau_star,
        val_coverage_s3=cov_s3,
        val_coverage_s1_star=cov_star,
        val_parquet_sha256=val.parquet_sha256,
        routing_manifest_sha256=val.meta.get("routing_manifest_sha256", ""),
    )


def _stats(values: pd.Series) -> dict[str, float | int | None]:
    v = pd.to_numeric(values).dropna().to_numpy(dtype=float)
    if v.size == 0:
        return {"n": 0, "mean": None, "p50": None, "p95": None}
    return {
        "n": int(v.size),
        "mean": float(v.mean()),
        "p50": float(np.percentile(v, 50)),
        "p95": float(np.percentile(v, 95)),
    }


def routed_summary(test: FoldData) -> dict[str, object]:
    """Operational numbers over the routed test objects (every one is in the denominator)."""
    l2, n = test.l2, len(test.l2)
    pos = pd.Index(test.ids).get_indexer(l2["source_id"])
    y, l1 = test.y[pos], test.l1_label[pos]
    status = l2["status"].astype(str)
    counts = status.value_counts().to_dict()
    ok = status == "OK"
    label = l2["label_l2"].astype(object)
    feats = test.routed.set_index("source_id").loc[l2["source_id"]]
    g = feats["phot_g_mean_mag"].to_numpy(dtype=float)
    total_tokens = pd.to_numeric(l2["prompt_tokens"]) + pd.to_numeric(
        l2["completion_tokens"]
    )
    return {
        "n_routed": n,
        "status_counts": {k: int(v) for k, v in counts.items()},
        "status_rates": {k: v / n for k, v in counts.items()},
        "unknown_rate": counts.get("UNKNOWN", 0) / n,
        "failure_rate": sum(v for k, v in counts.items() if k not in FAILURE_EXCLUDED)
        / n,
        "prompt_tokens": _stats(l2["prompt_tokens"]),
        "completion_tokens": _stats(l2["completion_tokens"]),
        "total_tokens": _stats(total_tokens),
        "calls_per_object": _stats(l2["attempts"]),
        "latency_ms": _stats(l2["latency_ms"]),
        "l2_label_equals_l1_argmax_rate": float(
            (label[ok].to_numpy() == l1[ok.to_numpy()]).mean()
        )
        if ok.any()
        else None,
        "models_reported": {
            str(k): int(v) for k, v in l2["model_reported"].value_counts().items()
        },
        "class_mix": {c: int((y == c).sum()) for c in CLASSES},
        "g_mag_edges": G_BINS.tolist(),
        "g_mag_counts": np.histogram(g[np.isfinite(g)], bins=G_BINS)[0].tolist(),
        "astrometric_params_solved_counts": {
            str(int(k)): int(v)
            for k, v in feats["astrometric_params_solved"].value_counts().items()
        },
    }


def _test_geometry(release_dir: Path, test: FoldData) -> L2.StrArr:
    """Latitude band per test object; refuses ids/pixels that disagree with the committed map."""
    if not np.array_equal(pixel_l5(test.ids), test.pixel):
        raise EvalError(
            "healpix_l5 in labels disagrees with source_id // 2**35 // 4**7"
        )
    pmap = build_pixel_map().set_index("healpix_l5")
    if (pmap["fold"].to_numpy()[test.pixel] != "test").any():
        raise EvalError(
            "test rows lie in pixels that the committed map assigns elsewhere"
        )
    return pmap["lat_band"].to_numpy()[test.pixel].astype(str)


def run_evaluation(
    release_dir: Path,
    model: str,
    arm: str,
    enable_thinking: bool,
    n_boot: int = M.N_BOOT,
) -> dict[str, object]:
    manifest = release_dir / "layer2" / MANIFEST_NAME
    try:
        payload = json.loads(manifest.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise EvalError(f"cannot read {MANIFEST_NAME}: {exc}") from exc
    tau = float(payload["tau"])
    a_keep2 = float(payload.get("a_keep2", A_KEEP2_DEFAULT))
    val = load_fold(release_dir, "val", manifest, model, arm, enable_thinking)
    test = load_fold(release_dir, "test", manifest, model, arm, enable_thinking)
    for key in ("template_sha256", "enable_thinking"):
        if val.meta.get(key) != test.meta.get(key):
            raise EvalError(f"val and test Layer 2 results differ in {key}")
    fit = fit_on_val(val, tau, a_keep2)
    band = _test_geometry(release_dir, test)
    status, label, s_l2, _ = align_l2(test, fit.verbal_calibration)
    inp = L2.Layer2Inputs(
        y=test.y,
        pixel=test.pixel,
        band=band,
        l1_label=test.l1_label,
        l1_conf=test.l1_conf,
        l2_status=status,
        l2_label=label,
        s_l2=s_l2,
        tau=tau,
        tau2=fit.tau2,
        tau_star=fit.tau_star,
    )
    n_pix = int(np.unique(test.pixel).size)
    ops = routed_summary(test)
    meta = {
        "header": (
            f"PROVISIONAL: Layer 2 evaluation on {len(test.ids)} test rows in {n_pix} "
            f"level-5 pixels, {ops['n_routed']} routed at tau={tau:.2f}; regime BALANCED; "
            "cells below the reporting floor are marked n/a"
        ),
        "release": release_dir.name,
        "n_test": len(test.ids),
        "n_test_pixels": n_pix,
        "model_requested": model,
        "arm": arm,
        "prompt_version": test.meta.get("prompt_version"),
        "template_sha256": test.meta.get("template_sha256"),
        "enable_thinking": test.meta.get("enable_thinking"),
        "temperature": test.meta.get("temperature"),
        "created_utc": {
            "val": val.meta.get("created_utc"),
            "test": test.meta.get("created_utc"),
        },
        "git_sha": test.meta.get("git_sha"),
        "floor": {"min_objects": M.MIN_OBJECTS, "min_pixels": M.MIN_PIXELS},
        "regime": "BALANCED",
    }
    return {
        "meta": meta,
        "fit": asdict(fit),
        "operations": ops,
        "eval": L2.evaluate(
            inp,
            n_boot,
            FORCED_REASON_INFEASIBLE if fit.tau2_status == TAU2_INFEASIBLE else None,
        ),
    }
