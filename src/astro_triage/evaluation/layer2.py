"""Layer 2 evaluation core (ADR 0005 D7): systems S0..S3, routed-subset views, paired bootstrap.

Pure computation on aligned test arrays; all I/O and the val-only fitting live in
``layer2_data``. Regime BALANCED only (Layer 2 outputs no class-probability vector).
Every object routed to Layer 2 stays in every denominator: an abstention or any non-OK
Layer 2 status is a non-answer, never a dropped row. A prediction outside ``CLS`` is a
non-answer by definition, so abstention and failure share one code path.
All units are evaluated on the SAME bootstrap replicates (one pixel draw per replicate,
seed ``eval.metrics.SEED``): paired differences are replicate-by-replicate.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Final, Literal

import numpy as np
from numpy.typing import NDArray

from astro_triage.abstention.policy import ABSTAIN, DEFAULT_GRID, combine
from astro_triage.evaluation.runner import make_cell
from eval import metrics as M
from eval.reweight import EmptyClassError

CLS = M.CLASSES
FloatArr = NDArray[np.float64]
StrArr = NDArray[np.str_]
BoolArr = NDArray[np.bool_]
IntArr = NDArray[np.int64]
ObjArr = NDArray[np.object_]
Policy = Literal["abstain_on_fail", "fallback_l1"]

# Negative-result rule, fixed BEFORE any test number is seen (ADR D7.8): a headline pair shows
# a gain iff the paired CI of this metric lies entirely above 0 on a cell that passes the floor.
HEADLINE_METRIC: Final = "macro_f1"
HEADLINE_PAIRS: Final[tuple[tuple[str, str], ...]] = (("S2", "S0"), ("S3", "S1*"))
PAIRS: Final[tuple[tuple[str, str], ...]] = (
    *HEADLINE_PAIRS,
    ("S3", "S1"),
    ("L2_forced", "L1_routed"),
    ("L2_selective", "L1_ok"),
)
METRIC_KEYS: Final[tuple[str, ...]] = (
    *(f"{m}/{c}" for m in ("precision", "recall", "f1") for c in CLS),
    "macro_f1",
    "accuracy",
    "accuracy_answered",
    "coverage",
    "ece",
)


@dataclass(frozen=True)
class Layer2Inputs:
    """Test fold, aligned per object. ``l2_*`` hold NOT_ROUTED / "" / NaN where not routed."""

    y: StrArr
    pixel: IntArr
    band: StrArr
    l1_label: StrArr
    l1_conf: FloatArr
    l2_status: ObjArr
    l2_label: ObjArr
    s_l2: FloatArr
    tau: float
    tau2: float | None
    tau_star: float


@dataclass(frozen=True)
class Unit:
    pop: str
    pred: StrArr
    conf: FloatArr


def build_units(inp: Layer2Inputs) -> tuple[dict[str, Unit], dict[str, BoolArr]]:
    n = len(inp.y)
    routed = inp.l1_conf < inp.tau
    ok = routed & (inp.l2_status == "OK")
    no_status = np.full(n, "NONE", dtype=object)
    no_label = np.full(n, "", dtype=object)
    no_score = np.full(n, np.nan)

    def comb(tau: float, policy: Policy, tau2: float | None, with_l2: bool) -> StrArr:
        out = combine(
            inp.l1_label,
            inp.l1_conf,
            inp.l2_status if with_l2 else no_status,
            inp.l2_label if with_l2 else no_label,
            inp.s_l2 if with_l2 else no_score,
            tau,
            tau2,
            policy,
        )
        return out.astype(str)

    s3 = comb(inp.tau, "abstain_on_fail", inp.tau2, True)
    used_l2 = routed & (
        s3 != ABSTAIN
    )  # same trust rule as S2: its Layer 2 answers are these
    mixed = np.where(used_l2, inp.s_l2, inp.l1_conf)
    forced = np.where(inp.l2_status == "OK", inp.l2_label, "NONE").astype(str)
    units = {
        "S0": Unit("all", inp.l1_label, inp.l1_conf),
        "S1": Unit("all", comb(inp.tau, "abstain_on_fail", None, False), inp.l1_conf),
        "S1*": Unit(
            "all", comb(inp.tau_star, "abstain_on_fail", None, False), inp.l1_conf
        ),
        "S2": Unit("all", comb(inp.tau, "fallback_l1", inp.tau2, True), mixed),
        "S3": Unit("all", s3, mixed),
        "L1_routed": Unit("routed", inp.l1_label, inp.l1_conf),
        "L2_forced": Unit("routed", forced, inp.s_l2),
        "L1_ok": Unit("routed_ok", inp.l1_label, inp.l1_conf),
        "L2_selective": Unit("routed_ok", forced, inp.s_l2),
    }
    return units, {"all": np.ones(n, dtype=bool), "routed": routed, "routed_ok": ok}


def _ece(conf: FloatArr, correct: BoolArr, w: FloatArr) -> float:
    if conf.size == 0 or np.isnan(conf).any() or w.sum() <= 0:
        return math.nan
    curve = M.reliability_curve(conf, correct, w, M.N_BINS, "equal_mass")
    ok = ~np.isnan(curve["freq"])
    gap = np.abs(curve["freq"][ok] - curve["mean_conf"][ok])
    return float((curve["weight"][ok] / curve["weight"][ok].sum() * gap).sum())


def unit_metrics(
    y: StrArr, pred: StrArr, conf: FloatArr, w: FloatArr
) -> dict[str, float]:
    """Weighted metrics of one resample; abstained / failed objects stay in every denominator."""
    ans = np.isin(pred, CLS)
    hit = ans & (pred == y)
    cm = M.class_metrics(y, pred, np.zeros(len(y), dtype=np.int64), w, 0, 0)
    out: dict[str, float] = {}
    for c in CLS:
        out[f"precision/{c}"] = float(cm[c]["purity"])
        out[f"recall/{c}"] = float(cm[c]["completeness"])
        out[f"f1/{c}"] = float(cm[c]["f1"])
    out["macro_f1"] = float(np.mean([out[f"f1/{c}"] for c in CLS]))
    total, answered = float(w.sum()), float(w[ans].sum())
    out["accuracy"] = float(w[hit].sum() / total)
    out["coverage"] = answered / total
    out["accuracy_answered"] = float(w[hit].sum() / answered) if answered else math.nan
    out["ece"] = _ece(conf[ans], hit[ans], w[ans])
    return out


def _replicate(
    inp: Layer2Inputs,
    units: dict[str, Unit],
    masks: dict[str, BoolArr],
    idx: IntArr,
) -> dict[str, dict[str, float]]:
    nan = {k: math.nan for k in METRIC_KEYS}
    out: dict[str, dict[str, float]] = {}
    for pop, mask in masks.items():
        sub = idx[mask[idx]]
        w: FloatArr | None = None
        if sub.size:
            try:
                w = M.regime_weights(inp.y[sub])
            except EmptyClassError:
                w = None
        for name, u in units.items():
            if u.pop == pop:
                out[name] = (
                    nan
                    if w is None
                    else unit_metrics(inp.y[sub], u.pred[sub], u.conf[sub], w)
                )
    return out


def _cell(
    point: float,
    samples: FloatArr,
    floor: dict[str, tuple[int, int, bool]],
    key: str,
    n_boot: int,
) -> dict[str, object]:
    cls = key.split("/")[-1]
    rows = [floor[cls]] if cls in CLS else list(floor.values())
    n, pixels, _ = min(rows, key=lambda r: (r[2], r[0], r[1]))
    ok = all(r[2] for r in rows)
    cell = make_cell(point, samples, ok, n_boot)
    if not ok:
        cell["status"] = f"n/a (n={n}, pixels={pixels})"
    return cell


def risk_coverage(inp: Layer2Inputs) -> list[dict[str, float]]:
    """Layer 1 selective accuracy over the tau grid (raw, unweighted), test fold."""
    hit = inp.l1_label == inp.y
    rows = []
    for tau in DEFAULT_GRID:
        kept = inp.l1_conf >= tau
        rows.append(
            {
                "tau": float(tau),
                "coverage": float(kept.mean()),
                "accuracy": float(hit[kept].mean()) if kept.any() else math.nan,
            }
        )
    return rows


def headline_verdict(
    diffs: dict[str, dict[str, dict[str, object]]],
    forced_reason: str | None = None,
) -> dict[str, object]:
    """Apply the pre-registered rule to the headline pairs (see ``HEADLINE_METRIC``).

    ``forced_reason`` can only force ``no_gain``; it never produces a gain.
    """
    pairs: dict[str, dict[str, object]] = {}
    for a, b in HEADLINE_PAIRS:
        cell = diffs[f"{a} - {b}"][HEADLINE_METRIC]
        lo = cell["lo"]
        evaluable = cell["status"] == "ok" and isinstance(lo, float)
        pairs[f"{a} - {b}"] = {
            **cell,
            "evaluable": evaluable,
            "gain": bool(evaluable and isinstance(lo, float) and lo > 0),
        }
    gain = forced_reason is None and any(p["gain"] for p in pairs.values())
    verdict: dict[str, object] = {
        "metric": HEADLINE_METRIC,
        "pairs": pairs,
        "n_evaluable": sum(bool(p["evaluable"]) for p in pairs.values()),
        "outcome": "gain" if gain else "no_gain",
    }
    if forced_reason is not None:
        verdict["forced_reason"] = forced_reason
    return verdict


def evaluate(
    inp: Layer2Inputs, n_boot: int = M.N_BOOT, forced_reason: str | None = None
) -> dict[str, object]:
    units, masks = build_units(inp)
    floors = {p: M.floor_status(inp.y[m], inp.pixel[m]) for p, m in masks.items()}
    point = _replicate(inp, units, masks, np.arange(len(inp.y)))
    draws = {u: {k: np.full(n_boot, np.nan) for k in METRIC_KEYS} for u in units}
    rng = np.random.default_rng(M.SEED)
    for r in range(n_boot):
        idx = M.bootstrap_pixel_draw(inp.pixel, inp.band, rng)
        for name, vals in _replicate(inp, units, masks, idx).items():
            for k, v in vals.items():
                draws[name][k][r] = v

    out_units: dict[str, object] = {}
    for name, u in units.items():
        fl, m = floors[u.pop], masks[u.pop]
        ans = m & np.isin(u.pred, CLS)
        out_units[name] = {
            "population": u.pop,
            "n": int(m.sum()),
            "n_answered": int(ans.sum()),
            "raw_coverage": float(ans.sum() / m.sum()) if m.any() else None,
            "raw_accuracy_answered": float((u.pred[ans] == inp.y[ans]).mean())
            if ans.any()
            else None,
            "metrics": {
                k: _cell(point[name][k], draws[name][k], fl, k, n_boot)
                for k in METRIC_KEYS
            },
        }
    diffs = {
        f"{a} - {b}": {
            k: _cell(
                point[a][k] - point[b][k],
                draws[a][k] - draws[b][k],
                floors[units[a].pop],
                k,
                n_boot,
            )
            for k in METRIC_KEYS
        }
        for a, b in PAIRS
    }
    return {
        "n_boot": n_boot,
        "seed": M.SEED,
        "floor": {
            p: {c: {"n": n, "pixels": px, "passes": ok} for c, (n, px, ok) in f.items()}
            for p, f in floors.items()
        },
        "units": out_units,
        "diffs": diffs,
        "verdict": headline_verdict(diffs, forced_reason),
        "risk_coverage": risk_coverage(inp),
    }
