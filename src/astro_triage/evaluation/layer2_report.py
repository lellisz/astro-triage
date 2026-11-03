"""Render and write the Layer 2 evaluation (``eval_l2.json`` and ``eval_l2.md``; ADR 0005 D7).

The markdown states numbers, intervals and flags only: no ranking is claimed, and the header
is built from the data of the run. Paths are never printed.
"""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
from typing import Final

import numpy as np

from astro_triage.evaluation.layer2 import HEADLINE_METRIC, HEADLINE_PAIRS
from eval import metrics as M

CLS = M.CLASSES
FORBIDDEN_WORDS: Final = ("superior", "melhor", "better", "worse", "outperform")
SYSTEMS: Final = ("S0", "S1", "S1*", "S2", "S3")
ROUTED_UNITS: Final = ("L1_routed", "L2_forced", "L1_ok", "L2_selective")
SUMMARY_KEYS: Final = (
    "coverage",
    "accuracy",
    "accuracy_answered",
    "macro_f1",
    "ece",
)

LIMITATIONS: Final = (
    (
        "Regime BALANCED (P0) only: Layer 2 outputs no class-probability vector, so prior shift "
        "and the REALISTIC regime are not reported for Layer 2."
    ),
    (
        "Routed objects are the hard, SDSS-selected tail of the test fold; no claim outside that "
        "subset. Its class mix and histograms are printed below."
    ),
    (
        "Verbalized confidence is a score, not a probability; s_l2 is its val-fitted histogram "
        "binning. ECE of S2/S3 mixes Layer 1 probabilities and s_l2."
    ),
    "Zero-shot Layer 2 reads the same 21 features as Layer 1 (ADR 0002 cut (c) logic).",
    "Temperature 0 on a hosted model: runs are dated and may not be reproducible.",
    "Cells below the reporting floor are n/a; no raw value is printed for them.",
)


def to_jsonable(obj: object) -> object:
    if isinstance(obj, dict):
        return {str(k): to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return to_jsonable(obj.tolist())
    if isinstance(obj, (np.floating, float)):
        return float(obj) if math.isfinite(float(obj)) else None
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    return obj


def _fmt(cell: dict, diff: bool = False) -> str:
    status, pt = cell["status"], cell["point"]
    if status != "ok" or pt is None:
        return status if status.startswith("n/a") else f"n/a ({status})"
    text = f"{pt:.3f} [{cell['lo']:.3f}, {cell['hi']:.3f}]"
    if diff and (cell["lo"] > 0 or cell["hi"] < 0):
        text += " CI excl. 0"
    return text


def _table(header: list[str], rows: list[list[str]]) -> list[str]:
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    return [*lines, *("| " + " | ".join(r) + " |" for r in rows), ""]


def _unit_rows(
    units: dict, names: tuple[str, ...], keys: tuple[str, ...]
) -> list[list[str]]:
    return [[n, *(_fmt(units[n]["metrics"][k]) for k in keys)] for n in names]


def verdict_text(res: dict) -> str:
    v, meta = res["eval"]["verdict"], res["meta"]
    tokens = res["operations"]["total_tokens"]["mean"]
    per_obj = "n/a" if tokens is None else f"{tokens:.0f}"
    pairs = "; ".join(f"{k}: {_fmt(p, diff=True)}" for k, p in v["pairs"].items())
    if v["outcome"] == "no_gain":
        text = (
            f"Layer 2 (text, Nemotron {meta['model_requested']}, prompt {meta['prompt_version']}) "
            f"adds no measurable value over Layer 1 at {per_obj} tokens/object. "
            f"Headline pairs ({v['metric']}, paired 95% CI): {pairs}."
        )
        if v["n_evaluable"] < len(HEADLINE_PAIRS):
            text += (
                f" {len(HEADLINE_PAIRS) - v['n_evaluable']} headline cell(s) below the reporting "
                "floor or undefined: no claim can be made from them."
            )
        forced = v.get("forced_reason")
        if forced:
            text += f" Forced verdict: {forced}."
        return text + (
            " No prompt or threshold is changed after seeing test; a new attempt is a new "
            "prompt version evaluated first on val."
        )
    return (
        f"Positive paired difference with a CI above 0 on {v['metric']} (descriptive, no "
        f"ranking claimed). Headline pairs (paired 95% CI): {pairs}."
    )


def _tau2_text(fit: dict) -> str:
    """tau2 line; readers branch on ``tau2_status`` because a null tau2 is ambiguous."""
    status = fit["tau2_status"]
    if status == "fitted":
        return f"tau2 {fit['tau2']:.2f} (status: {status})"
    return f"tau2 not a number (status: {status})"


def _uncalibrated_notice(fit: dict) -> list[str]:
    if not str(fit["tau2_status"]).startswith("not_applied"):
        return []
    return [
        (
            "Uncalibrated: fewer than 100 OK val answers, so tau2 is not applied. S2 and S3 "
            "trust every OK Layer 2 answer in {STAR, GALAXY, QSO}; their ECE is n/a (no score)."
        ),
        "",
    ]


def _status_rows(ops: dict) -> list[list[str]]:
    return [
        [k, str(ops["status_counts"][k]), f"{ops['status_rates'][k]:.3f}"]
        for k in sorted(ops["status_counts"])
    ]


def _stat_row(name: str, s: dict) -> list[str]:
    return [
        name,
        str(s["n"]),
        *("n/a" if s[k] is None else f"{s[k]:.1f}" for k in ("mean", "p50", "p95")),
    ]


def render_markdown(res: dict) -> str:
    meta, fit, ops, ev = res["meta"], res["fit"], res["operations"], res["eval"]
    units, diffs = ev["units"], ev["diffs"]
    L: list[str] = [f"# {meta['header']}", ""]
    L += [
        "Test fold only. Numbers, intervals and flags; no ranking is claimed.",
        "",
        "## Result",
        "",
        verdict_text(res),
        "",
        *_uncalibrated_notice(fit),
        "## Setup",
        "",
        f"- model requested {meta['model_requested']}; reported {json.dumps(ops['models_reported'])}",
        (
            f"- prompt {meta['prompt_version']} (template sha256 {meta['template_sha256']}), arm "
            f"{meta['arm']}, thinking {meta['enable_thinking']}, temperature {meta['temperature']}"
        ),
        f"- run dates (UTC) {json.dumps(meta['created_utc'])}; git {meta['git_sha']}",
        (
            f"- routing tau {fit['tau']:.2f} (frozen manifest); {_tau2_text(fit)}; "
            f"verbal calibration: {fit['calibration_status']}"
        ),
        (
            f"- tau2 rule: smallest grid value with kept val accuracy >= {fit['a_keep2']} "
            f"(no routed-fraction cap), {fit['n_val_ok']} OK val answers of {fit['n_val_routed']} "
            "routed"
        ),
        (
            f"- S1* tau* = {fit['tau_star']:.4f} chosen on val: coverage {fit['val_coverage_s1_star']:.4f} "
            f"vs S3 {fit['val_coverage_s3']:.4f}"
        ),
        (
            f"- bootstrap: pixel cluster within latitude bands, B={ev['n_boot']}, seed {ev['seed']}, "
            "95% percentile intervals, the same replicates for every system and view (paired)"
        ),
        (
            f"- floor: >= {meta['floor']['min_objects']} objects in >= {meta['floor']['min_pixels']} "
            "pixels per class; overall metrics need every class to pass"
        ),
        (
            "- metrics are BALANCED-weighted; abstentions and non-OK statuses stay in every "
            "denominator (accuracy), accuracy_answered uses answered objects only; ECE is top-label, "
            f"{M.N_BINS} equal-mass bins, answered objects"
        ),
        "",
        "## Systems (all test objects)",
        "",
        (
            "S0 Layer 1 alone; S1 Layer 1 abstaining on routed; S1* S1 with tau* from val; "
            "S2 Layer 1 + Layer 2 with Layer 1 fallback; S3 Layer 1 + Layer 2 + abstention."
        ),
        "",
    ]
    L += _table(["system", *SUMMARY_KEYS], _unit_rows(units, SYSTEMS, SUMMARY_KEYS))
    for kind in ("precision", "recall", "f1"):
        keys = tuple(f"{kind}/{c}" for c in CLS)
        L += [f"Per-class {kind}:", ""]
        L += _table(["system", *CLS], _unit_rows(units, SYSTEMS, keys))
    L += [
        f"Paired differences, first minus second (headline metric {HEADLINE_METRIC}):",
        "",
    ]
    dkeys = (*SUMMARY_KEYS, *(f"f1/{c}" for c in CLS))
    L += _table(
        ["pair", *dkeys],
        [[p, *(_fmt(c[k], diff=True) for k in dkeys)] for p, c in diffs.items()],
    )
    L += ["## Routed subset (test objects with Layer 1 confidence below tau)", ""]
    L += [
        "forced: any non-OK Layer 2 status is a wrong answer (OK-only subset in the selective rows).",
        "",
    ]
    L += _table(
        ["view", "n", "answered", *SUMMARY_KEYS],
        [
            [
                n,
                str(units[n]["n"]),
                str(units[n]["n_answered"]),
                *(_fmt(units[n]["metrics"][k]) for k in SUMMARY_KEYS),
            ]
            for n in ROUTED_UNITS
        ],
    )
    L += _table(
        ["pair", *dkeys],
        [
            [p, *(_fmt(diffs[p][k], diff=True) for k in dkeys)]
            for p in ("L2_forced - L1_routed", "L2_selective - L1_ok")
        ],
    )
    L += [
        f"Selective coverage: {units['L2_selective']['n']} OK answers of {ops['n_routed']} routed.",
        "",
        "## Layer 2 operations (routed test objects)",
        "",
    ]
    L += _table(["status", "n", "rate"], _status_rows(ops))
    L += [
        (
            f"UNKNOWN rate {ops['unknown_rate']:.3f}; failure rate (non-OK, non-UNKNOWN) "
            f"{ops['failure_rate']:.3f}; Layer 2 label equals the Layer 1 argmax in "
            f"{ops['l2_label_equals_l1_argmax_rate']} of OK answers"
        ),
        "",
    ]
    L += _table(
        ["per routed object", "n", "mean", "p50", "p95"],
        [
            _stat_row("prompt tokens", ops["prompt_tokens"]),
            _stat_row("completion tokens", ops["completion_tokens"]),
            _stat_row("total tokens", ops["total_tokens"]),
            _stat_row("calls (attempts)", ops["calls_per_object"]),
            _stat_row("latency ms", ops["latency_ms"]),
        ],
    )
    names = [
        f"[{a:g},{b:g})" for a, b in zip(ops["g_mag_edges"], ops["g_mag_edges"][1:])
    ]
    L += [
        "## Routed subset composition",
        "",
        (
            f"class mix {json.dumps(ops['class_mix'])}; astrometric_params_solved "
            f"{json.dumps(ops['astrometric_params_solved_counts'])}"
        ),
        "",
        *_table(["G magnitude", *names], [["count", *map(str, ops["g_mag_counts"])]]),
        "## Floor status",
        "",
    ]
    for pop, per in ev["floor"].items():
        L.append(
            f"- {pop}: "
            + "; ".join(
                f"{c} n={v['n']} pixels={v['pixels']} {'ok' if v['passes'] else 'n/a'}"
                for c, v in per.items()
            )
        )
    L += ["", "## What this does NOT show", "", *[f"- {x}" for x in LIMITATIONS], ""]
    text = "\n".join(L)
    low = text.lower()
    hit = [w for w in FORBIDDEN_WORDS if w in low]
    if hit:
        raise ValueError(f"report text contains ranking words {hit}; refusing to write")
    return text


def write_outputs(res: dict, outdir: Path) -> list[Path]:
    outdir.mkdir(parents=True, exist_ok=True)
    clean = to_jsonable(res)
    if not isinstance(clean, dict):
        raise TypeError("result must be a dict")
    payloads = {
        "eval_l2.json": json.dumps(clean, indent=1, allow_nan=False),
        "eval_l2.md": render_markdown(clean),
    }
    paths = []
    for name, text in payloads.items():
        path, tmp = outdir / name, outdir / f"{name}.tmp"
        try:
            tmp.write_text(text, encoding="utf-8")
            os.replace(tmp, path)
        finally:
            tmp.unlink(missing_ok=True)
        paths.append(path)
    return paths
