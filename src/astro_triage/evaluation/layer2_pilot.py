"""Pilot report numbers and projection from the Layer 2 cache and ledgers (ADR 0005 17.7).

Reads only: the result cache (without its writer lock), the call ledgers, the routing manifest
and ``labels.parquet`` of the val fold (and fold counts of the target release). It makes no
network call. Every sampled object is reported; one with no terminal record is ``pending``.
"""

from __future__ import annotations

import itertools
import json
import math
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Final, cast

import numpy as np
import pandas as pd

from astro_triage.abstention.policy import CLASSES, score_columns
from astro_triage.evaluation.layer2_report import FORBIDDEN_WORDS
from astro_triage.llm.cache import TERMINAL, load_records
from astro_triage.llm.prompt import PROMPT_VERSION
from astro_triage.llm.routing import (
    DEFAULT_SEED,
    MANIFEST_NAME,
    model_slug,
    request_for,
    select_routed,
)
from astro_triage.split import build_pixel_map
from eval import metrics as M

Rec = dict[str, Any]
REPO_ROOT: Final = Path(__file__).resolve().parents[3]
DEFAULT_CACHE_DIR: Final = REPO_ROOT / "data/cache/layer2"
CANARY_N: Final = 20  # mirrors llm.runner.CANARY_N; that module imports the HTTP client
PLANNED_RUNS: Final = 2  # one val run and one test run, each with its canary
MIN_OK_CALIBRATION: Final = 100
PIXEL_FRACTIONS: Final = (0.05, 0.10, 0.25, 0.5, 1.0)
FLOOR_SENTENCE: Final = (
    "The reporting floor (>= 100 objects in >= 20 pixels per class) is not reachable with "
    "100 objects; all pilot numbers are descriptive."
)


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval for a proportion; (nan, nan) when n is 0."""
    if n <= 0:
        return (math.nan, math.nan)
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def _stats(values: Sequence[float]) -> Rec:
    v = np.asarray(values, dtype=float)
    if v.size == 0:
        return {"n": 0, "mean": None, "p50": None, "p95": None}
    pct = np.percentile(v, [50, 95])
    return {
        "n": int(v.size),
        "mean": float(v.mean()),
        "p50": float(pct[0]),
        "p95": float(pct[1]),
    }


def _proportion(k: int, n: int) -> Rec:
    lo, hi = wilson(k, n)
    return {"k": k, "n": n, "point": k / n if n else None, "lo": lo, "hi": hi}


def _macro_f1(y: np.ndarray, pred: np.ndarray, idx: np.ndarray) -> float:
    ys = y[idx]
    if not set(CLASSES) <= set(ys.tolist()):
        return math.nan
    zeros = np.zeros(len(idx), dtype=np.int64)
    cm = M.class_metrics(ys, pred[idx], zeros, M.regime_weights(ys), 0, 0)
    return float(np.mean([cm[c]["f1"] for c in CLASSES]))


def _ledger_stats(lines: list[Rec], cached_attempts: int) -> Rec:
    total = len(lines)
    return {
        "http_attempts": total,
        "http_429": sum(1 for r in lines if r.get("http_status") == 429),
        "retries": sum(1 for r in lines if int(r.get("attempt", 1)) > 1),
        # The ledger has no field that marks a canary call: this is every attempt that the
        # sampled cache records do not account for (canary, other folds, other runs).
        "canary_attempts_estimate": max(0, total - cached_attempts),
    }


def _read_ledgers(release_dir: Path) -> list[Rec]:
    lines: list[Rec] = []
    for path in sorted((release_dir / "layer2").glob("calls_*.jsonl")):
        for text in path.read_text(encoding="utf-8").splitlines():
            try:
                rec = json.loads(text)
            except ValueError:
                continue
            if isinstance(rec, dict):
                lines.append(rec)
    return lines


def _projection(n_obj: int, att: Rec, tok: Rec, lat: Rec, rpm_cap: float) -> Rec:
    out: Rec = {"objects": n_obj}
    for q in ("p50", "p95"):
        a, t, ms = att[q], tok[q], lat[q]
        out[f"calls_{q}"] = None if a is None else n_obj * a + CANARY_N * PLANNED_RUNS
        out[f"tokens_{q}"] = None if t is None else n_obj * t
        out[f"hours_{q}"] = None
        if ms is not None:
            out[f"hours_{q}"] = n_obj * max(60.0 / rpm_cap, ms / 1000.0) / 3600.0
    return out


def _target_counts(target_dir: Path, fallback_manifest: Path) -> Rec:
    source = "target release"
    try:
        payload = json.loads(
            (target_dir / "layer2" / MANIFEST_NAME).read_text(encoding="utf-8")
        )
        f_val = float(payload["val_routed_fraction"])
    except (OSError, ValueError, KeyError):
        payload = json.loads(fallback_manifest.read_text(encoding="utf-8"))
        f_val = float(payload["val_routed_fraction"])
        source = "estimate from another release"
    folds = pd.read_parquet(target_dir / "labels.parquet", columns=["fold"])["fold"]
    counts = folds.astype(str).value_counts()
    return {
        "f_val": f_val,
        "f_val_source": source,
        "n_val": int(counts.get("val", 0)),
        "n_test": int(counts.get("test", 0)),
    }


def _entry(
    model: str,
    arm: str,
    thinking: bool,
    rows: pd.DataFrame,
    y: np.ndarray,
    l1: np.ndarray,
    records: Mapping[str, Rec],
    ledger_all: list[Rec],
) -> tuple[Rec, np.ndarray]:
    """One (model, arm, thinking) entry and its per-object forced prediction."""
    n = len(rows)
    found = [
        records.get(request_for(r, model, arm, thinking)[1]) for _, r in rows.iterrows()
    ]
    recs = [r if r is not None and r.get("status") in TERMINAL else None for r in found]
    done = [r for r in recs if r is not None]
    status = ["pending" if r is None else str(r["status"]) for r in recs]
    counts = Counter(status)
    ok = np.array([s == "OK" for s in status])
    label = np.array(
        [str(r["label_l2"]) if r and r.get("label_l2") else "" for r in recs]
    )
    pred = np.where(ok, label, "NONE")
    hit = ok & (label == y)

    def col(name: str) -> list[float]:
        return [r[name] for r in done if r.get(name) is not None]

    total = [
        (r.get("prompt_tokens") or 0) + (r.get("completion_tokens") or 0) for r in done
    ]
    lines = [
        r
        for r in ledger_all
        if r.get("model") == model
        and r.get("arm") == arm
        and r.get("enable_thinking") is thinking
    ]
    idx = np.arange(n)
    entry: Rec = {
        "model": model,
        "arm": arm,
        "thinking": thinking,
        "n": n,
        "pending": counts.get("pending", 0),
        "status_counts": dict(sorted(counts.items())),
        "status_rates": {k: v / n for k, v in sorted(counts.items())},
        "latency_ms": _stats(col("latency_ms")),
        "prompt_tokens": _stats(col("prompt_tokens")),
        "completion_tokens": _stats(col("completion_tokens")),
        "total_tokens": _stats(total),
        "attempts": _stats(col("attempts")),
        "accuracy_forced": _proportion(int(hit.sum()), n),
        "accuracy_selective": _proportion(int(hit.sum()), int(ok.sum())),
        "macro_f1_forced": _macro_f1(y, pred, idx),
        "macro_f1_layer1": _macro_f1(y, l1, idx),
        "unknown_rate": counts.get("UNKNOWN", 0) / n,
        "l2_equals_l1_rate": float((label[ok] == l1[ok]).mean()) if ok.any() else None,
        "model_reported": dict(Counter(str(r.get("model_reported")) for r in done)),
        "ledger": _ledger_stats(lines, int(sum(col("attempts")))),
    }
    return entry, pred


def _add_projection(
    entry: Rec, n_total: int, n_val_routed: float, rpm_cap: float
) -> None:
    att, tok, lat = entry["attempts"], entry["total_tokens"], entry["latency_ms"]
    k_ok = int(entry["status_counts"].get("OK", 0))
    lo, hi = wilson(k_ok, entry["n"])
    entry["projection"] = {
        "target_all_routed": _projection(n_total, att, tok, lat, rpm_cap),
        "pixel_fraction": {
            str(q): _projection(round(q * n_total), att, tok, lat, rpm_cap)
            for q in PIXEL_FRACTIONS
        },
        "expected_ok_val_answers": {
            "point": n_val_routed * k_ok / entry["n"],
            "lo": n_val_routed * lo,
            "hi": n_val_routed * hi,
            "calibration_minimum": MIN_OK_CALIBRATION,
        },
    }


def _comparisons(
    boot: Mapping[tuple[str, str, bool], np.ndarray], entries: list[Rec]
) -> list[Rec]:
    point = {
        (e["model"], e["arm"], e["thinking"]): e["macro_f1_forced"] for e in entries
    }
    out: list[Rec] = []
    for ka, kb in itertools.combinations(boot, 2):
        if ka[1:] != kb[1:]:
            continue
        d = boot[ka] - boot[kb]
        d = d[~np.isnan(d)]
        out.append(
            {
                "a": ka[0],
                "b": kb[0],
                "arm": ka[1],
                "thinking": ka[2],
                "metric": "macro_f1_forced, first minus second",
                "label": "descriptive",
                "point": point[ka] - point[kb],
                "n_valid": int(d.size),
                "lo": float(np.percentile(d, 2.5)) if d.size else None,
                "hi": float(np.percentile(d, 97.5)) if d.size else None,
            }
        )
    return out


def pilot_report(
    release_dir: Path,
    models: Sequence[str],
    arms: Sequence[str],
    thinking_models: Sequence[str],
    sample: int,
    target_release_dir: Path,
    rpm_cap: float,
    *,
    cache_dir: Path = DEFAULT_CACHE_DIR,
    n_boot: int = M.N_BOOT,
) -> dict[str, object]:
    manifest = release_dir / "layer2" / MANIFEST_NAME
    rows = select_routed(release_dir, "val", manifest, sample, DEFAULT_SEED)
    labels = pd.read_parquet(
        release_dir / "labels.parquet",
        columns=["source_id", "sdss_class", "healpix_l5"],
        filters=[("fold", "==", "val")],
    )
    lab = rows[["source_id"]].merge(labels, on="source_id", how="left", validate="1:1")
    y = lab["sdss_class"].astype(str).str.upper().to_numpy()
    pixel = lab["healpix_l5"].to_numpy(dtype=np.int64)
    band = build_pixel_map()["lat_band"].to_numpy()[pixel].astype(str)
    l1 = np.asarray(CLASSES)[rows[score_columns()].to_numpy(dtype=float).argmax(1)]
    ledger_all = _read_ledgers(release_dir)
    rng = np.random.default_rng(M.SEED)
    draws = [M.bootstrap_pixel_draw(pixel, band, rng) for _ in range(n_boot)]
    target = _target_counts(target_release_dir, manifest)
    n_total = round(target["f_val"] * (target["n_val"] + target["n_test"]))

    entries: list[Rec] = []
    boot: dict[tuple[str, str, bool], np.ndarray] = {}
    for model in models:
        records = load_records(
            cache_dir / model_slug(model) / f"{PROMPT_VERSION}.jsonl"
        )
        for arm in arms:
            for thinking in (False, True) if model in thinking_models else (False,):
                entry, pred = _entry(
                    model, arm, thinking, rows, y, l1, records, ledger_all
                )
                _add_projection(
                    entry, n_total, target["f_val"] * target["n_val"], rpm_cap
                )
                boot[(model, arm, thinking)] = np.array(
                    [_macro_f1(y, pred, d) for d in draws]
                )
                entries.append(entry)
    legacy = [
        {
            "model": m,
            "arm": a,
            "thinking": "unknown",
            **_ledger_stats(
                [
                    r
                    for r in ledger_all
                    if r.get("model") == m
                    and r.get("arm") == a
                    and "enable_thinking" not in r
                ],
                0,
            ),
        }
        for m in models
        for a in arms
    ]
    return {
        "release": release_dir.name,
        "sample": sample,
        "seed": DEFAULT_SEED,
        "n_boot": n_boot,
        "rpm_cap": rpm_cap,
        "floor_statement": FLOOR_SENTENCE,
        "target": {"release": target_release_dir.name, "objects": n_total, **target},
        "entries": entries,
        "ledger_legacy": legacy,
        "model_comparison": _comparisons(boot, entries),
    }


def _f(v: object, spec: str = ".3f") -> str:
    if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v):
        return "n/a"
    return format(v, spec)


def _ci(p: Rec) -> str:
    return f"{_f(p['point'])} [{_f(p['lo'])}, {_f(p['hi'])}]"


def _row(cells: Sequence[object]) -> str:
    return "| " + " | ".join(str(c) for c in cells) + " |"


def render_pilot_markdown(rep: Mapping[str, object]) -> str:
    entries = cast(list[Rec], rep["entries"])
    target = cast(Rec, rep["target"])
    think = {True: "on", False: "off"}
    head = ["model", "arm", "thinking", "n", "pending", "status counts"]
    head += ["acc forced [95% CI]", "acc selective [95% CI]", "macro-F1 forced"]
    head += ["macro-F1 L1", "UNKNOWN", "L2 = L1 argmax", "latency ms p50/p95"]
    head += [
        "tokens mean/p95",
        "attempts mean/p95",
        "HTTP / 429 / retries / canary est.",
    ]
    out = [
        f"# Layer 2 pilot report: {rep['release']}, {rep['sample']} val routed objects",
        "",
        str(rep["floor_statement"]),
        "",
        "## Per model, arm and reasoning setting",
        "",
        _row(head),
        "|" + "---|" * len(head),
    ]
    for e in entries:
        lat, tok, att, lg = (
            e["latency_ms"],
            e["total_tokens"],
            e["attempts"],
            e["ledger"],
        )
        out.append(
            _row(
                [
                    e["model"],
                    e["arm"],
                    think[e["thinking"]],
                    e["n"],
                    e["pending"],
                    json.dumps(e["status_counts"]),
                    _ci(e["accuracy_forced"]),
                    f"{_ci(e['accuracy_selective'])} (n={e['accuracy_selective']['n']})",
                    _f(e["macro_f1_forced"]),
                    _f(e["macro_f1_layer1"]),
                    _f(e["unknown_rate"]),
                    _f(e["l2_equals_l1_rate"]),
                    f"{_f(lat['p50'], '.0f')}/{_f(lat['p95'], '.0f')}",
                    f"{_f(tok['mean'], '.0f')}/{_f(tok['p95'], '.0f')}",
                    f"{_f(att['mean'], '.2f')}/{_f(att['p95'], '.0f')}",
                    " / ".join(
                        str(lg[k])
                        for k in (
                            "http_attempts",
                            "http_429",
                            "retries",
                            "canary_attempts_estimate",
                        )
                    ),
                ]
            )
        )
    out += ["", "## Model comparison (paired val pixel bootstrap, descriptive)", ""]
    for c in cast(list[Rec], rep["model_comparison"]):
        out.append(
            f"- {c['a']} minus {c['b']} ({c['arm']}, thinking {think[c['thinking']]}): "
            f"{_f(c['point'])} [{_f(c['lo'])}, {_f(c['hi'])}] on {c['metric']}"
        )
    out += [
        "",
        (
            f"## Projection for {target['release']} ({target['objects']} routed objects, val+test: "
            f"n_val={target['n_val']}, n_test={target['n_test']}; "
            f"val routed fraction {_f(target['f_val'], '.4f')}, {target['f_val_source']})"
        ),
        "",
        _row(
            ["model", "arm", "thinking", "scope", "objects", "calls p50..p95"]
            + ["tokens p50..p95", "hours p50..p95"]
        ),
        "|" + "---|" * 8,
    ]
    for e in entries:
        proj = e["projection"]
        scopes = [("all routed", proj["target_all_routed"])]
        scopes += [
            (f"pixel fraction {q}", p) for q, p in proj["pixel_fraction"].items()
        ]
        for name, p in scopes:
            out.append(
                _row(
                    [
                        e["model"],
                        e["arm"],
                        think[e["thinking"]],
                        name,
                        p["objects"],
                        f"{_f(p['calls_p50'], '.0f')}..{_f(p['calls_p95'], '.0f')}",
                        f"{_f(p['tokens_p50'], '.0f')}..{_f(p['tokens_p95'], '.0f')}",
                        f"{_f(p['hours_p50'], '.1f')}..{_f(p['hours_p95'], '.1f')}",
                    ]
                )
            )
    out += ["", "## Expected OK val answers against the calibration minimum", ""]
    for e in entries:
        x = e["projection"]["expected_ok_val_answers"]
        out.append(
            f"- {e['model']} / {e['arm']} / thinking {think[e['thinking']]}: "
            f"{_f(x['point'], '.0f')} (range {_f(x['lo'], '.0f')} to {_f(x['hi'], '.0f')}) "
            f"against a minimum of {x['calibration_minimum']}"
        )
    out += ["", "## Ledger lines without the reasoning setting", ""]
    out += [
        f"- {g['model']} / {g['arm']} / thinking unknown: {g['http_attempts']} attempts, "
        f"{g['http_429']} HTTP 429, {g['retries']} retries"
        for g in cast(list[Rec], rep["ledger_legacy"])
    ]
    text = "\n".join(out) + "\n"
    hit = [w for w in FORBIDDEN_WORDS if w in text.lower()]
    if hit:
        raise ValueError(f"report text contains ranking words {hit}; refusing to write")
    return text
