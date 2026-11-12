"""Test side of the DL comparison, part 2: block-bootstrap run over the frozen systems (ADR 0006 4.4).

Reads nothing itself: ``run_compare`` takes the already verified test set, systems and freeze from
``dl_eval`` and returns a dict. Thresholds come from the freeze only.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import numpy as np

from astro_triage.evaluation.dl_boot import (
    Selection,
    block_labels,
    class_codes,
    class_counts,
    interval,
    make_selection,
    prior_vector,
    purity,
    replicate_purities,
    seed_mean_diff,
)
from astro_triage.evaluation.dl_compare import (
    COMPLETENESS_LEVELS,
    DESCRIPTIVE_PAIRS,
    HEADLINE_PAIRS,
    MIN_BLOCKS,
    N_SEEDS,
    P3_BANDS,
    PRIMARY_COMPLETENESS,
    PRIMARY_PRIOR,
    TARGET_CLASSES,
    class_verdict,
    is_low_count,
    pair_verdict,
)
from astro_triage.evaluation.dl_eval import refuse
from astro_triage.evaluation.dl_val import NETS, json_float, sha256_file
from astro_triage.evaluation.runner import GRID, FloatArr, ModelData, TestSet
from eval import metrics as M

CSTARS: Final = tuple(f"{c:.2f}" for c in COMPLETENESS_LEVELS)


@dataclass(frozen=True)
class _Group:
    """One population: frozen selections, observed purities and the replicate purities."""

    point: dict[tuple[str, str, str, str], float]
    boot: dict[tuple[tuple[str, str, str], str], FloatArr]
    comp: dict[tuple[str, str, str], float]
    cont: dict[tuple[str, str, str], dict[str, int]]
    floor: dict[str, tuple[int, int, bool]]
    blocks: dict[str, int]


def _frozen(freeze: dict[str, object], name: str, cls: str, cs: str) -> float:
    thr = freeze["thresholds"]
    assert isinstance(thr, dict)
    return float(thr[name][cls][cs])


def _pairs() -> list[tuple[str, str, str]]:
    pairs = [(x, ref, "headline") for x, ref in HEADLINE_PAIRS]
    pairs += [(x, ref, "descriptive") for x, ref in DESCRIPTIVE_PAIRS]
    for n in NETS:  # fronts (2) and (3); front (1) is the headline pair
        pairs += [
            (f"{n}_noind", "lgbm_noind", "front2"),
            (f"{n}_full", f"{n}_noind", "front3"),
        ]
    return [*pairs, ("lgbm_full", "lgbm_noind", "front4")]


def _run_group(
    ts: TestSet,
    systems: dict[str, ModelData],
    freeze: dict[str, object],
    mask: np.ndarray,
    priors: tuple[str, ...],
    level: int,
    n_boot: int,
) -> _Group:
    y, band, pix = ts.y[mask], ts.band[mask], ts.pixel[mask]
    code = class_codes(y)
    block = block_labels(pix, band) if level == 3 else pix
    fams = sorted({f for x, ref, _ in _pairs() for f in (x, ref)})
    sels: dict[tuple[str, str, str], tuple[Selection, int]] = {}
    for name in (f"{f}_s{k}" for f in fams for k in range(N_SEEDS)):
        for cls in TARGET_CLASSES:
            ci = M.CLASSES.index(cls)
            for cs in CSTARS:
                score = systems[name].proba[mask, ci]
                sels[(name, cls, cs)] = (
                    make_selection(code, score >= _frozen(freeze, name, cls, cs)),
                    ci,
                )
    pis = {p: prior_vector(GRID[p]) for p in priors}
    boot = replicate_purities(code, block, band, sels, pis, n_boot, M.SEED)
    point, comp, cont = {}, {}, {}
    for key, (sel, ci) in sels.items():
        n, cnt = class_counts(code, sel)
        comp[key] = float(cnt[ci] / n[ci]) if n[ci] > 0 else math.nan
        cont[key] = {k: int(cnt[j]) for j, k in enumerate(M.CLASSES) if j != ci}
        for p, pi in pis.items():
            point[(*key, p)] = purity(n, cnt, pi, ci)
    blocks = {
        c: int(np.unique(block[code == M.CLASSES.index(c)]).size)
        for c in TARGET_CLASSES
    }
    return _Group(point, boot, comp, cont, M.floor_status(y, pix), blocks)


def _spread(v: list[float]) -> dict[str, float | None]:
    return {
        "min": json_float(min(v)),
        "median": json_float(float(np.median(v))),
        "max": json_float(max(v)),
    }


def _cell(g: _Group, x: str, ref: str, cls: str, cs: str, p: str) -> dict[str, object]:
    xs = [f"{x}_s{k}" for k in range(N_SEEDS)]
    ls = [f"{ref}_s{k}" for k in range(N_SEEDS)]
    px = [g.point[(n, cls, cs, p)] for n in xs]
    pl = [g.point[(n, cls, cs, p)] for n in ls]
    lo, hi, frac = interval(
        seed_mean_diff(
            [g.boot[((n, cls, cs), p)] for n in xs],
            [g.boot[((n, cls, cs), p)] for n in ls],
        )
    )
    return {
        "d": json_float(float(np.mean(px) - np.mean(pl))),
        "ci": [json_float(lo), json_float(hi)],
        "na": not math.isfinite(lo),
        "invalid_fraction": frac,
        "realised_completeness": {n: json_float(g.comp[(n, cls, cs)]) for n in xs + ls},
        "contaminants": {n: g.cont[(n, cls, cs)] for n in xs + ls},
        "purity_seed": {x: _spread(px), ref: _spread(pl)},
    }


def _headline(
    cell: dict[str, object],
    g: _Group,
    x: str,
    ref: str,
    cls: str,
    cs: str,
    p: str,
    delta_doc: dict[str, object],
) -> None:
    """ADR 0006 4.4 (i)-(v) and the class verdict; mutates the cell in place."""
    dc = delta_doc[ref.split("_")[1]][cls][cs][p]
    delta = math.nan if dc["delta"] is None else float(dc["delta"])
    names = [f"{f}_s{k}" for f in (x, ref) for k in range(N_SEEDS)]
    low = any(is_low_count(g.cont[(n, cls, cs)]) for n in names)
    floor_ok = g.floor[cls][2]
    reportable = bool(
        floor_ok
        and g.blocks[cls] >= MIN_BLOCKS
        and not low
        and not cell["na"]
        and math.isfinite(delta)
        and dc["val_checks_pass"]
    )
    d, (lo, hi) = cell["d"], cell["ci"]
    nan = math.nan
    vk = [float(x) for x in dc.get("v_k") or [] if x is not None and math.isfinite(x)]
    cell.update(
        v_k_spread=_spread(vk) if vk else None,
        delta=json_float(delta),
        reportable=reportable,
        low_count=low,
        blocks=g.blocks[cls],
        floor_ok=floor_ok,
        val_checks_pass=bool(dc["val_checks_pass"]),
        class_verdict=class_verdict(
            nan if d is None else d,
            nan if lo is None else lo,
            nan if hi is None else hi,
            delta,
            reportable,
        ),
    )


def _entry(
    g: _Group,
    g5: _Group,
    x: str,
    ref: str,
    role: str,
    cs: str,
    p: str,
    delta_doc: dict[str, object],
) -> dict[str, object]:
    cells: dict[str, dict[str, object]] = {}
    for cls in TARGET_CLASSES:
        cells[cls] = _cell(g, x, ref, cls, cs, p)
        if p == PRIMARY_PRIOR:
            cells[cls]["ci_l5"] = _cell(g5, x, ref, cls, cs, p)["ci"]
        if role == "headline":
            _headline(cells[cls], g, x, ref, cls, cs, p, delta_doc)
    entry: dict[str, object] = {"classes": cells, "role": role}
    if role == "headline":
        primary = (p, cs) == (PRIMARY_PRIOR, f"{PRIMARY_COMPLETENESS:.2f}")
        entry["role"] = "headline" if primary else "sensitivity"
        entry["verdict"] = pair_verdict(
            {c: str(cells[c]["class_verdict"]) for c in TARGET_CLASSES}
        )
    return entry


def run_compare(
    ts: TestSet,
    systems: dict[str, ModelData],
    freeze: dict[str, object],
    *,
    lgbm_dir: Path,
    n_boot: int = M.N_BOOT,
) -> dict[str, object]:
    """Block-bootstrap comparison on test (ADR 0006 4.4); thresholds come from the freeze only.

    Returns a dict and writes nothing. A ``verdict`` exists only for HEADLINE_PAIRS.
    """
    delta_path = lgbm_dir / "delta.json"
    if not delta_path.is_file() or sha256_file(delta_path) != freeze.get(
        "delta_sha256"
    ):
        raise refuse(f"delta.json differs from the freeze ({delta_path})")
    delta_doc = json.loads(delta_path.read_text(encoding="utf-8"))["delta"]
    everyone = np.ones(len(ts.y), dtype=bool)
    plan = [
        (everyone, ("P2", "P1"), 3),
        (np.isin(ts.band, P3_BANDS), ("P3",), 3),
        (everyone, ("P2",), 5),
    ]
    groups = [
        (ps, _run_group(ts, systems, freeze, m, ps, lv, n_boot)) for m, ps, lv in plan
    ]
    g5 = groups.pop()[1]  # level-5 descriptive run, P2 only
    out: dict[str, object] = {}
    for x, ref, role in _pairs():
        per_p: dict[str, dict[str, object]] = {}
        for ps, g in groups:
            for p in ps:
                per_p[p] = {
                    cs: _entry(g, g5, x, ref, role, cs, p, delta_doc) for cs in CSTARS
                }
        out[f"{x}-{ref}"] = per_p
    return {"n_boot": n_boot, "seed": M.SEED, "pairs": out}
