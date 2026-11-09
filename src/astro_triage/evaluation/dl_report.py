"""Pure formatting of the DL comparison result (ADR 0006 4.4, 4.5, 10; spec S13). No IO, no computation.

A verdict is printed only for the headline pairs (H1, H2); ensemble rows and fronts (2)-(4) carry no
verdict and are printed in their own descriptive section.
"""

from __future__ import annotations

import csv
import io
import json
import math
from typing import Final

import numpy as np

from astro_triage.evaluation.dl_compare import (
    HEADLINE_PAIRS,
    MIN_BLOCKS,
    TARGET_CLASSES,
)
from astro_triage.evaluation.runner import LIMITATIONS, _clean, make_header

NEGATIVE_SENTENCE: Final = (
    "On the same 21 Gaia DR3 features, MLP and FT-Transformer (5 seeds, untuned defaults) do not raise "
    "the purity of QSO and GALAXY at completeness 0.80 under the Delchambre+2023 prior beyond 2 "
    "seed-SDs of LightGBM with a block-bootstrap CI above zero."
)
TIE: Final = "tie: LightGBM stays"
HEAD_KEYS: Final = tuple(f"{x}-{ref}" for x, ref in HEADLINE_PAIRS)
PRIMARY: Final = ("P2", "0.80")


def _f(x: object, nd: int = 4) -> str:
    if not isinstance(x, (int, float)) or isinstance(x, bool) or not math.isfinite(x):
        return "n/a"
    return f"{x:.{nd}f}"


def _ci(ci: object) -> str:
    return "n/a" if not isinstance(ci, list) else f"[{_f(ci[0])}, {_f(ci[1])}]"


def _table(header: list[str], rows: list[list[str]]) -> list[str]:
    out = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    return out + ["| " + " | ".join(r) + " |" for r in rows] + [""]


def _span(vals: list[object]) -> str:
    ok = [float(v) for v in vals if isinstance(v, (int, float))]
    return "n/a" if not ok else f"{min(ok):.4g} / {float(np.median(ok)):.4g} / {max(ok):.4g}"


def _reasons(cell: dict) -> list[str]:
    out = []
    if not cell["floor_ok"]:
        out.append("class below the reporting floor on test")
    if cell["blocks"] < MIN_BLOCKS:
        out.append(f"only {cell['blocks']} blocks (< {MIN_BLOCKS})")
    if cell["low_count"]:
        out.append("low-count contaminants (< 10)")
    if cell["na"]:
        out.append(f"CI n/a ({cell['invalid_fraction']:.1%} invalid replicates)")
    if cell["delta"] is None:
        out.append("delta not finite")
    if not cell["val_checks_pass"]:
        out.append("delta's val cells fail the floor/low-count checks")
    return out


def _headline(pairs: dict) -> list[str]:
    rows, bad = [], []
    for key in HEAD_KEYS:
        for p, by_c in pairs[key].items():
            for cs, entry in by_c.items():
                for cls in TARGET_CLASSES:
                    c = entry["classes"][cls]
                    rows.append([key, p, cs, entry["role"], cls, _f(c["d"]), _ci(c["ci"]),
                                 _ci(c.get("ci_l5")), _f(c["delta"]), str(c["reportable"]),
                                 c["class_verdict"], entry["verdict"]])
                    if not c["reportable"]:
                        bad.append(f"- {key} {p} C*={cs} {cls}: {'; '.join(_reasons(c)) or 'unknown'}")
    out = ["## Headline pairs (H1, H2)", "",
           ("A non-reportable cell counts as a tie. `worse` is descriptive and never changes the verdict. "
           "No multiplicity correction is applied across the two headline pairs; requiring both classes "
           "is the only guard against chance wins. Only the P2 / C*=0.80 rows are the headline; "
           "the other rows are sensitivity and never override it."), ""]
    out += _table(["pair", "prior", "C*", "role", "class", "D", "CI (block)", "CI (level-5)", "delta",
                   "reportable", "class verdict", "pair verdict"], rows)
    out += ["### Cells not reportable", "", *(bad or ["none"]), ""]
    verdicts = [pairs[k][PRIMARY[0]][PRIMARY[1]]["verdict"] for k in HEAD_KEYS]
    if all(v == TIE for v in verdicts):
        out += ["### Result", "", NEGATIVE_SENTENCE, ""]
    return out


def _detail(pairs: dict) -> list[str]:
    rows = []
    for key in HEAD_KEYS:
        for cls in TARGET_CLASSES:
            c = pairs[key][PRIMARY[0]][PRIMARY[1]]["classes"][cls]
            for fam, spread in c["purity_seed"].items():
                names = [n for n in c["realised_completeness"] if n.rsplit("_s", 1)[0] == fam]
                cont = {k: _span([c["contaminants"][n][k] for n in names]) for k in c["contaminants"][names[0]]}
                rows.append([key, cls, fam, _span(list(spread.values())),
                             _span([c["realised_completeness"][n] for n in names]),
                             "; ".join(f"{k}: {v}" for k, v in cont.items())])
    return ["### Per system, P2 / C*=0.80 (descriptive, never read by the verdict)", "",
            *_table(["pair", "class", "system", "purity min / med / max",
                     "realised test completeness min / med / max", "contaminants min / med / max"], rows)]


def _descriptive_pairs(pairs: dict) -> list[str]:
    rows = []
    for key, by_p in pairs.items():
        if key in HEAD_KEYS:
            continue
        for cs, entry in by_p["P2"].items():
            for cls in TARGET_CLASSES:
                c = entry["classes"][cls]
                rows.append([key, entry["role"], cs, cls, _f(c["d"]), _ci(c["ci"]), _ci(c.get("ci_l5")),
                             "n/a" if c["na"] else "ok"])
    return ["## Descriptive pairs (ensembles and fronts 2-4; no verdict)", "",
            *_table(["pair", "role", "C*", "class", "D (P2)", "CI (block)", "CI (level-5)", "CI status"], rows)]


def _metrics(desc: dict) -> list[str]:
    keys = ("macro_f1", "accuracy", "ece/mean", "e_sq")
    rows = [[fam, *(f"{_f(m[k]['mean'])} ({_f(m[k]['min'])}-{_f(m[k]['max'])})" for k in keys)]
            for fam, m in desc["families"].items()]
    below = [c for c, f in desc["floor"].items() if not f["ok"]]
    note = f"Classes below the reporting floor on test: {', '.join(below)} (values not for claims)." if below else ""
    return ["## Descriptive metrics (seed mean of the metric, min-max over seeds)", "", desc["regime"], note, "",
            *_table(["system", "macro-F1", "accuracy", "ECE mean", "e_SQ"], rows)]


def _costs(desc: dict) -> list[str]:
    keys = ("epoch_seconds_mean", "epochs_run", "parameter_count", "predict_seconds_per_1m_rows")
    rows = []
    for fam in desc["families"]:
        found = [v for n, v in desc["costs"].items() if n.rsplit("_s", 1)[0] == fam and isinstance(v, dict)]
        rows.append([fam, *(_span([v.get(k) for v in found]) for k in keys)])
    return ["## Costs (copied from train_report_dl.json; min / med / max over seeds)", "",
            *_table(["system", *keys], rows)]


def render_markdown(res: dict) -> str:
    cleaned = _clean(res)
    assert isinstance(cleaned, dict)
    res = cleaned
    meta, pairs = res["meta"], res["compare"]["pairs"]
    lines = [
        "# Deep learning vs LightGBM on the test fold (ADR 0006)", "",
        f"**{make_header(meta['n_test'], meta['n_test_pixels'])}**", "",
        (f"n_boot {res['compare']['n_boot']}, seed {res['compare']['seed']}; freeze sha256 "
        f"{meta['freeze_sha256']}; git {meta['git_sha']}; owner approval: {meta['approval']}"), "",
        *_headline(pairs), *_detail(pairs), *_descriptive_pairs(pairs),
        *_metrics(res["descriptive"]), *_costs(res["descriptive"]),
        "## Limitations", "",
        ("- B0 (ADR 0006 section 6): zero hyperparameter search for both families; the deep models use "
        "untuned defaults and LightGBM a hand-set configuration, so unequal effort is not excluded (R2)."),
        ("- R17: the val-threshold sampling variance is not in the CI; a system whose threshold lands at "
        "a lower test completeness gets higher test purity. The realised test completeness is printed "
        "next to D and is not gated."),
        f"- {LIMITATIONS[0]}",
        "- Ensemble rows are descriptive only; a verdict exists for H1 and H2 alone.", "",
    ]
    return "\n".join(lines)


def to_json(res: dict) -> str:
    return json.dumps(_clean(res), indent=1, allow_nan=False) + "\n"


def risk_coverage_csv(rows: list[list[object]]) -> str:
    buf = io.StringIO(newline="")
    w = csv.writer(buf, lineterminator="\n")
    w.writerow(["system", "seed", "tau", "coverage", "accuracy"])
    w.writerows(rows)
    return buf.getvalue()
