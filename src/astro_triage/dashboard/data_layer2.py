"""Layer 2 (optional) frames and summary built from ``layer2/eval_l2.json``."""

from __future__ import annotations

import math
from typing import Final

import numpy as np
import pandas as pd

from astro_triage.dashboard.data_eval import frame_from_cells
from astro_triage.dashboard.data_release import (
    CLASSES,
)

LAYER2_SYSTEMS: Final[tuple[str, ...]] = ("S0", "S1", "S1*", "S2", "S3")
LAYER2_ROUTED: Final[tuple[str, ...]] = (
    "L1_routed",
    "L2_forced",
    "L1_ok",
    "L2_selective",
)
LAYER2_SHOWN_METRICS: Final[tuple[str, ...]] = ("accuracy", "macro_f1", "coverage")
LAYER2_COST_ROWS: Final = {
    "prompt_tokens": "tokens de entrada",
    "completion_tokens": "tokens de saída",
    "total_tokens": "tokens no total",
    "calls_per_object": "chamadas (tentativas)",
    "latency_ms": "latência (ms)",
}


def layer2_units_frame(units: dict, names: tuple[str, ...]) -> pd.DataFrame:
    rows = []
    for name in names:
        unit = units[name]
        row: dict = {
            "unit": name,
            "n": int(unit["n"]),
            "n_answered": int(unit["n_answered"]),
        }
        for key in LAYER2_SHOWN_METRICS:
            cell = unit["metrics"][key]
            usable = cell.get("status") == "ok" and cell.get("point") is not None
            row[key] = cell["point"] if usable else math.nan
            row[f"{key}_lo"] = (
                cell["lo"] if usable and cell["lo"] is not None else math.nan
            )
            row[f"{key}_hi"] = (
                cell["hi"] if usable and cell["hi"] is not None else math.nan
            )
            row[f"{key}_mark"] = "" if usable else "n/a"
        rows.append(row)
    return pd.DataFrame(rows)


def layer2_diffs_frame(diffs: dict) -> pd.DataFrame:
    frames = []
    for pair, cells in diffs.items():
        shown = {k: cells[k] for k in LAYER2_SHOWN_METRICS}
        frames.append(frame_from_cells(shown, {"pair": pair}))
    out = pd.concat(frames, ignore_index=True)
    out["mark"] = np.where(out["usable"], "", "n/a")
    out["excludes_zero"] = out["usable"] & ((out["lo"] > 0) | (out["hi"] < 0))
    return out


def layer2_summary(layer2: dict) -> dict:
    """Numbers ready to show: verdict of the pre-registered rule, systems, failures and cost."""
    raw = layer2["raw"]
    meta, ops, verdict = raw["meta"], raw["operations"], raw["eval"]["verdict"]
    pairs = []
    for pair, cell in verdict["pairs"].items():
        frame = frame_from_cells({verdict["metric"]: cell}, {"pair": pair})
        frame["evaluable"] = bool(cell["evaluable"])
        frame["gain"] = bool(cell["gain"])
        frame["mark"] = "" if cell["status"] == "ok" else "n/a"
        pairs.append(frame)
    status = pd.DataFrame(
        [
            {
                "status": key,
                "kind": "ok"
                if key == "OK"
                else "unknown"
                if key == "UNKNOWN"
                else "falha",
                "n": int(ops["status_counts"][key]),
                "rate": float(ops["status_rates"][key]),
            }
            for key in sorted(ops["status_counts"])
        ]
    )
    cost = pd.DataFrame(
        [
            {
                "item": label,
                "n": int(ops[key]["n"]),
                **{
                    stat: math.nan if ops[key][stat] is None else float(ops[key][stat])
                    for stat in ("mean", "p50", "p95")
                },
            }
            for key, label in LAYER2_COST_ROWS.items()
        ]
    )
    return {
        "outcome": verdict["outcome"],
        "metric": verdict["metric"],
        "n_evaluable": int(verdict["n_evaluable"]),
        "n_pairs": len(verdict["pairs"]),
        "pairs": pd.concat(pairs, ignore_index=True),
        "systems": layer2["systems"],
        "routed": layer2["routed"],
        "diffs": layer2["diffs"],
        "status": status,
        "unknown_rate": float(ops["unknown_rate"]),
        "failure_rate": float(ops["failure_rate"]),
        "n_routed": int(ops["n_routed"]),
        "class_mix": {c: int(ops["class_mix"].get(c, 0)) for c in CLASSES},
        "cost": cost,
        "model_requested": str(meta["model_requested"]),
        "models_reported": {str(k): int(v) for k, v in ops["models_reported"].items()},
        "prompt_version": str(meta["prompt_version"]),
        "floor": {
            **meta["floor"],
            "min_objects": int(meta["floor"]["min_objects"]),
            "min_pixels": int(meta["floor"]["min_pixels"]),
        },
        "n_boot": int(raw["eval"]["n_boot"]),
    }
