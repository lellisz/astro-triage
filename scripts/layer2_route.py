"""Build routing_v1.json (tau from the val fold only; ADR 0005 D2). The test fold is never read."""

from __future__ import annotations

import argparse
import sys
from datetime import UTC, datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

import numpy as np
import pandas as pd
import pyarrow as pa

from astro_triage.abstention.policy import (
    CLASSES,
    choose_threshold,
    l1_confidence,
    score_columns,
    select_val,
    sha256_file,
    write_routing_manifest,
)
from astro_triage.dataset.build import DEFAULT_OUTDIR

MANIFEST_VERSION = "routing_v1"
A_KEEP_L2_DEFAULT = 0.90  # tau2 is fitted later, on val Layer 2 answers (ADR D4.5)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release", required=True)
    parser.add_argument("--a-keep", type=float, default=0.99)
    parser.add_argument("--r-max", type=float, default=0.15)
    parser.add_argument(
        "--base", type=Path, default=DEFAULT_OUTDIR.parent, help="parent of <release>/"
    )
    args = parser.parse_args(argv)
    root = args.base / args.release
    out = root / "layer2" / f"{MANIFEST_VERSION}.json"
    try:
        if out.exists():
            raise FileExistsError(
                f"{out} exists; a frozen manifest is replaced only by a new version"
            )
        matches = sorted(root.glob("predictions_lgbm_full_*.parquet"))
        if len(matches) != 1:
            raise FileNotFoundError(
                f"expected one predictions_lgbm_full_*.parquet in {root}, found {len(matches)}"
            )
        pred_path = matches[0]
        # The fold filter is applied at read time: test rows are never loaded.
        labels = pd.read_parquet(
            root / "labels.parquet",
            columns=["source_id", "sdss_class", "fold"],
            filters=[("fold", "==", "val")],
        )
        val = select_val(labels)
        pred = pd.read_parquet(
            pred_path, columns=["source_id", *score_columns()]
        ).merge(val[["source_id", "sdss_class"]], on="source_id", how="inner")
        if len(pred) != len(val):
            raise ValueError(
                f"predictions cover {len(pred)} of {len(val)} val objects; refusing to fit"
            )
        # score_columns() follows CLASSES order and pred is selected by name, so argmax maps to CLASSES.
        probs = pred[score_columns()].to_numpy(dtype=float)
        pred_label = np.array(CLASSES)[probs.argmax(axis=1)]
        correct = pred_label == pred["sdss_class"].astype(str).str.upper().to_numpy()
        res = choose_threshold(
            l1_confidence(pred).to_numpy(), correct, args.a_keep, args.r_max
        )
        payload = {
            "version": MANIFEST_VERSION,
            "release": args.release,
            "tau": res["tau"],
            "tau_from_a_keep": res["tau_from_a_keep"],
            "r_max_conflict": res["conflict"],
            "a_keep": args.a_keep,
            "r_max": args.r_max,
            "grid": {"min": 0.50, "max": 0.99, "step": 0.01},
            "val_n": int(res["n"]),
            "val_routed_fraction": res["routed_fraction"],
            "val_accuracy_kept": res["accuracy_kept"],
            "tau2": None,
            "a_keep2": A_KEEP_L2_DEFAULT,
            "verbal_calibration": None,
            "predictions_file": pred_path.name,
            "predictions_sha256": sha256_file(pred_path),
            "created_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        }
        write_routing_manifest(out, payload)
    except (OSError, ValueError, KeyError, pa.ArrowException) as exc:
        print(f"routing failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    print(
        out.relative_to(REPO_ROOT).as_posix() if out.is_relative_to(REPO_ROOT) else out
    )
    print(
        f"tau={res['tau']:.2f} routed={res['routed_fraction']:.4f} "
        f"kept_acc={res['accuracy_kept']:.4f} conflict={res['conflict']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
