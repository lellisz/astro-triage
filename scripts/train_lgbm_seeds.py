"""Fit the LightGBM seed replicates of ADR 0006 section 3.8 into <release>/lgbm_seeds/ (val-only report)."""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

import pyarrow as pa

from astro_triage.dataset.build import DEFAULT_OUTDIR
from astro_triage.models import lgbm as lgbm_module
from astro_triage.models.lgbm import (
    MODEL_FULL,
    MODEL_NOIND,
    MODEL_VERSION,
    SEED,
    ModelSpec,
    TrainError,
    load_folds,
    run_model,
)

SEEDS_DIRNAME = "lgbm_seeds"
REPORT_NAME = "train_report_lgbm_seeds.json"
SPECS = {"full": MODEL_FULL, "noind": MODEL_NOIND}
N_SEEDS = 5
KEPT = (
    "model_id",
    "best_iteration",
    "temperature",
    "temperature_at_bound",
    "val_calibration_fitted_here",
    "prediction_rows",
    "prediction_file",
    "model_file",
)


def parse_seeds(text: str) -> list[int]:
    """Seed indices k from '0-4' or '0,2,3'; the seed itself is 20261025 + k."""
    try:
        if "-" in text:
            lo, hi = (int(p) for p in text.split("-", 1))
            ks = list(range(lo, hi + 1))
        else:
            ks = [int(p) for p in text.split(",")]
    except ValueError as exc:
        raise TrainError(f"--seeds {text!r} is not 'a-b' or 'a,b,c'") from exc
    if not ks or any(k < 0 or k >= N_SEEDS for k in ks) or len(set(ks)) != len(ks):
        raise TrainError(
            f"--seeds {text!r} must be distinct indices in 0..{N_SEEDS - 1}"
        )
    return ks


def _has_test_key(node: object) -> bool:
    if isinstance(node, dict):
        return any(
            "test" in str(k).lower() or _has_test_key(v) for k, v in node.items()
        )
    if isinstance(node, list):
        return any(_has_test_key(v) for v in node)
    return False


def _write_report(path: Path, fits: dict[str, object]) -> None:
    if _has_test_key(fits):
        raise TrainError(
            "seed report would contain a test-derived key; refusing to write"
        )
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=".report.", suffix=".tmp")
    os.close(fd)
    try:
        Path(tmp).write_text(json.dumps({"fits": fits}, indent=2), encoding="utf-8")
        os.replace(tmp, path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise


def run(
    outdir: Path, plan: list[tuple[str, ModelSpec, int]], measure_one: bool
) -> dict[str, object]:
    typed = outdir.as_posix()
    outdir = outdir.resolve()
    data_root = (lgbm_module.REPO_ROOT / "data").resolve()
    if not outdir.is_relative_to(data_root):
        raise TrainError(
            f"--outdir {typed!r} must be inside the repo's data/ directory"
        )
    seeds_dir = outdir / SEEDS_DIRNAME
    report_path = seeds_dir / REPORT_NAME
    fits: dict[str, object] = {}
    if report_path.exists():
        fits = dict(json.loads(report_path.read_text(encoding="utf-8"))["fits"])
    clash: list[str] = []
    for mid, _, _ in plan:
        pred = seeds_dir / f"predictions_{mid}_{MODEL_VERSION}.parquet"
        candidates = [
            pred,
            seeds_dir / "models" / f"{mid}.txt",
            pred.with_name(f"{pred.stem}.manifest.json"),
            seeds_dir / "models" / f"model_{mid}.manifest.json",
        ]
        clash += [p.relative_to(outdir).as_posix() for p in candidates if p.exists()]
        if mid in fits:
            clash.append(f"{REPORT_NAME}: fits[{mid!r}]")
    if clash:
        raise TrainError(f"refusing to overwrite existing {clash}; pick other --seeds")
    folds = load_folds(outdir)
    seeds_dir.mkdir(exist_ok=True)
    for mid, spec, k in plan:
        started = time.perf_counter()
        result = run_model(spec, folds, seeds_dir, seed=SEED + k, model_id=mid)
        wall = time.perf_counter() - started
        fits[mid] = {
            "seed": SEED + k,
            "wall_seconds": wall,
            **{key: result[key] for key in KEPT},
        }
        _write_report(report_path, fits)
        print(f"{mid}: seed {SEED + k}, wall {wall:.1f}s", file=sys.stderr)
    script = Path(__file__).name
    rel = outdir.relative_to(lgbm_module.REPO_ROOT.resolve()).as_posix()
    note = (
        "measure-one: the other nine fits were NOT run; report wall time to the human "
        f"(H-D6). Next: python scripts/{script} --outdir {rel} --features full "
        f"--seeds 1-4 ; python scripts/{script} --outdir {rel} --features noind "
        "--seeds 0-4"
        if measure_one
        else "done"
    )
    return {"fits_this_run": [mid for mid, _, _ in plan], "status": note}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR)
    parser.add_argument("--features", nargs="+", choices=sorted(SPECS), default=None)
    parser.add_argument("--seeds", default=None, help="indices 'a-b' or 'a,b' in 0..4")
    parser.add_argument(
        "--measure-one",
        action="store_true",
        help="fit only lgbm_full_s0 and report its wall time (H-D6)",
    )
    args = parser.parse_args(argv)
    try:
        if args.measure_one:
            if args.features is not None or args.seeds is not None:
                raise TrainError(
                    "--measure-one fits lgbm_full_s0 only; drop --features/--seeds"
                )
            plan = [("lgbm_full_s0", MODEL_FULL, 0)]
        else:
            ks = parse_seeds(args.seeds if args.seeds is not None else "0-4")
            plan = [
                (f"lgbm_{name}_s{k}", SPECS[name], k)
                for name in dict.fromkeys(args.features or sorted(SPECS))
                for k in ks
            ]
        summary = run(args.outdir, plan, args.measure_one)
    except TrainError as exc:
        print(f"seed training failed: {exc}", file=sys.stderr)
        return 1
    except (OSError, ValueError, KeyError, pa.ArrowException) as exc:
        print(f"seed training failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
