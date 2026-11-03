"""Evaluate M-full / M-noind against the Gaia DSC baselines on the test fold (no network)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

import pyarrow as pa

from astro_triage.dataset.build import DEFAULT_OUTDIR
from astro_triage.evaluation.runner import EvalError, evaluate, write_outputs
from eval.metrics import N_BOOT


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR)
    parser.add_argument("--n-boot", type=int, default=N_BOOT)
    parser.add_argument(
        "--workers", type=int, default=4, help="bootstrap worker processes (1 = serial)"
    )
    args = parser.parse_args(argv)
    try:
        res = evaluate(args.outdir, n_boot=args.n_boot, workers=args.workers)
        paths = write_outputs(res, args.outdir)
    except EvalError as exc:
        print(f"evaluation failed: {exc}", file=sys.stderr)
        return 1
    except (OSError, ValueError, pa.ArrowException) as exc:
        print(f"evaluation failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    meta = res["meta"]
    for path in paths:
        print(path.relative_to(REPO_ROOT).as_posix())
    print(f"bootstrap {meta['bootstrap_seconds']} s, total {meta['total_seconds']} s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
