"""Train M-full and M-noind LightGBM models, calibrate on val, write predictions and report."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

import pyarrow as pa

from astro_triage.dataset.build import DEFAULT_OUTDIR
from astro_triage.models.lgbm import TrainError, train_all


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR)
    args = parser.parse_args(argv)
    try:
        report = train_all(args.outdir)
    except TrainError as exc:
        print(f"training failed: {exc}", file=sys.stderr)
        return 1
    except (OSError, ValueError, pa.ArrowException) as exc:
        print(f"training failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
