"""Build labels.parquet and features.parquet from local XMatch outputs (100k or chunked; no network)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

import pyarrow as pa

from astro_triage.dataset.build import (
    DEFAULT_CANDIDATES,
    DEFAULT_DATASET_NAME,
    DEFAULT_OUTDIR,
    R_MATCH_ARCSEC,
    BuildError,
    build_dataset,
    format_report,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--r-match",
        type=float,
        default=R_MATCH_ARCSEC,
        help="match radius in arcsec (ADR 0004 G-A2 value by default)",
    )
    parser.add_argument(
        "--outdir",
        type=Path,
        default=None,
        help="output directory (default: data/interim/<dataset-name>)",
    )
    parser.add_argument(
        "--dataset-name",
        default=DEFAULT_DATASET_NAME,
        help="name of the output folder under data/interim/",
    )
    parser.add_argument(
        "--xmatch-dir",
        type=Path,
        default=None,
        help="directory with xmatch_{gaiadr3,paramp}_chunk*.parquet; default: the 100k files",
    )
    parser.add_argument("--candidates", type=Path, default=DEFAULT_CANDIDATES)
    args = parser.parse_args(argv)
    outdir: Path = (
        args.outdir
        if args.outdir is not None
        else DEFAULT_OUTDIR.parent / args.dataset_name
    )
    try:
        result = build_dataset(
            r_match=args.r_match,
            outdir=outdir,
            xmatch_dir=args.xmatch_dir,
            candidates_path=args.candidates,
            dataset_name=args.dataset_name,
        )
    except BuildError as exc:
        print(f"build failed: {exc}", file=sys.stderr)
        return 1
    except (OSError, ValueError, pa.ArrowException) as exc:
        print(f"build failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    print(format_report(result.report))
    for name, path in result.paths.items():
        try:
            shown = path.relative_to(REPO_ROOT).as_posix()
        except ValueError:
            shown = path.name
        print(f"{name}: {shown}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
