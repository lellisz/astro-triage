"""Extract SpecObj DR17 label candidates to Parquet (zstd) after the ADR 0002 basic filters."""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

import pyarrow.parquet as pq

from astro_triage.ingest.manifest import file_sha256
from astro_triage.labels.specobj import apply_basic_filters, load_candidates

DEFAULT_FITS = REPO_ROOT / "data" / "external" / "sdss" / "specObj-dr17.fits"
DEFAULT_OUT = REPO_ROOT / "data" / "external" / "sdss" / "specobj_candidates_v1.parquet"


def _display(path: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return resolved.name


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fits", type=Path, default=DEFAULT_FITS)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args(argv)

    started = time.perf_counter()
    try:
        table = load_candidates(args.fits)
    except (OSError, KeyError) as exc:
        print(f"failed to read {_display(args.fits)}: {exc}", file=sys.stderr)
        return 1
    table, report = apply_basic_filters(table)

    out: Path = args.out
    tmp = out.with_name(out.name + ".tmp")
    try:
        out.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(table, tmp, compression="zstd")
        os.replace(tmp, out)
    except OSError as exc:
        print(
            f"failed to write {_display(out)}: {exc}; check permissions and free disk space",
            file=sys.stderr,
        )
        tmp.unlink(missing_ok=True)
        return 1
    elapsed = time.perf_counter() - started

    for key, value in report.items():
        print(f"{key}: {value}")
    classes = table["class"].value_counts().to_pylist()
    for item in sorted(classes, key=lambda entry: entry["values"]):
        print(f"kept_{item['values']}: {item['counts']}")
    print(f"parquet: {_display(out)}")
    print(f"parquet_bytes: {out.stat().st_size}")
    print(f"parquet_sha256: {file_sha256(out)}")
    print(f"elapsed_seconds: {elapsed:.1f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
