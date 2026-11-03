"""Pilot report for Layer 2 from the result cache and the ledgers (ADR 0005 17.7; no network).

Reads the val pilot sample (``--sample``, the same seeded draw as ``layer2_run.py``) and writes
``pilot_report.json`` and ``pilot_report.md`` next to the routing manifest. Exit codes: 0 written,
1 refused or failed.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

import pyarrow as pa

from astro_triage.dataset.build import DEFAULT_OUTDIR
from astro_triage.evaluation.layer2_pilot import (
    DEFAULT_CACHE_DIR,
    pilot_report,
    render_pilot_markdown,
)
from astro_triage.evaluation.layer2_report import to_jsonable


def _positive(text: str) -> int:
    value = int(text)
    if value <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return value


def _write_atomic(path: Path, text: str) -> None:
    tmp = path.with_name(path.name + ".tmp")
    try:
        tmp.write_text(text, encoding="utf-8")
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release", required=True)
    parser.add_argument("--models", nargs="+", required=True)
    parser.add_argument(
        "--arms", nargs="+", choices=("blind", "with_l1"), required=True
    )
    parser.add_argument("--thinking-models", nargs="*", default=[])
    parser.add_argument("--sample", type=_positive, required=True)
    parser.add_argument("--target-release", required=True)
    parser.add_argument("--rpm-cap", type=float, default=30.0)
    parser.add_argument(
        "--base", type=Path, default=DEFAULT_OUTDIR.parent, help="parent of <release>/"
    )
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    args = parser.parse_args(argv)
    if args.rpm_cap <= 0:
        parser.error("--rpm-cap must be > 0")
    root = args.base / args.release
    try:
        rep = pilot_report(
            root,
            args.models,
            args.arms,
            args.thinking_models,
            args.sample,
            args.base / args.target_release,
            args.rpm_cap,
            cache_dir=args.cache_dir,
        )
        text = render_pilot_markdown(rep)
        payload = json.dumps(to_jsonable(rep), indent=1, allow_nan=False)
        outdir = root / "layer2"
        outdir.mkdir(parents=True, exist_ok=True)
        _write_atomic(outdir / "pilot_report.json", payload)
        _write_atomic(outdir / "pilot_report.md", text)
    except (OSError, ValueError, KeyError, pa.ArrowException) as exc:
        print(
            f"layer2 pilot report failed: {type(exc).__name__}: {exc}", file=sys.stderr
        )
        return 1
    for name in ("pilot_report.json", "pilot_report.md"):
        path = outdir / name
        print(
            path.relative_to(REPO_ROOT).as_posix()
            if path.is_relative_to(REPO_ROOT)
            else path.name
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
