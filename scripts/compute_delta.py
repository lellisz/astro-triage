"""Write <release>/lgbm_seeds/delta.json (ADR 0006 section 5, F5) from VAL rows only; refuses to overwrite."""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from astro_triage.dataset.build import DEFAULT_OUTDIR
from astro_triage.evaluation.dl_compare import compute_delta
from astro_triage.evaluation.runner import EvalError

SEEDS_DIRNAME = "lgbm_seeds"
DELTA_NAME = "delta.json"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR)
    args = parser.parse_args(argv)
    typed = args.outdir.as_posix()
    outdir = args.outdir.resolve()
    if not outdir.is_relative_to((REPO_ROOT / "data").resolve()):
        print(f"delta failed: --outdir {typed!r} must be inside data/", file=sys.stderr)
        return 1
    seeds_dir = outdir / SEEDS_DIRNAME
    target = seeds_dir / DELTA_NAME
    data_root = (REPO_ROOT / "data").resolve()
    if not (
        seeds_dir.resolve().is_relative_to(data_root)
        and target.parent.resolve().is_relative_to(data_root)
    ):
        print(
            f"delta failed: {SEEDS_DIRNAME} must resolve inside data/", file=sys.stderr
        )
        return 1
    if target.exists():
        print(f"delta failed: refusing to overwrite {target}", file=sys.stderr)
        return 1
    try:
        doc = compute_delta(outdir, seeds_dir)
        text = json.dumps(doc, indent=2, allow_nan=False)
        fd, tmp = tempfile.mkstemp(dir=seeds_dir, prefix=".delta.", suffix=".tmp")
        os.close(fd)
        try:
            Path(tmp).write_text(text, encoding="utf-8")
            os.link(tmp, target)
        finally:
            Path(tmp).unlink(missing_ok=True)
    except (EvalError, OSError, ValueError) as exc:
        print(f"delta failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    print(f"wrote {target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
