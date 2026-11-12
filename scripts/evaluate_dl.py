"""Single test exposure of the DL comparison (ADR 0006 5.2 and 14). Needs the owner's approval text (A6)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

import pyarrow as pa

from astro_triage.dataset.build import DEFAULT_OUTDIR
from astro_triage.evaluation import dl_desc, dl_eval, dl_report, dl_run
from astro_triage.evaluation.runner import EvalError
from astro_triage.models.lgbm import _atomic
from eval.metrics import N_BOOT

PREFIX = "dl evaluation refused: "


def _put(path: Path, text: str) -> None:
    _atomic(path, lambda tmp: Path(tmp).write_text(text, encoding="utf-8", newline="\n"))


def _free(paths: list[Path]) -> None:
    if taken := [p.name for p in paths if p.exists()]:
        raise EvalError(f"outputs already exist, nothing is overwritten: {taken}")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR)
    ap.add_argument("--n-boot", type=int, default=N_BOOT)
    ap.add_argument("--workers", type=int, default=4, help="threads, descriptive metrics")
    ap.add_argument("--dl-dirname", default="dl")
    ap.add_argument("--seeds-dirname", default="lgbm_seeds")
    ap.add_argument("--approval", required=True, help="owner's explicit approval text (A6)")
    a = ap.parse_args(argv)
    out = a.outdir.resolve()
    dl, lg = out / a.dl_dirname, out / a.seeds_dirname
    paths = [dl / "eval" / n for n in ("results_dl.json", "results_dl.md", "risk_coverage_dl.csv")]
    try:
        if not all(p.resolve().is_relative_to((REPO_ROOT / "data").resolve()) for p in (out, dl, lg)):
            raise EvalError("--outdir and the dirnames must resolve inside the repo's data/")
        _free(paths)
        costs = dl_desc.load_costs(dl)
        ts, freeze = dl_eval.open_test_set(out, dl, lg, a.approval)
        systems = dl_eval.load_dl_systems(out, ts, dl_dir=dl, lgbm_dir=lg)
        cmp = dl_run.run_compare(ts, systems, freeze, lgbm_dir=lg, n_boot=a.n_boot)
        desc, rows = dl_desc.describe(ts, systems, costs, workers=a.workers)
        res = dl_desc.assemble(ts, cmp, desc, str(freeze["_file_sha256"]), a.approval)
        texts = [dl_report.to_json(res), dl_report.render_markdown(res), dl_report.risk_coverage_csv(rows)]
        _free(paths)
        paths[0].parent.mkdir(exist_ok=True)
        try:
            for path, text in zip(paths, texts, strict=True):
                _put(path, text)
        except BaseException:
            for path in paths:  # _free above guarantees none existed before this run
                path.unlink(missing_ok=True)
            raise
        meta = res["meta"]
        dl_eval.write_marker(dl, freeze_sha=meta["freeze_sha256"], git_sha=meta["git_sha"])
    except EvalError as exc:
        print(str(exc) if str(exc).startswith(PREFIX) else PREFIX + str(exc), file=sys.stderr)
        return 1
    except (OSError, ValueError, pa.ArrowException) as exc:
        print(f"dl evaluation failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    print(*(p.relative_to(REPO_ROOT).as_posix() for p in paths), sep="\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
