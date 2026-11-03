"""Evaluate Layer 2 against Layer 1 on the test fold (ADR 0005 D7; no network, BALANCED only).

The verbal calibration, tau2 and tau* are fitted on the val fold only; the test fold is
evaluated with them. The evaluation is refused when any routed object lacks a terminal Layer 2
record or when the routing manifest does not match the predictions on disk.
Writes eval_l2.json and eval_l2.md (latest run) next to the routing manifest, plus a copy per
run named eval_l2_<arm>_think-<on|off>.{json,md} so another arm/thinking run does not lose it.
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

import pyarrow as pa

from astro_triage.dataset.build import DEFAULT_OUTDIR
from astro_triage.evaluation.layer2_data import run_evaluation
from astro_triage.evaluation.layer2_report import write_outputs
from astro_triage.evaluation.runner import EvalError
from eval.metrics import N_BOOT


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--arm", choices=("blind", "with_l1"), default="blind")
    parser.add_argument("--n-boot", type=int, default=N_BOOT)
    parser.add_argument(
        "--thinking",
        action="store_true",
        help="evaluate the run made with the model's reasoning mode enabled",
    )
    parser.add_argument(
        "--base", type=Path, default=DEFAULT_OUTDIR.parent, help="parent of <release>/"
    )
    args = parser.parse_args(argv)
    root = args.base / args.release
    try:
        res = run_evaluation(
            root, args.model, args.arm, args.thinking, n_boot=args.n_boot
        )
        paths = write_outputs(res, root / "layer2")
        # eval_l2.* stay the "latest" copy (what the dashboard reads); keep one per arm/thinking
        tag = f"{args.arm}_think-{'on' if args.thinking else 'off'}"
        for path in paths:
            shutil.copyfile(path, path.with_name(f"{path.stem}_{tag}{path.suffix}"))
    except EvalError as exc:
        print(f"layer2 evaluation refused: {exc}", file=sys.stderr)
        return 1
    except (OSError, ValueError, KeyError, pa.ArrowException) as exc:
        print(f"layer2 evaluation failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    for path in paths:
        print(
            path.relative_to(REPO_ROOT).as_posix()
            if path.is_relative_to(REPO_ROOT)
            else path.name
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
