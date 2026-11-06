"""Train the deep-learning replicates of ADR 0006 into <release>/dl/ (val-only report, GPU session check)."""

from __future__ import annotations

import os

# Must be set before torch is imported anywhere (cuBLAS reads it at first use); ADR 0006 amendment A.
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

import argparse
import dataclasses
import json
import sys
import time
from pathlib import Path
from typing import Literal

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

import numpy as np
import pyarrow as pa

from astro_triage.dataset.build import DEFAULT_OUTDIR
from astro_triage.evaluation.dl_compare import N_SEEDS
from astro_triage.evaluation.dl_val import sha256_file
from astro_triage.evaluation.runner import EvalError
from astro_triage.models.dl_session import (  # noqa: F401  (re-exported, spec S8)
    ENV_NAME,
    check_gpu_session,
    check_pi_src,
    fits_on_disk,
    gpu_facts,
    purity_or_discard,
    record_environment,
    run_ensemble,
    run_freeze,
    save_fit,
    unlink_all,
    verify_gpu,
)
from astro_triage.models.dl_session import read_json as _read_json  # noqa: F401
from astro_triage.models.dl_session import write_json as _write_json  # noqa: F401
from astro_triage.models.lgbm import (
    MODEL_VERSION,
    Fold,
    TrainError,
    load_folds,
)

REPORT_NAME = "train_report_dl.json"
SEEDS_DIRNAME = "lgbm_seeds"
PLATFORMS = ("local", "kaggle", "colab")


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


def strip_test_labels(fold: Fold) -> Fold:
    """Copy of the held-out fold with every label removed (ADR 0006 section 5.1, spec G16)."""
    frame = fold.frame.drop(columns=["sdss_class"])
    return dataclasses.replace(
        fold, labels=np.full(len(fold.labels), -1, dtype=np.int64), frame=frame
    )


def _inside_data(path: Path, typed: str, what: str) -> Path:
    resolved = path.resolve()
    if not resolved.is_relative_to((REPO_ROOT / "data").resolve()):
        raise TrainError(f"{what} {typed!r} must be inside the repo's data/ directory")
    return resolved


def _output_files(dl_dir: Path, model_id: str) -> list[Path]:
    pred = dl_dir / f"predictions_{model_id}_{MODEL_VERSION}.parquet"
    return [
        pred,
        pred.with_name(f"{pred.stem}.manifest.json"),
        dl_dir / "models" / f"{model_id}.pt",
        dl_dir / "models" / f"model_{model_id}.manifest.json",
        dl_dir / "models" / f"{model_id}.preproc.pkl",
    ]


def run(args: argparse.Namespace) -> list[str]:
    if args.ensemble and args.freeze:
        raise TrainError("--ensemble and --freeze are mutually exclusive")
    if args.arch is None and not (args.ensemble or args.freeze):
        raise TrainError("--arch is required for training")
    if args.freeze and (
        args.train_subset is not None or args.plan_c_subsample is not None
    ):
        raise TrainError(
            "freeze needs the full 40-file run; not allowed with subset flags"
        )
    if args.train_subset is not None and args.plan_c_subsample is not None:
        raise TrainError(
            "--train-subset (pilot) and --plan-c-subsample are mutually exclusive"
        )
    if args.device == "cuda" and args.platform is None:
        raise TrainError("--device cuda requires --platform {local,kaggle,colab}")
    for flag in ("train_subset", "plan_c_subsample"):
        if getattr(args, flag) is not None and getattr(args, flag) < 1:
            raise TrainError(f"--{flag.replace('_', '-')} must be >= 1")
    ks = parse_seeds(args.seeds)
    outdir = _inside_data(args.outdir, args.outdir.as_posix(), "--outdir")
    if args.train_subset is not None:
        dl_dir = outdir / "dl" / "pilot"
    elif args.plan_c_subsample is not None:
        dl_dir = outdir / f"dl_sub{args.plan_c_subsample}"
    else:
        dl_dir = outdir / "dl"
    _inside_data(dl_dir, dl_dir.as_posix(), "output directory")
    ckpt_dir = args.ckpt_dir if args.ckpt_dir is not None else dl_dir / "ckpt"
    ckpt_dir = _inside_data(ckpt_dir, ckpt_dir.as_posix(), "--ckpt-dir")
    subset = (
        args.train_subset if args.train_subset is not None else args.plan_c_subsample
    )
    pilot = args.train_subset is not None
    platform = args.platform or "local"
    lgbm_dir = _inside_data(outdir / SEEDS_DIRNAME, SEEDS_DIRNAME, "seeds directory")
    if args.freeze:
        target = run_freeze(outdir, dl_dir, lgbm_dir, args.approval_text)
        return [str(target)]

    env_path = outdir / "dl" / ENV_NAME
    facts = verify_gpu(env_path, platform) if args.device == "cuda" else None

    from astro_triage.models import dl_train

    dl_dir.mkdir(parents=True, exist_ok=True)
    env_path.parent.mkdir(parents=True, exist_ok=True)
    environment = record_environment(env_path, platform, facts)
    report_path = dl_dir / REPORT_NAME
    if args.ensemble:
        nets: list[Literal["mlp", "ftt"]] = [args.arch] if args.arch else ["mlp", "ftt"]
        return run_ensemble(
            outdir, dl_dir, lgbm_dir, nets, ks, report_path, platform, environment
        )
    folds = load_folds(outdir)
    folds["test"] = strip_test_labels(folds["test"])
    done: list[str] = []
    for name in dict.fromkeys(args.features):
        for k in ks:
            spec = dl_train.DLSpec(args.arch, name, k)
            mid = spec.model_id
            pred = dl_dir / f"predictions_{mid}_{MODEL_VERSION}.parquet"
            prior = fits_on_disk(report_path).get(mid)
            if (
                isinstance(prior, dict)
                and pred.exists()
                and prior.get("prediction_sha256") == sha256_file(pred)
            ):
                print(f"{mid}: already trained, skipped", file=sys.stderr)
                continue
            started = time.perf_counter()
            result = dl_train.train_one(
                spec,
                folds,
                dl_dir,
                ckpt_dir,
                device=args.device,
                max_epochs=args.max_epochs,
                patience=args.patience,
                train_subset=subset,
            )
            wall = time.perf_counter() - started
            if not result["beats_prior_logloss"] and not pilot:
                for key in ("prediction_file", "model_file"):
                    result.pop(key, None)
                save_fit(
                    report_path,
                    platform,
                    environment,
                    mid,
                    {**result, "wall_seconds": wall, "status": "failed_T2_beats_prior"},
                )
                stuck = unlink_all(_output_files(dl_dir, mid))
                if stuck:
                    raise TrainError(
                        f"{mid} does not beat the constant-prior val log-loss (T2); "
                        f"could not delete (close programs using them): {', '.join(map(str, stuck))}"
                    )
                raise TrainError(
                    f"{mid} does not beat the constant-prior val log-loss (T2); "
                    f"outputs discarded, entry kept in {REPORT_NAME}"
                )
            if subset is None:
                lgbm_pred = (
                    lgbm_dir / f"predictions_lgbm_{name}_s{k}_{MODEL_VERSION}.parquet"
                )
                check_pi_src(
                    result["train_class_fractions"],
                    lgbm_pred,
                    mid,
                    _output_files(dl_dir, mid),
                )
            purity = purity_or_discard(outdir, pred, mid, _output_files(dl_dir, mid))
            save_fit(
                report_path,
                platform,
                environment,
                mid,
                {
                    **result,
                    "wall_seconds": wall,
                    "prediction_sha256": sha256_file(pred),
                    "val_purity": purity,
                    "subset": subset,
                },
            )
            done.append(mid)
            print(f"{mid}: seed {spec.seed}, wall {wall:.1f}s", file=sys.stderr)
    return done


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR)
    parser.add_argument("--arch", choices=("mlp", "ftt"), default=None)
    parser.add_argument(
        "--features", nargs="+", choices=("full", "noind"), default=["full", "noind"]
    )
    parser.add_argument("--seeds", default="0-4", help="indices 'a-b' or 'a,b' in 0..4")
    parser.add_argument("--max-epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=16)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--platform", choices=PLATFORMS, default=None)
    parser.add_argument("--ckpt-dir", type=Path, default=None)
    parser.add_argument(
        "--train-subset",
        type=int,
        default=None,
        help="pilot: N train rows, writes dl/pilot/",
    )
    parser.add_argument(
        "--plan-c-subsample",
        type=int,
        default=None,
        help="plan C: N train rows, writes dl_sub<N>/",
    )
    parser.add_argument(
        "--ensemble",
        action="store_true",
        help="build ens_<arch>_full_s<k> (both nets without --arch)",
    )
    parser.add_argument(
        "--freeze", action="store_true", help="write dl/val_freeze.json once"
    )
    parser.add_argument(
        "--approval-text",
        default="",
        help="owner's explicit approval text for --freeze",
    )
    args = parser.parse_args(argv)
    try:
        done = run(args)
    except (
        TrainError,
        EvalError,
        OSError,
        ValueError,
        KeyError,
        pa.ArrowException,
    ) as exc:
        kind = "" if isinstance(exc, TrainError) else f"{type(exc).__name__}: "
        print(f"train_dl failed: {kind}{' '.join(str(exc).split())}", file=sys.stderr)
        return 1
    print(json.dumps({"trained": done}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
