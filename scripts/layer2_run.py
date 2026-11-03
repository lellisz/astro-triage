"""Run Layer 2 (NVIDIA API, text only) on the routed objects of one fold (ADR 0005).

Exit codes (ADR 0005 section 17.3):
  0 complete (every routed object terminal; full runs are consolidated)
  1 local error (missing key, locked cache, manifest, I/O); no API outcome
  2 fatal stop: MODEL_UNAVAILABLE, AUTH_ERROR, BAD_REQUEST or DRIFT; do not rerun blindly
  3 QUOTA_STOP: rerun with new approved caps
  4 incomplete and transient: CONSECUTIVE_RETRYABLE, CANARY_INCONCLUSIVE or pending objects
Re-running the same command resumes from the cache. The API key is read by the client
modules only and is never printed.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

import pyarrow as pa

from astro_triage.cost.quota import Ledger, QuotaGuard
from astro_triage.dataset.build import DEFAULT_OUTDIR
from astro_triage.llm.cache import CacheLockedError, ResultCache
from astro_triage.llm.prompt import PROMPT_VERSION
from astro_triage.llm.routing import model_slug
from astro_triage.llm.runner import consolidate, run
from astro_triage.llm.secrets import MissingKeyError, load_api_key, redact

DEFAULT_CACHE_DIR = REPO_ROOT / "data/cache/layer2"
EXIT_LOCAL = 1
EXIT_FATAL = 2
EXIT_QUOTA = 3
EXIT_INCOMPLETE = 4
_FATAL_STOPS = frozenset({"MODEL_UNAVAILABLE", "AUTH_ERROR", "BAD_REQUEST", "DRIFT"})
_TRANSIENT_STOPS = frozenset({"CONSECUTIVE_RETRYABLE", "CANARY_INCONCLUSIVE"})


def _positive(text: str) -> int:
    value = int(text)
    if value <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return value


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release", required=True)
    parser.add_argument("--fold", required=True, choices=("val", "test"))
    parser.add_argument("--model", required=True)
    parser.add_argument("--arm", choices=("blind", "with_l1"), default="blind")
    parser.add_argument(
        "--thinking", action="store_true", help="enable the model's reasoning mode"
    )
    parser.add_argument(
        "--sample",
        type=_positive,
        default=None,
        help="seeded subset of routed objects (pilot); no parquet is written",
    )
    parser.add_argument(
        "--max-calls", type=int, required=True, help="HTTP attempts, retries included"
    )
    parser.add_argument(
        "--max-tokens-total", type=int, required=True, help="prompt + completion tokens"
    )
    parser.add_argument("--rpm-cap", type=float, default=30.0)
    parser.add_argument("--no-canary", action="store_true")
    parser.add_argument(
        "--base", type=Path, default=DEFAULT_OUTDIR.parent, help="parent of <release>/"
    )
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    args = parser.parse_args(argv)
    release_dir = args.base / args.release
    run_id = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    cache_path = args.cache_dir / model_slug(args.model) / f"{PROMPT_VERSION}.jsonl"
    key = ""
    try:
        key = load_api_key()
        guard = QuotaGuard(
            rpm_cap=args.rpm_cap,
            max_calls=args.max_calls,
            max_tokens_total=args.max_tokens_total,
        )
        ledger = Ledger(release_dir / "layer2" / f"calls_{run_id}.jsonl")
        with ResultCache(cache_path) as cache:
            report = run(
                release_dir,
                args.fold,
                args.model,
                args.arm,
                args.thinking,
                guard,
                cache,
                ledger,
                key,
                sample=args.sample,
                canary=not args.no_canary,
                run_id=run_id,
            )
            print(json.dumps(report, indent=2, sort_keys=True))
            stop = report["stop_reason"]
            if stop == "QUOTA_STOP":
                return EXIT_QUOTA
            if stop in _FATAL_STOPS:
                return EXIT_FATAL
            if stop in _TRANSIENT_STOPS or report["pending"]:
                print(f"not consolidated: {report['pending']} pending; rerun later")
                return EXIT_INCOMPLETE
            if args.sample is not None:
                print(f"not consolidated: sample={args.sample} (pilot, no parquet)")
                return 0
            out = consolidate(
                release_dir,
                args.fold,
                args.model,
                args.arm,
                cache,
                enable_thinking=args.thinking,
            )
        print(
            out.relative_to(REPO_ROOT).as_posix()
            if out.is_relative_to(REPO_ROOT)
            else out
        )
    except (
        OSError,
        ValueError,
        KeyError,
        MissingKeyError,
        CacheLockedError,
        pa.ArrowException,
    ) as exc:
        print(
            f"layer2 run failed: {type(exc).__name__}: {redact(str(exc), key)}",
            file=sys.stderr,
        )
        return EXIT_LOCAL
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
