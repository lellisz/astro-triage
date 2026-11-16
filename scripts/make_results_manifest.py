"""Provenance table (file, bytes, SHA-256, time and its source) for the full_v0 DL comparison results."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from astro_triage.dataset.build import DEFAULT_OUTDIR
from astro_triage.models.lgbm import _atomic

PREFIX = "results manifest refused: "
FIXED = (
    "dl/eval/results_dl.md",
    "dl/eval/results_dl.json",
    "dl/eval/risk_coverage_dl.csv",
    "dl/val_freeze.json",
    "dl/test_opened.json",
    "dl/train_report_dl.json",
    "dl/environment.json",
    "lgbm_seeds/delta.json",
    "lgbm_seeds/train_report_lgbm_seeds.json",
)
GLOBS = ("dl/predictions_*.parquet", "lgbm_seeds/predictions_*.parquet")
ABSENT = "AUSENTE"
NOTE = (
    'Os manifests das predicoes trazem `release: "dataset_100k_v0"` (item G15 em '
    "`docs/specs/dl-implementation.md` e ADR 0006 §17); este arquivo recalcula todos os "
    "SHA-256 como verificacao independente, em vez de confiar no campo `file_sha256` dos manifests."
)
GIT_SHA_RE = re.compile(r"[0-9a-f]{40}")
SHA256_RE = re.compile(r"[0-9a-f]{64}")


class ManifestError(Exception):
    pass


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        while chunk := fh.read(1 << 20):
            h.update(chunk)
    return h.hexdigest()


def _iso(ts: float) -> str:
    return datetime.fromtimestamp(ts, UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def _json_field(path: Path, *keys: str) -> object:
    try:
        node: object = json.loads(path.read_text(encoding="utf-8"))
        for key in keys:
            node = node[key]  # type: ignore[index]
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise ManifestError(
            f"cannot read {'.'.join(keys)} from {path}: {type(exc).__name__}: {exc}"
        ) from exc
    return node


def _hex(value: object, pattern: re.Pattern[str], label: str) -> str:
    if not isinstance(value, str) or not pattern.fullmatch(value):
        raise ManifestError(f"{label} is not a valid lowercase hex digest")
    return value


def _utc(value: object, label: str) -> str:
    if not isinstance(value, str):
        raise ManifestError(f"{label} is not an ISO-8601 string")
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError as exc:
        raise ManifestError(f"{label} is not a valid ISO-8601 timestamp") from exc
    if parsed.utcoffset() != timedelta(0):
        raise ManifestError(f"{label} must carry a UTC offset (Z or +00:00)")
    return _iso(parsed.timestamp())


def _time(out: Path, rel: str) -> tuple[str, str]:
    path = out / rel
    if rel == "dl/test_opened.json":
        return _utc(_json_field(path, "opened_at"), "opened_at"), "opened_at"
    if rel.endswith(".parquet"):
        sidecar = path.with_suffix(".manifest.json")
        if sidecar.exists():
            return (
                _utc(_json_field(sidecar, "obtained_utc"), "obtained_utc"),
                "obtained_utc",
            )
    return _iso(path.stat().st_mtime), "mtime"


def _row(out: Path, rel: str) -> tuple[str, bool]:
    path = out / rel
    if not path.is_file():
        return f"| `{rel}` | {ABSENT} | {ABSENT} | {ABSENT} | {ABSENT} |", False
    when, source = _time(out, rel)
    return (
        f"| `{rel}` | {path.stat().st_size} | `{_sha256(path)}` | {when} | {source} |",
        True,
    )


def build(out: Path) -> tuple[str, int]:
    results = out / "dl" / "eval" / "results_dl.json"
    git_sha = _hex(_json_field(results, "meta", "git_sha"), GIT_SHA_RE, "meta.git_sha")
    freeze_sha = _hex(
        _json_field(results, "meta", "freeze_sha256"), SHA256_RE, "meta.freeze_sha256"
    )
    rels = list(FIXED)
    for pattern in GLOBS:
        rels += sorted(p.relative_to(out).as_posix() for p in out.glob(pattern))
    rows = [_row(out, rel) for rel in rels]
    lines = [
        "# Proveniencia dos resultados full_v0",
        "",
        f"- git_sha: `{git_sha}`",
        f"- freeze_sha256: `{freeze_sha}`",
        f"- gerado em: {_iso(datetime.now(UTC).timestamp())}",
        f"- {NOTE}",
        "",
        "| arquivo | bytes | sha256 | tempo (UTC) | fonte do tempo |",
        "|---|---|---|---|---|",
        *(line for line, _ in rows),
        "",
    ]
    return "\n".join(lines), sum(1 for _, ok in rows if not ok)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR)
    ap.add_argument(
        "--out", type=Path, required=True, help="markdown file to write (outside data/)"
    )
    a = ap.parse_args(argv)
    out, target = a.outdir.resolve(), a.out.resolve()
    data = (REPO_ROOT / "data").resolve()
    try:
        if not out.is_relative_to(data):
            raise ManifestError("--outdir must resolve inside the repo's data/")
        if target.is_relative_to(data):
            raise ManifestError("--out must be outside the repo's data/")
        if target.exists():
            raise ManifestError(f"{target} already exists, nothing is overwritten")
        if not target.parent.is_dir():
            raise ManifestError(f"directory of --out does not exist: {target.parent}")
        text, missing = build(out)
        _atomic(
            target,
            lambda tmp: Path(tmp).write_text(text, encoding="utf-8", newline="\n"),
        )
    except ManifestError as exc:
        print(PREFIX + str(exc), file=sys.stderr)
        return 1
    except OSError as exc:
        print(f"results manifest failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    print(target)
    if missing:
        print(f"{PREFIX}{missing} file(s) marked {ABSENT}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
