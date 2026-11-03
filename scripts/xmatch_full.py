"""Cross-match ALL SpecObj candidates with Gaia DR3 through CDS XMatch, in resumable chunks.

Chunk ``i`` is the candidate rows ``[i*N, (i+1)*N)`` of the Parquet file, in file order. Per
chunk and per catalogue (gaiadr3, paramp) there is ONE https request (selection=all, text id
with prefix 'S', see ADR 0004 D3). Each answer is validated, written as an atomic zstd Parquet
``xmatch_<catalogue>_chunk<iii>.parquet`` plus a manifest, and an existing valid file is
skipped, so a rerun resumes. A request that fails after the retries of ``xmatch.py`` stops
the whole run (no extra requests) and leaves the finished chunks in place.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Final

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
from xmatch_batch import (
    DEFAULT_CANDIDATES,
    PASSES,
    SID_PREFIX,
    make_manifest,
    write_parquet,
    write_upload_csv,
)

from astro_triage.ingest.manifest import file_sha256
from astro_triage.ingest.xmatch import (
    XMATCH_MAX_ROWS,
    IngestError,
    read_xmatch_csv,
    xmatch_to_file,
)

DEFAULT_OUTDIR = REPO_ROOT / "data" / "interim" / "xmatch_full"
PLAN_NAME: Final = "xmatch_full_plan.json"
REPORT_NAME: Final = "xmatch_full_report.json"
MAX_CHUNK_SIZE: Final = (
    1_500_000  # keeps the upload CSV (about 55 bytes/row) under 100 MB
)
MAX_UPLOAD_BYTES: Final = 95 * 1024 * 1024
# Output rows per uploaded row of the 100k batch at 5 arcsec (52,524 and 51,850 per 100,000).
# A smaller radius returns fewer rows, so this is an upper bound.
ROWS_PER_SENT: Final = 52_524 / 100_000
MAX_ESTIMATED_ROWS: Final = 1_200_000
ECHO_TOLERANCE_ARCSEC: Final = 1e-5
SELECTION: Final = "all"
CANDIDATE_COLUMNS: Final[tuple[str, ...]] = (
    "specobjid",
    "plug_ra",
    "plug_dec",
    "class",
)


class XMatchFullError(Exception):
    """The run cannot continue; the message says what to do. Finished chunks stay on disk."""


@dataclass(frozen=True)
class Plan:
    n_candidates: int
    chunk_size: int
    radius: float
    candidates_sha256: str

    @property
    def n_chunks(self) -> int:
        return math.ceil(self.n_candidates / self.chunk_size)

    def bounds(self, index: int) -> tuple[int, int]:
        start = index * self.chunk_size
        return start, min(start + self.chunk_size, self.n_candidates)

    def as_json(self) -> dict[str, object]:
        return {
            "n_candidates": self.n_candidates,
            "chunk_size": self.chunk_size,
            "radius_arcsec": self.radius,
            "selection": SELECTION,
            "candidates_sha256": self.candidates_sha256,
        }


def effective_chunk_size(requested: int) -> int:
    """Validate ``requested`` and shrink it when the estimated output passes the row budget."""
    if requested <= 0:
        raise XMatchFullError("--chunk-size must be positive")
    if requested > MAX_CHUNK_SIZE:
        raise XMatchFullError(
            f"--chunk-size {requested} exceeds {MAX_CHUNK_SIZE}: the upload would pass "
            "the 100 MB limit of CDS XMatch"
        )
    if requested * ROWS_PER_SENT > MAX_ESTIMATED_ROWS:
        return int(MAX_ESTIMATED_ROWS / ROWS_PER_SENT)
    return requested


def chunk_path(outdir: Path, label: str, index: int) -> Path:
    return outdir / f"xmatch_{label}_chunk{index:03d}.parquet"


def is_valid_chunk(path: Path) -> int:
    """Return the row count of a usable chunk file, or 0 when absent, unreadable or empty."""
    try:
        meta = pq.ParquetFile(path).metadata
        names = set(meta.schema.names)
    except (OSError, pa.ArrowException):
        return 0
    if not {"sid", "Source"} <= names:
        return 0
    return int(meta.num_rows)


def load_candidates(path: Path) -> pa.Table:
    try:
        table = pq.read_table(path, columns=list(CANDIDATE_COLUMNS))
    except (OSError, KeyError, pa.ArrowException) as exc:
        raise XMatchFullError(
            f"cannot read candidates {path.name}: {exc}; run scripts/extract_specobj.py"
        ) from exc
    if table.num_rows == 0:
        raise XMatchFullError(f"{path.name} has no rows")
    if pc.count_distinct(table["specobjid"]).as_py() != table.num_rows:
        raise XMatchFullError(f"{path.name} has duplicated specobjid")
    return table


def chunk_frame(table: pa.Table, start: int, stop: int) -> pd.DataFrame:
    sub = table.slice(start, stop - start).to_pandas()
    return pd.DataFrame(
        {
            "sid": SID_PREFIX + sub["specobjid"].astype(str),
            "ra": sub["plug_ra"].astype("float64"),
            "dec": sub["plug_dec"].astype("float64"),
            "cls": sub["class"].astype(str),
        }
    )


def validate_answer(table: pa.Table, sent: pd.DataFrame, radius: float) -> None:
    """Ids, class and echoed positions must equal the upload (ADR 0004 D3 assertion)."""
    if table.num_rows >= XMATCH_MAX_ROWS:
        raise XMatchFullError(f"answer has {table.num_rows} rows: truncated at the cap")
    needed = {"sid", "ra", "dec", "cls", "angDist", "Source"}
    missing = needed - set(table.column_names)
    if missing:
        raise XMatchFullError(f"answer lacks columns {sorted(missing)}")
    if table["Source"].null_count:
        raise XMatchFullError("answer has NULL Source values")
    got = table.select(["sid", "ra", "dec", "cls", "angDist"]).to_pandas()
    expected = sent.set_index("sid")
    unknown = ~got["sid"].isin(expected.index)
    if unknown.any():
        raise XMatchFullError(
            f"{int(unknown.sum())} returned sid are not in the upload"
        )
    ref = expected.loc[got["sid"]]
    if not (ref["cls"].to_numpy() == got["cls"].to_numpy()).all():
        raise XMatchFullError("returned class differs from the upload")
    worst = 0.0
    if len(got):
        diff_ra = np.abs(ref["ra"].to_numpy() - got["ra"].to_numpy()).max()
        diff_dec = np.abs(ref["dec"].to_numpy() - got["dec"].to_numpy()).max()
        worst = float(max(diff_ra, diff_dec)) * 3600.0
    if worst > ECHO_TOLERANCE_ARCSEC:
        raise XMatchFullError(
            f"echoed positions differ from the upload by up to {worst:.2e} arcsec"
        )
    if len(got) and float(got["angDist"].max()) > radius + 1e-6:
        raise XMatchFullError("a returned pair is farther than the requested radius")


def _write_json(path: Path, payload: dict[str, object]) -> None:
    fd, tmp_name = tempfile.mkstemp(dir=path.parent, prefix=".json.", suffix=".tmp")
    os.close(fd)
    try:
        Path(tmp_name).write_text(
            json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        os.replace(tmp_name, path)
    except BaseException:
        Path(tmp_name).unlink(missing_ok=True)
        raise


def check_plan(outdir: Path, plan: Plan) -> None:
    """Refuse to mix chunks made with another chunk size, radius or candidates file."""
    path = outdir / PLAN_NAME
    if path.exists():
        try:
            old = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise XMatchFullError(f"cannot read {PLAN_NAME}: {exc}") from exc
        if old != plan.as_json():
            raise XMatchFullError(
                f"{PLAN_NAME} differs from this run (chunk size, radius or candidates "
                "changed); use a new --outdir or restore the old arguments"
            )
    else:
        _write_json(path, plan.as_json())


def clean_leftovers(*dirs: Path) -> None:
    for directory in dirs:
        for pattern in (".*.tmp", ".upload_*.csv", ".xmatch_*.raw.csv"):
            for leftover in directory.glob(pattern):
                leftover.unlink(missing_ok=True)


def request_chunk(
    plan: Plan,
    index: int,
    label: str,
    catalog: str,
    sent: pd.DataFrame,
    upload: Path,
    upload_sha: str,
    outdir: Path,
    workdir: Path,
) -> dict[str, object]:
    raw = workdir / f".xmatch_{label}_chunk{index:03d}.raw.csv"
    started = time.perf_counter()
    try:
        stats = xmatch_to_file(
            upload,
            raw,
            catalog=catalog,
            radius_arcsec=plan.radius,
            selection=SELECTION,
            ra_col="ra",
            dec_col="dec",
        )
        table = read_xmatch_csv(raw)
        validate_answer(table, sent, plan.radius)
        start, stop = plan.bounds(index)
        parquet = chunk_path(outdir, label, index)
        write_parquet(table, parquet)
        params = (
            f"request=xmatch; distMaxArcsec={plan.radius:g}; selection={SELECTION}; "
            f"RESPONSEFORMAT=csv; cat2={catalog}; colRA1=ra; colDec1=dec; "
            f"cat1=<upload csv, columns sid,ra,dec,cls, {len(sent)} rows, sid prefix "
            f"'{SID_PREFIX}', sha256 {upload_sha}, candidate rows [{start}, {stop})>"
        )
        make_manifest(
            parquet,
            name=parquet.stem,
            params=params,
            rows=table.num_rows,
            notes=[
                f"bytes_received={stats.bytes_received}",
                f"attempts={stats.attempts}",
                f"candidates_sha256={plan.candidates_sha256}",
                f"chunk_index={index}; chunk_size={plan.chunk_size}",
            ],
        )
        rows_returned = table.num_rows
    finally:
        raw.unlink(missing_ok=True)
    del table
    gc.collect()
    return {
        "chunk": index,
        "catalog": label,
        "rows_sent": len(sent),
        "rows_returned": rows_returned,
        "bytes": stats.bytes_received,
        "seconds": round(time.perf_counter() - started, 1),
        "attempts": stats.attempts,
    }


def coverage_check(
    plan: Plan, table: pa.Table, outdir: Path
) -> tuple[dict[str, object], list[str]]:
    """Every candidate is in exactly one chunk; every returned object belongs to its chunk."""
    errors: list[str] = []
    sent_total = 0
    matched: dict[str, int] = {label: 0 for label, _ in PASSES}
    seen: dict[str, set[str]] = {label: set() for label, _ in PASSES}
    for index in range(plan.n_chunks):
        start, stop = plan.bounds(index)
        sent_total += stop - start
        ids = set(chunk_frame(table, start, stop)["sid"])
        for label, _ in PASSES:
            found = set(
                pq.read_table(chunk_path(outdir, label, index), columns=["sid"])
                .column("sid")
                .to_pylist()
            )
            if not found <= ids:
                errors.append(f"chunk {index} {label}: objects outside the chunk")
            if found & seen[label]:
                errors.append(
                    f"chunk {index} {label}: object already seen in another chunk"
                )
            seen[label] |= found
            matched[label] += len(found)
    if sent_total != plan.n_candidates:
        errors.append(f"chunks cover {sent_total} of {plan.n_candidates} candidates")
    summary: dict[str, object] = {
        "candidates_sent": sent_total,
        "matched_objects": matched,
        "match_rate": {
            label: matched[label] / sent_total if sent_total else None
            for label in matched
        },
    }
    return summary, errors


def run(args: argparse.Namespace) -> dict[str, object]:
    outdir: Path = args.outdir
    workdir: Path = args.workdir if args.workdir is not None else outdir / ".work"
    chunk_size = effective_chunk_size(args.chunk_size)
    if not 0 < args.radius <= 180:
        raise XMatchFullError("--radius must be in (0, 180] arcsec")
    outdir.mkdir(parents=True, exist_ok=True)
    workdir.mkdir(parents=True, exist_ok=True)
    clean_leftovers(outdir, workdir)

    table = load_candidates(args.candidates)
    plan = Plan(table.num_rows, chunk_size, args.radius, file_sha256(args.candidates))
    check_plan(outdir, plan)
    max_requests = plan.n_chunks * len(PASSES)
    estimate = int(chunk_size * ROWS_PER_SENT)
    note = "" if chunk_size == args.chunk_size else f" (reduced from {args.chunk_size})"
    print(
        f"[candidatos] {plan.n_candidates} rows; chunk_size={chunk_size}{note}; "
        f"chunks={plan.n_chunks}; max requests={max_requests}; estimated output "
        f"<= {estimate} rows per chunk and catalogue (budget {MAX_ESTIMATED_ROWS}, "
        f"cap {XMATCH_MAX_ROWS})",
        flush=True,
    )

    results: list[dict[str, object]] = []
    issued = 0
    for index in range(plan.n_chunks):
        start, stop = plan.bounds(index)
        pending = [
            (label, catalog)
            for label, catalog in PASSES
            if is_valid_chunk(chunk_path(outdir, label, index)) == 0
        ]
        for label, _ in PASSES:
            tag = f"[xmatch lote {index + 1}/{plan.n_chunks} {label}]"
            rows = is_valid_chunk(chunk_path(outdir, label, index))
            if rows:
                print(f"{tag} skip: valid file with {rows} rows", flush=True)
        if not pending:
            continue
        sent = chunk_frame(table, start, stop)
        upload = workdir / f".upload_chunk{index:03d}.csv"
        write_upload_csv(sent, upload)
        try:
            if upload.stat().st_size > MAX_UPLOAD_BYTES:
                raise XMatchFullError(
                    f"upload of chunk {index} is {upload.stat().st_size} bytes; "
                    "lower --chunk-size"
                )
            upload_sha = file_sha256(upload)
            for label, catalog in pending:
                tag = f"[xmatch lote {index + 1}/{plan.n_chunks} {label}]"
                if issued >= max_requests:
                    raise XMatchFullError("request budget exhausted; stopping")
                if issued:
                    time.sleep(args.sleep)
                issued += 1
                try:
                    result = request_chunk(
                        plan,
                        index,
                        label,
                        catalog,
                        sent,
                        upload,
                        upload_sha,
                        outdir,
                        workdir,
                    )
                except IngestError as exc:
                    raise XMatchFullError(
                        f"{tag} request failed after the retries: {exc}. Finished chunks "
                        "are kept; rerun the same command to resume"
                    ) from exc
                results.append(result)
                print(
                    f"{tag} rows_sent={result['rows_sent']} "
                    f"rows_returned={result['rows_returned']} bytes={result['bytes']} "
                    f"seconds={result['seconds']}",
                    flush=True,
                )
        finally:
            upload.unlink(missing_ok=True)
        del sent
        gc.collect()

    summary, errors = coverage_check(plan, table, outdir)
    report: dict[str, object] = {
        "plan": plan.as_json(),
        "requests_this_run": issued,
        "results_this_run": results,
        "coverage": summary,
        "coverage_errors": errors,
    }
    _write_json(outdir / REPORT_NAME, report)
    print(
        f"[xmatch] requests this run: {issued}; "
        f"rows returned this run: {sum(int(str(r['rows_returned'])) for r in results)}; "
        f"bytes: {sum(int(str(r['bytes'])) for r in results)}",
        flush=True,
    )
    print(f"[xmatch] coverage: {json.dumps(summary)}", flush=True)
    if errors:
        raise XMatchFullError("coverage check failed: " + "; ".join(errors))
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidates", type=Path, default=DEFAULT_CANDIDATES)
    parser.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR)
    parser.add_argument(
        "--workdir",
        type=Path,
        default=None,
        help="scratch directory for the upload CSV and raw answers (default: <outdir>/.work)",
    )
    parser.add_argument("--chunk-size", type=int, default=1_000_000)
    parser.add_argument("--radius", type=float, default=2.0)
    parser.add_argument("--sleep", type=float, default=20.0)
    args = parser.parse_args(argv)
    if args.sleep < 0:
        parser.error("--sleep must be >= 0")
    try:
        run(args)
    except XMatchFullError as exc:
        print(f"xmatch_full stopped: {exc}", file=sys.stderr)
        return 1
    except (OSError, ValueError, pa.ArrowException) as exc:
        print(f"xmatch_full failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
