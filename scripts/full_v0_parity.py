"""Row-data-free fingerprint of a full_v0 release and its comparison with a reference.

Subcommands: ``reference`` (local only, writes the fingerprint) and ``check`` (compares a
rebuilt release with the reference and writes a report). Exit codes: 0 match, 3 mismatch,
1 error. Output never holds a per-object value, only counts, digests and column aggregates.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import tempfile
from pathlib import Path
from typing import Final

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

SCHEMA_ID: Final = "full_v0_fingerprint/1"
LABELS_NAME: Final = "labels.parquet"
FEATURES_NAME: Final = "features.parquet"
ID_COLUMN: Final = "source_id"
CLASS_COLUMN: Final = "sdss_class"
FOLD_COLUMN: Final = "fold"
FILE_FIELDS: Final = ("labels_file_sha256", "features_file_sha256")
CRITERIA: Final = (
    "rows",
    "by_class",
    "by_fold",
    "labels_schema",
    "features_schema",
    "source_id_unique",
    "source_id_sha256",
)
# Per-column digests: a mismatch is reported as ``<labels|features>.column.<name>``.
COLUMN_FIELDS: Final = ("labels_column_sha256", "features_column_sha256")
# Float columns that build.py DERIVES from trigonometry or hypot may differ in the last
# ulp between platforms (libm / SIMD). pm (np.hypot of pmra/pmdec) is float32: the float32
# cast is the main protection against ulp noise; the rounding to 1e-3 mas/yr is only a
# secondary safeguard and does nothing for pm above ~8192 mas/yr (float32 spacing there
# exceeds 1e-3). Limitation: a value straddling a rounding boundary can still flip the
# digest (false MISMATCH, never a false match beyond the tolerance).
# parallax_over_error and bp_rp are plain IEEE division/subtraction, correctly rounded
# everywhere, so they stay exact; ra, dec and copied columns too.
DERIVED_DECIMALS: Final = {"pm": 3}
# l and b (astropy SkyCoord galactic: numpy sin/cos/arctan2) are float64 and NOT hashed:
# rounding then hashing flips on any single row near a rounding boundary. They carry
# row-free aggregates instead, compared with STATS_RTOL / STATS_ATOL. Why 1e-12: per-row
# libm ulp noise is ~8e-14 deg; worst case, all the same sign over ~2e6 rows, is ~5e-16
# relative on the sum, so 1e-12 keeps >3 orders of magnitude margin; the abs tol 1e-9
# covers values near zero.
STATS_COLUMNS: Final = frozenset({"l", "b"})
STATS_FIELDS: Final = ("labels_column_stats", "features_column_stats")
STATS_RTOL: Final = 1e-12
STATS_ATOL: Final = 1e-9
_STATS_EXACT: Final = ("count", "null_count", "nan_count")
_STATS_APPROX: Final = ("min", "max", "sum", "sumsq")
_CANON_NAN: Final = np.array([np.nan], dtype="<f8")
_STRING_BATCH: Final = 100_000


class ParityError(Exception):
    """Missing file, malformed reference or unsupported column type."""


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _read(path: Path) -> pa.Table:
    if not path.is_file():
        raise ParityError(f"missing file {path.name}")
    try:
        return pq.read_table(path)
    except (OSError, pa.ArrowException) as exc:
        raise ParityError(f"cannot read {path.name}: {exc}") from exc


def _ids(table: pa.Table, name: str) -> np.ndarray:
    if ID_COLUMN not in table.column_names:
        raise ParityError(f"{name} has no {ID_COLUMN} column")
    column = table[ID_COLUMN]
    if not pa.types.is_integer(column.type) or column.null_count:
        raise ParityError(f"{name} {ID_COLUMN} must be a non-null integer column")
    return np.asarray(column.to_numpy(), dtype="<i8")


def _feed_column(
    digest: hashlib._Hash,
    column: pa.ChunkedArray,
    label: str,
    decimals: int | None = None,
) -> None:
    kind = column.type
    digest.update(
        pc.is_null(column).to_numpy(zero_copy_only=False).astype(np.uint8).tobytes()
    )
    if pa.types.is_integer(kind) or pa.types.is_boolean(kind):
        values = pc.cast(column, pa.int64()).fill_null(0)
        digest.update(np.asarray(values.to_numpy(), dtype="<i8").tobytes())
    elif pa.types.is_floating(kind):
        floats = np.asarray(
            pc.cast(column, pa.float64()).fill_null(float("nan")).to_numpy(),
            dtype="<f8",
        )
        if decimals is not None:
            floats = np.round(floats, decimals)
        floats = floats + 0.0  # + 0.0 folds -0.0 into 0.0 for every float column
        digest.update(np.where(np.isnan(floats), _CANON_NAN, floats).tobytes())
    elif (
        pa.types.is_string(kind)
        or pa.types.is_large_string(kind)
        or (
            pa.types.is_dictionary(kind)
            and (
                pa.types.is_string(kind.value_type)
                or pa.types.is_large_string(kind.value_type)
            )
        )
    ):
        text = pc.cast(column, pa.string()).fill_null("")
        for start in range(0, len(text), _STRING_BATCH):
            chunk = text.slice(start, _STRING_BATCH).to_pylist()
            digest.update(("\0".join(chunk) + "\0").encode("utf-8"))
    else:
        raise ParityError(f"unsupported column type {kind} in column {label}")


def _column_sha256(table: pa.Table, order: np.ndarray) -> dict[str, str]:
    indices = pa.array(order)
    digests: dict[str, str] = {}
    for name in sorted(table.column_names):
        if name in STATS_COLUMNS:
            continue
        digest = hashlib.sha256()
        column = table[name].take(indices)
        _feed_column(digest, column, name, DERIVED_DECIMALS.get(name))
        digests[name] = digest.hexdigest()
    return digests


def _column_stats(table: pa.Table, order: np.ndarray) -> dict[str, dict[str, object]]:
    """Row-free aggregates of the STATS_COLUMNS, over the source_id-sorted column."""
    indices = pa.array(order)
    stats: dict[str, dict[str, object]] = {}
    for name in sorted(STATS_COLUMNS & set(table.column_names)):
        column = table[name].take(indices)
        if not pa.types.is_floating(column.type):
            raise ParityError(f"column {name} must be floating point")
        values = np.asarray(
            pc.cast(column, pa.float64()).fill_null(float("nan")).to_numpy(),
            dtype="<f8",
        )
        finite = values[np.isfinite(values)]
        total: list[float] = []
        total_sq: list[float] = []
        for start in range(0, finite.size, _STRING_BATCH):  # fixed order, exact fsum
            chunk = finite[start : start + _STRING_BATCH]
            total.append(math.fsum(chunk.tolist()))
            total_sq.append(math.fsum((chunk * chunk).tolist()))
        stats[name] = {
            "count": len(column),
            "null_count": int(column.null_count),
            "nan_count": int(np.isnan(values).sum()) - int(column.null_count),
            "min": float(finite.min()) if finite.size else None,
            "max": float(finite.max()) if finite.size else None,
            "sum": math.fsum(total),
            "sumsq": math.fsum(total_sq),
        }
    return stats


def _stats_equal(ref: object, got: object) -> bool:
    if not isinstance(ref, dict) or not isinstance(got, dict):
        return False
    for key in _STATS_EXACT:
        if ref.get(key) != got.get(key):
            return False
    for key in _STATS_APPROX:
        r, g = ref.get(key), got.get(key)
        if r is None or g is None:
            if r is not g:
                return False
        elif not (
            isinstance(r, (int, float))
            and isinstance(g, (int, float))
            and math.isclose(r, g, rel_tol=STATS_RTOL, abs_tol=STATS_ATOL)
        ):
            return False
    return True


def _counts(table: pa.Table, name: str) -> dict[str, int]:
    if name not in table.column_names:
        raise ParityError(f"labels has no {name} column")
    counted = pc.value_counts(pc.cast(table[name], pa.string()))
    return {str(item["values"]): int(item["counts"]) for item in counted.to_pylist()}


def _schema(table: pa.Table) -> list[str]:
    return [f"{field.name}:{field.type}" for field in table.schema]


def fingerprint(outdir: Path) -> dict[str, object]:
    labels_path = outdir / LABELS_NAME
    features_path = outdir / FEATURES_NAME
    labels = _read(labels_path)
    features = _read(features_path)
    label_ids = _ids(labels, LABELS_NAME)
    feature_ids = _ids(features, FEATURES_NAME)
    label_order = np.argsort(label_ids, kind="stable")
    feature_order = np.argsort(feature_ids, kind="stable")
    sorted_ids = label_ids[label_order]
    unique = bool(sorted_ids.size < 2 or np.all(sorted_ids[1:] != sorted_ids[:-1]))
    return {
        "schema": SCHEMA_ID,
        "rows": labels.num_rows,
        "by_class": _counts(labels, CLASS_COLUMN),
        "by_fold": _counts(labels, FOLD_COLUMN),
        "labels_schema": _schema(labels),
        "features_schema": _schema(features),
        "source_id_unique": unique,
        "source_id_sha256": hashlib.sha256(
            sorted_ids.astype("<i8").tobytes()
        ).hexdigest(),
        "labels_column_sha256": _column_sha256(labels, label_order),
        "features_column_sha256": _column_sha256(features, feature_order),
        "labels_column_stats": _column_stats(labels, label_order),
        "features_column_stats": _column_stats(features, feature_order),
        "labels_file_sha256": _file_sha256(labels_path),
        "features_file_sha256": _file_sha256(features_path),
    }


def compare(ref: dict[str, object], got: dict[str, object]) -> list[str]:
    differing = [name for name in CRITERIA if ref.get(name) != got.get(name)]
    for field in COLUMN_FIELDS:
        ref_cols, got_cols = ref.get(field), got.get(field)
        if not isinstance(ref_cols, dict) or not isinstance(got_cols, dict):
            differing.append(field)
            continue
        prefix = field.removesuffix("_sha256").replace("_", ".", 1)
        differing.extend(
            f"{prefix}.{name}"
            for name in sorted(set(ref_cols) | set(got_cols))
            if ref_cols.get(name) != got_cols.get(name)
        )
    for field in STATS_FIELDS:
        ref_cols, got_cols = ref.get(field), got.get(field)
        if not isinstance(ref_cols, dict) or not isinstance(got_cols, dict):
            differing.append(field)
            continue
        prefix = field.removesuffix("_stats").replace("_", ".", 1)
        differing.extend(
            f"{prefix}.{name}"
            for name in sorted(set(ref_cols) | set(got_cols))
            if not _stats_equal(ref_cols.get(name), got_cols.get(name))
        )
    if "source_id_unique" not in differing and not (
        ref.get("source_id_unique") is True and got.get("source_id_unique") is True
    ):
        differing.append("source_id_unique")
    return differing


def _atomic_write(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=".parity.", suffix=".tmp")
    os.close(fd)
    try:
        Path(tmp).write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        os.replace(tmp, path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise


def _load_reference(path: Path) -> dict[str, object]:
    try:
        loaded = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ParityError(f"cannot read reference {path.name}: {exc}") from exc
    if not isinstance(loaded, dict) or loaded.get("schema") != SCHEMA_ID:
        raise ParityError(f"reference {path.name} is not a {SCHEMA_ID} document")
    missing = [
        name
        for name in (*CRITERIA, *COLUMN_FIELDS, *STATS_FIELDS, *FILE_FIELDS)
        if name not in loaded
    ]
    if missing:
        raise ParityError(f"reference {path.name} lacks fields {missing}")
    return loaded


def _cmd_reference(args: argparse.Namespace) -> int:
    _atomic_write(args.out, fingerprint(args.outdir))
    return 0


def _cmd_check(args: argparse.Namespace) -> int:
    ref = _load_reference(args.reference)
    got = fingerprint(args.outdir)
    differing = compare(ref, got)
    report: dict[str, object] = {
        "match": not differing,
        "differing_fields": differing,
        "file_sha256_equal": all(ref.get(f) == got.get(f) for f in FILE_FIELDS),
        "reference": ref,
        "got": got,
    }
    _atomic_write(args.report, report)
    for name in differing:
        print(f"MISMATCH: {name}")
    return 3 if differing else 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    ref = sub.add_parser("reference", help="write the fingerprint of a local release")
    ref.add_argument("--outdir", type=Path, required=True)
    ref.add_argument("--out", type=Path, required=True)
    ref.set_defaults(func=_cmd_reference)
    chk = sub.add_parser("check", help="compare a release with the reference")
    chk.add_argument("--reference", type=Path, required=True)
    chk.add_argument("--outdir", type=Path, required=True)
    chk.add_argument("--report", type=Path, required=True)
    chk.set_defaults(func=_cmd_check)
    args = parser.parse_args(argv)
    try:
        return int(args.func(args))
    except (ParityError, OSError, pa.ArrowException) as exc:
        print(f"parity failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
