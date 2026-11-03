"""Gate G-A1 of ADR 0004: cross-match a batch of SDSS positions with Gaia DR3 through CDS XMatch.

Two passes over the same uploaded positions (``vizier:I/355/gaiadr3`` and
``vizier:I/355/paramp``). Outputs, all under ``--outdir`` (data/interim, ignored by git): the
upload CSV, one zstd Parquet and one manifest per pass, and ``column_report.json``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tempfile
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Final

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from astro_triage.ingest.manifest import (
    XMATCH_ACKNOWLEDGEMENT,
    XMATCH_CITATIONS,
    XMATCH_TERMS,
    Provenance,
    build_manifest,
    file_sha256,
    write_manifest,
)
from astro_triage.ingest.xmatch import (
    XMATCH_MAX_ROWS,
    IngestError,
    XMatchStats,
    read_xmatch_csv,
    xmatch_to_file,
)

DEFAULT_CANDIDATES = (
    REPO_ROOT / "data" / "external" / "sdss" / "specobj_candidates_v1.parquet"
)
DEFAULT_OUTDIR = REPO_ROOT / "data" / "interim" / "xmatch"
PASSES: Final[tuple[tuple[str, str], ...]] = (
    ("gaiadr3", "vizier:I/355/gaiadr3"),
    ("paramp", "vizier:I/355/paramp"),
)
RELEASE: Final = "vizier:I/355"
# XMatch parses an all-digit column as double and corrupts 19-digit ids (observed 2026-10-26:
# 506711614938966016 came back as 5.0671161493896602E17), so the id is sent as text.
SID_PREFIX: Final = "S"

# (canonical raw column, VizieR table pass, expected VizieR name, alternative name) from
# schema section 9.2. An empty expected name means "none identified": the report then only
# checks whether a column with the canonical (Gaia Archive) name exists in either pass.
EXPECTED_RAW: Final[tuple[tuple[str, str, str, str], ...]] = (
    ("source_id", "gaiadr3", "Source", ""),
    ("ra", "gaiadr3", "RA_ICRS", "RAdeg"),
    ("dec", "gaiadr3", "DE_ICRS", "DEdeg"),
    ("l", "gaiadr3", "", ""),
    ("b", "gaiadr3", "", ""),
    ("ra_error", "gaiadr3", "", ""),
    ("dec_error", "gaiadr3", "", ""),
    ("parallax", "gaiadr3", "Plx", ""),
    ("parallax_error", "gaiadr3", "e_Plx", ""),
    ("parallax_over_error", "gaiadr3", "", ""),
    ("pm", "gaiadr3", "", ""),
    ("pmra", "gaiadr3", "pmRA", ""),
    ("pmra_error", "gaiadr3", "e_pmRA", ""),
    ("pmdec", "gaiadr3", "pmDE", ""),
    ("pmdec_error", "gaiadr3", "e_pmDE", ""),
    ("ruwe", "gaiadr3", "RUWE", ""),
    ("astrometric_excess_noise", "gaiadr3", "epsi", ""),
    ("astrometric_excess_noise_sig", "gaiadr3", "", ""),
    ("astrometric_params_solved", "gaiadr3", "", ""),
    ("ipd_gof_harmonic_amplitude", "gaiadr3", "IPDgofha", ""),
    ("ipd_frac_multi_peak", "gaiadr3", "IPDfmp", ""),
    ("ipd_frac_odd_win", "gaiadr3", "IPDfow", ""),
    ("phot_g_mean_mag", "gaiadr3", "Gmag", ""),
    ("phot_g_mean_flux_over_error", "gaiadr3", "RFG", ""),
    ("phot_bp_mean_mag", "gaiadr3", "BPmag", ""),
    ("phot_rp_mean_mag", "gaiadr3", "RPmag", ""),
    ("bp_rp", "gaiadr3", "BP-RP", ""),
    ("phot_bp_rp_excess_factor", "gaiadr3", "E(BP/RP)", ""),
    ("in_qso_candidates", "gaiadr3", "QSO", ""),
    ("in_galaxy_candidates", "gaiadr3", "Gal", ""),
    ("classprob_dsc_combmod_quasar", "gaiadr3", "PQSO", ""),
    ("classprob_dsc_combmod_galaxy", "gaiadr3", "PGal", ""),
    ("classprob_dsc_combmod_star", "gaiadr3", "PSS", ""),
    ("classprob_dsc_allosmod_quasar", "paramp", "PQSO-A", ""),
    ("classprob_dsc_allosmod_galaxy", "paramp", "PGal-A", ""),
    ("classprob_dsc_allosmod_star", "paramp", "Pstar-A", ""),
    ("dr13_objid", "gaiadr3", "", ""),
    ("dr13_angular_distance_arcsec", "gaiadr3", "", ""),
    ("dr13_number_of_neighbours", "gaiadr3", "", ""),
    ("dr13_number_of_mates", "gaiadr3", "", ""),
)


def _display(path: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return resolved.name


def sample_positions(candidates: Path, n: int, seed: int) -> pd.DataFrame:
    table = pq.read_table(
        candidates, columns=["specobjid", "plug_ra", "plug_dec", "class"]
    )
    if n > table.num_rows:
        raise ValueError(f"--n {n} exceeds the {table.num_rows} candidates")
    rng = np.random.default_rng(seed)
    idx = rng.choice(table.num_rows, size=n, replace=False)
    sub = table.take(pa.array(idx)).to_pandas()
    return pd.DataFrame(
        {
            "sid": SID_PREFIX + sub["specobjid"].astype(str),
            "ra": sub["plug_ra"].astype("float64"),
            "dec": sub["plug_dec"].astype("float64"),
            "cls": sub["class"].astype(str),
        }
    )


def write_upload_csv(sample: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(dir=path.parent, prefix=".upload.", suffix=".tmp")
    os.close(fd)
    try:
        sample.to_csv(tmp_name, index=False, float_format="%.9f")
        os.replace(tmp_name, path)
    except BaseException:
        Path(tmp_name).unlink(missing_ok=True)
        raise


def write_parquet(table: pa.Table, path: Path) -> None:
    fd, tmp_name = tempfile.mkstemp(dir=path.parent, prefix=".parquet.", suffix=".tmp")
    os.close(fd)
    try:
        pq.write_table(table, tmp_name, compression="zstd")
        os.replace(tmp_name, path)
    except BaseException:
        Path(tmp_name).unlink(missing_ok=True)
        raise


def column_comparison(columns: dict[str, list[str]]) -> list[dict[str, str]]:
    report: list[dict[str, str]] = []
    for canonical, table_pass, expected, alternative in EXPECTED_RAW:
        if table_pass not in columns:
            status = "unknown"
        elif expected:
            have = columns[table_pass]
            if expected in have:
                status = "present"
            elif alternative and alternative in have:
                status = f"renamed:{alternative}"
            else:
                status = "absent"
        else:
            lowered = {name.lower() for names in columns.values() for name in names}
            status = "present" if canonical in lowered else "absent"
        report.append(
            {
                "canonical": canonical,
                "table": table_pass,
                "expected_vizier": expected or "none identified",
                "status": status,
            }
        )
    return report


def make_manifest(
    parquet: Path,
    *,
    name: str,
    params: str,
    rows: int,
    notes: list[str],
) -> Path:
    digest = file_sha256(parquet)
    provenance = Provenance(
        table_name=name,
        source="CDS XMatch",
        release=RELEASE,
        obtained_utc=datetime.now(UTC).isoformat(timespec="seconds"),
        query_text=params,
        query_sha256=hashlib.sha256(params.encode()).hexdigest(),
        row_count=rows,
        file_sha256=digest,
        file_bytes=parquet.stat().st_size,
        terms=XMATCH_TERMS,
        citations=list(XMATCH_CITATIONS),
        acknowledgement=XMATCH_ACKNOWLEDGEMENT,
        notes=notes,
    )
    return write_manifest(build_manifest(parquet, provenance), parquet.parent)


def _quantiles(values: pd.Series) -> str:
    q = values.quantile([0.5, 0.95])
    return f"median={q.loc[0.5]:.4g} p95={q.loc[0.95]:.4g} max={values.max():.4g}"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=100_000)
    parser.add_argument("--seed", type=int, default=20261025)
    parser.add_argument("--radius", type=float, default=5.0)
    parser.add_argument("--selection", choices=("best", "all"), default="all")
    parser.add_argument("--candidates", type=Path, default=DEFAULT_CANDIDATES)
    parser.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR)
    args = parser.parse_args(argv)
    if args.n <= 0:
        parser.error("--n must be positive")

    started = time.perf_counter()
    outdir: Path = args.outdir
    outdir.mkdir(parents=True, exist_ok=True)
    try:
        sample = sample_positions(args.candidates, args.n, args.seed)
    except (OSError, ValueError, KeyError) as exc:
        print(f"failed to build the sample: {exc}", file=sys.stderr)
        return 1
    upload = outdir / f"upload_n{args.n}_seed{args.seed}.csv"
    try:
        write_upload_csv(sample, upload)
    except OSError as exc:
        print(
            f"failed to write {_display(upload)}: {exc}; check free disk space",
            file=sys.stderr,
        )
        return 1
    upload_sha = file_sha256(upload)
    print(f"rows_sent: {len(sample)}")

    columns: dict[str, list[str]] = {}
    tables: dict[str, pa.Table] = {}
    stats: dict[str, XMatchStats] = {}
    for label, catalog in PASSES:
        stem = f"xmatch_{label}_n{args.n}_r{args.radius:g}_{args.selection}"
        raw = outdir / f"{stem}.raw.csv"
        params = (
            f"request=xmatch; distMaxArcsec={args.radius:g}; selection={args.selection}; "
            f"RESPONSEFORMAT=csv; cat2={catalog}; colRA1=ra; colDec1=dec; "
            f"cat1=<upload csv, columns sid,ra,dec,cls, {len(sample)} rows, "
            f"sha256 {upload_sha}, seed {args.seed}>"
        )
        try:
            stats[label] = xmatch_to_file(
                upload,
                raw,
                catalog=catalog,
                radius_arcsec=args.radius,
                selection=args.selection,
                ra_col="ra",
                dec_col="dec",
            )
            table = read_xmatch_csv(raw)
        except IngestError as exc:
            print(f"pass {label} failed, stopping: {exc}", file=sys.stderr)
            return 1
        except (OSError, pa.ArrowException) as exc:
            print(f"pass {label}: cannot read the answer: {exc}", file=sys.stderr)
            return 1
        try:
            parquet = outdir / f"{stem}.parquet"
            write_parquet(table, parquet)
            raw.unlink(missing_ok=True)
            manifest = make_manifest(
                parquet,
                name=stem,
                params=params,
                rows=table.num_rows,
                notes=[
                    f"bytes_received={stats[label].bytes_received}",
                    f"attempts={stats[label].attempts}",
                    "sample: rng(seed).choice(n_candidates, n, replace=False)",
                ],
            )
        except (OSError, ValueError) as exc:
            print(f"pass {label}: cannot write outputs: {exc}", file=sys.stderr)
            return 1
        columns[label] = list(table.column_names)
        tables[label] = table
        print(
            f"rows_returned_{label}: {table.num_rows} "
            f"(limit {XMATCH_MAX_ROWS}) columns={table.num_columns} "
            f"bytes={stats[label].bytes_received} parquet={_display(parquet)} "
            f"manifest={_display(manifest)}"
        )

    report = {
        "columns": columns,
        "raw_comparison": column_comparison(columns),
    }
    report_path = outdir / "column_report.json"
    report_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(f"column_report: {_display(report_path)}")

    gaia = tables["gaiadr3"].select(["sid", "angDist"]).to_pandas()
    per_object = gaia.groupby("sid")["angDist"].agg(["size", "min"])
    print(f"pairs_per_object ({args.selection}): {_quantiles(per_object['size'])}")
    print(f"matched_objects: {len(per_object)} of {len(sample)}")
    classes = sample.set_index("sid")["cls"]
    matched_cls = classes.reindex(per_object.index)
    sent = sample["cls"].value_counts()
    matched = matched_cls.value_counts()
    for cls in sorted(sent.index):
        hit = int(matched.get(cls, 0))
        print(f"match_rate_{cls}: {hit}/{int(sent[cls])} = {hit / int(sent[cls]):.3%}")
    print(f"angDist_nearest_arcsec: {_quantiles(per_object['min'])}")
    allos = tables["paramp"].column("PQSO-A") if "PQSO-A" in columns["paramp"] else None
    if allos is not None and allos.length() > 0:
        frac = 1 - allos.null_count / allos.length()
        print(f"paramp_rows_PQSO-A_non_null: {frac:.3%} of {allos.length()}")
    else:
        print("paramp_rows_PQSO-A_non_null: column PQSO-A absent or no rows")
    print(f"bytes_received_total: {sum(s.bytes_received for s in stats.values())}")
    print(f"elapsed_seconds: {time.perf_counter() - started:.1f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
