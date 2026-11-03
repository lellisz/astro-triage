"""Build the labels and features tables from the local 100k XMatch outputs.

Pure local work, no network. Rules: ADR 0002 section 3 (one row per Gaia source,
ambiguous matches dropped and counted), ADR 0004 D1/D5/D7 (column map, NULL rules, DSC
coverage layers) and ``docs/specs/schema.md`` sections 3-5.
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
import re
import subprocess
import tempfile
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Final

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
from astropy import units as u
from astropy.coordinates import SkyCoord

from astro_triage.ingest.manifest import (
    GAIA_ACKNOWLEDGEMENT,
    GAIA_TERMS,
    SDSS_ACKNOWLEDGEMENT,
    SDSS_CITATIONS,
    SDSS_TERMS,
    XMATCH_ACKNOWLEDGEMENT,
    XMATCH_CITATIONS,
    XMATCH_TERMS,
    Provenance,
    build_manifest,
    file_sha256,
    write_manifest,
)
from astro_triage.split import assign_folds, pixel_l5

__all__ = [
    "R_MATCH_ARCSEC",
    "BuildError",
    "BuildResult",
    "build_dataset",
    "discover_xmatch_files",
    "format_report",
]

# ADR 0004 gate G-A2 (E1): smallest radius at which the true-match count reaches its plateau.
R_MATCH_ARCSEC: Final = 0.75

REPO_ROOT: Final = Path(__file__).resolve().parents[3]
DEFAULT_GAIA: Final = (
    REPO_ROOT / "data/interim/xmatch/xmatch_gaiadr3_n100000_r5_all.parquet"
)
DEFAULT_PARAMP: Final = (
    REPO_ROOT / "data/interim/xmatch/xmatch_paramp_n100000_r5_all.parquet"
)
DEFAULT_CANDIDATES: Final = (
    REPO_ROOT / "data/external/sdss/specobj_candidates_v1.parquet"
)
DEFAULT_COLUMN_MAP: Final = REPO_ROOT / "manifests/column_map_vizier_v1.csv"
DEFAULT_DATASET_NAME: Final = "dataset_100k_v0"
DEFAULT_OUTDIR: Final = REPO_ROOT / "data/interim" / DEFAULT_DATASET_NAME
_DATASET_NAME_RE: Final = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]*")
_CHUNK_FILE_RE: Final = re.compile(r"xmatch_(gaiadr3|paramp)_chunk(\d+)\.parquet")

CLASSES: Final[tuple[str, ...]] = ("STAR", "GALAXY", "QSO")
FOLDS: Final[tuple[str, ...]] = ("train", "val", "test")
SCHEMA_VERSION: Final = 0
REPORT_FLOOR_OBJECTS: Final = 100
TOP_NULL_COLUMNS: Final = 10

# schema.md section 4: every column except source_id is float32.
FEATURE_COLUMNS: Final[tuple[str, ...]] = (
    "parallax",
    "parallax_error",
    "parallax_over_error",
    "pm",
    "pmra",
    "pmra_error",
    "pmdec",
    "pmdec_error",
    "ruwe",
    "astrometric_excess_noise",
    "astrometric_excess_noise_sig",
    "astrometric_params_solved",
    "ipd_gof_harmonic_amplitude",
    "ipd_frac_multi_peak",
    "ipd_frac_odd_win",
    "phot_g_mean_mag",
    "phot_g_mean_flux_over_error",
    "phot_bp_mean_mag",
    "phot_rp_mean_mag",
    "bp_rp",
    "phot_bp_rp_excess_factor",
)
# schema.md section 4 also marks phot_g_mean_mag and phot_g_mean_flux_over_error non-null
# (G < 20 selection); the XMatch path has no such selection and 25 pairs have no G photometry.
NON_NULL_FEATURES: Final[frozenset[str]] = frozenset(
    {
        "astrometric_excess_noise",
        "astrometric_excess_noise_sig",
        "astrometric_params_solved",
    }
)
FORBIDDEN_FEATURE_COLUMNS: Final[frozenset[str]] = frozenset(
    {
        "classprob_dsc_combmod_quasar",
        "classprob_dsc_combmod_galaxy",
        "classprob_dsc_combmod_star",
        "classprob_dsc_allosmod_quasar",
        "classprob_dsc_allosmod_galaxy",
        "classprob_dsc_allosmod_star",
        "in_qso_candidates",
        "in_galaxy_candidates",
        "ap_row_present",
        "ra",
        "dec",
        "l",
        "b",
        "ra_error",
        "dec_error",
        "sdss_z",
        "sdss_z_err",
    }
)
ALLOSMOD: Final[tuple[str, ...]] = (
    "classprob_dsc_allosmod_quasar",
    "classprob_dsc_allosmod_galaxy",
    "classprob_dsc_allosmod_star",
)
COMBMOD: Final[tuple[str, ...]] = (
    "classprob_dsc_combmod_quasar",
    "classprob_dsc_combmod_galaxy",
    "classprob_dsc_combmod_star",
)
# Columns that must exist in the candidate file and are copied into labels.
CANDIDATE_COLUMNS: Final[tuple[str, ...]] = (
    "specobjid",
    "plate",
    "mjd",
    "fiberid",
    "class",
    "subclass",
    "z",
    "z_err",
    "zwarning",
    "plug_ra",
    "plug_dec",
    "specprimary",
)

_DICT: Final = pa.dictionary(pa.int8(), pa.string())


def _field(name: str, kind: pa.DataType, nullable: bool = False) -> pa.Field:
    return pa.field(name, kind, nullable=nullable)


LABELS_SCHEMA: Final = pa.schema(
    [
        _field("source_id", pa.int64()),
        _field("sdss_class", _DICT),
        _field("sdss_z", pa.float32()),
        _field("sdss_z_err", pa.float32(), nullable=True),
        _field("n_specobj_candidates", pa.int16()),
        _field("sdss_plate", pa.int32()),
        _field("sdss_mjd", pa.int32()),
        _field("sdss_fiberid", pa.int32()),
        _field("sdss_specobjid", pa.string()),
        _field("sdss_ra", pa.float64()),
        _field("sdss_dec", pa.float64()),
        _field("angular_distance_arcsec", pa.float32()),
        _field("sdss_subclass", pa.string(), nullable=True),
        _field("sdss_zwarning", pa.int32()),
        _field("healpix_l5", pa.int16()),
        _field("fold", _DICT),
        _field("ra", pa.float64()),
        _field("dec", pa.float64()),
        _field("l", pa.float64()),
        _field("b", pa.float64()),
        _field("ap_row_present", pa.bool_()),
        *[_field(name, pa.float32(), nullable=True) for name in (*COMBMOD, *ALLOSMOD)],
    ]
)
FEATURES_SCHEMA: Final = pa.schema(
    [
        _field("source_id", pa.int64()),
        *[
            _field(name, pa.float32(), nullable=name not in NON_NULL_FEATURES)
            for name in FEATURE_COLUMNS
        ],
    ]
)


class BuildError(Exception):
    """An input violates a documented precondition; the message names the fix."""


class BuildResult:
    def __init__(self, report: dict[str, object], paths: dict[str, Path]) -> None:
        self.report = report
        self.paths = paths


def load_column_map(path: Path) -> dict[str, dict[str, str]]:
    """Return ``{pass: {canonical: vizier}}`` for the rows with status ``mapped``."""
    out: dict[str, dict[str, str]] = {"gaiadr3": {}, "paramp": {}}
    with Path(path).open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            if row["status"] != "mapped" or not row["vizier_column"]:
                continue
            for name, mapping in out.items():
                if row["note"].startswith(f"pass={name}"):
                    mapping[row["canonical_name"]] = row["vizier_column"]
    for name, mapping in out.items():
        if not mapping:
            raise BuildError(f"column map {path} has no mapped column for pass {name}")
    return out


def _as_paths(value: Path | Sequence[Path]) -> list[Path]:
    return [value] if isinstance(value, Path) else list(value)


def discover_xmatch_files(xmatch_dir: Path) -> tuple[list[Path], list[Path]]:
    """Return the per-chunk gaiadr3 and paramp Parquets of ``xmatch_full.py``, by chunk index.

    Both passes must exist for exactly the same chunk indices: a lone file would silently
    drop the pairs of one chunk from the global ambiguity rules.
    """
    found: dict[str, dict[int, Path]] = {"gaiadr3": {}, "paramp": {}}
    for path in Path(xmatch_dir).iterdir():
        match = _CHUNK_FILE_RE.fullmatch(path.name)
        if match:
            found[match.group(1)][int(match.group(2))] = path
    gaia, paramp = found["gaiadr3"], found["paramp"]
    if not gaia:
        raise BuildError(f"no xmatch_gaiadr3_chunk*.parquet in {xmatch_dir.name}")
    if gaia.keys() != paramp.keys():
        raise BuildError(
            "gaiadr3 and paramp chunk indices differ: "
            f"only gaiadr3 {sorted(gaia.keys() - paramp.keys())}, "
            f"only paramp {sorted(paramp.keys() - gaia.keys())}"
        )
    order = sorted(gaia)
    return [gaia[i] for i in order], [paramp[i] for i in order]


def _read_renamed(
    paths: Sequence[Path], mapping: dict[str, str], extra: list[str]
) -> pd.DataFrame:
    wanted = [*extra, *mapping.values()]
    parts: list[pa.Table] = []
    for path in paths:
        try:
            parts.append(pq.read_table(path, columns=wanted))
        except (KeyError, pa.ArrowInvalid) as exc:
            raise BuildError(
                f"{path.name} lacks a column named in the column map: {exc}"
            ) from exc
    table = (
        parts[0]
        if len(parts) == 1
        else pa.concat_tables(parts, promote_options="permissive")
    )
    frame = table.to_pandas()
    return frame.rename(columns={v: k for k, v in mapping.items()})


def read_pairs(
    gaia_path: Path | Sequence[Path],
    paramp_path: Path | Sequence[Path],
    candidates_path: Path,
    column_map: dict[str, dict[str, str]],
) -> pd.DataFrame:
    """One row per (SDSS object, Gaia source) pair, with the paramp LEFT JOIN and labels.

    Several files per pass are concatenated first, so the join and every later rule see
    all pairs at once.
    """
    pairs = _read_renamed(
        _as_paths(gaia_path), column_map["gaiadr3"], ["sid", "angDist", "cls"]
    )
    n_pairs = len(pairs)
    paramp_map = {
        k: v for k, v in column_map["paramp"].items() if k.startswith("classprob_")
    }
    paramp = _read_renamed(_as_paths(paramp_path), paramp_map, ["Source"]).rename(
        columns={"Source": "source_id"}
    )
    paramp_ids = paramp["source_id"].drop_duplicates()
    paramp = paramp.drop_duplicates("source_id")
    pairs = pairs.merge(paramp, on="source_id", how="left", validate="many_to_one")
    if len(pairs) != n_pairs:
        raise BuildError("the paramp LEFT JOIN changed the number of gaiadr3 rows")
    pairs["ap_row_present"] = pairs["source_id"].isin(paramp_ids)
    pairs["specobjid"] = pairs["sid"].str.removeprefix("S")
    if not pairs["specobjid"].str.fullmatch(r"\d+").all():
        raise BuildError("sid values are not 'S' followed by digits")

    ids = pairs["specobjid"].unique().tolist()
    cand = pq.read_table(
        candidates_path,
        columns=list(CANDIDATE_COLUMNS),
        filters=[("specobjid", "in", ids)],
    ).to_pandas()
    if cand["specobjid"].duplicated().any():
        raise BuildError("the candidates file has duplicated specobjid")
    if len(cand) != len(ids):
        raise BuildError(
            f"{len(ids) - len(cand)} SDSS objects of the XMatch output are not in "
            f"{candidates_path.name}"
        )
    cand["class"] = cand["class"].str.strip().str.upper()
    cand["subclass"] = cand["subclass"].str.strip().replace("", None)
    if not (cand["specprimary"] == 1).all() or not (cand["zwarning"] == 0).all():
        raise BuildError("candidates must satisfy SPECPRIMARY == 1 and ZWARNING == 0")
    if not cand["class"].isin(CLASSES).all():
        raise BuildError("candidates contain a class outside STAR/GALAXY/QSO")
    cand = (
        cand.drop(columns=["specprimary"])
        .add_prefix("sdss_")
        .rename(columns={"sdss_plug_ra": "sdss_ra", "sdss_plug_dec": "sdss_dec"})
    )
    pairs = pairs.merge(
        cand, left_on="specobjid", right_on="sdss_specobjid", how="left", validate="m:1"
    )
    if (pairs["cls"].str.upper() != pairs["sdss_class"]).any():
        raise BuildError("class echoed by XMatch differs from the candidates file")
    return pairs.drop(columns=["specobjid", "cls"])


def _by_class(frame: pd.DataFrame) -> dict[str, int]:
    counts = frame["sdss_class"].value_counts()
    return {c: int(counts.get(c, 0)) for c in CLASSES}


def resolve_ambiguity(
    pairs: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Apply ADR 0002 section 3, in the order of the ADR; return kept pairs and counts."""
    n_cand = pairs.groupby("source_id")["sid"].nunique().rename("n_specobj_candidates")
    pairs = pairs.merge(n_cand, left_on="source_id", right_index=True)

    gaia_per_obj = pairs.groupby("sid")["source_id"].nunique()
    multi = pairs["sid"].isin(gaia_per_obj[gaia_per_obj > 1].index)
    step1 = pairs[multi]
    rest = pairs[~multi]
    lost_sources = set(step1["source_id"]) - set(rest["source_id"])

    classes = rest.groupby("source_id")["sdss_class"].nunique()
    conflict_ids = classes[classes > 1].index
    conflict = rest[rest["source_id"].isin(conflict_ids)]
    agree = rest[~rest["source_id"].isin(conflict_ids)]
    ordered = agree.sort_values(["source_id", "angDist", "sid"], kind="stable")
    kept = ordered.drop_duplicates("source_id", keep="first")
    collapsed = ordered[ordered.duplicated("source_id", keep="first")]

    counts: dict[str, object] = {
        "specobj_multi_gaia": {
            "specobj_by_class": _by_class(step1.drop_duplicates("sid")),
            "pairs": len(step1),
            "gaia_sources_left_without_candidate": len(lost_sources),
        },
        "source_class_conflict": {
            "gaia_sources": len(conflict_ids),
            "specobj_by_class": _by_class(conflict.drop_duplicates("sid")),
        },
        "same_class_duplicates_collapsed": {
            "gaia_sources": int(collapsed["source_id"].nunique()),
            "specobj_dropped_by_class": _by_class(collapsed.drop_duplicates("sid")),
        },
    }
    return kept.reset_index(drop=True), counts


def _galactic(ra: pd.Series, dec: pd.Series) -> tuple[np.ndarray, np.ndarray]:
    coord = SkyCoord(
        ra=ra.to_numpy() * u.deg, dec=dec.to_numpy() * u.deg, frame="icrs"
    ).galactic
    return (
        np.asarray(coord.l.deg, dtype=np.float64),
        np.asarray(coord.b.deg, dtype=np.float64),
    )


def derive_columns(frame: pd.DataFrame) -> pd.DataFrame:
    """Add l, b, parallax_over_error, pm (ADR 0004 D1) and the fold columns."""
    out = frame.copy()
    out["l"], out["b"] = _galactic(out["ra"], out["dec"])
    err = out["parallax_error"].where(out["parallax_error"] != 0)
    out["parallax_over_error"] = out["parallax"] / err
    out["pm"] = np.hypot(out["pmra"], out["pmdec"])
    if np.isinf(out[["parallax_over_error", "pm"]]).to_numpy().any():
        raise BuildError("a derived column is infinite; check the astrometric inputs")
    out["healpix_l5"] = pixel_l5(out["source_id"].to_numpy(dtype=np.int64))
    out["fold"] = assign_folds(out["source_id"].to_numpy(dtype=np.int64))
    return out


def _to_table(frame: pd.DataFrame, schema: pa.Schema) -> pa.Table:
    arrays: list[pa.Array] = []
    for field in schema:
        column = frame[field.name]
        if pa.types.is_float32(field.type):
            column = column.astype(np.float32)
        array = pa.array(column, type=field.type, from_pandas=True)
        if pa.types.is_floating(field.type) and (
            pc.is_inf(array).true_count or pc.is_nan(array).true_count
        ):
            raise BuildError(f"{field.name}: NaN or inf survived the NULL conversion")
        if not field.nullable and array.null_count:
            raise BuildError(
                f"{field.name} is non-nullable but has {array.null_count} NULL values"
            )
        arrays.append(array)
    return pa.Table.from_arrays(arrays, schema=schema)


def _git_sha() -> str:
    try:
        done = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            check=False,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return "unavailable"
    return done.stdout.strip() if done.returncode == 0 else "unavailable"


def _write_parquet(table: pa.Table, path: Path, metadata: dict[str, str]) -> None:
    table = table.sort_by("source_id").replace_schema_metadata(metadata)
    fd, tmp_name = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.stem}.", suffix=".tmp"
    )
    os.close(fd)
    try:
        pq.write_table(table, tmp_name, compression="zstd")
        os.replace(tmp_name, path)
    except BaseException:
        Path(tmp_name).unlink(missing_ok=True)
        raise


def _write_json(payload: dict[str, object], path: Path) -> None:
    fd, tmp_name = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.stem}.", suffix=".tmp"
    )
    os.close(fd)
    try:
        Path(tmp_name).write_text(
            json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        os.replace(tmp_name, path)
    except BaseException:
        Path(tmp_name).unlink(missing_ok=True)
        raise


def _fractions(series: pd.Series, by: pd.Series) -> dict[str, float]:
    grouped = series.groupby(by).mean()
    return {c: float(grouped.get(c, float("nan"))) for c in CLASSES}


def build_report(
    labels: pd.DataFrame,
    n_input: int,
    n_after_r: int,
    discards: dict[str, object],
    r_match: float,
    n_sid_input: int,
    n_sid_after_r: int,
) -> dict[str, object]:
    cls = labels["sdss_class"]
    by_fold = {f: int((labels["fold"] == f).sum()) for f in FOLDS}
    by_class_fold = {
        c: {f: int(((cls == c) & (labels["fold"] == f)).sum()) for f in FOLDS}
        for c in CLASSES
    }
    nulls = labels.attrs["feature_null_fraction"]
    top_nulls = sorted(nulls.items(), key=lambda kv: kv[1], reverse=True)
    covered = labels[list(ALLOSMOD)].notna().all(axis=1)
    partial = labels[list(ALLOSMOD)].isna().any(axis=1) & labels[
        list(ALLOSMOD)
    ].notna().any(axis=1)
    dsc: dict[str, dict[str, object]] = {}
    for c in CLASSES:
        sel = cls == c
        n = int(sel.sum())
        no_row = int((sel & ~labels["ap_row_present"]).sum())
        no_allos = int((sel & labels["ap_row_present"] & ~covered).sum())
        dsc[c] = {
            "n": n,
            "layer1_no_paramp_row": no_row,
            "layer2_row_without_allosmod": no_allos,
            "covered_S_dsc": int((sel & covered).sum()),
            "layer1_fraction": no_row / n if n else None,
            "layer2_fraction": no_allos / n if n else None,
        }
    test = labels[labels["fold"] == "test"]
    pix_all = test.groupby("healpix_l5").size()
    pixels_per_class = {
        c: int(
            (
                test[test["sdss_class"] == c].groupby("healpix_l5").size()
                >= REPORT_FLOOR_OBJECTS
            ).sum()
        )
        for c in CLASSES
    }
    fold_per_pixel = labels.groupby("healpix_l5")["fold"].nunique()
    return {
        "r_match_arcsec": r_match,
        "input_pairs": n_input,
        "input_specobj": n_sid_input,
        "pairs_after_r_match": n_after_r,
        "specobj_after_r_match": n_sid_after_r,
        "discards_in_order": {
            "outside_r_match": {
                "pairs": n_input - n_after_r,
                "specobj_without_any_pair": n_sid_input - n_sid_after_r,
            },
            **discards,
        },
        "final_rows": len(labels),
        "final_by_class": _by_class(labels),
        "final_by_fold": by_fold,
        "final_by_class_and_fold": by_class_fold,
        "feature_null_fraction_top10": dict(top_nulls[:TOP_NULL_COLUMNS]),
        "solved3_fraction_by_class": _fractions(
            labels["astrometric_params_solved"] == 3, cls
        ),
        "dsc_coverage_by_class": dsc,
        "dsc_allosmod_partial_null_rows": int(partial.sum()),
        "test_pixels_total": int(pix_all.size),
        "test_pixels_with_at_least_100_objects": int(
            (pix_all >= REPORT_FLOOR_OBJECTS).sum()
        ),
        "test_pixels_with_at_least_100_objects_by_class": pixels_per_class,
        "pixels_in_more_than_one_fold": int((fold_per_pixel > 1).sum()),
    }


def format_report(report: dict[str, object]) -> str:
    return json.dumps(report, indent=2, ensure_ascii=False)


def _manifest(
    path: Path,
    name: str,
    rows: int,
    params: str,
    inputs: list[str],
    release: str = DEFAULT_DATASET_NAME,
) -> Path:
    provenance = Provenance(
        table_name=name,
        source="derived",
        release=release,
        obtained_utc=datetime.now(UTC).isoformat(timespec="seconds"),
        query_text=params,
        query_sha256=hashlib.sha256(params.encode()).hexdigest(),
        row_count=rows,
        file_sha256=file_sha256(path),
        file_bytes=path.stat().st_size,
        terms=f"{XMATCH_TERMS} | {GAIA_TERMS} | {SDSS_TERMS}",
        citations=[*XMATCH_CITATIONS, *SDSS_CITATIONS],
        acknowledgement=(
            f"{XMATCH_ACKNOWLEDGEMENT} {GAIA_ACKNOWLEDGEMENT} {SDSS_ACKNOWLEDGEMENT}"
        ),
        notes=[f"input: {item}" for item in inputs],
    )
    return write_manifest(build_manifest(path, provenance), path.parent)


def build_dataset(
    *,
    r_match: float = R_MATCH_ARCSEC,
    gaia_path: Path | Sequence[Path] = DEFAULT_GAIA,
    paramp_path: Path | Sequence[Path] = DEFAULT_PARAMP,
    xmatch_dir: Path | None = None,
    candidates_path: Path = DEFAULT_CANDIDATES,
    column_map_path: Path = DEFAULT_COLUMN_MAP,
    outdir: Path = DEFAULT_OUTDIR,
    dataset_name: str = DEFAULT_DATASET_NAME,
) -> BuildResult:
    if not r_match > 0:
        raise BuildError("r_match must be positive")
    if not _DATASET_NAME_RE.fullmatch(dataset_name):
        raise BuildError(f"invalid dataset_name: {dataset_name!r}")
    if xmatch_dir is not None:
        gaia_path, paramp_path = discover_xmatch_files(xmatch_dir)
    column_map = load_column_map(column_map_path)
    pairs = read_pairs(gaia_path, paramp_path, candidates_path, column_map)
    n_input, n_sid_input = len(pairs), int(pairs["sid"].nunique())
    pairs = pairs[pairs["angDist"] <= r_match]
    n_after_r, n_sid_after_r = len(pairs), int(pairs["sid"].nunique())
    if pairs.empty:
        raise BuildError(f"no pair within r_match = {r_match} arcsec")

    kept, discards = resolve_ambiguity(pairs)
    labels = derive_columns(kept).rename(columns={"angDist": "angular_distance_arcsec"})
    labels["n_specobj_candidates"] = labels["n_specobj_candidates"].astype(np.int16)
    labels.attrs["feature_null_fraction"] = {
        name: float(labels[name].isna().mean()) for name in FEATURE_COLUMNS
    }

    forbidden = FORBIDDEN_FEATURE_COLUMNS & set(FEATURE_COLUMNS)
    if forbidden:
        raise BuildError(
            f"forbidden columns configured as features: {sorted(forbidden)}"
        )
    labels_table = _to_table(labels, LABELS_SCHEMA)
    features_table = _to_table(labels, FEATURES_SCHEMA)

    outdir.mkdir(parents=True, exist_ok=True)
    for leftover in outdir.glob("*.tmp"):
        leftover.unlink()
    inputs = [
        p.name
        for p in (
            *_as_paths(gaia_path),
            *_as_paths(paramp_path),
            candidates_path,
            column_map_path,
        )
    ]
    meta = {
        "schema_version": str(SCHEMA_VERSION),
        "created_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "git_sha": _git_sha(),
        "inputs": json.dumps(inputs),
        "r_match_arcsec": str(r_match),
        "match_engine": "CDS XMatch",
        "xmatch_catalogue": "vizier:I/355/gaiadr3",
        "sdss_epoch_rule": "none: unpropagated XMatch angDist (G-A2 offset run is unpropagated)",
    }
    paths = {
        "labels": outdir / "labels.parquet",
        "features": outdir / "features.parquet",
        "report": outdir / "dataset_report.json",
    }
    _write_parquet(labels_table, paths["labels"], {**meta, "schema_name": "labels"})
    _write_parquet(
        features_table,
        paths["features"],
        {**meta, "schema_name": "features", "labels_file": paths["labels"].name},
    )
    report = build_report(
        labels, n_input, n_after_r, discards, r_match, n_sid_input, n_sid_after_r
    )
    _write_json(report, paths["report"])

    params = (
        f"local build; r_match_arcsec={r_match}; ambiguity per ADR 0002 section 3; "
        "column map manifests/column_map_vizier_v1.csv; folds via astro_triage.split"
    )
    for key in ("labels", "features"):
        _manifest(
            paths[key],
            f"{dataset_name}_{key}",
            len(labels),
            params,
            inputs,
            release=dataset_name,
        )
    return BuildResult(report, paths)
