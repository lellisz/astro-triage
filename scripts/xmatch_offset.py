"""Gate G-A2 of ADR 0004 (experiment E1): choose ``r_match`` from chance coincidences.

Recreates the 100,000-position sample of the real G-A1 batch, shifts every declination by
+180 arcsec, sends it in ONE XMatch request (gaiadr3, same parameters as the real run) and
compares, per SDSS class and radius, the distinct objects matched in the real and in the
shifted run. Every pair of the shifted run is a chance coincidence. The decision criterion
is the fixed one of ADR 0004 section 7, gate G-A2.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import tempfile
import time
from pathlib import Path
from typing import Final

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from xmatch_batch import (
    DEFAULT_CANDIDATES,
    DEFAULT_OUTDIR,
    _display,
    make_manifest,
    sample_positions,
    write_parquet,
    write_upload_csv,
)

from astro_triage.ingest.manifest import file_sha256
from astro_triage.ingest.xmatch import IngestError, read_xmatch_csv, xmatch_to_file

CATALOG: Final = "vizier:I/355/gaiadr3"
RADIUS_ARCSEC: Final = 5.0
SELECTION: Final = "all"
DEC_SHIFT_DEG: Final = 180.0 / 3600.0
GRID_ARCSEC: Final[tuple[float, ...]] = (0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0)
CLASSES: Final[tuple[str, ...]] = ("STAR", "GALAXY", "QSO")
PLATEAU_FRACTION: Final = 0.995
MAX_FALSE_FRACTION: Final = 0.01
GROWTH_FACTOR: Final = 1.005
LARGE_FALSE_FRACTION: Final = 0.05
ECHO_TOLERANCE_ARCSEC: Final = 1e-5
VERDICT_RERUN: Final = "needs rerun at 10 arcsec"
VERDICT_ESCALATE: Final = "escalate"


class ExperimentError(Exception):
    """A precondition of the experiment failed; the message says what to check."""


def build_shifted_sample(
    n: int, seed: int, candidates: Path, real: Path
) -> tuple[pd.DataFrame, pd.DataFrame]:
    sample = sample_positions(candidates, n, seed)
    real_sids = pq.read_table(real, columns=["sid"]).column("sid").to_pylist()
    if not set(real_sids) <= set(sample["sid"]):
        extra = len(set(real_sids) - set(sample["sid"]))
        raise ExperimentError(
            f"{extra} sid of the real batch are not in the recreated sample; "
            "the candidates file or the seed differ from the real run"
        )
    real_all = pq.read_table(real, columns=["sid", "ra", "dec", "cls"]).to_pandas()
    unique = real_all.drop_duplicates("sid").set_index("sid")
    check = sample.set_index("sid").reindex(unique.index)
    same_cls = bool((check["cls"] == unique["cls"]).all())
    ra_diff = float(np.abs(check["ra"] - unique["ra"]).max()) * 3600.0
    dec_diff = float(np.abs(check["dec"] - unique["dec"]).max()) * 3600.0
    if not same_cls or max(ra_diff, dec_diff) > ECHO_TOLERANCE_ARCSEC:
        raise ExperimentError(
            "the recreated sample disagrees with the real batch "
            f'(class equal={same_cls}, max ra diff={ra_diff:.2e}", '
            f'max dec diff={dec_diff:.2e}")'
        )
    shifted = sample.copy()
    shifted["dec"] = shifted["dec"] + DEC_SHIFT_DEG
    top = float(shifted["dec"].max())
    if top > 90.0:
        raise ExperimentError(
            f"{int((shifted['dec'] > 90.0).sum())} shifted dec exceed 90 deg (max {top:.6f})"
        )
    return sample, shifted


def validate_echo(table: pa.Table, shifted: pd.DataFrame) -> None:
    got = table.select(["sid", "ra", "dec", "cls"]).to_pandas()
    expected = shifted.set_index("sid")
    unknown = ~got["sid"].isin(expected.index)
    if unknown.any():
        raise ExperimentError(
            f"{int(unknown.sum())} returned sid are not in the upload"
        )
    ref = expected.loc[got["sid"]]
    if not (ref["cls"].to_numpy() == got["cls"].to_numpy()).all():
        raise ExperimentError("returned class differs from the upload")
    diff_ra = np.abs(ref["ra"].to_numpy() - got["ra"].to_numpy()).max() * 3600.0
    diff_dec = np.abs(ref["dec"].to_numpy() - got["dec"].to_numpy()).max() * 3600.0
    worst = float(max(diff_ra, diff_dec))
    if worst > ECHO_TOLERANCE_ARCSEC:
        raise ExperimentError(
            f"echoed positions differ from the upload by up to {worst:.2e} arcsec"
        )


def nearest_per_object(pairs: pd.DataFrame) -> pd.Series:
    return pairs.groupby("sid")["angDist"].min()


def count_curves(nearest: pd.Series, classes: pd.Series) -> dict[str, list[int]]:
    cls = classes.reindex(nearest.index)
    out: dict[str, list[int]] = {}
    for name in CLASSES:
        values = nearest[cls == name].to_numpy()
        out[name] = [int((values <= r).sum()) for r in GRID_ARCSEC]
    return out


def decide(
    n_true: dict[str, list[int]],
    n_real: dict[str, list[int]],
    n_false: dict[str, list[int]],
) -> dict[str, object]:
    last = len(GRID_ARCSEC) - 1
    i4 = GRID_ARCSEC.index(4.0)
    r_c: dict[str, float] = {}
    growing: list[str] = []
    for name in CLASSES:
        plateau = n_true[name][last]
        r_c[name] = next(
            r
            for r, n in zip(GRID_ARCSEC, n_true[name], strict=True)
            if n >= PLATEAU_FRACTION * plateau
        )
        if plateau > n_true[name][i4] * GROWTH_FACTOR:
            growing.append(name)
    r_match = max(r_c.values())
    idx = GRID_ARCSEC.index(r_match)
    frac = {
        name: n_false[name][idx] / n_real[name][idx] if n_real[name][idx] else math.nan
        for name in CLASSES
    }
    if growing:
        verdict = VERDICT_RERUN
    elif all(f <= MAX_FALSE_FRACTION for f in frac.values()):
        verdict = f"accepted: r_match = {r_match:g} arcsec"
    else:
        verdict = VERDICT_ESCALATE
    return {
        "r_c": r_c,
        "r_match": r_match,
        "false_fraction_at_r_match": frac,
        "growing_at_5_arcsec": growing,
        "verdict": verdict,
    }


def format_table(
    n_real: dict[str, list[int]],
    n_false: dict[str, list[int]],
    n_true: dict[str, list[int]],
) -> str:
    lines = [
        f"{'class':<8}{'r_arcsec':>9}{'N_real':>9}{'N_false':>9}{'N_true':>9}{'f_false':>10}"
    ]
    for name in CLASSES:
        for i, r in enumerate(GRID_ARCSEC):
            real = n_real[name][i]
            frac = n_false[name][i] / real if real else math.nan
            lines.append(
                f"{name:<8}{r:>9g}{real:>9d}{n_false[name][i]:>9d}"
                f"{n_true[name][i]:>9d}{frac:>10.4%}"
            )
    return "\n".join(lines)


def write_json(path: Path, payload: dict[str, object]) -> None:
    fd, tmp_name = tempfile.mkstemp(dir=path.parent, prefix=".report.", suffix=".tmp")
    os.close(fd)
    try:
        Path(tmp_name).write_text(
            json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        os.replace(tmp_name, path)
    except BaseException:
        Path(tmp_name).unlink(missing_ok=True)
        raise


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=100_000)
    parser.add_argument("--seed", type=int, default=20261025)
    parser.add_argument("--candidates", type=Path, default=DEFAULT_CANDIDATES)
    parser.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR)
    args = parser.parse_args(argv)
    if args.n <= 0:
        parser.error("--n must be positive")

    started = time.perf_counter()
    outdir: Path = args.outdir
    real_path = (
        outdir / f"xmatch_gaiadr3_n{args.n}_r{RADIUS_ARCSEC:g}_{SELECTION}.parquet"
    )
    stem = f"xmatch_gaiadr3_offset180_n{args.n}_r{RADIUS_ARCSEC:g}_{SELECTION}"
    raw = outdir / f"{stem}.raw.csv"
    parquet = outdir / f"{stem}.parquet"
    upload = outdir / f"upload_offset180_n{args.n}_seed{args.seed}.csv"
    try:
        sample, shifted = build_shifted_sample(
            args.n, args.seed, args.candidates, real_path
        )
        print(f"sample_sid_set_matches_real_batch: True (rows={len(sample)})")
        print(f"max_shifted_dec_deg: {shifted['dec'].max():.6f}")
        write_upload_csv(shifted, upload)
        upload_sha = file_sha256(upload)
        if raw.exists():
            print(f"reusing the answer already on disk: {_display(raw)}")
            stats_bytes, attempts = raw.stat().st_size, 0
        else:
            stats = xmatch_to_file(
                upload,
                raw,
                catalog=CATALOG,
                radius_arcsec=RADIUS_ARCSEC,
                selection=SELECTION,
                ra_col="ra",
                dec_col="dec",
            )
            stats_bytes, attempts = stats.bytes_received, stats.attempts
        table = read_xmatch_csv(raw)
        validate_echo(table, shifted)
        real_pairs = pq.read_table(real_path, columns=["sid", "angDist"]).to_pandas()
    except IngestError as exc:
        print(f"XMatch request failed, stopping: {exc}", file=sys.stderr)
        return 1
    except ExperimentError as exc:
        print(f"precondition failed: {exc}", file=sys.stderr)
        return 1
    except (OSError, ValueError, KeyError, pa.ArrowException) as exc:
        print(f"failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1

    params = (
        f"request=xmatch; distMaxArcsec={RADIUS_ARCSEC:g}; selection={SELECTION}; "
        f"RESPONSEFORMAT=csv; cat2={CATALOG}; colRA1=ra; colDec1=dec; "
        f"cat1=<upload csv, columns sid,ra,dec,cls, {len(shifted)} rows, dec shifted by "
        f"+180 arcsec, sha256 {upload_sha}, seed {args.seed}>"
    )
    try:
        write_parquet(table, parquet)
        raw.unlink(missing_ok=True)
        manifest = make_manifest(
            parquet,
            name=stem,
            params=params,
            rows=table.num_rows,
            notes=[
                f"bytes_received={stats_bytes}",
                f"attempts={attempts}",
                "E1/G-A2 offset run: every pair is a chance coincidence",
                "sample: rng(seed).choice(n_candidates, n, replace=False), dec + 180 arcsec",
            ],
        )
    except (OSError, ValueError) as exc:
        print(f"cannot write outputs: {exc}", file=sys.stderr)
        return 1
    print(f"rows_returned_offset: {table.num_rows} bytes={stats_bytes}")
    print(f"parquet: {_display(parquet)} manifest: {_display(manifest)}")

    classes = sample.set_index("sid")["cls"]
    n_real = count_curves(nearest_per_object(real_pairs), classes)
    n_false = count_curves(
        nearest_per_object(table.select(["sid", "angDist"]).to_pandas()), classes
    )
    n_true = {
        name: [a - b for a, b in zip(n_real[name], n_false[name], strict=True)]
        for name in CLASSES
    }
    empty = [name for name in CLASSES if n_real[name][-1] == 0]
    if empty:
        print(f"no real matches for class {empty}; cannot decide", file=sys.stderr)
        return 1
    print(format_table(n_real, n_false, n_true))
    decision = decide(n_true, n_real, n_false)
    r_c = decision["r_c"]
    assert isinstance(r_c, dict)
    for name in CLASSES:
        print(f"r_c_{name}: {r_c[name]:g}")
    print(f"r_match: {decision['r_match']:g}")
    print(f"growing_at_5_arcsec: {decision['growing_at_5_arcsec']}")
    print(f"verdict: {decision['verdict']}")

    last = len(GRID_ARCSEC) - 1
    warn: dict[str, float] = {}
    for name in (*CLASSES, "ALL"):
        real_n = (
            sum(n_real[c][last] for c in CLASSES)
            if name == "ALL"
            else n_real[name][last]
        )
        false_n = (
            sum(n_false[c][last] for c in CLASSES)
            if name == "ALL"
            else n_false[name][last]
        )
        warn[name] = false_n / real_n
    print(
        "false_fraction_at_5_arcsec: "
        + " ".join(f"{k}={v:.3%}" for k, v in warn.items())
    )
    large = [k for k, v in warn.items() if v > LARGE_FALSE_FRACTION]
    if large:
        print(f"WARNING: N_false(5)/N_real(5) > 5% for {large}")

    report: dict[str, object] = {
        "adr": "0004 G-A2 (E1)",
        "grid_arcsec": list(GRID_ARCSEC),
        "n": args.n,
        "seed": args.seed,
        "dec_shift_arcsec": 180,
        "xmatch": {
            "cat2": CATALOG,
            "radius_arcsec": RADIUS_ARCSEC,
            "selection": SELECTION,
            "rows_returned": table.num_rows,
            "bytes_received": stats_bytes,
            "upload_sha256": upload_sha,
        },
        "curves": {
            name: {
                "N_real": n_real[name],
                "N_false": n_false[name],
                "N_true": n_true[name],
                "false_fraction": [
                    f / r if r else None
                    for f, r in zip(n_false[name], n_real[name], strict=True)
                ],
            }
            for name in CLASSES
        },
        "false_fraction_at_5_arcsec": warn,
        "large_false_fraction_classes": large,
        **decision,
    }
    report_path = outdir / "e1_report.json"
    try:
        write_json(report_path, report)
    except OSError as exc:
        print(f"cannot write {_display(report_path)}: {exc}", file=sys.stderr)
        return 1
    print(f"e1_report: {_display(report_path)}")
    print(f"elapsed_seconds: {time.perf_counter() - started:.1f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
