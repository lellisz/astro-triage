"""Tests for the labels/features builder (ADR 0002 section 3, ADR 0004 D1/D7, schema.md 3-4).

All inputs are tiny synthetic files with the same layout as the real XMatch outputs; nothing
under ``data/`` is read. Expected values are computed by hand in the comments.
"""

from __future__ import annotations

import csv
import json
import math
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
import pytest

from astro_triage.dataset import build as build_module
from astro_triage.dataset.build import BuildError, build_dataset

REPO_ROOT = Path(__file__).resolve().parents[1]
COLUMN_MAP = REPO_ROOT / "manifests" / "column_map_vizier_v1.csv"
PIXEL_MAP = REPO_ROOT / "manifests" / "split_pixel_map_v1.csv"
PIXEL_SPAN = 2**49  # Gaia source_id // 2**49 is the level-5 NESTED pixel.

GAIA_DEFAULTS: dict[str, Any] = {
    "RAdeg": 10.0,
    "DEdeg": 20.0,
    "e_RAdeg": 0.1,
    "e_DEdeg": 0.1,
    "Plx": 1.0,
    "e_Plx": 0.5,
    "pmRA": 1.0,
    "e_pmRA": 0.1,
    "pmDE": 1.0,
    "e_pmDE": 0.1,
    "RUWE": 1.0,
    "epsi": 0.2,
    "sepsi": 1.0,
    "Solved": 31,
    "IPDgofha": 0.01,
    "IPDfmp": 0,
    "IPDfow": 0,
    "Gmag": 18.0,
    "RFG": 50.0,
    "BPmag": 18.5,
    "RPmag": 17.5,
    "BP-RP": 1.0,
    "E(BP/RP)": 1.2,
    "PQSO": 0.9,
    "PGal": 0.05,
    "PSS": 0.05,
}
NAN = float("nan")


def pixel_folds() -> dict[int, str]:
    with PIXEL_MAP.open(encoding="utf-8", newline="") as handle:
        return {int(r["healpix_l5"]): r["fold"] for r in csv.DictReader(handle)}


def pixels_of(fold: str, count: int) -> list[int]:
    found = [p for p, f in pixel_folds().items() if f == fold]
    return found[:count]


def src(pixel: int, n: int = 1) -> int:
    return pixel * PIXEL_SPAN + n


def gaia_row(
    sid: str, source: int, ang: float, cls: str = "STAR", **over: Any
) -> dict[str, Any]:
    return {
        "sid": f"S{sid}",
        "angDist": ang,
        "cls": cls,
        "Source": np.int64(source),
        **GAIA_DEFAULTS,
        **over,
    }


def cand_row(sid: str, cls: str = "STAR", **over: Any) -> dict[str, Any]:
    return {
        "specobjid": sid,
        "plate": 1000,
        "mjd": 55000,
        "fiberid": 12,
        "class": cls,
        "subclass": "",
        "z": 0.1,
        "z_err": 0.001,
        "zwarning": 0,
        "plug_ra": 10.0,
        "plug_dec": 20.0,
        "specprimary": 1,
        **over,
    }


def para_row(
    source: int, qa: float, ga: float, sa: float, **over: Any
) -> dict[str, Any]:
    return {
        "Source": np.int64(source),
        "PQSO-A": qa,
        "PGal-A": ga,
        "Pstar-A": sa,
        "PQSO": 0.777,  # 5-class combmod values: must never end up in allosmod
        "PGal": 0.666,
        **over,
    }


def run_build(
    tmp_path: Path,
    gaia: list[dict[str, Any]],
    paramp: list[dict[str, Any]],
    cands: list[dict[str, Any]],
    **kwargs: Any,
) -> Any:
    gaia_path = tmp_path / "gaia.parquet"
    paramp_path = tmp_path / "paramp.parquet"
    cand_path = tmp_path / "cand.parquet"
    pd.DataFrame(gaia).to_parquet(gaia_path)
    pd.DataFrame(paramp).to_parquet(paramp_path)
    pd.DataFrame(cands).to_parquet(cand_path)
    return build_dataset(
        gaia_path=gaia_path,
        paramp_path=paramp_path,
        candidates_path=cand_path,
        column_map_path=COLUMN_MAP,
        outdir=tmp_path / "out",
        **kwargs,
    )


def read(result: Any, key: str) -> pd.DataFrame:
    return pq.read_table(result.paths[key]).to_pandas().set_index("source_id")


# --------------------------------------------------------------------------- scenario R / B

# Fixed ids: 19-digit specobjid strings.
O = {n: f"{i:019d}" for n, i in {"a": 11, "b": 12, "c": 13}.items()}


def test_r_match_limit_is_inclusive_and_parametrisable(tmp_path: Path) -> None:
    p = pixels_of("train", 3)
    gaia = [
        gaia_row(O["a"], src(p[0]), 0.75),  # exactly at the default radius: kept
        gaia_row(O["b"], src(p[1]), 0.7501),  # just above: dropped
        gaia_row(O["c"], src(p[2]), 0.5),
    ]
    cands = [cand_row(O[k]) for k in "abc"]
    res = run_build(tmp_path, gaia, [para_row(src(p[0]), 0.1, 0.2, 0.3)], cands)
    assert sorted(read(res, "labels").index) == sorted([src(p[0]), src(p[2])])
    out = res.report["discards_in_order"]["outside_r_match"]
    assert out == {"pairs": 1, "specobj_without_any_pair": 1}
    assert res.report["input_pairs"] == 3 and res.report["pairs_after_r_match"] == 2

    # Parametrised: r = 0.5 keeps only the pair at exactly 0.5.
    res2 = (
        run_build(
            tmp_path / "r2", gaia, [para_row(src(p[0]), 0, 0, 0)], cands, r_match=0.5
        )
        if ((tmp_path / "r2").mkdir() is None)
        else None
    )
    assert list(read(res2, "labels").index) == [src(p[2])]
    assert res2.report["r_match_arcsec"] == 0.5
    assert res2.report["discards_in_order"]["outside_r_match"]["pairs"] == 2


def test_r_match_validation(tmp_path: Path) -> None:
    p = pixels_of("train", 1)
    gaia = [gaia_row(O["a"], src(p[0]), 0.9)]
    with pytest.raises(BuildError, match="no pair within"):
        run_build(tmp_path, gaia, [para_row(src(p[0]), 0, 0, 0)], [cand_row(O["a"])])
    (tmp_path / "x").mkdir()
    with pytest.raises(BuildError, match="positive"):
        run_build(
            tmp_path / "x",
            gaia,
            [para_row(src(p[0]), 0, 0, 0)],
            [cand_row(O["a"])],
            r_match=0.0,
        )


def _ambiguity_inputs() -> tuple[
    list[dict[str, Any]], list[dict[str, Any]], dict[str, int]
]:
    p = pixels_of("train", 8)
    ids = {k: src(p[i]) for i, k in enumerate("abcdefgh")}
    s = lambda n: f"{n:019d}"
    gaia = [
        # O10 has two Gaia neighbours inside the radius -> specobj_multi_gaia.
        gaia_row(s(10), ids["a"], 0.2),
        gaia_row(s(10), ids["b"], 0.3),
        # Gaia c is matched by a GALAXY and a QSO object -> source_class_conflict.
        gaia_row(s(20), ids["c"], 0.1, "GALAXY"),
        gaia_row(s(21), ids["c"], 0.2, "QSO"),
        # Gaia d is matched by two STAR objects: the closer (O31, 0.2) is kept.
        gaia_row(s(30), ids["d"], 0.3, "STAR"),
        gaia_row(s(31), ids["d"], 0.2, "STAR"),
        # Gaia e: a clean QSO match.
        gaia_row(s(40), ids["e"], 0.4, "QSO"),
        # O60 is outside the radius.
        gaia_row(s(60), ids["f"], 0.9, "STAR"),
    ]
    cands = [
        cand_row(s(10)),
        cand_row(s(20), "GALAXY"),
        cand_row(s(21), "QSO"),
        cand_row(s(30)),
        cand_row(s(31)),
        cand_row(s(40), "QSO"),
        cand_row(s(60)),
    ]
    return gaia, cands, ids


def test_ambiguity_rules_and_counts_add_up(tmp_path: Path) -> None:
    gaia, cands, ids = _ambiguity_inputs()
    res = run_build(tmp_path, gaia, [para_row(ids["a"], 0.1, 0.2, 0.3)], cands)
    labels = read(res, "labels")
    # Only d (closest same-class object) and e survive.
    assert sorted(labels.index) == sorted([ids["d"], ids["e"]])
    assert labels.loc[ids["d"], "sdss_specobjid"] == f"{31:019d}"
    assert labels.loc[ids["d"], "angular_distance_arcsec"] == pytest.approx(0.2)
    assert labels.loc[ids["d"], "n_specobj_candidates"] == 2
    assert labels.loc[ids["e"], "n_specobj_candidates"] == 1
    assert ids["a"] not in labels.index and ids["b"] not in labels.index
    assert ids["c"] not in labels.index

    rep = res.report
    d = rep["discards_in_order"]
    assert d["outside_r_match"] == {"pairs": 1, "specobj_without_any_pair": 1}
    assert d["specobj_multi_gaia"] == {
        "specobj_by_class": {"STAR": 1, "GALAXY": 0, "QSO": 0},
        "pairs": 2,
        "gaia_sources_left_without_candidate": 2,
    }
    assert d["source_class_conflict"] == {
        "gaia_sources": 1,
        "specobj_by_class": {"STAR": 0, "GALAXY": 1, "QSO": 1},
    }
    assert d["same_class_duplicates_collapsed"] == {
        "gaia_sources": 1,
        "specobj_dropped_by_class": {"STAR": 1, "GALAXY": 0, "QSO": 0},
    }
    # Conservation: 7 SDSS objects in = 2 kept + 1 multi + 2 conflict + 1 collapsed + 1 outside.
    assert rep["input_specobj"] == 7
    assert rep["input_pairs"] == 8
    dropped = (
        sum(d["specobj_multi_gaia"]["specobj_by_class"].values())
        + sum(d["source_class_conflict"]["specobj_by_class"].values())
        + sum(d["same_class_duplicates_collapsed"]["specobj_dropped_by_class"].values())
        + d["outside_r_match"]["specobj_without_any_pair"]
    )
    assert rep["final_rows"] == 2
    assert rep["input_specobj"] == rep["final_rows"] + dropped
    assert rep["final_by_class"] == {"STAR": 1, "GALAXY": 0, "QSO": 1}
    assert sum(rep["final_by_fold"].values()) == 2


# --------------------------------------------------------------------------- scenario C


@pytest.fixture(scope="module")
def world(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Any]:
    tmp_path = tmp_path_factory.mktemp("world")
    train, val, test = (pixels_of(f, 3) for f in ("train", "val", "test"))
    ids = {
        "s1": src(train[0], 1),
        "s2": src(val[0], 1),
        "s3": src(test[0], 1),
        "s4": src(train[1], 1),
        "s5": src(val[1], 1),
        "s6": src(train[0], 2),  # same pixel as s1
    }
    pix = {
        "s1": train[0],
        "s2": val[0],
        "s3": test[0],
        "s4": train[1],
        "s5": val[1],
        "s6": train[0],
    }
    sobj = {
        "s1": "1234567890123456789",  # > 2**53: a float round-trip would corrupt it
        "s2": "2000000000000000002",
        "s3": "2000000000000000003",
        "s4": "2000000000000000004",
        "s5": "2000000000000000005",
        "s6": "2000000000000000006",
    }
    gaia = [
        # poe = 2.0 / 0.5 = 4.0 ; pm = hypot(3, 4) = 5.0
        gaia_row(
            sobj["s1"], ids["s1"], 0.1, "STAR", Plx=2.0, e_Plx=0.5, pmRA=3.0, pmDE=4.0
        ),
        # Solved == 3: astrometry NULL on disk, never imputed.
        gaia_row(
            sobj["s2"],
            ids["s2"],
            0.1,
            "GALAXY",
            Plx=NAN,
            e_Plx=NAN,
            pmRA=NAN,
            e_pmRA=NAN,
            pmDE=NAN,
            e_pmDE=NAN,
            RUWE=NAN,
            Solved=3,
        ),
        # e_Plx == 0 -> parallax_over_error NULL, never inf.
        gaia_row(sobj["s3"], ids["s3"], 0.1, "QSO", Plx=1.0, e_Plx=0.0),
        # Sgr A* position; Plx present but e_Plx NULL -> poe NULL.
        gaia_row(
            sobj["s4"],
            ids["s4"],
            0.1,
            "STAR",
            RAdeg=266.41683,
            DEdeg=-29.00781,
            Plx=1.0,
            e_Plx=NAN,
        ),
        # Plx NULL, e_Plx present -> poe NULL.
        gaia_row(sobj["s5"], ids["s5"], 0.1, "GALAXY", Plx=NAN, e_Plx=0.5),
        gaia_row(sobj["s6"], ids["s6"], 0.1, "QSO", Plx=3.0, e_Plx=1.5),
    ]
    paramp = [
        para_row(ids["s1"], 0.11, 0.22, 0.33),
        para_row(ids["s2"], NAN, NAN, NAN),  # row present, allosmod NULL
        para_row(ids["s5"], 0.5, NAN, NAN),  # partial allosmod
        para_row(ids["s6"], 0.61, 0.62, 0.63),
        para_row(
            src(test[2], 9), 0.9, 0.9, 0.9
        ),  # Gaia source not in any pair: ignored
        # s3 and s4 deliberately have no paramp row.
    ]
    cands = [
        cand_row(sobj["s1"], "STAR", z=0.0001, subclass="G2"),
        cand_row(sobj["s2"], "GALAXY", z=0.3, z_err=NAN),
        cand_row(sobj["s3"], "QSO ", z=1.5),
        cand_row(sobj["s4"], "STAR"),
        cand_row(sobj["s5"], "GALAXY"),
        cand_row(sobj["s6"], "QSO"),
    ]
    res = run_build(tmp_path, gaia, paramp, cands)
    return {"res": res, "ids": ids, "pix": pix, "sobj": sobj, "gaia": gaia}


def test_sid_prefix_removed_and_19_digit_specobjid_preserved(
    world: dict[str, Any],
) -> None:
    labels = read(world["res"], "labels")
    table = pq.read_table(world["res"].paths["labels"])
    assert pa.types.is_string(table.schema.field("sdss_specobjid").type)
    for key, sobj in world["sobj"].items():
        got = labels.loc[world["ids"][key], "sdss_specobjid"]
        assert got == sobj and len(got) == 19 and got.isdigit()
    assert labels.loc[world["ids"]["s1"], "sdss_specobjid"] == "1234567890123456789"


def test_malformed_sid_is_rejected(tmp_path: Path) -> None:
    p = pixels_of("train", 1)
    row = gaia_row(O["a"], src(p[0]), 0.1)
    row["sid"] = "X" + O["a"]
    with pytest.raises(BuildError, match="'S' followed by digits"):
        run_build(tmp_path, [row], [para_row(src(p[0]), 0, 0, 0)], [cand_row(O["a"])])


def test_canonical_names_galactic_and_derived_columns(world: dict[str, Any]) -> None:
    ids = world["ids"]
    labels = read(world["res"], "labels")
    feats = read(world["res"], "features")
    # Sgr A* (ra, dec = 266.41683, -29.00781): (l, b) = (359.944, -0.046) within 0.01 deg.
    assert labels.loc[ids["s4"], "l"] == pytest.approx(359.944, abs=0.01)
    assert labels.loc[ids["s4"], "b"] == pytest.approx(-0.046, abs=0.01)
    assert labels.loc[ids["s4"], "ra"] == pytest.approx(266.41683)
    assert labels.loc[ids["s4"], "dec"] == pytest.approx(-29.00781)
    # Canonical names come from the column map (VizieR names must not leak).
    for vizier in ("Plx", "e_Plx", "pmRA", "RUWE", "Solved", "BP-RP", "E(BP/RP)"):
        assert vizier not in feats.columns
    row = feats.loc[ids["s1"]]
    assert row["parallax"] == pytest.approx(2.0)
    assert row["parallax_error"] == pytest.approx(0.5)
    assert row["parallax_over_error"] == pytest.approx(4.0)  # 2.0 / 0.5
    assert row["pm"] == pytest.approx(5.0)  # hypot(3, 4)
    assert row["bp_rp"] == pytest.approx(1.0)
    assert row["phot_bp_rp_excess_factor"] == pytest.approx(1.2)
    assert row["astrometric_params_solved"] == 31
    assert feats.loc[ids["s6"], "parallax_over_error"] == pytest.approx(
        2.0
    )  # 3.0 / 1.5
    # NULL poe when Plx NULL, e_Plx NULL, or e_Plx == 0.
    for key in ("s2", "s3", "s4", "s5"):
        assert math.isnan(feats.loc[ids[key], "parallax_over_error"]), key


def test_null_in_every_float_column_is_never_nan_or_inf(world: dict[str, Any]) -> None:
    ids = world["ids"]
    for key in ("labels", "features"):
        table = pq.read_table(world["res"].paths[key])
        for field in table.schema:
            if pa.types.is_floating(field.type):
                col = table[field.name]
                assert True not in pc.is_nan(col).to_pylist(), (key, field.name)
                assert True not in pc.is_inf(col).to_pylist(), (key, field.name)
    table = pq.read_table(world["res"].paths["features"])
    by_id = {sid: i for i, sid in enumerate(table["source_id"].to_pylist())}
    i2 = by_id[ids["s2"]]  # Solved == 3 row
    assert table["astrometric_params_solved"][i2].as_py() == 3
    for name in (
        "parallax",
        "parallax_error",
        "parallax_over_error",
        "pm",
        "pmra",
        "pmra_error",
        "pmdec",
        "pmdec_error",
        "ruwe",
    ):
        assert table[name][i2].as_py() is None, name
    # Not imputed elsewhere either: exactly the rows we nulled are NULL.
    assert table["parallax"].null_count == 2  # s2, s5
    assert table["pmra"].null_count == 1
    assert table["ruwe"].null_count == 1
    assert table["parallax_over_error"].null_count == 4  # s2, s3, s4, s5
    assert table["parallax_error"].null_count == 2  # s2, s4
    assert table["pm"].null_count == 1


def test_features_exclude_forbidden_columns_and_z_stays_in_labels(
    world: dict[str, Any],
) -> None:
    feats = pq.read_table(world["res"].paths["features"])
    labels = pq.read_table(world["res"].paths["labels"])
    forbidden = {
        "in_qso_candidates",
        "in_galaxy_candidates",
        "ap_row_present",
        "ra",
        "dec",
        "l",
        "b",
        "z",
        "z_err",
        "sdss_z",
        "sdss_z_err",
        "classe",
        "class",
        "fold",
        "sdss_class",
        "healpix_l5",
    }
    assert not forbidden & set(feats.column_names)
    assert not [c for c in feats.column_names if c.startswith("classprob_dsc_")]
    assert not [c for c in feats.column_names if c.startswith("sdss_")]
    assert {"sdss_z", "sdss_z_err", "fold", "sdss_class"} <= set(labels.column_names)
    assert feats.column_names[0] == "source_id"
    assert all(t == pa.float32() for t in feats.schema.types[1:])


def test_ap_row_present_left_join_and_allosmod_columns(world: dict[str, Any]) -> None:
    ids = world["ids"]
    labels = read(world["res"], "labels")
    # LEFT JOIN never loses a Gaia row: 6 sources in, 6 labels out.
    assert len(labels) == 6
    assert labels["ap_row_present"].to_dict() == {
        ids["s1"]: True,
        ids["s2"]: True,  # row present although allosmod is NULL
        ids["s3"]: False,
        ids["s4"]: False,
        ids["s5"]: True,
        ids["s6"]: True,
    }
    aq, ag, ast = (
        "classprob_dsc_allosmod_quasar",
        "classprob_dsc_allosmod_galaxy",
        "classprob_dsc_allosmod_star",
    )
    # allosmod comes from PQSO-A/PGal-A/Pstar-A, never from the 5-class PQSO/PGal (0.777/0.666).
    r1 = labels.loc[ids["s1"]]
    assert (r1[aq], r1[ag], r1[ast]) == (
        pytest.approx(0.11),
        pytest.approx(0.22),
        pytest.approx(0.33),
    )
    r6 = labels.loc[ids["s6"]]
    assert r6[aq] == pytest.approx(0.61) and r6[ag] == pytest.approx(0.62)
    for key in ("s2", "s3", "s4"):
        assert labels.loc[ids[key], [aq, ag, ast]].isna().all(), key
    assert labels.loc[ids["s5"], aq] == pytest.approx(0.5)
    assert labels.loc[ids["s5"], [ag, ast]].isna().all()
    assert not np.isclose(
        labels[[aq, ag]].to_numpy(dtype=float), 0.777, equal_nan=True
    ).any()
    assert not np.isclose(
        labels[[aq, ag]].to_numpy(dtype=float), 0.666, equal_nan=True
    ).any()
    # combmod comes from the gaiadr3 pass (PQSO=0.9, PGal=0.05, PSS=0.05), also for rows without
    # a paramp row.
    assert labels.loc[ids["s3"], "classprob_dsc_combmod_quasar"] == pytest.approx(0.9)
    assert labels.loc[ids["s3"], "classprob_dsc_combmod_star"] == pytest.approx(0.05)


def test_dsc_coverage_report(world: dict[str, Any]) -> None:
    dsc = world["res"].report["dsc_coverage_by_class"]
    assert dsc["STAR"]["n"] == 2 and dsc["STAR"]["layer1_no_paramp_row"] == 1  # s4
    assert dsc["STAR"]["covered_S_dsc"] == 1  # s1
    assert dsc["GALAXY"]["layer1_no_paramp_row"] == 0
    assert (
        dsc["GALAXY"]["layer2_row_without_allosmod"] == 2
    )  # s2 (all NULL), s5 (partial)
    assert dsc["GALAXY"]["covered_S_dsc"] == 0
    assert dsc["QSO"]["layer1_no_paramp_row"] == 1  # s3
    assert dsc["QSO"]["covered_S_dsc"] == 1  # s6
    assert world["res"].report["dsc_allosmod_partial_null_rows"] == 1  # s5


def test_fold_matches_pixel_map_and_pixels_never_split(world: dict[str, Any]) -> None:
    pixel_map = pixel_folds()
    labels = read(world["res"], "labels")
    assert labels.index.is_unique
    seen_folds = set()
    for key, source_id in world["ids"].items():
        pixel = world["pix"][key]
        assert labels.loc[source_id, "healpix_l5"] == pixel
        assert labels.loc[source_id, "fold"] == pixel_map[pixel], key
        seen_folds.add(labels.loc[source_id, "fold"])
    assert seen_folds == {"train", "val", "test"}  # the check is not trivial
    # s1 and s6 share a pixel and therefore a fold.
    assert (
        labels.loc[world["ids"]["s1"], "fold"] == labels.loc[world["ids"]["s6"], "fold"]
    )
    assert world["res"].report["pixels_in_more_than_one_fold"] == 0
    by_fold = world["res"].report["final_by_fold"]
    assert by_fold == {"train": 3, "val": 2, "test": 1}  # s1,s4,s6 / s2,s5 / s3


def test_report_file_matches_returned_report_and_counts(world: dict[str, Any]) -> None:
    res = world["res"]
    on_disk = json.loads(res.paths["report"].read_text(encoding="utf-8"))
    assert on_disk == json.loads(json.dumps(res.report))
    assert on_disk["final_rows"] == 6
    assert on_disk["final_by_class"] == {"STAR": 2, "GALAXY": 2, "QSO": 2}
    assert on_disk["final_by_class_and_fold"]["STAR"] == {
        "train": 2,
        "val": 0,
        "test": 0,
    }
    assert on_disk["input_pairs"] == on_disk["pairs_after_r_match"] == 6
    assert on_disk["discards_in_order"]["outside_r_match"]["pairs"] == 0
    assert on_disk["solved3_fraction_by_class"]["GALAXY"] == pytest.approx(
        0.5
    )  # s2 of s2,s5
    assert on_disk["solved3_fraction_by_class"]["STAR"] == 0.0


# --------------------------------------------------------------------------- atomic writes


def _small_inputs() -> tuple[
    list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]
]:
    p = pixels_of("train", 1)
    return (
        [gaia_row(O["a"], src(p[0]), 0.1)],
        [para_row(src(p[0]), 0.1, 0.2, 0.3)],
        [cand_row(O["a"])],
    )


@pytest.mark.parametrize("fail_at", [1, 3])
def test_failed_replace_leaves_no_tmp_and_previous_output_intact(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fail_at: int
) -> None:
    gaia, paramp, cands = _small_inputs()
    first = run_build(tmp_path, gaia, paramp, cands)
    outdir = first.paths["labels"].parent
    before = {p.name: p.read_bytes() for p in outdir.iterdir() if p.suffix != ".tmp"}

    real_replace = os.replace
    calls = {"n": 0}

    def flaky(src_path: Any, dst_path: Any) -> None:
        calls["n"] += 1
        if calls["n"] == fail_at:
            raise OSError("simulated replace failure")
        real_replace(src_path, dst_path)

    monkeypatch.setattr(build_module.os, "replace", flaky)
    with pytest.raises(OSError, match="simulated replace failure"):
        build_dataset(
            gaia_path=tmp_path / "gaia.parquet",
            paramp_path=tmp_path / "paramp.parquet",
            candidates_path=tmp_path / "cand.parquet",
            column_map_path=COLUMN_MAP,
            outdir=outdir,
        )
    monkeypatch.undo()

    assert list(outdir.glob("*.tmp")) == []
    assert list(outdir.glob(".*")) == []
    # Files not yet replaced are byte-identical to the previous run and still readable.
    names = ["labels.parquet", "features.parquet", "dataset_report.json"]
    for name in names[fail_at - 1 :]:
        assert (outdir / name).read_bytes() == before[name], name
    assert pq.read_table(outdir / "labels.parquet").num_rows == 1
    assert (
        json.loads((outdir / "dataset_report.json").read_text(encoding="utf-8"))[
            "final_rows"
        ]
        == 1
    )
