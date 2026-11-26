"""Tests for the Phase 1 evaluation runner and scripts/evaluate.py (ADR 0002 s4, ADR 0003).

All data is synthetic and written to ``tmp_path``; nothing under ``data/`` is read.
Source ids are built as ``pixel * 2**35 * 4**7 + serial`` so that the runner's pixel checks
(``healpix_l5`` and the committed pixel -> fold map) hold for the synthetic rows.
"""

from __future__ import annotations

import importlib.util
import json
import re
import sys
from dataclasses import dataclass, field
from functools import cache
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from astro_triage.evaluation import runner as R
from astro_triage.split import build_pixel_map
from eval import metrics as M

CLS = M.CLASSES  # ("STAR", "GALAXY", "QSO")
DIV = 2**35 * 4**7
REPO_ROOT = Path(__file__).resolve().parents[1]
NAN = float("nan")


# ------------------------------------------------------------------------ synthetic data


@cache
def _pixel_map() -> pd.DataFrame:
    return build_pixel_map()


FILE_ORDER = ("STAR", "QSO", "GALAXY")  # star, qso, galaxy: differs from CLS order


def _proba(idx: np.ndarray, hi: float = 0.8) -> np.ndarray:
    """Probabilities in CLS order with ``hi`` on class ``idx`` and the rest split evenly."""
    p = np.full((len(idx), 3), (1.0 - hi) / 2.0)
    p[np.arange(len(idx)), idx] = hi
    return p


@dataclass
class Dataset:
    labels: pd.DataFrame
    features: pd.DataFrame
    preds: dict[str, pd.DataFrame]
    report: dict
    test_sid: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=np.int64))

    def test_rows(self, cls: str | None = None) -> pd.Index:
        m = self.labels["fold"] == "test"
        if cls is not None:
            m &= self.labels["sdss_class"] == cls
        return self.labels.index[m]

    def set_dsc(
        self, which: str, rows: pd.Index, proba_cls_order: np.ndarray | None
    ) -> None:
        """Write DSC probabilities (CLS order) by column NAME; None writes NaN."""
        for j, c in enumerate(CLS):
            col = f"classprob_dsc_{which}_{ {'STAR': 'star', 'GALAXY': 'galaxy', 'QSO': 'quasar'}[c] }"
            self.labels.loc[rows, col] = (
                NAN if proba_cls_order is None else proba_cls_order[:, j]
            )

    def set_lgbm(self, name: str, sid: np.ndarray, proba_cls_order: np.ndarray) -> None:
        df = self.preds[name]
        pos = df.index[df["source_id"].isin(sid)]
        order = (
            df.loc[pos, "source_id"].map({s: i for i, s in enumerate(sid)}).to_numpy()
        )
        for j, c in enumerate(CLS):
            df.loc[pos, f"p_cal_{c.lower()}"] = proba_cls_order[order, j]

    def write(self, outdir: Path) -> Path:
        outdir.mkdir(parents=True, exist_ok=True)
        self.labels.to_parquet(outdir / "labels.parquet")
        self.features.to_parquet(outdir / "features.parquet")
        for name, df in self.preds.items():
            df.to_parquet(outdir / f"predictions_{name}_v0.parquet")
        (outdir / "train_report.json").write_text(
            json.dumps(self.report), encoding="utf-8"
        )
        return outdir


def make_data(
    per_band: int = 9,
    counts: dict[str, int] | None = None,
    solved: str = "cycle",
) -> Dataset:
    """``counts`` = objects per class per test pixel; DSC and LightGBM are perfect by default."""
    counts = counts or {"STAR": 5, "GALAXY": 5, "QSO": 5}
    pmap = _pixel_map()
    test_px = []
    for band in ("A", "B", "C"):
        sel = pmap[(pmap["fold"] == "test") & (pmap["lat_band"] == band)]
        test_px += sel["healpix_l5"].astype(int).to_list()[:per_band]
    train_px = int(pmap.loc[pmap["fold"] == "train", "healpix_l5"].iloc[0])
    val_px = int(pmap.loc[pmap["fold"] == "val", "healpix_l5"].iloc[0])
    band_of = dict(zip(pmap["healpix_l5"].astype(int), pmap["lat_band"]))

    recs = []
    serial = 1
    for px in test_px:
        for c in CLS:
            for _ in range(counts[c]):
                recs.append((px * DIV + serial, c, "test", px))
                serial += 1
    for fold, px in (("train", train_px), ("val", val_px)):
        for c in CLS:
            recs.append((px * DIV + serial, c, fold, px))
            serial += 1
    lab = pd.DataFrame(recs, columns=["source_id", "sdss_class", "fold", "healpix_l5"])
    lab["b"] = lab["healpix_l5"].map(
        lambda p: {"A": 5.0, "B": 20.0, "C": 60.0}[band_of[int(p)]]
    )
    lab["ap_row_present"] = True
    true_idx = lab["sdss_class"].map({c: i for i, c in enumerate(CLS)}).to_numpy()
    dsc_names = {"STAR": "star", "GALAXY": "galaxy", "QSO": "quasar"}
    for which in ("allosmod", "combmod"):
        p = _proba(true_idx)
        for c in FILE_ORDER:
            lab[f"classprob_dsc_{which}_{dsc_names[c]}"] = p[:, CLS.index(c)]

    feat = pd.DataFrame({"source_id": lab["source_id"].to_numpy()})
    n = len(feat)
    if solved == "cycle":  # each class cycles over the three strata
        within = (
            lab.groupby("sdss_class").cumcount().to_numpy()
        )  # cycles inside each class
        feat["astrometric_params_solved"] = np.array([3, 31, 95])[within % 3]
    else:
        feat["astrometric_params_solved"] = 95
    feat["phot_g_mean_mag"] = 15.0 + (np.arange(n) % 5)
    feat["bp_rp"] = 0.5 + 0.25 * (np.arange(n) % 5)

    vt = lab["fold"].isin(["val", "test"]).to_numpy()
    preds = {}
    for name in R.LGBM_MODELS:
        df = pd.DataFrame({"source_id": lab.loc[vt, "source_id"].to_numpy()})
        p = _proba(true_idx[vt])
        for c in FILE_ORDER:
            df[f"p_cal_{c.lower()}"] = p[:, CLS.index(c)]
        preds[name] = df
    report = {
        "training_weights": "none (natural class frequencies)",
        "models": [
            {
                "model_id": name,
                "train_class_fractions": {c: 1.0 / 3.0 for c in CLS},
            }
            for name in R.LGBM_MODELS
        ],
    }
    ds = Dataset(lab, feat, preds, report)
    ds.test_sid = lab.loc[lab["fold"] == "test", "source_id"].to_numpy()
    return ds


def run(ds: Dataset, tmp: Path, n_boot: int = 20) -> dict:
    outdir = ds.write(tmp)
    return R.evaluate(outdir, n_boot=n_boot, workers=1)


def _blk(res: dict, pop: str, pid: str, variant: str = "as_shipped") -> dict:
    return res["populations"][pop]["priors"][pid]["variants"][variant]


# ------------------------------------------------------------------------------ 1) names


def test_dsc_columns_are_mapped_by_name_not_position() -> None:
    labels = pd.DataFrame(
        {  # file order star, quasar, galaxy
            "classprob_dsc_allosmod_star": [0.7],
            "classprob_dsc_allosmod_quasar": [0.2],
            "classprob_dsc_allosmod_galaxy": [0.1],
        }
    )
    got = R._dsc_columns(labels, "allosmod")
    # CLS order is STAR, GALAXY, QSO: a positional read would give [0.7, 0.2, 0.1]
    assert got.tolist() == [[0.7, 0.1, 0.2]]


def test_lgbm_proba_is_mapped_by_name() -> None:
    pred = pd.DataFrame(
        {"p_cal_star": [0.6], "p_cal_qso": [0.3], "p_cal_galaxy": [0.1]}
    )
    assert R._lgbm_proba(pred, "m").tolist() == [[0.6, 0.1, 0.3]]


@pytest.fixture(scope="module")
def names_result(tmp_path_factory: pytest.TempPathFactory) -> dict:
    ds = make_data(per_band=2, counts={"STAR": 2, "GALAXY": 2, "QSO": 2})
    gal = ds.test_rows("GALAXY")
    sid = ds.labels.loc[gal[:4], "source_id"].to_numpy()
    # first 4 GALAXY rows predicted QSO by lgbm_full (file columns are star, qso, galaxy)
    ds.set_lgbm("lgbm_full", sid, _proba(np.full(4, CLS.index("QSO"))))
    qso = ds.test_rows("QSO")
    ds.set_dsc("allosmod", qso[:3], _proba(np.full(3, CLS.index("GALAXY"))))
    return run(ds, tmp_path_factory.mktemp("names"))


def test_accuracy_and_confusion_follow_class_names(names_result: dict) -> None:
    d = _blk(names_result, "a_all", "P0")["models"]["lgbm_full"]["detail"]
    # rows true STAR, GALAXY, QSO; columns predicted STAR, GALAXY, QSO
    assert d["confusion_counts"] == [[12, 0, 0], [0, 8, 4], [0, 0, 12]]
    assert d["accuracy"] == pytest.approx(32 / 36)
    assert d["n_predicted"] == {"STAR": 12, "GALAXY": 8, "QSO": 16}
    perfect = _blk(names_result, "a_all", "P0")["models"]["lgbm_noind"]["detail"]
    assert perfect["confusion_counts"] == [[12, 0, 0], [0, 12, 0], [0, 0, 12]]


def test_dsc_confusion_follows_class_names(names_result: dict) -> None:
    d = _blk(names_result, "c_sdsc", "P0")["models"]["dsc_allosmod"]["detail"]
    assert d["confusion_counts"] == [[12, 0, 0], [0, 12, 0], [0, 3, 9]]
    assert d["accuracy"] == pytest.approx(33 / 36)


# ------------------------------------------------------------------- 2) DSC absence


@pytest.fixture(scope="module")
def missing_result(tmp_path_factory: pytest.TempPathFactory) -> dict:
    ds = make_data(
        per_band=2, counts={"STAR": 2, "GALAXY": 2, "QSO": 2}, solved="cycle"
    )
    qso = ds.test_rows("QSO")
    ds.set_dsc(
        "allosmod", qso, None
    )  # all 12 QSO rows lack allosmod: STAR-mapping is visible
    star = ds.test_rows("STAR")
    ds.labels.loc[star[0], "classprob_dsc_allosmod_galaxy"] = NAN  # one NaN column only
    gal = ds.test_rows("GALAXY")
    ds.set_dsc("combmod", gal[:4], None)
    return run(ds, tmp_path_factory.mktemp("missing"))


def test_missing_allosmod_rows_define_s_dsc_for_every_model(
    missing_result: dict,
) -> None:
    pop = missing_result["populations"]["c_sdsc"]
    assert pop["n_rows"] == 36 - 12 - 1  # 23 rows with all three allosmod probabilities
    assert pop["n_by_class"] == {"STAR": 11, "GALAXY": 12, "QSO": 0}
    for name in ("lgbm_full", "lgbm_noind", "dsc_allosmod"):
        d = _blk(missing_result, "c_sdsc", "P0")["models"][name]["detail"]
        assert int(np.sum(d["confusion_counts"])) == 23, name
        assert [sum(r) for r in d["confusion_counts"]] == [11, 12, 0], name


def test_missing_dsc_is_not_mapped_to_star_and_is_not_an_error(
    missing_result: dict,
) -> None:
    d = _blk(missing_result, "c_sdsc", "P0")["models"]["dsc_allosmod"]["detail"]
    # DSC is perfect on covered rows: 11 STAR + 12 GALAXY, nothing predicted QSO or extra STAR
    assert d["n_predicted"] == {"STAR": 11, "GALAXY": 12, "QSO": 0}
    assert d["accuracy"] == 1.0


def test_lgbm_full_population_keeps_all_rows_while_dsc_cut_does_not(
    missing_result: dict,
) -> None:
    assert missing_result["populations"]["a_all"]["n_rows"] == 36
    assert set(missing_result["populations"]["a_all"]["models"]) == set(R.LGBM_MODELS)


def test_combmod_population_uses_its_own_rows(missing_result: dict) -> None:
    pop = missing_result["populations"]["a_combmod"]
    assert pop["n_rows"] == 32
    assert pop["n_by_class"] == {"STAR": 12, "GALAXY": 8, "QSO": 12}
    d = _blk(missing_result, "a_combmod", "P0")["models"]["lgbm_full"]["detail"]
    assert int(np.sum(d["confusion_counts"])) == 32


def test_coverage_counts_per_class_and_per_solved(missing_result: dict) -> None:
    cov = missing_result["coverage"]
    l2 = cov["overall"]["layer2_allosmod_non_null"]
    assert l2["STAR"] == {"n_covered": 11, "n": 12}
    assert l2["GALAXY"] == {"n_covered": 12, "n": 12}
    assert l2["QSO"] == {"n_covered": 0, "n": 12}
    assert l2["ALL"] == {"n_covered": 23, "n": 36}
    assert cov["overall"]["combmod_non_null"]["GALAXY"] == {"n_covered": 8, "n": 12}
    by = cov["by_solved"]
    assert sum(by[c]["n"] for c in by) == 36
    assert sum(by[c]["layer2_allosmod_non_null"]["ALL"]["n_covered"] for c in by) == 23
    for c in by:  # per-class counts of a stratum add up to its total
        t = by[c]["layer2_allosmod_non_null"]
        assert sum(t[k]["n"] for k in CLS) == t["ALL"]["n"]
        assert sum(t[k]["n_covered"] for k in CLS) == t["ALL"]["n_covered"]


# ---------------------------------------------------------------- 3) combmod star score


def _comb_dataset() -> tuple[Dataset, np.ndarray]:
    ds = make_data(per_band=2, counts={"STAR": 1, "GALAXY": 1, "QSO": 1})
    rows = ds.test_rows()[:4]
    # (star delivered, galaxy, quasar); column values are written by name
    vals = [
        (0.9, 0.1, 0.2),  # star score 0.7, argmax STAR
        (0.1, 0.2, 0.5),  # star score 0.3, argmax QSO
        (0.0, 0.6, 0.7),  # q + g = 1.3 > 1: star score clipped to 0, no NaN
        (0.5, 0.5, 0.0),  # star score 0.5, tie with galaxy: STAR wins (first index)
    ]
    for r, (s, g, q) in zip(rows, vals):
        ds.labels.loc[r, "classprob_dsc_combmod_star"] = s
        ds.labels.loc[r, "classprob_dsc_combmod_galaxy"] = g
        ds.labels.loc[r, "classprob_dsc_combmod_quasar"] = q
    return ds, ds.labels.loc[rows, "source_id"].to_numpy()


def test_combmod_star_score_is_one_minus_quasar_minus_galaxy(tmp_path: Path) -> None:
    ds, sid = _comb_dataset()
    ts = R.load_test_set(ds.write(tmp_path))
    md = ts.models["dsc_combmod"]
    pos = {s: i for i, s in enumerate(ts.source_id)}
    got = md.proba[[pos[s] for s in sid]]
    assert np.isfinite(got).all()
    np.testing.assert_allclose(got[0], [0.7, 0.1, 0.2])
    np.testing.assert_allclose(got[1], [0.3, 0.2, 0.5])
    np.testing.assert_allclose(got[2], [0.0, 0.6, 0.7])  # clipped, not negative or NaN
    np.testing.assert_allclose(got[3], [0.5, 0.5, 0.0])
    assert M.predict_argmax(got).tolist() == ["STAR", "QSO", "QSO", "STAR"]
    # the delivered star column is only compared, never used: row 0 differs by 0.2, row 2 by 0
    assert ts.combmod_star_col_max_abs_diff == pytest.approx(0.2)


def test_combmod_with_sum_above_one_survives_the_prior_shift() -> None:
    p = np.array([[0.0, 0.6, 0.7]])  # clipped star score, row sums to 1.3
    out = R.shift_prior(p, R.COMBMOD_PI_SRC, R.GRID["P1"])
    assert np.isfinite(out).all()
    assert out.sum() == pytest.approx(1.0)


def test_allosmod_argmax_runs_over_the_three_classes(tmp_path: Path) -> None:
    ds = make_data(per_band=2, counts={"STAR": 1, "GALAXY": 1, "QSO": 1})
    rows = ds.test_rows()[:3]
    ds.set_dsc(
        "allosmod",
        rows,
        np.array([[0.2, 0.7, 0.1], [0.1, 0.1, 0.8], [0.6, 0.3, 0.1]]),  # CLS order
    )
    ts = R.load_test_set(ds.write(tmp_path))
    pos = {s: i for i, s in enumerate(ts.source_id)}
    idx = [pos[s] for s in ds.labels.loc[rows, "source_id"]]
    pred = M.predict_argmax(ts.models["dsc_allosmod"].proba[idx])
    assert pred.tolist() == ["GALAXY", "QSO", "STAR"]


# ------------------------------------------------------------------ 4) prior shift


def test_shift_prior_hand_value() -> None:
    p = np.array([[0.4, 0.4, 0.2]])  # STAR, GALAXY, QSO
    src = {"STAR": 0.5, "GALAXY": 0.25, "QSO": 0.25}
    tgt = {"STAR": 0.8, "GALAXY": 0.1, "QSO": 0.1}
    # ratio = [1.6, 0.4, 0.4]; p * ratio = [0.64, 0.16, 0.08]; sum 0.88
    got = R.shift_prior(p, src, tgt)
    np.testing.assert_allclose(got, [[8 / 11, 2 / 11, 1 / 11]])
    np.testing.assert_allclose(R.shift_prior(p, src, src), p)  # same prior: identity


def test_variant_proba_never_shifts_without_pi_src_or_as_shipped() -> None:
    p = np.array([[0.4, 0.4, 0.2]])
    tgt = R.GRID["P1"]
    assert R.variant_proba(p, None, tgt, "prior_matched") is p
    assert R.variant_proba(p, {c: 1 / 3 for c in CLS}, tgt, "as_shipped") is p
    assert not np.allclose(
        R.variant_proba(p, {c: 1 / 3 for c in CLS}, tgt, "prior_matched"), p
    )


@pytest.fixture(scope="module")
def shift_result(tmp_path_factory: pytest.TempPathFactory) -> dict:
    ds = make_data(per_band=2, counts={"STAR": 2, "GALAXY": 2, "QSO": 2})
    qso = ds.test_rows("QSO")
    sid = ds.labels.loc[qso, "source_id"].to_numpy()
    # QSO prob 0.7 vs STAR 0.2: native argmax QSO, but P1 prior (STAR 0.998) flips it to STAR
    p = np.tile([0.2, 0.1, 0.7], (len(sid), 1))
    ds.set_lgbm("lgbm_full", sid, p)
    ds.set_dsc("allosmod", qso, p)
    return run(ds, tmp_path_factory.mktemp("shift"))


def test_prior_matched_shifts_lgbm_but_not_allosmod(shift_result: dict) -> None:
    ship = _blk(shift_result, "c_sdsc", "P1", "as_shipped")["models"]
    match = _blk(shift_result, "c_sdsc", "P1", "prior_matched")["models"]
    assert ship["lgbm_full"]["detail"]["confusion_counts"][2] == [0, 0, 12]
    assert match["lgbm_full"]["detail"]["confusion_counts"][2] == [
        12,
        0,
        0,
    ]  # flipped to STAR
    assert match["lgbm_full"]["shifted"] is True
    assert match["dsc_allosmod"]["shifted"] is False
    # allosmod is identical in both variants: same metrics and same detail
    assert ship["dsc_allosmod"]["detail"] == match["dsc_allosmod"]["detail"]
    for m, cell in ship["dsc_allosmod"]["metrics"].items():
        other = match["dsc_allosmod"]["metrics"].get(m)
        if other is not None:  # ECE only exists on prior_matched
            assert cell["point"] == other["point"] or (
                cell["point"] != cell["point"] and other["point"] != other["point"]
            ), m
    assert ship["dsc_allosmod"]["detail"]["confusion_counts"][2] == [0, 0, 12]


# -------------------------------------------------------------------------- 5) regimes


@pytest.fixture(scope="module")
def ident_result(tmp_path_factory: pytest.TempPathFactory) -> dict:
    """Floor-passing data (33 pixels, 165 objects per class); lgbm_full == lgbm_noind."""
    ds = make_data(per_band=11)
    sid = ds.test_sid
    rng = np.random.default_rng(0)
    true = (
        ds.labels.loc[ds.labels["fold"] == "test", "sdss_class"]
        .map({c: i for i, c in enumerate(CLS)})
        .to_numpy()
    )
    wrong = rng.random(len(sid)) < 0.15
    idx = np.where(wrong, (true + 1) % 3, true)
    for name in R.LGBM_MODELS:
        ds.set_lgbm(name, sid, _proba(idx))
    return run(ds, tmp_path_factory.mktemp("ident"))


@pytest.fixture(scope="module")
def short_result(tmp_path_factory: pytest.TempPathFactory) -> dict:
    """GALAXY has 2 objects per pixel (54 < 100); 27 test pixels (18 outside band A)."""
    ds = make_data(per_band=9, counts={"STAR": 5, "GALAXY": 2, "QSO": 5})
    return run(ds, tmp_path_factory.mktemp("short"))


def test_balanced_is_uniform_and_realistic_uses_the_grid(ident_result: dict) -> None:
    pri = ident_result["populations"]["a_all"]["priors"]
    assert pri["P0"]["regime"] == R.BALANCED
    assert pri["P0"]["prior"] == {c: pytest.approx(1 / 3) for c in CLS}
    for pid in ("P1", "P2", "P3"):
        assert pri[pid]["regime"] == R.REALISTIC
        assert pri[pid]["prior"] == R.GRID[pid]
        assert sum(R.GRID[pid].values()) == pytest.approx(1.0)
    # reweighting: the weighted confusion rows sum to the regime prior of the true class
    for pid, expected in (
        ("P0", {c: 1 / 3 for c in CLS}),
        ("P1", R.GRID["P1"]),
        ("P2", R.GRID["P2"]),
    ):
        d = _blk(ident_result, "a_all", pid)["models"]["lgbm_full"]["detail"]
        rows = np.asarray(d["confusion_weighted"]).sum(axis=1)
        np.testing.assert_allclose(rows, [expected[c] for c in CLS], rtol=1e-9)


def test_low_count_flag_only_in_realistic(ident_result: dict) -> None:
    p0 = _blk(ident_result, "a_all", "P0")["models"]["lgbm_full"]["detail"][
        "purity_flag"
    ]
    assert p0 == {c: None for c in CLS}
    # 15% errors over 165 objects per class: far more than 10 contaminants in every column
    p1 = _blk(ident_result, "a_all", "P1")["models"]["lgbm_full"]["detail"][
        "purity_flag"
    ]
    assert p1 == {c: None for c in CLS}


def test_low_count_flag_appears_in_realistic_with_few_contaminants(
    short_result: dict,
) -> None:
    # perfect predictions: 0 contaminants (< 10) in every column
    p1 = _blk(short_result, "a_all", "P1")["models"]["lgbm_full"]["detail"][
        "purity_flag"
    ]
    assert p1 == {c: "low-count" for c in CLS}
    p0 = _blk(short_result, "a_all", "P0")["models"]["lgbm_full"]["detail"][
        "purity_flag"
    ]
    assert all(v is None for v in p0.values())


def test_cells_above_the_floor_are_ok(ident_result: dict) -> None:
    cells = _blk(ident_result, "a_all", "P0")["models"]["lgbm_full"]["metrics"]
    assert cells["purity/STAR"]["status"] == "ok"
    assert cells["macro_f1"]["status"] == "ok"
    assert cells["purity/STAR"]["lo"] <= cells["purity/STAR"]["hi"]
    p3 = ident_result["populations"]["a_all"]["priors"]["P3"]
    assert p3["floor"]["STAR"]["pixels"] == 22  # bands B and C only


def test_below_floor_cell_is_marked_and_raw_value_is_flagged(
    short_result: dict, tmp_path: Path
) -> None:
    pri = short_result["populations"]["a_all"]["priors"]
    cells = pri["P0"]["variants"]["as_shipped"]["models"]["lgbm_full"]["metrics"]
    assert cells["purity/GALAXY"]["status"] == "n/a (floor)"
    assert cells["purity/GALAXY"]["lo"] is None and cells["purity/GALAXY"]["hi"] is None
    assert cells["purity/STAR"]["status"] == "ok"
    assert (
        cells["macro_f1"]["status"] == "n/a (floor)"
    )  # one class below floor taints macro
    assert pri["P0"]["floor"]["GALAXY"] == {"n": 54, "pixels": 27, "passes": False}
    # P3 keeps bands B and C only: 18 pixels < 20, so even STAR fails the floor there
    assert (
        pri["P2"]["variants"]["as_shipped"]["models"]["lgbm_full"]["metrics"][
            "purity/STAR"
        ]["status"]
        == "ok"
    )
    p3 = pri["P3"]
    assert (
        p3["n_rows"] == 12 * 18 and pri["P2"]["n_rows"] == 12 * 27
    )  # (5+2+5) per pixel
    assert (
        p3["variants"]["as_shipped"]["models"]["lgbm_full"]["metrics"]["purity/STAR"][
            "status"
        ]
        == "n/a (floor)"
    )
    text = R.render_markdown(R._clean(short_result))
    assert "n/a (floor) raw" in text
    assert "below-floor, not for claims" in text
    cell = R._fmt_cell(cells["purity/GALAXY"])
    assert cell.startswith("n/a (floor) raw ") and cell.endswith(
        "below-floor, not for claims"
    )


def test_low_count_in_s_sweep_is_realistic_only(short_result: dict) -> None:
    sweep = short_result["populations"]["a_all"]["s_sweep"]["lgbm_full|as_shipped"][
        "points"
    ]
    assert len(sweep) == len(R.S_QSO) * len(R.S_GALAXY)
    assert any(
        "low-count" in e["purity_flag"].values() for e in sweep if "purity_flag" in e
    )
    assert all(not e["floor_ok"]["GALAXY"] for e in sweep if "floor_ok" in e)


# -------------------------------------------------------------------------- 6) pairing


def test_run_boot_shares_replicates_across_models() -> None:
    rng = np.random.default_rng(1)
    n = 120
    pixel = np.repeat(np.arange(40), 3).astype(np.int64)
    band = np.where(pixel < 20, "A", "B")
    y = np.tile(np.asarray(CLS), n // 3)
    idx = np.array([CLS.index(c) for c in y])
    good = _proba(idx, hi=0.9)
    good[::5] = _proba(
        (idx[::5] + 1) % 3, hi=0.9
    )  # some errors so metrics are non-trivial
    shuffled = good[rng.permutation(n)]
    task = R.BootTask(
        "t",
        y,
        pixel,
        band,
        {"a": good, "b": good.copy(), "c": shuffled},
        {"a": None, "b": None, "c": None},
        ("P0",),
        30,
    )
    out = R.run_boot(task)
    ka, kb, kc = (R._key(m, "as_shipped", "P0") for m in "abc")
    for m in out["draws"][ka]:
        np.testing.assert_array_equal(out["draws"][ka][m], out["draws"][kb][m])
    assert out["point"][ka] == out["point"][kb]
    # shuffled model is worse on every replicate of macro_f1: paired difference is negative
    d = out["draws"][kc]["macro_f1"] - out["draws"][ka]["macro_f1"]
    assert np.isfinite(d).all() and (d < 0).all()
    assert out["point"][kc]["macro_f1"] < out["point"][ka]["macro_f1"]


def test_identical_models_have_exactly_zero_paired_difference(
    ident_result: dict,
) -> None:
    n_ok = 0
    for variant in R.VARIANTS:
        diffs = _blk(ident_result, "a_all", "P0", variant)["diffs"][
            "lgbm_full - lgbm_noind"
        ]
        for m, cell in diffs.items():
            assert cell["point"] == 0.0, m
            if cell["status"] == "ok":
                n_ok += 1
                assert (cell["lo"], cell["hi"]) == (0.0, 0.0), m
    assert n_ok > 0
    # also on the DSC-restricted cut and in the REALISTIC regime
    d = _blk(ident_result, "c_sdsc", "P1", "prior_matched")["diffs"][
        "lgbm_full - lgbm_noind"
    ]
    assert all(
        c["point"] == 0.0 or c["point"] != c["point"] for c in d.values()
    )  # 0 or undefined


@pytest.fixture(scope="module")
def shuffled_result(tmp_path_factory: pytest.TempPathFactory) -> dict:
    ds = make_data(per_band=11)
    sid = ds.test_sid
    true = (
        ds.labels.loc[ds.labels["fold"] == "test", "sdss_class"]
        .map({c: i for i, c in enumerate(CLS)})
        .to_numpy()
    )
    perm = np.random.default_rng(3).permutation(len(sid))
    ds.set_lgbm(
        "lgbm_full", sid, _proba(true)[perm]
    )  # scores unrelated to the true class
    return run(ds, tmp_path_factory.mktemp("shuf"))


def test_shuffled_model_has_negative_paired_difference(shuffled_result: dict) -> None:
    diffs = _blk(shuffled_result, "a_all", "P0")["diffs"]["lgbm_full - lgbm_noind"]
    cell = diffs["macro_f1"]
    assert cell["status"] == "ok"
    assert cell["point"] < -0.3
    assert cell["hi"] < 0  # interval excludes zero on the negative side
    assert diffs["f1/STAR"]["point"] < 0


# -------------------------------------------------------------------------- 7) strata


def test_strata_counts_add_up_to_the_total(missing_result: dict) -> None:
    st = missing_result["strata"]
    assert set(st) == {"3", "31", "95"}
    assert sum(v["n_rows"] for v in st.values()) == 36
    for c in CLS:
        assert sum(v["n_by_class"][c] for v in st.values()) == 12
    for v in st.values():
        assert sum(v["n_by_class"].values()) == v["n_rows"]
        n_cm = int(np.sum(v["models"]["lgbm_full|as_shipped"]["confusion_counts"]))
        assert n_cm == v["n_rows"]
    # DSC coverage of the strata adds up to the overall S_dsc
    assert sum(v["dsc"]["dsc_allosmod"]["n_covered"] for v in st.values()) == 23
    assert sum(v["dsc"]["dsc_combmod"]["n_covered"] for v in st.values()) == 32


def test_stratum_below_floor_is_n_a(missing_result: dict) -> None:
    for v in missing_result["strata"].values():  # 12 rows per stratum: far below 100
        assert (
            v["models"]["lgbm_full|as_shipped"]["macro_f1"]["status"] == "n/a (floor)"
        )
        assert v["floor"]["STAR"]["passes"] is False
        d = v["diffs"]["lgbm_full - lgbm_noind|as_shipped"]["macro_f1"]
        assert d["status"] == "n/a (floor)"


def test_stratum_with_a_missing_class_is_n_a_not_a_crash(tmp_path: Path) -> None:
    ds = make_data(per_band=2, counts={"STAR": 2, "GALAXY": 2, "QSO": 2})
    gal_ids = ds.labels.loc[ds.test_rows("GALAXY"), "source_id"]
    ds.features.loc[
        ds.features["source_id"].isin(gal_ids), "astrometric_params_solved"
    ] = 95
    res = run(ds, tmp_path)  # strata 3 and 31 hold no GALAXY at all
    for code in ("3", "31"):
        assert res["strata"][code]["n_by_class"]["GALAXY"] == 0
        assert (
            res["strata"][code]["models"]["lgbm_full|as_shipped"]["macro_f1"]["status"]
            == "n/a (floor)"
        )


def test_empty_stratum_is_n_a_not_a_crash(tmp_path: Path) -> None:
    res = run(
        make_data(
            per_band=2, counts={"STAR": 2, "GALAXY": 2, "QSO": 2}, solved="all95"
        ),
        tmp_path,
    )
    assert res["strata"]["3"]["n_rows"] == 0


# --------------------------------------------------------------- 8) text rigour, JSON


FORBIDDEN = ("superior", "melhor", "better", "worse", "outperform")


def test_markdown_is_provisional_and_free_of_ranking_words(
    ident_result: dict, tmp_path: Path
) -> None:
    paths = R.write_outputs(ident_result, tmp_path)
    text = paths[1].read_text(encoding="utf-8")
    assert text.lstrip("# ").startswith("PROVISIONAL")
    low = text.lower()
    for word in FORBIDDEN:
        assert word not in low, word
    assert not re.search(r"[A-Za-z]:[\\/]", text)  # no drive-letter path
    assert not re.search(r"/(Users|home)/", text)
    assert str(tmp_path) not in text
    assert "felip" not in low


def test_header_is_built_from_the_run_not_hardcoded() -> None:
    header = R.make_header(292556, 597)
    assert header.startswith("PROVISIONAL")
    assert "292556" in header and "597" in header
    assert "100k" not in header


def test_results_json_is_valid_strict_json(short_result: dict, tmp_path: Path) -> None:
    paths = R.write_outputs(short_result, tmp_path)
    raw = paths[0].read_text(encoding="utf-8")

    def reject(token: str) -> None:
        raise AssertionError(f"non-finite token {token} in results.json")

    data = json.loads(raw, parse_constant=reject)
    assert data["meta"]["header"].startswith("PROVISIONAL")
    assert "NaN" not in raw and "Infinity" not in raw
    # the runner encodes undefined values as null (not as NaN)
    assert R._clean(
        {"x": float("nan"), "y": np.float64("inf"), "z": [np.int64(2)]}
    ) == {
        "x": None,
        "y": None,
        "z": [2],
    }
    assert str(tmp_path) not in raw
    for p in paths:
        assert p.exists()


# ------------------------------------------------------------------------- 9) alignment


def _expect_error(ds: Dataset, tmp: Path, pattern: str) -> None:
    with pytest.raises(R.EvalError, match=pattern):
        R.load_test_set(ds.write(tmp))


def test_baseline_dataset_loads(tmp_path: Path) -> None:
    ds = make_data(per_band=2, counts={"STAR": 1, "GALAXY": 1, "QSO": 1})
    ts = R.load_test_set(ds.write(tmp_path))
    assert len(ts.y) == 18 and ts.n_val_test_predictions == 18 + 3


def test_prediction_row_order_does_not_matter(tmp_path: Path) -> None:
    ds = make_data(per_band=2, counts={"STAR": 1, "GALAXY": 1, "QSO": 1})
    ds.set_lgbm("lgbm_full", ds.test_sid[:1], _proba(np.array([CLS.index("QSO")])))
    base = R.load_test_set(ds.write(tmp_path / "a"))
    ds.preds["lgbm_full"] = (
        ds.preds["lgbm_full"].sample(frac=1.0, random_state=5).reset_index(drop=True)
    )
    shuf = R.load_test_set(ds.write(tmp_path / "b"))
    np.testing.assert_array_equal(
        base.models["lgbm_full"].proba, shuf.models["lgbm_full"].proba
    )


def test_feature_id_missing_from_features_raises(tmp_path: Path) -> None:
    ds = make_data(per_band=2, counts={"STAR": 1, "GALAXY": 1, "QSO": 1})
    ds.features = ds.features.iloc[1:].reset_index(drop=True)
    _expect_error(ds, tmp_path, "same source_id set")


def test_duplicate_id_in_labels_raises(tmp_path: Path) -> None:
    ds = make_data(per_band=2, counts={"STAR": 1, "GALAXY": 1, "QSO": 1})
    ds.labels = pd.concat([ds.labels, ds.labels.iloc[[0]]], ignore_index=True)
    _expect_error(ds, tmp_path, "not unique")


def test_duplicate_id_in_features_raises(tmp_path: Path) -> None:
    ds = make_data(per_band=2, counts={"STAR": 1, "GALAXY": 1, "QSO": 1})
    ds.features = pd.concat([ds.features, ds.features.iloc[[0]]], ignore_index=True)
    _expect_error(ds, tmp_path, "not unique")


@pytest.mark.parametrize("model", R.LGBM_MODELS)
def test_prediction_missing_a_test_id_raises(tmp_path: Path, model: str) -> None:
    ds = make_data(per_band=2, counts={"STAR": 1, "GALAXY": 1, "QSO": 1})
    ds.preds[model] = ds.preds[model][ds.preds[model]["source_id"] != ds.test_sid[0]]
    _expect_error(ds, tmp_path, "cover exactly val \\+ test")


def test_duplicate_prediction_id_raises(tmp_path: Path) -> None:
    ds = make_data(per_band=2, counts={"STAR": 1, "GALAXY": 1, "QSO": 1})
    df = ds.preds["lgbm_noind"]
    ds.preds["lgbm_noind"] = pd.concat([df, df.iloc[[0]]], ignore_index=True)
    _expect_error(ds, tmp_path, "cover exactly val \\+ test")


def test_extra_prediction_id_raises(tmp_path: Path) -> None:
    ds = make_data(per_band=2, counts={"STAR": 1, "GALAXY": 1, "QSO": 1})
    extra = ds.preds["lgbm_full"].iloc[[0]].copy()
    extra["source_id"] = extra["source_id"] + 1
    ds.preds["lgbm_full"] = pd.concat([ds.preds["lgbm_full"], extra], ignore_index=True)
    _expect_error(ds, tmp_path, "cover exactly val \\+ test")


def test_lgbm_probabilities_not_summing_to_one_raise(tmp_path: Path) -> None:
    ds = make_data(per_band=2, counts={"STAR": 1, "GALAXY": 1, "QSO": 1})
    ds.preds["lgbm_full"].loc[0, "p_cal_star"] = 0.1
    _expect_error(ds, tmp_path, "do not sum to 1")


def test_allosmod_not_summing_to_one_raises(tmp_path: Path) -> None:
    ds = make_data(per_band=2, counts={"STAR": 1, "GALAXY": 1, "QSO": 1})
    ds.labels.loc[ds.test_rows()[0], "classprob_dsc_allosmod_star"] = 0.0
    _expect_error(ds, tmp_path, "allosmod probabilities do not sum")


def test_healpix_column_disagreeing_with_source_id_raises(tmp_path: Path) -> None:
    ds = make_data(per_band=2, counts={"STAR": 1, "GALAXY": 1, "QSO": 1})
    ds.labels.loc[ds.test_rows()[0], "healpix_l5"] += 1
    _expect_error(ds, tmp_path, "healpix_l5")


def test_missing_input_file_raises_evalerror(tmp_path: Path) -> None:
    with pytest.raises(R.EvalError, match="labels.parquet"):
        R.load_test_set(tmp_path)


# ------------------------------------------------------------------- scripts/evaluate.py


def _load_script():
    spec = importlib.util.spec_from_file_location(
        "evaluate_script", REPO_ROOT / "scripts" / "evaluate.py"
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules["evaluate_script"] = mod
    spec.loader.exec_module(mod)
    return mod


def test_script_reports_failure_with_exit_code_1(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    mod = _load_script()
    assert (
        mod.main(["--outdir", str(tmp_path), "--n-boot", "20", "--workers", "1"]) == 1
    )
    assert "evaluation failed" in capsys.readouterr().err


def test_script_end_to_end_writes_the_four_outputs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    mod = _load_script()
    ds = make_data(per_band=2, counts={"STAR": 2, "GALAXY": 2, "QSO": 2})
    out = ds.write(tmp_path / "out")
    monkeypatch.setattr(
        mod, "REPO_ROOT", tmp_path
    )  # script prints paths relative to the repo
    assert mod.main(["--outdir", str(out), "--n-boot", "20", "--workers", "1"]) == 0
    lines = capsys.readouterr().out.splitlines()
    assert lines[:4] == [
        "out/eval/results.json",
        "out/eval/results.md",
        "out/eval/confusion_overall.csv",
        "out/eval/confusion_by_solved.csv",
    ]
    for rel in lines[:4]:
        assert (tmp_path / rel).stat().st_size > 0
    json.loads((tmp_path / lines[0]).read_text(encoding="utf-8"))
