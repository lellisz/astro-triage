"""dl_compare: threshold, weighted purity, low-count and delta (ADR 0006 sections 4, 5, 15).

Synthetic data only (tmp_path); test-fold rows are poisoned on purpose to prove val-only reads.
"""

from __future__ import annotations

import json
import math
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from astro_triage.evaluation import dl_compare as dc
from astro_triage.evaluation import dl_val as dv
from astro_triage.evaluation.runner import GRID, EvalError
from astro_triage.split import build_pixel_map

SHIFT = 2**49  # source_id // 2**35 // 4**7 == pixel
P2 = GRID["P2"]
ScoreFn = Callable[[str, int], np.ndarray]


# --------------------------------------------------------------------- pure helpers
def test_threshold_is_kth_largest_with_k_ceil_c_n():
    y = np.array(["QSO"] * 5 + ["STAR"] * 2)
    s = np.array([0.9, 0.2, 0.5, 0.7, 0.3, 0.99, 0.01])
    # QSO sorted desc: 0.9 0.7 0.5 0.3 0.2 ; n=5
    assert dc.threshold_at_completeness(y, s, "QSO", 0.8) == 0.3  # k = 4
    assert dc.threshold_at_completeness(y, s, "QSO", 0.81) == 0.2  # k = ceil(4.05) = 5
    assert dc.threshold_at_completeness(y, s, "QSO", 0.2) == 0.9  # k = 1
    assert dc.threshold_at_completeness(y, s, "QSO", 1.0) == 0.2  # k = n


def test_threshold_ignores_other_classes():
    y = np.array(["QSO", "QSO", "STAR"])
    s = np.array([0.5, 0.4, 0.99])
    assert dc.threshold_at_completeness(y, s, "QSO", 0.5) == 0.5


def test_threshold_float_noise_choice_ceil_round():
    """Coder choice 1: ceil(round(C*n, 9)). Note 0.9*30 == 27.0 exactly (no trap there)."""
    y = np.array(["QSO"] * 30)
    s = np.arange(30, 0, -1, dtype=float)  # 30 .. 1
    assert dc.threshold_at_completeness(y, s, "QSO", 0.9) == 4.0  # k = 27
    # a genuine float trap: 0.07 * 100 = 7.000000000000001 -> naive ceil gives 8
    assert math.ceil(0.07 * 100) == 8
    y100 = np.array(["QSO"] * 100)
    s100 = np.arange(100, 0, -1, dtype=float)
    assert dc.threshold_at_completeness(y100, s100, "QSO", 0.07) == 94.0  # k = 7


def test_threshold_ties_keep_realised_completeness_above_target():
    y = np.array(["QSO"] * 5)
    s = np.array([0.9, 0.8, 0.8, 0.8, 0.5])
    t = dc.threshold_at_completeness(y, s, "QSO", 0.6)  # k = 3
    assert t == 0.8
    realised = (s >= t).mean()
    assert realised == 0.8
    assert realised >= 0.6


@pytest.mark.parametrize("cstar", [0.8, 0.9])
@pytest.mark.parametrize("n", [1, 7, 30, 101])
def test_threshold_val_completeness_at_least_target(cstar, n):
    rng = np.random.default_rng(n)
    s = np.round(rng.random(n), 1)  # heavy ties
    y = np.array(["GALAXY"] * n)
    t = dc.threshold_at_completeness(y, s, "GALAXY", cstar)
    assert (s >= t).sum() / n >= cstar - 1e-12
    # minimal: the next distinct higher threshold would fall below C*
    higher = s[s > t]
    if higher.size:
        assert (s >= higher.min()).sum() / n < cstar + 1e-12


def test_threshold_errors():
    y = np.array(["QSO", "STAR"])
    s = np.array([0.5, 0.5])
    with pytest.raises(EvalError):
        dc.threshold_at_completeness(y, s, "GALAXY", 0.8)  # no rows
    with pytest.raises(EvalError):
        dc.threshold_at_completeness(y, np.array([np.nan, 0.1]), "QSO", 0.8)
    with pytest.raises(EvalError):
        dc.threshold_at_completeness(y, s, "QSO", 0.0)  # k = 0
    with pytest.raises(EvalError):
        dc.threshold_at_completeness(y, s, "QSO", 1.5)  # k > n


def test_weighted_purity_equals_adr0003_case_7():
    y = np.array(["STAR"] * 1000 + ["QSO"] * 1000)
    score = np.concatenate([[1.0] * 10, [0.0] * 990, [1.0] * 950, [0.0] * 50])
    got = dc.weighted_purity(y, score, "QSO", 0.5, {"STAR": 0.998, "QSO": 0.002})
    assert got == pytest.approx(0.1599, abs=1e-4)


def test_weighted_purity_p2_hand_value():
    # 4 QSO, 6 STAR, 5 GALAXY; selected: 3 QSO, 2 STAR, 1 GALAXY (score >= 0.5)
    y = np.array(["QSO"] * 4 + ["STAR"] * 6 + ["GALAXY"] * 5)
    score = np.array([0.9, 0.8, 0.7, 0.1] + [0.6, 0.5] + [0.0] * 4 + [0.5] + [0.0] * 4)
    pq_, ps, pg = 0.000989, 0.998813, 0.000198  # ADR 0003 D4, P2
    num = 3 * pq_ / 4
    expected = num / (num + 2 * ps / 6 + 1 * pg / 5)
    got = dc.weighted_purity(y, score, "QSO", 0.5, P2)
    assert got == pytest.approx(expected, rel=1e-5)
    assert 0.0 < got < 0.01  # star-dominated prior collapses QSO purity


def test_weighted_purity_nan_when_nothing_selected():
    y = np.array(["QSO", "STAR", "GALAXY"])
    assert math.isnan(dc.weighted_purity(y, np.array([0.1, 0.2, 0.3]), "QSO", 0.9, P2))


def test_contaminant_counts_raw_and_exclude_own_class():
    y = np.array(["QSO", "STAR", "STAR", "GALAXY", "QSO"])
    s = np.array([0.9, 0.9, 0.2, 0.9, 0.9])
    assert dc.contaminant_counts(y, s, "QSO", 0.5) == {"STAR": 1, "GALAXY": 1}


def test_is_low_count_total_reading():
    assert dc.is_low_count({"STAR": 4, "QSO": 5}) is True
    assert dc.is_low_count({"STAR": 0, "QSO": 12}) is False
    assert dc.is_low_count({"STAR": 5, "QSO": 5}) is False  # total 10 is not < 10
    assert dc.is_low_count({"STAR": 9, "QSO": 0}) is True
    assert dc.is_low_count({"STAR": 9, "GALAXY": 9}) is False  # each < 10, total 18


def test_block_ids_adr_hand_case():
    assert dc.block_ids(np.array([7207], dtype=np.int64))[0] == 450
    assert list(dc.block_ids(np.array([0, 15, 16, 31, 32]))) == [0, 0, 1, 1, 2]


# --------------------------------------------------------------------- synthetic release
def _pixels() -> pd.DataFrame:
    return build_pixel_map()


def val_pixels(per_band: int = 40) -> dict[str, list[int]]:
    """Val pixels per band, one per distinct pixel//16 group (so blocks == pixels)."""
    pm = _pixels()
    out: dict[str, list[int]] = {}
    for band in "ABC":
        sub = pm[(pm.fold == "val") & (pm.lat_band == band)]
        seen: set[int] = set()
        pix: list[int] = []
        for p in sub.healpix_l5.astype(int):
            if p // 16 not in seen:
                seen.add(p // 16)
                pix.append(p)
            if len(pix) == per_band:
                break
        out[band] = pix
    return out


def band_of(pixel: np.ndarray) -> np.ndarray:
    return _pixels()["lat_band"].to_numpy()[pixel].astype(str)


def make_release(
    root: Path,
    pix: np.ndarray,
    cls: np.ndarray,
    scores: ScoreFn,
    variant: str = "nan",
    n_test: int = 30,
    drop_val_id_from: tuple[str, int] | None = None,
) -> tuple[Path, Path, np.ndarray]:
    """labels.parquet + lgbm_seeds/predictions_lgbm_<fs>_s<k>_v0.parquet.

    ``scores(fs, k)`` gives a (n_val, 3) array in STAR, GALAXY, QSO order. Test rows
    (other-fold pixels) are appended to both files; ``variant`` controls their poison.
    """
    pm = _pixels()
    n = pix.size
    val_ids = pix.astype(np.int64) * SHIFT + np.arange(1, n + 1)
    test_pix = pm[pm.fold == "test"].healpix_l5.to_numpy(dtype=np.int64)[:n_test]
    test_ids = test_pix * SHIFT + 7
    rng = np.random.default_rng(1)
    if variant == "nan":
        t_cls = np.array(["STAR", "QSO", "GALAXY"] * n_test)[:n_test]
        t_pix = test_pix
        t_ids = test_ids
    else:  # garbage: unknown classes, wrong pixel, duplicated ids
        t_cls = np.array(["???"] * n_test)
        t_pix = test_pix + 1
        t_ids = np.full(n_test, test_ids[0])
    labels = pd.DataFrame(
        {
            "source_id": np.concatenate([val_ids, t_ids]),
            "sdss_class": np.concatenate([cls, t_cls]),
            "healpix_l5": np.concatenate([pix, t_pix]).astype(np.int64),
            "fold": ["val"] * n + ["test"] * n_test,
        }
    )
    labels = labels.sample(frac=1.0, random_state=3)  # file order != val order
    root.mkdir(parents=True, exist_ok=True)
    pq.write_table(
        pa.Table.from_pandas(labels, preserve_index=False), root / "labels.parquet"
    )
    sd = root / "lgbm_seeds"
    sd.mkdir(exist_ok=True)
    for fs in dc.FEATURE_SETS:
        for k in range(dc.N_SEEDS):
            sc = scores(fs, k)
            if variant == "nan":
                tp = np.full((n_test, 3), np.nan)
            else:
                tp = rng.choice([-5.0, 5.0, np.inf, 0.5], size=(n_test, 3))
            ids = np.concatenate([val_ids, test_ids])  # unique even in "garbage"
            allp = np.vstack([sc, tp])
            df = pd.DataFrame(
                {
                    "source_id": ids,
                    "p_cal_star": allp[:, 0],
                    "p_cal_galaxy": allp[:, 1],
                    "p_cal_qso": allp[:, 2],
                }
            )
            if drop_val_id_from == (fs, k):
                df = df[df.source_id != val_ids[0]]
            df = df.iloc[::-1]
            pq.write_table(
                pa.Table.from_pandas(df, preserve_index=False),
                sd / f"predictions_lgbm_{fs}_s{k}_v0.parquet",
            )
    return root, sd, val_ids


def random_data(per_band: int = 40) -> tuple[np.ndarray, np.ndarray, ScoreFn]:
    vp = val_pixels(per_band)
    allp = np.array(vp["A"] + vp["B"] + vp["C"])
    pix = np.tile(allp, 5)  # 600 rows
    cls = np.array((["STAR"] * 3 + ["QSO"] * 2 + ["GALAXY"] * 2) * 100)[: pix.size]
    pix = pix[: cls.size]

    def scores(fs: str, k: int) -> np.ndarray:
        rng = np.random.default_rng(100 * k + (fs == "noind"))
        out = rng.random((cls.size, 3)) * 0.5
        for j, c in enumerate(("STAR", "GALAXY", "QSO")):
            out[cls == c, j] += rng.random((cls == c).sum()) * 0.6
        return out

    return pix, cls, scores


def ref_cell(y, band, sc_by_seed, cls, cstar, prior, pop):
    """Independent re-derivation of one cell (numpy only)."""
    j = {"GALAXY": 1, "QSO": 2}[cls]
    v, thr = [], []
    for sc in sc_by_seed:
        s = sc[:, j]
        srt = np.sort(s[y == cls])[::-1]
        k = math.ceil(round(cstar * srt.size, 9))
        t = srt[k - 1]
        thr.append(t)
        yy, ss = y[pop], s[pop]
        n_c = {c: (yy == c).sum() for c in ("STAR", "GALAXY", "QSO")}
        w = np.array([prior[c] / n_c[c] for c in yy])
        sel = ss >= t
        v.append(w[sel & (yy == cls)].sum() / w[sel].sum())
    return v, thr


@pytest.fixture(scope="module")
def release(tmp_path_factory):
    pix, cls, scores = random_data()
    root = tmp_path_factory.mktemp("rel")
    out, sd, ids = make_release(root / "r", pix, cls, scores)
    return pix, cls, scores, out, sd, ids


# --------------------------------------------------------------------- compute_delta
def test_delta_matches_independent_reference_for_all_cells(release):
    pix, cls, scores, out, sd, _ = release
    doc = dv.compute_delta(out, sd)
    band = band_of(pix)
    for fs in ("full", "noind"):
        sc = [scores(fs, k) for k in range(5)]
        for c in ("QSO", "GALAXY"):
            for cstar in (0.8, 0.9):
                for p in ("P1", "P2", "P3"):
                    pop = (
                        np.isin(band, ("B", "C"))
                        if p == "P3"
                        else np.ones(cls.size, bool)
                    )
                    v, thr = ref_cell(cls, band, sc, c, cstar, GRID[p], pop)
                    cell = doc["delta"][fs][c][f"{cstar:.2f}"][p]
                    assert cell["v_k"] == pytest.approx(v, rel=1e-9)
                    assert cell["thresholds"] == pytest.approx(thr, rel=1e-12)
                    assert cell["delta"] == pytest.approx(
                        2 * np.std(v, ddof=1), rel=1e-9
                    )
                    assert cell["v_mean"] == pytest.approx(np.mean(v), rel=1e-9)


def test_delta_schema_nesting_and_sha(release):
    _pix, cls, _, out, sd, _ids = release
    doc = dv.compute_delta(out, sd)
    assert doc["schema_name"] == "astro_triage.lgbm_seeds_delta"
    assert (doc["fold"], doc["n_seeds"], doc["delta_ddof"]) == ("val", 5, 1)
    assert doc["delta_sd_multiplier"] == 2.0
    assert doc["primary_prior"] == "P2" and doc["primary_completeness"] == 0.80
    assert doc["n_val_rows"] == cls.size
    assert doc["n_val_by_class"] == {
        c: int((cls == c).sum()) for c in ("STAR", "GALAXY", "QSO")
    }
    assert set(doc["priors"]) == {"P1", "P2", "P3"}
    assert set(doc["delta"]) == {"full", "noind"}
    for fs in doc["delta"]:
        assert set(doc["delta"][fs]) == {"QSO", "GALAXY"}
        assert set(doc["delta"][fs]["QSO"]) == {"0.80", "0.90"}
        assert set(doc["delta"][fs]["QSO"]["0.80"]) == {"P1", "P2", "P3"}
        assert set(doc["prediction_sha256"][fs]) == {
            f"lgbm_{fs}_s{k}" for k in range(5)
        }
        assert all(len(h) == 64 for h in doc["prediction_sha256"][fs].values())
    assert len(doc["labels_sha256"]) == 64
    cell = doc["delta"]["full"]["QSO"]["0.80"]["P2"]
    assert set(cell) == {
        "delta", "v_k", "v_mean", "thresholds", "contaminant_counts", "low_count",
        "floor", "blocks", "val_checks_pass", "val_completeness",
    }  # fmt: skip
    assert len(cell["v_k"]) == len(cell["thresholds"]) == len(cell["low_count"]) == 5
    assert all(c >= 0.8 for c in cell["val_completeness"])
    json.dumps(doc, allow_nan=False)  # fully serialisable


def test_delta_hand_values_two_sd_ddof1(tmp_path):
    """5 QSO, 5 GALAXY, 5 STAR; seed k puts k STAR above the QSO threshold -> v_k by hand."""
    vp = val_pixels()
    pix = np.array(vp["B"][:15])
    cls = np.array(["QSO"] * 5 + ["GALAXY"] * 5 + ["STAR"] * 5)

    def scores(fs: str, k: int) -> np.ndarray:
        sc = np.full((15, 3), 0.1)
        sc[:5, 2] = [0.9, 0.8, 0.7, 0.6, 0.5]  # QSO: k = ceil(0.8*5) = 4 -> t = 0.6
        sc[:5, 1] = sc[5:10, 1] = np.linspace(0.9, 0.5, 5)
        sc[10 : 10 + k, 2] = 0.95  # k contaminating stars
        return sc

    out, sd, _ = make_release(tmp_path / "h", pix, cls, scores)
    doc = dv.compute_delta(out, sd)
    for p in ("P1", "P2", "P3"):
        pq_, ps = GRID[p]["QSO"], GRID[p]["STAR"]
        v = [4 * pq_ / (4 * pq_ + k * ps) for k in range(5)]
        cell = doc["delta"]["full"]["QSO"]["0.80"][p]
        assert cell["thresholds"] == pytest.approx([0.6] * 5)
        assert cell["v_k"] == pytest.approx(v, rel=1e-9)
        assert cell["delta"] == pytest.approx(2 * np.std(v, ddof=1), rel=1e-9)
        assert cell["delta"] != pytest.approx(2 * np.std(v, ddof=0), rel=1e-3)
        assert cell["contaminant_counts"][3] == {"STAR": 3, "GALAXY": 0}
        assert cell["low_count"] == [True] * 5
        assert cell["val_checks_pass"] is False  # floor fails, 15 rows


def test_p3_population_is_bands_b_and_c_threshold_from_all_val(release):
    pix, cls, scores, out, sd, _ = release
    doc = dv.compute_delta(out, sd)
    band = band_of(pix)
    bc = np.isin(band, ("B", "C"))
    p3 = doc["delta"]["full"]["QSO"]["0.80"]["P3"]
    p2 = doc["delta"]["full"]["QSO"]["0.80"]["P2"]
    assert p3["thresholds"] == p2["thresholds"]  # fitted on all val
    assert p3["floor"]["n"] == int(((cls == "QSO") & bc).sum())
    assert p2["floor"]["n"] == int((cls == "QSO").sum())
    assert p3["floor"]["n"] < p2["floor"]["n"]
    n_c = sum(p3["contaminant_counts"][0].values())
    sc0 = scores("full", 0)[:, 2]
    assert n_c == int(((sc0 >= p3["thresholds"][0]) & (cls != "QSO") & bc).sum())


def test_block_floor_is_distinct_pixel16_band_pairs(tmp_path):
    """Pixels 66 (band C) and 71 (band B) share pixel//16 == 4: two blocks, not one."""
    pm = _pixels().set_index("healpix_l5")
    assert (pm.loc[66, "fold"], pm.loc[71, "fold"]) == ("val", "val")
    assert (pm.loc[66, "lat_band"], pm.loc[71, "lat_band"]) == ("C", "B")
    vp = val_pixels()
    qso_pix = [66, 71, 66, 71, 66, 71]
    # GALAXY in band B too, so P3 (bands B+C) still has every class
    pix = np.array(qso_pix + vp["B"][:5] + vp["B"][5:10])
    cls = np.array(["QSO"] * 6 + ["STAR"] * 5 + ["GALAXY"] * 5)
    out, sd, _ = make_release(
        tmp_path / "b", pix, cls, lambda fs, k: np.full((16, 3), 0.5)
    )
    cell = dv.compute_delta(out, sd)["delta"]["full"]["QSO"]["0.80"]["P2"]
    assert cell["blocks"] == 2
    assert cell["floor"]["pixels"] == 2


# --------------------------------------------------------------------- val-only guard
def _strip(doc: dict) -> dict:
    # File hashes identify the input files (which differ in their test rows by
    # construction); every other key must be identical.
    return {
        k: v for k, v in doc.items() if k not in {"prediction_sha256", "labels_sha256"}
    }


def _close(a: object, b: object) -> bool:
    if isinstance(a, dict) and isinstance(b, dict):
        return a.keys() == b.keys() and all(_close(a[k], b[k]) for k in a)
    if isinstance(a, list) and isinstance(b, list):
        return len(a) == len(b) and all(_close(x, y) for x, y in zip(a, b, strict=True))
    if isinstance(a, float) and isinstance(b, float):
        return a == pytest.approx(b, rel=1e-12, abs=1e-15)
    return a == b


def test_test_rows_never_affect_output(tmp_path):
    pix, cls, scores = random_data(per_band=25)
    o1, s1, _ = make_release(tmp_path / "n", pix, cls, scores, variant="nan", n_test=30)
    o2, s2, _ = make_release(
        tmp_path / "g", pix, cls, scores, variant="garbage", n_test=55
    )
    d1, d2 = dv.compute_delta(o1, s1), dv.compute_delta(o2, s2)
    # labels.parquet row order differs between the two files, which moves the last ulp
    assert _close(_strip(d1), _strip(d2))
    # sanity: the poison is really on disk
    garbage = pq.read_table(s2 / "predictions_lgbm_full_s0_v0.parquet").to_pandas()
    assert (~np.isfinite(garbage.p_cal_qso)).any() or (
        garbage.p_cal_qso.abs() > 1
    ).any()
    lab = pq.read_table(o2 / "labels.parquet").to_pandas()
    assert (lab.sdss_class == "???").any() and not lab.source_id.is_unique


def test_labels_read_with_val_filter_and_never_whole_file(tmp_path, monkeypatch):
    pix, cls, scores = random_data(per_band=25)
    out, sd, _ = make_release(tmp_path / "s", pix, cls, scores)
    calls: list[tuple[str, object, object]] = []
    real = pq.read_table

    def spy(path, *a, **kw):
        calls.append((Path(path).name, kw.get("filters"), kw.get("columns")))
        return real(path, *a, **kw)

    monkeypatch.setattr(dv.pq, "read_table", spy)
    dv.compute_delta(out, sd)
    label_calls = [c for c in calls if c[0] == "labels.parquet"]
    assert len(label_calls) == 1
    assert label_calls[0][1] == [("fold", "==", "val")]
    assert "fold" not in label_calls[0][2]
    assert all(c[0] != "labels.parquet" or c[1] for c in calls)


def test_missing_val_id_in_predictions_raises(tmp_path):
    pix, cls, scores = random_data(per_band=25)
    out, sd, _ = make_release(
        tmp_path / "m", pix, cls, scores, drop_val_id_from=("noind", 3)
    )
    with pytest.raises(EvalError, match="1 val ids have no prediction"):
        dv.compute_delta(out, sd)


def test_missing_seed_file_and_nonfinite_val_score_raise(tmp_path):
    pix, cls, scores = random_data(per_band=25)
    out, sd, _ids = make_release(tmp_path / "f", pix, cls, scores)
    (sd / "predictions_lgbm_full_s2_v0.parquet").unlink()
    with pytest.raises(EvalError, match="missing seed predictions"):
        dv.compute_delta(out, sd)

    def nan_scores(fs: str, k: int) -> np.ndarray:
        sc = scores(fs, k)
        if (fs, k) == ("full", 0):
            sc[0, 2] = np.nan  # a VAL row
        return sc

    out2, sd2, _ = make_release(tmp_path / "f2", pix, cls, nan_scores)
    with pytest.raises(EvalError, match="non-finite val p_cal"):
        dv.compute_delta(out2, sd2)


def test_labels_inconsistent_pixel_or_missing_file_raise(tmp_path):
    pix, cls, scores = random_data(per_band=25)
    out, sd, _ = make_release(tmp_path / "p", pix, cls, scores)
    lab = pq.read_table(out / "labels.parquet").to_pandas()
    lab.loc[lab.fold == "val", "healpix_l5"] += 1
    pq.write_table(
        pa.Table.from_pandas(lab, preserve_index=False), out / "labels.parquet"
    )
    with pytest.raises(EvalError, match="healpix_l5"):
        dv.compute_delta(out, sd)
    (out / "labels.parquet").unlink()
    with pytest.raises(EvalError, match="cannot read val labels"):
        dv.compute_delta(out, sd)


# --------------------------------------------------------------------- reproducibility
def _reshuffle(root: Path, sd: Path, seed: int, labels: bool, preds: bool) -> None:
    """Rewrite the parquet files with rows in another (seeded) order, same content."""
    files = []
    if labels:
        files.append(root / "labels.parquet")
    if preds:
        files += sorted(sd.glob("predictions_lgbm_*_v0.parquet"))
    for i, f in enumerate(files):
        df = pq.read_table(f).to_pandas()
        df = df.sample(frac=1.0, random_state=seed + i)
        pq.write_table(pa.Table.from_pandas(df, preserve_index=False), f)


def _twin_docs(tmp_path: Path, labels: bool, preds: bool) -> tuple[dict, dict]:
    pix, cls, scores = random_data(per_band=25)
    o1, s1, _ = make_release(tmp_path / "a", pix, cls, scores)
    o2, s2, _ = make_release(tmp_path / "b", pix, cls, scores)
    _reshuffle(o1, s1, 1000, labels, preds)
    _reshuffle(o2, s2, 2000, labels, preds)
    return dv.compute_delta(o1, s1), dv.compute_delta(o2, s2)


def test_delta_bit_identical_when_labels_row_order_differs(tmp_path):
    d1, d2 = _twin_docs(tmp_path, labels=True, preds=False)
    assert _strip(d1) == _strip(d2)  # exact, no approx
    assert d1["labels_sha256"] != d2["labels_sha256"]  # the shuffle is really on disk


def test_delta_bit_identical_when_prediction_row_order_differs(tmp_path):
    d1, d2 = _twin_docs(tmp_path, labels=False, preds=True)
    assert _strip(d1) == _strip(d2)
    assert d1["prediction_sha256"] != d2["prediction_sha256"]


def test_reproducibility_tests_detect_missing_sort(tmp_path, monkeypatch):
    """Guard against vacuity: with sort_values neutralised the exact check must fail."""
    monkeypatch.setattr(
        pd.DataFrame, "sort_values", lambda self, *a, **kw: self, raising=True
    )
    d1, d2 = _twin_docs(tmp_path, labels=True, preds=False)
    assert _strip(d1) != _strip(d2)
