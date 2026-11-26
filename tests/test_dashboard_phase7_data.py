"""Phase 7 data pieces: read_release, sky tables, scatter sample (tiny synthetic data, tmp_path)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from astro_triage.dashboard import data as D
from astro_triage.dashboard import data_galaxies_z as GZ
from astro_triage.dashboard import data_sky as SKY
from astro_triage.dashboard.data_galaxies import GALAXY_Z_EDGES
from astro_triage.dashboard.data_load import read_release

APP = Path(D.__file__).resolve().parent / "app.py"
M = ("lgbm_full", "lgbm_noind")


# ----------------------------------------------------------------- (b) read_release
def fake_release(root: Path, nan_score: bool = True, dup: bool = False) -> Path:
    root.mkdir(parents=True)
    ids = [1, 2, 3, 4, 5, 6]
    pd.DataFrame(
        {
            "source_id": ids,
            "sdss_class": ["star", "GALAXY", "QSO", "GALAXY", "star", "QSO"],
            "sdss_z": [0.0, 0.1, 1.5, 0.2, 0.0, 2.0],
            "healpix_l5": [10, 10, 10, 10, 11, 11],
            "fold": ["train", "train", "test", "test", "test", "val"],
            "ra": 1.0,
            "dec": 2.0,
            "l": 3.0,
            "b": 4.0,
            "classprob_dsc_allosmod_star": [0.1] * 6,
            "unused_extra": 9,
        }
    ).to_parquet(root / "labels.parquet")
    pd.DataFrame(
        {
            "source_id": ids,
            "phot_g_mean_mag": [15.0 + i for i in ids],
            "astrometric_params_solved": 31,
            "bp_rp": [0.5 + 0.25 * i for i in ids],
            "junk": 0,
        }
    ).to_parquet(root / "features.parquet")
    probs = {
        3: (0.1, 0.2, 0.7),
        4: (0.2, 0.7, 0.1),
        5: (0.8, 0.1, 0.1),
    }  # id 4 is an error
    for m in M:
        rows = {"source_id": [*probs, *([4] if dup else [])]}
        for j, c in enumerate(("star", "galaxy", "qso")):
            col = [p[j] for p in probs.values()] + ([0.0] if dup else [])
            rows[f"p_cal_{c}"], rows[f"p_raw_{c}"] = col, col
        df = pd.DataFrame(rows)
        if nan_score and m == "lgbm_full":
            df.loc[df["source_id"] == 5, "p_cal_star"] = np.nan
        df.to_parquet(root / f"predictions_{m}_v0.parquet")
    return root


def test_read_release_columns_counts_and_unscored(tmp_path: Path) -> None:
    rel = read_release(fake_release(tmp_path / "r"), "r", {"meta": {}}, {})
    assert rel.n_unscored == 1 and list(rel.test["source_id"]) == [3, 4]
    want = [
        "source_id",
        "sdss_class",
        "sdss_z",
        "healpix_l5",
        "fold",
        "ra",
        "dec",
        "l",
        "b",
    ]
    want += [
        "classprob_dsc_allosmod_star",
        "true",
        "phot_g_mean_mag",
        "astrometric_params_solved",
        "bp_rp",
    ]
    want += ["dsc_available"]
    for m in M:
        want += [f"p_{m}_{c}" for c in ("STAR", "GALAXY", "QSO")]
        want += [f"praw_{m}_{c}" for c in ("STAR", "GALAXY", "QSO")]
    want += [f"{k}_{m}" for m in M for k in ("pred", "conf", "err")]
    assert list(rel.test.columns) == want
    assert list(rel.test["true"]) == ["QSO", "GALAXY"]  # upper-cased
    assert list(rel.test["pred_lgbm_full"]) == ["QSO", "GALAXY"]
    assert list(rel.test["err_lgbm_full"]) == [False, False]
    assert list(rel.test["conf_lgbm_full"]) == [0.7, 0.7]
    assert list(rel.test["phot_g_mean_mag"]) == [
        18.0,
        19.0,
    ]  # aligned by id, not position
    assert not isinstance(rel.test["fold"].dtype, pd.CategoricalDtype)  # plain strings
    assert set(rel.test["fold"]) == {"test"}
    pc = rel.pixel_counts
    assert list(pc.columns) == ["healpix_l5", "fold", "n"]
    got = {(r.healpix_l5, r.fold): r.n for r in pc.itertuples()}
    assert got == {(10, "train"): 2, (10, "test"): 2, (11, "test"): 1, (11, "val"): 1}
    assert pc["n"].sum() == 6  # unscored rows still count on the map


def test_read_release_without_nan_scores_keeps_all(tmp_path: Path) -> None:
    rel = read_release(
        fake_release(tmp_path / "r", nan_score=False), "r", {"meta": {}}, {}
    )
    assert rel.n_unscored == 0 and list(rel.test["source_id"]) == [3, 4, 5]


def test_read_release_duplicate_source_id_raises(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="source_id repetido"):
        read_release(fake_release(tmp_path / "r", dup=True), "r", {"meta": {}}, {})


# ------------------------------------------------------------------------ (c) sky
def test_pixel_table_uses_pixel_counts_and_band_table_matches_direct() -> None:
    pc = pd.DataFrame(
        {
            "healpix_l5": [0, 0, 4000, 9000],
            "fold": ["train", "test", "test", "val"],
            "n": [3, 2, 7, 1],
        }
    )
    test = pd.DataFrame(
        {
            "healpix_l5": [0, 0, 4000, 4000, 4000, 9000],
            "err_m": [True, False, False, True, True, False],
        }
    )
    rel = D.Release("r", Path("."), {"meta": {}}, {}, pc, test, 0)
    px = SKY.pixel_table(rel)
    assert dict(zip(px["healpix_l5"], px["n"])) == {
        0: 5,
        4000: 7,
        9000: 1,
    }  # summed over folds
    assert len(px) == 3 and set(px["band"]) <= {"A", "B", "C"}
    st_all = SKY.test_pixel_stats(rel, "m", px)
    bt = SKY.lat_band_table(st_all)
    st = st_all.set_index("healpix_l5")
    assert dict(st["n"]) == {0: 2, 4000: 3, 9000: 1} and dict(st["errors"]) == {
        0: 1,
        4000: 2,
        9000: 0,
    }
    for row in bt.itertuples():  # direct computation from the objects
        pix = px.loc[px["band"] == row.band, "healpix_l5"]
        sub = test[test["healpix_l5"].isin(pix)]
        assert (row.n, row.errors, row.n_pixels) == (
            len(sub),
            int(sub["err_m"].sum()),
            len(set(sub["healpix_l5"])),
        )
        lo, hi = SKY.wilson(row.errors, row.n)
        assert (row.ci_lo, row.ci_hi) == pytest.approx((lo, hi), nan_ok=True)
    assert bt["n"].sum() == len(test) and bt["errors"].sum() == test["err_m"].sum()


# -------------------------------------------------------------- (g) scatter sample
def scatter_df(n: int = 12000, seed: int = 1) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    z = np.exp(rng.normal(-2.5, 1.2, n))
    z[:150] = rng.uniform(-0.05, GZ.Z_LOW, 150)  # z <= Z_LOW band, negatives included
    return pd.DataFrame(
        {
            "id": np.arange(n),
            "z": z,
            "G": rng.uniform(15, 21, n),
            "error": rng.random(n) < 0.05,
            "confidence": rng.random(n),
            "pred": rng.choice(["QSO", "STAR"], n),
        }
    )


def test_scatter_sample_is_deterministic_whole_and_capped() -> None:
    df = scatter_df()
    out = GZ.scatter_sample(df)
    pd.testing.assert_frame_equal(out, GZ.scatter_sample(df))
    assert not out["id"].equals(GZ.scatter_sample(df, seed=7)["id"])
    assert set(df.loc[df["error"], "id"]) <= set(out["id"])  # every error
    low = df[df["z"] <= GZ.Z_LOW]
    assert set(low["id"]) <= set(out["id"])  # the whole z <= Z_LOW band
    assert out["id"].is_unique
    right = out[~out["error"]]
    assert (
        len(right) <= GZ.MAX_SCATTER_POINTS and len(right) >= GZ.MAX_SCATTER_POINTS - 20
    )
    out_small = GZ.scatter_sample(df, max_points=100)  # limit below the low-z count
    assert len(out_small[~out_small["error"]]) == int((~low["error"]).sum())
    assert (
        len(GZ.scatter_sample(df.iloc[:200])) == 200
    )  # under the cap: nothing dropped


def test_scatter_sample_is_stratified_by_z_band() -> None:
    df = scatter_df()
    out = GZ.scatter_sample(df, max_points=3000)
    edges = (GZ.Z_LOW, *GALAXY_Z_EDGES)

    def counts(frame: pd.DataFrame) -> np.ndarray:
        z = frame.loc[~frame["error"] & (frame["z"] > GZ.Z_LOW), "z"]
        return np.bincount(
            np.searchsorted(edges, z, side="right"), minlength=len(edges) + 1
        )

    full, got = counts(df), counts(out)
    assert (full > 0).sum() > 2  # several bands populated
    budget = 3000 - int((~df["error"] & (df["z"] <= GZ.Z_LOW)).sum())
    assert np.all(np.abs(got - budget * full / full.sum()) <= 1.0 + 1e-9)


