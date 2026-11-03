"""Tests for the HEALPix level-5 pixel split (ADR 0003 D3; schema invariants S1-S4, S7)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy_healpix import HEALPix

from astro_triage.split import (
    N_PIXELS,
    assign_folds,
    build_pixel_map,
    pixel_l5,
)

CSV_PATH = Path(__file__).resolve().parents[1] / "manifests" / "split_pixel_map_v1.csv"
HP5 = 2**35 * 4**7  # source_id span of one level-5 pixel
CUTS = (11.54, 30.0)


@pytest.fixture(scope="module")
def csv_map() -> pd.DataFrame:
    return pd.read_csv(CSV_PATH, dtype={"lat_band": str, "fold": str})


@pytest.fixture(scope="module")
def built_map() -> pd.DataFrame:
    return build_pixel_map()


def test_s1_shape_columns_and_key(csv_map: pd.DataFrame) -> None:
    assert list(csv_map.columns) == ["healpix_l5", "lat_band", "fold"]
    assert len(csv_map) == 12_288 == N_PIXELS
    assert csv_map["healpix_l5"].is_unique
    assert csv_map["healpix_l5"].tolist() == list(range(12_288))
    assert not csv_map.isna().any().any()


def test_one_fold_per_pixel_and_domains(csv_map: pd.DataFrame) -> None:
    assert set(csv_map["fold"]) == {"train", "val", "test"}
    assert set(csv_map["lat_band"]) == {"A", "B", "C"}
    # S7: no pixel in two folds (also checked via groupby, independent of key uniqueness).
    assert (csv_map.groupby("healpix_l5")["fold"].nunique() == 1).all()


def test_s3_counts_follow_floor_formulas(csv_map: pd.DataFrame) -> None:
    for band, grp in csv_map.groupby("lat_band"):
        n = len(grp)
        counts = grp["fold"].value_counts()
        n_train = n * 70 // 100
        n_val = n * 15 // 100
        assert counts["train"] == n_train, band
        assert counts["val"] == n_val, band
        assert counts["test"] == n - n_train - n_val, band
        # Floor-derived shares stay within +-1 of the real-valued shares for train and val.
        # (Test is the remainder and can exceed +-1; see report on S3 vs ADR 0003.)
        assert abs(counts["train"] - 0.70 * n) <= 1
        assert abs(counts["val"] - 0.15 * n) <= 1


def test_s2_rebuild_equals_versioned_csv(
    csv_map: pd.DataFrame, built_map: pd.DataFrame
) -> None:
    got = built_map.astype({"healpix_l5": "int64"})
    pd.testing.assert_frame_equal(
        got, csv_map.astype({"healpix_l5": "int64"}), check_dtype=False
    )
    again = build_pixel_map()
    pd.testing.assert_frame_equal(built_map, again)


@pytest.mark.parametrize(
    ("source_id", "expected"),
    [
        (0, 0),
        (HP5 - 1, 0),
        (HP5, 1),
        (HP5 * 12_287, 12_287),
        (HP5 * 12_288 - 1, 12_287),
    ],
)
def test_pixel_l5_edges_scalar_and_array(source_id: int, expected: int) -> None:
    assert pixel_l5(source_id) == expected
    out = pixel_l5(np.array([source_id], dtype=np.int64))
    assert isinstance(out, np.ndarray)
    assert out.tolist() == [expected]


def test_pixel_l5_matches_astropy_healpix_on_random_sample() -> None:
    rng = np.random.default_rng(7)
    pix5 = rng.integers(0, N_PIXELS, size=1000)
    # A level-12 child inside each level-5 pixel (nested: children are contiguous).
    pix12 = pix5 * 4**7 + rng.integers(0, 4**7, size=1000)
    offset = rng.integers(0, 2**35, size=1000)
    source_id = (pix12 * 2**35 + offset).astype(np.int64)

    hp12 = HEALPix(nside=2**12, order="nested")
    hp5 = HEALPix(nside=32, order="nested")
    lon, lat = hp12.healpix_to_lonlat(pix12)
    expected = hp5.lonlat_to_healpix(lon, lat)

    np.testing.assert_array_equal(pixel_l5(source_id), expected)
    np.testing.assert_array_equal(pixel_l5(source_id), pix5)


def test_assign_folds_matches_map_and_is_deterministic(csv_map: pd.DataFrame) -> None:
    rng = np.random.default_rng(11)
    pix = rng.integers(0, N_PIXELS, size=2000)
    sid = (pix * HP5 + rng.integers(0, HP5, size=2000)).astype(np.int64)
    got = assign_folds(sid)
    expected = csv_map["fold"].to_numpy(dtype=str)[pix]
    np.testing.assert_array_equal(got, expected)
    np.testing.assert_array_equal(assign_folds(sid), got)
    # Same pixel, different offsets -> same fold.
    sid2 = (pix * HP5 + 1).astype(np.int64)
    np.testing.assert_array_equal(assign_folds(sid2), got)


def test_lat_band_matches_independent_recomputation(csv_map: pd.DataFrame) -> None:
    rng = np.random.default_rng(3)
    pix = rng.choice(N_PIXELS, size=500, replace=False)
    hp = HEALPix(nside=32, order="nested")
    lon, lat = hp.healpix_to_lonlat(pix)
    gal = SkyCoord(ra=lon.to(u.deg), dec=lat.to(u.deg), frame="icrs").galactic
    abs_b = np.abs(gal.b.deg)
    near_cut = np.minimum(np.abs(abs_b - CUTS[0]), np.abs(abs_b - CUTS[1])) < 1e-6
    keep = ~near_cut
    expected = np.where(abs_b < CUTS[0], "A", np.where(abs_b < CUTS[1], "B", "C"))
    got = csv_map["lat_band"].to_numpy(dtype=str)[pix]
    np.testing.assert_array_equal(got[keep], expected[keep])


def _independent_folds(shared_rng: bool = False) -> pd.DataFrame:
    """Rebuild band and fold per pixel from ADR 0003 rule 2, without build_pixel_map."""
    hp = HEALPix(nside=32, order="nested")
    lon, lat = hp.healpix_to_lonlat(np.arange(12_288))
    icrs = SkyCoord(ra=lon.to(u.deg), dec=lat.to(u.deg), frame="icrs")
    abs_b = np.abs(icrs.galactic.b.deg)
    band = np.where(abs_b < CUTS[0], "A", np.where(abs_b < CUTS[1], "B", "C"))
    fold = np.empty(12_288, dtype=object)
    shared = np.random.default_rng(20261025)
    for label in ("A", "B", "C"):
        rng = shared if shared_rng else np.random.default_rng(20261025)
        pixels = np.sort(np.flatnonzero(band == label))
        order = rng.permutation(pixels)
        n = len(pixels)
        n_train, n_val = n * 70 // 100, n * 15 // 100
        fold[order[:n_train]] = "train"
        fold[order[n_train : n_train + n_val]] = "val"
        fold[order[n_train + n_val :]] = "test"
    return pd.DataFrame(
        {
            "healpix_l5": np.arange(12_288),
            "lat_band": band.astype(str),
            "fold": fold.astype(str),
        }
    )


def test_s2_independent_reconstruction_matches_csv(csv_map: pd.DataFrame) -> None:
    got = _independent_folds()
    # Pixels whose centre sits within 1e-6 deg of a band edge could flip with float noise.
    pd.testing.assert_frame_equal(
        got, csv_map.astype({"healpix_l5": "int64"}), check_dtype=False
    )


# ----------------------------------------------------------- input validation (pixel_l5)


def test_pixel_l5_negative_source_id_raises_with_count() -> None:
    ids = np.array([HP5, -1, 5, -7], dtype=np.int64)
    with pytest.raises(ValueError, match=r"2 of 4"):
        pixel_l5(ids)
    with pytest.raises(ValueError, match=r"1 of 1"):
        pixel_l5(-1)


def test_pixel_l5_pixel_above_range_raises_with_count() -> None:
    ids = np.array([HP5 * 12_288, HP5 * 12_287, HP5 * 12_300], dtype=np.int64)
    with pytest.raises(ValueError, match=r"2 of 3"):
        pixel_l5(ids)
    with pytest.raises(ValueError, match="12287"):
        pixel_l5(HP5 * 12_288)


def test_pixel_l5_accepts_series_same_as_array() -> None:
    arr = np.array([0, HP5, HP5 * 12_287 + 5, HP5 * 3 + 9], dtype=np.int64)
    out = pixel_l5(pd.Series(arr))
    assert isinstance(out, np.ndarray)
    np.testing.assert_array_equal(out, pixel_l5(arr))
    np.testing.assert_array_equal(assign_folds(pd.Series(arr)), assign_folds(arr))


@pytest.mark.parametrize("bad", [np.array([1.0, 2.0]), pd.Series([1.5, 2.5]), 3.0])
def test_pixel_l5_float_dtype_raises(bad: object) -> None:
    with pytest.raises(ValueError, match="integer"):
        pixel_l5(bad)  # type: ignore[call-overload]


def test_pixel_l5_scalar_returns_int() -> None:
    out = pixel_l5(HP5 * 7 + 3)
    assert type(out) is int and out == 7


def test_assign_folds_propagates_validation_errors() -> None:
    with pytest.raises(ValueError, match="1 of 2"):
        assign_folds(np.array([0, -5], dtype=np.int64))
    with pytest.raises(ValueError, match="integer"):
        assign_folds(pd.Series([1.0, 2.0]))
