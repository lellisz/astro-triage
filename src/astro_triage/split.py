"""Sky-region split by HEALPix level-5 pixel (ADR 0003 D3)."""

from __future__ import annotations

from typing import overload

import numpy as np
import pandas as pd
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy_healpix import HEALPix
from numpy.typing import NDArray

SEED = 20261025
NSIDE = 32
N_PIXELS = 12 * NSIDE**2
BAND_EDGES_DEG = (11.54, 30.0)
# Percent shares; integer arithmetic keeps floor(share * n) exact.
TRAIN_PCT = 70
VAL_PCT = 15

# Gaia source_id = healpix_l12 * 2**35 + ...; level 5 drops 7 levels of 4 children each.
_SOURCE_ID_DIVISOR = 2**35 * 4**7


@overload
def pixel_l5(source_id: int) -> int: ...
@overload
def pixel_l5(source_id: NDArray[np.int64] | pd.Series) -> NDArray[np.int64]: ...
def pixel_l5(
    source_id: int | NDArray[np.int64] | pd.Series,
) -> int | NDArray[np.int64]:
    """Return the level-5 NESTED HEALPix pixel (0..12287) of a Gaia source_id.

    Raises ValueError for non-integer input, negative ids or ids outside the pixel range.
    """
    ids = np.asarray(source_id)
    if ids.dtype.kind not in "iu":
        raise ValueError(
            f"source_id must be integer-typed, got dtype {ids.dtype} "
            "(missing values or floats cannot be mapped to a pixel)."
        )
    pixels = ids // _SOURCE_ID_DIVISOR
    invalid = (ids < 0) | (pixels < 0) | (pixels >= N_PIXELS)
    n_invalid = int(np.count_nonzero(invalid))
    if n_invalid:
        raise ValueError(
            f"{n_invalid} of {ids.size} source_id values are invalid "
            f"(negative or pixel outside 0..{N_PIXELS - 1})."
        )
    if ids.ndim == 0:
        return int(pixels)
    return pixels.astype(np.int64, copy=False)


def _centre_abs_galactic_latitude() -> NDArray[np.float64]:
    hp = HEALPix(nside=NSIDE, order="nested")
    lon, lat = hp.healpix_to_lonlat(np.arange(N_PIXELS))
    icrs = SkyCoord(ra=lon.to(u.deg), dec=lat.to(u.deg), frame="icrs")
    return np.abs(np.asarray(icrs.galactic.b.deg, dtype=np.float64))


def build_pixel_map() -> pd.DataFrame:
    """Build the pixel -> latitude band -> fold map, one row per pixel, sorted by pixel."""
    abs_b = _centre_abs_galactic_latitude()
    band = np.where(
        abs_b < BAND_EDGES_DEG[0], "A", np.where(abs_b < BAND_EDGES_DEG[1], "B", "C")
    )
    fold = np.empty(N_PIXELS, dtype=object)
    for label in ("A", "B", "C"):
        rng = np.random.default_rng(SEED)
        pixels = np.flatnonzero(band == label)
        shuffled = rng.permutation(pixels)
        n = len(pixels)
        n_train = n * TRAIN_PCT // 100
        n_val = n * VAL_PCT // 100
        fold[shuffled[:n_train]] = "train"
        fold[shuffled[n_train : n_train + n_val]] = "val"
        fold[shuffled[n_train + n_val :]] = "test"
    return pd.DataFrame(
        {
            "healpix_l5": np.arange(N_PIXELS, dtype=np.int16),
            "lat_band": band.astype(str),
            "fold": fold.astype(str),
        }
    )


def assign_folds(source_id: NDArray[np.int64] | pd.Series) -> NDArray[np.str_]:
    """Return the fold ('train', 'val' or 'test') of each source_id via the pixel map."""
    pixels = pixel_l5(np.asarray(source_id))
    folds = build_pixel_map()["fold"].to_numpy(dtype=str)
    return folds[pixels]
