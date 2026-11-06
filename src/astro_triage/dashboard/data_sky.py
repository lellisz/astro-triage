"""Sky numbers: Mollweide projection, HEALPix pixel table, per-pixel and per-band error rates."""

from __future__ import annotations

import math
from typing import Final

import numpy as np
import pandas as pd

from astro_triage.dashboard.data_release import (
    Release,
    wilson,
)

HEALPIX_NSIDE: Final = 32
BAND_EDGES_DEG: Final = (
    11.54,
    30.0,
)  # A < 11.54 <= B < 30 <= C, on |b| of the pixel centre
TOP_REGIONS: Final = 6  # rows of the "regions with most errors" table


def mollweide(l_deg: np.ndarray, b_deg: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Mollweide x, y in units where the map spans x in +-2*sqrt(2), y in +-sqrt(2).

    Galactic centre in the middle, longitude increasing to the left.
    """
    lon = np.asarray(l_deg, dtype=float)
    lon = np.where(lon > 180.0, lon - 360.0, lon)
    lam, phi = -np.radians(lon), np.radians(np.asarray(b_deg, dtype=float))
    theta = phi.copy()
    for _ in range(40):
        f = 2 * theta + np.sin(2 * theta) - np.pi * np.sin(phi)
        g = 2 + 2 * np.cos(2 * theta)
        theta = theta - np.divide(f, g, out=np.zeros_like(f), where=np.abs(g) > 1e-9)
    return 2 * math.sqrt(2) / math.pi * lam * np.cos(theta), math.sqrt(2) * np.sin(
        theta
    )


def inverse_mollweide(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Galactic (l in [0, 360), b) in degrees from Mollweide x, y; inverse of ``mollweide``."""
    xa, ya = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    theta = np.arcsin(np.clip(ya / math.sqrt(2), -1.0, 1.0))
    b = np.degrees(
        np.arcsin(np.clip((2 * theta + np.sin(2 * theta)) / np.pi, -1.0, 1.0))
    )
    cos_t = np.cos(theta)
    lam = np.divide(
        xa * np.pi,
        2 * math.sqrt(2) * cos_t,
        out=np.zeros_like(xa),
        where=cos_t > 1e-9,
    )
    return np.mod(-np.degrees(lam), 360.0), b


def pixel_table(rel: Release) -> pd.DataFrame:
    """One row per HEALPix pixel: fold, objects, Galactic centre, map x/y and latitude band."""
    from astropy.coordinates import SkyCoord
    from astropy_healpix import HEALPix

    px = (
        rel.pixel_counts.groupby("healpix_l5")
        .agg(fold=("fold", "first"), n=("n", "sum"))
        .reset_index()
    )
    lon, lat = HEALPix(nside=HEALPIX_NSIDE, order="nested").healpix_to_lonlat(
        px["healpix_l5"].to_numpy()
    )
    gal = SkyCoord(lon, lat, frame="icrs").galactic
    px["l_centre"], px["b_centre"] = gal.l.deg, gal.b.deg
    px["x"], px["y"] = mollweide(px["l_centre"].to_numpy(), px["b_centre"].to_numpy())
    px["band"] = np.where(
        px["b_centre"].abs() < BAND_EDGES_DEG[0],
        "A",
        np.where(px["b_centre"].abs() < BAND_EDGES_DEG[1], "B", "C"),
    )
    return px


def test_pixel_stats(rel: Release, model: str, pixels: pd.DataFrame) -> pd.DataFrame:
    """Per test pixel: objects, errors and error rate."""
    g = (
        rel.test.groupby("healpix_l5")[f"err_{model}"]
        .agg(n="size", errors="sum")
        .reset_index()
    )
    out = g.merge(pixels[["healpix_l5", "x", "y", "band"]], on="healpix_l5", how="left")
    out["rate"] = out["errors"] / out["n"]
    return out


def lat_band_table(stats: pd.DataFrame) -> pd.DataFrame:
    """Test objects and errors by latitude band A/B/C of the pixel centre, with Wilson intervals.

    ``stats`` is the output of :func:`test_pixel_stats`.
    """
    rows = []
    for band in ("A", "B", "C"):
        sub = stats[stats["band"] == band]
        n, k = int(sub["n"].sum()), int(sub["errors"].sum())
        lo, hi = wilson(k, n)
        rows.append(
            {
                "band": band,
                "n_pixels": len(sub),
                "n": n,
                "errors": k,
                "rate": k / n if n else math.nan,
                "ci_lo": lo,
                "ci_hi": hi,
            }
        )
    return pd.DataFrame(rows)


def chance_share_of_pixels_with_error(stats: pd.DataFrame, error_rate: float) -> float:
    """Share of test pixels expected to hold at least one error if errors fell at random."""
    return float((1.0 - (1.0 - error_rate) ** stats["n"].to_numpy()).mean())


def top_error_regions(
    stats: pd.DataFrame, floor: int, top: int = TOP_REGIONS
) -> pd.DataFrame:
    """The ``top`` regions with the highest error rate among those with at least ``floor`` objects.

    Adds the Galactic position ``l``, ``b`` recovered from the map coordinates.
    """
    ok = (stats["n"] >= floor) & np.isfinite(stats["x"]) & np.isfinite(stats["y"])
    out = (
        stats[ok]
        .sort_values(["rate", "n"], ascending=False, kind="stable")
        .head(top)
        .copy()
    )
    out["l"], out["b"] = inverse_mollweide(out["x"].to_numpy(), out["y"].to_numpy())
    return out
