"""Sky page numbers and map payload. Pure: no streamlit, nothing per object.

Cards read the artifact ``meta`` and are checked against the per-region aggregate; when they
differ, the aggregate is shown and the difference is kept in ``meta_*`` for the page to report.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np
import pandas as pd

from astro_triage.dashboard.data_release import wilson
from astro_triage.dashboard.kit_types import Interval, Value


@dataclass(frozen=True)
class SkyCards:
    n_regions: int
    n_objects: int
    n_below: int
    floor: int
    max_rate: Value
    meta_regions: int | None = None
    meta_objects: int | None = None

    @property
    def n_reliable(self) -> int:
        return self.n_regions - self.n_below

    @property
    def diverges(self) -> bool:
        return (self.meta_regions, self.meta_objects) != (None, None) and (
            self.meta_regions != self.n_regions or self.meta_objects != self.n_objects
        )


def _meta_int(meta: Mapping, key: str) -> int | None:
    value = meta.get(key)
    return int(value) if isinstance(value, (int, float)) else None


def _max_rate(reliable: pd.DataFrame) -> Value:
    if reliable.empty:
        return Value(None, None)
    top = reliable.sort_values(["rate", "n"], ascending=False, kind="stable").iloc[0]
    point = float(top["rate"])
    lo, hi = wilson(int(top["errors"]), int(top["n"]))
    if not (np.isfinite(lo) and np.isfinite(hi)):
        return Value(point, None)
    return Value(point, Interval(min(lo, point), max(hi, point)))


def cards(stats: pd.DataFrame, meta: Mapping) -> SkyCards:
    """Counts from the per-region aggregate; the maximum rate only over regions at the floor."""
    floor = int(meta["floor"]["min_objects"])
    below = stats["n"] < floor
    return SkyCards(
        n_regions=len(stats),
        n_objects=int(stats["n"].sum()),
        n_below=int(below.sum()),
        floor=floor,
        max_rate=_max_rate(stats[~below]),
        meta_regions=_meta_int(meta, "n_test_pixels"),
        meta_objects=_meta_int(meta, "n_test"),
    )


def map_payload(stats: pd.DataFrame, floor: int, show_count: bool) -> pd.DataFrame:
    """Closed columns sent to the map: region id, position, one layer, below-floor flag.

    The rate of a region below the floor is withheld (NaN): it is n/a on the map.
    """
    below = stats["n"] < floor
    out = stats[["healpix_l5", "x", "y"]].copy()
    if show_count:
        out["n"] = stats["n"].to_numpy()
    else:
        out["rate"] = stats["rate"].where(~below).to_numpy()
    out["below_floor"] = below.to_numpy()
    return out.reset_index(drop=True)
