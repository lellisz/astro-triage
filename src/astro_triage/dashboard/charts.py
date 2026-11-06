"""Plotly figures for the dashboard, in the editorial dark style.

Design rules: no colour per class (classes are told apart by symbol), no hue at all: errors are
told apart by shape and label, everything else is warm neutral greys (see ``theme``).


Facade: the figures live in the ``charts_*`` modules and are re-exported here.
"""

# ruff: noqa: F401
from __future__ import annotations

from astro_triage.dashboard.charts_base import (
    BELOW_FLOOR,
    BG,
    CLASS_PT,
    ERROR,
    GLYPH,
    HAIRLINE,
    MODEL_PT,
    MODEL_STYLE,
    NEUTRAL,
    SYMBOL,
    TEXT,
    TEXT_MUTED,
    TEXT_SOFT,
    _base,
    label,
)
from astro_triage.dashboard.charts_calibration import (
    calibration,
    confidence_hist,
)
from astro_triage.dashboard.charts_galaxies import (
    RATE_LOG_RANGE,
    RATE_TICKS,
    base_marker_traces,
    floor_marker,
    low_n_rows,
    n_labels,
    rate_bars,
    tiny_rate_rows,
    zero_error_rows,
)
from astro_triage.dashboard.charts_galaxies_rates import rate_dots
from astro_triage.dashboard.charts_galaxies_scatter import galaxy_scatter
from astro_triage.dashboard.charts_layer2 import layer2_f1
from astro_triage.dashboard.charts_model import dumbbell, error_bars, pair_text, per_100
from astro_triage.dashboard.charts_overview import (
    _DTICKS,
    error_types,
    f1_with_ci,
    importance,
    paired_diffs,
    paired_dtick,
    recall_dots,
    waffle,
)
from astro_triage.dashboard.charts_triage import triage_curve
