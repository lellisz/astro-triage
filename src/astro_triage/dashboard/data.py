"""Data layer of the dashboard: reads one release and computes every number the screens show.

No Streamlit import here, so everything is testable without a browser. Nothing on the
screens is typed by hand: counts, rates, intervals and sentences all come from these functions.

Conventions that matter:
- Classes are always mapped by NAME (``p_cal_star`` ...), never by position: the prediction
  parquet orders its columns star, qso, galaxy, unlike the display order STAR, GALAXY, QSO.
- ``healpix_l5`` ids are HEALPix level 5 (nside 32), NESTED, in ICRS (checked against ra/dec).
  The latitude band of a pixel uses the Galactic latitude of its centre (schema.md, lat_band).
- Metrics below the reporting floor carry ``status != "ok"``; they are exposed as not usable.


Facade: the computations live in the ``data_*`` modules and are re-exported here.
``load_release`` and ``load_layer2`` stay in this module because they resolve ``INTERIM_DIR`` from
its namespace (tests patch ``data.INTERIM_DIR``).
"""

# ruff: noqa: F401
from __future__ import annotations

import json
from pathlib import Path

from astro_triage.dashboard.data_eval import (
    feature_importance,
    frame_from_cells,
    metrics_frame,
    model_summary,
    paired_diffs_frame,
    population_rows,
    results_variant,
)
from astro_triage.dashboard.data_galaxies import (
    GALAXY_G_EDGES,
    GALAXY_Z_EDGES,
    SOLVED_NAMES,
    bin_label,
    error_rate_table,
    galaxy_scatter,
    rate_row,
    wilson_interval,
)
from astro_triage.dashboard.data_galaxies_z import (
    Z_LOW,
    GalaxyTotals,
    ZLowSummary,
    error_rate_table_z,
    galaxy_totals,
    z_low_label,
    z_low_sentence,
    z_low_summary,
)
from astro_triage.dashboard.data_headline import (
    PAIR_NAMES,
    error_directions,
    error_pairs,
    load_delta,
    purity_points,
)
from astro_triage.dashboard.data_layer2 import (
    LAYER2_COST_ROWS,
    LAYER2_ROUTED,
    LAYER2_SHOWN_METRICS,
    LAYER2_SYSTEMS,
    layer2_diffs_frame,
    layer2_summary,
    layer2_units_frame,
)
from astro_triage.dashboard.data_load import read_release
from astro_triage.dashboard.data_overview import (
    CONFIDENCE_EDGES,
    Headline,
    abstention_curve,
    calibration_points,
    confidence_histogram,
    confusion,
    errors_by_type,
    headline,
    reliability_rows,
)
from astro_triage.dashboard.data_release import (
    CLASSES,
    DEFAULT_RELEASE,
    INTERIM_DIR,
    MODELS,
    POPULATIONS,
    PRIOR_REGIMES,
    REPO_ROOT,
    SCORE_COLUMN,
    SEED,
    VARIANTS,
    Release,
    classes_upper,
    single_file,
    wilson,
)
from astro_triage.dashboard.data_sky import (
    BAND_EDGES_DEG,
    HEALPIX_NSIDE,
    chance_share_of_pixels_with_error,
    lat_band_table,
    mollweide,
    pixel_table,
    test_pixel_stats,
)
from astro_triage.dashboard.data_triage import triage_summary


def load_release(name: str = DEFAULT_RELEASE, base: Path | None = None) -> Release:
    """Read results, training report, labels, features and both models' predictions."""
    root = (base or INTERIM_DIR) / name
    results = json.loads((root / "eval" / "results.json").read_text(encoding="utf-8"))
    report = json.loads((root / "train_report.json").read_text(encoding="utf-8"))
    return read_release(root, name, results, report)


def load_layer2(rel: Release | str, base: Path | None = None) -> dict | None:
    """Read ``layer2/eval_l2.json`` of a release; None when the release has no Layer 2 run.

    Returns the parsed document under ``raw`` plus ready-to-show frames: ``systems`` (S0-S3, S1*),
    ``routed`` (the routed-subset views) and ``diffs`` (paired differences with the ``n/a`` mark).
    """
    root = rel.root if isinstance(rel, Release) else (base or INTERIM_DIR) / rel
    path = root / "layer2" / "eval_l2.json"
    if not path.is_file():
        return None
    raw = json.loads(path.read_text(encoding="utf-8"))
    units = raw["eval"]["units"]
    return {
        "raw": raw,
        "systems": layer2_units_frame(units, LAYER2_SYSTEMS),
        "routed": layer2_units_frame(units, LAYER2_ROUTED),
        "diffs": layer2_diffs_frame(raw["eval"]["diffs"]),
    }
