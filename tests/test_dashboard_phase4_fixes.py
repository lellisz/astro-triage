"""Phase 4 fixes still live: CI label, pixel stats columns, charts re-exports."""

from __future__ import annotations

from types import SimpleNamespace

import pandas as pd

from astro_triage.dashboard import data_sky, fmt


# (d) CI label
def test_ci_label_and_users():
    assert fmt.CI_LEVEL == 0.95
    assert fmt.pt_ci_label() == "IC 95%"
    assert "(IC 95%: 1,0% a 3,0%)" in fmt.pt_pct_interval(0.02, 0.01, 0.03)
    assert "(IC 95%: 0,100 a 0,300)" in fmt.pt_interval(0.2, 0.1, 0.3, "")
    assert fmt.pt_interval(0.2, 0.1, 0.3, "n/d") == "n/d"


# (e) data_sky
def test_pixel_stats_columns():
    t = pd.DataFrame(
        {"l": [10.0, 200.0, 359.0], "b": [5.0, -30.0, 60.0], "healpix_l5": [1, 1, 2],
         "err_lgbm_full": [True, False, False], "true": ["STAR", "QSO", "GALAXY"]}
    )  # fmt: skip
    rel = SimpleNamespace(test=t)
    px = pd.DataFrame(
        {"healpix_l5": [1, 2], "x": [0.0, 1.0], "y": [0.0, 1.0], "band": ["A", "C"]}
    )
    st = data_sky.test_pixel_stats(rel, "lgbm_full", px)
    assert "passes_floor" not in st.columns
    assert st.set_index("healpix_l5").loc[1, ["n", "errors"]].tolist() == [2, 1]
    assert st.set_index("healpix_l5").loc[1, "rate"] == 0.5


# (f) charts.py
