"""Sky page, phase 6: cards from the artifact, closed map payload, one layer, narrow viewport."""

from __future__ import annotations

import re

import numpy as np
import pandas as pd
import pytest

from astro_triage.dashboard import sky_map_html, sky_model, sky_text
from astro_triage.dashboard.kit_types import Value

FLOOR = 100


def _stats() -> pd.DataFrame:
    n = np.array([500, 400, 300, 600, 50, 20, 150])
    k = np.array([100, 4, 60, 12, 5, 20, 0])
    return pd.DataFrame(
        {
            "healpix_l5": np.arange(len(n)) + 7000,
            "n": n,
            "errors": k,
            "rate": k / n,
            "x": np.linspace(-2.0, 2.0, len(n)),
            "y": np.linspace(-1.0, 1.0, len(n)),
            "band": list("ABCABCA"),
        }
    )


def _meta(regions: int | None = None, objects: int | None = None) -> dict:
    s = _stats()
    return {
        "floor": {"min_objects": FLOOR},
        "n_test_pixels": len(s) if regions is None else regions,
        "n_test": int(s["n"].sum()) if objects is None else objects,
    }


def test_colored_plus_below_equals_test_regions():
    s = _stats()
    c = sky_model.cards(s, _meta())
    coloured = int((s["n"] >= FLOOR).sum())
    assert coloured + c.n_below == c.n_regions == len(s)
    assert c.n_reliable == coloured


def test_cards_read_from_artifact_meta_and_aggregate():
    s = _stats()
    c = sky_model.cards(s, _meta())
    assert (c.n_regions, c.n_objects, c.floor) == (len(s), int(s["n"].sum()), FLOOR)
    assert (c.meta_regions, c.meta_objects) == (c.n_regions, c.n_objects)
    assert not c.diverges
    assert isinstance(c.max_rate, Value)
    assert c.max_rate.ci is not None
    assert c.max_rate.ci.lo <= c.max_rate.point <= c.max_rate.ci.hi


def test_cards_report_divergence_between_meta_and_aggregate():
    c = sky_model.cards(_stats(), _meta(regions=len(_stats()) + 1))
    assert c.diverges and c.n_regions == len(_stats())
    assert "n/a" in sky_text.diverge_note(None, None)


def test_max_rate_ignores_regions_below_floor():
    s = _stats()
    assert s["rate"].max() == 1.0  # the region with n=20 is all errors
    c = sky_model.cards(s, _meta())
    assert c.max_rate.point == pytest.approx(0.2)


def test_max_rate_is_na_without_reliable_regions():
    s = _stats().assign(n=1)
    c = sky_model.cards(s, _meta())
    assert c.max_rate == Value(None, None)


def test_max_rate_zero_keeps_point_inside_interval():
    s = _stats().assign(errors=0, rate=0.0)
    c = sky_model.cards(s, _meta())
    assert c.max_rate.point == 0.0 and c.max_rate.ci.lo == 0.0


def test_map_payload_has_only_region_columns():
    s = _stats().assign(source_id=1, ra=1.0, dec=1.0)
    rate = sky_model.map_payload(s, FLOOR, show_count=False)
    count = sky_model.map_payload(s, FLOOR, show_count=True)
    assert list(rate.columns) == ["healpix_l5", "x", "y", "rate", "below_floor"]
    assert list(count.columns) == ["healpix_l5", "x", "y", "n", "below_floor"]
    assert len(rate) <= s["healpix_l5"].nunique()
    assert rate.loc[rate["below_floor"], "rate"].isna().all()


def test_count_layer_only_in_tech():
    import inspect

    from astro_triage.dashboard import views_sky

    src = inspect.getsource(views_sky.sky_page)
    assert src.index("if tech:") < src.index("_pick_mode()")
    s = _stats()
    simple = sky_map_html.build_map(s, True, FLOOR)
    assert "objetos ou mais" not in simple and "n/a: menos de" in simple
    assert "objetos ou mais" in sky_map_html.build_map(s, False, FLOOR)


def test_map_has_no_fixed_width_wider_than_narrow_viewport():
    html = sky_map_html.build_map(_stats(), True, FLOOR)
    fixed = re.findall(r"(?<![-\w])(?:min-)?width:(\d+(?:\.\d+)?)px", html)
    assert all(float(w) <= 390 for w in fixed), fixed
    # the map itself stays fluid: all of the row, minus the margin reserved for latitude labels
    assert re.search(r"\.at-map\{[^}]*width:(?:100%|calc\(100% - \d+px\))", html)


def test_legend_has_linear_scale_and_na_outline_but_no_log():
    html = sky_map_html.build_map(_stats(), True, FLOOR)
    assert "linear-gradient" in html and "at-lr at-ring" in html
    hidden = sky_map_html.build_map(_stats(), True, FLOOR, hide_below=True)
    assert "at-lr" not in hidden.split("</style>")[1]


def test_count_card_has_no_interval_and_escapes():
    html = sky_text.count_card("<b>", "1", "x")
    assert "&lt;b&gt;" in html and "hs2-between" not in html
