"""Sky map v3 (HTML/CSS Mollweide): pure builders, inverse projection, simple/technical texts."""

from __future__ import annotations

import math
import re
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from test_dashboard_phase4_sky2 import _literals, has_hand_written_digits

from astro_triage.dashboard import data_sky, sky_map_html, sky_text, theme

DASH = Path(__file__).resolve().parents[1] / "src" / "astro_triage" / "dashboard"
FLOOR = 100
TECH_TERMS = re.compile(r"healpix|nside|wilson|\bIC\b", re.IGNORECASE)


def _stats() -> pd.DataFrame:
    l_deg = np.array([0.0, 10.0, 90.0, 200.0, 300.0, 45.0])
    b_deg = np.array([0.0, 40.0, -30.0, 60.0, -10.0, 5.0])
    x, y = data_sky.mollweide(l_deg, b_deg)
    n = np.array([500, 400, 300, 600, 50, 200])
    k = np.array([100, 4, 60, 12, 5, 30])
    band = ["A", "C", "B", "C", "A", "A"]
    return pd.DataFrame(
        {"n": n, "errors": k, "rate": k / n, "x": x, "y": y, "band": band}
    )


def _bands() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "band": ["A", "B", "C"],
            "n_pixels": [3, 1, 2],
            "n": [750, 300, 1000],
            "errors": [135, 60, 16],
            "rate": [0.18, 0.2, 0.016],
            "ci_lo": [0.15, 0.16, 0.01],
            "ci_hi": [0.21, 0.25, 0.026],
        }
    )


# inverse projection
@pytest.mark.parametrize("l_deg", [1.0, 30.0, 89.0, 180.5, 270.0, 359.0])
@pytest.mark.parametrize("b_deg", [-80.0, -45.0, -11.54, 0.0, 12.0, 60.0, 85.0])
def test_inverse_mollweide_round_trip(l_deg, b_deg):
    x, y = data_sky.mollweide(np.array([l_deg]), np.array([b_deg]))
    l_back, b_back = data_sky.inverse_mollweide(x, y)
    assert b_back[0] == pytest.approx(b_deg, abs=1e-6)
    assert l_back[0] == pytest.approx(l_deg, abs=1e-6)


def test_inverse_mollweide_centre_and_poles_are_finite():
    l_back, b_back = data_sky.inverse_mollweide(
        np.array([0.0, 0.0, 0.0]), np.array([0.0, math.sqrt(2), -math.sqrt(2)])
    )
    assert np.all(np.isfinite(l_back)) and np.all((0 <= l_back) & (l_back < 360))
    assert b_back == pytest.approx([0.0, 90.0, -90.0])


# graticule geometry
def _theta(lat_deg: float) -> float:
    lo, hi, target = (
        -math.pi / 2,
        math.pi / 2,
        math.pi * math.sin(math.radians(lat_deg)),
    )
    for _ in range(80):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if 2 * mid + math.sin(2 * mid) < target else (lo, mid)
    return (lo + hi) / 2


@pytest.mark.parametrize("lat", [-60.0, -30.0, -11.54, 0.0, 11.54, 30.0, 60.0])
def test_parallel_width_is_cos_theta_and_top_is_sin_theta(lat):
    top, left, width = sky_map_html.parallel_geometry(lat)
    th = _theta(lat)
    assert width == pytest.approx(math.cos(th) * 100, abs=1e-6)
    assert top == pytest.approx(50 - math.sin(th) * 50, abs=1e-6)
    assert left == pytest.approx((100 - width) / 2, abs=1e-6)


def test_parallels_are_mirrored_about_the_equator():
    n, s = sky_map_html.parallel_geometry(30.0), sky_map_html.parallel_geometry(-30.0)
    assert n[0] + s[0] == pytest.approx(100.0)
    assert n[2] == pytest.approx(s[2])


def test_meridian_width_is_lon_over_180():
    assert sky_map_html.meridian_style(90) == "left:25.00%;width:50.00%"
    assert sky_map_html.meridian_style(180) == "left:0.00%;width:100.00%"


def test_longitude_labels_grow_to_the_left_and_skip_0_and_180():
    pos = dict(sky_map_html.lon_label_positions())
    assert sorted(pos) == [30, 60, 90, 120, 150, 210, 240, 270, 300, 330]
    assert pos[90] == pytest.approx(25.0) and pos[270] == pytest.approx(75.0)
    assert pos[30] == pytest.approx(50 - 30 / 180 * 50)
    assert pos[330] == pytest.approx(50 + 30 / 180 * 50)
    assert all(pos[a] < pos[b] for a, b in [(150, 30), (30, 330), (330, 210)])


def test_graticule_labels_and_dashed_band_edges():
    g = sky_map_html.graticule()
    for text in ("+30°", "−30°", "+60°", "−60°", ">0°</span>"):
        assert text in g
    assert g.count("at-pd") == 5  # equator and four band limits (+-11,54 and +-30)
    assert g.count("at-lat") == 5 and g.count("at-lon") == 10


# map block
def test_map_is_one_single_line_without_blank_lines_or_trace():
    html = sky_map_html.build_map(_stats(), True, FLOOR)
    assert "\n" not in html and "trace" not in html.lower()
    assert html.count("<i class='at-pt") == len(_stats())


def test_rate_mode_rings_below_floor_and_hide_option():
    s = _stats()
    shown = sky_map_html.build_map(s, True, FLOOR)
    hidden = sky_map_html.build_map(s, True, FLOOR, hide_below=True)
    assert shown.count("at-pt at-ring") == 1 and hidden.count("at-pt at-ring") == 0
    assert sky_map_html.build_map(s, False, FLOOR).count("at-pt at-ring") == 0


def test_colour_cap_is_percentile_95_of_displayed_values():
    s = _stats()
    rates = s.loc[s["n"] >= FLOOR, "rate"]
    assert sky_map_html.color_cap(s, True, FLOOR) == pytest.approx(
        np.percentile(rates, 95)
    )
    assert sky_map_html.color_cap(s, False, FLOOR) == pytest.approx(
        np.percentile(s["n"], 95)
    )


def test_highest_value_gets_the_top_of_the_ramp_and_legend_is_computed():
    s = _stats()
    html = sky_map_html.build_map(s, True, FLOOR)
    assert theme.ramp(1.0) in html
    cap = sky_map_html.color_cap(s, True, FLOOR)
    assert f"{sky_text.fmt.pt_pct(cap)} ou mais" in html
    assert "objetos ou mais" in sky_map_html.build_map(s, False, FLOOR)


def test_text_is_escaped():
    assert "<script>" not in sky_map_html.lead_line("<script>alert(1)</script>")
    assert "&lt;script&gt;" in sky_map_html.lead_line("<script>")


# texts
def test_map_lead_sentence_uses_computed_values_and_named_threshold():
    text = sky_text.map_lead(_stats(), FLOOR)
    s = _stats()
    overall = s["errors"].sum() / s["n"].sum()
    assert text.startswith(f"A taxa geral é {sky_text.fmt.pt_pct(overall)}.")
    assert "Em 3 das 5 regiões com dados confiáveis" in text
    assert sky_text.ERROR_HIGH == 0.10 and "ela passa de 10%." in text


def test_simple_mode_has_no_technical_terms():
    s = _stats()
    simple = [
        sky_text.TITLE,
        sky_text.lead(len(s)),
        sky_text.how(True, 1, FLOOR, False),
        sky_text.how(True, 1, FLOOR, True),
        sky_text.map_lead(s, FLOOR),
        sky_map_html.build_map(s, True, FLOOR),
    ]
    for text in simple:
        assert not TECH_TERMS.search(re.sub(r"<[^>]*>", "", text)), text


def test_technical_tables():
    bands = sky_text.bands_box(_bands())
    assert bands.count("<tr>") == 4 and "IC 95%" in bands
    assert "A · |b| &lt; 11,54° (plano da Via Láctea)" in bands
    assert "C · acima de 30°" in bands and "15,00% a 21,00%" in bands
    top = sky_text.top_box(data_sky.top_error_regions(_stats(), FLOOR), FLOOR)
    assert "Posição (l, b)" in top and "mínimo de 100 objetos" in top
    assert "nside" in top and "Regiões com mais erro" in top


def test_top_error_regions_order_floor_and_position():
    top = data_sky.top_error_regions(_stats(), FLOOR)
    assert list(top["n"]) == [500, 300, 200, 600, 400]  # 20%, 20%, 15%, 2%, 1%
    assert (top["n"] >= FLOOR).all()
    first = top.iloc[0]
    assert (first["l"] % 360, first["b"]) == pytest.approx((0.0, 0.0), abs=1e-6)
    assert sky_text.position_text(359.6, -29.7) == "l = 0°, b = −30°"
    assert sky_text.position_text(90.2, 5.4) == "l = 90°, b = +5°"
    assert len(data_sky.top_error_regions(_stats(), FLOOR, top=2)) == 2


# sweeps
@pytest.mark.parametrize("name", ["views_sky.py", "sky_text.py"])
def test_no_hand_written_digits_in_text_modules(name):
    bad = [
        (n.lineno, n.value[:60])
        for n in _literals(DASH / name)
        if isinstance(n.value, str) and has_hand_written_digits(n.value)
    ]
    assert not bad, bad


@pytest.mark.parametrize("name", ["views_sky.py", "sky_text.py", "sky_map_html.py"])
def test_size_hex_and_forbidden_words(name):
    src = (DASH / name).read_text(encoding="utf-8")
    assert len(src.splitlines()) <= 200
    assert not re.findall(r"#[0-9a-fA-F]{6}\b", src)
    assert "PROVIS" not in src.upper() and "console.log" not in src
