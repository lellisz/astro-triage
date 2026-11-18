"""Galáxias and Método v3: pure builders (log position, Wilson, title rule, extremes, modes)."""

from __future__ import annotations

import math
import re

import pandas as pd

from astro_triage.dashboard import html_galaxies as HG
from astro_triage.dashboard import html_galaxies_tech as HT
from astro_triage.dashboard import html_method as HM
from astro_triage.dashboard.data_galaxies import rate_row, wilson_interval
from astro_triage.dashboard.data_galaxies_z import GalaxyTotals

TECH_TERMS = re.compile(r"Wilson|logar|\bIC\b|n = |at-rowg", re.IGNORECASE)


def _band(label, lo, hi, n, k):
    return rate_row(label, lo, hi, pd.Series([True] * k + [False] * (n - k)))


def _tables(spike=True):
    ends, mid = (50, 5) if spike else (1, 30)
    by_g = pd.DataFrame(
        [
            _band("< 18", -math.inf, 18.0, 100, ends),
            _band("18–19", 18.0, 19.0, 1000, mid),
            _band("≥ 19", 19.0, math.inf, 1000, mid),
            _band("sem medida", math.nan, math.nan, 7, 0),
        ]
    )
    by_z = pd.DataFrame(
        [
            _band("z ≤ 0,001", -math.inf, 0.001, 100, ends),
            _band("0,001–0,1", 0.001, 0.1, 1000, mid),
            _band("≥ 0,1", 0.1, math.inf, 100, ends),
            _band("vazia", 5.0, 6.0, 0, 0),
        ]
    )
    return by_g, by_z


TOT = GalaxyTotals(n=1200, errors=60, to_qso=40, to_star=20)
FLOOR = 100  # reporting floor used by the fixtures


def test_log_position_endpoints_and_clamp():
    assert HT.log_pos(HT.LOG_LO_PCT) == 0.0
    assert math.isclose(HT.log_pos(HT.LOG_HI_PCT), 100.0)
    assert math.isclose(HT.log_pos(1.0), 100.0 / 3)
    assert HT.log_pos(0.0) == 0.0 and HT.log_pos(-1.0) == 0.0
    assert HT.log_pos(1e6) == 100.0 and HT.log_pos(math.nan) == 0.0


def test_wilson_interval_contains_rate_and_feeds_the_row():
    lo, hi = wilson_interval(5, 100)
    assert 0 < lo < 0.05 < hi < 1
    row = _band("x", 0.0, 1.0, 100, 5)
    assert (row["ci_lo"], row["ci_hi"]) == (lo, hi)
    zero = wilson_interval(0, 50)
    assert zero[0] == 0.0 and zero[1] > 0


def test_extreme_bands_chosen_by_position_not_label():
    by_g, by_z = _tables()
    picks = dict(HG.extreme_bands(by_g, by_z, FLOOR))
    assert picks["As mais brilhantes"].name == "G < 18"
    assert picks["As mais próximas"].name == "z ≤ 0,001"
    assert picks["As mais distantes"].name == "z ≥ 0,1"
    shuffled = by_z.iloc[::-1].reset_index(drop=True)
    assert dict(HG.extreme_bands(by_g, shuffled, FLOOR))["As mais distantes"].name == "z ≥ 0,1"


def test_title_rule_is_decided_by_the_data():
    by_g, by_z = _tables(spike=True)
    assert HG.TITLE_SPIKE in HG.simple_html(TOT, by_g, by_z, FLOOR)
    by_g, by_z = _tables(spike=False)
    out = HG.simple_html(TOT, by_g, by_z, FLOOR)
    assert HG.TITLE_NEUTRAL in out and HG.TITLE_SPIKE not in out
    assert "não é parelho" not in out


def test_simple_mode_has_four_bars_and_no_technical_terms():
    by_g, by_z = _tables()
    out = HG.simple_html(TOT, by_g, by_z, FLOOR)
    assert out.count("class='at-rowe'") == 4
    assert "Como ler:" in out and "viraram quasar" in out and "viraram estrela" in out
    assert not TECH_TERMS.search(out)
    assert "\n\n" not in out
    assert HG.ERROR_HIGH == 0.10
    assert out.count(f"background:{HG.theme.ACCENT}") == 3  # three ends above 10%


def test_title_neutral_when_extreme_below_floor():
    by_g, by_z = _tables(spike=True)
    out = HG.simple_html(TOT, by_g, by_z, FLOOR + 1)  # ends (n=100) fall under the floor
    assert HG.TITLE_NEUTRAL in out and "não é parelho" not in out
    assert out.count("class='at-rowe'") == 1
    assert HG.TITLE_SPIKE in HG.simple_html(TOT, *_tables(spike=True), FLOOR)


def test_title_neutral_when_wilson_lower_bound_does_not_pass_overall():
    by_g, by_z = _tables(spike=True)
    # 8 errors in 100: rate 8% > overall 5%, but the Wilson lower bound is below 5%
    by_g.loc[0] = _band("< 18", -math.inf, 18.0, 100, 8)
    by_z.loc[0] = _band("z ≤ 0,001", -math.inf, 0.001, 100, 8)
    by_z.loc[2] = _band("≥ 0,1", 0.1, math.inf, 100, 8)
    assert 0.08 > TOT.rate
    out = HG.simple_html(TOT, by_g, by_z, FLOOR)
    assert HG.TITLE_NEUTRAL in out and HG.TITLE_SPIKE not in out


def test_bar_colour_threshold():
    assert HG.rate_color(HG.ERROR_HIGH + 1e-9) == HG.theme.ACCENT
    assert HG.rate_color(HG.ERROR_HIGH) == HG.theme.SOFT


def test_technical_mode_has_both_log_tables():
    by_g, by_z = _tables()
    out = HT.tech_html(TOT, by_g, by_z, FLOOR)
    assert "Por brilho" in out and "Por distância" in out
    assert out.count("class='at-rowg'") == 2 + 3 + 4  # two headers, G rows, z rows (empty band kept)
    assert "Wilson" in out and "logarítmica" in out
    assert "7 galáxias sem magnitude" in out
    assert ">sem medida<" not in out
    assert "\n\n" not in out


def test_escape_and_no_trace():
    by_g, by_z = _tables()
    by_g.loc[0, "label"] = "<b>&"
    out = HT.tech_html(TOT, by_g, by_z, FLOOR)
    assert "<b>&" not in out and "&lt;b&gt;&amp;" in out
    both = out + HG.simple_html(TOT, by_g, by_z, FLOOR) + HM.simple_html() + HM.glossary_html()
    assert not re.search(r"\btrace\b", both, re.IGNORECASE)


def test_method_simple_has_one_sentence_and_hint():
    out = HM.simple_html()
    assert HM.LEAD in out and "Modo técnico" in out
    assert "ADR" not in out and not re.search(r"provis", out, re.IGNORECASE)


def test_mix_sentence_and_glossary_terms():
    text = HM.mix_sentence({"STAR": 0.5, "GALAXY": 0.3, "QSO": 0.2})
    assert "estrela 50%" in text and "galáxia 30%" in text and "quasar 20%" in text
    terms = [t.lower() for t, _ in HM.GLOSSARY]
    for need in ("pureza", "completude", "espectro", "prior", "piso de relato", "ic"):
        assert need in terms
