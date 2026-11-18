"""Visão geral v2: pure HTML builders, simple vs technical mode, escaping, numbers from data."""

from __future__ import annotations

from dataclasses import replace

import pytest

from astro_triage.dashboard import data_overview_page as P
from astro_triage.dashboard import html_overview as HO
from astro_triage.dashboard import html_overview_plots as HP
from astro_triage.dashboard import html_overview_triage as HT


def _pt(p, c, lo=None, hi=None, key="as_shipped"):
    return {
        "key": key,
        "purity": p,
        "completeness": c,
        "purity_lo": lo,
        "purity_hi": hi,
    }


def _data(**kw) -> P.PageData:
    base = P.PageData(
        n_pairs=1234567,
        n_test=54321,
        n_errors=777,
        n_pixels=321,
        accuracy=0.9123,
        class_mix={"STAR": 0.5, "GALAXY": 0.3, "QSO": 0.2},
        qso_hits=4321,
        qso_total=5555,
        shipped=_pt(0.0321, 0.9456, 0.0211, 0.0432),
        matched=_pt(0.6543, 0.1770, 0.5111, 0.8222, "prior_matched"),
        validation=_pt(0.0560, 0.8, key="validation"),
        directions=[
            ("STAR", "QSO", 400),
            ("QSO", "STAR", 200),
            ("QSO", "GALAXY", 80),
            ("GALAXY", "QSO", 50),
            ("GALAXY", "STAR", 30),
            ("STAR", "GALAXY", 17),
        ],
        triage={
            "threshold": 0.9,
            "coverage": 0.8770,
            "accuracy_kept": 0.9876,
            "n_set_aside": 6712,
            "errors_set_aside": 555,
            "errors_total": 777,
        },
    )
    return replace(base, **kw)


def _all(d, tech):
    return "".join(
        (
            HO.data_box_html(d) if tech else "",
            HP.errors_html(d, tech),
            HT.triage_html(d, tech),
        )
    )


@pytest.mark.parametrize("tech", [False, True])
def test_blocks_are_single_line_without_blank_lines(tech):
    for html in (_all(_data(), tech),):
        assert "\n" not in html and "trace" not in html.lower()


def test_data_box_comes_from_data():
    html = HO.data_box_html(_data())
    for part in (
        "1.234.567 pares",
        "54.321 objetos em 321 regiões",
        "estrela 50%",
        "quasar 20%",
    ):
        assert part in html
    assert "LightGBM com calibração" in html and "DSC" in html


def test_errors_title_and_bar_counts():
    simple, tech = HP.errors_html(_data(), False), HP.errors_html(_data(), True)
    assert "51% dos 777 erros são o mesmo engano" in simple
    assert "Estrelas que o modelo tomou por quasar." in simple
    assert simple.count("at-rowe") == 3 and tech.count("at-rowe") == 6
    assert "estrela → quasar" in simple and "galáxia → estrela" not in simple
    assert "galáxia → estrela" in tech and "3,9%" in tech


def test_errors_no_errors_does_not_divide_by_zero():
    html = HP.errors_html(_data(directions=[("STAR", "QSO", 0)]), False)
    assert "at-rowe" not in html and "Sem erros" in html


def test_triage_numbers_note_and_points():
    html = HT.triage_html(_data(), False)
    assert "limiar 0,90 · decide 87,7% · acerta 98,8%" in html
    assert "decidir tudo · acerta 91,23%" in html
    assert "Os outros 6.712 vão para um espectro" in html
    assert (
        "O eixo vertical começa em 91%" in html
    )  # accuracy 0.9123 is below the 96% floor
    assert "ao-pa" in html and "ao-pb" in html and "Detalhe técnico" not in html


def test_triage_axis_floor_is_the_default_when_points_are_high():
    html = HT.triage_html(_data(accuracy=0.9639), False)
    assert "O eixo vertical começa em 96%" in html


def test_triage_technical_box():
    html = HT.triage_html(_data(), True)
    assert "decide 47.609 objetos" in html and "erra 222 deles" in html
    assert "concentram 555 dos 777 erros" in html


def test_triage_missing_curve():
    assert "Sem curva" in HT.triage_html(_data(triage=None), True)


def test_total_pairs_and_class_mix():
    assert P.total_pairs({"n_per_fold": {"train": 5, "val": 3, "test": 2}}) == 10
    assert P.total_pairs({}) is None and P.total_pairs({"n_per_fold": "x"}) is None
    mix = P.class_mix_of({"n_by_class": {"STAR": 1, "GALAXY": 1, "QSO": 2}})
    assert mix == {"STAR": 0.25, "GALAXY": 0.25, "QSO": 0.5}
    assert P.class_mix_of({}) == {}


def test_usable_triage_rejects_nan():
    ok = _data().triage
    assert P.usable_triage(ok) is ok
    assert P.usable_triage({**ok, "coverage": float("nan")}) is None
    assert P.usable_triage(None) is None


def test_triage_gain_sentence_only_when_deciding_beats_deciding_all():
    assert HT.HOW_GAIN in HT.triage_html(_data(), False)  # 98,8% > 91,23%
    assert HT.HOW_GAIN not in HT.triage_html(_data(accuracy=0.995), False)


def test_triage_label_anchored_by_side_of_the_point():
    assert HT.label_side(0.0) == "ao-la-l" and HT.label_side(HT.PERCENT) == "ao-la-r"
    assert "ao-la ao-la-l" in HT.triage_html(_data(), False) or "ao-la ao-la-r" in HT.triage_html(_data(), False)


