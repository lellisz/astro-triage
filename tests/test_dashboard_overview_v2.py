"""Visão geral v2: pure HTML builders, simple vs technical mode, escaping, numbers from data."""

from __future__ import annotations

from dataclasses import replace

from astro_triage.dashboard import data_overview_page as P
from astro_triage.dashboard import html_overview as HO


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






