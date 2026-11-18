"""Comparação page v2: grid, fixed point, power phrase, absolute purity, no typed verdict text."""

from __future__ import annotations

import re
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

from test_dashboard_comparacao import TIE, _results

from astro_triage.dashboard import compare_model as CM
from astro_triage.dashboard import compare_text as CT
from astro_triage.dashboard import views_dl

DASH = Path(views_dl.__file__).parent


def _model():
    return CM.build(_results(ftt=(TIE, "win", "tie")))


def _html(model):
    return views_dl.page_html(model, "")


def _render(monkeypatch, results, state=None):
    chunks = []
    fake = SimpleNamespace(
        markdown=lambda t, **_: chunks.append(t),
        expander=lambda *_: nullcontext(),
        session_state=state or {},
    )
    monkeypatch.setattr(views_dl, "st", fake)
    monkeypatch.setattr(views_dl.DC, "load_results_dl", lambda r: results)
    monkeypatch.setattr(views_dl.DD, "adr_status", lambda: "Proposed")
    monkeypatch.setattr(views_dl.DD, "freeze_exists", lambda r: True)
    monkeypatch.setattr(views_dl.DD, "load_limitations", lambda r: None)
    views_dl.dl_page(SimpleNamespace(name="full_v0", root=Path("nowhere")))
    return " ".join(chunks)


def test_grid_has_four_cells_qso_galaxy_by_mlp_ftt():
    got = [(g.cls, g.net) for g in _model().grid]
    assert got == [
        ("QSO", "MLP"),
        ("QSO", "FTT"),
        ("GALAXY", "MLP"),
        ("GALAXY", "FTT"),
    ]
    assert [n for n, _ in _model().pairs] == ["MLP", "FTT"]


def test_page_ignores_sidebar_prior(monkeypatch):
    a = _render(monkeypatch, _results(ftt=(TIE, "win", "tie")), {"ui_prior": "P1"})
    b = _render(monkeypatch, _results(ftt=(TIE, "win", "tie")), {"ui_prior": "P3"})
    assert a == b and "P2 · C* = 0,80" in a and "Regra · prior" in a


def test_power_phrase_only_on_underpowered_ties():
    assert CT.POWER_PHRASE in _html(_model())
    narrow = _results(ftt=(TIE, "win", "tie"))
    for pair in narrow["compare"]["pairs"].values():
        for levels in pair.values():
            for cell in levels.values():
                if cell["role"] == "headline":
                    cell["classes"]["GALAXY"]["ci"] = [-0.0003, 0.0003]
    model = CM.build(narrow)
    assert CT.POWER_PHRASE not in _html(model)
    assert [g.underpowered for g in model.grid] == [False] * 4


def test_absolute_purity_labelled_not_ci_and_read_from_artifact():
    model = _model()
    assert model.purity and model.purity[0][0] == CM.REFERENCE_SYSTEM
    html = _html(model)
    for token in (CT.NOT_CI_LABEL, "5,00%", "4,00%", "6,00%"):
        assert token in html


def test_no_verdict_text_typed_in_views():
    na_word = CT.STATE_WORD[CM.ClassVerdict.NA]
    words = [*CT.STATE_WORD.values(), *CT.PAIR_WORD.values(), CT.POWER_PHRASE]
    for name in (
        "views_dl.py",
        "compare_model.py",
        "views_galaxies.py",
        "html_galaxy_prior.py",
    ):
        text = (DASH / name).read_text(encoding="utf-8")
        for w in words:
            if w != na_word:
                assert f'"{w}' not in text, (name, w)


def test_unavailable_when_artifact_missing(monkeypatch):
    out = _render(monkeypatch, None)
    assert "não disponível" in out and not re.search(r"\d,\d", out)


def test_answer_has_no_banned_words():
    text = CT.answer(_model())
    assert not re.search(r"\b(só|melhor|equivalente)\b", text)
    assert "MLP" in text and "FTT" in text


def test_inconsistent_artifact_is_visible(monkeypatch):
    out = _render(monkeypatch, _results(ftt=("X wins", "win", "tie")))
    assert "ilegíveis" in out and "difere" in out
