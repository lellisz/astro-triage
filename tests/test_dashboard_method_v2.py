"""Método page v2: one rule text, ADR board read from the file headers, three limitations."""

from __future__ import annotations

import re
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest
from test_dashboard_comparacao import TIE, _results

from astro_triage.dashboard import adr_board as AB
from astro_triage.dashboard import compare_model as CM
from astro_triage.dashboard import data_dl as DD
from astro_triage.dashboard import html_method as HM
from astro_triage.dashboard import method_notes as notes
from astro_triage.dashboard import verdict_rule as VR
from astro_triage.dashboard import views_dl, views_method
from astro_triage.dashboard.html_system import NA, method_note

DASH = Path(views_method.__file__).parent
ADR_DIR = Path(__file__).resolve().parents[1] / "docs" / "adr"
REAL = sorted(ADR_DIR.glob("*.md"))
EXPECTED_LABEL = {"accepted": "Aceita", "proposed": "Proposta", "closed": "Encerrada"}
_REL = SimpleNamespace(
    meta={
        "n_test": 10,
        "n_test_pixels": 3,
        "n_boot": 7,
        "floor": {"min_objects": 5, "min_pixels": 2},
        "pi_src_combmod": {"QSO": 0.001},
    },
    test=pd.DataFrame({"true": ["STAR", "GALAXY", "QSO"]}),
    results={"limitations": ["Selection by crossmatch: x."]},
)


def _adr(tmp_path, name, status_line):
    (tmp_path / name).write_text(
        f"# ADR {name[:4]}: titulo\n\n{status_line}\n", encoding="utf-8"
    )


def _method_html(monkeypatch, tech):
    sent: list[str] = []
    monkeypatch.setattr(views_method.st, "markdown", lambda h, **_: sent.append(h))
    monkeypatch.setattr(views_method, "tech_on", lambda: tech)
    views_method.method_page(_REL)
    return "".join(sent)


def _line(html, number):
    return re.search(rf"<li>ADR {number}: .*?</li>", html).group(0)


def test_rule_text_method_equals_comparison(monkeypatch):
    note = method_note(VR.rule_text())
    comparison = views_dl.page_html(CM.build(_results(ftt=(TIE, "win", "tie"))), "")
    assert note in comparison and note in HM.rule_html()
    for tech in (False, True):
        assert note in _method_html(monkeypatch, tech)


def test_rule_has_a_single_source(monkeypatch):
    monkeypatch.setattr(VR, "rule_lines", lambda: ("REGRA UNICA",))
    comparison = views_dl.page_html(CM.build(_results(ftt=(TIE, "win", "tie"))), "")
    assert "REGRA UNICA" in comparison and "REGRA UNICA" in HM.rule_html()


@pytest.mark.parametrize("path", REAL, ids=lambda p: p.name)
def test_status_shown_equals_adr_header(path):
    text = path.read_text(encoding="utf-8")
    head = "\n".join(text.splitlines()[: AB.ADR_HEADER_LINES])
    found = re.search(r"^- Status:\s*\**\s*([A-Za-z]+)", head, re.MULTILINE)
    kind = found.group(1).lower() if found else None
    expected = kind if kind in EXPECTED_LABEL else "na"
    entry = next(e for e in AB.load_board(ADR_DIR) if e.number == path.name[:4])
    assert entry.chip == expected
    line = _line(HM.adr_board_html(AB.load_board(ADR_DIR)), entry.number)
    assert f">{EXPECTED_LABEL.get(expected, NA)}</span>" in line


def test_board_covers_every_adr_file_and_page_shows_it(monkeypatch):
    board = AB.load_board(ADR_DIR)
    assert [e.number for e in board] == [p.name[:4] for p in REAL]
    assert all(e.title for e in board)
    html = _method_html(monkeypatch, True)
    assert all(f"ADR {e.number}:" in html for e in board)


def test_proposed_never_shown_as_accepted(tmp_path):
    _adr(tmp_path, "0004-a.md", "- Status: **Proposed (gates pending).** text")
    _adr(tmp_path, "0006-b.md", "- Status: **Proposed** (draft)")
    board = AB.load_board(tmp_path)
    html = HM.adr_board_html(board)
    for e in board:
        assert e.chip == "proposed"
        line = _line(html, e.number)
        assert ">Proposta</span>" in line and "Aceita" not in line
    real = {e.number: e.chip for e in AB.load_board(ADR_DIR)}
    assert real["0004"] == "proposed" and real["0006"] == "proposed"


def test_unknown_or_missing_status_is_na(tmp_path):
    _adr(tmp_path, "0001-a.md", "- Status: Rascunho (x)")
    _adr(tmp_path, "0002-b.md", "- Date: x")
    _adr(tmp_path, "0003-c.md", "- Status: Accepted (no bold)")
    (tmp_path / "0005-d.md").mkdir()
    board = {e.number: e for e in AB.load_board(tmp_path)}
    assert board["0001"].chip == "na" and board["0002"].chip == "na"
    assert board["0003"].chip == "accepted"
    assert board["0005"].chip == "na" and board["0005"].status_word is None
    assert AB.chip_kind(None) == "na" and AB.chip_kind("Superseded") == "na"
    assert AB.parse_status(["# t", "- Status: **Closed without x**"]) == "Closed"
    assert ">n/a</span>" in _line(HM.adr_board_html(list(board.values())), "0001")
    assert AB.load_board(tmp_path / "missing") == ()
    assert NA in HM.adr_board_html(())


def test_tracked_filter_and_data_dl_delegation(tmp_path):
    _adr(tmp_path, "0006-a.md", "- Status: Accepted")
    _adr(tmp_path, "0007-b.md", "- Status: Proposed")
    assert [e.number for e in AB.load_board(tmp_path, {"0006-a.md"})] == ["0006"]
    assert DD.adr_status(tmp_path) == "Accepted"


def test_old_next_step_card_absent():
    old = ("Comparar com " + "redes neurais", "Próximo " + "passo")
    for path in DASH.glob("*.py"):
        text = path.read_text(encoding="utf-8")
        assert not any(s in text for s in old), path.name


def test_three_limitations_present():
    out = HM.three_limits_html(["Selection by crossmatch: the matched x."])
    assert HM.THREE_LIMITS_TITLE in out
    assert out.count("<li>") == 3
    assert notes.NEIGHBOUR_NOTE in out and notes.PRIOR_NOTE in out
    assert "Seleção por crossmatch" in out or "Selection by crossmatch" in out
    assert f"<li>{NA}</li>" in HM.three_limits_html([])


def test_page_shows_the_three_limitations_and_links_as_na(monkeypatch):
    for tech in (False, True):
        html = _method_html(monkeypatch, tech)
        assert HM.THREE_LIMITS_TITLE in html and notes.PRIOR_NOTE in html
        assert HM.LINKS_TITLE in html and HM.GLOSSARY_TITLE in html


def test_glossary_missing_term_is_na():
    out = HM.glossary_html((("Termo", None), ("Outro", " "), ("Ok", "definição")))
    assert out.count(f"<span>{NA}</span>") == 2 and "definição" in out
    assert "<details" in HM.glossary_details()


def test_adr_board_has_no_hand_written_digits():
    src = (DASH / "adr_board.py").read_text(encoding="utf-8")
    assert "0006" not in src and "0004" not in src
    entry = AB.load_board(ADR_DIR)[0]
    assert entry.number == REAL[0].name[:4]
