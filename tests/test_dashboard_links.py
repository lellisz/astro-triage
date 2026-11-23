"""Método page: the Links section is read from references.json and rendered safely."""

from __future__ import annotations

import json

from test_dashboard_method_v2 import _method_html
from test_dashboard_phase4_sky2 import has_hand_written_digits

from astro_triage.dashboard import html_method as HM
from astro_triage.dashboard.html_system import NA

ANCHOR_ATTRS = 'target="_blank" rel="noopener noreferrer"'


def _file_refs():
    return json.loads(HM.REFERENCES_PATH.read_text(encoding="utf-8"))


def _point_to(monkeypatch, tmp_path, text):
    path = tmp_path / "refs.json"
    path.write_text(text, encoding="utf-8")
    monkeypatch.setattr(HM, "REFERENCES_PATH", path)


def test_seven_entries_rendered_in_order_with_safe_anchors():
    refs = _file_refs()
    out = HM.links_html()
    assert len(refs) == 7
    assert out.count("<li>") == 7
    positions = []
    for ref in refs:
        anchor = f'<a href="{ref["url"]}" {ANCHOR_ATTRS}>'
        assert anchor in out
        positions.append(out.index(anchor))
    assert positions == sorted(positions)


EXPECTED_LINKS = (
    ("Gaia DR3: resumo do release", "https://doi.org/10.1051/0004-6361/202243940"),
    ("Gaia DSC: classificação de fontes", "https://doi.org/10.1051/0004-6361/202243423"),
    (
        "Classificação de quasares e galáxias no Gaia DR2",
        "https://doi.org/10.1093/mnras/stz2947",
    ),
    ("Gaia DR3 (ESA): página do release", "https://www.cosmos.esa.int/web/gaia/dr3"),
    ("SDSS", "https://www.sdss.org/"),
    (
        "LightGBM: artigo",
        (
            "https://proceedings.neurips.cc/paper/2017/hash/"
            "6449f44a102fde848669bdd9eb6b76fa-Abstract.html"
        ),
    ),
    ("LightGBM: código", "https://github.com/lightgbm-org/LightGBM"),
)


def test_file_matches_the_verified_references_independently():
    """Pinned apart from the file: a wrong edit of a URL, a label or the order must fail.

    Each URL was checked against Crossref, arXiv or an HTTP 200 on 2026-11-23.
    """
    got = tuple((r["label"], r["url"]) for r in _file_refs())
    assert got == EXPECTED_LINKS


def test_reference_text_has_no_hand_written_numbers():
    """The file sits outside the digit scans of the .py modules: apply the same rule here."""
    for ref in _file_refs():
        for field in ("label", "note"):
            assert not has_hand_written_digits(ref[field]), (field, ref[field])


def test_file_urls_are_https_unique_and_fields_filled():
    refs = _file_refs()
    urls = [r["url"] for r in refs]
    assert all(u.startswith("https://") for u in urls)
    assert len(set(urls)) == len(urls)
    assert all(r["label"].strip() and r["note"].strip() for r in refs)


def test_text_and_attributes_are_escaped(monkeypatch, tmp_path):
    evil = [
        {
            "label": "<script>alert(1)</script>",
            "note": 'x" onmouseover="y<b>',
            "url": 'https://a.example/?q="><script>z</script>',
        }
    ]
    _point_to(monkeypatch, tmp_path, json.dumps(evil))
    out = HM.links_html()
    assert "<script>" not in out and "<b>" not in out
    assert '"><script' not in out
    assert "&lt;script&gt;" in out and "&quot;" in out


def test_non_https_and_malformed_entries_are_dropped(monkeypatch, tmp_path):
    entries = [
        {"label": "a", "note": "n", "url": "http://a.example/"},
        {"label": "b", "note": "n", "url": "javascript:alert(1)"},
        {"label": "c", "note": "n"},
        {"label": " ", "note": "n", "url": "https://c.example/"},
        "texto",
        {"label": "ok", "note": "n", "url": "https://ok.example/"},
    ]
    _point_to(monkeypatch, tmp_path, json.dumps(entries))
    out = HM.links_html()
    assert out.count("<li>") == 1 and "https://ok.example/" in out
    assert "javascript:" not in out and "http://a.example" not in out


def test_missing_file_invalid_json_or_empty_list_show_na(monkeypatch, tmp_path):
    monkeypatch.setattr(HM, "REFERENCES_PATH", tmp_path / "nao_existe.json")
    assert NA in HM.links_html() and "<a " not in HM.links_html()
    for text in ("{nao e json", "[]", '{"a": 1}', ""):
        _point_to(monkeypatch, tmp_path, text)
        out = HM.links_html()
        assert HM.LINKS_TITLE in out and NA in out and "<a " not in out


def test_method_page_shows_links_in_both_modes(monkeypatch):
    refs = _file_refs()
    for tech in (False, True):
        html = _method_html(monkeypatch, tech)
        assert all(f'<a href="{r["url"]}"' in html for r in refs)


def test_references_file_has_no_adr_acronym():
    text = HM.REFERENCES_PATH.read_text(encoding="utf-8")
    assert "ADR" not in text
