"""Overview headline and galaxies note of the web export: keys by verdict, no digits, no n/a invention."""

from __future__ import annotations

import dataclasses
import json
import re
from pathlib import Path

import pytest
from test_dashboard_dl import _digit_literals
from test_web_export_parity import NAME, build_release

from astro_triage.dashboard import compare_model as CM
from astro_triage.dashboard import compare_text as CT
from astro_triage.dashboard import paths
from astro_triage.dashboard.data_dl import PairVerdict
from astro_triage.web_export import __main__ as export
from astro_triage.web_export import pages, web_text
from astro_triage.web_export.types import clean

TEXT_FIELDS = ("question", "title", "lead", "caveat", "summary_en")


@pytest.fixture
def synthetic(tmp_path, monkeypatch):
    build_release(tmp_path)
    monkeypatch.setenv(paths.ENV_VAR, str(tmp_path))
    return export._load(NAME)


def _headline(rel) -> dict:
    return clean(pages.overview(rel, None))["headline"]


def _force_pairs(monkeypatch, verdicts: tuple[PairVerdict, ...]) -> None:
    real = CM.build

    def build(results):
        model = real(results)
        nets = [n for n, _ in model.pairs]
        return dataclasses.replace(model, pairs=tuple(zip(nets, verdicts, strict=True)))

    monkeypatch.setattr(CM, "build", build)


def test_a_winning_pair_gives_network_wins(synthetic) -> None:
    h = _headline(synthetic)
    assert h["key"] == web_text.KEY_WIN == "network_wins"
    assert h["title"] == web_text.TITLES[web_text.KEY_WIN]
    assert h["caveat"] is None
    assert [p["net"] for p in h["pairs"]] == ["MLP", "FTT"]
    assert h["pairs"][1]["text"] == CT.PAIR_WORD[PairVerdict.WIN]


def test_all_ties_give_no_detectable_difference(synthetic, monkeypatch) -> None:
    _force_pairs(monkeypatch, (PairVerdict.TIE, PairVerdict.TIE))
    h = _headline(synthetic)
    assert h["key"] == "no_detectable_difference"
    assert h["title"] == "Nenhuma rede venceu o LightGBM"
    assert CT.POWER_PHRASE in h["caveat"]
    assert "MLP" in h["lead"] and "FTT" in h["lead"]
    assert [p["text"] for p in h["pairs"]] == [CT.PAIR_WORD[PairVerdict.TIE]] * 2
    # The owner's wording: a missing difference is never called a tie on the Overview.
    shown = " ".join([h["title"], h["lead"], h["caveat"], *(p["text"] for p in h["pairs"])])
    assert "empate" not in shown.lower()


def test_any_win_beats_ties(synthetic, monkeypatch) -> None:
    _force_pairs(monkeypatch, (PairVerdict.WIN, PairVerdict.TIE))
    assert _headline(synthetic)["key"] == "network_wins"


def test_missing_results_dl_is_unavailable(tmp_path, monkeypatch) -> None:
    build_release(tmp_path, dl=False)
    monkeypatch.setenv(paths.ENV_VAR, str(tmp_path))
    h = _headline(export._load(NAME))
    assert h["key"] == "unavailable"
    assert h["pairs"] == [] and h["caveat"] is None
    assert h["lead"] and h["title"] == web_text.TITLES[web_text.KEY_NA]


def test_unreadable_results_dl_is_unavailable(synthetic) -> None:
    broken = synthetic.root / "dl" / "eval" / "results_dl.json"
    broken.write_text("{not json", encoding="utf-8")
    h = _headline(synthetic)
    assert h["key"] == "unavailable" and h["pairs"] == []


def test_fixed_texts_have_no_digit_literals() -> None:
    assert list(_digit_literals(Path(web_text.__file__))) == []


def test_exported_texts_have_no_digits(synthetic, monkeypatch) -> None:
    _force_pairs(monkeypatch, (PairVerdict.TIE, PairVerdict.TIE))
    h = _headline(synthetic)
    shown = [h[f] for f in TEXT_FIELDS if h[f]] + [p["text"] for p in h["pairs"]]
    for text in shown:
        assert not re.search(r"\d", re.sub(r"[A-Za-z_]+\d+", "", text)), text


def test_contrast_uses_the_shipped_row_and_formats_from_it(synthetic) -> None:
    c = _headline(synthetic)["contrast"]
    assert c["accuracy"]["text"] in c["text"]
    assert c["qso_purity"]["text"] in c["text"]
    assert c["qso_purity"]["point"] == pytest.approx(0.31)


def test_contrast_text_is_na_when_a_value_is_missing() -> None:
    assert web_text.contrast_text("n/a", "31,0%") == "n/a"
    assert web_text.contrast_text("72,5%", "n/a") == "n/a"
    rows = [
        {
            "rule": "as_shipped",
            "prior": pages.ui_state.SHIPPED_PRIOR,
            "accuracy": {"value": None, "text": "n/a"},
            "purity": {"point": None, "text": "n/a"},
        }
    ]
    assert pages._contrast(rows)["text"] == "n/a"


def test_headline_carries_no_per_object_id(synthetic) -> None:
    assert "source_id" not in json.dumps(_headline(synthetic))


def test_galaxies_note_is_fixed_and_digit_free(synthetic) -> None:
    note = clean(pages.galaxies(synthetic))["error_rows_note"]
    assert note == web_text.ERROR_ROWS_NOTE
    assert not re.search(r"\d", note)


def test_contrast_connective_follows_the_values() -> None:
    low = web_text.contrast_text("96,4%", "1,6%")
    assert "mas só" in low
    high = web_text.contrast_text("96,4%", "98,0%", purity_lower=False)
    assert "só" not in high and "mas" not in high
    assert pages._below(0.016, 0.964) and not pages._below(0.98, 0.964)
    assert not pages._below(None, 0.5) and not pages._below(0.5, None)
