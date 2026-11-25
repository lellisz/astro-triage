"""Phase 9: no tab of the dashboard shows a decision-record citation (``ADR``), in either mode.

The sweep runs the real app under ``AppTest`` over a synthetic release built in ``tmp_path`` by
the project's own evaluation runner (so ``results.json.limitations`` is the producer's
``LIMITATIONS``, with the ``ADR 000N`` citations in it) plus a ``results_dl.json`` and a
``results_dl.md`` whose limitations carry the ``B0 (ADR 0006 section 6)`` line. Nothing is read
from or written to the repository's ``data/``.
"""

from __future__ import annotations

import logging
import re

import pytest

from astro_triage.dashboard import limits_pt as LP
from astro_triage.evaluation import runner as R

ADR = re.compile(r"\bADRs?\b")
RELEASE = "rel"
# Elements whose text the reader sees; anything else is scanned through its serialized proto.


# --------------------------------------------------------------------------- the sweep
def _walk(node):
    """Every element under an ``AppTest`` block, containers included."""
    for child in getattr(node, "children", {}).values():
        yield child
        yield from _walk(child)


# ------------------------------------------------------------------------- limits_pt units
def test_drop_slot_consumes_a_number_and_prints_nothing() -> None:
    en = "Sensitivity sweep S has point estimates only (no bootstrap), per ADR 0003."
    pt, translated = LP.translate(en)
    assert translated and "0003" not in pt and "{" not in pt and "ADR" not in pt
    assert pt == (
        "A varredura de sensibilidade S tem apenas estimativas pontuais (sem bootstrap), "
        "conforme o protocolo de avaliação."
    )


def test_drop_slot_keeps_the_neighbouring_numbers() -> None:
    en = (
        "combmod has 3 of its 5 classes available; its star score is 1 - p_quasar - "
        "p_galaxy (absorbs WD and binary), ADR 0003 'Effect of the combmod embedded "
        "prior' item 3. The delivered classprob_dsc_combmod_star column is not used."
    )
    pt, translated = LP.translate(en)
    assert translated
    assert pt.startswith("O combmod tem 3 das suas 5 classes disponíveis; seu escore de estrela é 1 - ")
    assert "0003" not in pt and "item" not in pt and "ADR" not in pt


def test_every_model_has_as_many_slots_as_the_producer_has_numbers() -> None:
    """Each producer sentence that cites a record is translated (a mismatch silently falls back)."""
    cited = [t for t in R.LIMITATIONS if ADR.search(t)]
    assert cited
    for original in cited:
        pt, translated = LP.translate(original)
        assert translated and not ADR.search(pt), original
    for key, pt in LP.LIMITS_PT.items():
        assert key.count("{}") == len(LP._SLOT.findall(pt)), key


def test_slot_count_mismatch_falls_back_to_original_and_logs(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    monkeypatch.setitem(LP.LIMITS_PT, "Foo {} bar {}.", "Foo {}.")
    with caplog.at_level(logging.WARNING):
        assert LP.translate("Foo 1 bar 2.") == ("Foo 1 bar 2.", False)
    assert "do not fit" in caplog.text
    unknown = "Totally new limitation of 5 kinds."
    assert LP.translate(unknown) == (unknown, False)


@pytest.mark.parametrize(
    "text",
    [
        "",
        " ",
        "ADR",
        "ADRs",
        "ADR 0002",
        "per ADR 0003",
        "(ADR 0006 section 6)",
        "x (per ADR 0003; see 5) y",
        "in the ADRs: it is never shifted",
        "B0 (ADR 0006 section 6): zero hyperparameter search",
        "ADR 0004. Origem",
        "adr 0004 lowercase is not a citation",
        "ADRX and XADR are not citations",
        "ÁDR 0004 café ADR 0005 ñ",
        "(ADR 1) (ADR 12345)",
        "no citation at all.",
        "\n\tADR\n0004\n",
        "ADR " * 200,
        "(" * 50 + "ADR 0001" + ")" * 50,
    ],
)
def test_omit_record_refs_is_total_and_idempotent(text: str) -> None:
    once = LP.omit_record_refs(text)
    assert isinstance(once, str)
    assert LP.omit_record_refs(once) == once
    assert not ADR.search(once), once


def test_omit_record_refs_removes_only_the_citation() -> None:
    assert (
        LP.omit_record_refs("B0 (ADR 0006 section 6): zero search.")
        == "B0: zero search."
    )
    assert LP.omit_record_refs("Sweep S, per ADR 0003.") == "Sweep S."
    assert LP.omit_record_refs("not shifted in the ADRs: it") == "not shifted: it"
    assert LP.omit_record_refs("no citation at all.") == "no citation at all."
    assert LP.omit_record_refs("ADRX stays") == "ADRX stays"


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("defined ADRs: it is never shifted", "defined: it is never shifted"),
        ("foo ADR 0004 . bar", "foo. bar"),
        ("x (per ADR 0003) , y", "x, y"),
    ],
)
def test_omit_record_refs_closes_the_space_before_punctuation(text: str, expected: str) -> None:
    """The normalisation keeps the punctuation mark (a broken backreference replaced it)."""
    out = LP.omit_record_refs(text)
    assert out == expected
    assert not any(ord(c) < 32 for c in out if c not in "\n\t")


def test_omit_record_refs_never_raises_on_the_producer_texts() -> None:
    for original in R.LIMITATIONS:
        assert not ADR.search(LP.omit_record_refs(original))


def test_omit_record_refs_drops_the_whole_titled_citation() -> None:
    en = (
        "its star score is 1 - p_quasar - p_galaxy (absorbs WD and binary), ADR 0003 "
        "'Effect of the combmod embedded prior' item 3. The delivered "
        "classprob_dsc_combmod_star column is not used."
    )
    assert LP.omit_record_refs(en) == (
        "its star score is 1 - p_quasar - p_galaxy (absorbs WD and binary). The delivered "
        "classprob_dsc_combmod_star column is not used."
    )


def test_omit_record_refs_capitalizes_the_sentence_a_citation_opened() -> None:
    en = (
        "partial circularity. ADR 0002 cut (b) (outside DSC training): not available: "
        "DR14Q/DR15 not ingested."
    )
    assert LP.omit_record_refs(en) == (
        "partial circularity. Cut (b) (outside DSC training): not available: "
        "DR14Q/DR15 not ingested."
    )


def test_omit_record_refs_leaves_text_without_citation_byte_identical() -> None:
    for text in ("p < .05  and  double  spaces .", "  lead and trail  ", "a ,b . c"):
        assert LP.omit_record_refs(text) == text


def test_omit_record_refs_leaves_no_dangling_item_on_the_producer_texts() -> None:
    for original in R.LIMITATIONS:
        assert not re.search(r"\bitem \d+", LP.omit_record_refs(original)), original
