"""Phase 9: no tab of the dashboard shows a decision-record citation (``ADR``), in either mode.

The sweep runs the real app under ``AppTest`` over a synthetic release built in ``tmp_path`` by
the project's own evaluation runner (so ``results.json.limitations`` is the producer's
``LIMITATIONS``, with the ``ADR 000N`` citations in it) plus a ``results_dl.json`` and a
``results_dl.md`` whose limitations carry the ``B0 (ADR 0006 section 6)`` line. Nothing is read
from or written to the repository's ``data/``.
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path

import pytest
from test_dashboard_comparacao import _results as dl_results
from test_dashboard_shell import APP, EXPECTED_PAGES, open_page
from test_evaluation import make_data
from test_evaluation import run as run_evaluation

from astro_triage.dashboard import data as D
from astro_triage.dashboard import html_method as HM
from astro_triage.dashboard import limits_pt as LP
from astro_triage.dashboard import paths, sidebar
from astro_triage.evaluation import runner as R

ADR = re.compile(r"\bADRs?\b")
RELEASE = "rel"
B0_LINE = (
    "- B0 (ADR 0006 section 6): zero hyperparameter search for both families; "
    "the deep models use the grid frozen on val."
)
# Elements whose text the reader sees; anything else is scanned through its serialized proto.
TEXT_ATTRS = ("value", "label", "body", "help", "placeholder")


# --------------------------------------------------------------------------- the sweep
def _walk(node):
    """Every element under an ``AppTest`` block, containers included."""
    for child in getattr(node, "children", {}).values():
        yield child
        yield from _walk(child)


def visible_text(at) -> str:
    """Markdown, expander and widget labels, captions, alerts and rendered HTML of the page."""
    chunks: list[str] = []
    for root in (at.main, at.sidebar):
        for el in _walk(root):
            for attr in TEXT_ATTRS:
                got = getattr(el, attr, None)
                if isinstance(got, str):
                    chunks.append(got)
            proto = getattr(el, "proto", None)
            if proto is not None:
                chunks.append(str(proto))
    return "\n".join(chunks)


def adr_hits(at) -> list[str]:
    """Each citation found, with a little context around it."""
    text = visible_text(at)
    return [text[max(0, m.start() - 40) : m.end() + 40] for m in ADR.finditer(text)]


@pytest.fixture(scope="module")
def release_root(tmp_path_factory: pytest.TempPathFactory) -> Path:
    root = tmp_path_factory.mktemp("no_adr")
    rel = root / "interim" / RELEASE
    ds = make_data(per_band=3, counts={"STAR": 2, "GALAXY": 2, "QSO": 2})
    n = len(ds.labels)
    classes = ds.labels["sdss_class"]
    ds.labels["sdss_z"] = [
        0.0 if c == "STAR" else 0.1 + 0.01 * (i % 40) for i, c in enumerate(classes)
    ]
    ds.labels["ra"] = [(7.0 * i) % 360.0 for i in range(n)]
    ds.labels["dec"] = [(3.0 * i) % 120.0 - 60.0 for i in range(n)]
    ds.labels["l"] = [(11.0 * i) % 360.0 for i in range(n)]
    for k, entry in enumerate(ds.report["models"]):
        entry.update(
            n_features=3 + k,
            best_iteration=10,
            temperature=1.0 + k,
            gain_top10=[{"feature": f"phot_g_mean_mag_{k}", "share": 1.0 - 0.1 * k}],
            val_calibration_fitted_here={"accuracy": 0.9, "macro_f1": 0.9},
        )
    noind = ds.preds["lgbm_noind"]  # distinct from lgbm_full: equal charts share an element id
    noind["p_cal_star"] = noind["p_cal_star"] * 0.9 + 0.02
    noind["p_cal_galaxy"] = noind["p_cal_galaxy"] * 0.9 + 0.03
    noind["p_cal_qso"] = noind["p_cal_qso"] * 0.9 + 0.05  # rows still sum to 1
    for pred in ds.preds.values():
        for c in ("star", "galaxy", "qso"):
            pred[f"p_raw_{c}"] = pred[f"p_cal_{c}"]
    R.write_outputs(run_evaluation(ds, rel, n_boot=10), rel)
    dl_eval = rel / "dl" / "eval"
    dl_eval.mkdir(parents=True)
    (dl_eval / "results_dl.json").write_text(json.dumps(dl_results()), encoding="utf-8")
    (dl_eval / "results_dl.md").write_text(
        f"# results\n\n## Limitations\n\n{B0_LINE}\n- Second note, no citation.\n",
        encoding="utf-8",
    )
    return root


@pytest.fixture
def start(release_root: Path, monkeypatch: pytest.MonkeyPatch):
    """Factory of a running ``AppTest`` over the synthetic release."""
    import streamlit as st
    from streamlit.testing.v1 import AppTest

    interim = release_root / "interim"
    monkeypatch.setenv(paths.ENV_VAR, str(release_root))
    monkeypatch.setattr(D, "INTERIM_DIR", interim)
    st.cache_resource.clear()
    st.cache_data.clear()

    def run() -> AppTest:
        return AppTest.from_file(str(APP), default_timeout=120).run()

    yield run
    st.cache_resource.clear()
    st.cache_data.clear()


def render(start, title: str, technical: bool):
    at = start()
    if technical:
        at.toggle(sidebar.W_TECH).set_value(True).run()
    at = open_page(at, title)
    return at


def test_artifacts_really_carry_the_citations(release_root: Path) -> None:
    """Without these inputs the sweep below would pass for the wrong reason."""
    res = json.loads(
        (release_root / "interim" / RELEASE / "eval" / "results.json").read_text("utf-8")
    )
    assert res["limitations"] == list(R.LIMITATIONS)
    assert any(ADR.search(t) for t in res["limitations"])
    md = (release_root / "interim" / RELEASE / "dl/eval/results_dl.md").read_text("utf-8")
    assert "ADR 0006" in md


@pytest.mark.parametrize("technical", [False, True], ids=["simple", "technical"])
@pytest.mark.parametrize("title", EXPECTED_PAGES)
def test_no_tab_shows_an_adr_citation(start, title: str, technical: bool) -> None:
    at = render(start, title, technical)
    assert not at.exception, [e.value for e in at.exception]
    assert not at.error, [e.value for e in at.error]  # a page error would hide its text
    assert at.toggle(sidebar.W_TECH).value is technical
    assert visible_text(at).strip(), "nothing was rendered"
    assert adr_hits(at) == []


@pytest.mark.parametrize("technical", [False, True], ids=["simple", "technical"])
def test_sweep_has_detection_power(
    start, monkeypatch: pytest.MonkeyPatch, technical: bool
) -> None:
    """The same sweep fails when a tab prints a citation: guards against a vacuous pass."""
    monkeypatch.setattr(HM, "links_html", lambda: "<p>Origem: ADR 0004.</p>")
    at = render(start, "Método", technical)
    assert not at.exception
    hits = adr_hits(at)
    assert hits and "ADR 0004" in hits[0]


def test_method_page_does_render_the_limitations(start) -> None:
    """The artifact's limitations reach the Método page (translated or filtered)."""
    at = render(start, "Método", True)
    text = visible_text(at)
    assert "crossmatch" in text.lower()
    assert "ADR" not in text


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
