"""Galáxias, phase 5: prior rule, bars, GALAXY verdict cards (rule of phase 4), floor, payload.

Also the three open items of phase 4: unreadable DL artifact, the verdict-text sweep over more
modules, and a cell in the producer's format (not reportable, token "tie").
"""

from __future__ import annotations

import copy
import json
import re
from pathlib import Path
from types import SimpleNamespace

import pytest
from test_dashboard_comparacao import TIE, _results
from test_dashboard_galaxy_prior import (
    NARROW_CELL,
    NOTE_FULL,
    block,
    error_release,
    fake_release,
    nets,
)

from astro_triage.dashboard import compare_model as CM
from astro_triage.dashboard import compare_text as CT
from astro_triage.dashboard import data_dl as DD
from astro_triage.dashboard import data_galaxy_prior as gp
from astro_triage.dashboard import html_galaxy_prior as hg
from astro_triage.dashboard import html_system as hs
from astro_triage.dashboard import ui_state, views_dl, views_galaxies
from astro_triage.dashboard import verdict_rule as VR
from astro_triage.dashboard.data_galaxies import GALAXY_G_EDGES, error_rate_table
from astro_triage.dashboard.data_galaxies_z import error_rate_table_z, galaxy_totals

V = gp.VerdictKey
DASH = Path(views_galaxies.__file__).parent
STATE = ui_state.DEFAULT
TECH = ui_state.UiState(ui_state.Rule.SHIPPED, ui_state.SHIPPED_PRIOR, True)


# prior sentence ---------------------------------------------------------------------
def test_sentence_ignores_p3():
    for p3 in ((0.0, 0.01), (0.99, 1.0), (None, None)):
        for p2, want in (((0.7, 0.9), V.DEPENDS), ((0.5, 0.9), V.NO_DIFFERENCE)):
            assert gp.galaxy_verdict(block(p2=p2, p3=p3), nets()) is want
    assert gp.VERDICT_PRIORS == ("P1", "P2")


def test_sentence_claims_dependence_only_if_p1_p2_disjoint():
    overlapping = block(p1=(0.4, 0.75), p2=(0.7, 0.9), p3=(0.0, 0.05))
    assert gp.prior_depends(overlapping) is False
    assert (
        hg.verdict_text(gp.galaxy_verdict(overlapping, nets()))
        == hg.VERDICT_TEXT[V.NO_DIFFERENCE]
    )
    assert "depende" not in hg.VERDICT_TEXT[V.NO_DIFFERENCE]
    assert gp.prior_depends(block()) is True


# rows and floor ---------------------------------------------------------------------
def _both_blocks():
    rel = fake_release()
    node = rel.results["populations"]["a_all"]["priors"]["P1"]
    for n in rel.results["populations"]["a_all"]["priors"].values():
        n["variants"]["prior_matched"] = copy.deepcopy(node["variants"]["as_shipped"])
    return rel


def test_galaxy_rows_four_rows_in_overview_order_and_one_selected():
    rel = _both_blocks()
    shipped = gp.load_galaxy_prior_block(rel, "as_shipped")
    matched = gp.load_galaxy_prior_block(rel, "prior_matched")
    purity, completeness = gp.galaxy_rows(shipped, matched, STATE)
    assert len(purity) == len(completeness) == 4
    assert [r.label for r in purity] == [r.label for r in completeness]
    assert [r.selected for r in purity] == [True, False, False, False]
    assert purity[-1].note is not None and purity[0].note is None  # P3 is partial
    assert all(r.value.point is not None and r.value.ci for r in purity)


def test_galaxy_rows_below_floor_are_na_values():
    rel = fake_release(n=1)
    blk = gp.load_galaxy_prior_block(rel)
    purity, completeness = gp.galaxy_rows(blk, blk, STATE)
    assert all(
        r.value.point is None and r.value.ci is None for r in (*purity, *completeness)
    )


def test_band_below_floor_is_na_without_bar():
    rel = error_release(n_far=9, floor=10)
    rows = gp.galaxy_error_rows(
        galaxy_totals(rel, "lgbm_full"),
        error_rate_table(rel, "lgbm_full", gp.GALAXY, "G", GALAXY_G_EDGES),
        error_rate_table_z(rel, "lgbm_full"),
        10,
    )
    out = hg.errors_html(rows)
    far = out[out.index(gp.FARTHEST_LABEL) :]
    assert hs.NA in far and "hs2-fill" not in far and "hs2-ci" not in far
    assert (
        out.count("hs2-fill") == 3
    )  # the other three bands keep their bar and interval
    assert out.count("hs2-ci") >= 3


# verdict cards: one source ---------------------------------------------------------
def _galaxy_cells(results):
    return gp.dl_galaxy_cells(results)


def test_two_galaxy_verdict_cards_reuse_phase4():
    results = _results(ftt=(TIE, "win", "tie"))
    cells = _galaxy_cells(results)
    model = CM.build(results)
    out = hg.verdict_cards_html(cells)
    for net, name in (("mlp_full", "MLP"), ("ftt_full", "FT-Transformer")):
        short = CM.net_name(net)
        g = next(x for x in model.grid if x.cls == "GALAXY" and x.net == short)
        assert CM.grid_cell(name, cells[net]).verdict == g.verdict
        assert hs.verdict_card(f"{name} × {CT.REFERENCE_NAME}", g.verdict) in out
    assert out.count("hs2-card") == 2 and CT.STATE_WORD[DD.ClassVerdict.TIE] in out
    assert out.count(CT.POWER_PHRASE) == sum(
        g.underpowered for g in model.grid if g.cls == "GALAXY"
    )


def test_verdict_cards_na_without_artifact_and_no_number_made_up():
    out = hg.verdict_cards_html(dict.fromkeys(gp.NETWORKS))
    assert out.count("hs2-card") == 2 and hs.NA in out
    assert not re.search(r"\d,\d", out)


def test_verdict_cards_do_not_retype_the_rule():
    text = (DASH / "html_galaxy_prior.py").read_text(encoding="utf-8")
    assert "dl_compare" not in text and "class_verdict(" not in text
    assert "VR." not in text and "verdict_rule" not in text


def test_inconsistent_artifact_is_data_error_not_a_card():
    results = _results(ftt=("X wins", "win", "tie"))
    with pytest.raises(DD.DlDataError):
        gp.dl_galaxy_cells(results)


# page payload -----------------------------------------------------------------------
def _page(monkeypatch, state, results, load_error=None):
    chunks: list[str] = []
    rel = error_release()
    full = _both_blocks()
    rel.results = full.results
    rel.results["limitations"] = [NOTE_FULL]
    rel.name = "full_v0"
    fake_st = SimpleNamespace(
        markdown=lambda t, **_: chunks.append(t),
        session_state={
            ui_state.KEY_TECH: state.tech,
            ui_state.KEY_RULE: state.rule,
            ui_state.KEY_PRIOR: state.prior,
        },
    )

    def load(_rel):
        if load_error:
            raise load_error
        return results

    monkeypatch.setattr(views_galaxies, "st", fake_st)
    monkeypatch.setattr(
        views_galaxies,
        "DC",
        SimpleNamespace(
            load_results_dl=load,
            error_rate_table=error_rate_table,
            error_rate_table_z=error_rate_table_z,
            galaxy_totals=galaxy_totals,
        ),
    )
    views_galaxies.galaxies_page(rel)
    return "\n".join(chunks)


@pytest.mark.parametrize("state", [STATE, TECH], ids=["simple", "technical"])
def test_payload_has_no_source_id_nor_scatter(monkeypatch, state):
    out = _page(monkeypatch, state, _results(ftt=(TIE, "win", "tie")))
    for banned in ("source_id", "plotly", "<svg", "<canvas", "scatter", "customdata"):
        assert banned not in out.lower()
    assert not re.search(r"\b(ra|dec)\b\s*[:=\[]", out)
    assert "hs-dots" not in out


def test_tables_only_in_technical_mode(monkeypatch):
    simple = _page(monkeypatch, STATE, None)
    tech = _page(monkeypatch, TECH, None)
    assert hg.TECH_DETAILS not in simple and "<details" not in simple
    assert hg.TECH_DETAILS in tech and "<details" in tech


def test_page_leaves_seal_and_rule_line_to_the_shell_and_has_unavailable_cards(
    monkeypatch,
):
    out = _page(monkeypatch, STATE, None)
    # app.shell_header already prints the seal and the rule/prior line on every tab
    assert ui_state.rule_prior_line(STATE) not in out
    assert out.count("hs2-card") >= 4  # purity, completeness, two verdicts


def test_unreadable_artifact_is_n_a_not_a_traceback(monkeypatch):
    err = json.JSONDecodeError("bad", "{", 1)
    out = _page(monkeypatch, STATE, None, load_error=err)
    assert views_galaxies.UNREADABLE in out and "Traceback" not in out


def test_no_comparison_with_other_classes(monkeypatch):
    out = _page(monkeypatch, TECH, _results(ftt=(TIE, "win", "tie")))
    text = re.sub(r"<[^>]+>", " ", out)
    for sentence in re.split(r"[.;]", text):
        low = sentence.lower()
        if "estrela" in low or "quasar" in low:
            assert not re.search(r"\b(mais|menos|maior|menor|que as?)\b", low), sentence


def test_crossmatch_rates_come_from_the_artifact_not_the_module(monkeypatch):
    out = _page(monkeypatch, STATE, None)
    assert "31,1" in out and "91,1" in out and "65,7" in out and hg.CROSSMATCH_ORIGIN in out
    assert "ADR" not in out
    for name in ("html_galaxy_prior.py", "data_galaxy_prior.py", "views_galaxies.py"):
        text = (DASH / name).read_text(encoding="utf-8")
        assert not re.search(r"\b(31|91|65)[.,]\d", text)


# open items of phase 4 ----------------------------------------------------------------
def test_dl_page_unreadable_artifact_is_a_data_error_without_numbers(monkeypatch):
    chunks: list[str] = []
    monkeypatch.setattr(
        views_dl, "st", SimpleNamespace(markdown=lambda t, **_: chunks.append(t))
    )

    def boom(_rel):
        raise json.JSONDecodeError("bad", "{", 1)

    monkeypatch.setattr(views_dl.DC, "load_results_dl", boom)
    views_dl.dl_page(SimpleNamespace(name="full_v0", root=Path("nowhere")))
    out = " ".join(chunks)
    assert views_dl.UNREADABLE in out and not re.search(r"\d,\d", out)


def test_no_verdict_text_typed_in_more_modules():
    from test_dashboard_compare_v2 import DASH as CMP_DASH

    na_word = CT.STATE_WORD[CM.ClassVerdict.NA]
    words = [*CT.STATE_WORD.values(), *CT.PAIR_WORD.values(), CT.POWER_PHRASE]
    names = (
        "views_dl.py",
        "compare_model.py",
        "views_galaxies.py",
        "html_galaxy_prior.py",
        "data_galaxy_prior.py",
    )
    assert "compare_text.py" not in names
    for name in names:
        text = (CMP_DASH / name).read_text(encoding="utf-8")
        for w in words:
            if w != na_word:
                assert f'"{w}' not in text, (name, w)


def test_producer_format_not_reportable_with_tie_token_is_na_through_headline_and_rule():
    raw = _results(ftt=(TIE, "win", "tie"))
    cell = raw["compare"]["pairs"]["ftt_full-lgbm_full"]["P2"]["0.80"]
    cell["classes"]["GALAXY"].update(
        {"reportable": False, "class_verdict": "tie", "na": False}
    )
    head = DD.headline(raw)
    ftt = next(c for c in head.cells if c.pair.startswith("ftt"))
    assert ftt.classes["GALAXY"].verdict is DD.ClassVerdict.NA
    for c in head.cells:
        VR.check(c)  # raises on a mismatch with the recomputed verdict
    assert VR.underpowered(ftt.classes["GALAXY"]) is False
    grid = CM.build(raw).grid
    assert (
        next(g for g in grid if g.cls == "GALAXY" and g.net == "FTT").state
        is DD.ClassVerdict.NA
    )


def _with_blocks(monkeypatch, shipped, matched):
    by_variant = {"as_shipped": shipped, "prior_matched": matched}
    monkeypatch.setattr(
        views_galaxies,
        "load_galaxy_prior_block",
        lambda _r, v="as_shipped": by_variant[v],
    )


OVERLAP = {"p1": (0.4, 0.75), "p2": (0.7, 0.9)}


@pytest.mark.parametrize(
    ("shipped", "matched", "want"),
    [
        (block(**OVERLAP), block(), V.DEPENDS),  # bars disjoint, as_shipped overlaps
        (
            block(),
            block(**OVERLAP),
            V.NO_DIFFERENCE,
        ),  # bars overlap, as_shipped disjoint
        (block(), block(), V.DEPENDS),
        (block(**OVERLAP), block(**OVERLAP), V.NO_DIFFERENCE),
    ],
)
def test_sentence_follows_the_p1_p2_intervals_the_bars_show(
    monkeypatch, shipped, matched, want
):
    _with_blocks(monkeypatch, shipped, matched)
    out = _page(monkeypatch, STATE, None)
    scope = f"{hg.VERDICT_SCOPE} {ui_state.RULE_LABEL[ui_state.Rule.MATCHED]}."
    assert f"{hg.verdict_text(want)} {scope}" in out
    other = V.NO_DIFFERENCE if want is V.DEPENDS else V.DEPENDS
    assert hg.verdict_text(other) not in out
    purity, _ = gp.galaxy_rows(shipped, matched, STATE)
    p1, p2 = (r.value.ci for r in purity[1:3])  # the "ajustada" P1 and P2 bars
    disjoint = p1.hi < p2.lo or p2.hi < p1.lo
    assert disjoint is (want is V.DEPENDS)


@pytest.mark.parametrize("prior", ["P1", "P3"])
def test_verdict_block_states_the_fixed_point_whatever_the_sidebar(monkeypatch, prior):
    state = ui_state.UiState(ui_state.Rule.MATCHED, prior, False)
    results = _results(ftt=(TIE, "win", "tie"))
    out = _page(monkeypatch, state, results)
    model = CM.build(results)
    fixed = CT.fixed_point_line(ui_state.LINE_LABEL, model.prior, model.level)
    assert ui_state.rule_prior_line(state) not in out  # printed once, by the shell
    verdict_block = out[out.index(hg.DELTA_TITLE) :].split(hg.ERRORS_TITLE)[0]
    assert hs.method_note(fixed) in verdict_block


def test_narrow_cell_is_not_flagged(monkeypatch):
    assert gp.power_networks(nets(NARROW_CELL, NARROW_CELL)) == ()
