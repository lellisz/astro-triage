"""Phase 1b: sidebar navigation, global rule/prior/technical state and the seal on every page.

The app runs under ``AppTest`` against a synthetic release in ``tmp_path`` (nothing is read from
or written to the repository's ``data/``). The page bodies are stubbed: only the shell is tested.
What needs a real browser (menu cut at 1440 and 390, 44 px targets, visible focus) is not here.
"""

from __future__ import annotations

import ast
import json
import re
from pathlib import Path

import pandas as pd
import pytest
from test_dashboard_phase4_sky2 import has_hand_written_digits
from test_dashboard_phase7_app import sidebar_rel

from astro_triage.dashboard import data as D
from astro_triage.dashboard import data_cached as DC
from astro_triage.dashboard import (
    html_overview,
    html_system,
    paths,
    sidebar,
    ui_state,
    views_dl,
    views_galaxies,
    views_method,
    views_overview,
    views_sky,
)
from astro_triage.dashboard.ui_state import Rule
from astro_triage.dashboard.views_common import tech_on
from astro_triage.evaluation.dl_compare import P3_BANDS

DASH_DIR = Path(D.__file__).resolve().parent
APP = DASH_DIR / "app.py"
EXPECTED_PAGES = [
    "Visão geral",
    "Mapa do céu",
    "Galáxias",
    "Comparação com DL",
    "Método",
]
OTHER_PAGE = "Método"
BODY = "CORPO:"
# Occurrences of internal Streamlit selectors in dashboard sources; the shell may not add any.
INTERNAL_SELECTORS = {"style_kit.py": 1, "theme.py": 21}


# ------------------------------------------------------------------------- pure state
def test_normalize_defaults_and_stale_values() -> None:
    assert ui_state.normalize({}) == ui_state.DEFAULT
    stale = {"ui_rule": "gone", "ui_prior": "P9", "ui_tech": "yes"}
    assert ui_state.normalize(stale) == ui_state.DEFAULT
    assert ui_state.normalize({"ui_rule": ["x"]}).rule is Rule.SHIPPED
    got = ui_state.normalize(
        {"ui_rule": "prior_matched", "ui_prior": "P3", "ui_tech": True}
    )
    assert got == ui_state.UiState(Rule.MATCHED, "P3", True)
    assert ui_state.normalize({"ui_rule": Rule.MATCHED}).rule is Rule.MATCHED


def test_shipped_rule_only_enables_shipped_prior() -> None:
    assert ui_state.enabled_priors(Rule.SHIPPED) == (ui_state.SHIPPED_PRIOR,)
    assert ui_state.enabled_priors(Rule.MATCHED) == tuple(ui_state.PRIORS)
    forced = ui_state.normalize({"ui_rule": "as_shipped", "ui_prior": "P3"})
    assert forced.prior == ui_state.SHIPPED_PRIOR
    kept = ui_state.normalize({"ui_rule": "prior_matched", "ui_prior": "P3"})
    assert kept.prior == "P3"


def test_disabled_note_mentions_shipped_prior() -> None:
    other = [p for p in ui_state.PRIORS if p != ui_state.SHIPPED_PRIOR]
    assert other
    for prior in other:
        note = ui_state.disabled_note(Rule.SHIPPED, prior)
        assert note == f"só existe em {ui_state.SHIPPED_PRIOR}"
        assert ui_state.disabled_note(Rule.MATCHED, prior) is None
    assert ui_state.disabled_note(Rule.SHIPPED, ui_state.SHIPPED_PRIOR) is None


def test_p3_is_partial_and_has_notice() -> None:
    assert ui_state.is_partial("P3")
    assert not ui_state.is_partial(ui_state.SHIPPED_PRIOR)
    notice = sidebar.partial_notice("P3")
    assert notice is not None
    for band in P3_BANDS:
        assert f"{band}" in notice.replace(";", " ").replace(")", " ").split()
    assert html_overview.adr_split_ref() in notice
    assert sidebar.partial_notice(ui_state.SHIPPED_PRIOR) is None


def test_rule_prior_line_names_rule_and_prior() -> None:
    line = ui_state.rule_prior_line(ui_state.UiState(Rule.MATCHED, "P1", False))
    assert line.startswith(ui_state.LINE_LABEL)
    assert ui_state.RULE_LABEL[Rule.MATCHED] in line and "P1" in line


# ------------------------------------------------------------------------- the app
@pytest.fixture
def shell(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Factory of a running ``AppTest`` over a synthetic release with stubbed page bodies."""
    import streamlit as st
    from streamlit.testing.v1 import AppTest

    interim = tmp_path / "interim"
    results = interim / "rel" / "eval" / "results.json"
    results.parent.mkdir(parents=True)
    results.write_text(json.dumps({"meta": {}}), encoding="utf-8")
    monkeypatch.setenv(paths.ENV_VAR, str(tmp_path))
    monkeypatch.setattr(D, "INTERIM_DIR", interim)
    monkeypatch.setattr(D, "load_release", lambda name, base=None: sidebar_rel(None, 0))
    monkeypatch.setattr(D, "pixel_table", lambda rel: pd.DataFrame())
    monkeypatch.setattr(DC, "load_layer2", lambda rel: None, raising=False)

    def stub(name: str):
        def render(*_args: object) -> None:
            tech = tech_on()
            st.markdown(f"{BODY}{name}:tech={tech}")

        return render

    monkeypatch.setattr(views_overview, "overview", stub("overview"))
    monkeypatch.setattr(views_sky, "sky_page", stub("sky"))
    monkeypatch.setattr(views_galaxies, "galaxies_page", stub("galaxies"))
    monkeypatch.setattr(views_dl, "dl_page", stub("dl"))
    monkeypatch.setattr(views_method, "method_page", stub("method"))
    st.cache_resource.clear()
    st.cache_data.clear()

    def start() -> AppTest:
        return AppTest.from_file(str(APP), default_timeout=60).run()

    yield start
    st.cache_resource.clear()
    st.cache_data.clear()


def open_page(at, title: str):
    """``AppTest.switch_page`` only takes file-backed pages; callable pages go by page hash."""
    page_hash = next(
        h for h, info in at._registered_pages.items() if info["page_name"] == title
    )
    at._page_hash = page_hash
    return at.run()


def titles(at) -> list[str]:
    return [info["page_name"] for info in at._registered_pages.values()]


def main_text(at) -> str:
    return "\n".join(m.value for m in at.main.markdown)


def prior_widget(at):
    (widget,) = [w for w in at.button_group if w.key == sidebar.W_PRIOR]
    return widget


def choose_rule(at, rule: Rule):
    at.radio(sidebar.W_RULE).set_value(rule.value).run()


def test_navigation_has_five_pages_in_order(shell) -> None:
    at = shell()
    assert not at.exception
    assert titles(at) == EXPECTED_PAGES
    assert "Camada 2" not in titles(at)


def test_rule_prior_survive_page_switch(shell) -> None:
    at = shell()
    choose_rule(at, Rule.MATCHED)
    prior_widget(at).set_value("P3").run()
    assert at.session_state[ui_state.KEY_RULE] == Rule.MATCHED.value
    assert at.session_state[ui_state.KEY_PRIOR] == "P3"
    line = ui_state.rule_prior_line(ui_state.UiState(Rule.MATCHED, "P3", False))
    assert line in main_text(at)
    at = open_page(at, OTHER_PAGE)
    assert not at.exception
    assert f"{BODY}method" in main_text(at)
    assert at.session_state[ui_state.KEY_RULE] == Rule.MATCHED.value
    assert at.session_state[ui_state.KEY_PRIOR] == "P3"
    assert line in main_text(at)
    assert at.radio(sidebar.W_RULE).value == Rule.MATCHED.value
    assert prior_widget(at).value == "P3"


def test_technical_mode_survives_page_switch_and_reaches_views(shell) -> None:
    at = shell()
    assert "tech=False" in main_text(at)
    at.toggle(sidebar.W_TECH).set_value(True).run()
    assert "tech=True" in main_text(at)
    at = open_page(at, OTHER_PAGE)
    assert "tech=True" in main_text(at)
    assert at.session_state[ui_state.KEY_TECH] is True


@pytest.mark.parametrize("title", EXPECTED_PAGES)
def test_seal_on_every_page(shell, title: str) -> None:
    at = shell()
    at = open_page(at, title)
    assert not at.exception
    assert html_system.SEAL_PREFIX in main_text(at)
    assert not at.sidebar.children
    assert ui_state.LINE_LABEL in main_text(at)


def test_shipped_rule_disables_other_priors_with_note(shell) -> None:
    at = shell()
    widget = prior_widget(at)
    assert widget.disabled and widget.value == ui_state.SHIPPED_PRIOR
    note = f"só existe em {ui_state.SHIPPED_PRIOR}"
    captions = [c.value for c in at.caption]
    assert any(note in c for c in captions), captions
    choose_rule(at, Rule.MATCHED)
    assert not prior_widget(at).disabled
    assert not any(note in c.value for c in at.caption)
    choose_rule(at, Rule.SHIPPED)
    assert (
        prior_widget(at).disabled and prior_widget(at).value == ui_state.SHIPPED_PRIOR
    )


def test_leaving_p3_for_shipped_rule_corrects_the_prior(shell) -> None:
    at = shell()
    choose_rule(at, Rule.MATCHED)
    prior_widget(at).set_value("P3").run()
    choose_rule(at, Rule.SHIPPED)
    assert prior_widget(at).value == ui_state.SHIPPED_PRIOR
    assert ui_state.rule_prior_line(ui_state.DEFAULT) in main_text(at)


def test_p3_shows_a_single_partial_notice_in_content(shell) -> None:
    at = shell()
    notice = sidebar.partial_notice("P3")
    assert notice
    assert notice not in main_text(at)
    choose_rule(at, Rule.MATCHED)
    prior_widget(at).set_value("P3").run()
    assert notice not in main_text(at)  # the overview body (stubbed here) owns it
    at = open_page(at, OTHER_PAGE)
    assert main_text(at).count(notice) == 1
    assert notice not in [c.value for c in at.caption]
    for band in P3_BANDS:
        assert band in notice
    assert "ADR" in notice


def test_empty_interim_does_not_break_the_selector(shell, tmp_path: Path) -> None:
    (tmp_path / "interim" / "rel" / "eval" / "results.json").unlink()
    at = shell()
    assert not at.exception
    assert at.selectbox[0].options == [D.DEFAULT_RELEASE]


# --------------------------------------------------------------------------- sources
def test_no_new_internal_streamlit_selectors() -> None:
    found = {
        p.name: len(re.findall(r"data-testid=|st-emotion-cache", p.read_text("utf-8")))
        for p in DASH_DIR.glob("*.py")
    }
    found = {name: n for name, n in found.items() if n}
    for name, n in found.items():
        assert n <= INTERNAL_SELECTORS.get(name, 0), name


def test_shell_does_not_swap_the_page_selector() -> None:
    tree = ast.parse(APP.read_text(encoding="utf-8"))
    calls = {
        getattr(n.func, "attr", "") for n in ast.walk(tree) if isinstance(n, ast.Call)
    }
    assert "navigation" in calls and "segmented_control" not in calls
    assert "tabs" not in {
        n.attr for n in ast.walk(tree) if isinstance(n, ast.Attribute)
    }


@pytest.mark.parametrize("name", ["sidebar.py", "ui_state.py", "paths.py"])
def test_shell_text_has_no_hand_written_digits(name: str) -> None:
    tree = ast.parse((DASH_DIR / name).read_text(encoding="utf-8"))
    bad = [
        n.value
        for n in ast.walk(tree)
        if isinstance(n, ast.Constant)
        and isinstance(n.value, str)
        and has_hand_written_digits(n.value)
    ]
    assert not bad, bad
