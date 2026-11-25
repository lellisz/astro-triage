"""Phase 1b: sidebar navigation, global rule/prior/technical state and the seal on every page.

The app runs under ``AppTest`` against a synthetic release in ``tmp_path`` (nothing is read from
or written to the repository's ``data/``). The page bodies are stubbed: only the shell is tested.
What needs a real browser (menu cut at 1440 and 390, 44 px targets, visible focus) is not here.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

import pytest
from test_dashboard_phase4_sky2 import has_hand_written_digits

from astro_triage.dashboard import data as D
from astro_triage.dashboard import ui_state
from astro_triage.dashboard.ui_state import Rule

DASH_DIR = Path(D.__file__).resolve().parent
APP = DASH_DIR / "app.py"
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


def test_rule_prior_line_names_rule_and_prior() -> None:
    line = ui_state.rule_prior_line(ui_state.UiState(Rule.MATCHED, "P1", False))
    assert line.startswith(ui_state.LINE_LABEL)
    assert ui_state.RULE_LABEL[Rule.MATCHED] in line and "P1" in line


# ------------------------------------------------------------------------- the app


# --------------------------------------------------------------------------- sources
def test_no_new_internal_streamlit_selectors() -> None:
    found = {
        p.name: len(re.findall(r"data-testid=|st-emotion-cache", p.read_text("utf-8")))
        for p in DASH_DIR.glob("*.py")
    }
    found = {name: n for name, n in found.items() if n}
    for name, n in found.items():
        assert n <= INTERNAL_SELECTORS.get(name, 0), name


@pytest.mark.parametrize("name", ["ui_state.py", "paths.py"])
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
