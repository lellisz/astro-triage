"""Dashboard polish fixes (V1-V6): small guards for the rules the redesign relies on."""

from __future__ import annotations

from astro_triage.dashboard import verdict_rule as VR


def test_v4_method_box_states_the_full_win_rule_and_no_hand_written_result():
    text = " ".join(VR.rule_lines())
    assert "maior que δ" in text and "limite inferior" in text
    assert "maior que zero" in text and "nas duas classes" in text
    assert "todos os pares" not in text and "permanece" not in text


def test_v5_fragments_of_every_tab_carry_the_page_language(monkeypatch):
    from astro_triage.dashboard import views_method, views_overview, views_sky
    from astro_triage.dashboard.html_system import PAGE_LANG

    sent: list[str] = []
    for mod, name in (
        (views_overview, "_md"),
        (views_sky, "_md"),
        (views_method, "_html"),
    ):
        monkeypatch.setattr(mod.st, "markdown", lambda html, **_: sent.append(html))
        getattr(mod, name)("<p>x</p>")
    assert len(sent) == 3
    assert all(f"lang='{PAGE_LANG}'" in html for html in sent)


def test_v6_header_gets_a_solid_background_on_narrow_screens():
    from astro_triage.dashboard import theme

    css = theme.css()
    assert f"max-width: {theme.NARROW_SCREEN_PX}px" in css
    rule = css.split(f"max-width: {theme.NARROW_SCREEN_PX}px", 1)[1].split("}}", 1)[0]
    assert "stHeader" in rule and theme.BG in rule


def test_v1_data_dl_does_not_depend_on_data_headline():
    import subprocess
    import sys

    code = (
        "import sys, astro_triage.dashboard.data_dl as m;"
        "sys.exit(int('astro_triage.dashboard.data_headline' in sys.modules))"
    )
    import os
    from pathlib import Path

    src = Path(__file__).resolve().parents[1] / "src"
    env = {**os.environ, "PYTHONPATH": str(src)}
    done = subprocess.run([sys.executable, "-c", code], env=env, check=False)
    assert done.returncode == 0
