"""Phase 5 galaxies views: wording, SDSS opt-in, no downloads, no hand-written numbers."""

from __future__ import annotations

import ast
import re
from pathlib import Path

import pytest

# Same detector and literal walker as the phase 4 sky sweep (imported, not copied).
from test_dashboard_phase4_sky2 import _literals, has_hand_written_digits

DASH = Path(__file__).resolve().parents[1] / "src" / "astro_triage" / "dashboard"
VIEWS = ["views_galaxies.py", "views_galaxies_gallery.py"]
SWEPT = [
    "data_galaxies_z.py",
    "charts_galaxies_rates.py",
    "charts_galaxies_scatter.py",
    *VIEWS,
]
# CSS value, not a data number: the detector flags the "1" in grayscale(1). Exempted on purpose.
CSS_ONLY = {"filter: grayscale(1);"}
KNOWN = {96, 1.6, 72.5, 95.8, 163, 93.9, 112571, 1016}


def _src(name: str) -> str:
    return (DASH / name).read_text(encoding="utf-8")


def _strings(name: str) -> list[str]:
    return [n.value for n in _literals(DASH / name) if isinstance(n.value, str)]


# (7) views
@pytest.mark.parametrize("name", VIEWS)
def test_view_text_has_no_azul_or_trace(name):
    for s in _strings(name):
        # whole word "trace"; "tracejada" (dashed) is legitimate Portuguese
        assert not re.search(r"azul|\btrace\b", s, re.IGNORECASE), s


def test_sdss_toggle_is_off_by_default():
    src = _src("views_galaxies_gallery.py")
    calls = [
        n
        for n in ast.walk(ast.parse(src))
        if isinstance(n, ast.Call) and getattr(n.func, "attr", "") == "toggle"
    ]
    assert len(calls) == 1
    kw = {k.arg: k.value for k in calls[0].keywords}
    assert isinstance(kw["value"], ast.Constant) and kw["value"].value is False


def test_colour_selector_defaults_to_monochrome_and_offers_original():
    src = _src("views_galaxies_gallery.py")
    assert re.search(r'MODES\s*=\s*\(\s*"Monocromático",\s*"Cor original"\s*\)', src)
    assert re.search(r"st\.radio\([^)]*MODES[^)]*index=0", src, re.DOTALL)


def test_monochrome_uses_css_grayscale_filter():
    assert "filter: grayscale" in _src("views_galaxies_gallery.py")


def test_gallery_module_never_downloads_or_writes_images():
    tree = ast.parse(_src("views_galaxies_gallery.py"))
    banned_mod = {"requests", "urllib", "httpx", "http", "shutil", "pathlib", "PIL"}
    for n in ast.walk(tree):
        if isinstance(n, ast.Import):
            assert not banned_mod & {a.name.split(".")[0] for a in n.names}
        if isinstance(n, ast.ImportFrom):
            assert (n.module or "").split(".")[0] not in banned_mod
        if isinstance(n, ast.Call):
            f = n.func
            name = f.id if isinstance(f, ast.Name) else getattr(f, "attr", "")
            assert name not in {
                "open", "urlopen", "urlretrieve", "write_bytes", "write",
                "write_text", "download_button", "save",
            }, name  # fmt: skip


# (8) nothing hand-written
@pytest.mark.parametrize("name", SWEPT)
def test_no_known_metric_literals(name):
    bad = [
        (n.lineno, n.value)
        for n in _literals(DASH / name)
        if isinstance(n.value, (int, float))
        and not isinstance(n.value, bool)
        and n.value in KNOWN
    ]
    assert not bad, bad


@pytest.mark.parametrize("name", SWEPT)
def test_no_hand_written_digits_in_strings(name):
    bad = [
        (n.lineno, n.value[:70])
        for n in _literals(DASH / name)
        if isinstance(n.value, str)
        and n.value not in CSS_ONLY
        and has_hand_written_digits(n.value)
    ]
    assert not bad, bad


def test_sweep_detector_flags_a_planted_metric():
    assert has_hand_written_digits("o modelo errou 93,9% das galáxias")
