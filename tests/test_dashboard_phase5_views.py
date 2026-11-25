"""Phase 5 galaxies views: wording, no hand-written numbers."""

from __future__ import annotations

from pathlib import Path

import pytest

# Same detector and literal walker as the phase 4 sky sweep (imported, not copied).
from test_dashboard_phase4_sky2 import _literals, has_hand_written_digits

DASH = Path(__file__).resolve().parents[1] / "src" / "astro_triage" / "dashboard"
SWEPT = ["data_galaxies_z.py"]
KNOWN = {96, 1.6, 72.5, 95.8, 163, 93.9, 112571, 1016}


def _src(name: str) -> str:
    return (DASH / name).read_text(encoding="utf-8")


def _strings(name: str) -> list[str]:
    return [n.value for n in _literals(DASH / name) if isinstance(n.value, str)]


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
        and has_hand_written_digits(n.value)
    ]
    assert not bad, bad


def test_sweep_detector_flags_a_planted_metric():
    assert has_hand_written_digits("o modelo errou 93,9% das galáxias")
