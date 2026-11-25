"""Dashboard phase-1 fixes: floor sentence, paired ticks, calibration legend, limits, thresholds."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from astro_triage.dashboard import limits_pt

ROOT = Path(__file__).resolve().parents[1]


# 2. paired_dtick / paired_diffs


# 3. calibration


# 4. limits_pt
def test_limits_keys_are_nonempty_strings():
    assert limits_pt.LIMITS_PT
    assert all(isinstance(k, str) and k.strip() for k in limits_pt.LIMITS_PT)


def test_translate_known_and_unknown():
    key = next(iter(limits_pt.LIMITS_PT))
    assert limits_pt.translate(key) == (limits_pt.LIMITS_PT[key], True)
    assert limits_pt.translate("never seen text") == ("never seen text", False)


@pytest.mark.parametrize("release", ["full_v0", "dataset_100k_v0"])
def test_every_limitation_in_results_json_is_translated(release):
    path = ROOT / "data" / "interim" / release / "eval" / "results.json"
    if not path.exists():
        pytest.skip(f"{path} missing")
    with open(path, encoding="utf-8") as fh:
        items = json.load(fh).get("limitations", [])
    missing = [t[:80] for t in items if not limits_pt.translate(t)[1]]
    assert not missing, missing
