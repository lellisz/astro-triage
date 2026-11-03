"""read_mode of kaggle/run_kaggle.py: a bad mode file must stop the run, never mean phase1."""

import importlib.util
import json
import os
from pathlib import Path

import pytest

DEFAULT = Path(__file__).resolve().parents[1] / "kaggle" / "run_kaggle.py"


@pytest.fixture
def rk(tmp_path, monkeypatch):
    path = Path(os.environ.get("KAGGLE_RUNNER_UNDER_TEST", DEFAULT))
    spec = importlib.util.spec_from_file_location("rk_mode_under_test", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    (tmp_path / "kaggle").mkdir()
    monkeypatch.setattr(mod, "REPO", str(tmp_path))
    mod.logs = []
    monkeypatch.setattr(mod, "log", mod.logs.append)
    mod.mode_path = tmp_path / "kaggle" / "kaggle_mode.json"
    return mod


def put(rk, payload):
    text = payload if isinstance(payload, str) else json.dumps(payload)
    rk.mode_path.write_text(text, encoding="utf-8")


def assert_rejected(rk, payload, fragment):
    put(rk, payload)
    with pytest.raises(SystemExit) as info:
        rk.read_mode()
    assert "astro-triage runner stopped" in str(info.value)
    assert fragment in str(info.value)
    assert any("ERROR" in line for line in rk.logs)


def test_absent_file_means_phase1(rk):
    assert not rk.mode_path.exists()
    assert rk.read_mode() == {"mode": "phase1"}


def test_phase1_file_accepted(rk):
    put(rk, {"mode": "phase1"})
    assert rk.read_mode() == {"mode": "phase1"}


@pytest.mark.parametrize("after", ["continue", "stop"])
def test_seeds_file_accepted_and_returned_verbatim(rk, after):
    payload = {"mode": "seeds", "after_measure_one": after}
    put(rk, payload)
    assert rk.read_mode() == payload


@pytest.mark.parametrize("value", ["phase2", "", "SEEDS", None, 1, ["seeds"]])
def test_invalid_mode_value_rejected(rk, value):
    assert_rejected(rk, {"mode": value}, "'mode' must be")


def test_missing_mode_key_rejected(rk):
    assert_rejected(rk, {"after_measure_one": "continue"}, "'mode' must be")


def test_seeds_without_after_measure_one_rejected(rk):
    assert_rejected(rk, {"mode": "seeds"}, "differ from")


@pytest.mark.parametrize("value", ["halt", "", "Continue", None, True, 0])
def test_seeds_invalid_after_measure_one_rejected(rk, value):
    assert_rejected(
        rk, {"mode": "seeds", "after_measure_one": value}, "'after_measure_one' must be"
    )


@pytest.mark.parametrize("text", ["{not json", "", '{"mode": "seeds",'])
def test_corrupt_json_rejected(rk, text):
    assert_rejected(rk, text, "cannot read")


@pytest.mark.parametrize("text", ["[]", '"seeds"', "null", "42", '["mode"]'])
def test_non_object_json_rejected(rk, text):
    assert_rejected(rk, text, "'mode' must be")


def test_invalid_utf8_rejected(rk):
    rk.mode_path.write_bytes(b'{"mode": "\xff\xfe"}')
    with pytest.raises(SystemExit):
        rk.read_mode()


def test_unknown_extra_key_rejected_for_both_modes(rk):
    # Actual behaviour: the key set must match exactly, so any extra key stops the run.
    assert_rejected(rk, {"mode": "phase1", "extra": 1}, "differ from")
    assert_rejected(
        rk, {"mode": "seeds", "after_measure_one": "stop", "extra": 1}, "differ from"
    )


def test_phase1_with_after_measure_one_rejected(rk):
    assert_rejected(rk, {"mode": "phase1", "after_measure_one": "stop"}, "differ from")


def test_directory_in_place_of_file_falls_back_to_phase1(rk):
    rk.mode_path.mkdir()
    # isfile() is False for a directory, which is treated like an absent file.
    assert rk.read_mode() == {"mode": "phase1"}
