"""load_freeze / verify_freeze (spec S11, ADR 0006 sections 14 and 17): every refusal is exercised."""

from __future__ import annotations

import pytest
from dl_eval_kit import edit_freeze, frozen_env, write_freeze

from astro_triage.evaluation import dl_eval as E
from astro_triage.evaluation.runner import EvalError


def test_load_freeze_missing(tmp_path):
    with pytest.raises(EvalError, match="val_freeze.json missing"):
        E.load_freeze(tmp_path)


@pytest.mark.parametrize("text", ["{not json", "", "\x00\x01"])
def test_load_freeze_unreadable_json(tmp_path, text):
    (tmp_path / E.FREEZE_NAME).write_text(text, encoding="utf-8")
    with pytest.raises(EvalError, match="cannot read"):
        E.load_freeze(tmp_path)


def test_load_freeze_undecodable_bytes(tmp_path):
    (tmp_path / E.FREEZE_NAME).write_bytes(b"\xff\xfe\xfa")
    with pytest.raises(EvalError, match="cannot read"):
        E.load_freeze(tmp_path)


def good(**over):
    doc = {
        "schema_name": E.FREEZE_SCHEMA,
        "schema_version": 1,
        "approval": "ok",
        "prediction_sha256": {},
        "train_class_fractions": {},
    }
    doc.update(over)
    return doc


@pytest.mark.parametrize(
    "doc",
    [
        [],
        "x",
        None,
        good(schema_name="other"),
        good(schema_version=2),
        good(schema_version="1"),
    ],
)
def test_load_freeze_wrong_schema(tmp_path, doc):
    write_freeze(tmp_path, doc)
    with pytest.raises(EvalError, match="is not a astro_triage.dl_val_freeze v1"):
        E.load_freeze(tmp_path)


@pytest.mark.parametrize("approval", ["", "  \n\t ", None, 7])
def test_load_freeze_empty_approval(tmp_path, approval):
    write_freeze(tmp_path, good(approval=approval))
    with pytest.raises(EvalError, match="no approval text"):
        E.load_freeze(tmp_path)


@pytest.mark.parametrize("key", ["prediction_sha256", "train_class_fractions"])
@pytest.mark.parametrize("bad", [None, [], "x"])
def test_load_freeze_sections_must_be_objects(tmp_path, key, bad):
    write_freeze(tmp_path, good(**{key: bad}))
    with pytest.raises(EvalError, match=f"{key} is not an object"):
        E.load_freeze(tmp_path)


def test_load_freeze_valid_roundtrip(tmp_path):
    doc = good()
    write_freeze(tmp_path, doc)
    assert E.load_freeze(tmp_path) == doc


@pytest.mark.parametrize("part", ["pilot", "Pilot", "PILOT"])
def test_load_freeze_pilot_dir_refused_even_with_valid_freeze(tmp_path, part):
    d = tmp_path / part / "run"
    d.mkdir(parents=True)
    write_freeze(d, good())
    with pytest.raises(EvalError, match="pilot directory"):
        E.load_freeze(d)


def test_load_freeze_pilot_substring_is_not_a_component(tmp_path):
    d = tmp_path / "pilotage"
    d.mkdir()
    write_freeze(d, good())
    assert E.load_freeze(d)["approval"] == "ok"


def test_verify_freeze_happy(tmp_path):
    out, dl, lg, doc = frozen_env(tmp_path)
    assert E.verify_freeze(out, dl, lg) == doc


def test_verify_freeze_labels_tampered_or_absent(tmp_path):
    out, dl, lg, _ = frozen_env(tmp_path)
    lab = out / "labels.parquet"
    orig = lab.read_bytes()
    lab.write_bytes(orig + b"x")
    with pytest.raises(EvalError, match="labels sha256 differs"):
        E.verify_freeze(out, dl, lg)
    lab.unlink()
    with pytest.raises(EvalError, match="labels sha256 differs"):
        E.verify_freeze(out, dl, lg)
    lab.write_bytes(orig)
    E.verify_freeze(out, dl, lg)


def test_verify_freeze_delta_tampered_or_absent(tmp_path):
    out, dl, lg, _ = frozen_env(tmp_path)
    delta = lg / "delta.json"
    orig = delta.read_bytes()
    delta.write_bytes(orig + b" ")
    with pytest.raises(EvalError, match=r"delta.json sha256 differs"):
        E.verify_freeze(out, dl, lg)
    delta.unlink()
    with pytest.raises(EvalError, match=r"delta.json sha256 differs"):
        E.verify_freeze(out, dl, lg)


@pytest.mark.parametrize(
    "name", ["mlp_full_s0", "ens_ftt_full_s4", "lgbm_noind_s2", "ftt_noind_s3"]
)
def test_verify_freeze_prediction_tampered_names_the_system(tmp_path, name):
    out, dl, lg, _ = frozen_env(tmp_path)
    path = E._paths(dl, lg)[name]
    path.write_bytes(path.read_bytes() + b"x")
    with pytest.raises(EvalError, match="sha256 differs from the freeze for") as ei:
        E.verify_freeze(out, dl, lg)
    assert name in str(ei.value) and str(ei.value).count("'") == 2


def test_verify_freeze_prediction_missing_listed(tmp_path):
    out, dl, lg, _ = frozen_env(tmp_path)
    paths = E._paths(dl, lg)
    for n in ("mlp_full_s1", "lgbm_full_s0"):
        paths[n].unlink()
    with pytest.raises(EvalError, match="2 planned files missing") as ei:
        E.verify_freeze(out, dl, lg)
    assert paths["mlp_full_s1"].name in str(ei.value)


def test_verify_freeze_system_set_not_exactly_40(tmp_path):
    out, dl, lg, _ = frozen_env(tmp_path)
    edit_freeze(dl, lambda d: d["prediction_sha256"].pop("mlp_full_s0"))
    with pytest.raises(EvalError, match="exactly the planned systems"):
        E.verify_freeze(out, dl, lg)
    edit_freeze(
        dl, lambda d: d["prediction_sha256"].update(mlp_full_s0="0", extra_s9="0")
    )
    with pytest.raises(EvalError, match="exactly the planned systems"):
        E.verify_freeze(out, dl, lg)


def test_verify_freeze_constant_mutated(tmp_path, monkeypatch):
    out, dl, lg, _ = frozen_env(tmp_path)
    real = E.current_constants
    monkeypatch.setattr(E, "current_constants", lambda: {**real(), "n_boot": 999})
    with pytest.raises(EvalError, match="constants differ from current code: n_boot$"):
        E.verify_freeze(out, dl, lg)


def test_verify_freeze_constant_extra_in_code_or_missing_in_code(tmp_path, monkeypatch):
    out, dl, lg, _ = frozen_env(tmp_path)
    real = E.current_constants
    monkeypatch.setattr(E, "current_constants", lambda: {**real(), "new_const": 1})
    with pytest.raises(EvalError, match="differ from current code: new_const$"):
        E.verify_freeze(out, dl, lg)
    monkeypatch.setattr(
        E,
        "current_constants",
        lambda: {k: v for k, v in real().items() if k != "rng_seed"},
    )
    with pytest.raises(EvalError, match="differ from current code: rng_seed$"):
        E.verify_freeze(out, dl, lg)


def test_verify_freeze_constant_extra_or_absent_in_freeze(tmp_path):
    out, dl, lg, _ = frozen_env(tmp_path)
    edit_freeze(dl, lambda d: d["constants"].update(sneaky=True))
    with pytest.raises(EvalError, match="differ from current code: sneaky$"):
        E.verify_freeze(out, dl, lg)
    edit_freeze(
        dl, lambda d: (d["constants"].pop("sneaky"), d["constants"].pop("min_blocks"))
    )
    with pytest.raises(EvalError, match="differ from current code: min_blocks$"):
        E.verify_freeze(out, dl, lg)


@pytest.mark.parametrize("bad", [None, [], "x", 3])
def test_verify_freeze_constants_not_an_object(tmp_path, bad):
    out, dl, lg, _ = frozen_env(tmp_path)
    edit_freeze(dl, lambda d: d.__setitem__("constants", bad))
    with pytest.raises(EvalError, match="constants differ"):
        E.verify_freeze(out, dl, lg)


def test_verify_freeze_constants_checked_before_files(tmp_path, monkeypatch):
    out, dl, lg, _ = frozen_env(tmp_path)
    E._paths(dl, lg)["mlp_full_s0"].unlink()
    monkeypatch.setattr(E, "current_constants", lambda: {"x": 1})
    with pytest.raises(EvalError, match="constants differ"):
        E.verify_freeze(out, dl, lg)


def test_verify_freeze_pilot_dir_refused(tmp_path):
    out, dl, lg, _ = frozen_env(tmp_path / "Pilot")
    with pytest.raises(EvalError, match="pilot directory"):
        E.verify_freeze(out, dl, lg)
