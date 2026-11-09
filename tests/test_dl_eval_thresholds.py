"""F1: verify_freeze validates freeze['thresholds'] (structure, range, LightGBM == delta.json) before any test read."""

from __future__ import annotations

import json
import shutil

import pytest
from dl_eval_kit import edit_freeze, frozen_env, read_freeze

from astro_triage.evaluation import dl_eval as E
from astro_triage.evaluation.runner import EvalError

LG = "lgbm_full_s2"


@pytest.fixture(scope="module")
def base(tmp_path_factory):
    root = tmp_path_factory.mktemp("thr")
    return root, frozen_env(root / "env", real_delta=True)


@pytest.fixture
def env(base, tmp_path):
    shutil.copytree(base[0] / "env", tmp_path / "env")
    return tmp_path / "env" / "rel", tmp_path / "env" / "dl", tmp_path / "env" / "lg"


@pytest.fixture
def spy(monkeypatch):
    calls: list = []
    monkeypatch.setattr(E, "load_test_set", lambda out: calls.append(out) or object())
    monkeypatch.setitem(E._TEST_OPENED, "done", False)
    return calls


def _set(sys_, cls, c, v):
    def fn(d):
        d["thresholds"][sys_][cls][c] = v

    return fn


def _drop(path):
    def fn(d):
        o = d["thresholds"]
        for k in path[:-1]:
            o = o[k]
        del o[path[-1]]

    return fn


def _extra_sys(d):
    d["thresholds"]["lgbm_full_s9"] = d["thresholds"][LG]


def _extra_cls(d):
    d["thresholds"]["mlp_full_s0"]["STAR"] = d["thresholds"]["mlp_full_s0"]["QSO"]


def _bad_cls(d):
    d["thresholds"]["mlp_full_s0"]["QSO"] = []


def _bad_lvl(d):
    d["thresholds"]["mlp_full_s0"]["QSO"]["0.80"] = {"x": 1}


STRUCT = {
    "missing": lambda d: d.pop("thresholds"),
    "list": lambda d: d.update(thresholds=[]),
    "none": lambda d: d.update(thresholds=None),
    "sys_missing": _drop(["mlp_noind_s3"]),
    "sys_extra": _extra_sys,
    "cls_missing": _drop(["mlp_full_s0", "QSO"]),
    "cls_extra": _extra_cls,
    "cls_not_dict": _bad_cls,
    "lvl_missing": _drop(["ftt_full_s1", "GALAXY", "0.80"]),
    "lvl_not_dict": _bad_lvl,
}


@pytest.mark.parametrize("name", STRUCT)
def test_bad_structure_refused_before_test_read(env, spy, name):
    out, dl, lg = env
    edit_freeze(dl, STRUCT[name])
    with pytest.raises(EvalError, match="freeze thresholds"):
        E.open_test_set(out, dl, lg, "ok")
    assert spy == []


@pytest.mark.parametrize("v", [None, "0.5", True, False, float("nan"), float("inf"), 0, 0.0, -0.1, 1.0001, 2, [0.5]])
@pytest.mark.parametrize("sys_", ["mlp_full_s0", LG])
def test_bad_value_refused_before_test_read(env, spy, v, sys_):
    out, dl, lg = env
    edit_freeze(dl, _set(sys_, "QSO", "0.80", v))
    with pytest.raises(EvalError, match="freeze thresholds"):
        E.verify_freeze(out, dl, lg)
    with pytest.raises(EvalError, match="freeze thresholds"):
        E.open_test_set(out, dl, lg, "ok")
    assert spy == []


def test_boundary_one_and_int_one_accepted(env):
    out, dl, lg = env
    edit_freeze(dl, _set("mlp_full_s0", "QSO", "0.80", 1.0))
    edit_freeze(dl, _set("mlp_full_s1", "QSO", "0.80", 1))
    E.verify_freeze(out, dl, lg)


def test_lightgbm_threshold_differing_from_delta_refused_naming_item(env, spy):
    out, dl, lg = env
    old = read_freeze(dl)["thresholds"][LG]["GALAXY"]["0.80"]
    edit_freeze(dl, _set(LG, "GALAXY", "0.80", old / 2))
    with pytest.raises(EvalError, match=r"lgbm_full_s2/GALAXY/0\.80 differ from delta\.json"):
        E.open_test_set(out, dl, lg, "ok")
    assert spy == []


def test_dl_threshold_may_differ_from_lightgbm(env):
    out, dl, lg = env
    edit_freeze(dl, _set("ftt_noind_s4", "GALAXY", "0.90", 0.123))
    assert E.verify_freeze(out, dl, lg)["thresholds"]["ftt_noind_s4"]["GALAXY"]["0.90"] == 0.123


def test_matching_thresholds_accepted_and_delta_is_real(env, spy):
    out, dl, lg = env
    cells = json.loads((lg / "delta.json").read_text())["delta"]
    assert cells["full"]["QSO"]["0.80"]["P2"]["thresholds"][2] == read_freeze(dl)["thresholds"][LG]["QSO"]["0.80"]
    _, freeze = E.open_test_set(out, dl, lg, "ok")
    assert len(spy) == 1 and "thresholds" in freeze


def test_delta_without_delta_key_refused(env):
    out, dl, lg = env
    from astro_triage.evaluation import dl_val as V

    doc = json.loads((lg / "delta.json").read_text())
    del doc["delta"]
    (lg / "delta.json").write_text(json.dumps(doc))
    edit_freeze(dl, lambda d: d.update(delta_sha256=V.sha256_file(lg / "delta.json")))
    with pytest.raises(EvalError):
        E.verify_freeze(out, dl, lg)
