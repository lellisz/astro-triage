"""Proofs for ``layer2_report`` (markdown and JSON) and ``scripts/layer2_evaluate.py --help``.

The full result dict comes from the synthetic release of ``test_layer2_eval``; the verdict
branches reuse its meta/fit/operations with an ``eval`` block computed on the toy arrays.
No network, no key, nothing under ``data/``.
"""

from __future__ import annotations

import copy
import json
import os
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from test_layer2_eval import MODEL, REPO, TOY_STATUS, build_release, toy

from astro_triage.evaluation import layer2 as L2
from astro_triage.evaluation import layer2_data as D
from astro_triage.evaluation import layer2_report as R

FORBIDDEN = ("superior", "melhor", "better", "worse", "outperform")
NO_GAIN = "adds no measurable value over Layer 1"


@pytest.fixture(scope="module")
def base_res(tmp_path_factory: pytest.TempPathFactory) -> dict:
    rel = build_release(tmp_path_factory.mktemp("rep"))
    return D.run_evaluation(rel.rel, MODEL, "blind", enable_thinking=False, n_boot=20)


def with_eval(base: dict, reps: int, **kw) -> dict:
    res = copy.deepcopy(base)
    res["eval"] = L2.evaluate(toy(reps, **kw), n_boot=30)
    return res


def dead_l2() -> dict:
    """Every routed call times out: S2 == S0 and S3 == S1* (tau* = tau)."""
    return {
        "tau_star": 0.6,
        "status": np.array(
            ["TIMEOUT" if s != "NOT_ROUTED" else s for s in TOY_STATUS], dtype=object
        ),
        "label": np.full(12, "", dtype=object),
        "s_l2": np.full(12, np.nan),
    }


def write(res: dict, tmp_path: Path) -> tuple[str, str]:
    json_path, md_path = R.write_outputs(res, tmp_path)
    return json_path.read_text(encoding="utf-8"), md_path.read_text(encoding="utf-8")


def strict_load(raw: str) -> dict:
    def reject(token: str) -> None:
        raise AssertionError(f"non-finite token {token}")

    return json.loads(raw, parse_constant=reject)


# ------------------------------------------------------------------------------ markdown
def test_markdown_is_provisional_with_header_built_from_data(
    base_res: dict, tmp_path: Path
) -> None:
    _, md = write(base_res, tmp_path)
    first = md.splitlines()[0]
    assert md.lstrip("# ").startswith("PROVISIONAL")
    assert first == "# " + base_res["meta"]["header"]
    assert "400 test rows" in first and "40 level-5 pixels" in first
    assert f"{base_res['operations']['n_routed']} routed at tau=0.70" in first
    for section in (
        "## Result",
        "## Setup",
        "## Systems",
        "## Routed subset",
        "## What this does NOT show",
    ):
        assert section in md


def test_markdown_has_no_ranking_words_paths_or_user_names(
    base_res: dict, tmp_path: Path
) -> None:
    _, md = write(base_res, tmp_path)
    low = md.lower()
    for word in FORBIDDEN:
        assert word not in low, word
    assert not re.search(r"[A-Za-z]:[\\/]", md)
    assert not re.search(r"/(Users|home)/", md)
    assert str(tmp_path) not in md and tmp_path.name not in md
    assert "felip" not in low and "appdata" not in low


def test_ranking_word_anywhere_refuses_and_writes_nothing(
    base_res: dict, tmp_path: Path
) -> None:
    bad = copy.deepcopy(base_res)
    bad["meta"]["model_requested"] = "better-model"
    out = tmp_path / "out"
    with pytest.raises(ValueError, match="ranking words"):
        R.write_outputs(bad, out)
    assert not list(out.glob("eval_l2*"))


def test_no_gain_branch_is_stated_when_the_floor_is_not_met(
    base_res: dict, tmp_path: Path
) -> None:
    res = with_eval(base_res, 1)  # 12 objects: positive S2-S0 point, below the floor
    raw, md = write(res, tmp_path)
    assert NO_GAIN in md
    assert "2 headline cell(s) below the reporting floor" in md
    assert strict_load(raw)["eval"]["verdict"]["outcome"] == "no_gain"


def test_no_gain_branch_when_the_interval_touches_zero(
    base_res: dict, tmp_path: Path
) -> None:
    res = with_eval(base_res, 30, **dead_l2())
    raw, md = write(res, tmp_path)
    assert strict_load(raw)["eval"]["verdict"]["n_evaluable"] == 2
    assert NO_GAIN in md
    assert "headline cell(s) below" not in md  # evaluable, the CI just is not above 0
    assert "CI excl. 0" not in md.split("## Systems")[0]


def test_gain_branch_omits_the_negative_sentence_and_flags_the_interval(
    base_res: dict, tmp_path: Path
) -> None:
    raw, md = write(with_eval(base_res, 30), tmp_path)
    assert strict_load(raw)["eval"]["verdict"]["outcome"] == "gain"
    assert NO_GAIN not in md
    assert "Positive paired difference with a CI above 0 on macro_f1" in md
    assert "CI excl. 0" in md
    assert not any(w in md.lower() for w in FORBIDDEN)


def is_value_row(line: str) -> bool:
    first = line.strip("| ").split("|")[0].strip()
    return first in {"S0", "S1", "S1*", "S2", "S3", *R.ROUTED_UNITS} or " - " in first


def test_cells_below_the_floor_print_na_and_no_raw_value(
    base_res: dict, tmp_path: Path
) -> None:
    _, md = write(with_eval(base_res, 1), tmp_path)
    assert "n/a (n=4, pixels=4)" in md
    rows = [ln for ln in md.splitlines() if ln.startswith("|") and is_value_row(ln)]
    assert len(rows) >= 5 + 3 * 5 + 5 + 4 + 2
    for line in rows:
        assert not re.search(r"\d\.\d{3}", line), (
            line
        )  # no number, interval or raw value


def test_floor_passing_cells_print_point_and_interval(
    base_res: dict, tmp_path: Path
) -> None:
    _, md = write(with_eval(base_res, 30), tmp_path)
    s0 = next(ln for ln in md.splitlines() if ln.startswith("| S0 |"))
    assert re.search(r"\d\.\d{3} \[\d\.\d{3}, \d\.\d{3}\]", s0)


@pytest.mark.parametrize(
    ("cell", "diff", "expect", "flagged"),
    [
        (
            {"status": "ok", "point": 0.2, "lo": 0.1, "hi": 0.3},
            True,
            "0.200 [0.100, 0.300]",
            True,
        ),
        ({"status": "ok", "point": -0.2, "lo": -0.3, "hi": -0.1}, True, "-0.200", True),
        (
            {"status": "ok", "point": 0.0, "lo": 0.0, "hi": 0.3},
            True,
            "0.000 [0.000, 0.300]",
            False,
        ),
        (
            {"status": "ok", "point": 0.2, "lo": 0.1, "hi": 0.3},
            False,
            "0.200 [0.100, 0.300]",
            False,
        ),
        (
            {"status": "n/a (n=4, pixels=4)", "point": 0.2, "lo": None, "hi": None},
            True,
            "n/a (n=4, pixels=4)",
            False,
        ),
        (
            {"status": "n/a (undefined)", "point": None, "lo": None, "hi": None},
            False,
            "n/a (undefined)",
            False,
        ),
    ],
)
def test_cell_formatting(cell: dict, diff: bool, expect: str, flagged: bool) -> None:
    text = R._fmt(cell, diff=diff)
    assert text.startswith(expect)
    assert ("CI excl. 0" in text) is flagged
    if cell["status"].startswith("n/a"):
        assert text == cell["status"]  # never the raw point


# ---------------------------------------------------------------------------------- JSON
def test_json_is_strict_and_matches_the_markdown_header(
    base_res: dict, tmp_path: Path
) -> None:
    raw, md = write(base_res, tmp_path)
    assert "NaN" not in raw and "Infinity" not in raw
    data = strict_load(raw)
    assert data["meta"]["header"] == md.splitlines()[0].removeprefix("# ")
    assert set(data) == {"meta", "fit", "operations", "eval"}
    assert data["eval"]["n_boot"] == 20
    assert not list(tmp_path.glob("*.tmp"))


def test_undefined_values_are_null_in_json(base_res: dict, tmp_path: Path) -> None:
    raw, _ = write(with_eval(base_res, 1), tmp_path)
    cells = strict_load(raw)["eval"]["units"]["L2_selective"]["metrics"]
    assert cells["macro_f1"]["lo"] is None and cells["macro_f1"]["hi"] is None
    assert cells["macro_f1"]["status"].startswith("n/a")


def test_to_jsonable_maps_non_finite_and_numpy_types() -> None:
    out = R.to_jsonable(
        {
            "a": float("inf"),
            "b": np.float32("nan"),
            "c": np.int64(3),
            "d": np.array([1.0, np.nan, -np.inf]),
            "e": (1, np.bool_(True)),
            5: float("-inf"),
        }
    )
    assert out == {
        "a": None,
        "b": None,
        "c": 3,
        "d": [1.0, None, None],
        "e": [1, True],
        "5": None,
    }
    json.dumps(out, allow_nan=False)


def test_write_outputs_is_idempotent_and_overwrites(
    base_res: dict, tmp_path: Path
) -> None:
    assert write(base_res, tmp_path) == write(base_res, tmp_path)


# ------------------------------------------------------------------------------------ CLI
def test_cli_help_exits_zero_without_network_or_key() -> None:
    env = {k: v for k, v in os.environ.items() if k != "NVIDIA_API_KEY"}
    done = subprocess.run(
        [sys.executable, str(REPO / "scripts" / "layer2_evaluate.py"), "--help"],
        capture_output=True,
        text=True,
        timeout=120,
        env=env,
        check=False,
    )
    assert done.returncode == 0, done.stderr
    out = done.stdout
    assert "--release" in out and "--model" in out and "--arm" in out
