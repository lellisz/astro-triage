"""Web export (ADR 0008, phase 1): type helpers, file invariants and CLI behaviour.

The synthetic release is built with the evaluation runner over the dataset of
``test_evaluation.make_data``; nothing here reads ``data/`` except the optional real-release
test, which is read-only and skips when ``ASTRO_DATA_ROOT`` has no ``full_v0``.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pytest
from test_evaluation import make_data

from astro_triage.evaluation import runner as R
from astro_triage.web_export import SCHEMA_VERSION
from astro_triage.web_export.__main__ import main
from astro_triage.web_export.types import clean, est, head, num

RELEASE = "synth_v0"
FILES = (
    "meta",
    "overview",
    "sky",
    "galaxies",
    "compare",
    "method",
    "fmt_samples",
    "graficos",
)
FORBIDDEN_KEYS = {"source_id", "ra", "dec"}
GRAFICOS_FORBIDDEN_KEYS = FORBIDDEN_KEYS | {"l", "b", "healpix_l5", "hp"}
GRAFICOS_BANNED_SOURCE = ("dl_eval", "predictions_mlp", "predictions_ftt", '"dl"')
TAG = re.compile(r"<[A-Za-z/!]")
SKY_MAX_BYTES = 250 * 1024
GRAFICOS_MAX_BYTES = 150 * 1024
GRAFICOS_MAX_ARRAY = 252


# ----------------------------------------------------------------------------- helpers
def _strict(text: str) -> object:
    def reject(token: str) -> None:
        raise AssertionError(f"non-finite token {token} in export")

    return json.loads(text, parse_constant=reject)


def _walk(obj: object, path: str = "$"):
    """Yield (path, key_or_None, value) for every node, keys included."""
    yield path, None, obj
    if isinstance(obj, dict):
        for k, v in obj.items():
            yield f"{path}.{k}", k, v
            yield from _walk(v, f"{path}.{k}")
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            yield from _walk(v, f"{path}[{i}]")


def _check_file(doc: dict, name: str, release: str) -> None:
    assert doc["provisional"] is True, name
    assert doc["schema_version"] == 1, name
    assert doc["release"] == release, name


def _check_content(doc: object, name: str) -> None:
    for path, key, value in _walk(doc):
        assert key not in FORBIDDEN_KEYS, f"{name}: key {key} at {path}"
        if isinstance(value, str):
            assert not TAG.search(value), f"{name}: html tag at {path}: {value[:80]!r}"
            assert "ADR" not in value, f"{name}: 'ADR' at {path}: {value[:80]!r}"
        if isinstance(key, str):
            assert "ADR" not in key, f"{name}: 'ADR' in key at {path}"


def _check_sky(sky: dict) -> None:
    regions = sky["regions"]
    for r in regions:
        if r["below_floor"] is True:
            assert r["errors"] is None and r["rate"] is None, r
    for path, _, value in _walk(sky):
        if isinstance(value, list):
            assert len(value) <= len(regions), f"{path} has {len(value)} items"


def _check_graficos(doc: dict) -> None:
    """Below-floor items are null, no per-object key, no long array (ADR 0008 A1.5 R3, R4)."""
    for path, key, value in _walk(doc):
        assert key not in GRAFICOS_FORBIDDEN_KEYS, f"graficos: key {key} at {path}"
        if isinstance(value, list):
            assert len(value) <= GRAFICOS_MAX_ARRAY, f"{path} has {len(value)} items"
        if isinstance(value, dict) and value.get("below_floor") is True:
            for field in ("errors", "p_mean", "freq", "share", "rate"):
                if field in value:
                    sub = value[field]
                    assert sub["value" if "value" in sub else "point"] is None, path
            if "pred" in value:
                assert value["n"]["value"] is None, f"{path}: n of a confusion cell"
            else:
                assert value["n"]["value"] is not None, f"{path}: 1D charts keep n"
            for raw in ("raw_p_mean", "raw_freq"):
                assert value.get(raw) is None, path
    assert doc["page"] == "graficos" and doc["ece"] is None and doc["cmd"] is None


def _export_dir(out: Path, release: str) -> Path:
    return out / release


def _read_all(folder: Path, release: str) -> dict[str, dict]:
    docs = {}
    assert sorted(p.name for p in folder.iterdir()) == sorted(f"{n}.json" for n in FILES)
    for name in FILES:
        text = (folder / f"{name}.json").read_text(encoding="utf-8")
        doc = _strict(text)
        if name == "fmt_samples":
            # ADR 0008 Appendix A: a bare list [{fn, args, text}], so it has no head
            assert isinstance(doc, list) and doc
            assert all(set(s) == {"fn", "args", "text"} for s in doc)
            docs[name] = doc
            continue
        _check_file(doc, name, release)
        docs[name] = doc
    return docs


# ------------------------------------------------------------------------ 1. types
@pytest.mark.parametrize("bad", [None, float("nan"), float("inf"), float("-inf")])
def test_num_non_finite_is_null_and_na(bad: object) -> None:
    assert num(bad) == {"value": None, "text": "n/a"}
    assert num(np.float64(bad) if bad is not None else None)["value"] is None


@pytest.mark.parametrize("bad", [None, float("nan"), float("inf")])
@pytest.mark.parametrize("slot", ["point", "lo", "hi"])
def test_est_all_or_nothing(bad: object, slot: str) -> None:
    args = {"point": 0.5, "lo": 0.4, "hi": 0.6}
    args[slot] = bad
    out = est(**args)
    assert out["point"] is None and out["lo"] is None and out["hi"] is None
    assert out["text"] == "n/a" and out["ci_text"] == "n/a"
    full = est(0.5, 0.4, 0.6)
    assert full["point"] == 0.5 and full["lo"] == 0.4 and full["hi"] == 0.6


def test_zero_stays_zero() -> None:
    assert num(0.0)["value"] == 0.0 and num(0.0)["text"] != "n/a"
    assert num(0)["value"] == 0
    e = est(0.0, 0.0, 0.0)
    assert (e["point"], e["lo"], e["hi"]) == (0.0, 0.0, 0.0) and e["text"] != "n/a"
    assert est(0.0, 0.0, 0.2)["point"] == 0.0


@pytest.mark.parametrize("flag", [True, False, np.bool_(True)])
def test_bool_is_refused(flag: object) -> None:
    with pytest.raises(TypeError):
        num(flag)
    with pytest.raises(TypeError):
        est(flag, 0.1, 0.2)
    with pytest.raises(TypeError):
        est(0.1, flag, 0.2)
    with pytest.raises(TypeError):
        est(0.1, 0.0, flag)


def test_clean_converts_numpy_and_non_finite_recursively() -> None:
    raw = {
        "a": np.float64("nan"),
        "b": [np.int64(2), np.float32("inf"), (float("-inf"), 1.5)],
        "c": {"d": np.array([1.0, np.nan, np.inf]), "e": np.bool_(True)},
        1: np.str_("x"),
    }
    out = clean(raw)
    assert out == {
        "a": None,
        "b": [2, None, [None, 1.5]],
        "c": {"d": [1.0, None, None], "e": True},
        "1": "x",
    }
    assert type(out["b"][0]) is int
    json.dumps(out, allow_nan=False)


def test_head_is_provisional_and_versioned() -> None:
    h = head("r1", "meta")
    assert h["provisional"] is True
    assert h["schema_version"] == 1 == SCHEMA_VERSION
    assert h["release"] == "r1" and h["page"] == "meta"


# --------------------------------------------------------------- synthetic release
@pytest.fixture(scope="module")
def synth_root(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Data root holding ``interim/synth_v0`` with a full, evaluated synthetic release."""
    root = tmp_path_factory.mktemp("dataroot")
    rel = root / "interim" / RELEASE
    ds = make_data(per_band=15, counts={"STAR": 3, "GALAXY": 3, "QSO": 3})
    n = len(ds.labels)
    ds.labels["ra"] = 10.0
    ds.labels["dec"] = 20.0
    ds.labels["l"] = 30.0
    ds.labels["sdss_z"] = np.where(ds.labels["sdss_class"] == "STAR", 0.0, 0.3)
    assert n > 0
    for frame in ds.preds.values():
        for c in ("star", "galaxy", "qso"):
            frame[f"p_raw_{c}"] = frame[f"p_cal_{c}"]
    for m in ds.report["models"]:
        m.update(
            n_features=10,
            best_iteration=5,
            temperature=1.1,
            gain_top10=[{"feature": "f1", "share": 0.6}, {"feature": "f2", "share": 0.4}],
            val_calibration_fitted_here={"accuracy": 0.9, "macro_f1": 0.9},
        )
    ds.write(rel)
    R.write_outputs(R.evaluate(rel, n_boot=20, workers=1), rel)
    return root


@pytest.fixture
def export(synth_root: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("ASTRO_DATA_ROOT", str(synth_root))
    out = tmp_path / "out"

    def run(release: str = RELEASE, target: Path | None = None) -> int:
        return main(["--release", release, "--out", str(target or out)])

    run.out = out
    run.root = synth_root
    return run


# ----------------------------------------------------------- 2. invariants on export
def test_synthetic_export_invariants(export) -> None:
    assert export() == 0
    docs = _read_all(_export_dir(export.out, RELEASE), RELEASE)
    for name, doc in docs.items():
        _check_content(doc, name)
    _check_sky(docs["sky"])
    assert len(docs["sky"]["regions"]) > 0
    _check_graficos(docs["graficos"])


def test_graficos_below_floor_items_are_withheld() -> None:
    """Falsifier: the checker must reject a below-floor item that still carries a value."""
    cell = {"pred": "QSO", "n": {"value": None}, "share": {"point": None}, "below_floor": True}
    _check_graficos_items(cell)
    for bad in (
        {**cell, "n": {"value": 3}},
        {**cell, "share": {"point": 0.1}},
    ):
        with pytest.raises(AssertionError):
            _check_graficos_items(bad)
    with pytest.raises(AssertionError):
        _check_graficos_items({"n": {"value": None}, "rate": {"point": None}, "below_floor": True})


def _check_graficos_items(item: dict) -> None:
    doc = {
        "page": "graficos",
        "ece": None,
        "cmd": None,
        "items": [item],
    }
    _check_graficos(doc)


def test_graficos_sources_do_not_touch_deep_learning_outputs() -> None:
    from astro_triage.dashboard import data_charts
    from astro_triage.web_export import graficos

    for module in (graficos, data_charts):
        source = Path(module.__file__).read_text(encoding="utf-8")
        for banned in GRAFICOS_BANNED_SOURCE:
            assert banned not in source, f"{module.__name__} contains {banned}"


def test_graficos_import_does_not_load_streamlit() -> None:
    import os
    import subprocess
    import sys

    from astro_triage.web_export import graficos

    src = str(Path(graficos.__file__).parents[2])
    env = {**os.environ, "PYTHONPATH": src}
    code = (
        "import sys, astro_triage.web_export.graficos;"
        "assert 'streamlit' not in sys.modules and 'plotly' not in sys.modules"
    )
    done = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, env=env, check=False
    )
    assert done.returncode == 0, done.stderr


# ------------------------------------------------------------------------- 3. CLI
def test_out_inside_data_root_is_refused(
    export, capsys: pytest.CaptureFixture[str]
) -> None:
    inside = export.root / "interim" / "web_out"
    assert export(target=inside) != 0
    assert not inside.exists()
    inside2 = export.root / "exported"
    assert export(target=inside2) != 0
    assert not inside2.exists()
    assert "erro" in capsys.readouterr().err


def test_unknown_release_fails_and_creates_nothing(export) -> None:
    assert export(release="nao_existe") != 0
    assert not export.out.exists()


def test_reexport_replaces_and_leaves_no_temp_or_sibling_damage(export) -> None:
    export.out.mkdir()
    sibling = export.out / "outra_release"
    sibling.mkdir()
    (sibling / "keep.json").write_text("{}", encoding="utf-8")
    loose = export.out / "nota.txt"
    loose.write_text("x", encoding="utf-8")
    assert export() == 0
    target = _export_dir(export.out, RELEASE)
    stale = target / "meta.json"
    stale.write_text("STALE", encoding="utf-8")
    assert export() == 0
    assert _strict(stale.read_text(encoding="utf-8"))["release"] == RELEASE
    assert sorted(p.name for p in export.out.iterdir()) == sorted(
        ["nota.txt", "outra_release", RELEASE]
    )
    assert (sibling / "keep.json").read_text(encoding="utf-8") == "{}"
    assert loose.read_text(encoding="utf-8") == "x"


def _compare(export) -> dict:
    assert export() == 0
    return _strict((_export_dir(export.out, RELEASE) / "compare.json").read_text("utf-8"))


def test_compare_without_dl_results_is_unavailable(export) -> None:
    doc = _compare(export)
    assert doc["status"] == "unavailable"
    blocks = [k for k in doc if k not in {*head("r", "compare"), "status", "error_text"}]
    assert blocks
    assert all(doc[k] is None for k in blocks), {k: doc[k] for k in blocks}


def _malformed_root(synth_root: Path, tmp_path: Path) -> Path:
    import shutil

    root = tmp_path / "root"
    shutil.copytree(synth_root, root)
    bad = root / "interim" / RELEASE / "dl" / "eval"
    bad.mkdir(parents=True)
    (bad / "results_dl.json").write_text("{ not json", encoding="utf-8")
    return root


def test_compare_page_alone_reports_error_on_malformed_dl_results(
    synth_root: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Isolates compare(): the other page that reads results_dl (overview) is neutralised."""
    from astro_triage.dashboard import data_headline

    monkeypatch.setattr(data_headline, "load_results_dl", lambda rel: None)
    monkeypatch.setenv("ASTRO_DATA_ROOT", str(_malformed_root(synth_root, tmp_path)))
    out = tmp_path / "out"
    assert main(["--release", RELEASE, "--out", str(out)]) == 0
    doc = _strict((out / RELEASE / "compare.json").read_text("utf-8"))
    assert doc["status"] == "error"
    assert isinstance(doc["error_text"], str) and doc["error_text"]
    assert doc["answer_text"] is None and doc["grid"] is None


def test_compare_with_malformed_dl_results_is_error(
    synth_root: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """End to end: a malformed results_dl.json must degrade compare only, not abort."""
    monkeypatch.setenv("ASTRO_DATA_ROOT", str(_malformed_root(synth_root, tmp_path)))
    out = tmp_path / "out"
    assert main(["--release", RELEASE, "--out", str(out)]) == 0
    doc = _strict((out / RELEASE / "compare.json").read_text("utf-8"))
    assert doc["status"] == "error"
    assert isinstance(doc["error_text"], str) and doc["error_text"]
    assert doc["answer_text"] is None and doc["grid"] is None


# ------------------------------------------------------------------ 4. real release
def test_real_release_invariants(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import os

    root = os.environ.get("ASTRO_DATA_ROOT", "")
    real = "full_v0"
    if not root or not (Path(root) / "interim" / real / "eval" / "results.json").is_file():
        pytest.skip("ASTRO_DATA_ROOT sem interim/full_v0/eval/results.json")
    out = tmp_path / "out"
    assert main(["--release", real, "--out", str(out)]) == 0
    sky_path = out / real / "sky.json"
    assert sky_path.stat().st_size < SKY_MAX_BYTES
    docs = _read_all(out / real, real)
    for name, doc in docs.items():
        _check_content(doc, name)
    _check_sky(docs["sky"])
    _check_graficos(docs["graficos"])
    assert (out / real / "graficos.json").stat().st_size < GRAFICOS_MAX_BYTES


# ------------------------------------------- 5. REV-A01: never touch the data folder
def _snapshot(base: Path) -> dict[str, bytes | None]:
    """Every path under ``base`` (relative) with its bytes; folders map to None."""
    return {
        p.relative_to(base).as_posix(): (p.read_bytes() if p.is_file() else None)
        for p in sorted(base.rglob("*"))
    }


def test_release_named_data_with_root_at_out_data_is_refused_untouched(
    synth_root: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """``--release data`` makes the target ``<out>/data``: exactly the data root."""
    import shutil

    out = tmp_path / "out"
    root = out / "data"
    shutil.copytree(synth_root, root)
    shutil.copytree(root / "interim" / RELEASE, root / "interim" / "data")
    (root / "precious.parquet").write_bytes(b"do-not-touch")
    before = _snapshot(out)
    monkeypatch.setenv("ASTRO_DATA_ROOT", str(root))
    assert main(["--release", "data", "--out", str(out)]) == 1
    assert _snapshot(out) == before
    assert not [p for p in out.iterdir() if p.name != "data"]
    assert "pasta de dados" in capsys.readouterr().err


def test_target_with_foreign_file_is_refused_before_any_rename(
    export, capsys: pytest.CaptureFixture[str]
) -> None:
    target = export.out / RELEASE
    target.mkdir(parents=True)
    (target / "meta.json").write_text("OLD", encoding="utf-8")
    (target / "minha_nota.txt").write_text("estranho", encoding="utf-8")
    before = _snapshot(export.out)
    assert export() == 1
    assert _snapshot(export.out) == before
    assert [p.name for p in export.out.iterdir()] == [RELEASE]
    assert "nada foi movido" in capsys.readouterr().err
