"""Per-column rules of full_v0_parity.py (l/b: aggregate stats; pm: 3 decimals; rest exact).

Also make_kaggle_package.py tests. l/b edge cases live in test_full_v0_parity_stats.py.
Synthetic fixtures only. Swap modules for mutation checks via FULL_V0_PARITY_UNDER_TEST /
KAGGLE_PACKAGER_UNDER_TEST.
"""

import importlib.util
import json
import os
import re
import zipfile
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

ROOT = Path(__file__).resolve().parents[1]


def load(var, default):
    path = Path(os.environ.get(var, ROOT / default))
    spec = importlib.util.spec_from_file_location(path.stem + "_ut", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


par = load("FULL_V0_PARITY_UNDER_TEST", "scripts/full_v0_parity.py")
pkg = load("KAGGLE_PACKAGER_UNDER_TEST", "scripts/make_kaggle_package.py")
IDS = [1000000011, 1000000022, 1000000033, 1000000044]
BASE = {  # l/b values are not hashed (stats within 1e-9); pm is float32, rounded at 3 decimals
    "l": [12.3456789004, 100.0000000003, 0.5, 300.25],
    "b": [-45.1234567891, 2.0, 0.25, 80.75],
    "pm": [1.2344, 20.5, 3.25, 0.0],
    "bp_rp": [0.75, 1.5, -0.25, 2.0],
    "parallax_over_error": [5.5, 12.25, 1.0, 40.0],
    "ra": [10.5, 20.5, 30.5, 40.5],
    "dec": [-5.5, 6.5, 7.5, -8.5],
}


def table(**over):
    cols = {"source_id": IDS, **BASE}
    cols.update(over)
    types = {"pm": pa.float32(), "source_id": pa.int64()}
    return pa.table(
        {k: pa.array(v, types.get(k, pa.float64())) for k, v in cols.items()}
    )


def release(path, feat):
    path.mkdir(parents=True, exist_ok=True)
    lab = pa.table(
        {"source_id": IDS, "sdss_class": ["STAR"] * 4, "fold": ["train"] * 4}
    )
    pq.write_table(lab, path / "labels.parquet")
    pq.write_table(feat, path / "features.parquet")
    return path


def run(tmp_path, feat, capsys):
    ref = tmp_path / "ref.json"
    assert (
        par.main(
            [
                "reference",
                "--outdir",
                str(release(tmp_path / "a", table())),
                "--out",
                str(ref),
            ]
        )
        == 0
    )
    argv = [
        "check",
        "--reference",
        str(ref),
        "--outdir",
        str(release(tmp_path / "b", feat)),
    ]
    code = par.main([*argv, "--report", str(tmp_path / "rep.json")])
    return code, {x.split(": ")[1] for x in capsys.readouterr().out.splitlines()}


def bump(name, delta):
    values = list(BASE[name])
    values[0] += delta
    return table(**{name: values})


def ulp(name):
    values = list(BASE[name])
    values[0] = float(np.nextafter(values[0], np.inf))
    return table(**{name: values})


# fmt: off
@pytest.mark.parametrize("feat", [bump("l", 1e-12), bump("b", -1e-12), bump("pm", 1e-5)], ids=["l", "b", "pm"])
def test_sub_tolerance_noise_in_derived_columns_matches(tmp_path, capsys, feat):
    assert run(tmp_path, feat, capsys) == (0, set())


@pytest.mark.parametrize("name,delta", [("l", 1e-6), ("b", 1e-6), ("pm", 1e-2)])
def test_over_tolerance_in_derived_column_names_only_that_column(tmp_path, capsys, name, delta):
    assert run(tmp_path, bump(name, delta), capsys) == (3, {f"features.column.{name}"})


@pytest.mark.parametrize("name", ["bp_rp", "parallax_over_error", "ra", "dec"])
def test_one_ulp_in_exact_column_is_a_mismatch(tmp_path, capsys, name):
    assert run(tmp_path, ulp(name), capsys) == (3, {f"features.column.{name}"})


@pytest.mark.parametrize("name", ["l", "pm", "bp_rp", "dec"])
def test_negative_zero_hashes_like_zero(tmp_path, capsys, name):
    ref_vals = [0.0, *BASE[name][1:]]
    ref = tmp_path / "ref.json"
    release(tmp_path / "a", table(**{name: ref_vals}))
    assert par.main(["reference", "--outdir", str(tmp_path / "a"), "--out", str(ref)]) == 0
    other = release(tmp_path / "b", table(**{name: [-0.0, *BASE[name][1:]]}))
    argv = ["check", "--reference", str(ref), "--outdir", str(other), "--report", str(tmp_path / "r.json")]
    assert par.main(argv) == 0, capsys.readouterr().out


def test_column_on_one_side_only_is_a_mismatch(tmp_path, capsys):
    extra = table().append_column("extra", pa.array([1, 2, 3, 4]))
    code, got = run(tmp_path, extra, capsys)
    assert (code, got) == (3, {"features_schema", "features.column.extra"})
    reverse = table().drop_columns(["dec"])
    assert run(tmp_path / "x", reverse, capsys) == (3, {"features_schema", "features.column.dec"})
# fmt: on


def tree(root):
    for name in ("src", "scripts", "manifests", "kaggle"):
        (root / name).mkdir(parents=True, exist_ok=True)
        (root / name / "m.py").write_text("x = 1\n")
    (root / "kaggle/full_v0_reference.json").write_text("{}")
    (root / "requirements.txt").write_text("numpy==1\n")
    return root


def names(zpath):
    with zipfile.ZipFile(zpath) as z:
        return z.namelist()


SEEDS = {"mode": "seeds", "after_measure_one": "stop"}


def test_package_errors(tmp_path):
    root = tree(tmp_path / "r")
    (root / "kaggle/full_v0_reference.json").unlink()
    with pytest.raises(pkg.PackageError):
        pkg.build(root, tmp_path / "o", SEEDS)
    tree(root)
    (root / "kaggle/kaggle_mode.json").write_text("{}")
    with pytest.raises(pkg.PackageError):
        pkg.build(root, tmp_path / "o", SEEDS)


def test_phase1_build_refuses_stray_mode_file(tmp_path):
    root = tree(tmp_path / "r")
    (root / "kaggle/kaggle_mode.json").write_text("{}")
    with pytest.raises(pkg.PackageError):  # else phase1 would ship a mode file
        pkg.build(root, tmp_path / "o")


@pytest.mark.parametrize("mode", [None, SEEDS])
def test_real_repo_package_is_clean_and_small(tmp_path, mode):
    target, _, _ = pkg.build(ROOT, tmp_path, mode)
    listed = names(target)
    bad = re.compile(
        r"(?i)(^|/)(data|logs)/|\.env|kaggle\.json|token|\.parquet$|\.duckdb$|\.fits$"
    )
    assert target.stat().st_size < 5 * 1024 * 1024
    assert not [n for n in listed if bad.search(n)]
    assert ("kaggle/kaggle_mode.json" in listed) == (mode is not None)
    if mode:
        assert (
            json.loads(zipfile.ZipFile(target).read("kaggle/kaggle_mode.json")) == mode
        )
    assert not (ROOT / "kaggle/kaggle_mode.json").exists()


# fmt: off
def test_cli_flag_rules(tmp_path, monkeypatch):
    monkeypatch.setattr(pkg, "REPO_ROOT", tree(tmp_path / "r"))
    out = str(tmp_path / "o")
    bad = [["--after-measure-one", "stop"], ["--mode", "seeds"], ["--mode", "phase1", "--after-measure-one", "stop"]]
    for argv in bad:
        with pytest.raises(SystemExit) as exc:
            pkg.main(["--outdir", out, *argv])
        assert exc.value.code == 2
    assert pkg.main(["--outdir", out, "--mode", "seeds", "--after-measure-one", "continue"]) == 0
    assert "kaggle/kaggle_mode.json" in names(Path(out) / "astro_triage_pkg.zip")
# fmt: on
