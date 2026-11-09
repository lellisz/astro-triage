"""Shared fixture for the kaggle/run_kaggle_dl.py tests: fake /kaggle tree, fake subprocesses."""

import importlib.util
import io
import json
import subprocess
import zipfile
from pathlib import Path
from types import SimpleNamespace

import pytest

RUNNER = Path(__file__).resolve().parents[1] / "kaggle" / "run_kaggle_dl.py"
MODE = {
    "seeds": [0, 1],
    "max_epochs": 3,
    "patience": 2,
    "train_subset": None,
    "ensemble": False,
}
REPORT = "train_report_dl.json"


def put(path, text="x"):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(text, bytes):
        path.write_bytes(text)
    else:
        path.write_text(text)
    return path


def make_package(path, mode):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "w") as z:
        z.writestr("scripts/train_dl.py", "")
        z.writestr("requirements.txt", "numpy==1\ntorch==2\npytest\n")
        if mode is not None:
            z.writestr("kaggle/kaggle_dl_mode.json", json.dumps(mode))


@pytest.fixture
def rk(tmp_path, monkeypatch):
    spec = importlib.util.spec_from_file_location("rk_dl_ut", RUNNER)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    work = tmp_path / "kaggle/working"
    work.mkdir(parents=True)
    (tmp_path / "kaggle/input").mkdir()
    ds = f"{work}/repo/data/interim/full_v0"
    for name, value in {
        "INPUT": str(tmp_path / "kaggle/input"),
        "WORKING": str(work),
        "REPO": f"{work}/repo",
        "LOG_PATH": f"{work}/run_log.txt",
        "ENV_PATH": f"{work}/environment.json",
        "OUT_ZIP": f"{work}/astro_triage_dl_outputs.zip",
        "DATASET_DIR": ds,
        "DL_DIR": f"{ds}/dl",
    }.items():
        monkeypatch.setattr(mod, name, value)
    monkeypatch.setattr(mod, "DRY", False)
    mod.root, mod.work, mod.dl = tmp_path, work, Path(ds) / "dl"
    mod.calls, mod.handlers = [], {}
    mod.session_code, mod.train_code = 0, 0
    mod.fake_runs = mod.calls  # alias

    def fake_run(cmd, **kw):
        mod.calls.append(list(cmd))
        code, out = 0, ""
        if "-c" in cmd:
            code, out = mod.session_code, "gpu ok"
        elif "pip" in cmd and "list" in cmd:
            out = "numpy==1"
        return SimpleNamespace(returncode=code, stdout=out, stderr="")

    class FakePopen:
        def __init__(self, cmd, **kw):
            mod.calls.append(list(cmd))
            self.cmd = list(cmd)
            self.stdout = io.StringIO("line from child\n")

        def wait(self):
            hook = mod.handlers.get("ensemble" if "--ensemble" in self.cmd else "train")
            if hook:
                return hook(self.cmd)
            return mod.train_code

    monkeypatch.setattr(subprocess, "run", fake_run)
    monkeypatch.setattr(subprocess, "Popen", FakePopen)
    return mod


def stage_world(rk, mode=None, inputs=True, seeds=(0, 1)):
    """Package + Phase 1 inputs under the fake /kaggle/input."""
    make_package(rk.root / "kaggle/input/pkg/astro_triage_pkg.zip", mode or MODE)
    if inputs:
        base = rk.root / "kaggle/input/p1/full_v0"
        put(base / "labels.parquet")
        put(base / "features.parquet")
        for f in ("full", "noind"):
            for k in seeds:
                put(base / f"lgbm_seeds/predictions_lgbm_{f}_s{k}_v0.parquet")


def done_model(rk, mid, extra_fits=None):
    """A finished model: report entry + parquet on disk (under rk.dl)."""
    put(rk.dl / f"predictions_{mid}_v0.parquet")
    return {mid: {"prediction_sha256": "abc"}}


def write_report(directory, fits, envs=None):
    put(Path(directory) / REPORT, json.dumps({"environment": envs or {}, "fits": fits}))


def run_main(rk, argv=None):
    try:
        rk.main(argv or [])
        return 0
    except SystemExit as exc:
        return exc.code


def zip_names(rk):
    with zipfile.ZipFile(rk.OUT_ZIP) as z:
        return set(z.namelist())
