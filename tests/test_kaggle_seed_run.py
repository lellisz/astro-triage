"""Seeds mode of kaggle/run_kaggle.py: stage plans, remaining fits, parity path, zip guards.

Pure local: stage functions are patched, no network, no Kaggle, no training. The module
under test can be swapped with KAGGLE_RUNNER_UNDER_TEST (used for mutation checks).
"""

import importlib.util
import json
import os
import re
import sys
import zipfile
from pathlib import Path

import pytest

DEFAULT = Path(__file__).resolve().parents[1] / "kaggle" / "run_kaggle.py"
PHASE1 = "pacote pip retomada specobj candidatos xmatch dataset treino avaliação zip"
HEAD7 = PHASE1.split()[:7]
CONTINUE = [*HEAD7, "paridade", "retomada_seeds", "seeds_measure_one", "seeds_rest"]
# fmt: off
SEEDS_FUNCS = ["check_parity", "restore_seed_outputs", "seeds_script_cmd", "seeds_measure_one",
               "seeds_rest", "package_seeds", "run_seeds_tail", "read_seed_report", "remaining_fits"]
# fmt: on


@pytest.fixture
def rk(tmp_path, monkeypatch):
    path = Path(os.environ.get("KAGGLE_RUNNER_UNDER_TEST", DEFAULT))
    spec = importlib.util.spec_from_file_location("rk_under_test", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    w = tmp_path / "working"
    repo = w / "repo"
    ds = repo / "data/interim/full_v0"
    for d in (tmp_path / "input", w, repo / "kaggle", repo / "scripts", ds):
        d.mkdir(parents=True)
    values = {
        "INPUT": str(tmp_path / "input"), "WORKING": str(w), "TEMP": str(tmp_path / "t"),
        "REPO": str(repo), "XMATCH_DIR": str(w / "xm"), "XMATCH_WORK": str(tmp_path / "xw"),
        "LOG_PATH": str(w / "run_log.txt"), "ENV_PATH": str(w / "environment.json"),
        "OUT_ZIP": str(w / "out.zip"), "DATASET_DIR": str(ds),
        "PARITY_REPORT": str(w / "parity_report.json"),
        "SEEDS_RUN_PATH": str(w / "seeds_run.json"),
    }  # fmt: skip
    for key, value in values.items():
        monkeypatch.setattr(mod, key, value)
    monkeypatch.setattr(mod, "SEEDS_STATE", {})
    monkeypatch.setattr(mod, "check_disk", lambda stage: None)
    for name in ("find_package", "install_dependencies", "restore_previous_outputs",
                 "download_specobj", "build_candidates", "package_outputs"):  # fmt: skip
        monkeypatch.setattr(mod, name, lambda: None)
    (w / "environment.json").write_text("{}")
    (w / "run_log.txt").write_text("log")
    mod.calls, mod.logs, mod.state = [], [], {"fits": {}}
    monkeypatch.setattr(mod, "log", mod.logs.append)
    monkeypatch.setattr(mod, "read_seed_report", lambda: mod.state)

    def fake_stream(label, cmd, cwd=None):
        mod.calls.append((label, list(cmd)))
        if "--measure-one" in cmd:
            mod.state["fits"]["lgbm_full_s0"] = {"wall_seconds": 1.0}
        elif "--features" in cmd:
            name = cmd[cmd.index("--features") + 1]
            for k in cmd[cmd.index("--seeds") + 1].split(","):
                mod.state["fits"][f"lgbm_{name}_s{k}"] = {"wall_seconds": 1.0}

    monkeypatch.setattr(mod, "run_stream", fake_stream)
    return mod


def set_mode(rk, **mode):
    Path(rk.REPO, "kaggle/kaggle_mode.json").write_text(json.dumps(mode))


def stages(rk):
    return [m.split("]")[0][1:] for m in rk.logs if m.endswith("] start")]


def test_default_mode_and_phase1_commands_unchanged(rk):
    assert rk.read_mode() == {"mode": "phase1"}
    assert rk.plan_stages(rk.read_mode()) == PHASE1.split()
    Path(rk.REPO, "scripts/evaluate.py").write_text("")
    rk.main()
    py, repo, ds = sys.executable, rk.REPO, rk.DATASET_DIR
    assert rk.calls == [
        ("[xmatch] xmatch_full", [py, f"{repo}/scripts/xmatch_full.py", "--candidates",
            rk.CANDIDATES_PATH, "--outdir", rk.XMATCH_DIR, "--workdir", rk.XMATCH_WORK,
            "--chunk-size", "1000000", "--radius", "2.0", "--sleep", "20"]),
        ("[dataset] build_dataset", [py, f"{repo}/scripts/build_dataset.py",
            "--dataset-name", "full_v0", "--xmatch-dir", rk.XMATCH_DIR,
            "--candidates", rk.CANDIDATES_PATH]),
        ("[treino] train_lgbm", [py, f"{repo}/scripts/train_lgbm.py", "--outdir", ds]),
        ("[avaliação] evaluate", [py, f"{repo}/scripts/evaluate.py", "--outdir", ds]),
    ]  # fmt: skip


@pytest.mark.parametrize("after", ["stop", "continue"])
def test_seeds_plan_and_commands(rk, after):
    mode = {"mode": "seeds", "after_measure_one": after}
    expected = [*CONTINUE] if after == "continue" else CONTINUE[:-1]
    assert rk.plan_stages(mode) == [*expected, "zip_seeds"]
    set_mode(rk, **mode)
    rk.check_parity = lambda: True
    rk.package_seeds = lambda: None
    rk.main()
    assert stages(rk) == [*expected, "zip_seeds"]
    assert not any(
        re.search(r"train_lgbm\.py|evaluate", " ".join(c)) for _, c in rk.calls
    )
    assert ("lgbm_noind_s4" in rk.state["fits"]) == (after == "continue")


def test_remaining_fits(rk):
    full = {"full": [1, 2, 3, 4], "noind": [0, 1, 2, 3, 4]}
    assert rk.remaining_fits({}) == full
    assert rk.remaining_fits({"fits": {}}) == full
    one = rk.remaining_fits({"fits": {"lgbm_full_s0": {}, "lgbm_full_s2": {}}})
    assert one == {"full": [1, 3, 4], "noind": [0, 1, 2, 3, 4]}
    ten = {f"lgbm_{n}_s{k}": {} for n in ("full", "noind") for k in range(5)}
    assert rk.remaining_fits({"fits": ten}) == {"full": [], "noind": []}
    dup = ["lgbm_full_s1", "lgbm_full_s1"]  # a duplicated id must not duplicate output
    assert rk.remaining_fits({"fits": dup})["full"] == [2, 3, 4]


def test_seeds_rest_skips_restored_fits(rk):
    rk.state["fits"].update(
        {"lgbm_full_s0": {}, "lgbm_full_s1": {}, "lgbm_noind_s0": {}}
    )
    rk.seeds_rest()
    args = {
        c[c.index("--features") + 1]: c[c.index("--seeds") + 1] for _, c in rk.calls
    }
    assert args == {"full": "2,3,4", "noind": "1,2,3,4"}


@pytest.mark.parametrize(("code", "lines"), [(3, ["MISMATCH: rows", "noise"]), (1, [])])
def test_mismatch_or_error_packages_report_and_trains_nothing(rk, code, lines):
    set_mode(rk, mode="seeds", after_measure_one="continue")
    rk.run_stream_code = lambda label, cmd, cwd=None: (code, lines)
    with pytest.raises(SystemExit):
        rk.main()
    assert [c for c in rk.calls if "seeds" in c[0] or "treino" in c[0]] == []
    assert ("[paridade] MISMATCH: rows" in rk.logs) == (code == 3)
    assert not any("noise" in m for m in rk.logs)
    with zipfile.ZipFile(rk.OUT_ZIP) as z:
        names = z.namelist()
        report = json.loads(z.read("parity_report.json"))
    assert "parity_report.json" in names and not any(
        n.startswith("lgbm_") for n in names
    )
    assert report["match"] is False


def seeded_zip(rk, extra=()):
    base = Path(rk.DATASET_DIR, "lgbm_seeds")
    (base / "models").mkdir(parents=True)
    (base / "p.parquet").write_bytes(b"x")
    for name in extra:
        (base / name).write_bytes(b"x")
    Path(rk.PARITY_REPORT).write_text(json.dumps({"match": True}))
    rk.state["fits"] = {"lgbm_full_s0": {}}


def test_zip_seeds_members_and_no_test_keys(rk):
    seeded_zip(rk)
    rk.SEEDS_STATE.update({"mode": "seeds", "stage_seconds": {"zip_seeds": 1.0}})
    rk.package_seeds()
    with zipfile.ZipFile(rk.OUT_ZIP) as z:
        names = z.namelist()
        run = z.read("seeds_run.json").decode()
    ok = re.compile(
        r"lgbm_seeds/.*|parity_report\.json|seeds_run\.json|environment\.json|run_log\.txt"
    )
    assert names and all(ok.fullmatch(n) for n in names)
    assert not any(n.endswith(("labels.parquet", "features.parquet")) for n in names)
    assert "test" not in run.lower()


@pytest.mark.parametrize("banned", ["labels.parquet", "features.parquet"])
def test_zip_seeds_refuses_full_v0_parquets(rk, banned):
    seeded_zip(rk, extra=[banned])
    with pytest.raises(SystemExit):
        rk.package_seeds()
    assert not Path(rk.OUT_ZIP).exists()


def test_zip_seeds_requires_parity_report(rk):
    seeded_zip(rk)
    Path(rk.PARITY_REPORT).unlink()
    with pytest.raises(SystemExit):
        rk.package_seeds()


def test_seeds_functions_never_reference_test_labels_or_evaluate():
    src = Path(os.environ.get("KAGGLE_RUNNER_UNDER_TEST", DEFAULT)).read_text("utf-8")
    for name in SEEDS_FUNCS:
        body = re.search(rf"(?sm)^def {name}\(.*?(?=^def |^[A-Z_]+ =|\Z)", src)
        assert body, name
        assert not re.search(r"(?i)test|evaluate|train_lgbm\.py", body.group(0)), name
