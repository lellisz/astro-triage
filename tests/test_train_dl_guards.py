"""scripts/train_dl.py guards without torch: a fake dl_train stands in; GPU facts, unlink and report merge."""

from __future__ import annotations

import json
import sys
import types
from dataclasses import dataclass
from pathlib import Path

import pytest
from dl_refs import FRACS, write_lgbm_ref
from test_train_dl import make_fold

import astro_triage.models as models_pkg
from astro_triage.models import dl_session

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import train_dl as mod

T4 = {"torch": "2.5.1", "cuda": "12.1", "gpu": "Tesla T4", "capability": (7, 5), "arch_list": ["sm_75"]}
P100 = {**T4, "gpu": "Tesla P100", "capability": (6, 0), "arch_list": ["sm_60"]}
PRED = "predictions_mlp_full_s0_v0.parquet"


@dataclass
class Spec:
    arch: str
    features: str
    k: int

    @property
    def model_id(self) -> str:
        return f"{self.arch}_{self.features}_s{self.k}"

    @property
    def seed(self) -> int:
        return 20261025 + self.k


class Fake:
    """Stands in for dl_train: records calls, writes the prediction file, returns a fixed T2 verdict."""

    def __init__(self, beats: bool = True) -> None:
        self.beats, self.calls = beats, []
        self.DLSpec = Spec

    def train_one(self, spec, folds, outdir, ckpt_dir, **kw):
        self.calls.append((spec, folds))
        outdir.mkdir(parents=True, exist_ok=True)
        (outdir / f"predictions_{spec.model_id}_{mod.MODEL_VERSION}.parquet").write_bytes(b"p")
        return {"model_id": spec.model_id, "beats_prior_logloss": self.beats,
                "prediction_file": "p", "model_file": "m",
                "train_class_fractions": FRACS}


@pytest.fixture
def repo(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setattr(mod, "REPO_ROOT", tmp_path)
    (tmp_path / "data" / "rel").mkdir(parents=True)
    write_lgbm_ref(tmp_path / "data" / "rel", "full", 0, FRACS)
    folds = {n: make_fold() for n in ("train", "val", "test")}
    monkeypatch.setattr(mod, "load_folds", lambda _o: dict(folds))
    return tmp_path


@pytest.fixture
def fake(monkeypatch: pytest.MonkeyPatch) -> Fake:
    f = Fake()
    monkeypatch.setitem(sys.modules, "astro_triage.models.dl_train", f)
    monkeypatch.setattr(models_pkg, "dl_train", f, raising=False)
    return f


def gpu(monkeypatch: pytest.MonkeyPatch, facts: dict) -> None:
    monkeypatch.setattr(dl_session, "gpu_facts", lambda: dict(facts))


def run(repo: Path, *extra: str, arch: str = "mlp") -> int:
    base = ["--arch", arch, "--features", "full", "--seeds", "0", "--outdir", str(repo / "data" / "rel")]
    return mod.main([*base, *extra])


CUDA = ("--device", "cuda", "--platform", "kaggle")


def test_p100_exits_1_before_train_one_and_writes_no_environment(repo, fake, monkeypatch, capsys) -> None:
    gpu(monkeypatch, P100)
    assert run(repo, *CUDA) == 1
    assert "sm_60" in capsys.readouterr().err
    assert fake.calls == []
    assert not (repo / "data" / "rel" / "dl" / "environment.json").exists()


def test_train_one_receives_test_fold_without_labels_or_class_column(repo, fake) -> None:
    assert run(repo) == 0
    folds = fake.calls[0][1]
    assert (folds["test"].labels == -1).all()
    assert "sdss_class" not in folds["test"].frame.columns
    assert "sdss_class" in folds["val"].frame.columns and (folds["val"].labels >= 0).all()


def test_symlinked_dl_pointing_outside_data_refused(repo, fake, capsys) -> None:
    outside = repo / "outside"
    outside.mkdir()
    try:
        (repo / "data" / "rel" / "dl").symlink_to(outside, target_is_directory=True)
    except (OSError, NotImplementedError):
        pytest.skip("symlinks not permitted on this OS/user")
    assert run(repo) == 1
    assert "data/" in capsys.readouterr().err
    assert fake.calls == [] and list(outside.iterdir()) == []


def test_pin_from_pilot_run_refuses_other_torch_version(repo, fake, monkeypatch, capsys) -> None:
    gpu(monkeypatch, T4)
    assert run(repo, *CUDA, "--train-subset", "5") == 0
    dl = repo / "data" / "rel" / "dl"
    assert json.loads((dl / "environment.json").read_text())["kaggle"]["torch"] == "2.5.1"
    gpu(monkeypatch, {**T4, "torch": "2.6.0"})
    capsys.readouterr()
    assert run(repo, *CUDA) == 1
    assert "differs" in capsys.readouterr().err
    assert len(fake.calls) == 1 and not (dl / PRED).exists()


def test_t2_failure_entry_saved_before_unlink_and_stuck_files_listed(repo, fake, monkeypatch, capsys) -> None:
    fake.beats = False
    monkeypatch.setattr(dl_session.time, "sleep", lambda _s: None)
    dl = repo / "data" / "rel" / "dl"
    seen = []

    def deny(self, missing_ok=False):
        if "mlp_full_s0" in self.name:
            seen.append(json.loads((dl / mod.REPORT_NAME).read_text())["fits"]["mlp_full_s0"]["status"])
            raise PermissionError(self.name)

    monkeypatch.setattr(Path, "unlink", deny)
    assert run(repo) == 1
    err = capsys.readouterr().err
    assert "could not delete" in err and PRED in err
    assert seen and set(seen) == {"failed_T2_beats_prior"}
    assert (dl / PRED).exists()


def fail_n_times(monkeypatch, n: int, platform: str):
    state = {"calls": 0, "sleeps": []}
    real = Path.unlink

    def flaky(self, missing_ok=False):
        state["calls"] += 1
        if state["calls"] <= n:
            raise PermissionError("locked")
        real(self, missing_ok=missing_ok)

    monkeypatch.setattr(Path, "unlink", flaky)
    monkeypatch.setattr(dl_session, "sys", types.SimpleNamespace(platform=platform))
    monkeypatch.setattr(dl_session, "time", types.SimpleNamespace(sleep=state["sleeps"].append))
    return state


def test_unlink_retries_on_win32_then_succeeds(tmp_path, monkeypatch) -> None:
    f = tmp_path / "a"
    f.write_text("x")
    state = fail_n_times(monkeypatch, 2, "win32")
    assert dl_session.unlink_all([f]) == []
    assert state["calls"] == 3 and state["sleeps"] == [0.05, 0.1]
    monkeypatch.undo()
    assert not f.exists()


def test_unlink_gives_up_after_five_on_win32_and_tries_once_elsewhere(tmp_path, monkeypatch) -> None:
    f = tmp_path / "a"
    state = fail_n_times(monkeypatch, 99, "win32")
    assert dl_session.unlink_all([f]) == [f] and state["calls"] == 5
    state = fail_n_times(monkeypatch, 99, "linux")
    assert dl_session.unlink_all([f]) == [f] and state["calls"] == 1 and state["sleeps"] == []


def test_report_merges_platforms_and_architectures(repo, fake, monkeypatch) -> None:
    gpu(monkeypatch, T4)
    assert run(repo, "--device", "cuda", "--platform", "kaggle") == 0
    gpu(monkeypatch, {**T4, "torch": "2.6.0"})
    assert run(repo, "--device", "cuda", "--platform", "colab", arch="ftt") == 0
    report = json.loads((repo / "data" / "rel" / "dl" / mod.REPORT_NAME).read_text())
    assert {k: v["torch"] for k, v in report["environment"].items()} == {"kaggle": "2.5.1", "colab": "2.6.0"}
    fits = report["fits"]
    assert (fits["mlp_full_s0"]["platform"], fits["mlp_full_s0"]["torch_version"]) == ("kaggle", "2.5.1")
    assert (fits["ftt_full_s0"]["platform"], fits["ftt_full_s0"]["torch_version"]) == ("colab", "2.6.0")
