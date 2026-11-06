"""scripts/train_dl.py without torch: GPU session check, label stripping, refusals, import hygiene."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from astro_triage.models.lgbm import Fold, TrainError

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))
import train_dl as mod


def session(
    name="Tesla T4",
    cap=(7, 5),
    arch=("sm_75",),
    pinned=None,
    version="2.5.1",
    cuda="12.1",
) -> None:
    mod.check_gpu_session(version, cuda, name, cap, list(arch), pinned)


def test_p100_by_name_and_by_capability_refused() -> None:
    with pytest.raises(TrainError, match="sm_60"):
        session("Tesla P100-PCIE-16GB", (6, 0), ("sm_60", "sm_75"))
    with pytest.raises(TrainError, match="sm_60"):
        session("Unknown GPU", (6, 0), ("sm_60",))


def test_t4_with_listed_arch_and_equal_pin_passes() -> None:
    session(pinned=None)
    session(pinned="2.5.1")


def test_missing_arch_pin_mismatch_and_cpu_build_refused() -> None:
    with pytest.raises(TrainError, match="lacks sm_75"):
        session(arch=("sm_70", "sm_80"))
    with pytest.raises(TrainError, match="differs"):
        session(pinned="2.4.0")
    with pytest.raises(TrainError, match="CPU build"):
        session(cuda=None)


def make_fold(n: int = 6) -> Fold:
    y = (np.arange(n) % 3).astype(np.int64)
    names = np.array(["STAR", "GALAXY", "QSO"])[y]
    frame = pd.DataFrame({"x": np.arange(n, dtype=float), "sdss_class": names})
    return Fold(np.arange(n) + 100, y, frame)


def test_strip_test_labels_removes_labels_and_class_column_only_from_the_copy() -> None:
    fold = make_fold()
    out = mod.strip_test_labels(fold)
    assert (out.labels == -1).all() and len(out.labels) == 6
    assert "sdss_class" not in out.frame.columns
    np.testing.assert_array_equal(out.source_id, fold.source_id)
    np.testing.assert_array_equal(out.frame["x"], fold.frame["x"])
    assert "sdss_class" in fold.frame.columns and fold.labels.tolist() == [
        0,
        1,
        2,
        0,
        1,
        2,
    ]


@pytest.fixture
def repo(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setattr(mod, "REPO_ROOT", tmp_path)
    (tmp_path / "data" / "rel").mkdir(parents=True)
    return tmp_path


def one_line_failure(
    argv: list[str], capsys: pytest.CaptureFixture[str], needle: str
) -> None:
    assert mod.main(argv) == 1
    err = capsys.readouterr().err.strip().splitlines()
    assert len(err) == 1 and err[0].startswith("train_dl failed:") and needle in err[0]


@pytest.mark.parametrize("sub", ["outside", "data/../escape"])
def test_outdir_outside_data_refused(repo, capsys, sub) -> None:
    target = repo.parent / "outside_dl_rel" if sub == "outside" else repo / sub
    one_line_failure(["--arch", "mlp", "--outdir", str(target)], capsys, "data/")
    assert not (repo / "escape").exists() and not target.exists()


def test_ckpt_dir_outside_data_refused(repo, capsys) -> None:
    out = repo / "data" / "rel"
    argv = ["--arch", "mlp", "--outdir", str(out), "--ckpt-dir", str(repo / "ckpt")]
    one_line_failure(argv, capsys, "--ckpt-dir")
    assert not (out / "dl").exists()


def test_exclusive_subsets_and_cuda_needs_platform(repo, capsys) -> None:
    base = ["--arch", "mlp", "--outdir", str(repo / "data" / "rel")]
    both = [*base, "--train-subset", "5", "--plan-c-subsample", "5"]
    one_line_failure(both, capsys, "mutually exclusive")
    one_line_failure([*base, "--device", "cuda"], capsys, "--platform")
    one_line_failure([*base, "--train-subset", "0"], capsys, ">= 1")


@pytest.mark.parametrize("text", ["5", "0-5", "2,2", "a-b", "-1", ""])
def test_bad_seeds_rejected(text) -> None:
    with pytest.raises(TrainError):
        mod.parse_seeds(text)


def test_cublas_set_and_torch_not_imported_at_module_import() -> None:
    code = (
        f"import sys, os; sys.path.insert(0, {str(SCRIPTS)!r}); import train_dl; "
        "print('torch' in sys.modules, os.environ.get('CUBLAS_WORKSPACE_CONFIG'))"
    )
    env = {k: v for k, v in os.environ.items() if k != "CUBLAS_WORKSPACE_CONFIG"}
    res = subprocess.run(
        [sys.executable, "-I", "-c", code],
        capture_output=True,
        text=True,
        env=env,
        check=False,
    )
    assert res.stdout.strip() == "False :4096:8", res.stderr
