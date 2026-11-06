"""dl_train part 1 (spec S5): seeding, early stopping, checkpoint and resume on CPU."""

from __future__ import annotations

import contextlib
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from astro_triage.models import dl_train
from astro_triage.models.dl_nets import MLP_CONFIG, build_model, parameter_groups
from astro_triage.models.dl_train import (
    DLSpec,
    FitTensors,
    compat_key,
    fit_network,
    set_seeds,
)
from astro_triage.models.lgbm import TrainError

N, D = 300, 6


def _tensors() -> FitTensors:
    g = torch.Generator().manual_seed(123)
    x = torch.randn(N, D, generator=g)
    y = torch.randint(0, 3, (N,), generator=g)
    return FitTensors(x[:200], None, y[:200], x[200:], None, y[200:])


def _compat(**over: object) -> dict[str, object]:
    args: dict[str, object] = {
        "spec": DLSpec("mlp", "full", 0), "config": {**MLP_CONFIG, "batch_size": 64},
        "patience": 5, "preproc_sha": "aaa", "train_ids_sha": "bbb",
        "device_type": "cpu", "torch_version": "x",
    }
    args.update(over)
    return compat_key(**args)  # type: ignore[arg-type]


def _run(tmp: Path, *, max_epochs: int = 2, patience: int = 5, compat: dict | None = None):
    gen = set_seeds(1234)
    model = build_model("mlp", D, 0)
    opt = torch.optim.AdamW(parameter_groups(model, 1e-5), lr=1e-3)
    info = fit_network(
        model, opt, _tensors(), ckpt=tmp / "last.pt", compat=compat or _compat(),
        max_epochs=max_epochs, patience=patience, device="cpu", gen=gen,
    )
    return model.state_dict(), info


def _same(a: dict, b: dict) -> bool:
    return a.keys() == b.keys() and all(torch.equal(a[k], b[k]) for k in a)


def test_two_runs_bit_identical(tmp_path: Path) -> None:
    sd1, _ = _run(tmp_path / "a")
    sd2, _ = _run(tmp_path / "b")
    assert _same(sd1, sd2)


def _tree_equal(a: object, b: object) -> bool:
    if isinstance(a, dict):
        return a.keys() == b.keys() and all(_tree_equal(a[k], b[k]) for k in a)
    if isinstance(a, list):
        return len(a) == len(b) and all(map(_tree_equal, a, b))
    return torch.equal(a, b) if isinstance(a, torch.Tensor) else a == b


def _last(tmp: Path) -> dict:
    return torch.load(tmp / "last.pt", weights_only=True)


def _record(monkeypatch: pytest.MonkeyPatch, kill_at: int | None = None) -> list[int]:
    seen: list[int] = []

    def hook(epoch: int) -> None:
        seen.append(epoch)
        if epoch == kill_at:
            raise KeyboardInterrupt

    monkeypatch.setattr(dl_train, "_on_epoch_end", hook)
    return seen


@pytest.mark.parametrize(("first_max", "final_max", "kill_at"), [(3, 3, 1), (1, 2, None)])
def test_resume_matches_uninterrupted(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, first_max, final_max, kill_at) -> None:
    tmp, mp = tmp_path, monkeypatch
    ref_sd, ref_info = _run(tmp / "ref", max_epochs=final_max)
    _record(mp, kill_at)
    with contextlib.suppress(KeyboardInterrupt):
        _run(tmp / "run", max_epochs=first_max)
    assert _last(tmp / "run")["epoch"] == 1
    first_seconds = _last(tmp / "run")["epoch_seconds"][0]
    seen = _record(mp)
    sd, info = _run(tmp / "run", max_epochs=final_max)
    assert seen == list(range(2, final_max + 1))  # resumed, not restarted
    assert _same(sd, ref_sd)
    assert info["epochs_run"] == ref_info["epochs_run"] == final_max
    assert info["epoch_seconds"][0] == first_seconds
    a, b = _last(tmp / "run"), _last(tmp / "ref")
    assert len(a.pop("epoch_seconds")) == len(b.pop("epoch_seconds"))
    assert _tree_equal(a, b)  # model, optimizer, rng (cpu/cuda/shuffle), counters


@pytest.mark.parametrize(
    ("over", "key"),
    [
        ({"preproc_sha": "zzz"}, "preproc_sha"),
        ({"spec": DLSpec("mlp", "full", 1)}, "seed"),
        ({"patience": 9}, "patience"),
        ({"device_type": "cuda"}, "device_type"),
    ],
)
def test_incompatible_checkpoint_names_key(tmp_path: Path, over: dict, key: str) -> None:
    _run(tmp_path, max_epochs=1)
    patience = over.get("patience", 5)
    with pytest.raises(TrainError, match=key):
        _run(tmp_path, max_epochs=2, patience=patience, compat=_compat(**over))


def test_patience_argument_must_match_compat(tmp_path: Path) -> None:
    with pytest.raises(TrainError, match="patience"):
        _run(tmp_path, patience=3)  # compat says 5


def test_best_pt_mismatch_with_last_raises(tmp_path: Path) -> None:
    _run(tmp_path, max_epochs=1)
    pack = torch.load(tmp_path / "best.pt", weights_only=True)
    pack["best_epoch"] = 7
    torch.save(pack, tmp_path / "best.pt")
    with pytest.raises(TrainError, match="does not match"):
        _run(tmp_path, max_epochs=1)  # resumes with nothing left to run, final check fires


def test_stale_best_without_last_is_discarded(tmp_path: Path) -> None:
    tmp_path.mkdir(exist_ok=True)
    torch.save({"model": {}, "best_epoch": 9, "best_val_loss": 0.0}, tmp_path / "best.pt")
    assert _run(tmp_path, max_epochs=1)[1]["best_epoch"] == 1


def test_resume_cuda_rng_without_cuda_raises(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _run(tmp_path, max_epochs=1)
    state = _last(tmp_path)
    state["rng"]["torch_cuda"] = [torch.zeros(816, dtype=torch.uint8)]
    torch.save(state, tmp_path / "last.pt")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(TrainError, match="CUDA"):
        _run(tmp_path, max_epochs=2)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 2)  # count differs from stored 1
    with pytest.raises(TrainError, match="CUDA"):
        _run(tmp_path, max_epochs=2)


def test_patience_one_stops_and_restores_best(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    epoch1_sd, _ = _run(tmp_path / "e1", max_epochs=1)
    losses = iter([1.0, 2.0, 3.0, 4.0])
    monkeypatch.setattr(dl_train, "_val_logloss", lambda *a, **k: next(losses))
    sd, info = _run(tmp_path / "p", max_epochs=10, patience=1, compat=_compat(patience=1))
    assert info["epochs_run"] == 2
    assert info["best_epoch"] == 1
    assert _same(sd, epoch1_sd)


def test_torch_load_uses_weights_only(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[object] = []
    real = torch.load

    def spy(*args: object, **kwargs: object):
        calls.append(kwargs.get("weights_only"))
        return real(*args, **kwargs)

    _run(tmp_path, max_epochs=1)
    monkeypatch.setattr(torch, "load", spy)
    _run(tmp_path, max_epochs=2)
    assert len(calls) >= 2
    assert all(c is True for c in calls)


def test_nan_val_loss_raises(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(dl_train, "_val_logloss", lambda *a, **k: float("nan"))
    with pytest.raises(TrainError, match="non-finite"):
        _run(tmp_path)


@pytest.mark.parametrize(("kw", "key"), [({"max_epochs": 0}, "max_epochs"), ({"patience": 0}, "patience")])
def test_nonpositive_limits_raise(tmp_path: Path, kw: dict, key: str) -> None:
    with pytest.raises(TrainError, match=f"{key} must be >= 1"):
        _run(tmp_path, compat=_compat(patience=kw.get("patience", 5)), **kw)


def test_set_seeds_reproduces_randperm() -> None:
    a = torch.randperm(50, generator=set_seeds(7))
    b = torch.randperm(50, generator=set_seeds(7))
    c = torch.randperm(50, generator=set_seeds(8))
    assert torch.equal(a, b)
    assert not torch.equal(a, c)
