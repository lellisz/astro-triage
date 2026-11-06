"""scripts/train_dl.py guards that need the real train_one (CPU, 2 epochs): existing target, single pin."""

from __future__ import annotations

import json

import pytest

pytest.importorskip("torch")

from test_train_dl_e2e import PRED, argv, mod, repo  # noqa: F401

from astro_triage.models import dl_session


def test_existing_prediction_without_report_entry_is_refused(repo, capsys) -> None:  # noqa: F811
    dl = repo / "data" / "rel" / "dl"
    dl.mkdir(parents=True)
    (dl / PRED).write_bytes(b"precious")
    assert mod.main(argv(repo)) == 1
    assert "refusing to overwrite" in capsys.readouterr().err
    assert (dl / PRED).read_bytes() == b"precious"
    assert not (dl / "models" / "mlp_full_s0.pt").exists()


def test_pilot_pins_environment_and_other_torch_refused_on_cuda(repo, monkeypatch, capsys) -> None:  # noqa: F811
    assert mod.main(argv(repo, "--train-subset", "90")) == 0
    env_path = repo / "data" / "rel" / "dl" / "environment.json"
    pinned = json.loads(env_path.read_text(encoding="utf-8"))["local"]["torch"]
    t4 = {"torch": "0.0.0-other", "cuda": "12.1", "gpu": "Tesla T4", "capability": (7, 5), "arch_list": ["sm_75"]}
    monkeypatch.setattr(dl_session, "gpu_facts", lambda: dict(t4))
    capsys.readouterr()
    assert mod.main(argv(repo, "--device", "cuda", "--platform", "local")) == 1
    err = capsys.readouterr().err
    assert "differs" in err and pinned in err
    assert json.loads(env_path.read_text(encoding="utf-8"))["local"]["torch"] == pinned
    assert not (repo / "data" / "rel" / "dl" / PRED).exists()
