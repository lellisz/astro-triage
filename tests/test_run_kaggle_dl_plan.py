# ruff: noqa: F811
"""kaggle/run_kaggle_dl.py: dry run, mode file, system order, session, stages, inputs, ensemble."""

import json

import pytest
from kaggle_dl_kit import (  # noqa: F401
    MODE,
    make_package,
    put,
    rk,
    run_main,
    stage_world,
    write_report,
)


# fmt: off
def cfg(rk, **over):
    return put(rk.root / "mode.json", json.dumps({**MODE, **over}))


def logtext(rk):
    return (rk.work / "run_log.txt").read_text()


def trains(rk):
    return [c for c in rk.calls if any(str(p).endswith("train_dl.py") for p in c)]


def test_dry_run_prints_plan_runs_and_writes_nothing(rk, capsys):
    path = cfg(rk)
    before = sorted(p for p in rk.root.rglob("*"))
    assert run_main(rk, ["--dry-run", "--config", str(path)]) == 0
    out = capsys.readouterr().out
    assert "[plano]" in out and "ftt" in out and "never run: --freeze" in out
    assert rk.calls == []
    assert sorted(p for p in rk.root.rglob("*")) == before
    assert not (rk.work / "run_log.txt").exists()


@pytest.mark.parametrize("drop", ["seeds", "max_epochs", "patience", "train_subset", "ensemble"])
def test_missing_mode_key_fails_clearly(rk, drop, capsys):
    mode = {k: v for k, v in MODE.items() if k != drop}
    p = put(rk.root / "m.json", json.dumps(mode))
    code = run_main(rk, ["--dry-run", "--config", str(p)])
    assert code != 0 and "keys must be exactly" in str(code)


def test_extra_mode_key_and_bad_json_fail(rk):
    p = put(rk.root / "m.json", json.dumps({**MODE, "freeze": True}))
    assert "keys must be exactly" in str(run_main(rk, ["--dry-run", "--config", str(p)]))
    p = put(rk.root / "m2.json", "{not json")
    assert "cannot read" in str(run_main(rk, ["--dry-run", "--config", str(p)]))


def test_ensemble_with_train_subset_is_an_error(rk):
    p = cfg(rk, ensemble=True, train_subset=1000)
    assert "not allowed with 'train_subset'" in str(run_main(rk, ["--dry-run", "--config", str(p)]))


@pytest.mark.parametrize("over", [{"seeds": []}, {"seeds": [0, 0]}, {"seeds": [5]}, {"seeds": [True]},
                                  {"max_epochs": 0}, {"patience": "2"}, {"train_subset": 0}, {"ensemble": 1}])
def test_invalid_mode_values_fail(rk, over):
    assert run_main(rk, ["--dry-run", "--config", str(cfg(rk, **over))]) not in (0, None)


def test_system_order_and_never_freeze(rk):
    stage_world(rk, {**MODE, "ensemble": True})
    write_report(rk.dl, {})  # nothing finished
    rk.handlers["train"] = lambda cmd: 0
    assert run_main(rk) == 0
    order = [(c[c.index("--arch") + 1], c[c.index("--features") + 1]) for c in trains(rk) if "--ensemble" not in c]
    assert order == [("ftt", "full"), ("mlp", "full"), ("ftt", "noind"), ("mlp", "noind")]
    for c in rk.calls:
        assert "--freeze" not in c
        if any(str(p).endswith("train_dl.py") for p in c):
            assert c[c.index("--device") + 1] in ("cuda", "cpu") and c[c.index("--platform") + 1] == "kaggle"


def test_dry_run_plan_never_mentions_freeze_command(rk, capsys):
    run_main(rk, ["--dry-run", "--config", str(cfg(rk, ensemble=True))])
    plan = [l for l in capsys.readouterr().out.splitlines() if "train_dl.py" in l]
    assert plan and all("--freeze" not in l for l in plan)


def test_p100_session_aborts_before_any_training(rk):
    stage_world(rk)
    rk.session_code = 1
    assert "GPU/torch check failed" in str(run_main(rk))
    assert trains(rk) == []
    assert "[sessao_gpu] start" in logtext(rk)
    assert not (rk.work / "astro_triage_dl_outputs.zip").exists()  # zip is only in the train try/finally


def test_session_snippet_calls_verify_gpu_and_exits_on_trainerror(rk):
    s = rk.SESSION_SNIPPET
    assert "verify_gpu" in s and "TrainError" in s and "session check failed" in s


def test_stages_log_start_and_done_in_order(rk):
    stage_world(rk, {**MODE, "ensemble": True})
    assert run_main(rk) == 0
    lines = logtext(rk).splitlines()
    names = ["pacote", "pip", "entradas", "sessao_gpu", "retomada", "treino", "ensemble", "zip"]
    idx = []
    for n in names:
        s = next(i for i, l in enumerate(lines) if f"[{n}] start" in l)
        d = next(i for i, l in enumerate(lines) if f"[{n}] done" in l)
        assert s < d
        idx.append(s)
    assert idx == sorted(idx)


def test_failed_stage_still_logs_done(rk):
    stage_world(rk)
    rk.train_code = 3
    assert "exited with code 3" in str(run_main(rk))
    assert "[treino] done" in logtext(rk) and "[zip] done" in logtext(rk)


@pytest.mark.parametrize("missing", ["labels.parquet", "features.parquet", "lgbm_seeds/predictions_lgbm_full_s1_v0.parquet"])
def test_missing_inputs_fail_clearly_and_nothing_trains(rk, missing):
    stage_world(rk)
    (rk.root / "kaggle/input/p1/full_v0" / missing).unlink()
    code = run_main(rk)
    assert "[entradas] not found" in str(code) and missing in str(code)
    assert trains(rk) == []


def test_no_inputs_at_all_fails(rk):
    stage_world(rk, inputs=False)
    assert "[entradas] not found" in str(run_main(rk))


def test_missing_package_fails(rk):
    assert "[pacote]" in str(run_main(rk))


def test_package_without_mode_file_fails(rk):
    make_package(rk.root / "kaggle/input/pkg/astro_triage_pkg.zip", None)
    assert "kaggle_dl_mode.json missing" in str(run_main(rk))


def test_pip_excludes_torch_and_skip_list(rk):
    stage_world(rk)
    run_main(rk)
    pip = next(c for c in rk.calls if "install" in c)
    assert "numpy==1" in pip and not any(p.startswith(("torch", "pytest")) for p in pip)


def test_ensemble_runs_only_when_all_full_seeds_finished_and_on_cpu(rk):
    stage_world(rk, {**MODE, "ensemble": True})

    def hook(cmd):  # fake training of every full seed of mlp only
        if "--ensemble" not in cmd and cmd[cmd.index("--features") + 1] == "full" and cmd[cmd.index("--arch") + 1] == "mlp":
            for k in (0, 1):
                put(rk.dl / f"predictions_mlp_full_s{k}_v0.parquet")
            write_report(rk.dl, {f"mlp_full_s{k}": {"prediction_sha256": "h"} for k in (0, 1)})
        return 0
    rk.handlers["train"] = hook
    assert run_main(rk) == 0
    ens = [c for c in rk.calls if "--ensemble" in c]
    assert len(ens) == 1 and ens[0][ens[0].index("--arch") + 1] == "mlp"
    assert ens[0][ens[0].index("--device") + 1] == "cpu"
    assert "ftt: not every ftt_full seed is finished, skipped" in logtext(rk)


def test_ensemble_off_never_runs_ensemble(rk):
    stage_world(rk)
    run_main(rk)
    assert not [c for c in rk.calls if "--ensemble" in c]


def test_nothing_written_outside_working(rk):
    stage_world(rk, {**MODE, "ensemble": True})
    before = {p for p in rk.root.rglob("*") if "working" not in p.parts and "input" not in p.parts}
    run_main(rk)
    after = {p for p in rk.root.rglob("*") if "working" not in p.parts and "input" not in p.parts}
    assert after == before
