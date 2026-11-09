"""scripts/evaluate_dl.py (S13): single test exposure guards. Never touches the real data/."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import evaluate_dl as E

from astro_triage.evaluation import dl_desc, dl_eval, dl_report, dl_run
from astro_triage.evaluation.runner import EvalError

REAL_OPEN = dl_eval.open_test_set
OUT = ("results_dl.json", "results_dl.md", "risk_coverage_dl.csv")


class World:
    def __init__(self, root: Path, mp):
        self.root, self.calls = root, []
        self.out = root / "data" / "out"
        self.dl = self.out / "dl"
        self.dl.mkdir(parents=True)
        (self.dl / "train_report_dl.json").write_text('{"fits": {}}')
        (self.dl / dl_eval.FREEZE_NAME).write_text("{}")
        mp.setattr(E, "REPO_ROOT", root)

        def spy(name, ret=None):
            def f(*a, **k):
                self.calls.append(name)
                if isinstance(ret, Exception):
                    raise ret
                return ret

            return f

        self.spy, self.mp = spy, mp
        self.patch()

    def patch(self, **over):
        s, mp = self.spy, self.mp
        mp.setattr(dl_eval, "open_test_set", over.get("open", s("open", ("ts", {}))))
        mp.setattr(dl_eval, "load_dl_systems", s("load", {}))
        mp.setattr(dl_run, "run_compare", over.get("cmp", s("cmp", {})))
        mp.setattr(dl_desc, "describe", over.get("desc", s("desc", ({}, []))))
        mp.setattr(
            dl_desc,
            "assemble",
            s("asm", {"meta": {"freeze_sha256": "f", "git_sha": "g"}}),
        )
        mp.setattr(dl_report, "to_json", s("json", "{}\n"))
        mp.setattr(dl_report, "render_markdown", over.get("md", s("md", "# r\n")))
        mp.setattr(dl_report, "risk_coverage_csv", s("csv", "a\n"))

    def run(self, *extra, outdir=None, approval="yes"):
        args = ["--outdir", str(outdir or self.out), *extra]
        return E.main(args + (["--approval", approval] if approval is not None else []))

    def files(self):
        return (
            [p.name for p in (self.dl / "eval").glob("*")]
            if (self.dl / "eval").exists()
            else []
        )

    @property
    def marker(self):
        return self.dl / dl_eval.MARKER_NAME


@pytest.fixture
def w(tmp_path, monkeypatch):
    return World(tmp_path, monkeypatch)


def test_happy_path_marker_written_last(w, monkeypatch, capsys):
    seen = {}
    real = dl_eval.write_marker

    def wm(dl, **k):
        seen["files"] = sorted(w.files())
        return real(dl, **k)

    monkeypatch.setattr(dl_eval, "write_marker", wm)
    assert w.run() == 0
    assert seen["files"] == sorted(OUT) and w.marker.is_file()
    assert (
        w.calls.index("open")
        < w.calls.index("load")
        < w.calls.index("cmp")
        < w.calls.index("desc")
    )
    assert capsys.readouterr().out.split() == [f"data/out/dl/eval/{n}" for n in OUT]


@pytest.mark.parametrize("stage", ["cmp", "desc", "md"])
def test_exception_leaves_no_marker_and_no_report(w, stage, capsys):
    exc = {"cmp": EvalError("boom"), "desc": ValueError("bad"), "md": OSError("disk")}[
        stage
    ]
    w.patch(**{stage: w.spy(stage, exc)})
    assert w.run() == 1
    assert not w.marker.exists() and w.files() == []
    assert "dl evaluation" in capsys.readouterr().err


def test_write_failure_midway_leaves_no_marker(w, monkeypatch):
    n = {"i": 0}
    real = E._put

    def flaky(path, text):
        n["i"] += 1
        if n["i"] == 2:
            raise OSError("disk full")
        real(path, text)

    monkeypatch.setattr(E, "_put", flaky)
    assert w.run() == 1
    assert not w.marker.exists()
    assert (
        w.files() == []
    )  # no partial report may remain (spec: no report file without the full set)


def test_marker_present_refuses_with_real_open(w, capsys):
    w.patch(open=REAL_OPEN)
    w.marker.write_text("{}")
    assert w.run() == 1
    assert "test already opened" in capsys.readouterr().err
    assert w.files() == [] and "load" not in w.calls


def test_existing_outputs_not_overwritten(w, capsys):
    (w.dl / "eval").mkdir()
    (w.dl / "eval" / OUT[1]).write_text("KEEP")
    assert w.run() == 1
    assert (w.dl / "eval" / OUT[1]).read_text() == "KEEP" and w.files() == [OUT[1]]
    assert "already exist" in capsys.readouterr().err
    assert w.calls == [] and not w.marker.exists()


def test_outputs_appearing_during_run_not_overwritten(w):
    def cmp(*a, **k):
        (w.dl / "eval").mkdir(exist_ok=True)
        (w.dl / "eval" / OUT[0]).write_text("RACE")
        return {}

    w.patch(cmp=cmp)
    assert w.run() == 1
    assert (w.dl / "eval" / OUT[0]).read_text() == "RACE" and not w.marker.exists()


def test_outdir_outside_data_refused(w, tmp_path, capsys):
    bad = tmp_path / "elsewhere"
    (bad / "dl").mkdir(parents=True)
    assert w.run(outdir=bad) == 1
    assert "inside the repo's data/" in capsys.readouterr().err and w.calls == []
    (w.root / "dl").mkdir()
    (w.root / "dl" / "train_report_dl.json").write_text(
        '{"fits": {}}'
    )  # would load fine if allowed
    assert w.run("--dl-dirname", "../../dl") == 1 and w.calls == []
    assert "inside the repo's data/" in capsys.readouterr().err


@pytest.mark.parametrize("approval", ["", "   "])
def test_empty_approval_refused_by_real_open(w, approval, capsys):
    w.patch(open=REAL_OPEN)
    assert w.run(approval=approval) == 1
    assert (
        "approval" in capsys.readouterr().err
        and not w.marker.exists()
        and w.files() == []
    )


def test_missing_approval_flag_exits_nonzero(w):
    with pytest.raises(SystemExit) as e:
        w.run(approval=None)
    assert e.value.code == 2 and w.calls == []


def test_costs_loaded_before_test_set_opened(w, capsys):
    (w.dl / "train_report_dl.json").unlink()
    assert w.run() == 1
    assert "open" not in w.calls and "cannot read costs" in capsys.readouterr().err


def test_evalerror_message_prefixed_exactly_once(w, capsys):
    w.patch(cmp=w.spy("cmp", EvalError("boom")))
    assert w.run() == 1
    assert capsys.readouterr().err == "dl evaluation refused: boom\n"
    w.patch(cmp=w.spy("cmp", EvalError("dl evaluation refused: already prefixed")))
    assert w.run() == 1
    assert capsys.readouterr().err == "dl evaluation refused: already prefixed\n"
