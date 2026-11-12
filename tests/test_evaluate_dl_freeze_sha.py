"""evaluate_dl wiring: the freeze sha in results_dl.json and test_opened.json is that of the bytes loaded."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
from dl_eval_kit import frozen_env

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import evaluate_dl as E

from astro_triage.evaluation import dl_desc, dl_eval, dl_report, dl_run


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _run(tmp_path, monkeypatch, on_compare=None, on_read=None):
    """Real load_freeze/open_test_set/assemble/write_marker; only the heavy stages are synthetic."""
    root = tmp_path / "data"
    out, dl, _, _ = frozen_env(root)
    freeze_path = dl / dl_eval.FREEZE_NAME
    (dl / "train_report_dl.json").write_text('{"fits": {}}')
    ts = type("TS", (), {"y": np.zeros(3), "pixel": np.arange(3)})()
    loaded = _sha(freeze_path)

    def compare(*a, **k):
        if on_compare:
            on_compare(freeze_path)
        return {}

    if on_read:  # file changes right after load_freeze's first read_bytes (kills a second-read sha)
        real_read = Path.read_bytes

        def read_bytes(self):
            data = real_read(self)
            if self.resolve() == freeze_path.resolve():
                on_read(freeze_path)
            return data

        monkeypatch.setattr(Path, "read_bytes", read_bytes)
    monkeypatch.setattr(E, "REPO_ROOT", tmp_path)
    monkeypatch.setitem(dl_eval._TEST_OPENED, "done", False)
    monkeypatch.setattr(dl_eval, "load_test_set", lambda _out: ts)
    monkeypatch.setattr(dl_eval, "load_dl_systems", lambda *a, **k: {})
    monkeypatch.setattr(dl_run, "run_compare", compare)
    monkeypatch.setattr(dl_desc, "describe", lambda *a, **k: ({}, []))
    monkeypatch.setattr(dl_report, "render_markdown", lambda res: "# r\n")
    monkeypatch.setattr(dl_report, "risk_coverage_csv", lambda rows: "a\n")
    args = ["--outdir", str(out), "--dl-dirname", "../dl", "--seeds-dirname", "../lg", "--approval", "ok go"]
    assert E.main(args) == 0
    results = json.loads((dl / "eval" / "results_dl.json").read_text(encoding="utf-8"))
    marker = json.loads((dl / dl_eval.MARKER_NAME).read_text(encoding="utf-8"))
    return loaded, results["meta"]["freeze_sha256"], marker["freeze_sha256"], _sha(freeze_path)


def test_report_and_marker_carry_the_loaded_freeze_sha(tmp_path, monkeypatch):
    loaded, in_report, in_marker, on_disk = _run(tmp_path, monkeypatch)
    assert in_report == loaded == in_marker == on_disk


def test_sha_survives_freeze_file_change_after_load(tmp_path, monkeypatch):
    def tamper(p: Path) -> None:
        p.write_bytes(p.read_bytes() + b" ")

    loaded, in_report, in_marker, on_disk = _run(tmp_path, monkeypatch, on_compare=tamper)
    assert on_disk != loaded  # the tamper really happened after the load
    assert in_report == loaded and in_marker == loaded


def test_sha_is_of_the_single_read_not_a_later_one(tmp_path, monkeypatch):
    done = []

    def tamper_once(p: Path) -> None:
        if not done:  # only after the first read of the freeze
            done.append(1)
            p.write_text(p.read_text(encoding="utf-8") + " ", encoding="utf-8", newline="")

    loaded, in_report, in_marker, on_disk = _run(tmp_path, monkeypatch, on_read=tamper_once)
    assert on_disk != loaded
    assert in_report == loaded and in_marker == loaded
