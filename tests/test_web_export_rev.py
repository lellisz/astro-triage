"""Regressions for the integration review of the site figures in the web export.

REV-01: graficos.cmd null -> the full export (main) exits 0 and figures.json has no cmd-taxa.
REV-03: ``_remove_export`` retries only on PermissionError (a Windows scanner may hold a file).
REV-04: an OSError from ``write_figures`` becomes a one-line ExportError; main returns 1.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from test_web_export_invariants import RELEASE, synth_root  # noqa: F401  (module fixture)

import astro_triage.web_export.__main__ as M
from astro_triage.web_export import graficos


@pytest.fixture
def fast_retry(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(M, "RENAME_WAIT_S", 0)


def _export_folder(tmp_path: Path) -> Path:
    folder = tmp_path / ".rel.tmp-x.old"
    (folder / "figures").mkdir(parents=True)
    for n in ("meta.json", "overview.json"):
        (folder / n).write_text("{}", encoding="utf-8")
    for n in ("a.svg", "figures.json"):
        (folder / "figures" / n).write_text("x", encoding="utf-8")
    return folder


def _flaky(monkeypatch: pytest.MonkeyPatch, name: str, exc: type[BaseException], fails: int):
    """Make Path.<name> raise ``exc`` on its first ``fails`` calls; count all calls."""
    real = getattr(Path, name)
    calls = {"n": 0}

    def op(self, *a, **k):
        calls["n"] += 1
        if calls["n"] <= fails:
            raise exc(13, "held by another process", str(self))
        return real(self, *a, **k)

    monkeypatch.setattr(Path, name, op)
    return calls


# ------------------------------------------------------------------ REV-03 failure first
def test_rev03_persistent_permission_error_propagates_after_all_attempts(
    tmp_path, monkeypatch, fast_retry
):
    folder = _export_folder(tmp_path)
    calls = _flaky(monkeypatch, "unlink", PermissionError, fails=10_000)
    with pytest.raises(PermissionError):
        M._remove_export(folder)
    assert calls["n"] == M.RENAME_ATTEMPTS  # retried, then gave up on the first file
    monkeypatch.undo()
    assert folder.is_dir()  # nothing silently swallowed


def test_rev03_other_oserror_is_not_retried(tmp_path, monkeypatch, fast_retry):
    folder = _export_folder(tmp_path)
    calls = _flaky(monkeypatch, "unlink", FileNotFoundError, fails=10_000)
    with pytest.raises(FileNotFoundError):
        M._remove_export(folder)
    assert calls["n"] == 1


@pytest.mark.parametrize("name", ["unlink", "rmdir"])
def test_rev03_transient_permission_error_is_retried_and_removal_finishes(
    tmp_path, monkeypatch, fast_retry, name
):
    folder = _export_folder(tmp_path)
    calls = _flaky(monkeypatch, name, PermissionError, fails=1)
    M._remove_export(folder)  # before the fix: PermissionError on the first call
    assert not folder.exists()
    assert calls["n"] >= 2


def test_rev03_retry_waits_rename_wait_s_between_attempts(tmp_path, monkeypatch):
    folder = _export_folder(tmp_path)
    _flaky(monkeypatch, "unlink", PermissionError, fails=2)
    waits: list[float] = []
    monkeypatch.setattr(M, "RENAME_WAIT_S", 0.125)
    monkeypatch.setattr(M.time, "sleep", waits.append)
    M._remove_export(folder)
    assert waits == [0.125, 0.125]


# ------------------------------------------------------------------ REV-04
def _boom(*_a, **_k):
    raise OSError(28, "No space left on device")


def test_rev04_figures_oserror_is_one_line_export_error_in_run(
    synth_root, tmp_path, monkeypatch  # noqa: F811
):
    monkeypatch.setenv("ASTRO_DATA_ROOT", str(synth_root))
    monkeypatch.setattr(M, "write_figures", _boom)
    out = tmp_path / "out"
    with pytest.raises(M.ExportError) as info:
        M.run(RELEASE, out)
    msg = str(info.value)
    assert msg.startswith("figuras:") and "\n" not in msg
    assert info.value.__suppress_context__  # raised "from None"
    assert not (out / RELEASE).exists()
    assert [p.name for p in out.iterdir()] == []  # no half export, no temp left


def test_rev04_figures_oserror_main_returns_1_without_traceback(
    synth_root, tmp_path, monkeypatch, capsys  # noqa: F811
):
    monkeypatch.setenv("ASTRO_DATA_ROOT", str(synth_root))
    monkeypatch.setattr(M, "write_figures", _boom)
    out = tmp_path / "out"
    assert M.main(["--release", RELEASE, "--out", str(out)]) == 1
    cap = capsys.readouterr()
    lines = cap.err.strip().splitlines()
    assert len(lines) == 1, cap.err
    assert lines[0].startswith("erro: figuras:"), lines[0]
    assert "Traceback" not in cap.err + cap.out
    assert not (out / RELEASE).exists()


# ------------------------------------------------------------------ REV-01 full export
def test_rev01_main_exports_without_cmd_taxa_when_cmd_is_null(
    synth_root, tmp_path, monkeypatch, capsys  # noqa: F811
):
    real = graficos.graficos

    def no_cmd(rel):
        # same state the export produces for a release without BP-RP
        return {**real(rel), "cmd": None}

    monkeypatch.setattr(graficos, "graficos", no_cmd)
    monkeypatch.setenv("ASTRO_DATA_ROOT", str(synth_root))
    out = tmp_path / "out"
    assert M.main(["--release", RELEASE, "--out", str(out)]) == 0, capsys.readouterr().err
    doc = json.loads((out / RELEASE / "graficos.json").read_text(encoding="utf-8"))
    assert doc["cmd"] is None
    figs = out / RELEASE / "figures"
    manifest = json.loads((figs / "figures.json").read_text(encoding="utf-8"))
    assert "cmd-taxa" not in manifest["figures"]
    assert not (figs / "cmd-taxa.svg").exists()
    assert {"calibracao", "confusao", "quasar-pureza"} <= set(manifest["figures"])
