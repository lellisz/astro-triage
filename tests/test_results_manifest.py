"""scripts/make_results_manifest.py: sha256, time source, read-only test_opened, no per-object data, refusals."""

from __future__ import annotations

import hashlib
import json
import os
import re
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import make_results_manifest as M

OPENED_AT = "2026-11-10T12:00:00Z"
OBTAINED = "2026-11-09T08:30:00Z"
MTIME = 1_700_000_000  # 2023-11-14T22:13:20Z
GIT_SHA = "a1b2c3d4e5" * 4
FREEZE_SHA = "f" * 64


@pytest.fixture
def env(tmp_path, monkeypatch):
    monkeypatch.setattr(
        M, "REPO_ROOT", tmp_path
    )  # the guard reads the module constant at call time
    out = tmp_path / "data" / "interim" / "full_v0"
    for rel in M.FIXED:
        p = out / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(f"content of {rel}".encode())
    (out / "dl/eval/results_dl.json").write_text(
        json.dumps({"meta": {"git_sha": GIT_SHA, "freeze_sha256": FREEZE_SHA}})
    )
    (out / "dl/test_opened.json").write_text(json.dumps({"opened_at": OPENED_AT}))
    (out / "dl/predictions_s0_v0.parquet").write_bytes(b"pred-with-sidecar")
    (out / "dl/predictions_s0_v0.manifest.json").write_text(
        json.dumps({"obtained_utc": OBTAINED})
    )
    (out / "lgbm_seeds/predictions_s1_v0.parquet").write_bytes(b"pred-no-sidecar")
    for p in out.rglob("*"):
        if p.is_file() and p.name != "test_opened.json":
            os.utime(p, (MTIME, MTIME))
    (tmp_path / "docs").mkdir()
    return out


def run(capsys, out: Path, target: Path) -> tuple[int, str, str]:
    code = M.main(["--outdir", str(out), "--out", str(target)])
    cap = capsys.readouterr()
    return code, cap.out, cap.err


def rows(target: Path) -> dict[str, list[str]]:
    found = {}
    for line in target.read_text(encoding="utf-8").splitlines():
        m = re.match(r"\| `([^`]+)` \|(.*)\|$", line)
        if m:
            found[m.group(1)] = [c.strip() for c in m.group(2).split("|")]
    return found


def test_sha256_and_bytes_match_hashlib(env, tmp_path, capsys):
    target = tmp_path / "docs" / "MANIFEST.md"
    assert run(capsys, env, target)[0] == 0
    table = rows(target)
    assert len(table) == len(M.FIXED) + 2
    for rel, (size, sha, _when, _src) in table.items():
        data = (env / rel).read_bytes()
        assert sha == f"`{hashlib.sha256(data).hexdigest()}`", rel
        assert size == str(len(data)), rel
    header = target.read_text(encoding="utf-8")
    assert (
        f"git_sha: `{GIT_SHA}`" in header
        and FREEZE_SHA in header
        and "dataset_100k_v0" in header
        and "docs/specs/dl-implementation.md" in header
        and "ADR 0006 §17" in header
        and "backlog" not in header
    )


def test_time_source_by_file_type(env, tmp_path, capsys):
    target = tmp_path / "docs" / "MANIFEST.md"
    run(capsys, env, target)
    table = rows(target)
    assert table["dl/test_opened.json"][2:] == [OPENED_AT, "opened_at"]
    assert table["dl/predictions_s0_v0.parquet"][2:] == [OBTAINED, "obtained_utc"]
    assert table["lgbm_seeds/predictions_s1_v0.parquet"][2:] == [
        "2023-11-14T22:13:20Z",
        "mtime",
    ]
    assert table["dl/environment.json"][2:] == ["2023-11-14T22:13:20Z", "mtime"]
    assert all(r[3] for r in table.values())


def test_test_opened_untouched(env, tmp_path, capsys):
    p = env / "dl/test_opened.json"
    os.utime(p, (MTIME + 5, MTIME + 5))
    before = (p.read_bytes(), p.stat().st_mtime_ns)
    run(capsys, env, tmp_path / "docs" / "MANIFEST.md")
    assert (p.read_bytes(), p.stat().st_mtime_ns) == before


def test_no_per_object_data(env, tmp_path, capsys):
    target = tmp_path / "docs" / "MANIFEST.md"
    run(capsys, env, target)
    text = target.read_text(encoding="utf-8")
    assert not re.search(
        r"source_id|pixel|\bra\b|\bdec\b|\b\d{17,19}\b", text, re.IGNORECASE
    )
    assert all(len(r) == 4 for r in rows(target).values())


def test_existing_out_refused_not_overwritten(env, tmp_path, capsys):
    target = tmp_path / "docs" / "MANIFEST.md"
    target.write_text("KEEP", encoding="utf-8")
    code, _, err = run(capsys, env, target)
    assert code == 1 and "already exists" in err
    assert target.read_text(encoding="utf-8") == "KEEP"


def test_missing_file_is_ausente_exit_2_table_written(env, tmp_path, capsys):
    (env / "dl/environment.json").unlink()
    target = tmp_path / "docs" / "MANIFEST.md"
    code, _, err = run(capsys, env, target)
    assert code == 2 and "1 file(s) marked AUSENTE" in err
    assert rows(target)["dl/environment.json"] == ["AUSENTE"] * 4
    assert len(rows(target)) == len(M.FIXED) + 2


def test_out_inside_data_refused(env, tmp_path, capsys):
    target = env / "MANIFEST.md"
    code, _, err = run(capsys, env, target)
    assert code == 1 and "outside the repo's data/" in err
    assert not target.exists()


def _set(env: Path, rel: str, value, *keys: str) -> None:
    p = env / rel
    doc = json.loads(p.read_text(encoding="utf-8"))
    node = doc
    for k in keys[:-1]:
        node = node[k]
    node[keys[-1]] = value
    p.write_text(json.dumps(doc), encoding="utf-8")


HOSTILE = ["x | y", "a\nb", None, {"a": 1}, ["a"], "`x`", "not-valid"]
FIELDS = [
    ("dl/eval/results_dl.json", ("meta", "git_sha")),
    ("dl/eval/results_dl.json", ("meta", "freeze_sha256")),
    ("dl/test_opened.json", ("opened_at",)),
    ("dl/predictions_s0_v0.manifest.json", ("obtained_utc",)),
]


@pytest.mark.parametrize("value", HOSTILE, ids=repr)
@pytest.mark.parametrize("rel,keys", FIELDS, ids=[k[-1] for _, k in FIELDS])
def test_hostile_json_field_refused_no_out(env, tmp_path, capsys, rel, keys, value):
    _set(env, rel, value, *keys)
    target = tmp_path / "docs" / "MANIFEST.md"
    code, _, err = run(capsys, env, target)
    assert code == 1 and M.PREFIX in err
    assert not target.exists()


@pytest.mark.parametrize(
    "rel,keys,value",
    [
        ("dl/eval/results_dl.json", ("meta", "git_sha"), "A" * 40),
        ("dl/eval/results_dl.json", ("meta", "git_sha"), "a" * 39),
        ("dl/eval/results_dl.json", ("meta", "freeze_sha256"), "a" * 40),
        ("dl/eval/results_dl.json", ("meta", "freeze_sha256"), "F" * 64),
        ("dl/eval/results_dl.json", ("meta", "git_sha"), "a" * 40 + "\n"),
    ],
)
def test_hex_fields_are_strict(env, tmp_path, capsys, rel, keys, value):
    _set(env, rel, value, *keys)
    target = tmp_path / "docs" / "MANIFEST.md"
    assert run(capsys, env, target)[0] == 1
    assert not target.exists()


@pytest.mark.parametrize("value", ["2026-11-10T12:00:00-03:00", "2026-11-10T12:00:00"])
def test_timestamp_without_utc_offset_refused(env, tmp_path, capsys, value):
    _set(env, "dl/test_opened.json", value, "opened_at")
    target = tmp_path / "docs" / "MANIFEST.md"
    code, _, err = run(capsys, env, target)
    assert code == 1 and "UTC offset" in err
    assert not target.exists()


def test_utc_offset_normalized_to_z(env, tmp_path, capsys):
    _set(env, "dl/test_opened.json", "2026-11-10T12:00:00+00:00", "opened_at")
    _set(
        env,
        "dl/predictions_s0_v0.manifest.json",
        "2026-11-09T08:30:00.123+00:00",
        "obtained_utc",
    )
    target = tmp_path / "docs" / "MANIFEST.md"
    assert run(capsys, env, target)[0] == 0
    table = rows(target)
    assert table["dl/test_opened.json"][2] == "2026-11-10T12:00:00Z"
    assert table["dl/predictions_s0_v0.parquet"][2] == "2026-11-09T08:30:00Z"
    assert "+00:00" not in target.read_text(encoding="utf-8")
    assert all(r[2].endswith("Z") for r in table.values())


def test_outdir_outside_data_refused(env, tmp_path, capsys):
    other = tmp_path / "elsewhere"
    other.mkdir()
    code, _, err = run(capsys, other, tmp_path / "docs" / "MANIFEST.md")
    assert code == 1 and "inside the repo's data/" in err
