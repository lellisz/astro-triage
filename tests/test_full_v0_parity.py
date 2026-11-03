"""full_v0_parity.py (exact column sets); synthetic fixtures only.
Swap the module for mutation checks via FULL_V0_PARITY_UNDER_TEST."""

import importlib.util
import json
import os
import re
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

ROOT = Path(__file__).resolve().parents[1]


def load(var, default):
    path = Path(os.environ.get(var, ROOT / default))
    spec = importlib.util.spec_from_file_location(path.stem + "_ut", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


par = load("FULL_V0_PARITY_UNDER_TEST", "scripts/full_v0_parity.py")
IDS = [1000000011, 1000000022, 1000000033, 1000000044]


def labels(**over):
    cols = {
        "source_id": IDS, "sdss_class": ["STAR", "QSO", "GALAXY", "STAR"],
        "fold": ["train", "train", "val", "test"], "sdss_z": [0.1, None, 0.3, 0.4],
    }  # fmt: skip
    cols.update(over)
    return pa.table(cols)


def features(**over):
    cols = {"source_id": IDS, "g": [1.5, float("nan"), 3.5, 4.5], "n": [1, 2, 3, 4]}
    cols.update(over)
    return pa.table(cols)


def release(path, lab=None, feat=None, **kw):
    path.mkdir(parents=True, exist_ok=True)
    pq.write_table(lab or labels(), path / "labels.parquet", **kw)
    pq.write_table(feat or features(), path / "features.parquet", **kw)
    return path


# fmt: off
@pytest.fixture
def ref(tmp_path):
    out = tmp_path / "ref.json"
    argv = ["reference", "--outdir", str(release(tmp_path / "a")), "--out", str(out)]
    assert par.main(argv) == 0
    return out


def check(ref, outdir, report):
    argv = ["check", "--reference", str(ref), "--outdir", str(outdir), "--report", str(report)]
    return par.main(argv)
# fmt: on


def set_at(table, name, index, value):
    values = table[name].to_pylist()
    values[index] = value
    new = pa.array(values, table[name].type)
    return table.set_column(table.schema.get_field_index(name), name, new)


def test_match_on_copy_and_on_rewritten_file_bytes(tmp_path, ref, capsys):
    rep = tmp_path / "rep.json"
    assert check(ref, tmp_path / "a", rep) == 0
    other = release(tmp_path / "b", compression="gzip", row_group_size=1)
    assert check(ref, other, rep) == 0
    out = json.loads(rep.read_text())
    assert out["match"] is True and out["file_sha256_equal"] is False
    assert capsys.readouterr().out == ""


def cols(side, *names):
    return {f"{side}.column.{n}" for n in names}


LAB = ("source_id", "sdss_class", "fold", "sdss_z")
MUTATIONS = {
    "rows": ("lab", lambda: labels().slice(0, 3), {"rows", "by_class", "by_fold", "source_id_sha256"} | cols("labels", *LAB)),
    "by_class": ("lab", lambda: set_at(labels(), "sdss_class", 0, "QSO"), {"by_class"} | cols("labels", "sdss_class")),
    "by_fold": ("lab", lambda: set_at(labels(), "fold", 0, "val"), {"by_fold"} | cols("labels", "fold")),
    "schema": ("lab", lambda: labels().append_column("x", pa.array([1, 2, 3, 4])), {"labels_schema"} | cols("labels", "x")),
    "ids": ("lab", lambda: set_at(labels(), "source_id", 0, 1000000015), {"source_id_sha256"} | cols("labels", "source_id")),
    "ids_reorder": ("lab", lambda: set_at(labels(), "source_id", 0, 1000000055), {"source_id_sha256"} | cols("labels", *LAB)),
    "dup_id": ("lab", lambda: set_at(labels(), "source_id", 3, IDS[0]), {"source_id_sha256", "source_id_unique"} | cols("labels", *LAB)),
    "labels": ("lab", lambda: set_at(labels(), "sdss_z", 0, 0.11), cols("labels", "sdss_z")),
    "feat_int": ("feat", lambda: set_at(features(), "n", 1, 99), cols("features", "n")),
    "feat_float": ("feat", lambda: set_at(features(), "g", 0, 1.6), cols("features", "g")),
    "feat_name": ("feat", lambda: features().rename_columns(["source_id", "g", "m"]), {"features_schema"} | cols("features", "n", "m")),
}  # fmt: skip


@pytest.mark.parametrize("kind", MUTATIONS)
def test_mismatch_exit3_names_exact_fields(tmp_path, ref, capsys, kind):
    side, make, fields = MUTATIONS[kind]
    rep = tmp_path / "rep.json"
    assert check(ref, release(tmp_path / "b", **{side: make()}), rep) == 3
    lines = capsys.readouterr().out.splitlines()
    assert lines and all(re.fullmatch(r"MISMATCH: [a-z_0-9.]+", x) for x in lines)
    assert {x.split(": ")[1] for x in lines} == fields
    assert json.loads(rep.read_text())["match"] is False
    assert not any(str(i) in rep.read_text() + "".join(lines) for i in IDS)


def test_errors_exit1_and_leave_existing_report_alone(tmp_path, ref):
    rep = tmp_path / "rep.json"
    rep.write_text("OLD")
    missing = tmp_path / "m"
    missing.mkdir()
    corrupt = release(tmp_path / "c")
    (corrupt / "features.parquet").write_bytes(b"not parquet")
    bad_ref = tmp_path / "bad.json"
    bad_ref.write_text("{not json")
    a, none = tmp_path / "a", tmp_path / "none.json"
    for reference, outdir in [(ref, missing), (ref, corrupt), (bad_ref, a), (none, a)]:
        assert check(reference, outdir, rep) == 1
    assert rep.read_text() == "OLD"


def test_report_write_is_atomic(tmp_path, ref, monkeypatch):
    rep = tmp_path / "rep.json"
    rep.write_text("OLD")
    monkeypatch.setattr(par.os, "replace", lambda *a: 1 / 0)
    with pytest.raises(ZeroDivisionError):
        check(ref, tmp_path / "a", rep)
    assert rep.read_text() == "OLD"
    assert not list(tmp_path.glob(".parity.*"))
