"""l/b aggregate-statistics rules of full_v0_parity.py (K11-1: l and b are not hashed).

Synthetic fixtures only. Swap the module for mutation checks via FULL_V0_PARITY_UNDER_TEST.
"""

import importlib.util
import json
import os
import re
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

ROOT = Path(__file__).resolve().parents[1]
_path = Path(
    os.environ.get("FULL_V0_PARITY_UNDER_TEST", ROOT / "scripts/full_v0_parity.py")
)
_spec = importlib.util.spec_from_file_location("full_v0_parity_stats_ut", _path)
par = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(par)

IDS = [1000000011, 1000000022, 1000000033, 1000000044]
BASE = {
    "l": [12.3456789004, 100.0000000003, 0.5, 300.25],
    "b": [-45.1234567891, 2.0, 0.25, 80.75],
    "ra": [10.5, 20.5, 30.5, 40.5],
    "dec": [-5.5, 6.5, 7.5, -8.5],
}
SIDES = ["labels", "features"]
COLS = ["l", "b"]


def table(**over):
    cols = {"source_id": IDS, **BASE, **over}
    types = {"source_id": pa.int64()}
    return pa.table(
        {k: pa.array(v, types.get(k, pa.float64())) for k, v in cols.items()}
    )


def release(path, feat, lab):
    path.mkdir(parents=True, exist_ok=True)
    ids = lab["source_id"].to_pylist()
    meta = pa.table({"sdss_class": ["STAR"] * len(ids), "fold": ["train"] * len(ids)})
    for name in meta.column_names:
        lab = lab.append_column(name, meta[name])
    pq.write_table(lab, path / "labels.parquet")
    pq.write_table(feat, path / "features.parquet")
    return path


def check(tmp_path, capsys, feat=None, lab=None, edit=None, drop_ref=(), ref_tab=None):
    """Reference from a release (pristine by default), then check a variant."""
    ref = tmp_path / "ref.json"
    base = ref_tab or table()
    argv = ["reference", "--outdir", str(release(tmp_path / "a", base, base))]
    assert par.main([*argv, "--out", str(ref)]) == 0
    doc = json.loads(ref.read_text())
    if edit:
        edit(doc)
    for key in drop_ref:
        del doc[key]
    ref.write_text(json.dumps(doc))
    other = release(tmp_path / "b", feat or table(), lab or table())
    argv = ["check", "--reference", str(ref), "--outdir", str(other)]
    code = par.main([*argv, "--report", str(tmp_path / "rep.json")])
    out = capsys.readouterr().out
    return code, {x.split(": ")[1] for x in out.splitlines()}, out


def swap(side, name, values):
    """(feat, lab) where `name` holds `values` on one side only."""
    mod = table(**{name: values})
    return (mod, table()) if side == "features" else (table(), mod)


def shifted(side, name, delta, row=None):
    values = [v + delta if row in (None, i) else v for i, v in enumerate(BASE[name])]
    return swap(side, name, values)


def run(tmp_path, capsys, pair, **kw):
    code, names, _ = check(tmp_path, capsys, *pair, **kw)
    return code, names


@pytest.mark.parametrize("side", SIDES)
@pytest.mark.parametrize("name", COLS)
def test_noise_of_1e12_on_every_row_matches(tmp_path, capsys, side, name):
    pair = shifted(side, name, 1e-12)
    assert run(tmp_path, capsys, pair) == (0, set())


@pytest.mark.parametrize("side", SIDES)
@pytest.mark.parametrize("name", COLS)
@pytest.mark.parametrize("row", [0, 3])
def test_single_row_shift_of_1e6_names_only_that_column(
    tmp_path, capsys, side, name, row
):
    pair = shifted(side, name, 1e-6, row)
    assert run(tmp_path, capsys, pair) == (3, {f"{side}.column.{name}"})


@pytest.mark.parametrize("name", COLS)
def test_value_straddling_a_1e9_rounding_boundary_matches(tmp_path, capsys, name):
    low, high = 12.3456789005 - 1e-13, 12.3456789005 + 1e-13
    assert np.round(low, 9) != np.round(high, 9)  # the old rounded digest would flip
    rest = BASE[name][1:]
    ref, got = table(**{name: [low, *rest]}), table(**{name: [high, *rest]})
    assert check(tmp_path, capsys, got, got, ref_tab=ref)[:2] == (0, set())


@pytest.mark.parametrize("side", SIDES)
@pytest.mark.parametrize("field", ["count", "null_count", "nan_count"])
def test_exact_stat_fields_are_compared_exactly(tmp_path, capsys, side, field):
    def edit(doc):
        doc[f"{side}_column_stats"]["l"][field] += 1

    assert run(tmp_path, capsys, (), edit=edit) == (3, {f"{side}.column.l"})


@pytest.mark.parametrize("field", ["min", "max", "sum", "sumsq"])
def test_approx_stat_fields_have_a_tolerance_not_exactness(tmp_path, capsys, field):
    def small(doc):
        doc["features_column_stats"]["b"][field] += 1e-12

    def big(doc):
        doc["features_column_stats"]["b"][field] += 1e-3

    assert run(tmp_path, capsys, (), edit=small) == (0, set())
    assert run(tmp_path / "x", capsys, (), edit=big) == (3, {"features.column.b"})


def test_dropped_row_nulled_value_and_nan_value_are_all_caught(tmp_path, capsys):
    kept = table().take([0, 1, 2])
    code, got = run(tmp_path / "d", capsys, (kept, kept))
    assert code == 3 and {"features.column.l", "labels.column.l"} <= got
    l_rest, b_rest = BASE["l"][1:], BASE["b"]
    cases = [
        (swap("features", "l", [None, *l_rest]), {"features.column.l"}),
        (swap("features", "l", [float("nan"), *l_rest]), {"features.column.l"}),
        (swap("labels", "b", [b_rest[0], None, *b_rest[2:]]), {"labels.column.b"}),
    ]
    for i, (pair, names) in enumerate(cases):
        assert run(tmp_path / str(i), capsys, pair) == (3, names)


@pytest.mark.parametrize("name", ["ra", "dec"])
def test_ra_dec_ulp_is_caught_even_when_l_and_b_are_within_tolerance(
    tmp_path, capsys, name
):
    bumped = [float(np.nextafter(BASE[name][0], np.inf)), *BASE[name][1:]]
    over = {c: [v + 1e-12 for v in BASE[c]] for c in COLS} | {name: bumped}
    assert run(tmp_path, capsys, (table(**over),)) == (3, {f"features.column.{name}"})


def test_nan_in_l_at_the_same_rows_on_both_sides_matches(tmp_path, capsys):
    nan_l = [BASE["l"][0], float("nan"), BASE["l"][2], float("nan")]
    tab = table(l=nan_l)
    assert check(tmp_path, capsys, tab, tab, ref_tab=tab)[:2] == (0, set())
    got = par.fingerprint(tmp_path / "b")["features_column_stats"]["l"]
    assert (got["count"], got["nan_count"], got["null_count"]) == (4, 2, 0)
    assert got["min"] == nan_l[2] and got["max"] == nan_l[0]


def test_l_and_b_are_stats_not_digests_and_the_stats_are_correct(tmp_path):
    fp = par.fingerprint(release(tmp_path, table(), table()))
    for side in SIDES:
        assert not {"l", "b"} & set(fp[f"{side}_column_sha256"])
        for name in COLS:
            v = np.array(BASE[name])
            got = fp[f"{side}_column_stats"][name]
            assert (got["count"], got["null_count"], got["nan_count"]) == (4, 0, 0)
            assert (got["min"], got["max"]) == (v.min(), v.max())
            assert got["sum"] == pytest.approx(v.sum(), rel=1e-12)
            assert got["sumsq"] == pytest.approx((v * v).sum(), rel=1e-12)


@pytest.mark.parametrize(
    "drop", [["labels_column_stats"], ["features_column_stats"], list(par.STATS_FIELDS)]
)
def test_old_format_reference_is_rejected_with_exit_1_and_no_output(
    tmp_path, capsys, drop
):
    code, _, out = check(tmp_path, capsys, drop_ref=drop)
    assert code == 1 and out == ""
    assert not (tmp_path / "rep.json").exists()
    assert not list(tmp_path.glob(".parity.*"))


def test_mismatch_stdout_is_field_names_only_never_row_values(tmp_path, capsys):
    code, _, out = check(tmp_path, capsys, *shifted("features", "l", 1e-6, 0))
    assert code == 3 and out.strip()
    assert all(
        re.fullmatch(r"MISMATCH: [a-z_]+(\.[a-z_]+)*", x) for x in out.splitlines()
    )
    assert "12.3456" not in out and "100.0000" not in out
