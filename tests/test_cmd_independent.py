"""Independent proof of the colour-magnitude map (G7) of ``graficos.json``.

ADR 0008 Amendment 1 (A1.2 item 7, A1.8) and ``docs/specs/web-graficos.md`` (G7). Expected
values come from the spec and from hand arithmetic: every synthetic object is placed by hand in
a known cell, the Wilson numbers are checked against hand values, the labels against literals.
The real-release test recomputes the cell counts from the parquet files with plain numpy and
only ever reads the data root.
"""

from __future__ import annotations

import builtins
import json
import math
import os
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import pytest

from astro_triage.dashboard import data as D
from astro_triage.dashboard import fmt, paths
from astro_triage.dashboard.html_galaxies import SMALL_RATE
from astro_triage.web_export import graficos as G
from astro_triage.web_export.types import clean

MODEL = "lgbm_full"
CLASSES = ("STAR", "GALAXY", "QSO")
PER_OBJECT_KEYS = {"source_id", "ra", "dec", "l", "b", "healpix_l5", "hp"}
MAX_ARRAY = 252
MAX_BYTES = 150 * 1024
FLOOR = 7
FILLER = 60  # objects at G = 11 (off the grid) so that the other charts have enough rows
Z95 = 1.959963984540054
N_G, N_C = 18, 14


# ------------------------------------------------------------------ synthetic release
def _release(
    g: list[float],
    c: list[float],
    err: list[bool],
    floor: int = FLOOR,
    with_c: bool = True,
    extra: dict[str, object] | None = None,
) -> D.Release:
    """Test fold from hand-placed (G, BP-RP, error) triples plus ``FILLER`` off-grid objects."""
    g_all = np.array([*g, *([11.0] * FILLER)], dtype=float)
    c_all = np.array([*c, *([1.0] * FILLER)], dtype=float)
    e_all = np.array([*err, *([False] * FILLER)], dtype=bool)
    n = len(g_all)
    idx = np.arange(n)
    true = np.array(CLASSES)[idx % 3]
    pred_idx = (idx + e_all.astype(int)) % 3
    rng = np.random.default_rng(3)
    p = 0.1 + rng.uniform(0.0, 0.01, size=(n, 3))
    p[idx, pred_idx] += 0.8
    p /= p.sum(axis=1, keepdims=True)
    test = pd.DataFrame(
        {
            "true": true,
            "healpix_l5": idx % 5,
            "phot_g_mean_mag": g_all,
            "sdss_z": np.where(true == "STAR", 0.0, 0.5 + (idx % 7) * 0.3),
        }
    )
    if with_c:
        test["bp_rp"] = c_all
    for j, cls in enumerate(CLASSES):
        test[f"p_{MODEL}_{cls}"] = p[:, j]
        test[f"praw_{MODEL}_{cls}"] = p[:, j]
    test[f"pred_{MODEL}"] = np.array(CLASSES)[pred_idx]
    test[f"conf_{MODEL}"] = p.max(axis=1)
    test[f"err_{MODEL}"] = test["true"] != test[f"pred_{MODEL}"]
    for name, value in (extra or {}).items():
        test[name] = value
    results = {"meta": {"floor": {"min_objects": floor}}}
    return D.Release("synth", D.REPO_ROOT, results, {}, pd.DataFrame(), test, 0)


def _centre(gi: int, ci: int) -> tuple[float, float]:
    """Middle of cell (gi, ci), by arithmetic on the spec grid (12 + 0.5 gi, -0.5 + 0.25 ci)."""
    return 12.0 + 0.5 * gi + 0.25, -0.5 + 0.25 * ci + 0.125


def _from_spec(spec: dict[tuple[int, int], tuple[int, int]], floor: int = FLOOR) -> D.Release:
    """``spec[(gi, ci)] = (n, k)``: n objects in the cell, the first k of them wrong."""
    g: list[float] = []
    c: list[float] = []
    err: list[bool] = []
    for (gi, ci), (n, k) in spec.items():
        gc, cc = _centre(gi, ci)
        g += [gc] * n
        c += [cc] * n
        err += [i < k for i in range(n)]
    return _release(g, c, err, floor)


def _cmd(rel: D.Release) -> dict:
    doc = G.graficos(rel)
    assert doc["cmd"] is not None
    return doc["cmd"]


def _cell(cmd: dict, gi: int, ci: int) -> dict:
    cell = cmd["cells"][gi * N_C + ci]
    assert (cell["gi"], cell["ci"]) == (gi, ci)
    return cell


def _wilson(k: int, n: int, z: float = Z95) -> tuple[float, float]:
    p = k / n
    d = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / d
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0.0, centre - half), min(1.0, centre + half)


def _check_no_object_data(doc: object, path: str = "$") -> None:
    if isinstance(doc, dict):
        for k, v in doc.items():
            assert k not in PER_OBJECT_KEYS, f"{k} at {path}"
            _check_no_object_data(v, f"{path}.{k}")
    elif isinstance(doc, list):
        assert len(doc) <= MAX_ARRAY, f"{path}: {len(doc)} items"
        for i, v in enumerate(doc):
            _check_no_object_data(v, f"{path}[{i}]")


# ------------------------------------------------- 1. grid shape, order, fixed edges
def test_grid_has_252_cells_18_by_14_g_major_with_spec_edges() -> None:
    cmd = _cmd(_release([15.2], [0.3], [False]))
    assert len(cmd["cells"]) == 252 == N_G * N_C
    assert cmd["g_edges"] == [12.0 + 0.5 * i for i in range(19)]
    assert cmd["c_edges"] == [-0.5 + 0.25 * i for i in range(15)]
    assert (len(cmd["g_labels"]), len(cmd["c_labels"])) == (18, 14)
    assert [(c["gi"], c["ci"]) for c in cmd["cells"]] == [
        (k // N_C, k % N_C) for k in range(252)
    ]


def test_edges_and_labels_do_not_depend_on_the_data() -> None:
    base = _cmd(_release([15.2, 17.7], [0.3, 1.1], [False, True]))
    rng = np.random.default_rng(99)
    g = rng.uniform(0.0, 30.0, size=500).tolist()
    c = rng.uniform(-5.0, 9.0, size=500).tolist()
    other = _cmd(_release(g, c, [bool(x) for x in rng.integers(0, 2, size=500)]))
    for key in ("g_edges", "c_edges", "g_labels", "c_labels", "g_ticks", "c_ticks"):
        assert other[key] == base[key], key
    assert [(c["gi"], c["ci"]) for c in other["cells"]] == [
        (c["gi"], c["ci"]) for c in base["cells"]
    ]
    # falsifier: the data did move the counts, so the equality above is not vacuous
    assert other["n_outside"]["value"] != base["n_outside"]["value"]


# ------------------------------------------------- 2. half-open cells, outside, missing
NAN = float("nan")
INF = float("inf")
# (G, BP-RP, expected cell or "out" or "miss"), every cell below computed by hand
POINTS: list[tuple[float, float, object]] = [
    (15.0, 0.25, (6, 3)),  # exactly on both lower borders: the cell above/right
    (12.0, -0.5, (0, 0)),  # first edge belongs to the first cell
    (20.999, 2.999, (17, 13)),  # just under the last edges
    (14.999, 0.249, (5, 2)),  # just under a border: the cell below/left
    (13.0, 0.5, (2, 4)),
    (21.0, 1.0, "out"),  # exactly the top of the last G edge
    (15.0, 3.0, "out"),  # exactly the top of the last colour edge
    (11.999, 1.0, "out"),
    (15.0, -0.501, "out"),
    (INF, 1.0, "out"),
    (15.0, -INF, "out"),
    (NAN, 1.0, "miss"),
    (15.0, NAN, "miss"),
    (NAN, NAN, "miss"),
]


def test_half_open_borders_outside_and_missing() -> None:
    rel = _release(
        [p[0] for p in POINTS],
        [p[1] for p in POINTS],
        [i == 0 for i in range(len(POINTS))],  # only the first object is an error
        floor=1,
    )
    cmd = _cmd(rel)
    expected = {p[2] for p in POINTS if isinstance(p[2], tuple)}
    for gi in range(N_G):
        for ci in range(N_C):
            cell = _cell(cmd, gi, ci)
            if (gi, ci) in expected:
                assert cell["below_floor"] is False and cell["n"]["value"] == 1, (gi, ci)
                assert cell["errors"]["value"] == (1 if (gi, ci) == (6, 3) else 0), (gi, ci)
            else:
                assert cell["below_floor"] is True and cell["n"]["value"] is None, (gi, ci)
    assert cmd["n_missing"]["value"] == sum(p[2] == "miss" for p in POINTS) == 3
    assert cmd["n_outside"]["value"] == sum(p[2] == "out" for p in POINTS) + FILLER


# ------------------------------------------------- 3. floor
def test_floor_boundary_is_exact_and_hides_every_value_below_it() -> None:
    spec = {(2, 2): (FLOOR, 1), (2, 3): (FLOOR - 1, FLOOR - 1), (5, 5): (FLOOR + 3, 0)}
    rel = _from_spec(spec)
    floor = rel.meta["floor"]["min_objects"]
    assert floor == FLOOR
    doc = G.graficos(rel)
    assert doc["floor"] == floor
    cmd = doc["cmd"]
    for gi in range(N_G):
        for ci in range(N_C):
            cell = _cell(cmd, gi, ci)
            n_true = spec.get((gi, ci), (0, 0))[0]
            if n_true < floor:
                assert cell["below_floor"] is True, (gi, ci, n_true)
                assert cell["n"]["value"] is None and cell["errors"]["value"] is None
                assert cell["rate"]["point"] is None
                assert cell["rate"]["lo"] is None and cell["rate"]["hi"] is None
            else:
                assert cell["below_floor"] is False, (gi, ci, n_true)
                assert cell["n"]["value"] == n_true
    at_floor = _cell(cmd, 2, 2)
    assert at_floor["n"]["value"] == floor and at_floor["rate"]["point"] is not None
    assert _cell(cmd, 2, 3)["below_floor"] is True  # floor - 1
    shown = [c for c in cmd["cells"] if not c["below_floor"]]
    assert all(c["n"]["value"] >= floor for c in shown) and len(shown) == 2
    # the hidden cell's 100 % error rate must not leak through the cap
    assert cmd["cap"]["rate"] == pytest.approx(1 / floor)


def test_floor_above_everything_nulls_the_whole_map() -> None:
    cmd = _cmd(_from_spec({(1, 1): (30, 3)}, floor=10_000))
    assert all(c["below_floor"] and c["n"]["value"] is None for c in cmd["cells"])
    assert cmd["cap"]["rate"] is None and cmd["cap"]["count"] is None
    assert cmd["cap"]["rate_text"] is None and cmd["cap"]["count_text"] is None


# ------------------------------------------------- 4. rate and Wilson interval by hand
def test_rate_and_wilson_interval_checked_by_hand() -> None:
    cmd = _cmd(_from_spec({(3, 3): (10, 3), (4, 4): (7, 0)}))
    mid = _cell(cmd, 3, 3)
    assert mid["n"]["value"] == 10 and mid["errors"]["value"] == 3
    assert mid["rate"]["point"] == pytest.approx(0.3, abs=1e-12)
    # hand value of the 95 % Wilson interval for 3 of 10
    assert mid["rate"]["lo"] == pytest.approx(0.1078, abs=5e-4)
    assert mid["rate"]["hi"] == pytest.approx(0.6032, abs=5e-4)
    lo, hi = _wilson(3, 10)
    assert mid["rate"]["lo"] == pytest.approx(lo, abs=1e-9)
    assert mid["rate"]["hi"] == pytest.approx(hi, abs=1e-9)
    zero = _cell(cmd, 4, 4)  # zero errors: lower bound 0, upper z^2 / (n + z^2)
    assert zero["rate"]["point"] == 0.0
    assert zero["rate"]["lo"] == pytest.approx(0.0, abs=1e-12)
    assert zero["rate"]["hi"] == pytest.approx(Z95**2 / (7 + Z95**2), abs=1e-9)


# ------------------------------------------------- 5. no per-object data, small file
def test_no_forbidden_keys_no_long_list_and_small_json() -> None:
    doc = G.graficos(_from_spec({(1, 1): (30, 3), (2, 2): (50, 5)}))
    assert doc["cmd"] is not None
    _check_no_object_data(doc)
    size = len(json.dumps(clean(doc), ensure_ascii=False, allow_nan=False).encode("utf-8"))
    assert size < MAX_BYTES, size
    with pytest.raises(AssertionError):  # falsifiers of the checker
        _check_no_object_data({"cmd": [{"healpix_l5": 3}]})
    with pytest.raises(AssertionError):
        _check_no_object_data({"cells": list(range(MAX_ARRAY + 1))})


# ------------------------------------------------- 6. LightGBM only, no dl/ file opened
class _Opened:
    def __init__(self, mp: pytest.MonkeyPatch) -> None:
        self.paths: list[str] = []
        real_open = builtins.open
        real_read_parquet = pd.read_parquet

        def note(p: object) -> None:
            if isinstance(p, str | os.PathLike):
                self.paths.append(os.fspath(p))

        def open_(file, *a, **k):
            note(file)
            return real_open(file, *a, **k)

        def read_parquet(path, *a, **k):
            note(path)
            return real_read_parquet(path, *a, **k)

        mp.setattr(builtins, "open", open_)
        mp.setattr(pd, "read_parquet", read_parquet)
        for fn in ("read_schema", "read_table", "read_metadata"):
            mp.setattr(pq, fn, self._wrap(getattr(pq, fn), note))

    @staticmethod
    def _wrap(orig, note):
        def inner(source, *a, **k):
            note(source)
            return orig(source, *a, **k)

        return inner


def _is_dl(path: str) -> bool:
    p = path.replace("\\", "/")
    return "/dl/" in p or "predictions_mlp" in p or "predictions_ftt" in p or "dl_eval" in p


def test_map_uses_lgbm_full_only_and_ignores_dl_columns(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rel = _from_spec({(1, 1): (30, 3), (2, 2): (50, 5)})
    clean_cmd = _cmd(rel)
    n = len(rel.test)
    poisoned = rel.test.assign(
        err_mlp_full=True, err_ftt_full=True, pred_mlp_full="QSO", pred_ftt_full="QSO"
    )
    rel_dl = D.Release("synth", D.REPO_ROOT, rel.results, {}, pd.DataFrame(), poisoned, 0)
    seen_models: list[str] = []
    real_cells = G.data_charts.cmd_cells

    def spy(r, model, *a, **k):
        seen_models.append(model)
        return real_cells(r, model, *a, **k)

    monkeypatch.setattr(G.data_charts, "cmd_cells", spy)
    opened = _Opened(monkeypatch)
    with_dl_cols = _cmd(rel_dl)
    assert n > 0
    assert seen_models == ["lgbm_full"]
    assert with_dl_cols == clean_cmd  # the poisoned DL columns change nothing
    assert not [p for p in opened.paths if _is_dl(p)], opened.paths
    # falsifier: a real error column of another model would have moved the counts
    assert any(c["errors"]["value"] for c in clean_cmd["cells"] if not c["below_floor"])


# ------------------------------------------------- 7. texts belong to Python
def test_labels_and_ticks_match_pt_formatting_and_literals() -> None:
    cmd = _cmd(_release([15.2], [0.3], [False]))
    g_expected = [
        f"{fmt.pt_num(12.0 + 0.5 * i, 1)}–{fmt.pt_num(12.0 + 0.5 * (i + 1), 1)}"
        for i in range(18)
    ]
    c_expected = [
        f"{fmt.pt_num(-0.5 + 0.25 * i, 2)}–{fmt.pt_num(-0.5 + 0.25 * (i + 1), 2)}"
        for i in range(14)
    ]
    assert cmd["g_labels"] == g_expected and cmd["c_labels"] == c_expected
    assert (cmd["g_labels"][0], cmd["g_labels"][-1]) == ("12,0–12,5", "20,5–21,0")
    assert (cmd["c_labels"][0], cmd["c_labels"][-1]) == ("-0,50–-0,25", "2,75–3,00")
    assert [t["value"] for t in cmd["g_ticks"]] == [float(v) for v in range(12, 22)]
    for ticks, edges in ((cmd["g_ticks"], cmd["g_edges"]), (cmd["c_ticks"], cmd["c_edges"])):
        for t in ticks:
            assert t["value"] in edges
            assert float(t["text"].replace(",", ".")) == pytest.approx(t["value"])
            assert "." not in t["text"]  # decimal comma, Python owns the text


@pytest.mark.parametrize(
    ("spec", "rate", "rate_text", "count", "count_text"),
    [
        # the hidden cell has a 100 % rate and the largest rate: it must not set the cap
        ({(1, 1): (30, 3), (2, 2): (FLOOR - 1, FLOOR - 1)}, 0.1, "10,0%", 30, "30"),
        ({(1, 1): (101, 1)}, 1 / 101, "0,99%", 101, "101"),  # just under SMALL_RATE: 2 digits
        ({(1, 1): (100, 1)}, 0.01, "1,0%", 100, "100"),  # exactly SMALL_RATE: 1 digit
        ({(1, 1): (1234, 0)}, 0.0, "0,00%", 1234, "1.234"),  # maximum exactly 0.0: 2 digits
    ],
)
def test_cap_texts_and_digit_rule(
    spec: dict, rate: float, rate_text: str, count: int, count_text: str
) -> None:
    cap = _cmd(_from_spec(spec))["cap"]
    assert cap["rate"] == pytest.approx(rate, abs=1e-12)
    assert cap["count"] == count
    assert cap["rate_text"] == rate_text
    assert cap["count_text"] == count_text
    assert cap["rate_text"] == fmt.pt_pct(rate, 2 if rate < SMALL_RATE else 1)
    assert cap["count_text"] == fmt.pt_int(count)
    assert cap["rate_zero_text"] == fmt.pt_pct(0, 0) == "0%"
    assert cap["count_zero_text"] == fmt.pt_int(0) == "0"


def test_cap_is_null_when_no_cell_is_reportable() -> None:
    cap = _cmd(_from_spec({(1, 1): (FLOOR - 1, 2)}))["cap"]
    assert cap["rate"] is None and cap["count"] is None
    assert cap["rate_text"] is None and cap["count_text"] is None
    assert cap["rate_zero_text"] == "0%" and cap["count_zero_text"] == "0"


# ------------------------------------------------- 8. no BP-RP: cmd is null
def test_cmd_is_null_without_bp_rp_and_the_rest_still_builds() -> None:
    rel = _release([15.2], [0.3], [False], with_c=False)
    assert "bp_rp" not in rel.test.columns
    doc = G.graficos(rel)
    assert doc["cmd"] is None
    assert doc["confusion"]["rows"] and doc["error_by_g"]["series"]
    assert json.dumps(clean(doc), allow_nan=False)  # serialises without error
    assert G.graficos(_release([15.2], [0.3], [False]))["cmd"] is not None  # falsifier


# ------------------------------------------------- 9. the real release
def _root() -> Path | None:
    raw = os.environ.get("ASTRO_DATA_ROOT", "")
    if not raw:
        return None
    rel = Path(raw) / "interim" / "full_v0"
    return rel if (rel / "eval" / "results.json").is_file() else None


needs_real = pytest.mark.skipif(_root() is None, reason="ASTRO_DATA_ROOT sem full_v0")


@pytest.fixture(scope="module")
def real_release() -> D.Release:
    return D.load_release("full_v0", paths.interim_dir())


@pytest.fixture(scope="module")
def real_doc(real_release: D.Release) -> dict:
    return json.loads(json.dumps(clean(G.graficos(real_release)), allow_nan=False))


@pytest.fixture(scope="module")
def independent() -> pd.DataFrame:
    """Test fold rebuilt from the parquet files with plain pandas (no dashboard loader)."""
    root = _root()
    assert root is not None
    lab = pd.read_parquet(
        root / "labels.parquet",
        columns=["source_id", "sdss_class", "fold"],
        filters=[("fold", "==", "test")],
    )
    cols = ["source_id", "p_cal_star", "p_cal_galaxy", "p_cal_qso"]
    full = pd.read_parquet(root / "predictions_lgbm_full_v0.parquet", columns=cols)
    noind = pd.read_parquet(root / "predictions_lgbm_noind_v0.parquet", columns=cols)
    feat = pd.read_parquet(
        root / "features.parquet", columns=["source_id", "phot_g_mean_mag", "bp_rp"]
    )
    df = lab.merge(full, on="source_id", how="inner")
    df = df.merge(noind[["source_id", *cols[1:]]], on="source_id", how="inner", suffixes=("", "_n"))
    df = df.merge(feat, on="source_id", how="left")
    df = df.dropna(subset=[*cols[1:], *[c + "_n" for c in cols[1:]]]).reset_index(drop=True)
    probs = df[cols[1:]].to_numpy(dtype=float)
    true = df["sdss_class"].astype(str).str.upper()
    df["wrong"] = np.array(CLASSES)[probs.argmax(axis=1)] != true
    return df


@needs_real
def test_real_release_cells_partition_the_test_fold(
    real_doc: dict, independent: pd.DataFrame
) -> None:
    cmd = real_doc["cmd"]
    assert cmd is not None
    assert len(cmd["cells"]) == 252
    _check_no_object_data(real_doc)
    floor = real_doc["floor"]
    g = independent["phot_g_mean_mag"].to_numpy(dtype=float)
    c = independent["bp_rp"].to_numpy(dtype=float)
    miss = np.isnan(g) | np.isnan(c)
    with np.errstate(invalid="ignore"):
        inside = ~miss & (g >= 12.0) & (g < 21.0) & (c >= -0.5) & (c < 3.0)
    gi = np.floor((g[inside] - 12.0) / 0.5).astype(int)
    ci = np.floor((c[inside] + 0.5) / 0.25).astype(int)
    counts = np.zeros((N_G, N_C), dtype=int)
    np.add.at(counts, (gi, ci), 1)
    n_test = len(independent)
    assert cmd["n_missing"]["value"] == int(miss.sum())
    assert cmd["n_outside"]["value"] == int((~miss & ~inside).sum())
    reportable = 0
    hidden = 0
    for cell in cmd["cells"]:
        n = int(counts[cell["gi"], cell["ci"]])
        if n < floor:
            assert cell["below_floor"] is True and cell["n"]["value"] is None
            hidden += n
        else:
            assert cell["below_floor"] is False and cell["n"]["value"] == n
            reportable += cell["n"]["value"]
    assert reportable + hidden + cmd["n_missing"]["value"] + cmd["n_outside"]["value"] == n_test
    assert reportable > 0 and int(counts.sum()) == reportable + hidden
    size = len(json.dumps(real_doc, ensure_ascii=False, sort_keys=True).encode("utf-8"))
    assert size < MAX_BYTES, size
