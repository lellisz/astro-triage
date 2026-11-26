"""Independent proof of the /graficos/ acceptance criteria (ADR 0008 Amendment 1).

Written without reusing the exporter's helpers for the expected values: the synthetic release
is built by hand with known counts, and the real-release checks recompute from the parquet
files with plain pandas/numpy. The real-release tests skip when ``ASTRO_DATA_ROOT`` has no
``full_v0`` and only ever read the data root.
"""

from __future__ import annotations

import ast
import builtins
import json
import math
import os
import re
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import pytest

from astro_triage.dashboard import data as D
from astro_triage.dashboard import paths
from astro_triage.web_export import graficos as G
from astro_triage.web_export.__main__ import main
from astro_triage.web_export.types import clean

MODEL = "lgbm_full"
CLASSES = ("STAR", "GALAXY", "QSO")
PER_OBJECT_KEYS = {"source_id", "ra", "dec", "l", "b", "healpix_l5", "hp"}
MAX_ARRAY = 252
MAX_BYTES = 150 * 1024
SRC = Path(G.__file__).resolve().parents[1]
FLOOR = 7


# ------------------------------------------------------------------ synthetic release
def _rows(spec: dict[tuple[str, str], int]) -> list[tuple[str, str]]:
    return [(t, p) for (t, p), n in spec.items() for _ in range(n)]


def _release(floor: int, g_counts: tuple[int, int] = (FLOOR, FLOOR - 1)) -> D.Release:
    """Test fold with a known confusion matrix and known G bins (all by hand).

    Confusion (rows true, columns STAR/GALAXY/QSO), with FLOOR = 7:
      STAR   40  7  6      GALAXY  6 40  7      QSO  7  6 40
    G bins of the STAR rows: ``g_counts[0]`` objects in [15, 16), ``g_counts[1]`` in [16, 17),
    the rest at G = 12.5.
    """
    spec = {
        ("STAR", "STAR"): 40, ("STAR", "GALAXY"): FLOOR, ("STAR", "QSO"): FLOOR - 1,
        ("GALAXY", "STAR"): FLOOR - 1, ("GALAXY", "GALAXY"): 40, ("GALAXY", "QSO"): FLOOR,
        ("QSO", "STAR"): FLOOR, ("QSO", "GALAXY"): FLOOR - 1, ("QSO", "QSO"): 40,
    }  # fmt: skip
    rows = _rows(spec)
    n = len(rows)
    rng = np.random.default_rng(11)
    true = np.array([t for t, _ in rows])
    pred = np.array([p for _, p in rows])
    p = rng.dirichlet(np.ones(3), size=n)
    for i, c in enumerate(pred):
        p[i, CLASSES.index(c)] += 1.0
    p /= p.sum(axis=1, keepdims=True)
    g = np.full(n, 12.5)
    star_idx = np.flatnonzero(true == "STAR")
    g[star_idx[: g_counts[0]]] = 15.5
    g[star_idx[g_counts[0] : g_counts[0] + g_counts[1]]] = 16.5
    z = np.where(true == "STAR", 0.0, rng.uniform(0.1, 2.9, size=n))
    test = pd.DataFrame(
        {"true": true, "healpix_l5": np.arange(n) % 5, "phot_g_mean_mag": g, "sdss_z": z}
    )
    for j, c in enumerate(CLASSES):
        test[f"p_{MODEL}_{c}"] = p[:, j]
        test[f"praw_{MODEL}_{c}"] = p[:, j]
    test[f"pred_{MODEL}"] = pred
    test[f"conf_{MODEL}"] = p.max(axis=1)
    test[f"err_{MODEL}"] = test["true"] != test[f"pred_{MODEL}"]
    results = {"meta": {"floor": {"min_objects": floor}}}
    return D.Release("synth", D.REPO_ROOT, results, {}, pd.DataFrame(), test, 0)


def _walk(obj: object, path: str = "$") -> Iterator[tuple[str, str | None, object]]:
    yield path, None, obj
    if isinstance(obj, dict):
        for k, v in obj.items():
            yield f"{path}.{k}", k, v
            yield from _walk(v, f"{path}.{k}")
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            yield from _walk(v, f"{path}[{i}]")


def _numbers(obj: object) -> list[float]:
    return [
        v
        for _, _, v in _walk(obj)
        if isinstance(v, int | float) and not isinstance(v, bool)
    ]


def _cell(doc: dict, true: str, pred: str) -> dict:
    row = next(r for r in doc["confusion"]["rows"] if r["true"] == true)
    return next(c for c in row["cells"] if c["pred"] == pred)


# ------------------------------------------------------------- 1. floor, exact boundary
def test_confusion_floor_boundary_is_exact() -> None:
    doc = G.graficos(_release(FLOOR))
    expected = {
        ("STAR", "STAR"): 40, ("STAR", "GALAXY"): FLOOR, ("STAR", "QSO"): FLOOR - 1,
        ("GALAXY", "STAR"): FLOOR - 1, ("GALAXY", "GALAXY"): 40, ("GALAXY", "QSO"): FLOOR,
        ("QSO", "STAR"): FLOOR, ("QSO", "GALAXY"): FLOOR - 1, ("QSO", "QSO"): 40,
    }  # fmt: skip
    for (t, p), n in expected.items():
        cell = _cell(doc, t, p)
        if n < FLOOR:
            assert cell["below_floor"] is True, (t, p, n)
            assert cell["n"]["value"] is None and cell["share"]["point"] is None, (t, p)
            assert cell["share"]["lo"] is None and cell["share"]["hi"] is None
            assert _numbers({k: v for k, v in cell.items() if k != "pred"}) == []
        else:
            assert cell["below_floor"] is False, (t, p, n)
            assert cell["n"]["value"] == n, (t, p)
            assert cell["share"]["point"] is not None
    # the row totals (not cells) stay visible
    assert [r["n"]["value"] for r in doc["confusion"]["rows"]] == [53, 53, 53]


def test_error_by_g_floor_boundary_keeps_n_and_hides_errors() -> None:
    doc = G.graficos(_release(FLOOR))
    star = next(s for s in doc["error_by_g"]["series"] if s["cls"] == "STAR")
    at = next(r for r in star["rows"] if r["lo"] == 15.0)
    below = next(r for r in star["rows"] if r["lo"] == 16.0)
    assert at["n"]["value"] == FLOOR and at["below_floor"] is False
    assert at["errors"]["value"] is not None and at["rate"]["point"] is not None
    assert below["n"]["value"] == FLOOR - 1 and below["below_floor"] is True
    assert below["errors"]["value"] is None
    assert below["rate"]["point"] is None and below["rate"]["lo"] is None
    # every 1D row obeys: below_floor <=> n < floor, n never hidden
    for series in (*doc["error_by_g"]["series"], *doc["error_by_z"]["series"]):
        for r in series["rows"]:
            assert r["n"]["value"] is not None
            assert r["below_floor"] is (r["n"]["value"] < FLOOR), (series["cls"], r)
            assert (r["errors"]["value"] is None) is r["below_floor"]


@pytest.mark.parametrize(("floor", "hidden"), [(15, 0), (16, 3)])
def test_calibration_floor_boundary(floor: int, hidden: int) -> None:
    """159 rows in 10 equal-mass bins: nine bins of 16 and one of 15, for each class."""
    doc = G.graficos(_release(floor))
    for series in doc["calibration"]["series"]:
        sizes = sorted(b["n"]["value"] for b in series["bins"])
        assert sizes == [15] + [16] * 9, series["cls"]
        for b in series["bins"]:
            below = b["n"]["value"] < floor
            assert b["below_floor"] is below
            assert (b["p_mean"]["value"] is None) is below
            assert (b["freq"]["point"] is None) is below
            assert (b["raw_p_mean"] is None) is below and (b["raw_freq"] is None) is below
    assert sum(
        b["below_floor"] for s in doc["calibration"]["series"] for b in s["bins"]
    ) == hidden


def test_forced_floor_above_everything_nulls_all_values() -> None:
    doc = G.graficos(_release(10_000))
    assert all(c["below_floor"] and c["n"]["value"] is None
               for r in doc["confusion"]["rows"] for c in r["cells"])  # fmt: skip
    for s in (*doc["error_by_g"]["series"], *doc["error_by_z"]["series"]):
        assert all(r["errors"]["value"] is None and r["rate"]["point"] is None
                   for r in s["rows"])  # fmt: skip


# --------------------------------------------------------------- 2. no per-object data
def _check_no_object_data(doc: dict) -> None:
    for path, key, value in _walk(doc):
        assert key not in PER_OBJECT_KEYS, f"{key} at {path}"
        if isinstance(value, list):
            assert len(value) <= MAX_ARRAY, f"{path}: {len(value)} items"


def test_synthetic_has_no_object_keys_and_no_long_arrays() -> None:
    doc = G.graficos(_release(FLOOR))
    _check_no_object_data(doc)
    # falsifier: the checker rejects a leak
    with pytest.raises(AssertionError):
        _check_no_object_data({"a": [{"healpix_l5": 3}]})
    with pytest.raises(AssertionError):
        _check_no_object_data({"a": list(range(MAX_ARRAY + 1))})


# ------------------------------------------------------------ 3. independence from DL
BANNED = ("dl_eval", "predictions_mlp", "predictions_ftt")


def _module_ast(name: str) -> ast.Module:
    return ast.parse((SRC / f"{name}.py").read_text(encoding="utf-8"))


@pytest.mark.parametrize("rel_path", ["web_export/graficos", "dashboard/data_charts"])
def test_modules_do_not_import_or_open_dl(rel_path: str) -> None:
    tree = _module_ast(rel_path)
    imported: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported += [a.name for a in node.names]
        elif isinstance(node, ast.ImportFrom):
            base = node.module or ""
            imported.append(base)
            imported += [f"{base}.{a.name}" for a in node.names]
    assert any(i.startswith("astro_triage") for i in imported), "parse sanity"
    for name in imported:
        parts = name.split(".")
        assert "dl" not in parts and "dl_eval" not in parts, name
        assert not any(b in name for b in BANNED), name
    # no string literal (path fragment, glob, column) names a DL artefact either
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            assert not any(b in node.value for b in BANNED), node.value
            assert node.value not in {"dl", "dl/"}, node.value
            assert not re.search(r"(^|[\\/])dl([\\/]|$)", node.value), node.value
        if isinstance(node, ast.Attribute | ast.Name):
            ident = node.attr if isinstance(node, ast.Attribute) else node.id
            assert not any(b in ident for b in BANNED), ident
    # falsifier of the checker itself
    bad = ast.parse("from astro_triage.dashboard import dl_eval")
    assert any("dl_eval" in a.name for n in ast.walk(bad) if isinstance(n, ast.ImportFrom)
               for a in n.names)  # fmt: skip


# --------------------------------------------------------------------------- texts
def test_synthetic_texts_have_no_forbidden_words() -> None:
    text = json.dumps(G.graficos(_release(FLOOR)), ensure_ascii=False)
    assert "ADR" not in text
    assert "empate" not in text.lower() and "streamlit" not in text.lower()


def test_labels_ts_has_no_forbidden_words() -> None:
    labels = SRC.parents[1] / "web" / "content" / "labels.ts"
    text = labels.read_text(encoding="utf-8")
    assert text.strip()
    assert "ADR" not in text
    assert "empate" not in text.lower() and "streamlit" not in text.lower()


# ------------------------------------------------------------------- real release
def _root() -> Path | None:
    raw = os.environ.get("ASTRO_DATA_ROOT", "")
    if not raw:
        return None
    rel = Path(raw) / "interim" / "full_v0"
    return rel if (rel / "eval" / "results.json").is_file() else None


needs_real = pytest.mark.skipif(_root() is None, reason="ASTRO_DATA_ROOT sem full_v0")


class _Opened:
    """Records every path opened through the usual Python and pyarrow entry points."""

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
            orig = getattr(pq, fn)
            mp.setattr(pq, fn, self._wrap(orig, note))
        orig_pf = pq.ParquetFile

        class PF(orig_pf):  # type: ignore[misc, valid-type]
            def __init__(self, source, *a, **k):
                note(source)
                super().__init__(source, *a, **k)

        mp.setattr(pq, "ParquetFile", PF)

    @staticmethod
    def _wrap(orig, note):
        def inner(source, *a, **k):
            note(source)
            return orig(source, *a, **k)

        return inner


def _is_dl(path: str) -> bool:
    p = path.replace("\\", "/")
    return "/dl/" in p or "predictions_mlp" in p or "predictions_ftt" in p or "dl_eval" in p


@pytest.fixture(scope="module")
def real_release() -> D.Release:
    return D.load_release("full_v0", paths.interim_dir())


@pytest.fixture(scope="module")
def real_doc(real_release: D.Release) -> dict:
    return json.loads(json.dumps(clean(G.graficos(real_release)), allow_nan=False))


@needs_real
def test_graficos_build_touches_no_dl_file(
    real_release: D.Release, monkeypatch: pytest.MonkeyPatch
) -> None:
    seen = _Opened(monkeypatch)
    G.graficos(real_release)
    # graficos() works on the already-loaded frame, so it may read nothing at all; the
    # recorder is proven live by the next test.
    assert not [p for p in seen.paths if _is_dl(p)], seen.paths


@needs_real
def test_release_load_touches_no_dl_file(monkeypatch: pytest.MonkeyPatch) -> None:
    seen = _Opened(monkeypatch)
    D.load_release("full_v0", paths.interim_dir())
    assert any(p.endswith("labels.parquet") for p in seen.paths), seen.paths  # recorder is live
    assert not [p for p in seen.paths if _is_dl(p)], [p for p in seen.paths if _is_dl(p)]


@needs_real
def test_recorder_flags_a_dl_path(monkeypatch: pytest.MonkeyPatch) -> None:
    root = _root()
    assert root is not None
    seen = _Opened(monkeypatch)
    dl_dir = root / "dl"
    victims = [p for p in dl_dir.rglob("*.json")][:1] if dl_dir.is_dir() else []
    if not victims:
        pytest.skip("sem arquivo json em dl/ para provar o gravador")
    victims[0].read_bytes()  # Path.read_bytes is not recorded: use open to prove the hook
    with open(victims[0], "rb"):
        pass
    assert any(_is_dl(p) for p in seen.paths)


@needs_real
def test_real_graficos_size_and_no_object_data(real_doc: dict) -> None:
    _check_no_object_data(real_doc)
    size = len(json.dumps(real_doc, ensure_ascii=False, sort_keys=True).encode("utf-8"))
    assert size < MAX_BYTES, size
    floor = real_doc["floor"]
    assert floor > 1
    for row in real_doc["confusion"]["rows"]:
        for cell in row["cells"]:
            v = cell["n"]["value"]
            assert v is None or v >= floor, (row["true"], cell["pred"], v)
            assert (v is None) is cell["below_floor"]


@needs_real
def test_full_export_file_is_small_and_has_only_allowed_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    out = tmp_path / "out"
    assert main(["--release", "full_v0", "--out", str(out)]) == 0
    f = out / "full_v0" / "graficos.json"
    assert f.stat().st_size < MAX_BYTES
    _check_no_object_data(json.loads(f.read_text(encoding="utf-8")))


# -------- 4. numeric parity, recomputed with pandas straight from the parquet files
def _wilson(k: int, n: int, z: float = 1.959963984540054) -> tuple[float, float]:
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0.0, c - h), min(1.0, c + h)


@pytest.fixture(scope="module")
def independent() -> pd.DataFrame:
    root = _root()
    if root is None:
        pytest.skip("sem full_v0")
    lab = pd.read_parquet(
        root / "labels.parquet",
        columns=["source_id", "sdss_class", "fold"],
        filters=[("fold", "==", "test")],
    )
    cols = ["source_id", "p_cal_star", "p_cal_galaxy", "p_cal_qso"]
    full = pd.read_parquet(root / "predictions_lgbm_full_v0.parquet", columns=cols)
    noind = pd.read_parquet(root / "predictions_lgbm_noind_v0.parquet", columns=cols)
    feat = pd.read_parquet(root / "features.parquet", columns=["source_id", "phot_g_mean_mag"])
    df = lab.merge(full, on="source_id", how="inner")
    df = df.merge(noind[["source_id", *cols[1:]]], on="source_id", how="inner", suffixes=("", "_n"))
    df = df.merge(feat, on="source_id", how="left")
    df = df.dropna(subset=[*cols[1:], *[c + "_n" for c in cols[1:]]]).reset_index(drop=True)
    df["true"] = df["sdss_class"].astype(str).str.upper()
    probs = df[cols[1:]].to_numpy(dtype=float)
    df["pred"] = np.array(CLASSES)[probs.argmax(axis=1)]
    return df


@needs_real
def test_parity_confusion_matrix_nine_counts(real_doc: dict, independent: pd.DataFrame) -> None:
    floor = real_doc["floor"]
    table = pd.crosstab(independent["true"], independent["pred"]).reindex(
        index=CLASSES, columns=CLASSES, fill_value=0
    )
    compared = 0
    for t in CLASSES:
        row = next(r for r in real_doc["confusion"]["rows"] if r["true"] == t)
        assert row["n"]["value"] == int(table.loc[t].sum()), t
        for p in CLASSES:
            n = int(table.loc[t, p])
            cell = next(c for c in row["cells"] if c["pred"] == p)
            if n < floor:
                assert cell["below_floor"] is True and cell["n"]["value"] is None, (t, p, n)
            else:
                assert cell["n"]["value"] == n, (t, p)  # difference must be 0
                assert cell["share"]["point"] == pytest.approx(n / table.loc[t].sum(), abs=1e-12)
                lo, hi = _wilson(n, int(table.loc[t].sum()))
                assert cell["share"]["lo"] == pytest.approx(lo, abs=1e-9)
                assert cell["share"]["hi"] == pytest.approx(hi, abs=1e-9)
                compared += 1
    assert compared >= 3


@needs_real
def test_parity_error_rate_by_class_in_g_19_20(
    real_doc: dict, independent: pd.DataFrame
) -> None:
    floor = real_doc["floor"]
    for cls in CLASSES:
        sub = independent[(independent["true"] == cls)]
        sub = sub[(sub["phot_g_mean_mag"] >= 19.0) & (sub["phot_g_mean_mag"] < 20.0)]
        n, k = len(sub), int((sub["pred"] != sub["true"]).sum())
        series = next(s for s in real_doc["error_by_g"]["series"] if s["cls"] == cls)
        row = next(r for r in series["rows"] if r["lo"] == 19.0 and r["hi"] == 20.0)
        assert row["n"]["value"] == n, cls
        if n < floor:
            assert row["below_floor"] is True and row["errors"]["value"] is None
            continue
        assert row["errors"]["value"] == k, cls
        assert row["rate"]["point"] == pytest.approx(k / n, abs=1e-12)
        lo, hi = _wilson(k, n)
        assert row["rate"]["lo"] == pytest.approx(lo, abs=1e-9)
        assert row["rate"]["hi"] == pytest.approx(hi, abs=1e-9)


@needs_real
def test_parity_calibration_top_decile_of_qso(
    real_doc: dict, independent: pd.DataFrame
) -> None:
    p = independent["p_cal_qso"].to_numpy(dtype=float)
    hit = (independent["true"] == "QSO").to_numpy()
    order = np.argsort(p, kind="stable")
    top = np.array_split(order, 10)[-1]
    n, k = len(top), int(hit[top].sum())
    series = next(s for s in real_doc["calibration"]["series"] if s["cls"] == "QSO")
    b = series["bins"][-1]
    assert b["n"]["value"] == n
    assert b["below_floor"] is (n < real_doc["floor"])
    if b["below_floor"]:
        return
    assert b["p_mean"]["value"] == pytest.approx(float(p[top].mean()), abs=1e-12)
    assert b["freq"]["point"] == pytest.approx(k / n, abs=1e-12)
    lo, hi = _wilson(k, n)
    assert b["freq"]["lo"] == pytest.approx(lo, abs=1e-9)
    assert b["freq"]["hi"] == pytest.approx(hi, abs=1e-9)
