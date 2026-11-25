"""Web export parity (ADR 0008 section 6): every exported value equals its source function.

The export is a reshaping of dashboard functions; these tests compare, with exact equality, the
JSON documents with the direct output of those functions, on a synthetic release (tmp_path) and,
when ``ASTRO_DATA_ROOT`` has it, on the real ``full_v0``. A drift guard regenerates the
synthetic export and compares it with the versioned fixture the Next.js build reads.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from astro_triage.dashboard import compare_model as CM
from astro_triage.dashboard import data as D
from astro_triage.dashboard import (
    data_galaxy_prior,
    data_headline,
    data_sky,
    fmt,
    method_model,
    overview_model,
    paths,
    sky_model,
)
from astro_triage.dashboard.ui_state import UiState
from astro_triage.web_export import __main__ as export
from astro_triage.web_export import pages

ROOT = Path(__file__).resolve().parents[1]
FIXTURE_DIR = ROOT / "web" / "test" / "fixtures" / "fixture_v0"
NAME = "fixture_v0"
DOCS = ("meta", "overview", "sky", "galaxies", "compare", "method", "fmt_samples")
PIXELS = {100: 14, 101: 14, 102: 14, 103: 3}  # pixel -> test objects; 103 is below the floor
FLOOR = {"min_objects": 5, "min_pixels": 2}
CLASSES = ("STAR", "GALAXY", "QSO")
LIMITATIONS = [
    "Selection by crossmatch with SDSS favours bright objects.",
    "A second limitation, plain text.",
]
RESULTS_DL_MD = "# DL\n\n## Limitations\n\n- Only one seed was run.\n- Second bullet.\n\n## End\n"


# ------------------------------------------------------------------ synthetic release
def _cell(point: float, half: float = 0.02) -> dict:
    return {"status": "ok", "point": point, "lo": point - half, "hi": point + half}


def _models(pur: float, com: float, gpur: float, gcom: float) -> dict:
    metrics = {
        "purity/QSO": _cell(pur),
        "completeness/QSO": _cell(com),
        "purity/GALAXY": _cell(gpur),
        "completeness/GALAXY": _cell(gcom),
        "macro_f1": _cell(0.6),
    }
    return {
        "lgbm_full": {"metrics": metrics, "detail": {"accuracy": 0.7251}},
        "lgbm_noind": {"metrics": {"macro_f1": _cell(0.55)}, "detail": {}},
        "dsc_allosmod": {"metrics": {"macro_f1": _cell(0.5)}, "detail": {}},
    }


def _results() -> dict:
    priors = {}
    for i, p in enumerate(("P1", "P2", "P3")):
        priors[p] = {
            "prior": {"QSO": (0.005, 0.0158, 0.05)[i]},
            "prior_source": f"source {p}",
            "n_rows": 1000 + i,
            "floor": {"GALAXY": {"n": 30 + i, "passes": True}},
            "variants": {
                "as_shipped": {"models": _models(0.30 + i / 100, 0.90, 0.40, 0.85)},
                "prior_matched": {"models": _models(0.40 + i / 10, 0.80, 0.50 + i / 10, 0.7)},
            },
        }
    c_sdsc = {
        "n_rows": 321,
        "priors": {
            "P0": {
                "variants": {
                    "prior_matched": {
                        "models": _models(0.5, 0.5, 0.5, 0.5),
                        "diffs": {
                            "lgbm_full - dsc_allosmod": {"macro_f1": _cell(0.1, 0.05)},
                            "lgbm_full - lgbm_noind": {"macro_f1": _cell(0.05, 0.05)},
                        },
                    }
                }
            }
        },
    }
    return {
        "meta": {
            "header": "PROVISIONAL synthetic",
            "floor": FLOOR,
            "n_boot": 100,
            "n_test": sum(PIXELS.values()),
            "n_test_pixels": len(PIXELS),
            "pi_src_combmod": {"STAR": 0.9, "GALAXY": 0.09, "QSO": 0.01},
            "n_by_class": {"STAR": 20, "GALAXY": 15, "QSO": 10},
        },
        "populations": {"a_all": {"n_rows": 999, "priors": priors}, "c_sdsc": c_sdsc},
        "limitations": LIMITATIONS,
    }


def _train_report() -> dict:
    def entry(model: str) -> dict:
        return {
            "model_id": model,
            "n_features": 12,
            "best_iteration": 77,
            "temperature": 1.25,
            "train_class_fractions": {"STAR": 0.5, "GALAXY": 0.3, "QSO": 0.2},
            "gain_top10": [
                {"feature": "f_a", "share": 0.4},
                {"feature": "f_b", "share": 0.6},
            ],
            "val_calibration_fitted_here": {"accuracy": 0.7, "macro_f1": 0.6},
        }

    return {
        "models": [entry(m) for m in D.MODELS],
        "n_per_fold": {"train": 10, "val": 6, "test": sum(PIXELS.values())},
    }


def _dl_cell(win_galaxy: bool) -> dict:
    qso = {
        "d": 0.1,
        "ci": [0.05, 0.15],
        "delta": 0.02,
        "class_verdict": "win",
        "purity_seed": {
            "lgbm_full": {"min": 0.3, "median": 0.4, "max": 0.5},
            "mlp_full": {"min": 0.35, "median": 0.45, "max": 0.55},
        },
    }
    galaxy = (
        {"d": 0.1, "ci": [0.05, 0.15], "delta": 0.02, "class_verdict": "win"}
        if win_galaxy
        else {"d": 0.001, "ci": [-0.01, 0.01], "delta": 0.02, "class_verdict": "tie"}
    )
    return {
        "role": "headline",
        "verdict": "X wins" if win_galaxy else "tie: LightGBM stays",
        "classes": {"QSO": qso, "GALAXY": galaxy},
    }


def _results_dl() -> dict:
    return {
        "compare": {
            "pairs": {
                "mlp_full-lgbm_full": {"P2": {"0.80": _dl_cell(False)}},
                "ftt_full-lgbm_full": {"P2": {"0.80": _dl_cell(True)}},
            }
        }
    }


def build_release(base: Path, name: str = NAME, *, dl: bool = True) -> Path:
    """Write a complete synthetic release under ``base/interim/<name>``; return its root."""
    root = base / "interim" / name
    (root / "eval").mkdir(parents=True)
    rng = np.random.default_rng(20261025)
    rows = []
    sid = 0
    for hp, n_test in PIXELS.items():
        for fold, n in (("test", n_test), ("train", 2), ("val", 1)):
            for _ in range(n):
                sid += 1
                cls = CLASSES[sid % 3]
                z = {"STAR": 0.0, "GALAXY": 0.05 + sid / 100, "QSO": 0.6 + sid / 50}[cls]
                rows.append((sid, cls.lower(), z, hp, fold, 10.0 + hp % 7, 15.0 + sid))
    ids, cls_, z, hp, fold, lon, mag = zip(*rows, strict=True)
    labels = pd.DataFrame(
        {
            "source_id": ids,
            "sdss_class": cls_,
            "sdss_z": z,
            "healpix_l5": hp,
            "fold": fold,
            "ra": [r % 360 + 0.5 for r in lon],
            "dec": [(r % 80) - 40.0 for r in lon],
            "l": [(r * 37) % 360 for r in lon],
            "b": [((i * 13) % 140) - 70.0 for i, r in enumerate(lon)],
            "classprob_dsc_allosmod_star": rng.uniform(0, 1, len(ids)),
            "classprob_dsc_allosmod_galaxy": rng.uniform(0, 1, len(ids)),
            "classprob_dsc_allosmod_qso": rng.uniform(0, 1, len(ids)),
        }
    )
    labels.to_parquet(root / "labels.parquet")
    pd.DataFrame(
        {
            "source_id": ids,
            "phot_g_mean_mag": mag,
            "astrometric_params_solved": 31,
        }
    ).to_parquet(root / "features.parquet")
    test_ids = [i for i, f in zip(ids, fold, strict=True) if f == "test"]
    for m in D.MODELS:
        p = rng.dirichlet([1.2, 1.0, 0.8], len(test_ids))
        frame = {"source_id": test_ids}
        for j, c in enumerate(("star", "galaxy", "qso")):
            frame[f"p_cal_{c}"] = p[:, j]
            frame[f"p_raw_{c}"] = p[:, j]
        pd.DataFrame(frame).to_parquet(root / f"predictions_{m}_v0.parquet")
    write = lambda p, obj: p.write_text(json.dumps(obj), encoding="utf-8")
    write(root / "eval" / "results.json", _results())
    write(root / "train_report.json", _train_report())
    if dl:
        (root / "dl" / "eval").mkdir(parents=True)
        write(root / "dl" / "eval" / "results_dl.json", _results_dl())
        (root / "dl" / "eval" / "results_dl.md").write_text(RESULTS_DL_MD, encoding="utf-8")
    return root


@pytest.fixture
def synthetic(tmp_path, monkeypatch) -> D.Release:
    build_release(tmp_path)
    monkeypatch.setenv(paths.ENV_VAR, str(tmp_path))
    return export._load(NAME)


# ------------------------------------------------------------------ parity (acceptance 1, 2)
def _docs(rel: D.Release) -> dict:
    return export._documents(rel)


def _rows_by_state(doc: dict) -> dict:
    return {(r["rule"], r["prior"]): r for r in doc["rows"]}


def assert_parity(rel: D.Release) -> None:
    """Exact equality of the exported values with the direct output of their source."""
    docs = export.clean(_docs(rel))
    meta, ov, sky, gal, cmp_, met = (
        docs[k] for k in ("meta", "overview", "sky", "galaxies", "compare", "method")
    )
    # meta.states == ROW_ORDER
    assert [(s["rule"], s["prior"]) for s in meta["states"]] == [
        (r.value, p) for r, p in overview_model.ROW_ORDER
    ]
    # overview.rows[i].purity == overview_model.build(...) of the same state
    shipped = data_headline.shipped_state(rel, pages.MODEL)
    matched = data_headline.prior_states(rel, pages.MODEL)
    defs = data_headline.accuracy_defs(rel, pages.MODEL)
    assert len(ov["rows"]) == len(overview_model.ROW_ORDER)
    for i, (rule, prior) in enumerate(overview_model.ROW_ORDER):
        want = overview_model.build(shipped, matched, defs, UiState(rule, prior, False))
        v = want.selected.purity
        got = ov["rows"][i]
        assert (got["rule"], got["prior"]) == (rule.value, prior)
        assert got["purity"]["point"] == v.point
        assert got["purity"]["lo"] == (None if v.ci is None else v.ci.lo)
        assert got["purity"]["hi"] == (None if v.ci is None else v.ci.hi)
        c = want.selected.completeness
        assert got["completeness"]["point"] == c.point
    # sky.regions == test_pixel_stats
    stats = data_sky.test_pixel_stats(rel, pages.MODEL, D.pixel_table(rel))
    floor = int(rel.meta["floor"]["min_objects"])
    assert len(sky["regions"]) == len(stats) > 0
    for got, (_, row) in zip(sky["regions"], stats.iterrows(), strict=True):
        assert got["hp"] == int(row["healpix_l5"])
        assert got["n"] == int(row["n"])
        assert got["below_floor"] == (int(row["n"]) < floor)
    assert sky["cards"]["n_regions"]["value"] == sky_model.cards(stats, rel.meta).n_regions
    # galaxies.blocks == load_galaxy_prior_block
    for rule in ("as_shipped", "prior_matched"):
        block = data_galaxy_prior.load_galaxy_prior_block(rel, rule)
        assert len(gal["blocks"][rule]) == len(block.priors)
        for k, p in enumerate(block.priors):
            got = gal["blocks"][rule][k]
            assert got["prior"] == p.prior
            assert got["purity"]["point"] == (None if p.purity is None else p.purity.point)
            assert got["purity"]["lo"] == (None if p.purity is None else p.purity.lo)
            assert got["purity"]["hi"] == (None if p.purity is None else p.purity.hi)
    # compare.grid == compare_model.build(results)
    results = _results_dl_of(rel)
    if results is None:
        assert cmp_["status"] == "unavailable" and cmp_["grid"] is None
    else:
        grid = CM.build(results).grid
        assert cmp_["status"] == "ok" and len(cmp_["grid"]) == len(grid) > 0
        for k, g in enumerate(grid):
            assert cmp_["grid"][k]["state"] == g.state.value
            assert (cmp_["grid"][k]["cls"], cmp_["grid"][k]["net"]) == (g.cls, g.net)
    assert "Azul" not in gal["technical"]["missing_note"]
    nets = [n for n in gal["networks"].values() if n is not None]
    assert (gal["power_note"] is not None) == any(n["underpowered"] for n in nets)
    if results is not None:
        assert (cmp_["power_note"] is not None) == any(g.underpowered for g in grid)
        assert [p["reference"] for p in cmp_["purity"]].count(True) == 1
        for p in cmp_["purity"]:
            assert (p["net"] is None) == p["reference"]
        assert all(g["cls_label"] for g in cmp_["grid"])
    # method.facts.n_test == method_facts(rel).n_test
    facts = method_model.method_facts(rel)
    assert met["facts"]["n_test"]["text"] == facts.n_test
    assert met["facts"]["n_test"]["value"] == rel.meta["n_test"]
    assert met["facts"]["n_boot"]["text"] == facts.n_boot


def _results_dl_of(rel: D.Release) -> dict | None:
    from astro_triage.dashboard import data_dl

    return data_dl.load_results_dl(rel)


def test_parity_on_synthetic_release(synthetic) -> None:
    assert_parity(synthetic)


def test_parity_detects_a_changed_value(synthetic, monkeypatch) -> None:
    """Falsifier: if an exported number drifted, the parity check must fail."""
    real = pages.overview

    def drifted(rel, pixels):
        doc = real(rel, pixels)
        doc["rows"][0]["purity"]["point"] += 1e-9
        return doc

    monkeypatch.setattr(pages, "overview", drifted)
    with pytest.raises(AssertionError):
        assert_parity(synthetic)


def test_compare_without_dl_artifact_is_unavailable(tmp_path, monkeypatch) -> None:
    build_release(tmp_path, dl=False)
    monkeypatch.setenv(paths.ENV_VAR, str(tmp_path))
    rel = export._load(NAME)
    assert_parity(rel)
    assert export.clean(pages.compare(rel))["status"] == "unavailable"


def test_parity_on_real_release(monkeypatch) -> None:
    root = os.environ.get("ASTRO_DATA_ROOT")
    path = Path(root or "") / "interim" / "full_v0" / "eval" / "results.json"
    if not root or not path.is_file():
        pytest.skip("ASTRO_DATA_ROOT sem interim/full_v0/eval/results.json")
    rel = D.load_release("full_v0", Path(root) / "interim")
    assert_parity(rel)


# ------------------------------------------------------------------ format parity (acceptance 3)
def test_fmt_samples_equal_the_formatters() -> None:
    samples = pages.fmt_samples()
    assert len(samples) > 20
    for item in samples:
        assert item["text"] == getattr(fmt, item["fn"])(*item["args"]), item


def test_fmt_samples_cover_every_pt_function_once_declared() -> None:
    assert {i["fn"] for i in pages.fmt_samples()} == {n for n, _ in pages.FMT_SAMPLES}


# ------------------------------------------------------------------ drift guard (acceptance 4)
def _strip_stamp(doc: dict) -> dict:
    return {k: v for k, v in doc.items() if k != "generated_at"}


def test_versioned_fixture_matches_a_fresh_export(tmp_path, monkeypatch) -> None:
    """Regenerate with the exporter; any difference but ``generated_at`` is a stale fixture."""
    build_release(tmp_path / "data")
    monkeypatch.setenv(paths.ENV_VAR, str(tmp_path / "data"))
    out = tmp_path / "out"
    assert export.main(["--release", NAME, "--out", str(out)]) == 0
    fresh = out / NAME
    assert sorted(p.name for p in fresh.glob("*.json")) == sorted(f"{d}.json" for d in DOCS)
    assert FIXTURE_DIR.is_dir(), "run the exporter on the synthetic release to create it"
    assert sorted(p.name for p in FIXTURE_DIR.glob("*.json")) == sorted(
        f"{d}.json" for d in DOCS
    )
    for d in DOCS:
        a = json.loads((fresh / f"{d}.json").read_text(encoding="utf-8"))
        b = json.loads((FIXTURE_DIR / f"{d}.json").read_text(encoding="utf-8"))
        if isinstance(a, dict):
            a, b = _strip_stamp(a), _strip_stamp(b)
        assert a == b, f"{d}.json differs from the versioned fixture"


# ------------------------------------------------------------------ digit-free source (extra)
def test_method_model_source_has_no_hand_written_digits() -> None:
    from test_dashboard_dl import _digit_literals

    assert list(_digit_literals(Path(method_model.__file__))) == []
