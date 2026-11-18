"""Contract of the data layer of the Galaxias tab, prior-ruler phase (synthetic data)."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest
from test_dashboard_dl import _digit_literals

from astro_triage.dashboard import data_galaxy_prior as gp
from astro_triage.dashboard import verdict_rule as VR
from astro_triage.dashboard.data_dl import ClassCell, ClassVerdict
from astro_triage.dashboard.data_galaxies import (
    GALAXY_G_EDGES,
    GALAXY_Z_EDGES,
    error_rate_table,
)
from astro_triage.dashboard.data_galaxies_z import error_rate_table_z, galaxy_totals
from astro_triage.dashboard.data_galaxy_prior import (
    GalaxyPriorBlock,
    Interval,
    PriorGalaxy,
    VerdictKey,
)
from astro_triage.dashboard.html_galaxies import extreme_bands
from astro_triage.evaluation import dl_compare

MODULE = Path(gp.__file__)
V = VerdictKey


def cell(lo=0.0, hi=0.2, delta=0.05, d=0.1):
    """Cell as the loader would give it: the verdict token comes from the producer's function."""
    if None in (lo, hi, delta):
        verdict = ClassVerdict.NA if lo is None and hi is None else None
    else:
        verdict = ClassVerdict(dl_compare.class_verdict(d, lo, hi, delta, True))
    return ClassCell("GALAXY", d, lo, hi, delta, verdict, None, {})


def pg(prior, lo, hi, partial=False):
    iv = Interval((lo + hi) / 2, lo, hi) if lo is not None else None
    return PriorGalaxy(prior, "src", 10, iv, iv, partial)


def block(p1=(0.4, 0.6), p2=(0.7, 0.9), p3=(0.1, 0.2)):
    return GalaxyPriorBlock(
        "as_shipped",
        (pg("P1", *p1), pg("P2", *p2), pg("P3", *p3, partial=True)),
    )


# 1. galaxy_verdict -----------------------------------------------------------------
WIDE_CELL, NARROW_CELL = cell(0.0, 0.2, 0.05), cell(0.0, 0.02, 0.05)


def nets(mlp=NARROW_CELL, ftt=NARROW_CELL):
    return {"mlp_full": mlp, "ftt_full": ftt}


NO_DL = nets(None, None)
NA_CELL = cell(None, None, None)


@pytest.mark.parametrize(
    ("blk", "dl", "want"),
    [
        (block(), nets(WIDE_CELL), V.DEPENDS_WIDE),
        (block(), nets(), V.DEPENDS),
        (block(p2=(0.5, 0.9)), nets(NARROW_CELL, WIDE_CELL), V.WIDE),
        (block(p2=(0.5, 0.9)), nets(), V.NO_DIFFERENCE),
        # the prior axis does not need the DL: no usable network keeps the prior part
        (block(), NO_DL, V.DEPENDS),
        (block(p2=(0.5, 0.9)), NO_DL, V.NO_DIFFERENCE),
        (block(), {}, V.DEPENDS),
        (block(), nets(NA_CELL, cell(0.0, 0.2, None)), V.DEPENDS),
        # MLP n/a, FTT valid: the valid one decides the power axis
        (block(), nets(NA_CELL, WIDE_CELL), V.DEPENDS_WIDE),
        (block(), nets(NA_CELL, NARROW_CELL), V.DEPENDS),
        (block(p2=(0.5, 0.9)), nets(None, WIDE_CELL), V.WIDE),
        # unavailable only when the prior axis is
        (block(p1=(None, None)), nets(), V.UNAVAILABLE),
        (block(p2=(None, None)), nets(), V.UNAVAILABLE),
        (block(p1=(None, None)), NO_DL, V.UNAVAILABLE),
        (GalaxyPriorBlock("as_shipped", ()), nets(), V.UNAVAILABLE),
    ],
)
def test_verdict_two_axes_and_missing_data(blk, dl, want):
    assert gp.galaxy_verdict(blk, dl) is want


def test_verdict_touching_intervals_overlap_in_both_orders():
    for a, b in (((0.4, 0.6), (0.6, 0.9)), ((0.6, 0.9), (0.4, 0.6))):
        assert gp.galaxy_verdict(block(p1=a, p2=b), nets()) is V.NO_DIFFERENCE


def test_verdict_minimal_margin_is_disjoint_in_both_orders():
    for a, b in (((0.4, 0.6), (0.6001, 0.9)), ((0.6001, 0.9), (0.4, 0.6))):
        assert gp.galaxy_verdict(block(p1=a, p2=b), nets()) is V.DEPENDS


@pytest.mark.parametrize("p3", [(0.1, 0.2), (0.0, 1.0), (0.95, 0.99), (None, None)])
def test_verdict_ignores_p3(p3):
    for p2, want in (((0.7, 0.9), V.DEPENDS), ((0.5, 0.9), V.NO_DIFFERENCE)):
        got = gp.galaxy_verdict(block(p2=p2, p3=p3), nets())
        assert got is want


def test_power_networks_names_only_the_wide_ones_in_order():
    assert gp.power_networks(nets(WIDE_CELL, WIDE_CELL)) == ("mlp_full", "ftt_full")
    assert gp.power_networks(nets(NA_CELL, WIDE_CELL)) == ("ftt_full",)
    assert gp.power_networks(nets(WIDE_CELL, NARROW_CELL)) == ("mlp_full",)
    assert gp.power_networks(NO_DL) == ()


# 2. needs_power_note ---------------------------------------------------------------
def test_power_note_true_when_half_width_exceeds_delta():
    assert gp.needs_power_note(cell(0.0, 0.2, 0.05)) is True


def test_power_note_false_for_narrow_interval_and_exact_fit():
    assert gp.needs_power_note(cell(0.0, 0.02, 0.05)) is False
    assert (
        gp.needs_power_note(cell(-0.05, 0.05, 0.05)) is False
    )  # fits the band exactly


def test_power_note_is_the_phase4_rule_not_the_half_width():
    """Half-width equal to delta but shifted out of the band: the old criterion said no."""
    c = cell(0.0, 0.1, 0.05)
    assert (c.hi - c.lo) / 2 <= c.delta
    assert gp.needs_power_note(c) is VR.underpowered(c) is True


def test_power_note_agrees_with_verdict_rule_on_a_grid():
    for lo in (-0.2, -0.05, 0.0, 0.03):
        for hi in (0.02, 0.05, 0.1, 0.3):
            for d in (-0.1, 0.0, 0.1):
                if lo <= d <= hi:
                    c = cell(lo, hi, 0.05, d)
                    assert gp.needs_power_note(c) is VR.underpowered(c)


def test_power_note_false_for_a_win_and_for_below_the_floor():
    assert gp.needs_power_note(cell(0.01, 0.3, 0.05)) is False  # win, not a tie
    assert gp.needs_power_note(cell(None, None, None)) is False  # NA


@pytest.mark.parametrize(
    "c", [None, cell(0.0, 0.2, None), cell(None, 0.2, 0.05), cell(0.0, None, 0.05)]
)
def test_power_note_false_when_anything_is_missing(c):
    assert gp.needs_power_note(c) is False


# 3. floor --------------------------------------------------------------------------
def fake_release(n=100, passes=True, status="ok", min_objects=50, p3_floor=True):
    metric = {"status": status, "point": 0.5, "lo": 0.4, "hi": 0.6}
    node = {
        "prior_source": "s",
        "n_rows": 7,
        "floor": {"GALAXY": {"n": n, "passes": passes}},
        "variants": {
            "as_shipped": {
                "models": {
                    "lgbm_full": {
                        "metrics": {
                            "purity/GALAXY": metric,
                            "completeness/GALAXY": metric,
                        }
                    }
                }
            }
        },
    }
    priors = {p: node for p in ("P1", "P2", "P3")}
    results = {
        "meta": {"floor": {"min_objects": min_objects}},
        "populations": {"a_all": {"priors": priors}},
    }
    return SimpleNamespace(results=results, meta=results["meta"])


def test_floor_ok_gives_values():
    p = gp.load_galaxy_prior_block(fake_release()).get("P1")
    assert p.purity == Interval(0.5, 0.4, 0.6)
    assert p.completeness == Interval(0.5, 0.4, 0.6)
    assert p.n_rows == 7 and p.prior_source == "s"


def test_floor_exactly_at_min_objects_passes():
    assert gp.load_galaxy_prior_block(fake_release(n=50)).get("P1").purity is not None


@pytest.mark.parametrize(
    "kw", [{"n": 49}, {"passes": False}, {"status": "na"}, {"n": None}]
)
def test_floor_or_status_failure_gives_none_never_raw(kw):
    for p in gp.load_galaxy_prior_block(fake_release(**kw)).priors:
        assert p.purity is None and p.completeness is None


def test_missing_prior_node_is_none_and_p3_is_partial():
    rel = fake_release()
    del rel.results["populations"]["a_all"]["priors"]["P2"]
    blk = gp.load_galaxy_prior_block(rel)
    assert [p.prior for p in blk.priors] == ["P1", "P2", "P3"]
    assert blk.get("P2").purity is None
    assert [p.partial for p in blk.priors] == [False, False, True]


# 4. galaxy_error_rows --------------------------------------------------------------
def error_release(n_far=20, floor=10):
    g_bright, g_mid = GALAXY_G_EDGES[0] - 1, GALAXY_G_EDGES[1] + 0.2
    z_far = GALAXY_Z_EDGES[-1] + 0.5
    groups = [(g_bright, 0.0, 20, 5), (g_mid, 0.07, 20, 2), (g_mid, z_far, n_far, 3)]
    rows = []
    for g, z, n, err in groups:
        for i in range(n):
            e = i < err
            rows.append(
                {
                    "true": "GALAXY",
                    "phot_g_mean_mag": g,
                    "sdss_z": z,
                    "err_lgbm_full": e,
                    "pred_lgbm_full": "QSO" if e else "GALAXY",
                }
            )
    meta = {"floor": {"min_objects": floor}}
    return SimpleNamespace(test=pd.DataFrame(rows), meta=meta, results={"meta": meta})


LABELS = (gp.ALL_LABEL, gp.BRIGHTEST_LABEL, gp.NEAREST_LABEL, gp.FARTHEST_LABEL)


def rows_of(rel):
    """What the view does: fetch the totals and tables once, hand them to the pure rows."""
    return gp.galaxy_error_rows(
        galaxy_totals(rel, "lgbm_full"),
        error_rate_table(rel, "lgbm_full", gp.GALAXY, "G", GALAXY_G_EDGES),
        error_rate_table_z(rel, "lgbm_full"),
        int(rel.meta["floor"]["min_objects"]),
    )


def test_error_rows_do_not_scan_the_release():
    rel = error_release()
    args = (
        galaxy_totals(rel, "lgbm_full"),
        error_rate_table(rel, "lgbm_full", gp.GALAXY, "G", GALAXY_G_EDGES),
        error_rate_table_z(rel, "lgbm_full"),
    )
    assert gp.galaxy_error_rows(*args, 10) == rows_of(
        rel
    )  # no release in the signature


def test_error_rows_four_rows_in_order_with_counts():
    rows = rows_of(error_release())
    assert tuple(r[0] for r in rows) == LABELS
    assert [(r[1].n, r[1].errors) for r in rows] == [
        (60, 10),
        (20, 5),
        (20, 5),
        (20, 3),
    ]
    assert rows[0][1].rate == pytest.approx(10 / 60)
    assert rows[1][1].ci_lo <= rows[1][1].rate <= rows[1][1].ci_hi


def test_error_rows_band_below_floor_is_none_others_kept():
    rows = rows_of(error_release(n_far=9, floor=10))
    assert rows[3] == (gp.FARTHEST_LABEL, None)
    assert rows[0][1] is not None and rows[1][1] is not None


def test_error_rows_everything_below_floor_is_none():
    assert all(r[1] is None for r in rows_of(error_release(floor=10_000)))


def test_labels_match_extreme_bands_descriptions():
    """Contract: the module labels duplicate the literals of ``extreme_bands``.

    If ``extreme_bands`` changes its text, the band would silently become None.
    """
    rel = error_release()
    by_g = error_rate_table(rel, "lgbm_full", gp.GALAXY, "G", GALAXY_G_EDGES)
    texts = [
        what for what, _ in extreme_bands(by_g, error_rate_table_z(rel, "lgbm_full"), 1)
    ]
    assert texts == [gp.BRIGHTEST_LABEL, gp.NEAREST_LABEL, gp.FARTHEST_LABEL]


# 5. crossmatch_note ----------------------------------------------------------------
def test_crossmatch_note_returns_artifact_text_verbatim():
    text = "Selection by crossmatch: texto do artefato, intacto."
    rel = SimpleNamespace(results={"limitations": ["outra", text, 3]})
    assert gp.crossmatch_note(rel) == text


@pytest.mark.parametrize(
    "results",
    [{}, {"limitations": []}, {"limitations": ["outra"]}, {"limitations": "Selection"}],
)
def test_crossmatch_note_none_without_entry(results):
    assert gp.crossmatch_note(SimpleNamespace(results=results)) is None


def test_no_digit_literals_in_module():
    assert list(_digit_literals(MODULE)) == []


# smoke on the real release ---------------------------------------------------------
FULL_V0 = Path(__file__).resolve().parents[1] / "data" / "interim" / "full_v0"


@pytest.mark.skipif(not FULL_V0.exists(), reason="data/interim/full_v0 not available")
def test_smoke_real_release_invariants():
    from astro_triage.dashboard.data import load_release

    rel = load_release("full_v0")
    blk = gp.load_galaxy_prior_block(rel)
    assert [p.prior for p in blk.priors] == ["P1", "P2", "P3"]
    assert [p.partial for p in blk.priors] == [False, False, True]
    for p in blk.priors:
        assert p.purity is not None
        assert p.purity.lo <= p.purity.point <= p.purity.hi
    c1, c2 = blk.get("P1").completeness, blk.get("P2").completeness
    for f in ("point", "lo", "hi"):
        assert getattr(c1, f) == pytest.approx(
            getattr(c2, f), abs=1e-9
        )  # as delivered: equal up to float noise
    assert c1.lo <= c1.point <= c1.hi
    assert blk.get("P3").completeness is not None  # P3 differs (partial bands)
    assert len(rows_of(rel)) == 4


# 6. HTML of the tab (REV-01 texts, REV-02, REV-06, REV-07) ---------------------------
from astro_triage.dashboard import html_galaxy_prior as hg
from astro_triage.dashboard.limits_pt import translate

NOTE_100K = (
    "Selection by crossmatch: the matched class mix differs from the SDSS mix "
    "(match rate STAR 91.1%, QSO 65.7%, GALAXY 31.1% in the 100k batch); metrics "
    "describe the matched intersection, not the Gaia sky."
)
NOTE_FULL = (
    "Selection by crossmatch: the matched class mix differs from the SDSS mix "
    "(match rate STAR 91.1%, QSO 65.7%, GALAXY 31.1%, measured on the 100k pilot "
    "batch only and not recomputed for full_v0); metrics describe the matched "
    "intersection, not the Gaia sky."
)


def test_crossmatch_na_line_only_when_the_text_says_not_recomputed():
    assert translate(NOTE_100K)[1] and translate(NOTE_FULL)[1]
    assert "full_v0: n/a" in hg.crossmatch_html(NOTE_FULL, "full_v0")
    out = hg.crossmatch_html(NOTE_100K, "dataset_100k_v0")
    assert "n/a" not in out and "piloto" not in out
    assert "lote de 100k" in out


def test_crossmatch_without_note_shows_na_text_and_no_extra_line():
    out = hg.crossmatch_html(None, "full_v0")
    assert "full_v0: n/a" not in out


def test_power_phrase_comes_from_compare_text_and_only_for_wide_cells():
    from astro_triage.dashboard import compare_text as CT

    both = hg.verdict_cards_html(nets(WIDE_CELL, WIDE_CELL))
    assert both.count(CT.POWER_PHRASE) == 2
    one = hg.verdict_cards_html(nets(NARROW_CELL, WIDE_CELL))
    assert one.count(CT.POWER_PHRASE) == 1
    assert CT.POWER_PHRASE not in hg.verdict_cards_html(nets())
    assert CT.POWER_PHRASE not in hg.verdict_cards_html(nets(NA_CELL, None))


def test_verdict_texts_use_the_agreed_wording():
    assert (
        hg.verdict_text(V.DEPENDS) == "A pureza de galáxia depende do prior assumido."
    )
    assert hg.verdict_text(V.NO_DIFFERENCE) == "Sem diferença detectável entre P1 e P2."
    assert hg.verdict_text(V.DEPENDS_WIDE).endswith(
        "a comparação com as redes tem intervalos largos."
    )
    assert hg.verdict_text(V.WIDE).startswith(
        "Sem diferença detectável entre P1 e P2; "
    )


def test_delta_sentence_explains_the_band():
    assert "duas vezes o desvio entre sementes do LightGBM" in hg.DELTA_SENTENCE


def test_panel_and_crossmatch_titles_are_headings():
    out = hg.crossmatch_html(NOTE_100K, "r")
    assert "class='hs-pt'>" not in out
    assert "class='hs-pt' role='heading'" in out


def test_no_digit_literals_in_html_module():
    assert list(_digit_literals(Path(hg.__file__))) == []
