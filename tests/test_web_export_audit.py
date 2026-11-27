"""Audit of the displayed content (plan 03, tasks 1-24 and D-3): one test per "Como verificar".

Every test here would fail on the pre-audit export: the old phrases, the band rate above the
pixel floor, the fixed two decimals, the signed zero, the proportion diffs and the stale
"100k" limitation. Synthetic release only; nothing is read from ``data/``.
"""

from __future__ import annotations

import html
import json
import re
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest
from test_web_export_parity import NAME, build_release

from astro_triage.dashboard import compare_text as CT
from astro_triage.dashboard import data_dl, data_sky, paths
from astro_triage.dashboard import html_galaxy_prior as hg
from astro_triage.dashboard.data_dl import PairVerdict
from astro_triage.dashboard.data_galaxies_z import GalaxyTotals
from astro_triage.dashboard.data_galaxy_prior import galaxy_error_rows
from astro_triage.dashboard.limits_pt import translate
from astro_triage.web_export import __main__ as export
from astro_triage.web_export import graficos, pages, web_text
from astro_triage.web_export.types import clean, est

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
STALE_100K = "Sample of 100k Gaia sources only; r_match 0.75 arcsec on SDSS plug positions."
# The four compare.json limitations of full_v0 as results_dl.md writes them (plan task 17).
COMPARE_LIMITS = (
    (
        "B0 (ADR 0006 section 6): zero hyperparameter search for both families; the deep models "
        "use untuned defaults and LightGBM a hand-set configuration, so unequal effort is not "
        "excluded (R2)."
    ),
    (
        "R17: the val-threshold sampling variance is not in the CI; a system whose threshold lands "
        "at a lower test completeness gets higher test purity. The realised test completeness is "
        "printed next to D and is not gated."
    ),
    (
        "Selection by crossmatch: the matched class mix differs from the SDSS mix (match rate STAR "
        "91.1%, QSO 65.7%, GALAXY 31.1%, measured on the 100k batch only and not recomputed for "
        "other releases); metrics describe the matched intersection, not the Gaia sky."
    ),
    "Ensemble rows are descriptive only; a verdict exists for H1 and H2 alone.",
)
# method.json limitations[0].original of full_v0 (plan task 18).
METHOD_CROSSMATCH = (
    "Selection by crossmatch: the matched class mix differs from the SDSS mix (match rate STAR "
    "91.1%, QSO 65.7%, GALAXY 31.1% in the 100k batch); metrics describe the matched "
    "intersection, not the Gaia sky."
)


@pytest.fixture
def synthetic(tmp_path, monkeypatch):
    build_release(tmp_path)
    monkeypatch.setenv(paths.ENV_VAR, str(tmp_path))
    return export._load(NAME)


def _src_hits(pattern: str, *dirs: Path) -> list[str]:
    rx = re.compile(pattern)
    hits = []
    for d in dirs:
        for f in d.rglob("*.py"):
            if f.resolve() == Path(__file__).resolve():
                continue
            for i, line in enumerate(f.read_text(encoding="utf-8").splitlines(), 1):
                if rx.search(line):
                    hits.append(f"{f.name}:{i}")
    return hits


# ------------------------------------------------------------------ tasks 1-3, 6-8, 21 (texts)
def test_old_phrases_are_gone_from_src_and_tests() -> None:
    assert _src_hits(r"sem diferença detectável, fica", SRC, ROOT / "tests") == []
    assert _src_hits(r"Sem diferença detectável\"", SRC, ROOT / "tests") == []
    assert _src_hits(r"não são a pureza nem a completude", SRC) == []
    assert _src_hits(r"ponto fixo desta aba", SRC) == []
    assert _src_hits(r"Sem diferença detectável entre P1 e P2", SRC, ROOT / "tests") == []
    assert _src_hits(r"As mais próximas", SRC, ROOT / "tests") == []


def test_pair_word_and_tie_title_follow_d1() -> None:
    assert CT.PAIR_WORD[PairVerdict.TIE] == "sem vitória do par, fica o LightGBM"
    assert web_text.TITLES[web_text.KEY_TIE] == "Nenhuma rede venceu o LightGBM"
    assert CT.FIXED_NOTE == "ponto fixo da comparação com as redes; a barra de ajustes não o muda"


def test_caveat_is_scoped_to_the_underpowered_class() -> None:
    one = web_text.headline_caveat(CT.POWER_PHRASE, ["Galáxia"])
    assert "na classe galáxia" in one and CT.POWER_PHRASE in one
    assert "classe" not in web_text.headline_caveat(CT.POWER_PHRASE, [])


def test_charts_scope_names_the_diagonal_as_completeness() -> None:
    text = web_text.charts_scope(292556, "LightGBM completo")
    assert "A diagonal da matriz é a completude da regra como entregue" in text


# ------------------------------------------------------------------ task 4 (protocol line)
@pytest.mark.parametrize(("status", "expect_text"), [("Proposed", True), (None, True), ("Accepted", False)])
def test_protocol_text_follows_the_adr_status(synthetic, monkeypatch, status, expect_text) -> None:
    monkeypatch.setattr(data_dl, "adr_status", lambda *a, **k: status)
    h = clean(pages.overview(synthetic, None))["headline"]
    if expect_text:
        assert "em proposta" in h["protocol_text"]
    else:
        assert h["protocol_text"] is None


def test_protocol_text_present_when_dl_is_missing(tmp_path, monkeypatch) -> None:
    build_release(tmp_path, dl=False)
    monkeypatch.setenv(paths.ENV_VAR, str(tmp_path))
    monkeypatch.setattr(data_dl, "adr_status", lambda *a, **k: "Proposed")
    h = clean(pages.overview(export._load(NAME), None))["headline"]
    assert "em proposta" in h["protocol_text"]


def test_real_adr_0006_is_still_proposed_so_the_line_shows() -> None:
    # Guard: if the ADR is accepted this test must be revisited, not the line kept by accident.
    assert data_dl.adr_status() == "Proposed"


# ------------------------------------------------------------------ task 5, 12 (contrast)
def test_contrast_names_the_sky_prior_and_the_test_mix(synthetic) -> None:
    doc = clean(pages.overview(synthetic, None))
    text = doc["headline"]["contrast"]["text"]
    prior = doc["rows"][0]["qso_prior"]["text"]
    assert f"num céu com {prior} de quasares (prior P2)" in text
    assert "mistura do teste" in text
    assert text.startswith("Acurácia ")


def test_data_box_accuracy_has_the_headline_precision(synthetic) -> None:
    # The synthetic release draws its predictions at random, so the two accuracies differ in
    # value here (on full_v0 both are 0,9639); the task fixes the precision: one decimal.
    doc = clean(pages.overview(synthetic, None))
    one_decimal = re.compile(r"\d+,\d%")
    assert one_decimal.fullmatch(doc["technical"]["data_box"]["accuracy"]["text"])
    assert one_decimal.fullmatch(doc["headline"]["contrast"]["accuracy"]["text"])


# ------------------------------------------------------------------ tasks 9, 10 (sky bands, D-9)
def _band_frame(rows: list[tuple]) -> pd.DataFrame:
    return pd.DataFrame(rows, columns=["band", "n_pixels", "n", "errors", "rate", "ci_lo", "ci_hi"])


def test_band_below_the_pixel_floor_is_na(monkeypatch) -> None:
    frame = _band_frame(
        [
            ("A", 13, 1284, 29, 29 / 1284, 0.0158, 0.0322),
            ("B", 109, 37373, 1958, 1958 / 37373, 0.05, 0.055),
            ("C", 475, 253899, 8577, 8577 / 253899, 0.033, 0.0345),
        ]
    )
    monkeypatch.setattr(data_sky, "lat_band_table", lambda stats: frame)
    bands = pages._bands(pd.DataFrame(), 20)
    a, b, c = bands
    assert a["rate"]["point"] is None and a["rate"]["text"] == "n/a"
    assert a["rate"]["ci_text"] == "n/a" and a["reportable"] is False
    # the counts stay: the floor hides the rate, not the band
    assert (a["n_pixels"], a["n"], a["errors"]) == (13, 1284, 29)
    assert b["reportable"] is True and c["reportable"] is True
    assert b["rate"]["point"] == pytest.approx(1958 / 37373)
    # exactly at the floor is reportable
    monkeypatch.setattr(data_sky, "lat_band_table", lambda stats: _band_frame([("A", 20, 10, 1, 0.1, 0.05, 0.2)]))
    assert pages._bands(pd.DataFrame(), 20)[0]["reportable"] is True


def test_band_decimals_follow_small_rate() -> None:
    def row(r: float) -> SimpleNamespace:
        return SimpleNamespace(rate=r, ci_lo=r * 0.9, ci_hi=r * 1.1)

    assert pages._band_rate(row(0.0338), True)["text"] == "3,4%"
    assert pages._band_rate(row(0.0052), True)["text"] == "0,52%"


# ------------------------------------------------------------------ task 11
def test_galaxy_error_rate_uses_small_rate_decimals() -> None:
    r = SimpleNamespace(rate=0.009025, ci_lo=0.0085, ci_hi=0.0096)
    assert pages._error_rate(r)["text"] == "0,90%"
    r2 = SimpleNamespace(rate=0.0338, ci_lo=0.03, ci_hi=0.04)
    assert pages._error_rate(r2)["text"] == "3,4%"


# ------------------------------------------------------------------ tasks 13, 14, 15
def test_signed_zero_has_no_sign() -> None:
    assert est(-0.000445, -0.001, 0.0005, kind="num", digits=3, signed=True)["text"] == "0,000"
    assert est(0.0004, -0.001, 0.0005, kind="num", digits=3, signed=True)["text"] == "0,000"
    assert est(-0.0006, -0.001, 0.0005, kind="num", digits=3, signed=True)["text"] == "-0,001"


def test_diffs_in_percentage_points() -> None:
    e = est(0.0111, -0.028, 0.0582, kind="pp", signed=True)
    assert e["text"] == "+1,11 p.p."
    assert e["ci_text"] == "IC 95%: -2,80 p.p. a +5,82 p.p."
    assert e["point"] == pytest.approx(0.0111)  # the value stays a proportion


def test_network_diffs_are_in_pp_everywhere(synthetic) -> None:
    pages_src = (SRC / "astro_triage" / "web_export" / "pages.py").read_text(encoding="utf-8")
    assert "digits=4, signed=True" not in pages_src
    comp = clean(pages.compare(synthetic))
    assert comp["grid"], "synthetic compare has a grid"
    for g in comp["grid"]:
        if g["diff"]["point"] is not None:
            assert g["diff"]["text"].endswith(" p.p.")
    for n in clean(pages.galaxies(synthetic))["networks"].values():
        if n:
            assert n["diff"]["text"].endswith(" p.p.")


# ------------------------------------------------------------------ task 16
def test_dsc_population_and_prior_in_portuguese(synthetic) -> None:
    dsc = clean(pages.overview(synthetic, None))["technical"]["dsc"]
    for key in ("population_text", "prior_text"):
        assert "c_sdsc" not in dsc[key]
    assert dsc["n_rows"]["text"] in dsc["population_text"]
    assert dsc["prior_text"] == "prior P0: mistura do teste"


# ------------------------------------------------------------------ tasks 17, 18 (limitations)
@pytest.mark.parametrize("original", COMPARE_LIMITS)
def test_compare_limitations_are_translated(original) -> None:
    text, ok = translate(original)
    assert ok, original
    assert "(sem tradução)" not in text


def test_r17_translation_does_not_promise_the_realised_completeness() -> None:
    text, _ = translate(COMPARE_LIMITS[1])
    assert "não é exibida nesta página" in text


def test_compare_page_has_no_untranslated_limit(synthetic, monkeypatch) -> None:
    md = synthetic.root / "dl" / "eval" / "results_dl.md"
    bullets = "\n".join(f"- {s}" for s in COMPARE_LIMITS)
    md.write_text(f"# DL\n\n## Limitations\n\n{bullets}\n\n## End\n", encoding="utf-8")
    limits = clean(pages.compare(export._load(NAME)))["limitations"]
    assert len(limits) == 4
    for t in limits:
        assert "(sem tradução)" not in t


ONLY_THAT_BATCH = (
    "essa taxa foi medida só nesse lote, e as outras versões dos dados não têm medida própria"
)


def test_only_that_batch_sentence_has_no_internal_name_nor_recompute_claim() -> None:
    # REV-02: the owner's sentence names no internal release id and makes no "not recomputed" claim.
    assert "full_v0" not in ONLY_THAT_BATCH
    assert "não recalculad" not in ONLY_THAT_BATCH


def test_crossmatch_caveat_is_the_same_in_method_and_compare() -> None:
    # AUD-31 / task 18, owner's option (a): the method sentence says the rate is the batch's
    # own and full_v0 has no measure of its own; it does not claim "not recomputed".
    text, ok = translate(METHOD_CROSSMATCH)
    assert ok, text
    assert ONLY_THAT_BATCH in text, text
    assert "no lote de 100k" in text
    assert "não recalculad" not in text
    assert "(sem tradução)" not in text


def test_crossmatch_galaxy_note_has_no_na_line_for_the_method_sentence() -> None:
    out = hg.crossmatch_html(METHOD_CROSSMATCH, "full_v0")
    assert html.unescape(ONLY_THAT_BATCH) in html.unescape(out)
    assert "full_v0: n/a" not in out
    assert ": n/a" not in out


def test_crossmatch_galaxy_note_on_the_page_uses_the_new_sentence(tmp_path, monkeypatch) -> None:
    root = build_release(tmp_path)
    path = root / "eval" / "results.json"
    results = json.loads(path.read_text(encoding="utf-8"))
    results["limitations"] = [
        METHOD_CROSSMATCH if x.startswith("Selection by crossmatch") else x for x in results["limitations"]
    ]
    path.write_text(json.dumps(results), encoding="utf-8")
    monkeypatch.setenv(paths.ENV_VAR, str(tmp_path))
    rel = export._load(NAME)
    method_texts = [x["text"] for x in clean(pages.method(rel))["limitations"]]
    assert any(ONLY_THAT_BATCH in t for t in method_texts), method_texts
    note = clean(pages.galaxies(rel))["crossmatch_note"]
    assert ONLY_THAT_BATCH in note, note
    assert "n/a" not in note and "não recalculad" not in note


# ------------------------------------------------------------------ D-3 (stale 100k sentence)
def _release_with_stale_limit(tmp_path, monkeypatch, name: str):
    root = build_release(tmp_path, name)
    path = root / "eval" / "results.json"
    results = json.loads(path.read_text(encoding="utf-8"))
    results["limitations"] = [*results["limitations"], STALE_100K]
    path.write_text(json.dumps(results), encoding="utf-8")
    monkeypatch.setenv(paths.ENV_VAR, str(tmp_path))
    return export._load(name)


def test_full_v0_replaces_the_100k_sentence_and_marks_it(tmp_path, monkeypatch) -> None:
    rel = _release_with_stale_limit(tmp_path, monkeypatch, "full_v0")
    lim = clean(pages.method(rel))["limitations"]
    stale = [x for x in lim if x["original"] == STALE_100K]
    assert len(stale) == 1
    text = stale[0]["text"]
    assert "100k" not in text and "100 mil" not in text
    assert text.endswith("(texto atualizado na exportação)")
    assert "full_v0" in text and "(sem tradução)" not in text
    rows = sum(json.loads((rel.root / "train_report.json").read_text())["n_per_fold"].values())
    assert str(rows) in text  # small synthetic count, no thousands separator
    assert "0,75" in text


def test_other_releases_keep_the_artifact_sentence(tmp_path, monkeypatch) -> None:
    rel = _release_with_stale_limit(tmp_path, monkeypatch, NAME)
    lim = clean(pages.method(rel))["limitations"]
    text = next(x["text"] for x in lim if x["original"] == STALE_100K)
    assert "texto atualizado na exportação" not in text


# ------------------------------------------------------------------ tasks 19, 20
def test_web_method_has_no_usage_and_labels_wilson(synthetic) -> None:
    m = clean(pages.method(synthetic))
    assert m["usage"] == []
    assert any("método de Wilson, por objeto" in r and "regiões vizinhas" in r for r in m["reading"])


# ------------------------------------------------------------------ task 21
def test_galaxy_extreme_labels_are_neutral(synthetic) -> None:
    rows = clean(pages.galaxies(synthetic))["error_rows"]
    labels = [r["label"] for r in rows]
    joined = " | ".join(labels)
    assert "mais próximas" not in joined and "mais distantes" not in joined
    assert "As mais" not in joined
    assert labels[1:], "row 0 is the overall row; the extremes follow"
    neutral = ("mais brilhantes", "redshift quase nulo", "maior redshift")
    for lab in labels[1:]:
        # A band dropped by the floor has no edges to print, so only its neutral word stays;
        # a band that is kept prints its limits (asserted with real edges below).
        assert re.search(r"[Gz]\s*[<≤≥>]", lab) or lab in neutral, lab


def test_galaxy_extreme_labels_print_the_band_limits() -> None:
    def table(rows: list[tuple]) -> pd.DataFrame:
        return pd.DataFrame(rows, columns=["label", "lo", "hi", "n", "errors", "rate", "ci_lo"])

    by_g = table([("< 18", 0.0, 18.0, 500, 10, 0.02, 0.01), ("18 a 19", 18.0, 19.0, 900, 9, 0.01, 0.005)])
    by_z = table(
        [
            ("≤ 0,001", 0.0, 0.001, 400, 40, 0.1, 0.07),
            ("0,1 a 0,2", 0.1, 0.2, 800, 8, 0.01, 0.005),
            ("≥ 0,45", 0.45, 9.0, 300, 30, 0.1, 0.07),
        ]
    )
    totals = GalaxyTotals(n=2900, errors=97, to_qso=50, to_star=47)
    labels = [label for label, _ in galaxy_error_rows(totals, by_g, by_z, 100)]
    assert labels == [
        "Todas as galáxias",
        "G < 18 (mais brilhantes)",
        "z ≤ 0,001 (redshift quase nulo)",
        "z ≥ 0,45",
    ]
    # below the floor: the band is dropped and only the neutral word remains, with no rate
    rows = galaxy_error_rows(totals, by_g, by_z, 1000)
    assert [label for label, _ in rows[1:]] == ["mais brilhantes", "redshift quase nulo", "maior redshift"]
    assert all(rate is None for _, rate in rows[1:])


# ------------------------------------------------------------------ task 22
def test_cmd_note_prints_the_outside_share() -> None:
    note = web_text.cmd_note(1079, 57531, 100, 292556)
    assert "57.531 (19,7% do teste)" in note
    assert "do teste" not in web_text.cmd_note(0, 5, 100, 0)


# ------------------------------------------------------------------ task 23
def _cal_row(mean_pred: float, observed: float, lo: float, hi: float) -> pd.Series:
    return pd.Series({"n": 29256, "mean_pred": mean_pred, "observed": observed, "ci_lo": lo, "ci_hi": hi})


@pytest.mark.parametrize("p", [4.6e-05, 0.0003, 0.004, 0.0099])
def test_calibration_never_prints_zero_for_a_positive_share(p) -> None:
    b = graficos._cal_bin(_cal_row(p, p, p / 2, p * 2), 100)
    assert b["p_mean"]["text"] not in ("0,0%", "0,00%")
    assert not re.fullmatch(r"0,0+%", b["p_mean"]["text"])
    assert not re.fullmatch(r"0,0+%", b["freq"]["text"])
    assert "0,0% a 0,0%" not in b["freq"]["ci_text"]


def test_calibration_above_small_rate_keeps_one_decimal() -> None:
    assert graficos._cal_bin(_cal_row(0.614, 0.616, 0.6, 0.63), 100)["p_mean"]["text"] == "61,4%"


# ------------------------------------------------------------------ task 24
def test_front_roles_have_no_jargon() -> None:
    for k in (2, 3, 4):
        label = pages._role_label(f"{data_dl.FRONT_PREFIX}{k}")
        assert not label.lower().startswith("frente"), label
    assert pages._role_label(f"{data_dl.FRONT_PREFIX}3") == "efeito dos indicadores na rede"
