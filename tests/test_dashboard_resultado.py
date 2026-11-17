"""Overview 'Régua do prior' screen: data_headline states, HTML builders, rules, real release."""

from __future__ import annotations

import json
import re
from pathlib import Path
from types import SimpleNamespace

import pytest

# Same detector and literal walker as the phase 4 sky sweep (imported, not copied).
from test_dashboard_phase4_sky2 import _literals, has_hand_written_digits

from astro_triage.dashboard import data_headline as H
from astro_triage.dashboard import html_overview as HO
from astro_triage.dashboard import style_overview, theme

ROOT = Path(__file__).resolve().parents[1]
DASH = ROOT / "src" / "astro_triage" / "dashboard"
FULL_V0 = ROOT / "data" / "interim" / "full_v0"
XSS = "<script>x</script>"
PURITY = {"P1": 0.40, "P2": 0.70, "P3": 0.55}
QSO = {"P1": 0.005, "P2": 0.0158, "P3": 0.05}
WARN_P3 = "P3 vale só para parte do céu (bandas B e C; ADR 0003)"


def _m(point, status="ok"):
    return {"status": status, "point": point, "lo": point - 0.02, "hi": point + 0.02}


def _variant(pur, com, status="ok"):
    return {
        "models": {
            "lgbm_full": {
                "metrics": {"purity/QSO": _m(pur, status), "completeness/QSO": _m(com)},
                "detail": {"accuracy": 0.7251},
            }
        }
    }


def _results(drop=()):
    priors = {}
    for p in ("P1", "P2", "P3"):
        if p in drop:
            continue
        priors[p] = {
            "prior": {"QSO": QSO[p]},
            "variants": {
                "as_shipped": _variant(0.30, 0.90),
                "prior_matched": _variant(PURITY[p], 0.80),
            },
        }
    return {"populations": {"a_all": {"priors": priors}}}


def _rel(tmp_path, results=None, dl=None):
    if dl is not None:
        d = tmp_path / "dl" / "eval"
        d.mkdir(parents=True)
        (d / "results_dl.json").write_text(json.dumps(dl), encoding="utf-8")
    return SimpleNamespace(root=tmp_path, results=results or _results())


def _states(**over):
    out = {
        p: {
            "prior": p,
            "qso_prior": QSO[p],
            "purity": PURITY[p],
            "purity_lo": PURITY[p] - 0.02,
            "purity_hi": PURITY[p] + 0.02,
            "completeness": 0.8,
            "completeness_lo": 0.78,
            "completeness_hi": 0.82,
            "partial": p in H.PARTIAL_PRIORS,
        }
        for p in H.SKY_PRIORS
    }
    out.update(over)
    return out


# (1) prior_states
def test_prior_states_keys_partial_and_no_p0(tmp_path):
    st = H.prior_states(_rel(tmp_path))
    assert list(st) == ["P1", "P2", "P3"]
    assert [p for p, s in st.items() if s["partial"]] == ["P3"]
    assert "P0" not in json.dumps(st)
    assert st["P2"]["purity"] == pytest.approx(0.70)


def test_low_count_cell_nulls_only_that_prior(tmp_path):
    res = _results()
    res["populations"]["a_all"]["priors"]["P1"]["variants"]["prior_matched"]["models"][
        "lgbm_full"
    ]["metrics"]["purity/QSO"]["status"] = "low_count"
    st = H.prior_states(_rel(tmp_path, res))
    assert all(st["P1"][k] is None for k in H._METRIC_FIELDS)
    assert st["P1"]["qso_prior"] == QSO["P1"]
    assert st["P2"]["purity"] == pytest.approx(0.70)
    assert st["P2"]["completeness_hi"] == pytest.approx(0.82)


def test_missing_p3_is_all_none_without_exception(tmp_path):
    st = H.prior_states(_rel(tmp_path, _results(drop=("P3",))))
    assert list(st) == ["P1", "P2", "P3"]
    assert st["P3"]["qso_prior"] is None
    assert all(st["P3"][k] is None for k in H._METRIC_FIELDS)
    assert st["P2"]["purity"] == pytest.approx(0.70)


# (2) accuracy_defs
def test_accuracy_defs_without_results_dl(tmp_path):
    a = H.accuracy_defs(_rel(tmp_path))
    assert a["shipped"] == pytest.approx(0.7251)
    for k in ("seeds_mean", "seeds_min", "seeds_max", "n_seeds", "regime"):
        assert a[k] is None
    assert a["balanced"] is False


def test_accuracy_defs_reads_seeds(tmp_path):
    dl = {
        "descriptive": {
            "regime": "BALANCED unweighted macro",
            "families": {"lgbm_full": {"accuracy": {"mean": 0.9, "min": 0.88, "max": 0.92}}},
            "systems": {f"lgbm_full_s{i}": {} for i in range(5)} | {"dl_x_s0": {}},
        }
    }
    a = H.accuracy_defs(_rel(tmp_path, dl=dl))
    assert (a["seeds_mean"], a["seeds_min"], a["seeds_max"]) == (0.9, 0.88, 0.92)
    assert a["n_seeds"] == 5
    assert a["regime"] == "BALANCED unweighted macro"
    assert a["balanced"] is True


# (3) builders
def _metric_classes(html: str) -> list[tuple[str, str]]:
    return re.findall(r"<div class='ao-r-k'>([^<]*)</div><div class='([^']+)'>", html)


def test_state_numerals_share_class_and_show_ci():
    s = _states()["P2"]
    out = HO.state_html("t", "s", "P2", s)
    got = _metric_classes(out)
    assert [k for k, _ in got] == ["Pureza de quasar", "Completude de quasar"]
    assert len({c for _, c in got}) == 1
    assert out.count("IC ") == 2 and "IC 95%: 68,0% a 72,0%" in out
    assert "IC 95%: 78,0% a 82,0%" in out


def test_none_metrics_render_na_and_no_bar():
    out = HO.state_html("t", "s", "P2", {k: None for k in H._METRIC_FIELDS})
    assert out.count("n/a") >= 4  # two numerals, two CI lines, dot field
    assert "ao-r-bar" not in out and "hs-q" not in out
    assert "n/a" in HO.state_html("t", "s", "P2", None)


def test_prior_label_and_forbidden_phrase():
    out = HO.state_html("t", "s", "P2", _states()["P2"])
    assert "prior de quasar assumido" in out
    assert "quasares no céu" not in out


def test_partial_warning_only_for_p3():
    states = _states()
    p3 = HO.states_html(states["P2"], states, "P3")
    assert p3.count(WARN_P3) == 1
    for sel in ("P1", "P2"):
        out = HO.states_html(states["P2"], states, sel)
        assert "vale só para parte do céu" not in out


def test_legend_inscription_and_blue_dots():
    s = _states()["P2"]
    out = HO.state_html("t", "s", "P2", s)
    legend = "1 ponto ≈ 0,1% da seleção · composição esperada sob o prior, não contagem de objetos"
    assert legend in out
    assert out.count("class='hs-q'") == round(1000 * s["purity"]) == 700
    assert out.count("class='hs-n'") == 300
    insc = HO.inscription_html()
    assert "De cada 1.000 objetos que o modelo chama de quasar, quantos são?" in insc


# (4) accuracy
def test_accuracy_two_definitions_stacked_and_labelled():
    defs = {
        "shipped": 0.7251,
        "seeds_mean": 0.9639,
        "seeds_min": 0.9600,
        "seeds_max": 0.9660,
        "n_seeds": 5,
        "regime": "BALANCED unweighted",
        "balanced": True,
    }
    out = HO.accuracy_html(defs)
    assert out.count("ao-r-accv") == 2
    assert "72,51%" in out and "96,39%" in out
    assert "como entregue" in out and "média de 5 sementes, regime balanceado" in out
    assert "acerto alto não garante pureza alta de quasar" in out


def test_accuracy_none_is_na():
    for defs in (None, {"shipped": None, "seeds_mean": None, "n_seeds": None}):
        out = HO.accuracy_html(defs)
        assert out.count("n/a") >= 2 and "None" not in out and "nan" not in out


# (5) escape
def test_injected_text_is_escaped():
    states = _states()
    outs = [
        HO.state_html(XSS, XSS, XSS, states["P2"]),
        HO.states_html(states["P2"], states, XSS),
        HO.ruler_html(states, XSS),
        HO.accuracy_html({"shipped": XSS, "n_seeds": XSS, "regime": XSS, "balanced": XSS}),
        HO.state_html("t", "s", "P2", {"qso_prior": XSS}),
    ]
    for out in outs:
        assert "<script" not in out.lower()
    assert "&lt;script&gt;" in outs[0]


# (6) rules
@pytest.mark.parametrize("name", ["html_overview.py", "views_overview.py"])
def test_no_hand_written_digits(name):
    bad = [
        (n.lineno, n.value[:70])
        for n in _literals(DASH / name)
        if isinstance(n.value, str) and has_hand_written_digits(n.value)
    ]
    assert not bad, bad


def _all_html() -> str:
    states = _states()
    defs = {"shipped": 0.7, "seeds_mean": 0.9, "seeds_min": 0.8, "seeds_max": 0.95,
            "n_seeds": 5, "regime": "BALANCED unweighted", "balanced": True}  # fmt: skip
    return "\n".join(
        [HO.inscription_html(), HO.accuracy_html(defs), HO.accuracy_html(None)]
        + [HO.states_html(states["P2"], states, p) for p in H.SKY_PRIORS]
        + [HO.ruler_html(states, p) for p in H.SKY_PRIORS]
        + [HO.states_html(None, {}, "P3"), HO.ruler_html({}, "P2")]
    )


def test_no_forbidden_words():
    low = _all_html().lower()
    for word in ("melhor", "melhora", "ganho", "equivalente", "deploy", "trace"):
        assert not re.search(rf"\b{word}\b", low), word


def test_overview_css_only_theme_hex():
    tokens = {
        v.lower()
        for v in vars(theme).values()
        if isinstance(v, str) and re.fullmatch(r"#[0-9a-fA-F]{6}", v)
    }
    found = re.findall(r"#[0-9a-fA-F]{3,8}\b", style_overview.overview_css())
    assert [h for h in found if h.lower() not in tokens] == []


def test_builders_are_single_line():
    states = _states()
    for out in (
        _all_html().split("\n")
        + [HO.state_html("t", "s", "P3", states["P3"])]
    ):
        assert "\n" not in out and "\r" not in out
    # each builder emits one line (the joined helper above uses '\n' only as separator)
    assert len(HO.states_html(states["P2"], states, "P3").splitlines()) == 1
    assert len(HO.accuracy_html(None).splitlines()) == 1
    assert len(HO.ruler_html(states, "P2").splitlines()) == 1


# (7) real release
def _app():
    from astro_triage.dashboard import data as D
    from astro_triage.dashboard import views_overview

    views_overview.overview(D.load_release("full_v0"))


def _run_real():
    from streamlit.testing.v1 import AppTest

    return AppTest.from_function(_app, default_timeout=300).run()


def _md(at) -> str:
    return "\n".join(m.value for m in at.markdown)


@pytest.mark.skipif(not FULL_V0.is_dir(), reason="data/interim/full_v0 absent")
def test_real_release_renders_and_p3_warning():
    at = _run_real()
    assert not at.exception, at.exception
    assert not at.error
    md = _md(at)
    for s in ("PROVISÓRIO · full_v0", "72,5%", "17,3%", "1,6%", "95,8%", "96,39%", "96,33%"):
        assert s in md, s
    assert WARN_P3 not in md.split("Ajustada ao prior")[1].split("</section>")[0]
    assert "Próximo passo" not in md and "at-rowp" not in md
    sel = [w for w in at.button_group if w.key == "overview_prior"]
    assert len(sel) == 1
    sel[0].set_value("P3")
    at = at.run()
    assert not at.exception, at.exception
    assert not at.error
    md3 = _md(at)
    assert WARN_P3 in md3 and "1,6%" in md3
    assert "Próximo passo" not in md3 and "at-rowp" not in md3

