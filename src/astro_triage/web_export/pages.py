"""One builder per exported page; each returns the dict of ADR 0008, Appendix A.

Every number and sentence is produced by an existing pure dashboard function; this module
only reshapes it into the common JSON types. It never imports streamlit, ``data_cached`` or
a ``views_*`` module, and it computes no metric.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Final

import numpy as np
import pandas as pd

from astro_triage.dashboard import compare_model as CM
from astro_triage.dashboard import compare_text as CT
from astro_triage.dashboard import data as D
from astro_triage.dashboard import (
    data_dl,
    data_galaxy_prior,
    data_headline,
    data_sky,
    fmt,
    method_model,
    method_notes,
    overview_model,
    overview_text,
    sky_map_html,
    sky_model,
    sky_text,
    ui_state,
)
from astro_triage.dashboard import html_dl as HD
from astro_triage.dashboard import html_method as HM
from astro_triage.dashboard import verdict_rule as VR
from astro_triage.dashboard.charts_base import MODEL_PT
from astro_triage.dashboard.data_overview_page import PRIOR, PageData, load_page
from astro_triage.dashboard.data_release import CLASSES, wilson
from astro_triage.dashboard.html_galaxies import SMALL_RATE
from astro_triage.dashboard.html_galaxy_prior import VERDICT_SCOPE, verdict_text
from astro_triage.dashboard.html_system import SEAL_PREFIX
from astro_triage.dashboard.limits_pt import omit_record_refs, translate
from astro_triage.dashboard.ui_state import LINE_LABEL, UiState
from astro_triage.web_export import web_text
from astro_triage.web_export.types import est, from_value, head, num

MODEL: Final = "lgbm_full"
DSC_POPULATION: Final = "c_sdsc"
DSC_PRIOR: Final = "P0"
DSC_MODELS: Final = (*D.MODELS, "dsc_allosmod")
PAIR_SEPARATOR: Final = " - "
DSC_METRIC: Final = "macro_f1"
MAP_DIGITS: Final = 4
CURVE_POINTS: Final = 37
EDGE_LON_DEG: Final = 180
ACCEPTED_STATUS: Final = "Accepted"
ENSEMBLE_PREFIX: Final = "ens_"
COMPARE_BLOCKS: Final = (
    "fixed_point",
    "answer_text",
    "reasons_text",
    "protocol_accepted",
    "freeze_exists",
    "pairs",
    "grid",
    "power_note",
    "purity",
    "rule_lines",
    "descriptive",
    "limitations",
)
FMT_SAMPLES: Final = (
    ("pt_int", ((0,), (-5,), (999,), (1000,), (1234567,), (12.7,))),
    (
        "pt_pct",
        (
            (0, 0),
            (0.5, 0),
            (0.12345, 1),
            (0.12345, 2),
            (1, 0),
            (-0.0123, 1),
            (0.00012, 3),
        ),
    ),
    (
        "pt_num",
        (
            (0, 3),
            (-1.5, 3),
            (1234.5678, 2),
            (0.5, 2),
            (0.25, 4, True),
            (-0.25, 4, True),
            (0, 4, True),
        ),
    ),
    ("pt_g", ((0,), (0.001,), (0.1,), (1,), (10,), (100,), (1000,), (-2.5,))),
    ("pt_ci_label", ((),)),
)
UNREADABLE_DL: Final = (
    "O arquivo da avaliação do aprendizado profundo está ilegível; "
    "os vereditos ficam n/a."
)


def _state_label(rule: ui_state.Rule, prior: str) -> str:
    return f"{ui_state.RULE_LABEL[rule]} · {prior}"


def meta(rel: D.Release) -> dict:
    """meta.json: the allowed (rule, prior) pairs, defaults, labels and the sky grid."""
    default = ui_state.DEFAULT
    return {
        **head(rel.name, "meta"),
        "states": [
            {
                "rule": rule.value,
                "prior": prior,
                "label": _state_label(rule, prior),
            }
            for rule, prior in overview_model.ROW_ORDER
        ],
        "default_state": {
            "rule": default.rule.value,
            "prior": default.prior,
            "tech": default.tech,
        },
        "shipped_prior": ui_state.SHIPPED_PRIOR,
        "partial_priors": sorted(data_headline.PARTIAL_PRIORS),
        "rule_labels": {r.value: label for r, label in ui_state.RULE_LABEL.items()},
        "seal_text": f"{SEAL_PREFIX} · {rel.name}",
        "nside": data_sky.HEALPIX_NSIDE,
        "band_edges_deg": list(data_sky.BAND_EDGES_DEG),
        "floor": {
            "min_objects": int(rel.meta["floor"]["min_objects"]),
            "min_pixels": int(rel.meta["floor"]["min_pixels"]),
        },
        "n_boot": int(rel.meta["n_boot"]),
    }


def _qso_prior(
    shipped: dict | None, matched: dict, rule: ui_state.Rule, prior: str
) -> dict:
    cell = shipped if rule is ui_state.Rule.SHIPPED else matched.get(prior)
    return num((cell or {}).get("qso_prior"), kind="pct")


def _rows(rel: D.Release) -> list[dict]:
    shipped = data_headline.shipped_state(rel, MODEL)
    matched = data_headline.prior_states(rel, MODEL)
    try:
        defs = data_headline.accuracy_defs(rel, MODEL)
    except (data_dl.DlDataError, ValueError, OSError):
        # Unreadable results_dl.json: same output as when the file is absent.
        defs = data_headline.accuracy_defs(
            replace(rel, root=rel.root / "_results_dl_unreadable"), MODEL
        )
    rows = []
    for rule, prior in overview_model.ROW_ORDER:
        state = UiState(rule, prior, False)
        model = overview_model.build(shipped, matched, defs, state)
        chosen = model.selected
        rows.append(
            {
                "rule": rule.value,
                "prior": prior,
                "partial": chosen.partial,
                "qso_prior": _qso_prior(shipped, matched, rule, prior),
                "purity": from_value(chosen.purity),
                "completeness": from_value(chosen.completeness),
                "accuracy": num(model.accuracy.point, kind="pct"),
                "answer_text": overview_text.answer(model, state),
                "partial_notice": model.partial_notice,
            }
        )
    return rows


def _class_mix(d: PageData) -> dict:
    return {c: num(d.class_mix.get(c), kind="pct", digits=0) for c in D.CLASSES}


def _data_box(d: PageData) -> dict:
    return {
        "n_pairs": num(d.n_pairs),
        "n_test": num(d.n_test),
        "n_errors": num(d.n_errors),
        "n_pixels": num(d.n_pixels),
        "accuracy": num(d.accuracy, kind="pct", digits=2),
        "qso_hits": num(d.qso_hits),
        "qso_total": num(d.qso_total),
        "class_mix": _class_mix(d),
    }


def _validation(d: PageData) -> dict | None:
    if d.validation is None:
        return None
    return {
        "purity": num(d.validation["purity"], kind="pct"),
        "completeness": num(d.validation["completeness"], kind="pct"),
    }


def _error_directions(rel: D.Release) -> list[dict]:
    frame = D.error_directions(rel, MODEL)
    return [
        {
            "true": str(r.true),
            "pred": str(r.pred),
            "n": num(r.n),
            "share": num(r.share, kind="pct"),
        }
        for r in frame.itertuples()
    ]


def _triage(d: PageData) -> dict | None:
    t = d.triage
    if t is None:
        return None
    return {
        "threshold": num(t["threshold"], kind="num", digits=2),
        "coverage": num(t["coverage"], kind="pct"),
        "accuracy_kept": num(t["accuracy_kept"], kind="pct"),
        "n_set_aside": num(t["n_set_aside"]),
        "errors_set_aside": num(t["errors_set_aside"]),
        "errors_total": num(t["errors_total"]),
    }


def _macro_f1(frame: pd.DataFrame, model: str) -> dict:
    rows = frame[frame["model"] == model]
    if rows.empty:
        return est(None, None, None, kind="num")
    r = rows.iloc[0]
    return est(r["point"], r["lo"], r["hi"], kind="num")


def _dsc(rel: D.Release) -> dict:
    metrics = D.metrics_frame(rel, DSC_POPULATION, DSC_PRIOR, "prior_matched")
    f1 = metrics[metrics["metric"] == DSC_METRIC]
    diffs = D.paired_diffs_frame(rel, DSC_POPULATION, DSC_PRIOR, "prior_matched")
    diffs = diffs[(diffs["metric"] == DSC_METRIC) & diffs["usable"]]
    pairs = []
    for r in diffs.itertuples():
        a, _, b = str(r.pair).partition(PAIR_SEPARATOR)
        pairs.append(
            {
                "a": MODEL_PT.get(a, a),
                "b": MODEL_PT.get(b, b),
                "est": est(r.point, r.lo, r.hi, kind="num", signed=True),
            }
        )
    return {
        "population": DSC_POPULATION,
        "prior": DSC_PRIOR,
        "n_rows": num(D.population_rows(rel, DSC_POPULATION)),
        "macro_f1": [
            {"model": m, "label": MODEL_PT[m], "est": _macro_f1(f1, m)}
            for m in DSC_MODELS
        ],
        "paired_diffs": pairs,
    }


def _models(rel: D.Release) -> list[dict]:
    out = []
    for model in D.MODELS:
        summary = D.model_summary(rel, model)
        gains = D.feature_importance(rel, model).sort_values(
            "share", ascending=False, kind="stable"
        )
        out.append(
            {
                "model": model,
                "label": MODEL_PT[model],
                "n_features": int(summary["n_features"]),
                "best_iteration": int(summary["best_iteration"]),
                "temperature": num(summary["temperature"], kind="num"),
                "importance": [
                    {"feature": str(r.feature), "gain": num(r.share, kind="pct")}
                    for r in gains.itertuples()
                ],
            }
        )
    return out


def _confidence_hist(rel: D.Release) -> list[dict]:
    return [
        {
            "lo": float(r.lo),
            "hi": float(r.hi),
            "n_right": int(r.n) - int(r.n_wrong),
            "n_wrong": int(r.n_wrong),
        }
        for r in D.confidence_histogram(rel, MODEL).itertuples()
    ]


def _calibration(rel: D.Release) -> dict:
    return {
        cls: [
            {"p_mean": float(r.mean_pred), "freq": float(r.observed), "n": int(r.n)}
            for r in frame.itertuples()
        ]
        for cls, frame in D.calibration_points(rel, MODEL).items()
    }


def _technical(rel: D.Release) -> dict:
    page = load_page(
        rel,
        D.headline(rel, MODEL),
        D.purity_points(rel, MODEL, PRIOR),
        D.abstention_curve(rel, MODEL),
        MODEL,
    )
    return {
        "data_box": _data_box(page),
        "validation": _validation(page),
        "error_directions": _error_directions(rel),
        "triage": _triage(page),
        "dsc": _dsc(rel),
        "models": _models(rel),
        "confidence_hist": _confidence_hist(rel),
        "calibration": _calibration(rel),
    }


def _headline(rel: D.Release) -> dict:
    """The answer to the central question, from the Comparação model; no network, no answer."""
    model: CM.CompareModel | None = None
    lead = HD.UNAVAILABLE_NOTE
    try:
        results = data_dl.load_results_dl(rel)
        if results is not None:
            model = CM.build(results)
    except data_dl.DlDataError as exc:
        lead = str(exc)
    except (ValueError, OSError):
        lead = UNREADABLE_DL
    base = {"question": web_text.QUESTION, "caveat": None, "pairs": []}
    if model is None or not model.pairs:
        return {
            **base,
            "key": web_text.KEY_NA,
            "title": web_text.TITLES[web_text.KEY_NA],
            "lead": lead,
        }
    wins = [n for n, v in model.pairs if v is data_dl.PairVerdict.WIN]
    key = web_text.KEY_WIN if wins else web_text.KEY_TIE
    return {
        **base,
        "key": key,
        "title": web_text.TITLES[key],
        "lead": web_text.headline_lead(key, [n for n, _ in model.pairs], wins),
        "caveat": None if wins else web_text.headline_caveat(CT.POWER_PHRASE),
        "pairs": [
            {
                "net": net,
                "text": CT.PAIR_WORD[v],
            }
            for net, v in model.pairs
        ],
    }


def _below(a: float | None, b: float | None) -> bool:
    """True when both values exist and ``a`` is under ``b``; the caller words the contrast."""
    return a is not None and b is not None and a < b


def _contrast(rows: list[dict]) -> dict:
    shipped = next(
        r
        for r in rows
        if (r["rule"], r["prior"]) == (ui_state.Rule.SHIPPED.value, ui_state.SHIPPED_PRIOR)
    )
    return {
        "accuracy": shipped["accuracy"],
        "qso_purity": shipped["purity"],
        "text": web_text.contrast_text(
            shipped["accuracy"]["text"],
            shipped["purity"]["text"],
            purity_lower=_below(shipped["purity"]["point"], shipped["accuracy"]["value"]),
        ),
    }


def overview(rel: D.Release, pixels: pd.DataFrame) -> dict:
    """overview.json: the four QSO rows and the technical blocks.

    ``pixels`` is the HEALPix table shared with the sky page; the overview takes its pixel
    count from the headline, so it is not read here.
    """
    del pixels
    rows = _rows(rel)
    return {
        **head(rel.name, "overview"),
        "headline": {
            **_headline(rel),
            "contrast": _contrast(rows),
            "summary_en": web_text.SUMMARY_EN,
        },
        "rows": rows,
        "accuracy_note": overview_text.ACCURACY_NOTE,
        "technical": _technical(rel),
    }


def _curve(lon: list[float], lat: np.ndarray | list[float]) -> list[list[float]]:
    """Polyline in the map units of ``data_sky.mollweide``."""
    x, y = data_sky.mollweide(
        np.asarray(lon, dtype=float), np.asarray(lat, dtype=float)
    )
    return [
        [round(float(a), MAP_DIGITS), round(float(b), MAP_DIGITS)]
        for a, b in zip(x, y, strict=True)
    ]


def _graticule() -> dict:
    """Same lines and labels as ``sky_map_html.graticule``, in ``mollweide`` units."""
    steps = np.linspace(-90.0, 90.0, CURVE_POINTS)
    meridians = {0, EDGE_LON_DEG, -EDGE_LON_DEG}
    meridians |= {s * m for m in sky_map_html.MERIDIANS_DEG for s in (-1, 1)}
    edges = {s * e for e in data_sky.BAND_EDGES_DEG for s in (-1, 1)}
    parallels = sorted(edges | set(sky_map_html.LABELLED_LATS_DEG))
    labels = []
    for lat in sky_map_html.LABELLED_LATS_DEG:
        ((x, y),) = _curve([EDGE_LON_DEG], [lat])
        labels.append(
            {"x": x, "y": y, "text": sky_map_html._lat_text(lat), "kind": "lat"}
        )
    for lon, _ in sky_map_html.lon_label_positions():
        ((x, y),) = _curve([lon], [0.0])
        labels.append(
            {"x": x, "y": y, "text": f"{fmt.pt_int(lon)}°", "kind": "lon"}
        )
    return {
        "meridians": [_curve([lon] * CURVE_POINTS, steps) for lon in sorted(meridians)],
        "parallels": [
            _curve([EDGE_LON_DEG, -EDGE_LON_DEG], [lat, lat]) for lat in parallels
        ],
        "labels": labels,
    }


def _regions(stats: pd.DataFrame, floor: int) -> list[dict]:
    rate_layer = sky_model.map_payload(stats, floor, show_count=False)
    count_layer = sky_model.map_payload(stats, floor, show_count=True)
    rows = zip(
        rate_layer["healpix_l5"],
        rate_layer["x"],
        rate_layer["y"],
        stats["band"],
        count_layer["n"],
        rate_layer["below_floor"],
        stats["errors"],
        rate_layer["rate"],
        strict=True,
    )
    # Owner decision SEC-02 (2026-11-26): below the floor neither n nor hp leave the exporter,
    # because n = 1 of a known pixel says that a public object is in the test fold.
    return [
        {
            "hp": None if below else int(hp),
            "x": round(float(x), MAP_DIGITS),
            "y": round(float(y), MAP_DIGITS),
            "band": str(band),
            "n": None if below else int(n),
            "below_floor": bool(below),
            "errors": None if below else int(errors),
            "rate": None if below else float(rate),
        }
        for hp, x, y, band, n, below, errors, rate in rows
    ]


def _bands(stats: pd.DataFrame) -> list[dict]:
    names = sky_text.band_names()
    return [
        {
            "band": r.band,
            "label": names[r.band],
            "n_pixels": int(r.n_pixels),
            "n": int(r.n),
            "errors": int(r.errors),
            "rate": est(r.rate, r.ci_lo, r.ci_hi, digits=2),
        }
        for r in data_sky.lat_band_table(stats).itertuples(index=False)
    ]


def _top_regions(stats: pd.DataFrame, floor: int) -> list[dict]:
    out = []
    for r in data_sky.top_error_regions(stats, floor).itertuples(index=False):
        lo, hi = wilson(int(r.errors), int(r.n))
        out.append(
            {
                "hp": int(r.healpix_l5),
                "l": float(r.l),
                "b": float(r.b),
                "position_text": sky_text.position_text(r.l, r.b),
                "n": int(r.n),
                "errors": int(r.errors),
                "rate": est(r.rate, lo, hi),
            }
        )
    return out


def sky(rel: D.Release, pixels: pd.DataFrame) -> dict:
    """sky.json: cards, texts, colour caps, graticule and one row per test region."""
    stats = data_sky.test_pixel_stats(rel, MODEL, pixels)
    floor = int(rel.meta["floor"]["min_objects"])
    sc = sky_model.cards(stats, rel.meta)
    return {
        **head(rel.name, "sky"),
        "cards": {
            "n_regions": num(sc.n_regions),
            "n_objects": num(sc.n_objects),
            "n_below": num(sc.n_below),
            "max_rate": from_value(sc.max_rate),
            "diverges": sc.diverges,
            "diverge_note": (
                sky_text.diverge_note(sc.meta_regions, sc.meta_objects)
                if sc.diverges
                else None
            ),
        },
        "texts": {
            "title": sky_text.TITLE,
            "lead": sky_text.lead(sc.n_regions),
            "how_rate": web_text.how(True, sc.n_below, floor, False),
            "how_rate_hidden": web_text.how(True, sc.n_below, floor, True),
            "how_count": web_text.how(False, sc.n_below, floor, False),
            "how_count_hidden": web_text.how(False, sc.n_below, floor, True),
            "map_lead": sky_text.map_lead(stats, floor),
            "position_note": sky_text.POSITION_NOTE,
            "neighbour_note": method_notes.NEIGHBOUR_NOTE,
            "spread_note": sky_text.spread_note(stats),
        },
        "color_cap": {
            "rate": (
                sky_map_html.color_cap(stats, True, floor) if sc.n_reliable else None
            ),
            "count": sky_map_html.color_cap(stats, False, floor),
        },
        "graticule": _graticule(),
        "regions": _regions(stats, floor),
        "bands": _bands(stats),
        "top_regions": _top_regions(stats, floor),
    }


def _prior_blocks(block: data_galaxy_prior.GalaxyPriorBlock) -> list[dict]:
    def interval(iv: data_galaxy_prior.Interval | None) -> dict:
        if iv is None:
            return est(None, None, None)
        return est(iv.point, iv.lo, iv.hi)

    return [
        {
            "prior": p.prior,
            "prior_source": p.prior_source,
            "n_rows": num(p.n_rows),
            "purity": interval(p.purity),
            "completeness": interval(p.completeness),
            "partial": p.partial,
        }
        for p in block.priors
    ]


def _bars(
    shipped: data_galaxy_prior.GalaxyPriorBlock,
    matched: data_galaxy_prior.GalaxyPriorBlock,
) -> dict:
    purity, completeness = data_galaxy_prior.galaxy_rows(
        shipped, matched, ui_state.DEFAULT
    )

    def bars(rows: tuple) -> list[dict]:
        return [
            {
                "rule": rule.value,
                "prior": prior,
                "label": bar.label,
                "est": from_value(bar.value),
                "partial": prior in data_headline.PARTIAL_PRIORS,
            }
            for (rule, prior), bar in zip(overview_model.ROW_ORDER, rows, strict=True)
        ]

    return {"purity": bars(purity), "completeness": bars(completeness)}


def _dl_cells(
    rel: D.Release,
) -> tuple[dict[str, data_dl.ClassCell | None], str | None]:
    """GALAXY cell of each network, or a data-error text (no numbers) when unreadable."""
    none = dict.fromkeys(data_galaxy_prior.NETWORKS)
    try:
        results = data_dl.load_results_dl(rel)
        cells = none if results is None else data_galaxy_prior.dl_galaxy_cells(results)
        for cell in cells.values():
            if cell is not None:
                CM.grid_cell("", cell)
        return cells, None
    except (data_dl.DlDataError, ValueError, OSError) as exc:
        text = str(exc) if isinstance(exc, data_dl.DlDataError) else UNREADABLE_DL
        return none, text


def _networks(cells: dict[str, data_dl.ClassCell | None]) -> dict:
    out: dict[str, dict | None] = {}
    for net in data_galaxy_prior.NETWORKS:
        cell = cells.get(net)
        if cell is None:
            out[net] = None
            continue
        grid = CM.grid_cell(net, cell)
        out[net] = {
            "label": HD.system_label(net),
            "state": grid.state.value,
            "word": grid.verdict.word,
            "diff": est(cell.d, cell.lo, cell.hi, kind="num", digits=4, signed=True),
            "d": cell.d,
            "lo": cell.lo,
            "hi": cell.hi,
            "delta": cell.delta,
            "underpowered": grid.underpowered,
        }
    return out


def _power_note(underpowered: bool) -> str | None:
    """The tie is a lack of power, not equivalence: the Streamlit phrase, as plain text."""
    return CT.POWER_PHRASE if underpowered else None


def _error_rate(r: data_galaxy_prior.ErrorRate | None) -> dict:
    if r is None:
        return est(None, None, None)
    return est(r.rate, min(r.ci_lo, r.rate), max(r.ci_hi, r.rate))


def _error_rows(rel: D.Release, by_g: pd.DataFrame, by_z: pd.DataFrame) -> list[dict]:
    rows = data_galaxy_prior.galaxy_error_rows(
        D.galaxy_totals(rel, MODEL),
        by_g,
        by_z,
        int(rel.meta["floor"]["min_objects"]),
    )
    return [
        {
            "label": label,
            "n": num(None if r is None else r.n),
            "errors": num(None if r is None else r.errors),
            "rate": _error_rate(r),
        }
        for label, r in rows
    ]


def _bands_table(table: pd.DataFrame, floor: int) -> list[dict]:
    """Bands with a defined position; the "sem medida" row is left out, as in the dashboard."""
    out = []
    for r in table[table["lo"].notna()].itertuples(index=False):
        reportable = int(r.n) > 0 and int(r.n) >= floor
        digits = 2 if reportable and r.rate < SMALL_RATE else 1
        rate = (
            est(r.rate, r.ci_lo, r.ci_hi, digits=digits)
            if reportable
            else est(None, None, None)
        )
        out.append(
            {
                "label": str(r.label),
                "n": num(r.n),
                "errors": num(r.errors),
                "rate": rate,
                "reportable": reportable,
            }
        )
    return out


def galaxies(rel: D.Release) -> dict:
    """galaxies.json: GALAXY purity and completeness by rule and prior, verdict, networks."""
    shipped = data_galaxy_prior.load_galaxy_prior_block(rel, "as_shipped")
    matched = data_galaxy_prior.load_galaxy_prior_block(rel, "prior_matched")
    cells, dl_error = _dl_cells(rel)
    key = data_galaxy_prior.galaxy_verdict(matched, cells)
    unavailable = key is data_galaxy_prior.VerdictKey.UNAVAILABLE
    by_g = D.error_rate_table(rel, MODEL, "GALAXY", "G", D.GALAXY_G_EDGES)
    by_z = D.error_rate_table_z(rel, MODEL)
    floor = int(rel.meta["floor"]["min_objects"])
    note = data_galaxy_prior.crossmatch_note(rel)
    scope = f"{VERDICT_SCOPE} {ui_state.RULE_LABEL[ui_state.Rule.MATCHED]}."
    networks = _networks(cells)
    return {
        **head(rel.name, "galaxies"),
        "blocks": {
            "as_shipped": _prior_blocks(shipped),
            "prior_matched": _prior_blocks(matched),
        },
        "bars": _bars(shipped, matched),
        "verdict": {
            "key": key.value,
            "text": verdict_text(key),
            "scope_text": None if unavailable else scope,
        },
        "networks": networks,
        "power_note": _power_note(
            any(n is not None and n["underpowered"] for n in networks.values())
        ),
        "networks_note": (
            HD.UNAVAILABLE_NOTE
            if dl_error is None and all(n is None for n in networks.values())
            else None
        ),
        "dl_error": dl_error,
        "fixed_point_text": CT.fixed_point_line(
            LINE_LABEL, VR.RULE_PRIOR, str(VR.RULE_COMPLETENESS)
        ),
        "error_rows": _error_rows(rel, by_g, by_z),
        "error_rows_note": web_text.ERROR_ROWS_NOTE,
        "floor_note": (
            f"Faixas com menos de {fmt.pt_int(floor)} objetos não têm taxa "
            "confiável e aparecem como n/a."
        ),
        "crossmatch_note": translate(note)[0] if note else None,
        "technical": {
            "by_g": _bands_table(by_g, floor),
            "by_z": _bands_table(by_z, floor),
            "missing_note": web_text.missing_note(by_g),
        },
    }


def _limit_text(original: str) -> str:
    shown, translated = translate(original)
    return shown if translated else omit_record_refs(shown) + HM.UNTRANSLATED


def _compare_blocks(rel: D.Release, results: dict) -> dict:
    model = CM.build(results)
    hl = data_dl.headline(results)
    limitations = data_dl.load_limitations(rel)
    freeze = data_dl.freeze_exists(rel)
    status = data_dl.adr_status()
    below = sum(g.state is data_dl.ClassVerdict.NA for g in model.grid)
    return {
        "fixed_point": {
            "prior": model.prior,
            "level": model.level,
            "text": CT.fixed_point_line(LINE_LABEL, model.prior, model.level),
        },
        "answer_text": CT.answer(model),
        "reasons_text": HD.reasons_text(status, freeze, below, limitations is not None),
        "protocol_accepted": None if status is None else status == ACCEPTED_STATUS,
        "freeze_exists": freeze,
        "pairs": [
            {"net": net, "verdict": v.value, "text": CT.PAIR_WORD[v]}
            for net, v in model.pairs
        ],
        "grid": [
            {
                "cls": g.cls,
                "cls_label": HD.class_label(g.cls),
                "net": g.net,
                "state": g.state.value,
                "word": g.verdict.word,
                "d": g.verdict.d,
                "lo": None if g.verdict.ci is None else g.verdict.ci.lo,
                "hi": None if g.verdict.ci is None else g.verdict.ci.hi,
                "diff": est(
                    g.verdict.d,
                    None if g.verdict.ci is None else g.verdict.ci.lo,
                    None if g.verdict.ci is None else g.verdict.ci.hi,
                    kind="num",
                    digits=4,
                    signed=True,
                ),
                "delta": g.verdict.delta,
                "underpowered": g.underpowered,
            }
            for g in model.grid
        ],
        "power_note": _power_note(any(g.underpowered for g in model.grid)),
        "purity": [
            {
                "system": system,
                "reference": system == CM.REFERENCE_SYSTEM,
                "net": None
                if system == CM.REFERENCE_SYSTEM
                else CM.net_name(system.removeprefix(ENSEMBLE_PREFIX)),
                "label": CT.REFERENCE_NAME
                if system == CM.REFERENCE_SYSTEM
                else HD.system_label(system),
                "low": low,
                "median": median,
                "high": high,
                "detail_text": CT.purity_detail(median, low, high),
            }
            for system, low, median, high in model.purity
        ],
        "rule_lines": list(VR.rule_lines()),
        "descriptive": [
            {
                "pair": HD.pair_label(cell.pair),
                "role": HD.role_label(cell.role),
                "cls": HD.class_label(cls),
                "d": c.d,
                "lo": c.lo,
                "hi": c.hi,
                "diff": est(c.d, c.lo, c.hi, kind="num", digits=4, signed=True),
                "text": (
                    f"D {HD.signed(c.d)}; {fmt.pt_ci_label()} {HD.interval(c.lo, c.hi)}; "
                    f"δ {HD.plain(c.delta)}"
                ),
            }
            for cell in data_dl.descriptive(results, hl)
            for cls, c in cell.classes.items()
        ],
        "limitations": (
            None if limitations is None else [_limit_text(t) for t in limitations]
        ),
    }


def compare(rel: D.Release) -> dict:
    """compare.json: status ok, unavailable or error; every other block is null unless ok."""
    out = {**head(rel.name, "compare"), "status": "ok", "error_text": None}
    blocks: dict | None = None
    try:
        results = data_dl.load_results_dl(rel)
        if results is None:
            out["status"] = "unavailable"
        else:
            blocks = _compare_blocks(rel, results)
    except data_dl.DlDataError as exc:
        out.update(status="error", error_text=str(exc))
    except (ValueError, OSError):
        out.update(status="error", error_text=UNREADABLE_DL)
    return {**out, **(blocks or dict.fromkeys(COMPARE_BLOCKS))}


def method(rel: D.Release) -> dict:
    """method.json: formatted facts, the fixed texts, the limitations and the https links."""
    facts = method_model.method_facts(rel)
    raw = rel.meta
    mix = method_model.class_mix(rel)
    limitations = list(rel.results["limitations"])
    return {
        **head(rel.name, "method"),
        "facts": {
            "n_test": num(raw["n_test"], facts.n_test),
            "n_pixels": num(raw["n_test_pixels"], facts.n_pixels),
            "prior_qso": num(raw["pi_src_combmod"]["QSO"], facts.prior_qso),
            "min_objects": num(raw["floor"]["min_objects"], facts.min_objects),
            "min_pixels": num(raw["floor"]["min_pixels"], facts.min_pixels),
            "n_boot": num(raw["n_boot"], facts.n_boot),
            "completeness": num(VR.RULE_COMPLETENESS, facts.completeness),
        },
        "mix": {c: num(mix.get(c, 0.0), kind="pct", digits=0) for c in CLASSES},
        "mix_text": HM.mix_sentence(mix),
        "method_text": HM.METHOD_TEXT,
        "lead": HM.LEAD,
        "simple": HM.HINT,
        "reading": method_model.reading_paragraphs(facts),
        "usage": method_model.usage_paragraphs(),
        "rule_lines": list(VR.rule_lines()),
        "three_limits": [
            HM.crossmatch_limit(limitations),
            method_notes.NEIGHBOUR_NOTE,
            method_notes.PRIOR_NOTE,
        ],
        "limitations": [
            {"text": _limit_text(t), "original": omit_record_refs(t)}
            for t in limitations
        ],
        "glossary": [
            {"term": term, "definition": (text or "").strip() or None}
            for term, text in HM.GLOSSARY
        ],
        "links": [
            {"label": label, "note": note, "url": url}
            for label, note, url in HM._references()
            if url.startswith(HM.HTTPS_PREFIX)
        ],
        "sources": list(HM.SOURCES),
    }


def fmt_samples() -> list[dict]:
    """Reference outputs of every ``fmt.pt_*`` the client re-implements for axis ticks."""
    return [
        {"fn": name, "args": list(args), "text": getattr(fmt, name)(*args)}
        for name, cases in FMT_SAMPLES
        for args in cases
    ]
