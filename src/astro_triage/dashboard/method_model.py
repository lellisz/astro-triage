"""Pure model of the Método page: formatted facts and paragraphs, no Streamlit."""

from __future__ import annotations

from dataclasses import dataclass

from astro_triage.dashboard import data as D
from astro_triage.dashboard import fmt
from astro_triage.dashboard import method_notes as N
from astro_triage.dashboard import verdict_rule as VR


@dataclass(frozen=True)
class MethodFacts:
    """Figures from the release artifact, already formatted pt-BR."""

    n_test: str
    n_pixels: str
    prior_qso: str
    min_objects: str
    min_pixels: str
    n_boot: str
    completeness: str


def method_facts(rel: D.Release) -> MethodFacts:
    meta = rel.meta
    return MethodFacts(
        n_test=fmt.pt_int(meta["n_test"]),
        n_pixels=fmt.pt_int(meta["n_test_pixels"]),
        prior_qso=fmt.pt_pct(meta["pi_src_combmod"]["QSO"], 3),
        min_objects=fmt.pt_int(meta["floor"]["min_objects"]),
        min_pixels=fmt.pt_int(meta["floor"]["min_pixels"]),
        n_boot=fmt.pt_int(meta["n_boot"]),
        completeness=fmt.pt_num(VR.RULE_COMPLETENESS, 2),
    )


def reading_paragraphs(f: MethodFacts) -> list[str]:
    return [
        N.sample_note(f.n_test, f.n_pixels, f.prior_qso),
        N.PRIOR_NOTE,
        (
            "A pureza só tem sentido junto com a completude. A pureza à completude fixa de "
            f"{f.completeness} aparece na validação e, na comparação com aprendizado profundo, "
            "no teste."
        ),
        N.floor_note(f.min_objects, f.min_pixels),
        (
            "Os intervalos de acerto por faixa usam o método de Wilson. "
            + N.bootstrap_note(f.n_boot)
            + " "
            + N.INTERVAL_NOTE
        ),
        N.sky_scale_note(f.min_objects, True)
        + " "
        + N.MAP_NOTE
        + " "
        + N.NEIGHBOUR_NOTE,
        N.COUNT_UNDER_BAND_NOTE,
    ]


def usage_paragraphs() -> list[str]:
    return [N.SCATTER_OFF_NOTE]


def class_mix(rel: D.Release) -> dict[str, float]:
    return rel.test["true"].value_counts(normalize=True).to_dict()
