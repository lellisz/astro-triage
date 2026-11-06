"""Portuguese text for the evaluation's ``limitations``, matched by pattern.

Each key is the English sentence with every number replaced by ``{}``; the value is the
translation with ``{}`` in the same order. Numbers are taken from the original text and
re-formatted pt-BR, so none is written in this module.
"""

from __future__ import annotations

import logging
import re

_LOG = logging.getLogger(__name__)
_LOG_CHARS = 80
_SWAP = str.maketrans(",.", ".,")
_NUMBER = re.compile(r"(?<![\w.,])\d+(?:,\d{3})*(?:\.\d+)?k?%?(?!\w)")
_THOUSANDS = re.compile(r"\d+(?:,\d{3})+")

LIMITS_PT: dict[str, str] = {
    "No DR14Q crossmatch.": "Sem crossmatch com o DR14Q.",
    "REALISTIC tables are class-prior corrected; within-class distribution = SDSS spectroscopic targets (selection bias not corrected).": "As tabelas REALISTIC são corrigidas pelo prior de classe; distribuição dentro da classe = alvos espectroscópicos do SDSS (viés de seleção não corrigido).",
    "LightGBM scores are the temperature-calibrated probabilities (p_cal_*) fitted on val; test was not used for any fitting.": "Os escores do LightGBM são as probabilidades calibradas por temperatura (p_cal_*) ajustadas no val; o teste não foi usado em nenhum ajuste.",
    "DSC was trained partly on SDSS objects (DR14Q quasars, DR15 galaxies): partial circularity. ADR {} cut (b) (outside DSC training): not available: DR14Q/DR15 not ingested.": "O DSC foi treinado em parte com objetos do SDSS (quasares do DR14Q, galáxias do DR15): circularidade parcial. Recorte (b) da ADR {} (fora do treino do DSC): indisponível: DR14Q/DR15 não foram ingeridos.",
    "Missing astrometry (astrometric_params_solved = {}) is a pattern linked to the class, so the missingness itself can carry class information (see M-noind).": "A astrometria ausente (astrometric_params_solved = {}) é um padrão ligado à classe, então a própria ausência pode carregar informação de classe (ver M-noind).",
    "Prior shift of combmod uses pi_src = P2 (ADR {}). allosmod has no pi_src defined in the ADRs: it is never shifted, so its as_shipped and prior_matched rows are identical.": "A mudança de prior do combmod usa pi_src = P2 (ADR {}). O allosmod não tem pi_src definido nas ADRs: ele nunca é deslocado, então suas linhas como entregue e ajustado ao prior são idênticas.",
    "Sensitivity sweep S has point estimates only (no bootstrap), per ADR {}.": "A varredura de sensibilidade S tem apenas estimativas pontuais (sem bootstrap), conforme a ADR {}.",
    "Reporting floor (>= {} objects in >= {} test pixels per class) is not met by some cells; those are marked n/a (floor) and any raw value printed is below-floor, not for claims.": "O piso de relato (>= {} objetos em >= {} regiões de teste por classe) não é atingido por algumas células; elas são marcadas como n/a (abaixo do piso) e qualquer valor bruto exibido está abaixo do piso, não serve para afirmações.",
    "Only Gaia sources with an SDSS spectroscopic match are labelled (full_v0 release, {} rows); r_match {} arcsec on SDSS plug positions.": "Apenas fontes do Gaia com correspondência espectroscópica no SDSS recebem rótulo (versão dos dados full_v0, {} linhas); r_match de {} arcsec sobre as posições das fibras do SDSS.",
    "Sample of {} Gaia sources only; r_match {} arcsec on SDSS plug positions.": "Apenas uma amostra de {} fontes do Gaia; r_match de {} arcsec sobre as posições das fibras do SDSS.",
    "Selection by crossmatch: the matched class mix differs from the SDSS mix (match rate STAR {}, QSO {}, GALAXY {}, measured on the {} pilot batch only and not recomputed for full_v0); metrics describe the matched intersection, not the Gaia sky.": "Seleção por crossmatch: a mistura de classes dos objetos cruzados difere da mistura do SDSS (taxa de cruzamento estrela {}, quasar {}, galáxia {}, medida apenas no lote piloto de {} e não recalculada para o full_v0); as métricas descrevem a interseção dos catálogos, não o céu do Gaia.",
    "Selection by crossmatch: the matched class mix differs from the SDSS mix (match rate STAR {}, QSO {}, GALAXY {} in the {} batch); metrics describe the matched intersection, not the Gaia sky.": "Seleção por crossmatch: a mistura de classes dos objetos cruzados difere da mistura do SDSS (taxa de cruzamento estrela {}, quasar {}, galáxia {} no lote de {}); as métricas descrevem a interseção dos catálogos, não o céu do Gaia.",
    "combmod has {} of its {} classes available; its star score is {} - p_quasar - p_galaxy (absorbs WD and binary), ADR {} 'Effect of the combmod embedded prior' item {}. The delivered classprob_dsc_combmod_star column is not used.": "O combmod tem {} das suas {} classes disponíveis; seu escore de estrela é {} - p_quasar - p_galaxy (absorve WD e binária), ADR {} 'Efeito do prior embutido do combmod' item {}. A coluna entregue classprob_dsc_combmod_star não é usada.",
}


def _pt_number(token: str) -> str:
    if _THOUSANDS.fullmatch(token.rstrip("%")):
        return token.replace(",", ".")
    if "," in token and "." in token:
        return token.translate(_SWAP)
    return token.replace(".", ",")


def translate(text: str) -> tuple[str, bool]:
    """Portuguese text and whether a translation exists; the original comes back untranslated."""
    numbers = _NUMBER.findall(text)
    pt = LIMITS_PT.get(_NUMBER.sub("{}", text))
    if pt is None:
        return text, False
    if pt.count("{}") != len(numbers):
        _LOG.warning(
            "limitations: %d numbers do not fit the %d slots of the model: %.*s",
            len(numbers),
            pt.count("{}"),
            _LOG_CHARS,
            text,
        )
        return text, False
    parts = pt.split("{}")
    shown = parts[0]
    for token, tail in zip(numbers, parts[1:], strict=True):
        shown += _pt_number(token) + tail
    return shown, True


def limitation_lines(originals: list[str]) -> list[str]:
    """Markdown bullets for the evaluation's limitations; untranslated ones keep the original."""
    lines = []
    for text in originals:
        shown, translated = translate(text)
        lines.append(f"- {shown}" if translated else f"- {shown} (sem tradução)")
    return lines
