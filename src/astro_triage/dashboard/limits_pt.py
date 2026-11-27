"""Portuguese text for the evaluation's ``limitations``, matched by pattern.

Each key is the English sentence with every number replaced by ``{}``; the value is the
translation with ``{}`` in the same order. Numbers are taken from the original text and
re-formatted pt-BR, so none is written in this module. ``{-}`` consumes a number and prints
nothing (record numbers the page does not show).
"""

from __future__ import annotations

import logging
import re
from typing import Final

_LOG = logging.getLogger(__name__)
_LOG_CHARS = 80
_SWAP = str.maketrans(",.", ".,")
_NUMBER = re.compile(r"(?<![\w.,])\d+(?:,\d{3})*(?:\.\d+)?k?%?(?!\w)")
_THOUSANDS = re.compile(r"\d+(?:,\d{3})+")
_SLOT = re.compile(r"\{-?\}")
DROP: Final[str] = "{-}"
_REF_PAREN = re.compile(r"\s*\((?:per\s+)?ADR\s+\d{4}[^)]*\)")
_REF_PER = re.compile(r",?\s*per\s+ADR\s+\d{4}")
_REF_IN_THE = re.compile(r"\s+in\s+the\s+ADRs\b")
_REF_TITLED = re.compile(r",?\s*ADR\s+\d{4}\s+'[^']*'\s+item\s+\d+")
_REF_SENTENCE_START = re.compile(r"(\. )ADR\s+\d{4}\s+(\w)")
_REF_ANY = re.compile(r"\bADRs?\b(?:\s+\d{4})?\s*")
_DOUBLE_SPACE = re.compile(r" {2,}")
_SPACE_BEFORE_PUNCT = re.compile(r" ([.,:])")

LIMITS_PT: dict[str, str] = {
    "No DR14Q crossmatch.": "Sem crossmatch com o DR14Q.",
    "REALISTIC tables are class-prior corrected; within-class distribution = SDSS spectroscopic targets (selection bias not corrected).": "As tabelas REALISTIC são corrigidas pelo prior de classe; distribuição dentro da classe = alvos espectroscópicos do SDSS (viés de seleção não corrigido).",
    "LightGBM scores are the temperature-calibrated probabilities (p_cal_*) fitted on val; test was not used for any fitting.": "Os escores do LightGBM são as probabilidades calibradas por temperatura (p_cal_*) ajustadas no val; o teste não foi usado em nenhum ajuste.",
    "DSC was trained partly on SDSS objects (DR14Q quasars, DR15 galaxies): partial circularity. ADR {} cut (b) (outside DSC training): not available: DR14Q/DR15 not ingested.": "O DSC foi treinado em parte com objetos do SDSS (quasares do DR14Q, galáxias do DR15): circularidade parcial. O recorte fora do treino do DSC{-} está indisponível: DR14Q/DR15 não foram ingeridos.",
    "Missing astrometry (astrometric_params_solved = {}) is a pattern linked to the class, so the missingness itself can carry class information (see M-noind).": "A astrometria ausente (astrometric_params_solved = {}) é um padrão ligado à classe, então a própria ausência pode carregar informação de classe (ver M-noind).",
    "Prior shift of combmod uses pi_src = P2 (ADR {}). allosmod has no pi_src defined in the ADRs: it is never shifted, so its as_shipped and prior_matched rows are identical.": "A mudança de prior do combmod usa pi_src = P2{-}. O allosmod não tem pi_src definido: ele nunca é deslocado, então suas linhas como entregue e ajustado ao prior são idênticas.",
    "Sensitivity sweep S has point estimates only (no bootstrap), per ADR {}.": "A varredura de sensibilidade S tem apenas estimativas pontuais (sem bootstrap), conforme o protocolo de avaliação{-}.",
    "Reporting floor (>= {} objects in >= {} test pixels per class) is not met by some cells; those are marked n/a (floor) and any raw value printed is below-floor, not for claims.": "O piso de relato (>= {} objetos em >= {} regiões de teste por classe) não é atingido por algumas células; elas são marcadas como n/a (abaixo do piso) e qualquer valor bruto exibido está abaixo do piso, não serve para afirmações.",
    "Only Gaia sources with an SDSS spectroscopic match are labelled (full_v0 release, {} rows); r_match {} arcsec on SDSS plug positions.": "Apenas fontes do Gaia com correspondência espectroscópica no SDSS recebem rótulo (versão dos dados full_v0, {} linhas); r_match de {} arcsec sobre as posições das fibras do SDSS.",
    "Sample of {} Gaia sources only; r_match {} arcsec on SDSS plug positions.": "Apenas uma amostra de {} fontes do Gaia; r_match de {} arcsec sobre as posições das fibras do SDSS.",
    "Selection by crossmatch: the matched class mix differs from the SDSS mix (match rate STAR {}, QSO {}, GALAXY {}, measured on the {} pilot batch only and not recomputed for full_v0); metrics describe the matched intersection, not the Gaia sky.": "Seleção por crossmatch: a mistura de classes dos objetos cruzados difere da mistura do SDSS (taxa de cruzamento estrela {}, quasar {}, galáxia {}, medida apenas no lote piloto de {} e não recalculada para o full_v0); as métricas descrevem a interseção dos catálogos, não o céu do Gaia.",
    "Selection by crossmatch: the matched class mix differs from the SDSS mix (match rate STAR {}, QSO {}, GALAXY {} in the {} batch); metrics describe the matched intersection, not the Gaia sky.": "Seleção por crossmatch: a mistura de classes dos objetos cruzados difere da mistura do SDSS (taxa de cruzamento estrela {}, quasar {}, galáxia {} no lote de {}; essa taxa foi medida só nesse lote, e as outras versões dos dados não têm medida própria); as métricas descrevem a interseção dos catálogos, não o céu do Gaia.",
    "B0 (ADR {} section {}): zero hyperparameter search for both families; the deep models use untuned defaults and LightGBM a hand-set configuration, so unequal effort is not excluded (R2).": "B0{-}{-}: nenhuma busca de hiperparâmetros nas duas famílias; as redes usam os valores padrão sem ajuste e o LightGBM uma configuração escolhida à mão, então não se exclui um esforço desigual (R2).",
    "R17: the val-threshold sampling variance is not in the CI; a system whose threshold lands at a lower test completeness gets higher test purity. The realised test completeness is printed next to D and is not gated.": "R17: a variância amostral do limiar fixado na validação não está no IC; um sistema cujo limiar cai numa completude de teste menor ganha pureza de teste maior. A completude realizada no teste não é exibida nesta página e não entra no veredito.",
    "Selection by crossmatch: the matched class mix differs from the SDSS mix (match rate STAR {}, QSO {}, GALAXY {}, measured on the {} batch only and not recomputed for other releases); metrics describe the matched intersection, not the Gaia sky.": "Seleção por crossmatch: a mistura de classes dos objetos cruzados difere da mistura do SDSS (taxa de cruzamento estrela {}, quasar {}, galáxia {}, medida só no lote de {} e não recalculada para outras versões dos dados); as métricas descrevem a interseção dos catálogos, não o céu do Gaia.",
    "Ensemble rows are descriptive only; a verdict exists for H1 and H2 alone.": "As linhas dos conjuntos (ensembles) são apenas descritivas; só H1 e H2 têm veredito.",
    "combmod has {} of its {} classes available; its star score is {} - p_quasar - p_galaxy (absorbs WD and binary), ADR {} 'Effect of the combmod embedded prior' item {}. The delivered classprob_dsc_combmod_star column is not used.": "O combmod tem {} das suas {} classes disponíveis; seu escore de estrela é {} - p_quasar - p_galaxy (absorve WD e binária), efeito do prior embutido do combmod{-}{-}. A coluna entregue classprob_dsc_combmod_star não é usada.",
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
    slots = _SLOT.findall(pt)
    if len(slots) != len(numbers):
        _LOG.warning(
            "limitations: %d numbers do not fit the %d slots of the model: %.*s",
            len(numbers),
            len(slots),
            _LOG_CHARS,
            text,
        )
        return text, False
    tails = _SLOT.split(pt)
    shown = tails[0]
    for token, slot, tail in zip(numbers, slots, tails[1:], strict=True):
        shown += ("" if slot == DROP else _pt_number(token)) + tail
    return shown, True


def omit_record_refs(text: str) -> str:
    """``text`` without citations of decision records; only the citation is removed."""
    cleaned = _REF_SENTENCE_START.sub(lambda m: m.group(1) + m.group(2).upper(), text)
    for rule in (_REF_TITLED, _REF_PAREN, _REF_PER, _REF_IN_THE, _REF_ANY):
        cleaned = rule.sub("", cleaned)
    if cleaned == text:
        return text
    cleaned = _DOUBLE_SPACE.sub(" ", cleaned)
    return _SPACE_BEFORE_PUNCT.sub(r"\1", cleaned).strip()


def limitation_lines(originals: list[str]) -> list[str]:
    """Markdown bullets for the evaluation's limitations; untranslated ones keep the original."""
    lines = []
    for text in originals:
        shown, translated = translate(text)
        lines.append(f"- {shown}" if translated else f"- {shown} (sem tradução)")
    return lines
