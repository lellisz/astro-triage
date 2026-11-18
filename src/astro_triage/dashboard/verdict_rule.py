"""The verdict rule of ADR 0006, in one place for the Comparação and Método pages.

The class and pair verdicts are recomputed with the functions that produced the artifact
(``dl_compare``) and checked against the token it stored: a mismatch is an error, never a
silent choice. The rule text is built from the constants of ``dl_compare``. No IO, no number
written by hand.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Final

from astro_triage.dashboard import fmt
from astro_triage.dashboard.data_dl import (
    ClassCell,
    ClassVerdict,
    DlDataError,
    PairCell,
    PairVerdict,
    parse_pair_verdict,
)
from astro_triage.evaluation import dl_compare
from astro_triage.evaluation.runner import EvalError

RULE_PRIOR: Final = dl_compare.PRIMARY_PRIOR
RULE_COMPLETENESS: Final = dl_compare.PRIMARY_COMPLETENESS


def class_state(cell: ClassCell) -> ClassVerdict:
    """NA below the reporting floor; otherwise ``dl_compare.class_verdict`` on the artifact numbers."""
    if cell.verdict is ClassVerdict.NA:
        return ClassVerdict.NA
    if cell.verdict is None:
        raise DlDataError(f"a classe {cell.cls} não tem veredito (célula descritiva)")
    if cell.d is None or cell.lo is None or cell.hi is None or cell.delta is None:
        raise DlDataError(f"valor ausente na classe {cell.cls}")
    token = dl_compare.class_verdict(cell.d, cell.lo, cell.hi, cell.delta, True)
    return ClassVerdict(token)


def pair_state(classes: Mapping[str, ClassCell]) -> PairVerdict:
    """A pair wins only with a win in every target class; NA, tie and worse are not wins."""
    tokens = {cls: class_state(c).value for cls, c in classes.items()}
    try:
        token = dl_compare.pair_verdict(tokens)
    except EvalError as exc:
        raise DlDataError(str(exc)) from exc
    return parse_pair_verdict(token, "veredito recalculado")


def underpowered(cell: ClassCell) -> bool:
    """A tie whose interval does not fit inside [-delta, +delta]; below the floor is not a power issue."""
    if class_state(cell) is not ClassVerdict.TIE:
        return False
    assert cell.lo is not None and cell.hi is not None and cell.delta is not None
    return not (-cell.delta <= cell.lo and cell.hi <= cell.delta)


def check(cell: PairCell) -> None:
    """Raise ``DlDataError`` when the recomputed verdicts differ from the artifact tokens."""
    for cls, c in cell.classes.items():
        if c.verdict is not None and class_state(c) is not c.verdict:
            raise DlDataError(
                f"o veredito da classe {cls} em {cell.pair} difere do recalculado"
            )
    if cell.verdict is not None and pair_state(cell.classes) is not cell.verdict:
        raise DlDataError(f"o veredito do par {cell.pair} difere do recalculado")


def rule_lines() -> tuple[str, ...]:
    return (
        (
            "Vitória na classe: célula relatável, D maior que δ e limite inferior do "
            "intervalo maior que zero. Pior na classe: D menor que −δ e limite superior "
            "menor que zero; é descritivo e não decide o par."
        ),
        (
            "O par vence só com vitória nas duas classes. Célula abaixo do piso de relato "
            "conta como empate. Em empate, fica o LightGBM."
        ),
        "Sem correção de multiplicidade entre os dois pares.",
        (
            f"δ = {fmt.pt_g(dl_compare.DELTA_SD_MULTIPLIER)} desvios-padrão "
            f"(ddof = {dl_compare.DELTA_DDOF}) das {fmt.pt_int(dl_compare.N_SEEDS)} "
            "sementes do LightGBM, medido na validação."
        ),
        (
            f"{fmt.pt_ci_label()} por bootstrap de blocos do céu (bloco = pixel nível "
            f"{dl_compare.BLOCK_LEVEL} da divisão por {fmt.pt_int(dl_compare.BLOCK_DIVISOR)}, "
            f"ao menos {fmt.pt_int(dl_compare.MIN_BLOCKS)} blocos)."
        ),
        (
            "Métrica: pureza à completude fixa C* = "
            f"{fmt.pt_num(RULE_COMPLETENESS, 2)}, prior {RULE_PRIOR}; sensibilidade em "
            f"{', '.join(dl_compare.SENSITIVITY_PRIORS)}."
        ),
    )


def rule_text() -> str:
    return " ".join(rule_lines())
