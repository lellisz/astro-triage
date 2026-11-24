"""Verdict words and the answer sentence of the Comparação page: the only place they are written."""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Final

from astro_triage.dashboard import fmt
from astro_triage.dashboard.data_dl import ClassVerdict, PairVerdict
from astro_triage.dashboard.html_system import NA
from astro_triage.evaluation import dl_compare

if TYPE_CHECKING:
    from astro_triage.dashboard.compare_model import CompareModel

STATE_WORD: Final[Mapping[ClassVerdict, str]] = {
    ClassVerdict.WIN: "vitória",
    ClassVerdict.TIE: "empate",
    ClassVerdict.WORSE: "pior (descritivo)",
    ClassVerdict.NA: NA,
}
PAIR_WORD: Final[Mapping[PairVerdict, str]] = {
    PairVerdict.WIN: "vence o par, com vitória nas duas classes",
    PairVerdict.TIE: "empate, fica o LightGBM",
}
POWER_PHRASE: Final = "falta de poder, não igualdade"
NOT_CI_LABEL: Final = "não é IC"
REFERENCE_NAME: Final = "LightGBM"
FIXED_NOTE: Final = (
    "ponto fixo desta aba; a regra e o prior da barra de ajustes não o mudam"
)


def fixed_point_line(label: str, prior: str, level: str) -> str:
    """The "Regra · prior" line of this page: the protocol point, whatever the sidebar says."""
    return f"{label}: {prior} · C* = {fmt.pt_num(float(level), 2)} ({FIXED_NOTE})"


def purity_detail(mid: float, lo: float, hi: float) -> str:
    """Median of the seeds with the min-max range; the range is not a confidence interval."""
    return (
        f"mediana de {fmt.pt_int(dl_compare.N_SEEDS)} sementes {fmt.pt_pct(mid, 2)}; "
        f"mínimo a máximo {fmt.pt_pct(lo, 2)} a {fmt.pt_pct(hi, 2)} ({NOT_CI_LABEL})"
    )


def answer(model: CompareModel) -> str:
    """One sentence with the verdict of each pair, from templates."""
    parts = [
        f"{net} contra {REFERENCE_NAME}: {PAIR_WORD[verdict]}"
        for net, verdict in model.pairs
    ]
    return "; ".join(parts) + "."
