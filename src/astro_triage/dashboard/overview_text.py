"""Answer sentence and notes of the overview page, built from templates and the model."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Final

from astro_triage.dashboard import fmt
from astro_triage.dashboard.html_system import NA, between_text
from astro_triage.dashboard.kit_types import Value
from astro_triage.dashboard.overview_model import OverviewModel
from astro_triage.dashboard.ui_state import RULE_LABEL, UiState

ACCURACY_NOTE: Final = "acerto alto não garante pureza alta de quasar"
PARTIAL_TAG: Final = "parcial"
CARD_TITLES: Final[Mapping[str, str]] = {
    "purity": "Pureza de quasar",
    "completeness": "Completude de quasar",
    "accuracy": "Acerto geral",
}
CARD_DEFS: Final[Mapping[str, str]] = {
    "purity": "Entre os objetos que o modelo chama de quasar, a fração que é quasar.",
    "completeness": "Entre os quasares do teste, a fração que o modelo chama de quasar.",
    "accuracy": "Fração dos objetos de teste cuja classe prevista é a classe verdadeira.",
}


def _phrase(name: str, value: Value) -> str:
    if value.point is None:
        return f"{name} {NA}"
    return f"{name} {fmt.pt_pct(value.point)} ({between_text(value.ci)})"


def answer(model: OverviewModel, state: UiState) -> str:
    """One sentence with purity and completeness of the selected row, its rule and prior."""
    row = model.selected
    scope = f"regra {RULE_LABEL[state.rule]}, prior {state.prior}"
    if row.partial:
        scope += f", {PARTIAL_TAG}"
    return (
        f"Na {scope}: "
        f"{_phrase('pureza de quasar', row.purity)}; "
        f"{_phrase('completude de quasar', row.completeness)}."
    )
