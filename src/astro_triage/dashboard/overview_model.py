"""Structure of the overview page: four QSO rows, the selected one and the accuracy card.

Pure model, no Streamlit. Numbers are copied from the loaders; nothing is computed here.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Final

from astro_triage.dashboard import ui_state
from astro_triage.dashboard.html_overview import partial_note
from astro_triage.dashboard.kit_types import Interval, Value
from astro_triage.dashboard.ui_state import PRIORS, SHIPPED_PRIOR, Rule, UiState

ROW_ORDER: Final[tuple[tuple[Rule, str], ...]] = (
    (Rule.SHIPPED, SHIPPED_PRIOR),
    *((Rule.MATCHED, p) for p in PRIORS),
)
_NA: Final = Value(None, None)


@dataclass(frozen=True)
class QsoRow:
    rule: Rule
    prior: str
    purity: Value
    completeness: Value
    partial: bool
    selected: bool


@dataclass(frozen=True)
class OverviewModel:
    rows: tuple[QsoRow, ...]
    selected: QsoRow
    accuracy: Value
    partial_notice: str | None


def _value(cell: Mapping, key: str) -> Value | None:
    point, lo, hi = (cell.get(k) for k in (key, f"{key}_lo", f"{key}_hi"))
    if point is None or lo is None or hi is None:
        return None
    return Value(point, Interval(lo, hi))


def _pair(cell: Mapping | None) -> tuple[Value, Value]:
    """Purity and completeness of one cell; both n/a unless both are usable."""
    if cell:
        purity, completeness = _value(cell, "purity"), _value(cell, "completeness")
        if purity is not None and completeness is not None:
            return purity, completeness
    return _NA, _NA


def _accuracy(defs: Mapping, state: UiState) -> Value:
    point = defs.get("shipped")
    if state.rule is Rule.MATCHED or not isinstance(point, int | float):
        return _NA
    return Value(point, None)


def build(
    shipped: Mapping | None,
    matched: Mapping[str, Mapping],
    accuracy_defs: Mapping,
    state: UiState,
) -> OverviewModel:
    rows = []
    for rule, prior in ROW_ORDER:
        cell = shipped if rule is Rule.SHIPPED else matched.get(prior)
        purity, completeness = _pair(cell)
        rows.append(
            QsoRow(
                rule,
                prior,
                purity,
                completeness,
                ui_state.is_partial(prior),
                (rule, prior) == (state.rule, state.prior),
            )
        )
    chosen = [r for r in rows if r.selected]
    if len(chosen) != 1:
        raise ValueError("o estado não corresponde a nenhuma das quatro linhas")
    selected = chosen[0]
    notice = partial_note(selected.prior) if selected.partial else None
    return OverviewModel(tuple(rows), selected, _accuracy(accuracy_defs, state), notice)
