"""Pure model of the Comparação page: verdicts by the shared rule, the 2x2 grid and the purity."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Final

from astro_triage.dashboard import compare_text as T
from astro_triage.dashboard import data_dl as DD
from astro_triage.dashboard import verdict_rule as VR
from astro_triage.dashboard.data_dl import ClassVerdict, DlDataError, PairVerdict
from astro_triage.dashboard.kit_types import Interval, KitInputError, Verdict
from astro_triage.evaluation import dl_compare

REFERENCE_SYSTEM: Final = "lgbm_full"
_LEVEL_TOLERANCE: Final = 1e-9
_NET_NAMES: Final = {"mlp": "MLP", "ftt": "FTT"}


@dataclass(frozen=True)
class GridCell:
    net: str
    cls: str
    state: ClassVerdict
    verdict: Verdict
    underpowered: bool


@dataclass(frozen=True)
class CompareModel:
    pairs: tuple[tuple[str, PairVerdict], ...]
    grid: tuple[GridCell, ...]
    purity: tuple[tuple[str, float, float, float], ...]
    prior: str
    level: str


def net_name(pair: str) -> str:
    family = pair.partition("-")[0].partition("_")[0]
    return _NET_NAMES.get(family, family.upper())


def _verdict(c: DD.ClassCell, state: ClassVerdict) -> Verdict:
    word = T.STATE_WORD[state]
    if state is ClassVerdict.NA:
        return Verdict(word, None, None, c.delta)
    if c.lo is None or c.hi is None:
        raise DlDataError(f"intervalo ausente na classe {c.cls}")
    try:
        return Verdict(word, c.d, Interval(c.lo, c.hi), c.delta)
    except KitInputError as exc:
        raise DlDataError(f"intervalo inválido na classe {c.cls}: {exc}") from exc


def _check_point(head: DD.Headline) -> None:
    try:
        level = float(head.level)
    except ValueError as exc:
        raise DlDataError(f"completude ilegível: {head.level!r}") from exc
    if (
        head.prior != VR.RULE_PRIOR
        or abs(level - VR.RULE_COMPLETENESS) > _LEVEL_TOLERANCE
    ):
        raise DlDataError("o destaque do artefato não está no ponto fixo da regra")


def _purity(
    cells: tuple[DD.PairCell, ...],
) -> tuple[tuple[str, float, float, float], ...]:
    seen: dict[str, tuple[float, float, float]] = {}
    for cell in cells:
        qso = cell.classes.get("QSO")
        for system, stats in (qso.purity if qso else {}).items():
            seen.setdefault(system, stats)
    names = sorted(seen, key=lambda s: s != REFERENCE_SYSTEM)
    return tuple((s, *seen[s]) for s in names)


def build(results: Mapping) -> CompareModel:
    """Raises ``DlDataError`` for a malformed artifact or a verdict that disagrees with the rule."""
    head = DD.headline(results)
    _check_point(head)
    order = [f"{a}-{b}" for a, b in dl_compare.HEADLINE_PAIRS]
    unknown = [c.pair for c in head.cells if c.pair not in order]
    if unknown:
        raise DlDataError(f"par fora do protocolo no destaque: {unknown[0]}")
    cells = tuple(sorted(head.cells, key=lambda c: order.index(c.pair)))
    pairs: list[tuple[str, PairVerdict]] = []
    grid: dict[tuple[str, str], GridCell] = {}
    for cell in cells:
        VR.check(cell)
        net = net_name(cell.pair)
        pairs.append((net, VR.pair_state(cell.classes)))
        for cls, c in cell.classes.items():
            state = VR.class_state(c)
            grid[(cls, net)] = GridCell(
                net, cls, state, _verdict(c, state), VR.underpowered(c)
            )
    ordered = tuple(
        grid[(cls, net)]
        for cls in dl_compare.TARGET_CLASSES
        for net, _ in pairs
        if (cls, net) in grid
    )
    return CompareModel(tuple(pairs), ordered, _purity(cells), head.prior, head.level)
