"""Typed inputs of the kit components (no Streamlit, no HTML)."""

from __future__ import annotations

import math
from dataclasses import dataclass


class KitInputError(ValueError):
    """An input the kit refuses to draw (missing interval, non-finite number)."""


def _finite(name: str, x: float | None) -> None:
    if x is not None and not math.isfinite(x):
        raise KitInputError(f"{name} deve ser finito")


@dataclass(frozen=True)
class Interval:
    lo: float
    hi: float

    def __post_init__(self) -> None:
        _finite("lo", self.lo)
        _finite("hi", self.hi)
        if self.lo > self.hi:
            raise KitInputError("o intervalo exige lo <= hi")


@dataclass(frozen=True)
class Value:
    """A point with its interval; ``point=None`` means n/a and carries no interval."""

    point: float | None
    ci: Interval | None

    def __post_init__(self) -> None:
        _finite("point", self.point)
        if self.point is None and self.ci is not None:
            raise KitInputError("valor n/a não pode trazer intervalo")


@dataclass(frozen=True)
class BarRow:
    label: str
    value: Value
    selected: bool = False
    note: str | None = None


@dataclass(frozen=True)
class Verdict:
    """``word`` comes from the caller; this module never writes verdict text."""

    word: str
    d: float | None
    ci: Interval | None
    delta: float | None

    def __post_init__(self) -> None:
        _finite("d", self.d)
        _finite("delta", self.delta)
        if self.delta is not None and self.delta < 0:
            raise KitInputError("delta deve ser não negativo")
