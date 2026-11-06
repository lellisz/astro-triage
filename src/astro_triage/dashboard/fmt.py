"""pt-BR number formatting helpers for the dashboard."""

from __future__ import annotations

CI_LEVEL = 0.95


def pt_ci_label() -> str:
    return f"IC {CI_LEVEL * 100:.0f}%"


def pt_int(n: float) -> str:
    return f"{int(n):,}".replace(",", ".")


def pt_pct(x: float, digits: int = 1) -> str:
    return f"{x * 100:.{digits}f}".replace(".", ",") + "%"


def pt_g(x: float) -> str:
    return f"{x:g}".replace(".", ",")


def pt_num(x: float, digits: int = 3, signed: bool = False) -> str:
    return f"{x:{'+' if signed else ''}.{digits}f}".replace(".", ",")


def pt_pct_interval(point: float, lo: float, hi: float, digits: int = 1) -> str:
    return f"{pt_pct(point, digits)} ({pt_ci_label()}: {pt_pct(lo, digits)} a {pt_pct(hi, digits)})"


def pt_interval(
    point: float, lo: float, hi: float, mark: str, signed: bool = False
) -> str:
    if mark:
        return mark
    return (
        f"{pt_num(point, 3, signed)} ({pt_ci_label()}: {pt_num(lo, 3, signed)} a "
        f"{pt_num(hi, 3, signed)})"
    )
