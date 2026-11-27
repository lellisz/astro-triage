"""Converters from dashboard values to the common JSON types (Num, Est, Head)."""

from __future__ import annotations

import math
from datetime import UTC, datetime
from enum import Enum
from numbers import Real
from typing import Final

import numpy as np

from astro_triage.dashboard import fmt
from astro_triage.dashboard.kit_types import NA_WORD, Value
from astro_triage.web_export import SCHEMA_VERSION

_NUM_KINDS: Final = ("int", "pct", "num", "g")
_EST_KINDS: Final = ("pct", "num", "pp")
PP_SUFFIX: Final = " p.p."
PP_DIGITS: Final = 2


def _real(name: str, x: object) -> float | None:
    """``x`` as a python number, or None when it is None or not finite; bool is refused."""
    if isinstance(x, bool | np.bool_):
        raise TypeError(f"{name} não pode ser bool")
    if x is None:
        return None
    if not isinstance(x, Real):
        raise TypeError(f"{name} deve ser numérico, recebido {type(x).__name__}")
    if not math.isfinite(x):
        return None
    return x.item() if isinstance(x, np.generic) else x


def _signed_num(x: float, digits: int, signed: bool) -> str:
    """``fmt.pt_num``, except that a signed value rounding to zero prints "0,000", no sign."""
    if signed and round(x, digits) == 0:
        return fmt.pt_num(0.0, digits)
    return fmt.pt_num(x, digits, signed)


def _format(kind: str, x: float, digits: int | None, signed: bool) -> str:
    if kind == "int":
        return fmt.pt_int(x)
    if kind == "pct":
        return fmt.pt_pct(x, 1 if digits is None else digits)
    if kind == "num":
        return _signed_num(x, 3 if digits is None else digits, signed)
    if kind == "pp":
        return _signed_num(x * 100, PP_DIGITS if digits is None else digits, signed) + PP_SUFFIX
    return fmt.pt_g(x)


def num(
    value: float | None,
    text: str | None = None,
    *,
    kind: str = "int",
    digits: int | None = None,
    signed: bool = False,
) -> dict:
    """``{value, text}``; None, NaN or inf give ``value`` null and text "n/a", never 0."""
    if kind not in _NUM_KINDS:
        raise ValueError(f"kind desconhecido: {kind}")
    x = _real("value", value)
    if x is None:
        return {"value": None, "text": NA_WORD}
    shown = text if text is not None else _format(kind, x, digits, signed)
    return {"value": x, "text": shown}


def est(
    point: float | None,
    lo: float | None,
    hi: float | None,
    *,
    digits: int | None = None,
    kind: str = "pct",
    signed: bool = False,
) -> dict:
    """``{point, lo, hi, text, ci_text}``; any missing or non-finite part nulls all of it."""
    if kind not in _EST_KINDS:
        raise ValueError(f"kind desconhecido: {kind}")
    parts = [_real(n, v) for n, v in (("point", point), ("lo", lo), ("hi", hi))]
    if any(p is None for p in parts):
        return {
            "point": None,
            "lo": None,
            "hi": None,
            "text": NA_WORD,
            "ci_text": NA_WORD,
        }
    text, lo_text, hi_text = (_format(kind, v, digits, signed) for v in parts)
    return {
        "point": parts[0],
        "lo": parts[1],
        "hi": parts[2],
        "text": text,
        "ci_text": f"{fmt.pt_ci_label()}: {lo_text} a {hi_text}",
    }


def from_value(v: Value, *, digits: int | None = None) -> dict:
    """Est of a kit ``Value``; a point without interval is n/a here (use ``num`` for it)."""
    lo, hi = (None, None) if v.ci is None else (v.ci.lo, v.ci.hi)
    return est(v.point, lo, hi, digits=digits)


def head(release: str, page: str) -> dict:
    """First keys of every file. ``provisional`` is always true: there is no switch for it."""
    return {
        "schema_version": SCHEMA_VERSION,
        "release": release,
        "provisional": True,
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "page": page,
    }


def clean(obj: object) -> object:
    """Plain JSON-ready copy: numpy to python, NaN and inf to None, recursively."""
    if isinstance(obj, dict):
        return {str(k): clean(v) for k, v in obj.items()}
    if isinstance(obj, list | tuple):
        return [clean(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return clean(obj.tolist())
    if isinstance(obj, np.generic):
        return clean(obj.item())
    if isinstance(obj, Enum):
        return clean(obj.value)
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    return obj
