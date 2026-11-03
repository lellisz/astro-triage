"""Build the Layer 2 messages from one feature row and parse the model answer (ADR 0005 D3).

Only the 21 public Gaia DR3 feature columns (and, in the ``with_l1`` arm, the three calibrated
Layer 1 probabilities) enter the prompt, read by name. Identifiers, positions and labels never do,
so a memorized catalogue cannot be looked up. A NULL or NaN value is printed as MISSING; nothing is
imputed and no sentinel number is printed.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Mapping
from typing import Final, Literal, NamedTuple

__all__ = [
    "FEATURE_COLUMNS",
    "FORBIDDEN_SUBSTRINGS",
    "LABELS",
    "PROMPT_VERSION",
    "TEMPLATE_SHA256",
    "Answer",
    "ParseError",
    "build_messages",
    "parse_answer",
]

PROMPT_VERSION: Final = "p1"
LABELS: Final[tuple[str, ...]] = ("STAR", "GALAXY", "QSO", "UNKNOWN")

# Schema §4 rows 2-22, with the unit printed after each value.
_FEATURE_UNITS: Final[dict[str, str]] = {
    "parallax": "mas",
    "parallax_error": "mas",
    "parallax_over_error": "",
    "pm": "mas/yr",
    "pmra": "mas/yr",
    "pmra_error": "mas/yr",
    "pmdec": "mas/yr",
    "pmdec_error": "mas/yr",
    "ruwe": "",
    "astrometric_excess_noise": "mas",
    "astrometric_excess_noise_sig": "",
    "astrometric_params_solved": "",
    "ipd_gof_harmonic_amplitude": "",
    "ipd_frac_multi_peak": "percent",
    "ipd_frac_odd_win": "percent",
    "phot_g_mean_mag": "mag",
    "phot_g_mean_flux_over_error": "",
    "phot_bp_mean_mag": "mag",
    "phot_rp_mean_mag": "mag",
    "bp_rp": "mag",
    "phot_bp_rp_excess_factor": "",
}
FEATURE_COLUMNS: Final[tuple[str, ...]] = tuple(_FEATURE_UNITS)

# Names that must never appear in a prompt (ADR 0005 D3.1). Bare ra/dec/l/b/ra_error/dec_error cannot be listed
# as substrings; the leak test checks them as whole tokens.
FORBIDDEN_SUBSTRINGS: Final[tuple[str, ...]] = (
    "source_id",
    "sdss_",
    "classprob_dsc",
    "in_qso_candidates",
    "in_galaxy_candidates",
    "ap_row_present",
    "healpix_l5",
    "fold",
)

_L1_COLUMNS: Final[tuple[str, ...]] = ("p_cal_star", "p_cal_galaxy", "p_cal_qso")
_MISSING: Final = "MISSING (not measured)"
_TWO_PARAM_SENTENCE: Final = "2-parameter astrometric solution: parallax, proper motion and RUWE were not measured."

_SYSTEM_PROMPT: Final = (
    "You classify astronomical sources observed by the Gaia DR3 survey from their "
    "photometry and astrometry. Choose one label: STAR, GALAXY or QSO (quasar). "
    "Answer UNKNOWN when the data do not separate the classes; stars and quasars can be "
    "indistinguishable in these quantities, and UNKNOWN is a correct answer then. "
    "Values marked MISSING were not measured; do not guess them.\n"
    "Reply with exactly one JSON object and nothing else, with this form: "
    '{"label": "STAR" | "GALAXY" | "QSO" | "UNKNOWN", "confidence": <integer 0-100>}. '
    "No other keys, no explanation."
)
_USER_HEADER: Final = "Gaia DR3 source measurements:"
_L1_HEADER: Final = "Probabilities from a previous classifier:"
_USER_FOOTER: Final = "Classify this source."

# Hash of the template text, not of the filled prompt (ADR 0005 D3.6).
TEMPLATE_SHA256: Final = hashlib.sha256(
    "\n".join(
        (
            _SYSTEM_PROMPT,
            _USER_HEADER,
            *(f"{n} = <value> {u}".rstrip() for n, u in _FEATURE_UNITS.items()),
            _TWO_PARAM_SENTENCE,
            _L1_HEADER,
            *(f"{c} = <value>" for c in _L1_COLUMNS),
            _USER_FOOTER,
        )
    ).encode("utf-8")
).hexdigest()


class Answer(NamedTuple):
    label: str
    confidence: int


class ParseError(ValueError):
    """The model content is not a valid answer; ``status`` is the D4 terminal status."""

    def __init__(self, status: str, message: str) -> None:
        super().__init__(message)
        self.status = status


def _fmt(value: float) -> str:
    # Normalise -0.0 so equal values print identically.
    return "0" if value == 0 else f"{value:.4g}"


def build_messages(
    row: Mapping[str, float | None],
    arm: Literal["blind", "with_l1"],
    l1: Mapping[str, float] | None = None,
) -> list[dict[str, str]]:
    """System + user messages for one source; ``l1`` is read only in the ``with_l1`` arm."""
    lines = [_USER_HEADER]
    for name, unit in _FEATURE_UNITS.items():
        value = row.get(name)
        if value is None or not math.isfinite(value):
            lines.append(f"{name} = {_MISSING}")
        else:
            lines.append(f"{name} = {_fmt(value)} {unit}".rstrip())
    if row.get("astrometric_params_solved") == 3:
        lines.append(_TWO_PARAM_SENTENCE)
    if arm == "with_l1":
        if l1 is None:
            raise ValueError("arm 'with_l1' requires the l1 probabilities")
        lines.append(_L1_HEADER)
        lines.extend(f"{c} = {l1[c]:.2f}" for c in _L1_COLUMNS)
    lines.append(_USER_FOOTER)
    return [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {"role": "user", "content": "\n".join(lines)},
    ]


_FENCE_RE: Final = re.compile(r"^```[A-Za-z]*\s*(.*?)\s*```$", re.DOTALL)


def parse_answer(content: str | None) -> Answer:
    """Validate the model content against the one accepted schema or raise ``ParseError``."""
    if content is None or not content.strip():
        raise ParseError("EMPTY_CONTENT", "empty content")
    text = content.strip()
    fenced = _FENCE_RE.match(text)
    if fenced:
        text = fenced.group(1)
    try:
        obj = json.loads(text)
    except (ValueError, RecursionError):
        raise ParseError("INVALID_JSON", "content is not valid JSON") from None
    if not isinstance(obj, dict) or set(obj) != {"label", "confidence"}:
        raise ParseError(
            "SCHEMA_VIOLATION", "expected exactly the keys label, confidence"
        )
    label, confidence = obj["label"], obj["confidence"]
    if not isinstance(label, str) or label not in LABELS:
        raise ParseError("SCHEMA_VIOLATION", "label outside the accepted enum")
    if (
        not isinstance(confidence, int)
        or isinstance(confidence, bool)
        or not 0 <= confidence <= 100
    ):
        raise ParseError("SCHEMA_VIOLATION", "confidence must be an integer in 0..100")
    return Answer(label, confidence)
