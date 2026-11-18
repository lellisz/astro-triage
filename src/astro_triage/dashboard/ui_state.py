"""Global UI state of the dashboard shell: decision rule, sky prior and technical mode.

Pure model, no Streamlit. The values of ``Rule`` are the variant names of ``results.json``.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from typing import Final

from astro_triage.dashboard import data_headline

KEY_RULE: Final = "ui_rule"
KEY_PRIOR: Final = "ui_prior"
KEY_TECH: Final = "ui_tech"
KEYS: Final = (KEY_RULE, KEY_PRIOR, KEY_TECH)
PRIORS: Final = data_headline.SKY_PRIORS
SHIPPED_PRIOR: Final = data_headline.SHIPPED_PRIOR
LINE_LABEL: Final = "Regra · prior"


class Rule(StrEnum):
    SHIPPED = "as_shipped"
    MATCHED = "prior_matched"


RULE_LABEL: Final[Mapping[Rule, str]] = {
    Rule.SHIPPED: "como entregue",
    Rule.MATCHED: "ajustada ao prior",
}


@dataclass(frozen=True)
class UiState:
    rule: Rule
    prior: str
    tech: bool


DEFAULT: Final = UiState(Rule.SHIPPED, SHIPPED_PRIOR, False)


def normalize(raw: Mapping[str, object]) -> UiState:
    """Read the state from a session mapping; a missing or stale value falls back to the default."""
    try:
        rule = Rule(raw.get(KEY_RULE))
    except (ValueError, TypeError):
        rule = DEFAULT.rule
    prior = raw.get(KEY_PRIOR)
    if not isinstance(prior, str) or prior not in PRIORS:
        prior = DEFAULT.prior
    if rule is Rule.SHIPPED:
        prior = SHIPPED_PRIOR
    tech = raw.get(KEY_TECH)
    return UiState(rule, prior, tech if isinstance(tech, bool) else DEFAULT.tech)


def enabled_priors(rule: Rule) -> tuple[str, ...]:
    return (SHIPPED_PRIOR,) if rule is Rule.SHIPPED else tuple(PRIORS)


def disabled_note(rule: Rule, prior: str) -> str | None:
    if prior in enabled_priors(rule):
        return None
    return f"só existe em {SHIPPED_PRIOR}"


def is_partial(prior: str) -> bool:
    return prior in data_headline.PARTIAL_PRIORS


def rule_prior_line(state: UiState) -> str:
    return f"{LINE_LABEL}: {RULE_LABEL[state.rule]} · {state.prior}"
