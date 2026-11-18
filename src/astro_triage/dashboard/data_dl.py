"""Loader and pure model of the deep-learning evaluation (``dl/eval/results_dl.json``).

No Streamlit import. Hashes are dropped on load; nothing here is per object or a path.
The model reads the verdict vocabulary of ``dl_compare`` strictly: an unknown token raises
``DlDataError`` and is never mapped to a tie.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

import pandas as pd

from astro_triage.dashboard.adr_board import ADR_HEADER_LINES, parse_status
from astro_triage.dashboard.constants import ADR_NUMBER
from astro_triage.dashboard.data_release import REPO_ROOT, Release, strip_sha256
from astro_triage.evaluation.dl_compare import TARGET_CLASSES

_EVAL_DIR = ("dl", "eval")
HEADLINE_ROLE = "headline"
SENSITIVITY_ROLE = "sensitivity"
DESCRIPTIVE_ROLE = "descriptive"
FRONT_PREFIX = "front"
VERDICT_ROLES = (HEADLINE_ROLE, SENSITIVITY_ROLE)
# Vocabulary of dl_compare.pair_verdict (pinned against the producer in the tests).
PAIR_TIE_TOKEN = "tie: LightGBM stays"
PAIR_WIN_TOKEN = "X wins"  # literal: the producer never writes the system name


class DlDataError(ValueError):
    """The artifact does not follow the grammar the tab reads."""


class ClassVerdict(StrEnum):
    WIN = "win"
    TIE = "tie"
    WORSE = "worse"
    NA = (
        "n/a"  # internal state only (cell not reportable); never read from the artifact
    )


class PairVerdict(StrEnum):
    TIE = "tie"
    WIN = "win"


@dataclass(frozen=True)
class ClassCell:
    cls: str
    d: float | None
    lo: float | None
    hi: float | None
    delta: float | None
    verdict: ClassVerdict | None
    ci_l5: tuple[float, float] | None
    purity: Mapping[str, tuple[float, float, float]]


@dataclass(frozen=True)
class PairCell:
    pair: str
    prior: str
    level: str
    role: str
    classes: Mapping[str, ClassCell]
    verdict: PairVerdict | None

    @property
    def inconsistent(self) -> bool:
        """The pair verdict disagrees with the class verdicts (a win needs both classes)."""
        if self.verdict is None:
            return False
        both = all(c.verdict is ClassVerdict.WIN for c in self.classes.values())
        return both != (self.verdict is PairVerdict.WIN)


@dataclass(frozen=True)
class Headline:
    prior: str
    level: str
    cells: tuple[PairCell, ...]


def load_results_dl(rel: Release) -> dict | None:
    """Parsed ``dl/eval/results_dl.json`` without any ``*sha256`` key; None when absent."""
    path = rel.root.joinpath(*_EVAL_DIR, "results_dl.json")
    if not path.is_file():
        return None
    return strip_sha256(json.loads(path.read_text(encoding="utf-8")))


def load_risk_coverage_dl(rel: Release) -> pd.DataFrame | None:
    """``risk_coverage_dl.csv`` (system, seed, tau, coverage, accuracy); None when absent."""
    path = rel.root.joinpath(*_EVAL_DIR, "risk_coverage_dl.csv")
    if not path.is_file():
        return None
    return pd.read_csv(path)


def _finite(x: object) -> float | None:
    if isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(x):
        return None
    return float(x)


def parse_class_verdict(token: object, where: str) -> ClassVerdict:
    if isinstance(token, str):
        try:
            parsed = ClassVerdict(token)
        except ValueError:
            pass
        else:
            if parsed is not ClassVerdict.NA:
                return parsed
    raise DlDataError(f"veredito de classe desconhecido em {where}: {token!r}")


def parse_pair_verdict(token: object, where: str) -> PairVerdict:
    if token == PAIR_TIE_TOKEN:
        return PairVerdict.TIE
    if token == PAIR_WIN_TOKEN:
        return PairVerdict.WIN
    raise DlDataError(f"veredito de par desconhecido em {where}: {token!r}")


def _purity(raw: object) -> dict[str, tuple[float, float, float]]:
    out: dict[str, tuple[float, float, float]] = {}
    for system, stats in (raw if isinstance(raw, Mapping) else {}).items():
        got = [_finite(stats.get(k)) for k in ("min", "median", "max")]
        if None not in got:
            out[system] = (got[0], got[1], got[2])  # type: ignore[assignment]
    return out


def _parse_class(cls: str, raw: Mapping, role: str, where: str) -> ClassCell:
    judged = role in VERDICT_ROLES
    na = bool(raw.get("na")) or raw.get("reportable") is False
    verdict = None
    if judged:
        token = parse_class_verdict(raw.get("class_verdict"), where)
        verdict = ClassVerdict.NA if na else token
    ci = raw.get("ci")
    pair = ci if isinstance(ci, (list, tuple)) and len(ci) == 2 else (None, None)
    d, lo, hi = _finite(raw.get("d")), _finite(pair[0]), _finite(pair[1])
    delta = _finite(raw.get("delta"))
    if na:
        d = lo = hi = None
    elif judged and None in (d, lo, hi, delta):
        raise DlDataError(f"valor ausente ou não finito em {where}")
    l5 = raw.get("ci_l5")
    l5_pair = (
        (_finite(l5[0]), _finite(l5[1]))
        if isinstance(l5, (list, tuple)) and len(l5) == 2
        else (None, None)
    )
    return ClassCell(
        cls,
        d,
        lo,
        hi,
        delta,
        verdict,
        None if None in l5_pair else l5_pair,  # type: ignore[arg-type]
        _purity(raw.get("purity_seed")),
    )


def _parse_cell(pair: str, prior: str, level: str, raw: Mapping) -> PairCell:
    role = str(raw.get("role", ""))
    where = f"{pair} {prior} {level}"
    classes = {}
    for cls in TARGET_CLASSES:
        body = (raw.get("classes") or {}).get(cls)
        if body is None and role in VERDICT_ROLES:
            raise DlDataError(f"classe {cls} ausente em {where}")
        if body is not None:
            classes[cls] = _parse_class(cls, body, role, f"{where} {cls}")
    verdict = None
    if role in VERDICT_ROLES:
        verdict = parse_pair_verdict(raw.get("verdict"), where)
    return PairCell(pair, prior, level, role, classes, verdict)


def load_cells(results: Mapping) -> list[PairCell]:
    """Every (pair, prior, level) cell of ``compare.pairs``; raises ``DlDataError`` on bad tokens."""
    pairs = (results.get("compare") or {}).get("pairs")
    if not isinstance(pairs, Mapping):
        raise DlDataError("o artefato não tem compare.pairs")
    return [
        _parse_cell(pair, prior, level, raw)
        for pair, priors in pairs.items()
        for prior, levels in priors.items()
        for level, raw in levels.items()
    ]


def headline(results: Mapping) -> Headline:
    """The cells whose role is 'headline' (picked by role, never by position)."""
    cells = [c for c in load_cells(results) if c.role == HEADLINE_ROLE]
    if not cells:
        raise DlDataError("o artefato não tem célula com papel 'headline'")
    where = {(c.prior, c.level) for c in cells}
    if len(where) != 1:
        raise DlDataError("as células 'headline' não estão num único prior e C*")
    (prior, level) = next(iter(where))
    return Headline(prior, level, tuple(cells))


def sensitivity(results: Mapping, head: Headline) -> list[PairCell]:
    """Sensitivity cells of the headline pairs, ordered by prior, level and pair."""
    pairs = [c.pair for c in head.cells]
    cells = [
        c for c in load_cells(results) if c.role == SENSITIVITY_ROLE and c.pair in pairs
    ]
    return sorted(cells, key=lambda c: (c.prior, c.level, pairs.index(c.pair)))


def descriptive(results: Mapping, head: Headline) -> list[PairCell]:
    """Descriptive cells (ensembles and fronts) at the headline prior and C*."""
    return [
        c
        for c in load_cells(results)
        if (c.prior, c.level) == (head.prior, head.level)
        and (c.role == DESCRIPTIVE_ROLE or c.role.startswith(FRONT_PREFIX))
    ]


def below_floor(head: Headline) -> int:
    """Number of headline class cells the artifact marks as not reportable."""
    return sum(
        c.verdict is ClassVerdict.NA
        for cell in head.cells
        for c in cell.classes.values()
    )


def adr_status(adr_dir: Path | None = None) -> str | None:
    """Status word of the ADR header (e.g. 'Proposed'); None when it cannot be read."""
    base = adr_dir or REPO_ROOT / "docs" / "adr"
    try:
        path = min(base.glob(f"{ADR_NUMBER:04d}-*.md"))
        head = path.read_text(encoding="utf-8").splitlines()[:ADR_HEADER_LINES]
    except (OSError, ValueError):
        return None
    return parse_status(head)


def freeze_exists(rel: Release) -> bool:
    return rel.root.joinpath("dl", "val_freeze.json").is_file()


def load_limitations(rel: Release) -> list[str] | None:
    """Bullets under '## Limitations' of ``results_dl.md``; None when unreadable."""
    path = rel.root.joinpath(*_EVAL_DIR, "results_dl.md")
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except (OSError, ValueError):
        return None
    out: list[str] = []
    inside = False
    for line in lines:
        if line.startswith("## "):
            if inside:
                break
            inside = line[3:].strip() == "Limitations"
        elif inside and line.startswith("- "):
            out.append(line[2:].strip())
    return out or None
