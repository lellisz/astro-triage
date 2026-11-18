"""ADR board model: status word and title read from the header of each ADR file.

Pure and read-only. The ADR number comes from the file name; the status is never typed here.
A status outside the closed vocabulary, or no status line, is ``na``.
"""

from __future__ import annotations

import re
from collections.abc import Collection, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Final, Literal

from astro_triage.dashboard.data_release import REPO_ROOT

ADR_HEADER_LINES: Final = 12
ChipKind = Literal["accepted", "proposed", "closed", "na"]
_STATUS_LINE: Final = re.compile(r"^- Status:\s*(.*)$")
_STATUS_WORD: Final = re.compile(r"^[*_\s]*([A-Za-z]+)")
_TITLE_LINE: Final = re.compile(r"^#\s+ADR\s+\S+\s*[:—–-]\s*(.*)$")
_VOCABULARY: Final[dict[str, ChipKind]] = {
    "accepted": "accepted",
    "proposed": "proposed",
    "closed": "closed",
}


@dataclass(frozen=True)
class AdrEntry:
    number: str
    title: str
    status_word: str | None
    chip: ChipKind


def parse_status(lines: Sequence[str]) -> str | None:
    """First word of the first ``- Status:`` line in the header; None when absent."""
    for line in lines[:ADR_HEADER_LINES]:
        found = _STATUS_LINE.match(line)
        if found:
            word = _STATUS_WORD.match(found.group(1))
            return word.group(1) if word else None
    return None


def chip_kind(word: str | None) -> ChipKind:
    return "na" if word is None else _VOCABULARY.get(word.lower(), "na")


def _title(lines: Sequence[str], fallback: str) -> str:
    for line in lines:
        found = _TITLE_LINE.match(line)
        if found and found.group(1).strip():
            return found.group(1).strip()
    return fallback


def _entry(path: Path) -> AdrEntry:
    number, _, slug = path.stem.partition("-")
    try:
        lines = path.read_text(encoding="utf8").splitlines()
    except (OSError, ValueError):
        return AdrEntry(number, slug, None, "na")
    word = parse_status(lines)
    return AdrEntry(
        number, _title(lines[:ADR_HEADER_LINES], slug), word, chip_kind(word)
    )


def load_board(
    adr_dir: Path | None = None, tracked: Collection[str] | None = None
) -> tuple[AdrEntry, ...]:
    """One entry per ``NNNN-*.md`` in the ADR folder, by file name; ``tracked`` filters names."""
    base = adr_dir or REPO_ROOT / "docs" / "adr"
    try:
        paths = sorted(
            p for p in base.glob("*.md") if p.stem.partition("-")[0].isdigit()
        )
    except OSError:
        return ()
    if tracked is not None:
        paths = [p for p in paths if p.name in tracked]
    return tuple(_entry(p) for p in paths)
