"""Load the NVIDIA API key and scrub it from any text that may be logged.

The key comes from the environment, else from the ``NVIDIA_API_KEY`` line of the repository
``.env`` (only that line is parsed). It is never a default, never part of an exception message
and never printed.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from pathlib import Path
from typing import Final

from astro_triage.ingest.tap import _sanitize

__all__ = ["MissingKeyError", "load_api_key", "redact"]

KEY_NAME: Final = "NVIDIA_API_KEY"
_DEFAULT_DOTENV: Final = Path(__file__).resolve().parents[3] / ".env"


class MissingKeyError(RuntimeError):
    """NVIDIA_API_KEY is set neither in the environment nor in the .env file."""


def _read_dotenv_key(path: Path) -> str | None:
    try:
        text = path.read_text(encoding="utf-8-sig")
    except OSError:
        return None
    for line in text.splitlines():
        line = line.strip()
        if line.startswith("export "):
            line = line[len("export ") :].lstrip()
        name, sep, value = line.partition("=")
        if not sep or name.strip() != KEY_NAME:
            continue
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        return value or None
    return None


def load_api_key(
    env: Mapping[str, str] | None = None, dotenv_path: Path | None = None
) -> str:
    """Return the key; the environment wins over ``.env``.

    The default ``.env`` (repository root) is consulted only when ``env`` is not injected.
    """
    source = os.environ if env is None else env
    value = source.get(KEY_NAME, "").strip()
    if value:
        return value
    path = (
        dotenv_path
        if dotenv_path is not None
        else (_DEFAULT_DOTENV if env is None else None)
    )
    if path is not None:
        value = _read_dotenv_key(path) or ""
        if value:
            return value
    raise MissingKeyError(f"{KEY_NAME} is not set in the environment or in .env")


def redact(text: str, key: str | None, limit: int = 200) -> str:
    """Replace the literal key, mask Bearer/key= patterns, flatten and truncate."""
    if key:
        text = text.replace(key, "***")
    return _sanitize(text)[:limit]
