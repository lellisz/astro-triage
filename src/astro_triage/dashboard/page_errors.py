"""Short, path-free Portuguese descriptions of the errors that can break a page. No Streamlit."""

from __future__ import annotations

import re
from pathlib import Path

_PATH = re.compile(r"(?:[A-Za-z]:)?[\\/][\w.\-]+(?:[\\/][\w.\-]+)+")
_MAX_CHARS = 120


def describe_error(exc: BaseException) -> str:
    """What went wrong, without file paths or tracebacks (type and a short message only)."""
    if isinstance(exc, OSError) and exc.filename:
        return f"faltou o artefato {Path(str(exc.filename)).name}"
    if isinstance(exc, KeyError) and exc.args:
        return f"faltou o campo {exc.args[0]!r} em um artefato"
    message = _PATH.sub("…", str(exc)).strip()[:_MAX_CHARS]
    return f"{type(exc).__name__}: {message}" if message else type(exc).__name__


def page_error(exc: BaseException) -> str:
    return f"Não foi possível montar esta página: {describe_error(exc)}."
