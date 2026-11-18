"""Single resolver of the dashboard's data root; read-only: nothing here creates, moves or deletes."""

from __future__ import annotations

import os
from collections.abc import Mapping
from pathlib import Path
from typing import Final

ENV_VAR: Final = "ASTRO_DATA_ROOT"
_ABSENT: Final = frozenset({"", "~"})


class DataRootMissing(FileNotFoundError):
    """The data root or its ``interim`` folder is not a directory."""


def repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def data_root(environ: Mapping[str, str] | None = None) -> Path:
    """``$ASTRO_DATA_ROOT`` when set and meaningful, else ``<repo>/data``.

    A leading ``~`` is expanded to the home folder; a relative value is resolved against
    the current working directory.
    """
    raw = (os.environ if environ is None else environ).get(ENV_VAR, "").strip()
    if raw in _ABSENT:
        return repo_root() / "data"
    path = Path(raw).expanduser()
    return path if path.is_absolute() else Path.cwd() / path


def interim_dir(environ: Mapping[str, str] | None = None) -> Path:
    return data_root(environ) / "interim"


def require_dir(path: Path) -> Path:
    """``path`` itself when it is a directory; otherwise ``DataRootMissing`` without any path in the text."""
    if not path.is_dir():
        raise DataRootMissing(
            f"data folder not found; set {ENV_VAR} to the folder that holds interim/"
        )
    return path


def require_interim(environ: Mapping[str, str] | None = None) -> Path:
    return require_dir(interim_dir(environ))


def is_under_root(path: Path, root: Path) -> bool:
    try:
        Path(path).resolve().relative_to(Path(root).resolve())
    except ValueError:
        return False
    return True
