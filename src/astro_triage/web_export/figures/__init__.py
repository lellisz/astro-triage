"""Site figures drawn with matplotlib from the exported page JSON (one SVG per figure).

``write_figures`` reads ``<data_dir>/<release>/*.json`` and writes
``<out_dir>/<release>/figures/<id>.svg`` plus ``figures.json`` (title, caption and aria-label of
each figure, read by ``web/components/charts/MplFigure.tsx``). Every SVG is built in memory before
the first file is written, so a bad input never leaves a half set.
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any, Final

from astro_triage.web_export import SCHEMA_VERSION

PAGES: Final = ("meta", "overview", "galaxies", "compare", "graficos")
RELEASE_NAME: Final = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]*")
FIG_ID: Final = re.compile(r"[a-z0-9][a-z0-9-]*")
MANIFEST: Final = "figures.json"


class FigureError(Exception):
    """A refusal or failure with a short message that is safe to print."""


def _is_link(path: Path) -> bool:
    return path.is_symlink() or os.path.isjunction(path)


def _load(folder: Path, release: str) -> dict[str, Any]:
    pages: dict[str, Any] = {}
    for name in PAGES:
        file = folder / f"{name}.json"
        try:
            doc = json.loads(file.read_text(encoding="utf-8"))
        except OSError:
            raise FigureError(f"arquivo de dados ausente ou ilegível: {file}") from None
        except ValueError:
            raise FigureError(f"JSON inválido em {file}") from None
        if not isinstance(doc, dict):
            raise FigureError(f"{file}: esperado um objeto JSON")
        if doc.get("schema_version") != SCHEMA_VERSION:
            raise FigureError(f"{file}: schema_version {doc.get('schema_version')} difere de {SCHEMA_VERSION}")
        if doc.get("release") != release:
            raise FigureError(f"{file}: release {doc.get('release')} difere de {release}")
        if doc.get("provisional") is not True:
            raise FigureError(f"{file}: provisional deve ser true")
        pages[name] = doc
    return pages


def write_figures(data_dir: Path, out_dir: Path, release: str) -> list[Path]:
    """Render every figure of ``release`` and return the written SVG paths."""
    from astro_triage.web_export.figures.charts import builders

    if not RELEASE_NAME.fullmatch(release):
        raise FigureError("nome de release inválido")
    pages = _load(data_dir / release, release)
    try:
        figs = [build(pages) for build in builders(pages)]
    except (KeyError, TypeError, ValueError, IndexError, StopIteration) as exc:
        raise FigureError(f"JSON fora do formato esperado ({type(exc).__name__}: {exc})") from None
    ids = [f.id for f in figs]
    if len(set(ids)) != len(ids) or not all(FIG_ID.fullmatch(i) for i in ids):
        raise FigureError(f"ids de figura inválidos ou repetidos: {ids}")

    target = out_dir / release / "figures"
    if any(_is_link(p) for p in (out_dir, out_dir / release, target)):
        raise FigureError("a pasta de saída não pode ser link nem junction")
    target.mkdir(parents=True, exist_ok=True)
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "release": release,
        "provisional": True,
        "figures": {f.id: {"title": f.title, "caption": f.caption, "aria_label": f.aria_label} for f in figs},
    }
    written = []
    for f in figs:
        path = target / f"{f.id}.svg"
        path.write_text(f.svg, encoding="utf-8", newline="\n")
        written.append(path)
    (target / MANIFEST).write_text(
        json.dumps(manifest, ensure_ascii=False, sort_keys=True, indent=1), encoding="utf-8", newline="\n"
    )
    return written
