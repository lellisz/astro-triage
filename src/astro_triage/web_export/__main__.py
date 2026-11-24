"""CLI: ``python -m astro_triage.web_export --release <name> --out <dir>``.

Writes the seven JSON files of ADR 0008 into ``<out>/<release>/``. The data root is only
read; the files are written to a temporary folder inside ``<out>`` and moved into place once
all of them exist, so a failed run never leaves a half export.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import tempfile
import time
from collections.abc import Callable
from pathlib import Path
from typing import Final

from astro_triage.dashboard import data as D
from astro_triage.dashboard import paths
from astro_triage.web_export import pages
from astro_triage.web_export.types import clean

RELEASE_NAME: Final = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]*")
SUFFIX: Final = ".json"
READ_ERRORS: Final = (OSError, ValueError, KeyError)
RENAME_ATTEMPTS: Final = 20
RENAME_WAIT_S: Final = 0.25


class ExportError(Exception):
    """A refusal or failure with a short message that is safe to print."""


def _is_link(path: Path) -> bool:
    return path.is_symlink() or os.path.isjunction(path)


def _remove_flat(folder: Path) -> None:
    """Delete a folder that holds only plain files; anything else is left untouched."""
    entries = list(folder.iterdir())
    if any(_is_link(e) or not e.is_file() for e in entries):
        raise ExportError(
            f"a pasta {folder.name} tem conteúdo que não é arquivo comum; nada foi apagado"
        )
    for entry in entries:
        entry.unlink()
    folder.rmdir()


def _check_previous(target: Path, docs: dict[str, object]) -> None:
    """Refuse, before moving anything, a target that is not a previous export."""
    if not target.exists():
        return
    expected = {f"{name}{SUFFIX}" for name in docs}
    if not target.is_dir() or _is_link(target):
        raise ExportError("o destino existe e não é uma pasta de exportação; nada foi movido")
    for entry in target.iterdir():
        if _is_link(entry) or not entry.is_file() or entry.name not in expected:
            raise ExportError(
                "o destino existe e não é uma exportação anterior; nada foi movido"
            )


def _load(name: str) -> D.Release:
    try:
        return D.load_release(name, paths.interim_dir())
    except FileNotFoundError:
        raise ExportError(
            f"release '{name}' não encontrada; confira --release e {paths.ENV_VAR}"
        ) from None
    except READ_ERRORS as exc:
        raise ExportError(f"release '{name}' ilegível ({type(exc).__name__})") from None


def _documents(rel: D.Release) -> dict[str, object]:
    pixels = D.pixel_table(rel)
    builders: dict[str, Callable[[], object]] = {
        "meta": lambda: pages.meta(rel),
        "overview": lambda: pages.overview(rel, pixels),
        "sky": lambda: pages.sky(rel, pixels),
        "galaxies": lambda: pages.galaxies(rel),
        "compare": lambda: pages.compare(rel),
        "method": lambda: pages.method(rel),
        "fmt_samples": pages.fmt_samples,
    }
    docs = {}
    for name, build in builders.items():
        try:
            docs[name] = build()
        except READ_ERRORS as exc:
            raise ExportError(
                f"falha ao montar '{name}' ({type(exc).__name__})"
            ) from None
    return docs


def _write(folder: Path, docs: dict[str, object]) -> None:
    for name, doc in docs.items():
        text = json.dumps(
            clean(doc), allow_nan=False, ensure_ascii=False, sort_keys=True
        )
        (folder / f"{name}{SUFFIX}").write_text(text, encoding="utf-8", newline="\n")


def _rename(src: Path, dst: Path) -> None:
    """Rename with retries: on Windows a scanner may hold a fresh file for a moment."""
    for attempt in range(RENAME_ATTEMPTS):
        try:
            src.rename(dst)
            return
        except PermissionError:
            if attempt == RENAME_ATTEMPTS - 1:
                raise
            time.sleep(RENAME_WAIT_S)


def _publish(out: Path, release: str, docs: dict[str, object]) -> Path:
    target = out / release
    if _is_link(out) or _is_link(target):
        raise ExportError("--out e <out>/<release> não podem ser link nem junction")
    _check_previous(target, docs)
    out.mkdir(parents=True, exist_ok=True)
    tmp = Path(tempfile.mkdtemp(prefix=f".{release}.tmp-", dir=out))
    old = tmp.with_name(f"{tmp.name}.old")
    try:
        _write(tmp, docs)
        replaced = target.exists()
        if replaced:
            _rename(target, old)
        try:
            _rename(tmp, target)
        except OSError:
            if replaced:
                _rename(old, target)
            raise
        if replaced:
            _remove_flat(old)
    finally:
        if tmp.exists():
            _remove_flat(tmp)
    return target


def run(release: str, out: Path) -> Path:
    if not RELEASE_NAME.fullmatch(release):
        raise ExportError("nome de release inválido")
    root = paths.data_root()
    target = out / release
    if paths.is_under_root(out, root) or paths.is_under_root(target, root):
        raise ExportError("--out não pode ficar dentro da pasta de dados")
    if paths.is_under_root(root, target):
        raise ExportError("o destino não pode ser a pasta de dados nem um ancestral dela")
    rel = _load(release)
    return _publish(out, release, _documents(rel))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m astro_triage.web_export",
        description="Exporta os JSON da interface web a partir de uma release.",
    )
    parser.add_argument("--release", required=True, help="nome da release")
    parser.add_argument("--out", required=True, type=Path, help="pasta de saída")
    args = parser.parse_args(argv)
    try:
        run(args.release, args.out)
    except ExportError as exc:
        print(f"erro: {exc}", file=sys.stderr)
        return 1
    except OSError as exc:
        print(f"erro: não foi possível gravar ({type(exc).__name__})", file=sys.stderr)
        return 1
    print(f"exportado: {args.out / args.release}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
