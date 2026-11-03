"""Build the zip that is uploaded to Kaggle as a private dataset.

Contents: src/, scripts/, manifests/, kaggle/, requirements.txt and the ruff/pyproject
configuration when present. Nothing else (no data, logs, docs, tests, venv or git state).
The zip is reproducible (sorted entries, fixed timestamps) and is refused when it passes
5 MB, holds a data file (.parquet/.fits/.duckdb) or embeds an absolute path of the user.
"""

from __future__ import annotations

import argparse
import hashlib
import re
import sys
import zipfile
from pathlib import Path
from typing import Final

REPO_ROOT = Path(__file__).resolve().parents[1]

PACKAGE_DIRS: Final[tuple[str, ...]] = ("src", "scripts", "manifests", "kaggle")
PACKAGE_FILES: Final[tuple[str, ...]] = (
    "requirements.txt",
    "pyproject.toml",
    "ruff.toml",
    ".ruff.toml",
)
SKIP_DIRS: Final[frozenset[str]] = frozenset(
    {"__pycache__", ".pytest_cache", ".ruff_cache", ".mypy_cache", ".git", ".venv"}
)
SKIP_SUFFIXES: Final[frozenset[str]] = frozenset({".pyc", ".pyo", ".tmp", ".log"})
FORBIDDEN_SUFFIXES: Final[frozenset[str]] = frozenset({".parquet", ".fits", ".duckdb"})
MAX_ZIP_BYTES: Final = 5 * 1024 * 1024
ZIP_NAME: Final = "astro_triage_pkg.zip"
_FIXED_TIME: Final = (1980, 1, 1, 0, 0, 0)
_USER_PATH_RE: Final = re.compile(rb"[A-Za-z]:[\\/]+Users[\\/]|/Users/\w|/home/\w")


class PackageError(Exception):
    """The package violates one of the rules above; the message names the file."""


def collect(root: Path) -> list[Path]:
    """Return the files to ship, relative to ``root``, sorted."""
    found: list[Path] = []
    for name in PACKAGE_DIRS:
        base = root / name
        if not base.is_dir():
            raise PackageError(f"missing directory {name}/")
        for path in base.rglob("*"):
            rel = path.relative_to(root)
            if not path.is_file() or SKIP_DIRS & set(rel.parts):
                continue
            if path.suffix in SKIP_SUFFIXES:
                continue
            found.append(rel)
    for name in PACKAGE_FILES:
        if (root / name).is_file():
            found.append(Path(name))
    if Path("requirements.txt") not in found:
        raise PackageError("missing requirements.txt")
    return sorted(found, key=lambda p: p.as_posix())


def check_content(rel: Path, data: bytes, markers: list[bytes]) -> None:
    if rel.suffix in FORBIDDEN_SUFFIXES:
        raise PackageError(f"data file in the package: {rel.as_posix()}")
    if _USER_PATH_RE.search(data):
        raise PackageError(f"absolute user path in {rel.as_posix()}")
    for marker in markers:
        if marker and marker in data:
            raise PackageError(f"home directory path in {rel.as_posix()}")


def build(root: Path, outdir: Path) -> tuple[Path, str, list[tuple[str, int]]]:
    files = collect(root)
    home = str(Path.home()).encode()
    markers = [home, home.replace(b"\\", b"/")]
    outdir.mkdir(parents=True, exist_ok=True)
    target = outdir / ZIP_NAME
    tmp = target.with_name(target.name + ".tmp")
    listing: list[tuple[str, int]] = []
    try:
        with zipfile.ZipFile(tmp, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            for rel in files:
                data = (root / rel).read_bytes()
                check_content(rel, data, markers)
                info = zipfile.ZipInfo(rel.as_posix(), date_time=_FIXED_TIME)
                info.compress_type = zipfile.ZIP_DEFLATED
                info.external_attr = 0o644 << 16
                archive.writestr(info, data)
                listing.append((rel.as_posix(), len(data)))
        size = tmp.stat().st_size
        if size > MAX_ZIP_BYTES:
            raise PackageError(f"zip is {size} bytes, above the {MAX_ZIP_BYTES} limit")
        tmp.replace(target)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise
    return target, hashlib.sha256(target.read_bytes()).hexdigest(), listing


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outdir", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        target, digest, listing = build(REPO_ROOT, args.outdir)
    except (PackageError, OSError) as exc:
        print(f"package failed: {exc}", file=sys.stderr)
        return 1
    print(f"zip: {target.name}")
    print(f"bytes: {target.stat().st_size}")
    print(f"sha256: {digest}")
    print(f"files: {len(listing)}")
    for name, size in listing:
        print(f"  {name} ({size})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
