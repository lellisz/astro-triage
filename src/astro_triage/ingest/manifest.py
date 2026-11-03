"""Data manifest generator: provenance, exact query, terms of use and required citations.

One JSON manifest per ingested table, written to ``manifests/<table_name>.manifest.json``.
Raw data never goes to git; only scripts, queries and manifests do.

``tap._sanitize`` is deliberately not used on ``query_text``: it flattens whitespace and
truncates at 300 characters, which would break the "exact query" requirement. Instead the
query is rejected when it looks like it embeds a credential.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import re
import tempfile
from dataclasses import asdict, dataclass, field
from datetime import datetime
from importlib import metadata
from pathlib import Path
from typing import Any, Final, Literal

__all__ = [
    "GAIA_ACKNOWLEDGEMENT",
    "GAIA_CITATIONS",
    "GAIA_TERMS",
    "SDSS_ACKNOWLEDGEMENT",
    "SDSS_CITATIONS",
    "SDSS_TERMS",
    "VIZIER_ACKNOWLEDGEMENT",
    "VIZIER_CITATIONS",
    "VIZIER_TERMS",
    "XMATCH_ACKNOWLEDGEMENT",
    "XMATCH_CITATIONS",
    "XMATCH_TERMS",
    "Provenance",
    "Source",
    "build_manifest",
    "client_versions",
    "file_sha256",
    "write_manifest",
]

Source = Literal[
    "Gaia Archive TAP", "VizieR/CDS TAP", "CDS XMatch", "SDSS SAS", "derived"
]

_REPO_ROOT: Final = Path(__file__).resolve().parents[3]
_CHUNK_BYTES: Final = 1024 * 1024
_SCHEMA_VERSION: Final = 1
_TABLE_NAME_RE: Final = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]*")
_SECRET_RE: Final = re.compile(
    r"(?:password|passwd|pwd|token|authorization|cookie|jsessionid|secret"
    r"|api[_-]?key)['\"]?\s*[=:]\s*\S+|\bbearer\s+[\w.~+/=-]+",
    re.IGNORECASE,
)
_PENDING_ESA: Final = "to be verified against the ESA page"
_PENDING_CDS: Final = "to be verified: read the CDS acknowledgement page"
_PENDING_SDSS: Final = (
    "copy verbatim from https://www.sdss4.org/collaboration/citing-sdss/"
)

GAIA_TERMS: Final = "CC BY-NC 3.0 IGO (Gaia Archive data licence)"
GAIA_CITATIONS: Final[tuple[str, ...]] = (
    "Gaia Collaboration, Vallenari et al. 2023, A&A 674, A1, arXiv:2208.00211",
    "Apsis I: Creevey et al., arXiv:2206.05864 (cite if Apsis parameters are used; "
    "requirement " + _PENDING_ESA + ")",
    "DSC (Apsis III): Delchambre et al., arXiv:2206.06710 (cite if classprob_dsc_* "
    "are used; requirement " + _PENDING_ESA + ")",
    "Table DOI: https://doi.org/10.17876/gaia/dr.3/<table_identifier>",
)
GAIA_ACKNOWLEDGEMENT: Final = (
    "This work has made use of data from the European Space Agency (ESA) mission Gaia "
    "(https://www.cosmos.esa.int/gaia), processed by the Gaia Data Processing and "
    "Analysis Consortium (DPAC, https://www.cosmos.esa.int/web/gaia/dpac/consortium). "
    "Funding for the DPAC has been provided by national institutions, in particular the "
    "institutions participating in the Gaia Multilateral Agreement. "
    f"[{_PENDING_ESA}; text taken from the AIP mirror, medium confidence]"
)

VIZIER_TERMS: Final = (
    "VizieR table I/355 (Gaia DR3) is a copy of the Gaia data: Gaia terms apply ("
    + GAIA_TERMS
    + "); VizieR service terms "
    + _PENDING_CDS
)
VIZIER_CITATIONS: Final[tuple[str, ...]] = (
    *GAIA_CITATIONS,
    f"VizieR service citation: {_PENDING_CDS}",
)
VIZIER_ACKNOWLEDGEMENT: Final = f"VizieR/CDS acknowledgement text: {_PENDING_CDS}"

XMATCH_TERMS: Final = f"CDS XMatch service terms: {_PENDING_CDS}"
XMATCH_CITATIONS: Final[tuple[str, ...]] = (
    f"CDS XMatch service citation: {_PENDING_CDS}",
    f"VizieR service citation: {_PENDING_CDS}",
)
XMATCH_ACKNOWLEDGEMENT: Final = f"CDS XMatch acknowledgement text: {_PENDING_CDS}"

SDSS_TERMS: Final = (
    "SDSS Image Use Policy: images may be used for any purpose with credit "
    '"Sloan Digital Sky Survey". Rate limits and use with third-party APIs/VLMs are '
    "undocumented."
)
SDSS_CITATIONS: Final[tuple[str, ...]] = (
    "Abdurro'uf et al. 2022, ApJS 259, 35 (SDSS DR17)",
    "Blanton et al. 2017 (SDSS-IV overview)",
)
SDSS_ACKNOWLEDGEMENT: Final = f"SDSS-IV acknowledgement: {_PENDING_SDSS}"


@dataclass(frozen=True)
class Provenance:
    table_name: str
    source: Source
    release: str
    obtained_utc: str
    query_text: str
    query_sha256: str
    row_count: int
    file_sha256: str
    file_bytes: int
    terms: str
    citations: list[str]
    acknowledgement: str
    client_versions: dict[str, str] = field(default_factory=lambda: client_versions())
    notes: list[str] = field(default_factory=list)


def client_versions() -> dict[str, str]:
    versions = {"python": platform.python_version()}
    for package in ("astroquery", "pyarrow"):
        try:
            versions[package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            versions[package] = "not installed"
    return versions


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(_CHUNK_BYTES):
            digest.update(chunk)
    return digest.hexdigest()


def _display_path(path: Path) -> str:
    resolved = Path(path).resolve()
    try:
        return resolved.relative_to(_REPO_ROOT).as_posix()
    except ValueError:
        return resolved.name


def _validate(provenance: Provenance) -> None:
    if not _TABLE_NAME_RE.fullmatch(provenance.table_name):
        raise ValueError(
            f"table_name {provenance.table_name!r} must match {_TABLE_NAME_RE.pattern}"
        )
    try:
        datetime.fromisoformat(provenance.obtained_utc)
    except ValueError as exc:
        raise ValueError(
            f"obtained_utc {provenance.obtained_utc!r} is not ISO 8601"
        ) from exc
    if _SECRET_RE.search(provenance.query_text):
        raise ValueError(
            "query_text looks like it embeds a credential; remove it before writing a manifest"
        )
    if provenance.row_count < 0:
        raise ValueError("row_count must be >= 0")


def _assert_no_local_paths(text: str) -> None:
    home = Path.home()
    forbidden = {str(_REPO_ROOT), _REPO_ROOT.as_posix(), str(home), home.as_posix()}
    escaped = {json.dumps(item)[1:-1] for item in forbidden}
    for needle in forbidden | escaped:
        if needle and needle in text:
            raise ValueError(
                "manifest contains an absolute local path; use repository-relative paths"
            )


def build_manifest(path: Path, provenance: Provenance) -> dict[str, Any]:
    """Build the manifest dict for the file at ``path``.

    The file hash and size are recomputed from disk and must agree with ``provenance``.
    """
    target = Path(path)
    if not target.is_file():
        raise FileNotFoundError(f"data file not found: {_display_path(target)}")
    _validate(provenance)
    actual_sha = file_sha256(target)
    actual_bytes = target.stat().st_size
    if actual_sha != provenance.file_sha256 or actual_bytes != provenance.file_bytes:
        raise ValueError(
            "file_sha256/file_bytes in provenance do not match the file on disk; "
            "recompute them with file_sha256() and stat()"
        )
    expected_query_sha = hashlib.sha256(provenance.query_text.encode()).hexdigest()
    if expected_query_sha != provenance.query_sha256:
        raise ValueError("query_sha256 does not match sha256 of query_text (UTF-8)")
    manifest: dict[str, Any] = {
        "schema_version": _SCHEMA_VERSION,
        "file": _display_path(target),
        **asdict(provenance),
    }
    _assert_no_local_paths(json.dumps(manifest, ensure_ascii=False))
    return manifest


def write_manifest(manifest: dict[str, Any], out_dir: Path) -> Path:
    """Write ``<out_dir>/<table_name>.manifest.json`` atomically; return its path."""
    table_name = str(manifest.get("table_name", ""))
    if not _TABLE_NAME_RE.fullmatch(table_name):
        raise ValueError(f"manifest has an invalid table_name: {table_name!r}")
    text = json.dumps(manifest, indent=2, ensure_ascii=False, sort_keys=False) + "\n"
    _assert_no_local_paths(text)
    directory = Path(out_dir)
    directory.mkdir(parents=True, exist_ok=True)
    final = directory / f"{table_name}.manifest.json"
    fd, tmp_name = tempfile.mkstemp(
        dir=directory, prefix=f".{table_name}.", suffix=".tmp"
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(text)
        os.replace(tmp_name, final)
    except BaseException:
        Path(tmp_name).unlink(missing_ok=True)
        raise
    return final
