"""Offline tests for ``astro_triage.ingest.manifest``."""

from __future__ import annotations

import dataclasses
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import pytest

from astro_triage.ingest import manifest as m

MIB = 1024 * 1024
SOURCES = ("Gaia Archive TAP", "VizieR/CDS TAP", "CDS XMatch", "SDSS SAS")
QUERY = (
    "SELECT source_id, ra, dec\n"
    "FROM gaiadr3.gaia_source\n"
    "WHERE 1 = CONTAINS(POINT('ICRS', ra, dec),\n"
    "                   CIRCLE('ICRS', 10.0, 20.0, 0.5))\r\n"
    "  AND phot_g_mean_mag < 18\n"
)


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _prov(data: bytes, **overrides: Any) -> m.Provenance:
    base: dict[str, Any] = {
        "table_name": "gaia_test",
        "source": "Gaia Archive TAP",
        "release": "DR3",
        "obtained_utc": "2025-01-02T03:04:05+00:00",
        "query_text": QUERY,
        "query_sha256": _sha(QUERY.encode()),
        "row_count": 3,
        "file_sha256": _sha(data),
        "file_bytes": len(data),
        "terms": m.GAIA_TERMS,
        "citations": list(m.GAIA_CITATIONS),
        "acknowledgement": m.GAIA_ACKNOWLEDGEMENT,
    }
    base.update(overrides)
    return m.Provenance(**base)


@pytest.fixture
def data_file(tmp_path: Path) -> tuple[Path, bytes]:
    data = b"hello astro\n"
    path = tmp_path / "data.parquet"
    path.write_bytes(data)
    return path, data


# 1) file_sha256


@pytest.mark.parametrize(
    "size", [0, 1, MIB - 1, MIB, MIB + 1, int(2.5 * MIB)], ids=lambda s: f"{s}B"
)
def test_file_sha256_matches_hashlib(tmp_path: Path, size: int) -> None:
    data = bytes((i * 31 + 7) % 256 for i in range(size))
    path = tmp_path / "blob.bin"
    path.write_bytes(data)
    assert m.file_sha256(path) == hashlib.sha256(data).hexdigest()


def test_file_sha256_empty_is_known_constant(tmp_path: Path) -> None:
    path = tmp_path / "empty"
    path.write_bytes(b"")
    assert (
        m.file_sha256(path)
        == "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
    )


# 2) build_manifest


def test_build_manifest_fields_and_recomputed_values(
    data_file: tuple[Path, bytes],
) -> None:
    path, data = data_file
    out = m.build_manifest(path, _prov(data))
    assert out["schema_version"] == 1
    assert out["file"] == "data.parquet"
    assert out["table_name"] == "gaia_test"
    assert out["source"] == "Gaia Archive TAP"
    assert out["release"] == "DR3"
    assert out["row_count"] == 3
    assert out["query_sha256"] == _sha(QUERY.encode())
    assert out["file_sha256"] == _sha(b"hello astro\n")
    assert out["file_bytes"] == 12
    assert set(out["client_versions"]) >= {"python", "astroquery", "pyarrow"}
    for key in ("terms", "citations", "acknowledgement", "notes", "obtained_utc"):
        assert key in out


def test_build_manifest_rejects_wrong_declared_hash(
    data_file: tuple[Path, bytes],
) -> None:
    path, data = data_file
    with pytest.raises(ValueError, match="do not match"):
        m.build_manifest(path, _prov(data, file_sha256=_sha(b"other")))


def test_build_manifest_rejects_wrong_declared_size(
    data_file: tuple[Path, bytes],
) -> None:
    path, data = data_file
    with pytest.raises(ValueError, match="do not match"):
        m.build_manifest(path, _prov(data, file_bytes=len(data) + 1))


def test_build_manifest_rejects_wrong_query_sha(
    data_file: tuple[Path, bytes],
) -> None:
    path, data = data_file
    with pytest.raises(ValueError, match="query_sha256"):
        m.build_manifest(path, _prov(data, query_sha256=_sha(b"nope")))


def test_build_manifest_missing_file(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        m.build_manifest(tmp_path / "absent.parquet", _prov(b""))


@pytest.mark.parametrize("name", ["../x", "", "a/b", "a\\b", ".hidden", "x y", "a\n"])
def test_build_manifest_rejects_invalid_table_name(
    data_file: tuple[Path, bytes], name: str
) -> None:
    path, data = data_file
    with pytest.raises(ValueError, match="table_name"):
        m.build_manifest(path, _prov(data, table_name=name))


@pytest.mark.parametrize("stamp", ["yesterday", "", "2025-13-45", "01/02/2025"])
def test_build_manifest_rejects_non_iso_timestamp(
    data_file: tuple[Path, bytes], stamp: str
) -> None:
    path, data = data_file
    with pytest.raises(ValueError, match="ISO 8601"):
        m.build_manifest(path, _prov(data, obtained_utc=stamp))


def test_build_manifest_rejects_negative_row_count(
    data_file: tuple[Path, bytes],
) -> None:
    path, data = data_file
    with pytest.raises(ValueError, match="row_count"):
        m.build_manifest(path, _prov(data, row_count=-1))


def test_build_manifest_accepts_zero_row_count(data_file: tuple[Path, bytes]) -> None:
    path, data = data_file
    assert m.build_manifest(path, _prov(data, row_count=0))["row_count"] == 0


# 3) no personal paths


def test_file_inside_repo_is_relative(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo = tmp_path / "fakerepo"
    (repo / "data" / "raw").mkdir(parents=True)
    path = repo / "data" / "raw" / "t.parquet"
    path.write_bytes(b"abc")
    monkeypatch.setattr(m, "_REPO_ROOT", repo.resolve())
    out = m.build_manifest(path, _prov(b"abc"))
    assert out["file"] == "data/raw/t.parquet"


def test_file_outside_repo_only_name(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo = tmp_path / "fakerepo"
    repo.mkdir()
    elsewhere = tmp_path / "elsewhere" / "deep"
    elsewhere.mkdir(parents=True)
    path = elsewhere / "t.parquet"
    path.write_bytes(b"abc")
    monkeypatch.setattr(m, "_REPO_ROOT", repo.resolve())
    out = m.build_manifest(path, _prov(b"abc"))
    assert out["file"] == "t.parquet"


def _fake_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Path, Path]:
    repo = (tmp_path / "fake_repo_root").resolve()
    home = (tmp_path / "fake_home_dir").resolve()
    repo.mkdir()
    home.mkdir()
    monkeypatch.setattr(m, "_REPO_ROOT", repo)
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: home))
    return repo, home


@pytest.mark.parametrize("which", ["repo", "home"])
@pytest.mark.parametrize("style", ["native", "posix"])
def test_build_manifest_rejects_leaked_root_or_home(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, which: str, style: str
) -> None:
    repo, home = _fake_env(tmp_path, monkeypatch)
    leaked = repo if which == "repo" else home
    text = str(leaked) if style == "native" else leaked.as_posix()
    path = tmp_path / "t.parquet"
    path.write_bytes(b"abc")
    with pytest.raises(ValueError, match="absolute local path"):
        m.build_manifest(path, _prov(b"abc", notes=[f"saved at {text}/x"]))


def test_build_manifest_clean_with_fake_env_passes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _fake_env(tmp_path, monkeypatch)
    path = tmp_path / "t.parquet"
    path.write_bytes(b"abc")
    out = m.build_manifest(path, _prov(b"abc", notes=["nothing personal"]))
    assert out["file"] == "t.parquet"


def test_write_manifest_rejects_leaked_home(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _, home = _fake_env(tmp_path, monkeypatch)
    manifest = {"table_name": "t", "notes": [str(home)]}
    with pytest.raises(ValueError, match="absolute local path"):
        m.write_manifest(manifest, tmp_path / "out")
    assert not list((tmp_path / "out").glob("*"))


# 4) credentials and exact query


@pytest.mark.parametrize(
    "query",
    [
        "SELECT 1 -- password=abc",
        "SELECT 1 -- Bearer tok",
        "SELECT 1 WHERE x = 'a' -- API_KEY: k123",
        "SELECT 1 /* token = zzz */",
        "SELECT 1 -- Authorization: Basic abcdef",
    ],
)
def test_build_manifest_rejects_credentials(
    data_file: tuple[Path, bytes], query: str
) -> None:
    path, data = data_file
    prov = _prov(data, query_text=query, query_sha256=_sha(query.encode()))
    with pytest.raises(ValueError, match="credential"):
        m.build_manifest(path, prov)


def test_multiline_query_preserved_exactly(
    data_file: tuple[Path, bytes], tmp_path: Path
) -> None:
    path, data = data_file
    out = m.build_manifest(path, _prov(data))
    assert out["query_text"] == QUERY
    written = m.write_manifest(out, tmp_path / "out")
    loaded = json.loads(written.read_text(encoding="utf-8"))
    assert loaded["query_text"] == QUERY
    assert loaded["query_text"].encode() == QUERY.encode()
    assert "\r\n" in loaded["query_text"]
    assert (
        hashlib.sha256(loaded["query_text"].encode()).hexdigest()
        == (loaded["query_sha256"])
    )


def test_long_query_not_truncated(data_file: tuple[Path, bytes]) -> None:
    path, data = data_file
    query = "SELECT " + ", ".join(f"col{i}" for i in range(500)) + " FROM t"
    assert len(query) > 300
    out = m.build_manifest(
        path, _prov(data, query_text=query, query_sha256=_sha(query.encode()))
    )
    assert out["query_text"] == query


# 5) write_manifest


def _manifest(path: Path, data: bytes, **overrides: Any) -> dict[str, Any]:
    return m.build_manifest(path, _prov(data, **overrides))


def test_write_manifest_creates_readable_indented_json(
    data_file: tuple[Path, bytes], tmp_path: Path
) -> None:
    path, data = data_file
    manifest = _manifest(path, data)
    out_dir = tmp_path / "new" / "manifests"
    final = m.write_manifest(manifest, out_dir)
    assert final == out_dir / "gaia_test.manifest.json"
    text = final.read_text(encoding="utf-8")
    assert json.loads(text) == manifest
    assert text.startswith('{\n  "schema_version": 1,')
    assert text.endswith("}\n")
    assert [p.name for p in out_dir.iterdir()] == ["gaia_test.manifest.json"]


def test_write_manifest_overwrites_existing(
    data_file: tuple[Path, bytes], tmp_path: Path
) -> None:
    path, data = data_file
    out_dir = tmp_path / "out"
    m.write_manifest(_manifest(path, data, release="DR3"), out_dir)
    final = m.write_manifest(_manifest(path, data, release="DR4"), out_dir)
    assert json.loads(final.read_text(encoding="utf-8"))["release"] == "DR4"
    assert [p.name for p in out_dir.iterdir()] == ["gaia_test.manifest.json"]


def test_write_manifest_atomic_on_replace_failure(
    data_file: tuple[Path, bytes],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path, data = data_file
    out_dir = tmp_path / "out"
    final = m.write_manifest(_manifest(path, data, release="OLD"), out_dir)
    before = final.read_bytes()

    def boom(src: str, dst: str) -> None:
        raise OSError("simulated replace failure")

    monkeypatch.setattr(os, "replace", boom)
    with pytest.raises(OSError, match="simulated"):
        m.write_manifest(_manifest(path, data, release="NEW"), out_dir)
    monkeypatch.undo()

    assert final.read_bytes() == before
    assert [p.name for p in out_dir.iterdir()] == ["gaia_test.manifest.json"]
    assert not list(out_dir.glob("*.tmp"))


def test_write_manifest_failure_without_prior_leaves_nothing(
    data_file: tuple[Path, bytes],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path, data = data_file
    out_dir = tmp_path / "out"
    manifest = _manifest(path, data)

    def boom(src: str, dst: str) -> None:
        raise OSError("simulated replace failure")

    monkeypatch.setattr(os, "replace", boom)
    with pytest.raises(OSError):
        m.write_manifest(manifest, out_dir)
    monkeypatch.undo()
    assert list(out_dir.iterdir()) == []


@pytest.mark.parametrize("name", ["../x", "", "a/b"])
def test_write_manifest_rejects_bad_table_name(tmp_path: Path, name: str) -> None:
    with pytest.raises(ValueError, match="table_name"):
        m.write_manifest({"table_name": name}, tmp_path / "out")
    assert not (tmp_path / "x.manifest.json").exists()


def test_write_manifest_missing_table_name(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="table_name"):
        m.write_manifest({}, tmp_path / "out")


# 6) source constants

_SOURCE_CONSTANTS = {
    "Gaia Archive TAP": (m.GAIA_TERMS, m.GAIA_CITATIONS, m.GAIA_ACKNOWLEDGEMENT),
    "VizieR/CDS TAP": (m.VIZIER_TERMS, m.VIZIER_CITATIONS, m.VIZIER_ACKNOWLEDGEMENT),
    "CDS XMatch": (m.XMATCH_TERMS, m.XMATCH_CITATIONS, m.XMATCH_ACKNOWLEDGEMENT),
    "SDSS SAS": (m.SDSS_TERMS, m.SDSS_CITATIONS, m.SDSS_ACKNOWLEDGEMENT),
}
_MARKERS = ("to be verified", "copy verbatim")


@pytest.mark.parametrize("source", SOURCES)
def test_source_manifest_has_nonempty_terms_citations_ack(
    tmp_path: Path, source: str
) -> None:
    terms, citations, ack = _SOURCE_CONSTANTS[source]
    path = tmp_path / "f.bin"
    path.write_bytes(b"x")
    prov = _prov(
        b"x",
        source=source,
        terms=terms,
        citations=list(citations),
        acknowledgement=ack,
    )
    out = m.build_manifest(path, prov)
    assert out["source"] == source
    assert out["terms"].strip()
    assert out["acknowledgement"].strip()
    assert out["citations"]
    assert all(c.strip() for c in out["citations"])


def _has_marker(text: str) -> bool:
    return any(marker in text for marker in _MARKERS)


@pytest.mark.parametrize("source", ["Gaia Archive TAP", "VizieR/CDS TAP"])
def test_gaia_and_vizier_flag_pending_items(source: str) -> None:
    terms, citations, ack = _SOURCE_CONSTANTS[source]
    joined = " ".join([terms, ack, *citations])
    assert _has_marker(joined)
    assert _has_marker(ack) or _has_marker(terms)


def test_cds_constants_are_fully_pending() -> None:
    for const in (
        m.VIZIER_ACKNOWLEDGEMENT,
        m.XMATCH_TERMS,
        m.XMATCH_ACKNOWLEDGEMENT,
        *m.XMATCH_CITATIONS,
    ):
        assert "to be verified" in const


def test_sdss_ack_is_pending_and_not_invented() -> None:
    assert "copy verbatim" in m.SDSS_ACKNOWLEDGEMENT
    assert "https://www.sdss4.org/collaboration/citing-sdss/" in (
        m.SDSS_ACKNOWLEDGEMENT
    )
    # The real SDSS-IV boilerplate must not be fabricated in this module.
    lowered = m.SDSS_ACKNOWLEDGEMENT.lower()
    for invented in ("alfred p. sloan foundation", "funding for the sloan"):
        assert invented not in lowered
    assert len(m.SDSS_ACKNOWLEDGEMENT) < 200


def test_constants_survive_json_roundtrip(tmp_path: Path) -> None:
    terms, citations, ack = _SOURCE_CONSTANTS["SDSS SAS"]
    path = tmp_path / "f.bin"
    path.write_bytes(b"x")
    prov = _prov(
        b"x",
        source="SDSS SAS",
        terms=terms,
        citations=list(citations),
        acknowledgement=ack,
    )
    final = m.write_manifest(m.build_manifest(path, prov), tmp_path / "o")
    loaded = json.loads(final.read_text(encoding="utf-8"))
    assert loaded["acknowledgement"] == ack
    assert loaded["citations"] == list(citations)


def test_provenance_is_frozen(data_file: tuple[Path, bytes]) -> None:
    _, data = data_file
    prov = _prov(data)
    with pytest.raises(dataclasses.FrozenInstanceError):
        prov.row_count = 5  # type: ignore[misc]
