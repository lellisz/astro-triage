"""Proofs for ``llm.secrets`` (ADR 0005 D3 section 5): key loading and redaction.

No real ``.env`` or environment variable is read: ``env`` is always injected and every file
lives in ``tmp_path``. The key is an obviously fake string.
"""

from pathlib import Path

import pytest

from astro_triage.llm.secrets import MissingKeyError, load_api_key, redact

FAKE = "FAKE-KEY-123"


def _dotenv(tmp_path: Path, raw: bytes) -> Path:
    path = tmp_path / ".env"
    path.write_bytes(raw)
    return path


def test_environment_wins_over_dotenv(tmp_path: Path) -> None:
    path = _dotenv(tmp_path, b"NVIDIA_API_KEY=FROM-DOTENV\n")
    assert load_api_key({"NVIDIA_API_KEY": FAKE}, path) == FAKE


def test_blank_environment_value_falls_back_to_dotenv(tmp_path: Path) -> None:
    path = _dotenv(tmp_path, b"NVIDIA_API_KEY=FROM-DOTENV\n")
    assert load_api_key({"NVIDIA_API_KEY": "   "}, path) == "FROM-DOTENV"


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        (b"\xef\xbb\xbfNVIDIA_API_KEY=FAKE-KEY-123\n", FAKE),  # UTF-8 BOM
        (b"NVIDIA_API_KEY=FAKE-KEY-123\r\nOTHER=1\r\n", FAKE),  # CRLF
        (b'NVIDIA_API_KEY="FAKE-KEY-123"\n', FAKE),  # double quotes
        (b"NVIDIA_API_KEY='FAKE-KEY-123'\n", FAKE),  # single quotes
        (b"export NVIDIA_API_KEY=FAKE-KEY-123\n", FAKE),
        (b"# c\nOTHER=x\n  NVIDIA_API_KEY = FAKE-KEY-123  \n", FAKE),
        (b"\xef\xbb\xbfNVIDIA_API_KEY='FAKE-KEY-123'\r\n", FAKE),  # all together
    ],
)
def test_dotenv_formats(tmp_path: Path, raw: bytes, expected: str) -> None:
    assert load_api_key({}, _dotenv(tmp_path, raw)) == expected


@pytest.mark.parametrize(
    "raw",
    [
        b"",
        b"OTHER=1\n",
        b"NVIDIA_API_KEY=\n",
        b'NVIDIA_API_KEY=""\n',
        b"XNVIDIA_API_KEY=FAKE-KEY-123\n",
    ],
)
def test_missing_key_raises_without_path(tmp_path: Path, raw: bytes) -> None:
    path = _dotenv(tmp_path, raw)
    with pytest.raises(MissingKeyError) as info:
        load_api_key({}, path)
    text = f"{info.value!s} {info.value!r}"
    assert str(tmp_path) not in text
    assert ".env" in text  # only the file name, never the directory


def test_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(MissingKeyError):
        load_api_key({}, tmp_path / "absent.env")


def test_injected_env_without_dotenv_never_touches_repo_dotenv() -> None:
    # env injected and no dotenv_path: the repository .env must not be consulted.
    with pytest.raises(MissingKeyError):
        load_api_key({})


def test_redact_replaces_literal_key() -> None:
    out = redact(f"request failed for {FAKE} at host", FAKE)
    assert FAKE not in out
    assert "***" in out


def test_redact_key_with_regex_metacharacters() -> None:
    key = "a.b*c+d(e)[f]{1}^$|?\\z"
    out = redact(f"boom {key} then {key}", key)
    assert key not in out
    assert out.count("***") == 2
    # A regex-based replace would treat '.' as a wildcard and eat this unrelated text.
    assert "aXb" in redact("aXb is harmless", key)


def test_redact_without_the_replace_would_leak() -> None:
    # Plain word: _sanitize alone cannot know it is a secret, only str.replace removes it.
    assert redact("the secret word is hunter2 ok", "hunter2") == (
        "the secret word is *** ok"
    )


def test_redact_none_key_and_limit() -> None:
    assert redact("plain", None) == "plain"
    assert len(redact("x" * 1000, FAKE, limit=50)) == 50
    assert "abc.def" not in redact("Authorization: Bearer abc.def", None)
    assert "abc.def" not in redact("x Bearer abc.def y", None)
