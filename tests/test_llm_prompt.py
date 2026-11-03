"""Proofs for ``llm.prompt`` (ADR 0005 D3, D4): leakage, MISSING handling, arms and parsing."""

import math
import re

import pytest

from astro_triage.llm.prompt import (
    FEATURE_COLUMNS,
    FORBIDDEN_SUBSTRINGS,
    PROMPT_VERSION,
    TEMPLATE_SHA256,
    Answer,
    ParseError,
    build_messages,
    parse_answer,
)

SOURCE_ID = 4295806720338829312
# Columns that must never reach the prompt even if present in the row mapping.
LEAKY = {
    "source_id": SOURCE_ID,
    "ra": 187.2779,
    "dec": 2.0524,
    "l": 293.1,
    "b": 64.4,
    "ra_error": 0.0321,
    "dec_error": 0.0287,
    "label": 2,
    "fold": 1,
    "sdss_class": 2,
}
L1 = {"p_cal_qso": 0.11, "p_cal_star": 0.62, "p_cal_galaxy": 0.27}
_BARE_TOKENS = re.compile(r"(?<!\w)(?:ra|dec|l|b|ra_error|dec_error)(?!\w)")


def make_row(**over: float | None) -> dict[str, float | None]:
    row: dict[str, float | None] = {n: 1.5 for n in FEATURE_COLUMNS}
    row["astrometric_params_solved"] = 31
    row.update(over)
    return row


def text_of(messages: list[dict[str, str]]) -> str:
    return "\n".join(m["content"] for m in messages)


@pytest.mark.parametrize("arm", ["blind", "with_l1"])
def test_prompt_does_not_leak_identity_or_position(arm: str) -> None:
    row = {**make_row(), **LEAKY}
    text = text_of(build_messages(row, arm, L1))  # type: ignore[arg-type]
    low = text.lower()
    for forbidden in FORBIDDEN_SUBSTRINGS:
        assert forbidden not in low, forbidden
    assert str(SOURCE_ID) not in text
    assert _BARE_TOKENS.findall(low) == []
    for value in ("187.2779", "293.1", "64.4"):
        assert value not in text


def test_bare_token_check_is_not_fooled_by_pmra_error() -> None:
    # pmra_error contains the substring ra_error but is a legitimate feature.
    text = text_of(build_messages(make_row(), "blind"))
    assert "pmra_error" in text
    assert _BARE_TOKENS.findall(text.lower()) == []
    assert _BARE_TOKENS.findall("ra_error = 1 dec = 2")  # the detector does detect


def test_all_21_features_printed_in_order() -> None:
    assert len(FEATURE_COLUMNS) == 21
    user = build_messages(make_row(), "blind")[1]["content"]
    names = [ln.split(" = ")[0] for ln in user.splitlines() if " = " in ln]
    assert names == list(FEATURE_COLUMNS)


def test_two_parameter_solution_has_nine_missing_and_fixed_sentence() -> None:
    absent = {
        n: None
        for n in (
            "parallax",
            "parallax_error",
            "parallax_over_error",
            "pm",
            "pmra",
            "pmra_error",
            "pmdec",
            "pmdec_error",
        )
    }
    row = make_row(astrometric_params_solved=3, ruwe=math.nan, **absent)
    user = build_messages(row, "blind")[1]["content"]
    assert user.count("MISSING (not measured)") == 9
    assert (
        "2-parameter astrometric solution: parallax, proper motion and RUWE "
        "were not measured." in user
    )
    assert "nan" not in user.lower().replace("not measured", "")
    assert "astrometric_params_solved = 3" in user


def test_five_or_six_parameter_solution_has_no_sentence() -> None:
    user = build_messages(make_row(), "blind")[1]["content"]
    assert "MISSING" not in user
    assert "2-parameter" not in user


def test_nothing_is_imputed_and_zero_is_not_missing() -> None:
    row = make_row(parallax=0.0, pmra=-0.0, ruwe=None)
    user = build_messages(row, "blind")[1]["content"]
    assert "parallax = 0 mas" in user
    assert "pmra = 0 mas/yr" in user  # -0.0 normalised
    assert "-0" not in user
    assert "ruwe = MISSING (not measured)" in user


def test_blind_arm_ignores_l1() -> None:
    with_probs = build_messages(make_row(), "blind", L1)
    assert with_probs == build_messages(make_row(), "blind")
    assert "p_cal" not in text_of(with_probs)
    assert "previous classifier" not in text_of(with_probs)


def test_with_l1_reads_probabilities_by_name_in_fixed_order() -> None:
    user = build_messages(make_row(), "with_l1", L1)[1]["content"]
    tail = user.splitlines()[-4:-1]
    # L1 dict was built qso, star, galaxy; printing must be by column name.
    assert tail == [
        "p_cal_star = 0.62",
        "p_cal_galaxy = 0.27",
        "p_cal_qso = 0.11",
    ]
    assert "Probabilities from a previous classifier:" in user


def test_with_l1_requires_probabilities() -> None:
    with pytest.raises(ValueError):
        build_messages(make_row(), "with_l1")
    with pytest.raises(KeyError):
        build_messages(make_row(), "with_l1", {"p_cal_star": 0.5})


def test_prompt_is_deterministic_and_template_hash_is_stable() -> None:
    assert build_messages(make_row(), "blind") == build_messages(make_row(), "blind")
    assert re.fullmatch(r"[0-9a-f]{64}", TEMPLATE_SHA256)
    assert PROMPT_VERSION == "p1"


OK = '{"label": "QSO", "confidence": 73}'


@pytest.mark.parametrize(
    ("content", "expected"),
    [
        (OK, Answer("QSO", 73)),
        (f"```json\n{OK}\n```", Answer("QSO", 73)),
        (f"```\n{OK}\n```", Answer("QSO", 73)),
        (f"  \n{OK}\n  ", Answer("QSO", 73)),
        ('{"confidence": 0, "label": "UNKNOWN"}', Answer("UNKNOWN", 0)),
        ('{"label": "STAR", "confidence": 100}', Answer("STAR", 100)),
    ],
)
def test_parse_answer_accepts(content: str, expected: Answer) -> None:
    assert parse_answer(content) == expected


@pytest.mark.parametrize(
    ("content", "status"),
    [
        ('{"label": "STAR", "confidence": 90, "why": "x"}', "SCHEMA_VIOLATION"),
        ('{"label": "STAR"}', "SCHEMA_VIOLATION"),
        ('{"label": "STAR", "confidence": 101}', "SCHEMA_VIOLATION"),
        ('{"label": "STAR", "confidence": -1}', "SCHEMA_VIOLATION"),
        ('{"label": "star", "confidence": 50}', "SCHEMA_VIOLATION"),
        ('{"label": "", "confidence": 50}', "SCHEMA_VIOLATION"),
        ('{"label": null, "confidence": 50}', "SCHEMA_VIOLATION"),
        ('{"label": "STAR", "confidence": true}', "SCHEMA_VIOLATION"),
        ('{"label": "STAR", "confidence": 90.0}', "SCHEMA_VIOLATION"),
        ('{"label": "STAR", "confidence": "90"}', "SCHEMA_VIOLATION"),
        ('{"label": "STAR", "confidence": NaN}', "SCHEMA_VIOLATION"),
        ('["STAR", 90]', "SCHEMA_VIOLATION"),
        ("42", "SCHEMA_VIOLATION"),
        ("{", "INVALID_JSON"),
        (f"Sure! {OK}", "INVALID_JSON"),
        ("```json\n{\n```", "INVALID_JSON"),
        ("", "EMPTY_CONTENT"),
        ("   \n\t", "EMPTY_CONTENT"),
        (None, "EMPTY_CONTENT"),
    ],
)
def test_parse_answer_rejects_with_right_status(
    content: str | None, status: str
) -> None:
    with pytest.raises(ParseError) as info:
        parse_answer(content)
    assert info.value.status == status


def test_parse_answer_deeply_nested_input_is_invalid_json_not_a_crash() -> None:
    with pytest.raises(ParseError) as info:
        parse_answer("[" * 100000)
    assert info.value.status == "INVALID_JSON"
