"""Proofs for ``llm.runner`` and ``scripts/layer2_run.py`` (ADR 0005 D4-D6): synthetic release,
fake opener, fake clock, no network, no real sleep, no real key."""

from __future__ import annotations

import hashlib
import importlib.util
import io
import json
import logging
import os
import random
import subprocess
import sys
import urllib.error
from datetime import UTC, datetime
from email.message import Message
from pathlib import Path
from typing import Any, Self

import numpy as np
import pandas as pd
import pytest

from astro_triage.cost.quota import Ledger, QuotaGuard
from astro_triage.llm.cache import ResultCache
from astro_triage.llm.client import CallResult, FatalCallError, chat
from astro_triage.llm.prompt import FEATURE_COLUMNS
from astro_triage.llm.routing import (
    MANIFEST_NAME,
    ManifestMismatchError,
    request_for,
    select_routed,
)
from astro_triage.llm.runner import _classify, run

KEY = "FAKE-KEY-123"
MODEL = "nvidia/fake-model"
REPO = Path(__file__).resolve().parents[1]
TAU = 0.6
FIXED_NOW = datetime(2026, 10, 27, 12, 0, 0, tzinfo=UTC)

VAL_IDS = [4_000_000_000_000 + i for i in range(30)]
TEST_IDS = [4_000_000_000_000 + i for i in range(30, 40)]
TRAIN_IDS = [4_000_000_000_000 + i for i in range(40, 50)]
VAL_ROUTED = [s for i, s in enumerate(VAL_IDS) if i % 5 != 0]  # 24 routed
TEST_ROUTED = [
    TEST_IDS[1],
    TEST_IDS[2],
    TEST_IDS[3],
    TEST_IDS[6],
    TEST_IDS[7],
    TEST_IDS[8],
]


def make_release(base: Path) -> Path:
    """Synthetic release ``base/rel``: 30 val (24 routed), 10 test (6 routed), 10 train."""
    rel = base / "rel"
    (rel / "layer2").mkdir(parents=True)
    rng = np.random.default_rng(7)
    ids = VAL_IDS + TEST_IDS + TRAIN_IDS
    fold = ["val"] * 30 + ["test"] * 10 + ["train"] * 10
    pd.DataFrame({"source_id": ids, "fold": fold}).astype(
        {"source_id": "int64"}
    ).to_parquet(rel / "labels.parquet")
    feats = pd.DataFrame(
        {c: rng.normal(size=50).astype("float32") for c in FEATURE_COLUMNS}
    )
    feats.loc[3, "bp_rp"] = np.nan
    feats.insert(0, "source_id", np.array(ids, dtype="int64"))
    feats["ra"] = 1.0
    feats.to_parquet(rel / "features.parquet")
    routed = set(VAL_ROUTED) | set(TEST_ROUTED) | set(TRAIN_IDS)
    rows = []
    for sid in ids:
        if sid in routed:
            rows.append((sid, 0.5, 0.3, 0.2))
        else:
            rows.append((sid, 0.9, 0.05, 0.05))
    pred = pd.DataFrame(
        rows, columns=["source_id", "p_cal_star", "p_cal_qso", "p_cal_galaxy"]
    )
    pred = pred.sample(frac=1.0, random_state=3).reset_index(drop=True)  # shuffled
    pred = pred[["p_cal_galaxy", "source_id", "p_cal_star", "p_cal_qso"]]
    pred.to_parquet(rel / "predictions.parquet")
    write_manifest(rel)
    return rel


def write_manifest(rel: Path, sha: str | None = None) -> None:
    digest = (
        sha or hashlib.sha256((rel / "predictions.parquet").read_bytes()).hexdigest()
    )
    (rel / "layer2" / MANIFEST_NAME).write_text(
        json.dumps(
            {
                "predictions_file": "predictions.parquet",
                "predictions_sha256": digest,
                "tau": TAU,
            }
        ),
        encoding="utf-8",
    )


class Resp:
    def __init__(self, body: bytes, status: int = 200) -> None:
        self.status = status
        self._body = body

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *exc: object) -> None:
        return None

    def read(self, n: int = -1) -> bytes:
        return self._body if n < 0 else self._body[:n]


def answer(
    label: str = "STAR",
    finish: str | None = "stop",
    content: str | None = None,
) -> Resp:
    text = (
        content
        if content is not None
        else json.dumps({"label": label, "confidence": 80})
    )
    choice: dict[str, Any] = {"message": {"content": text}}
    if finish is not None:
        choice["finish_reason"] = finish
    body = {
        "model": MODEL,
        "choices": [choice],
        "usage": {"prompt_tokens": 100, "completion_tokens": 7},
    }
    return Resp(json.dumps(body).encode())


def http_error(code: int) -> urllib.error.HTTPError:
    body = io.BytesIO(f'{{"error": "bad key {KEY}"}}'.encode())
    return urllib.error.HTTPError("https://x/y", code, "msg " + KEY, Message(), body)


class Opener:
    """Plays a script, one item per HTTP attempt; the last item repeats forever."""

    def __init__(self, *script: object) -> None:
        self.script = list(script) or [answer()]
        self.calls = 0

    def __call__(self, request: Any, timeout: float) -> Any:
        item = self.script[min(self.calls, len(self.script) - 1)]
        self.calls += 1
        if isinstance(item, BaseException):
            raise item
        return item


def guard(max_calls: int = 10_000) -> QuotaGuard:
    return QuotaGuard(
        rpm_cap=6000.0,
        max_calls=max_calls,
        max_tokens_total=10**9,
        clock=lambda: 0.0,
        sleep=lambda s: None,
    )


def go(
    rel: Path,
    fold: str,
    opener: Opener,
    *,
    max_calls: int = 10_000,
    canary: bool = False,
    arm: str = "blind",
    sample: int | None = None,
    cache_dir: Path | None = None,
    key: str = KEY,
) -> dict[str, Any]:
    cdir = cache_dir or rel.parent / "cache"
    with ResultCache(cdir / "c.jsonl") as cache:
        return run(
            rel,
            fold,  # type: ignore[arg-type]
            MODEL,
            arm,
            False,
            guard(max_calls),
            cache,
            Ledger(rel / "layer2" / "calls.jsonl"),
            key,
            sample=sample,
            canary=canary,
            run_id="r1",
            opener=opener,
            sleep=lambda s: None,
            rng=random.Random(1),
            now=lambda: FIXED_NOW,
        )


def read_cache(rel: Path) -> list[dict[str, Any]]:
    path = rel.parent / "cache" / "c.jsonl"
    return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x]


def read_ledger(rel: Path) -> list[dict[str, Any]]:
    path = rel / "layer2" / "calls.jsonl"
    return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x]


# ---------------------------------------------------------------- manifest / routing


def test_routed_selection_uses_fold_and_tau(tmp_path: Path) -> None:
    rel = make_release(tmp_path)
    val = select_routed(rel, "val", rel / "layer2" / MANIFEST_NAME, None)
    test = select_routed(rel, "test", rel / "layer2" / MANIFEST_NAME, None)
    assert val["source_id"].tolist() == sorted(VAL_ROUTED)
    assert test["source_id"].tolist() == sorted(TEST_ROUTED)  # train rows never leak in
    assert (val["l1_conf"] < TAU).all()


@pytest.mark.parametrize("fold", ["test", "val"])
def test_fold_refused_when_predictions_hash_does_not_match(
    tmp_path: Path, fold: str
) -> None:
    rel = make_release(tmp_path)
    pred = pd.read_parquet(rel / "predictions.parquet")
    pred.loc[0, "p_cal_star"] = (
        0.1234  # predictions changed after the manifest froze tau
    )
    pred.to_parquet(rel / "predictions.parquet")
    opener = Opener()
    with pytest.raises(ManifestMismatchError):
        go(rel, fold, opener)
    assert opener.calls == 0


@pytest.mark.parametrize("fold", ["test", "val"])
def test_fold_refused_when_manifest_sha_is_wrong(tmp_path: Path, fold: str) -> None:
    rel = make_release(tmp_path)
    write_manifest(rel, sha="0" * 64)
    opener = Opener()
    with pytest.raises(ManifestMismatchError):
        go(rel, fold, opener)
    assert opener.calls == 0


# ---------------------------------------------------------------- _classify


def _res(
    status: str = "OK", finish: str | None = "stop", content: str | None = None
) -> CallResult:
    c = '{"label":"QSO","confidence":42}' if content is None else content
    return CallResult(status, 200, c, finish, MODEL, 10, 5, 1, 0, 1)


def test_classify_stop_ok() -> None:
    assert _classify(_res()) == ("OK", "QSO", 42)


def test_classify_unknown_is_answered() -> None:
    assert _classify(_res(content='{"label":"UNKNOWN","confidence":10}')) == (
        "UNKNOWN",
        "UNKNOWN",
        10,
    )


def test_classify_length_is_truncated_without_label() -> None:
    assert _classify(_res(status="TRUNCATED", finish="length")) == (
        "TRUNCATED",
        None,
        None,
    )


@pytest.mark.parametrize("finish", ["content_filter", None, "tool_calls", ""])
def test_classify_finish_reason_outside_stop_length_is_schema_violation(
    finish: str | None,
) -> None:
    assert _classify(_res(finish=finish)) == ("SCHEMA_VIOLATION", None, None)


def test_classify_bad_content_uses_parse_status() -> None:
    assert _classify(_res(content="nope"))[0] == "INVALID_JSON"
    assert _classify(_res(content=""))[0] == "EMPTY_CONTENT"
    assert (
        _classify(_res(content='{"label":"CAT","confidence":1}'))[0]
        == "SCHEMA_VIOLATION"
    )


def test_classify_passes_transport_status_through() -> None:
    res = CallResult("RATE_LIMITED", 429, None, None, None, None, None, 1, 0, 4)
    assert _classify(res) == ("RATE_LIMITED", None, None)


# ---------------------------------------------------------------- run: statuses end to end


@pytest.mark.parametrize(
    ("resp", "status"),
    [
        (answer(finish="length"), "TRUNCATED"),
        (answer(finish="content_filter"), "SCHEMA_VIOLATION"),
        (answer(finish=None), "SCHEMA_VIOLATION"),
        (answer(content="not json"), "INVALID_JSON"),
        (answer(content=""), "EMPTY_CONTENT"),
    ],
)
def test_terminal_statuses_are_cached_and_not_retried(
    tmp_path: Path, resp: Resp, status: str
) -> None:
    rel = make_release(tmp_path)
    opener = Opener(resp)
    rep = go(rel, "test", opener)
    assert rep["stop_reason"] is None
    assert rep["status_counts"] == {status: 6}
    assert rep["pending"] == 0
    assert opener.calls == 6  # deterministic outcomes never retried
    assert {r["status"] for r in read_cache(rel)} == {status}
    assert all(r["label_l2"] is None for r in read_cache(rel))


def test_429_is_retryable_not_terminal_and_resumed_next_run(tmp_path: Path) -> None:
    rel = make_release(tmp_path)
    rep = go(rel, "test", Opener(http_error(429)))
    # 4 attempts per object (1 + 3 retries); stops after 3 consecutive retryable objects
    assert rep["stop_reason"] == "CONSECUTIVE_RETRYABLE"
    assert rep["status_counts"] == {"RATE_LIMITED": 3}
    assert rep["pending"] == 6  # nobody is terminal: all stay in the denominator
    assert all(r["status"] == "RATE_LIMITED" for r in read_cache(rel))
    assert len(read_ledger(rel)) == 12
    good = Opener(answer("GALAXY"))
    rep2 = go(rel, "test", good)
    assert rep2["stop_reason"] is None
    assert rep2["pending"] == 0
    assert good.calls == 6  # the non-terminal records were retried


def test_two_retryable_then_success_resets_the_consecutive_counter(
    tmp_path: Path,
) -> None:
    rel = make_release(tmp_path)
    script: list[object] = []
    for _ in range(2):  # objects 1-2 fail for all 4 attempts
        script += [http_error(503)] * 4
    script += [answer()]  # then everything is fine
    rep = go(rel, "test", Opener(*script))
    assert rep["stop_reason"] is None
    assert rep["status_counts"] == {"OK": 4, "SERVER_ERROR": 2}
    assert rep["pending"] == 2


@pytest.mark.parametrize(
    ("code", "stop"),
    [
        (401, "AUTH_ERROR"),
        (403, "AUTH_ERROR"),
        (404, "MODEL_UNAVAILABLE"),
        (400, "BAD_REQUEST"),
    ],
)
def test_fatal_http_status_stops_after_ledger_record(
    tmp_path: Path, code: int, stop: str
) -> None:
    rel = make_release(tmp_path)
    opener = Opener(http_error(code))
    rep = go(rel, "test", opener)
    assert rep["stop_reason"] == stop
    assert opener.calls == 1  # no retry, no second object
    led = read_ledger(rel)
    assert len(led) == 1
    assert led[0]["http_status"] == code
    assert led[0]["status"] == stop
    assert rep["pending"] == 6
    # the fatal marker is not terminal: it can never be consolidated as a result
    assert read_cache(rel)[0]["status"] == stop


def test_chat_raises_fatal_after_after_attempt() -> None:
    seen: list[str] = []
    with pytest.raises(FatalCallError) as err:
        chat(
            [{"role": "user", "content": "x"}],
            MODEL,
            key=KEY,
            enable_thinking=False,
            max_tokens=64,
            before_attempt=lambda: None,
            after_attempt=lambda r: seen.append(r.status),
            opener=Opener(http_error(401)),
            sleep=lambda s: None,
        )
    assert seen == ["AUTH_ERROR"]
    assert err.value.status == "AUTH_ERROR"


def test_model_mismatch_is_counted(tmp_path: Path) -> None:
    rel = make_release(tmp_path)
    body = json.loads(answer()._body)
    body["model"] = "other/model"
    rep = go(rel, "test", Opener(Resp(json.dumps(body).encode())))
    assert rep["model_mismatch"] == 6


def test_sample_runs_subset_deterministically(tmp_path: Path) -> None:
    rel = make_release(tmp_path)
    a = go(rel, "val", Opener(), sample=5)
    assert a["routed"] == 5 and a["calls_made"] == 5
    rel2 = make_release(tmp_path / "b")
    go(rel2, "val", Opener(), sample=5)
    ids_a = sorted(r["source_id"] for r in read_cache(rel))
    ids_b = sorted(r["source_id"] for r in read_cache(rel2))
    assert ids_a == ids_b


# ---------------------------------------------------------------- canary


def _fill_val(rel: Path) -> None:
    rep = go(rel, "val", Opener(answer("STAR")))
    assert rep["pending"] == 0


def test_canary_skipped_with_fewer_than_20_cached(tmp_path: Path) -> None:
    rel = make_release(tmp_path)
    rep = go(rel, "val", Opener(), canary=True, sample=10)
    assert rep["canary"] == {"status": "skipped", "cached_val": 0}
    assert rep["stop_reason"] is None


def test_canary_all_agree_no_drift_and_makes_20_calls(tmp_path: Path) -> None:
    rel = make_release(tmp_path)
    _fill_val(rel)
    opener = Opener(answer("STAR"))
    rep = go(rel, "val", opener, canary=True)
    # 17.5.4: status "pass" (not "done"), with the three-way tally
    assert rep["canary"] == {
        "status": "pass",
        "agree": 20,
        "disagree": 0,
        "inconclusive": 0,
        "n": 20,
    }
    assert rep["stop_reason"] is None
    assert rep["calls_made"] == 20 and opener.calls == 20
    assert rep["cached_skipped"] == 24


@pytest.mark.parametrize(("disagree", "drift"), [(2, False), (3, True), (20, True)])
def test_canary_drift_threshold_is_18_of_20(
    tmp_path: Path, disagree: int, drift: bool
) -> None:
    rel = make_release(tmp_path)
    _fill_val(rel)
    script = [answer("GALAXY")] * disagree + [answer("STAR")]
    rep = go(rel, "val", Opener(*script), canary=True)
    assert rep["canary"]["agree"] == 20 - disagree
    assert rep["stop_reason"] == ("DRIFT" if drift else None)


def test_canary_drift_stops_before_the_main_loop(tmp_path: Path) -> None:
    rel = make_release(tmp_path)
    _fill_val(rel)
    # one routed object loses its cache record: it must not be called after DRIFT
    path = rel.parent / "cache" / "c.jsonl"
    lines = path.read_text(encoding="utf-8").splitlines()
    path.write_text("\n".join(lines[:-1]) + "\n", encoding="utf-8")
    opener = Opener(answer("GALAXY"))
    rep = go(rel, "val", opener, canary=True)
    assert rep["stop_reason"] == "DRIFT"
    assert opener.calls == 20  # canary only
    assert rep["pending"] == 1


def test_canary_unusable_answers_count_as_disagreement(tmp_path: Path) -> None:
    rel = make_release(tmp_path)
    _fill_val(rel)
    rep = go(rel, "val", Opener(answer(finish="length")), canary=True)
    assert rep["canary"]["agree"] == 0
    assert rep["stop_reason"] == "DRIFT"


def test_canary_single_transient_429_is_absorbed_by_retry(tmp_path: Path) -> None:
    rel = make_release(tmp_path)
    _fill_val(rel)
    rep = go(rel, "val", Opener(http_error(429), answer("STAR")), canary=True)
    assert rep["canary"]["agree"] == 20
    assert rep["stop_reason"] is None


def test_canary_does_not_write_to_the_result_cache(tmp_path: Path) -> None:
    rel = make_release(tmp_path)
    _fill_val(rel)
    before = (rel.parent / "cache" / "c.jsonl").read_bytes()
    go(rel, "val", Opener(answer("GALAXY")), canary=True)
    assert (rel.parent / "cache" / "c.jsonl").read_bytes() == before


def test_canary_on_test_fold_uses_val_cache(tmp_path: Path) -> None:
    rel = make_release(tmp_path)
    _fill_val(rel)
    rep = go(rel, "test", Opener(answer("STAR")), canary=True)
    assert rep["canary"]["status"] == "pass"  # 17.5.4
    assert rep["calls_made"] == 20 + 6


def test_canary_sustained_rate_limit_on_three_calls_is_inconclusive_not_drift(
    tmp_path: Path,
) -> None:
    """17.5.2-3: 3 canary calls that exhaust retries on 429 are transport failures (outside the
    denominator): 17 agreements < CANARY_MIN_AGREE -> CANARY_INCONCLUSIVE, never DRIFT."""
    rel = make_release(tmp_path)
    _fill_val(rel)
    script = [http_error(429)] * 12 + [answer("STAR")]
    rep = go(rel, "val", Opener(*script), canary=True)
    assert rep["canary"]["agree"] == 17
    assert rep["canary"]["inconclusive"] == 3 and rep["canary"]["disagree"] == 0
    assert rep["canary"]["status"] == "inconclusive"
    assert rep["stop_reason"] == "CANARY_INCONCLUSIVE"


# ---------------------------------------------------------------- secrets


def _all_text(root: Path) -> str:
    chunks = []
    for p in root.rglob("*"):
        if p.is_file() and not p.name.endswith(".lock"):
            chunks.append(p.read_bytes().decode("latin-1"))
    return "\n".join(chunks)


def test_key_never_reaches_ledger_cache_or_logs(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    rel = make_release(tmp_path)
    caplog.set_level(logging.DEBUG)
    go(rel, "test", Opener(http_error(429), answer(), http_error(500), answer()))
    go(rel, "val", Opener(http_error(401)))  # fatal path, error body contains the key
    assert KEY not in _all_text(tmp_path)
    assert KEY not in caplog.text
    assert "nvapi-" not in _all_text(tmp_path)


def test_key_not_in_run_report(tmp_path: Path) -> None:
    rel = make_release(tmp_path)
    rep = go(rel, "test", Opener(http_error(401)))
    assert KEY not in json.dumps(rep)


# ---------------------------------------------------------------- request identity


def test_cache_key_changes_with_arm_and_thinking(tmp_path: Path) -> None:
    rel = make_release(tmp_path)
    row = select_routed(rel, "test", rel / "layer2" / MANIFEST_NAME, None).iloc[0]
    keys = {
        request_for(row, MODEL, "blind", False)[1],
        request_for(row, MODEL, "with_l1", False)[1],
        request_for(row, MODEL, "blind", True)[1],
        request_for(row, "other", "blind", False)[1],
    }
    assert len(keys) == 4
    with pytest.raises(ValueError):
        request_for(row, MODEL, "bogus", False)


# ---------------------------------------------------------------- CLI


def _env() -> dict[str, str]:
    env = {k: v for k, v in os.environ.items() if k != "NVIDIA_API_KEY"}
    env["NVIDIA_API_KEY"] = KEY
    return env


def _cli(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(REPO / "scripts" / "layer2_run.py"), *args],
        capture_output=True,
        text=True,
        env=_env(),
        timeout=120,
        check=False,
    )


def test_cli_help_exits_zero() -> None:
    done = _cli("--help")
    assert done.returncode == 0
    assert "--max-calls" in done.stdout and "--max-tokens-total" in done.stdout


def test_cli_requires_max_calls() -> None:
    done = _cli(
        "--release", "x", "--fold", "val", "--model", MODEL, "--max-tokens-total", "100"
    )
    assert done.returncode == 2
    assert "--max-calls" in done.stderr


def test_cli_requires_max_tokens_total() -> None:
    done = _cli("--release", "x", "--fold", "val", "--model", MODEL, "--max-calls", "5")
    assert done.returncode == 2
    assert "--max-tokens-total" in done.stderr


def test_cli_rejects_non_positive_sample() -> None:
    done = _cli(
        "--release", "x", "--fold", "val", "--model", MODEL,
        "--max-calls", "5", "--max-tokens-total", "100", "--sample", "0",
    )  # fmt: skip
    assert done.returncode == 2


def _load_cli() -> Any:
    spec = importlib.util.spec_from_file_location(
        "layer2_run_under_test", REPO / "scripts" / "layer2_run.py"
    )
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.mark.parametrize(
    ("stop", "code"),
    [
        ("QUOTA_STOP", 3),
        ("AUTH_ERROR", 2),
        ("MODEL_UNAVAILABLE", 2),
        ("BAD_REQUEST", 2),
        ("DRIFT", 2),
        ("CONSECUTIVE_RETRYABLE", 4),  # 17.3: transient, moved from 2 to 4
        ("CANARY_INCONCLUSIVE", 4),
    ],
)
def test_cli_main_maps_stop_reason_to_exit_code(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    stop: str,
    code: int,
) -> None:
    mod = _load_cli()
    monkeypatch.setattr(mod, "load_api_key", lambda: KEY)
    monkeypatch.setattr(
        mod, "run", lambda *a, **k: {"stop_reason": stop, "pending": 1, "run_id": "r"}
    )
    rc = mod.main(
        [
            "--release", "rel", "--fold", "val", "--model", MODEL,
            "--max-calls", "5", "--max-tokens-total", "100",
            "--base", str(tmp_path), "--cache-dir", str(tmp_path / "cache"),
        ]
    )  # fmt: skip
    assert rc == code
    out = capsys.readouterr()
    assert KEY not in out.out + out.err
    assert not list((tmp_path / "cache").rglob("*.lock"))  # lock released


def test_cli_main_missing_key_exits_1_without_leaking(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    mod = _load_cli()
    from astro_triage.llm.secrets import MissingKeyError

    def boom() -> str:
        raise MissingKeyError("NVIDIA_API_KEY is not set")

    monkeypatch.setattr(mod, "load_api_key", boom)
    rc = mod.main(
        [
            "--release", "rel", "--fold", "val", "--model", MODEL,
            "--max-calls", "5", "--max-tokens-total", "100",
            "--base", str(tmp_path), "--cache-dir", str(tmp_path / "cache"),
        ]
    )  # fmt: skip
    assert rc == 1
    assert "MissingKeyError" in capsys.readouterr().err


# ------------------------------------------------- Amendment 1: canary tally and exit codes
def _canary_rel(tmp_path: Path) -> Path:
    """Val fully cached, then one record dropped: a pending object proves the main loop never ran."""
    rel = make_release(tmp_path)
    _fill_val(rel)
    path = rel.parent / "cache" / "c.jsonl"
    lines = path.read_text(encoding="utf-8").splitlines()
    path.write_text("\n".join(lines[:-1]) + "\n", encoding="utf-8")
    return rel


def _timeouts(k: int) -> list[object]:
    return [TimeoutError("timed out")] * (
        4 * k
    )  # 4 attempts per call, all transport failures


@pytest.mark.parametrize(
    ("timeouts", "disagree", "status", "stop"),
    [
        (2, 0, "pass", None),  # 18 conclusive agreements: pass (17.5 acceptance)
        (0, 2, "pass", None),  # <= 2 disagreements pass
        (1, 2, "pass", None),  # 19 conclusive, 2 disagree
        (3, 0, "inconclusive", "CANARY_INCONCLUSIVE"),  # 17 conclusive < 18
        (
            3,
            1,
            "inconclusive",
            "CANARY_INCONCLUSIVE",
        ),  # 16 agree + 1 disagree = 17 < 18
        (0, 3, "drift", "DRIFT"),  # >= 3 disagreements
        (3, 3, "drift", "DRIFT"),  # DRIFT is decided before INCONCLUSIVE
    ],
)
def test_canary_decision_table(
    tmp_path: Path, timeouts: int, disagree: int, status: str, stop: str | None
) -> None:
    rel = _canary_rel(tmp_path)
    script = _timeouts(timeouts) + [answer("GALAXY")] * disagree + [answer("STAR")]
    opener = Opener(*script)
    rep = go(rel, "val", opener, canary=True)
    c = rep["canary"]
    assert (c["inconclusive"], c["disagree"], c["n"]) == (timeouts, disagree, 20)
    assert c["agree"] == 20 - timeouts - disagree
    assert c["status"] == status and rep["stop_reason"] == stop
    if stop is not None:
        assert rep["pending"] == 1 and opener.calls == 4 * timeouts + (20 - timeouts)
    else:
        assert rep["pending"] == 0  # the main loop ran and finished the dropped object


def test_canary_transport_failure_is_outside_the_denominator_even_for_5xx(
    tmp_path: Path,
) -> None:
    rel = _canary_rel(tmp_path)
    script = [http_error(503)] * 12 + [answer("STAR")]
    rep = go(rel, "val", Opener(*script), canary=True)
    assert rep["canary"]["inconclusive"] == 3 and rep["canary"]["disagree"] == 0
    assert rep["stop_reason"] == "CANARY_INCONCLUSIVE"


def test_full_cache_rerun_zero_calls_without_canary_and_20_with(tmp_path: Path) -> None:
    rel = make_release(tmp_path)
    _fill_val(rel)
    quiet = Opener(answer("GALAXY"))
    rep = go(rel, "val", quiet, canary=False)
    assert (
        rep["calls_made"] == 0
        and quiet.calls == 0
        and rep["canary"] == {"status": "disabled"}
    )
    loud = Opener(answer("STAR"))
    assert go(rel, "val", loud, canary=True)["calls_made"] == 20 and loud.calls == 20


def test_ledger_lines_carry_enable_thinking_and_allowlist_still_rejects_others(
    tmp_path: Path,
) -> None:
    rel = make_release(tmp_path)
    go(rel, "test", Opener(answer("STAR")))
    lines = read_ledger(rel)
    assert lines and all(x["enable_thinking"] is False for x in lines)
    led = Ledger(tmp_path / "x" / "l.jsonl")
    base = {k: v for k, v in lines[0].items()}
    led.append({**base, "enable_thinking": True})
    with pytest.raises(ValueError, match="allowlist"):
        led.append({**base, "surprise": 1})


def _cli_run(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    opener: Opener,
    *extra: str,
) -> tuple[int, Path]:
    """layer2_run.main over a synthetic release; only the HTTP opener is replaced."""
    mod = _load_cli()
    rel = make_release(tmp_path)
    monkeypatch.setattr(mod, "load_api_key", lambda: KEY)
    real = mod.run
    monkeypatch.setattr(
        mod,
        "run",
        lambda *a, **k: real(
            *a, **k, opener=opener, sleep=lambda s: None, rng=random.Random(1)
        ),
    )
    rc = mod.main(
        [
            "--release", "rel", "--fold", "test", "--model", MODEL, "--no-canary",
            "--max-calls", "500", "--max-tokens-total", "1000000", "--rpm-cap", "60000",
            "--base", str(tmp_path), "--cache-dir", str(tmp_path / "cache"), *extra,
        ]
    )  # fmt: skip
    return rc, rel


def _parquets(rel: Path) -> list[Path]:
    return list((rel / "layer2").glob("predictions_l2_*.parquet"))


def test_exit_4_three_consecutive_503(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    rc, rel = _cli_run(tmp_path, monkeypatch, Opener(http_error(503)))
    assert rc == 4
    assert (
        json.loads(capsys.readouterr().out.split("not consolidated")[0])["stop_reason"]
        == "CONSECUTIVE_RETRYABLE"
    )
    assert not _parquets(rel)


def test_exit_4_one_object_ending_timeout_among_terminal_others_no_parquet(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    script = [
        *_timeouts(1),
        answer("STAR"),
    ]  # first routed object times out, rest answer
    rc, rel = _cli_run(tmp_path, monkeypatch, Opener(*script))
    assert rc == 4
    assert not _parquets(rel)


def test_exit_2_for_404_and_exit_0_for_a_complete_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    rc, rel = _cli_run(tmp_path / "a", monkeypatch, Opener(http_error(404)))
    assert rc == 2 and not _parquets(rel)
    rc, rel = _cli_run(tmp_path / "b", monkeypatch, Opener(answer("STAR")))
    assert rc == 0
    assert [p.name for p in _parquets(rel)] and "_think-off_test" in _parquets(rel)[
        0
    ].name


def test_exit_0_for_a_complete_sample_run_with_no_parquet(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    rc, rel = _cli_run(tmp_path, monkeypatch, Opener(answer("STAR")), "--sample", "3")
    assert rc == 0 and not _parquets(rel)


def test_exit_1_missing_key_writes_no_ledger_line(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    mod = _load_cli()
    rel = make_release(tmp_path)
    from astro_triage.llm.secrets import MissingKeyError

    def boom() -> str:
        raise MissingKeyError("NVIDIA_API_KEY is not set")

    monkeypatch.setattr(mod, "load_api_key", boom)
    rc = mod.main(
        [
            "--release", "rel", "--fold", "test", "--model", MODEL,
            "--max-calls", "5", "--max-tokens-total", "100",
            "--base", str(tmp_path), "--cache-dir", str(tmp_path / "cache"),
        ]
    )  # fmt: skip
    assert rc == 1
    assert not list((rel / "layer2").glob("calls_*.jsonl"))


def test_exit_3_when_the_call_cap_is_reached(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    mod = _load_cli()
    rel = make_release(tmp_path)
    monkeypatch.setattr(mod, "load_api_key", lambda: KEY)
    real = mod.run
    monkeypatch.setattr(
        mod,
        "run",
        lambda *a, **k: real(
            *a, **k, opener=Opener(answer("STAR")), sleep=lambda s: None
        ),
    )
    rc = mod.main(
        [
            "--release", "rel", "--fold", "test", "--model", MODEL, "--no-canary",
            "--max-calls", "2", "--max-tokens-total", "1000000", "--rpm-cap", "60000",
            "--base", str(tmp_path), "--cache-dir", str(tmp_path / "cache"),
        ]
    )  # fmt: skip
    assert rc == 3 and not _parquets(rel)


def test_exit_code_constants_follow_the_table() -> None:
    mod = _load_cli()
    assert (mod.EXIT_LOCAL, mod.EXIT_FATAL, mod.EXIT_QUOTA, mod.EXIT_INCOMPLETE) == (
        1,
        2,
        3,
        4,
    )
