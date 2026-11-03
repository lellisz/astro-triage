"""Proofs for ``llm.consolidate`` (ADR 0005 §16): resume equivalence, refusal of partial runs,
parquet schema invariants. Synthetic release and fake opener from ``test_layer2_runner``."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from test_layer2_runner import (
    FIXED_NOW,
    KEY,
    MODEL,
    TEST_ROUTED,
    VAL_ROUTED,
    Opener,
    answer,
    go,
    http_error,
    make_release,
    read_cache,
)

from astro_triage.llm.cache import ResultCache
from astro_triage.llm.consolidate import (
    SCHEMA,
    IncompleteRunError,
    consolidate,
    output_path,
)
from astro_triage.llm.routing import ManifestMismatchError


def cons(rel: Path, fold: str = "test", arm: str = "blind") -> Path:
    with ResultCache(rel.parent / "cache" / "c.jsonl") as cache:
        return consolidate(
            rel,
            fold,
            MODEL,
            arm,
            cache,
            now=lambda: FIXED_NOW,  # type: ignore[arg-type]
        )


def test_interrupted_then_resumed_equals_continuous_run(tmp_path: Path) -> None:
    cont = make_release(tmp_path / "cont")
    split = make_release(tmp_path / "split")

    rep = go(cont, "val", Opener(answer("QSO")))
    assert rep["pending"] == 0

    first = go(split, "val", Opener(answer("QSO")), max_calls=7)
    assert first["stop_reason"] == "QUOTA_STOP"
    assert first["calls_done"] == 7 and first["pending"] == len(VAL_ROUTED) - 7
    with pytest.raises(IncompleteRunError):
        cons(split, "val")
    assert not output_path(split, "val", MODEL, "blind", enable_thinking=False).exists()

    resume_opener = Opener(answer("QSO"))
    second = go(split, "val", resume_opener)
    assert second["stop_reason"] is None
    assert second["cached_skipped"] == 7
    assert second["calls_made"] == len(VAL_ROUTED) - 7 == resume_opener.calls

    a, b = cons(cont, "val"), cons(split, "val")
    ta, tb = pq.read_table(a), pq.read_table(b)
    drop = ["latency_ms"]  # wall-clock, the only non-deterministic column
    assert ta.drop_columns(drop).equals(tb.drop_columns(drop))
    ma, mb = dict(ta.schema.metadata), dict(tb.schema.metadata)
    for volatile in (b"created_utc", b"git_sha"):
        ma.pop(volatile, None)
        mb.pop(volatile, None)
    assert ma == mb


def test_second_complete_run_makes_zero_calls(tmp_path: Path) -> None:
    rel = make_release(tmp_path)
    go(rel, "test", Opener())
    again = Opener()
    rep = go(rel, "test", again)
    assert again.calls == 0
    assert rep["calls_made"] == 0 and rep["calls_done"] == 0
    assert rep["cached_skipped"] == len(TEST_ROUTED)
    assert rep["pending"] == 0 and rep["stop_reason"] is None


def test_consolidate_refuses_routed_object_without_record(tmp_path: Path) -> None:
    rel = make_release(tmp_path)
    go(rel, "test", Opener(), max_calls=4)
    with pytest.raises(IncompleteRunError, match="2 of 6"):
        cons(rel)


def test_consolidate_refuses_empty_cache(tmp_path: Path) -> None:
    rel = make_release(tmp_path)
    with pytest.raises(IncompleteRunError):
        cons(rel)


def test_consolidate_refuses_non_terminal_records(tmp_path: Path) -> None:
    rel = make_release(tmp_path)
    go(rel, "test", Opener(http_error(503)))  # SERVER_ERROR records are not terminal
    with pytest.raises(IncompleteRunError):
        cons(rel)


def test_consolidate_is_refused_when_manifest_hash_mismatches(tmp_path: Path) -> None:
    import pandas as pd

    rel = make_release(tmp_path)
    go(rel, "test", Opener())
    pred = pd.read_parquet(rel / "predictions.parquet")
    pred.loc[0, "p_cal_qso"] = 0.0777
    pred.to_parquet(rel / "predictions.parquet")
    with pytest.raises(ManifestMismatchError):
        cons(rel)


def test_parquet_schema_invariants_and_denominator(tmp_path: Path) -> None:
    rel = make_release(tmp_path)
    script = [
        answer("STAR"),
        answer(content="not json"),
        answer(content='{"label":"UNKNOWN","confidence":10}'),
        answer(finish="length"),
        answer(finish="content_filter"),
        answer(content=""),
    ]
    rep = go(rel, "test", Opener(*script))
    assert rep["pending"] == 0
    out = cons(rel)

    assert out == output_path(rel, "test", MODEL, "blind", enable_thinking=False)
    assert not list(out.parent.glob("*.tmp"))  # atomic write leaves no temp file

    pf = pq.ParquetFile(out)
    names = pf.schema_arrow.names
    assert names == [f.name for f in SCHEMA]
    assert pf.schema_arrow.field("source_id").type == pa.int64()
    assert pa.types.is_dictionary(pf.schema_arrow.field("status").type)
    for rg in range(pf.metadata.num_row_groups):
        for col in range(pf.metadata.num_columns):
            assert pf.metadata.row_group(rg).column(col).compression == "ZSTD"

    df = pq.read_table(out).to_pandas()
    assert df["source_id"].tolist() == sorted(TEST_ROUTED)
    assert df["source_id"].is_monotonic_increasing
    assert len(df) == len(TEST_ROUTED)  # every routed object is in the denominator
    assert df["status"].astype(str).tolist() == [
        "OK",
        "INVALID_JSON",
        "UNKNOWN",
        "TRUNCATED",
        "SCHEMA_VIOLATION",
        "EMPTY_CONTENT",
    ]
    answered = df["status"].astype(str).isin(["OK", "UNKNOWN"])
    assert (df["label_l2"].notna() == answered).all()
    assert (df["confidence_raw"].notna() == answered).all()
    assert df["label_l2"].astype(str)[answered].tolist() == ["STAR", "UNKNOWN"]
    assert (df["l1_conf"] < 0.6).all()
    assert (df["called_utc"] == pd_ts(FIXED_NOW)).all()

    meta = {k.decode(): v.decode() for k, v in pf.schema_arrow.metadata.items()}
    assert meta["fold"] == "test" and meta["arm"] == "blind"
    assert meta["model_requested"] == MODEL
    assert meta["enable_thinking"] == "false" and meta["max_tokens"] == "64"
    assert meta["release"] == "rel"
    assert len(meta["routing_manifest_sha256"]) == 64


def pd_ts(dt: datetime) -> object:
    import pandas as pd

    return pd.Timestamp(dt.astimezone(UTC))


def test_arms_write_distinct_files(tmp_path: Path) -> None:
    rel = make_release(tmp_path)
    go(rel, "test", Opener(), arm="blind")
    cache_blind = rel.parent / "cache" / "c.jsonl"
    assert len(read_cache(rel)) == len(TEST_ROUTED)
    go(rel, "test", Opener(), arm="with_l1")
    assert len(cache_blind.read_text(encoding="utf-8").splitlines()) == 2 * len(
        TEST_ROUTED
    )
    a = cons(rel, arm="blind")
    b = cons(rel, arm="with_l1")
    assert a != b and a.exists() and b.exists()


def test_no_key_in_generated_parquet_or_cache(tmp_path: Path) -> None:
    rel = make_release(tmp_path)
    go(rel, "test", Opener(http_error(429), answer()))
    out = cons(rel)
    blob = b"".join(
        p.read_bytes()
        for p in tmp_path.rglob("*")
        if p.is_file() and p.suffix != ".lock"
    )
    assert KEY.encode() not in blob
    assert out.exists()
