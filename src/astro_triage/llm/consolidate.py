"""Turn the Layer 2 result cache into the predictions parquet of one release fold (ADR 0005 §16).

Every routed object must have a terminal record: an incomplete run is refused, never
consolidated, so no metric can be computed on a partial subset.
"""

from __future__ import annotations

import os
import subprocess
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Final, Literal

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from astro_triage.abstention.policy import sha256_file
from astro_triage.llm.cache import TERMINAL, ResultCache
from astro_triage.llm.client import BASE_URL
from astro_triage.llm.prompt import PROMPT_VERSION, TEMPLATE_SHA256
from astro_triage.llm.routing import (
    MANIFEST_NAME,
    max_tokens_for,
    model_slug,
    request_for,
    select_routed,
)

__all__ = ["SCHEMA", "IncompleteRunError", "consolidate", "output_path"]

_DICT: Final = pa.dictionary(pa.int8(), pa.string())
SCHEMA: Final = pa.schema(
    [
        pa.field("source_id", pa.int64(), nullable=False),
        pa.field("l1_conf", pa.float32(), nullable=False),
        pa.field("status", _DICT, nullable=False),
        pa.field("label_l2", _DICT, nullable=True),
        pa.field("confidence_raw", pa.int8(), nullable=True),
        pa.field("model_reported", pa.string(), nullable=True),
        pa.field("prompt_tokens", pa.int32(), nullable=True),
        pa.field("completion_tokens", pa.int32(), nullable=True),
        pa.field("latency_ms", pa.int32(), nullable=False),
        pa.field("attempts", pa.int8(), nullable=False),
        pa.field("finish_reason", pa.string(), nullable=True),
        pa.field("called_utc", pa.timestamp("ms", tz="UTC"), nullable=False),
    ]
)
_ANSWERED: Final = frozenset({"OK", "UNKNOWN"})
_REPO_ROOT: Final = Path(__file__).resolve().parents[3]


class IncompleteRunError(ValueError):
    """Some routed object has no terminal record in the cache."""


def output_path(
    release_dir: Path, fold: str, model: str, arm: str, *, enable_thinking: bool
) -> Path:
    think = "on" if enable_thinking else "off"
    name = (
        f"predictions_l2_{model_slug(model)}_{PROMPT_VERSION}_{arm}_think-{think}_"
        f"{fold}.parquet"
    )
    return release_dir / "layer2" / name


def _git_sha() -> str:
    try:
        done = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=_REPO_ROOT,
            capture_output=True,
            text=True,
            check=False,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return "unavailable"
    return done.stdout.strip() if done.returncode == 0 else "unavailable"


def _column(records: list[dict[str, Any]], name: str, type_: pa.DataType) -> pa.Array:
    return pa.array([r.get(name) for r in records], type=type_)


def consolidate(
    release_dir: Path,
    fold: Literal["val", "test"],
    model: str,
    arm: str,
    cache: ResultCache,
    *,
    enable_thinking: bool = False,
    now: Callable[[], datetime] = lambda: datetime.now(UTC),
) -> Path:
    """Write the parquet for all routed objects of ``fold``; raise if any lacks a terminal record."""
    manifest = release_dir / "layer2" / MANIFEST_NAME
    routed = select_routed(release_dir, fold, manifest, None)
    records: list[dict[str, Any]] = []
    missing: list[int] = []
    for _, row in routed.iterrows():
        _, key = request_for(row, model, arm, enable_thinking)
        rec = cache.get(key)
        if rec is None or rec.get("status") not in TERMINAL:
            missing.append(int(row["source_id"]))
        else:
            records.append(rec)
    if missing:
        raise IncompleteRunError(
            f"{len(missing)} of {len(routed)} routed objects lack a terminal record "
            f"(first: {missing[:5]}); finish the run before consolidating"
        )
    status = [r["status"] for r in records]
    for r in records:
        answered = r["status"] in _ANSWERED
        if (r.get("label_l2") is not None) != answered or (
            (r.get("confidence_raw") is not None) != answered
        ):
            raise ValueError(
                f"cache record for source_id {r['source_id']} breaks the label/status rule"
            )
    ts = pd.to_datetime([r["called_utc"] for r in records], utc=True, format="ISO8601")
    table = pa.Table.from_arrays(
        [
            pa.array(routed["source_id"].to_numpy(), type=pa.int64()),
            pa.array(routed["l1_conf"].to_numpy(), type=pa.float32()),
            pa.array(status, type=_DICT),
            _column(records, "label_l2", _DICT),
            _column(records, "confidence_raw", pa.int8()),
            _column(records, "model_reported", pa.string()),
            _column(records, "prompt_tokens", pa.int32()),
            _column(records, "completion_tokens", pa.int32()),
            _column(records, "latency_ms", pa.int32()),
            _column(records, "attempts", pa.int8()),
            _column(records, "finish_reason", pa.string()),
            pa.array(ts.to_numpy(), type=pa.timestamp("ms", tz="UTC")),
        ],
        schema=SCHEMA,
    )
    meta = {
        "schema_version": "1",
        "release": release_dir.name,
        "fold": fold,
        "model_requested": model,
        "prompt_version": PROMPT_VERSION,
        "template_sha256": TEMPLATE_SHA256,
        "arm": arm,
        "enable_thinking": str(enable_thinking).lower(),
        "max_tokens": str(max_tokens_for(enable_thinking)),
        "temperature": "0",
        "endpoint": BASE_URL,
        "routing_manifest_sha256": sha256_file(manifest),
        "created_utc": now().astimezone(UTC).isoformat(timespec="seconds"),
        "git_sha": _git_sha(),
    }
    table = table.replace_schema_metadata(meta)
    out = output_path(release_dir, fold, model, arm, enable_thinking=enable_thinking)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(out.name + ".tmp")
    try:
        pq.write_table(table, tmp, compression="zstd")
        os.replace(tmp, out)
    finally:
        tmp.unlink(missing_ok=True)
    return out
