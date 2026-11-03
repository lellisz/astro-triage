"""Select the routed objects of a release fold and derive the request identity (ADR 0005 D2, D5.4).

An object is routed to Layer 2 iff its Layer 1 confidence is below the frozen ``tau`` of
``routing_v1.json``. Both folds are refused unless the manifest's predictions hash matches the
predictions file on disk, so a threshold chosen on other predictions can never be applied.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any, Final, Literal

import numpy as np
import pandas as pd

from astro_triage.abstention.policy import l1_confidence, score_columns, sha256_file
from astro_triage.llm.cache import cache_key
from astro_triage.llm.prompt import (
    FEATURE_COLUMNS,
    PROMPT_VERSION,
    build_messages,
)

__all__ = [
    "DEFAULT_SEED",
    "MANIFEST_NAME",
    "ManifestMismatchError",
    "max_tokens_for",
    "model_slug",
    "request_for",
    "select_routed",
]

DEFAULT_SEED: Final = 20261025
MANIFEST_NAME: Final = "routing_v1.json"
_MAX_TOKENS_THINKING_OFF: Final = 64
_MAX_TOKENS_THINKING_ON: Final = 4096


class ManifestMismatchError(ValueError):
    """The routing manifest does not describe the predictions file on disk."""


def model_slug(model: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", model)


def max_tokens_for(enable_thinking: bool) -> int:
    return _MAX_TOKENS_THINKING_ON if enable_thinking else _MAX_TOKENS_THINKING_OFF


def _load_manifest(release_dir: Path, manifest: Path) -> tuple[float, Path]:
    payload: Any = json.loads(manifest.read_text(encoding="utf-8"))
    name = payload.get("predictions_file")
    expected = payload.get("predictions_sha256")
    tau = payload.get("tau")
    if not isinstance(name, str) or not isinstance(expected, str):
        raise ManifestMismatchError(f"{manifest.name} lacks the predictions hash")
    if isinstance(tau, bool) or not isinstance(tau, (int, float)):
        raise ManifestMismatchError(f"{manifest.name} lacks a numeric tau")
    pred_path = release_dir / Path(name).name
    if sha256_file(pred_path) != expected:
        raise ManifestMismatchError(
            f"{pred_path.name} does not match the sha256 frozen in {manifest.name}; "
            "refusing to route (a new manifest version is required)"
        )
    return float(tau), pred_path


def select_routed(
    release_dir: Path,
    fold: Literal["val", "test"],
    manifest: Path,
    sample: int | None,
    seed: int = DEFAULT_SEED,
) -> pd.DataFrame:
    """Routed objects of ``fold`` sorted by ``source_id``: ids, l1 columns, ``l1_conf``, features.

    ``sample`` draws that many routed objects with ``default_rng(seed)`` (all if it is not
    smaller than the routed count). Labels are read only for ``source_id`` and ``fold``.
    """
    if fold not in ("val", "test"):
        raise ValueError(f"unknown fold: {fold!r}")
    if sample is not None and sample <= 0:
        raise ValueError("sample must be positive")
    tau, pred_path = _load_manifest(release_dir, manifest)
    labels = pd.read_parquet(
        release_dir / "labels.parquet",
        columns=["source_id", "fold"],
        filters=[("fold", "==", fold)],
    )
    pred = pd.read_parquet(pred_path, columns=["source_id", *score_columns()])
    pred = labels[["source_id"]].merge(pred, on="source_id", how="left", validate="1:1")
    if pred[score_columns()].isna().any(axis=None):
        raise ValueError(f"predictions do not cover every {fold} object; refusing")
    pred["l1_conf"] = l1_confidence(pred)
    routed = pred[pred["l1_conf"] < tau].sort_values("source_id", ignore_index=True)
    if routed.empty:
        raise ValueError(f"no routed objects in the {fold} fold at tau={tau}")
    if sample is not None and sample < len(routed):
        pick = np.sort(
            np.random.default_rng(seed).choice(len(routed), size=sample, replace=False)
        )
        routed = routed.iloc[pick].reset_index(drop=True)
    feats = pd.read_parquet(
        release_dir / "features.parquet",
        columns=["source_id", *FEATURE_COLUMNS],
        filters=[("source_id", "in", routed["source_id"].tolist())],
    )
    feats[list(FEATURE_COLUMNS)] = feats[list(FEATURE_COLUMNS)].astype("float64")
    out = routed.merge(
        feats, on="source_id", how="left", validate="1:1", indicator=True
    )
    if (out.pop("_merge") != "both").any():
        raise ValueError("features.parquet lacks rows for some routed objects")
    return out


def request_for(
    row: pd.Series, model: str, arm: str, enable_thinking: bool
) -> tuple[list[dict[str, str]], str]:
    """Messages and cache key of one routed row (the key hashes the filled prompt)."""
    if arm not in ("blind", "with_l1"):
        raise ValueError(f"unknown arm: {arm!r}")
    features = {c: float(row[c]) for c in FEATURE_COLUMNS}
    l1 = {c: float(row[c]) for c in score_columns()}
    messages = build_messages(features, arm, l1)
    filled = hashlib.sha256(
        "\n".join(m["content"] for m in messages).encode("utf-8")
    ).hexdigest()
    key = cache_key(
        model,
        PROMPT_VERSION,
        arm,
        enable_thinking,
        max_tokens_for(enable_thinking),
        filled,
        int(row["source_id"]),
    )
    return messages, key
