"""F3/F4: the freeze sha is that of the bytes parsed, not of a later re-read; lgbm_reference rule."""

from __future__ import annotations

import hashlib

import numpy as np
import pytest
from dl_eval_kit import frozen_env

from astro_triage.evaluation import dl_desc, dl_eval
from astro_triage.evaluation.dl_val import lgbm_reference


def _assemble(sha: str) -> dict:
    ts = type("TS", (), {"y": np.zeros(3), "pixel": np.arange(3)})()
    return dl_desc.assemble(ts, {}, {}, sha, "ok go")


def test_load_freeze_returns_sha_of_the_bytes_read(tmp_path):
    _, dl, _, _ = frozen_env(tmp_path)
    doc = dl_eval.load_freeze(dl)
    expected = hashlib.sha256((dl / dl_eval.FREEZE_NAME).read_bytes()).hexdigest()
    assert doc["_file_sha256"] == expected


def test_assemble_keeps_loaded_sha_after_file_changes(tmp_path):
    _, dl, _, _ = frozen_env(tmp_path)
    path = dl / dl_eval.FREEZE_NAME
    original = hashlib.sha256(path.read_bytes()).hexdigest()
    doc = dl_eval.load_freeze(dl)
    path.write_bytes(path.read_bytes() + b" ")
    meta = _assemble(str(doc["_file_sha256"]))["meta"]
    assert meta["freeze_sha256"] == original
    assert meta["freeze_sha256"] != hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.mark.parametrize(
    ("system", "ref"),
    [("ftt_noind_s3", "lgbm_noind_s3"), ("mlp_full_s0", "lgbm_full_s0")],
)
def test_lgbm_reference(system, ref):
    assert lgbm_reference(system) == ref
