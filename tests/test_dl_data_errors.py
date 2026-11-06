"""Tests for astro_triage.models.dl_data: error paths and save/load (spec S3, ADR 0006 section 15)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from astro_triage.models.dl_data import (
    Preprocessor,
    load_preprocessor,
    save_preprocessor,
)
from astro_triage.models.lgbm import INDICATOR_COLUMN, TrainError

FEATS = ("a", "b", INDICATOR_COLUMN)
N = 1200


def make_frame(n: int = N, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {
            "a": rng.normal(size=n),
            "b": rng.normal(size=n),
            INDICATOR_COLUMN: rng.choice([31.0, 95.0], n),
        }
    )


@pytest.fixture
def fitted() -> Preprocessor:
    return Preprocessor(FEATS).fit(make_frame(seed=1))


def test_null_without_train_indicator_names_column(fitted: Preprocessor) -> None:
    f = make_frame(10, seed=2)
    f.loc[3, "b"] = np.nan
    with pytest.raises(TrainError, match="column b"):
        fitted.transform(f)


def test_unseen_solved_code_rejected(fitted: Preprocessor) -> None:
    f = make_frame(10, seed=2)
    f.loc[0, INDICATOR_COLUMN] = 999.0
    with pytest.raises(TrainError, match="999"):
        fitted.transform(f)


def test_null_solved_code_rejected_in_transform_and_fit(fitted: Preprocessor) -> None:
    f = make_frame(10, seed=2)
    f.loc[0, INDICATOR_COLUMN] = np.nan
    with pytest.raises(TrainError, match="NULL codes"):
        fitted.transform(f)
    t = make_frame(seed=1)
    t.loc[0, INDICATOR_COLUMN] = np.nan
    with pytest.raises(TrainError, match="NULL codes"):
        Preprocessor(FEATS).fit(t)


def test_non_finite_output_rejected(fitted: Preprocessor) -> None:
    f = make_frame(10, seed=2)
    f.loc[0, "a"] = np.inf
    with pytest.raises(TrainError, match="not finite"):
        fitted.transform(f)


def test_missing_feature_columns_rejected(fitted: Preprocessor) -> None:
    with pytest.raises(TrainError, match="lacks feature columns"):
        fitted.transform(make_frame(5).drop(columns=["b"]))
    with pytest.raises(TrainError, match="lacks feature columns"):
        Preprocessor(FEATS).fit(make_frame().drop(columns=[INDICATOR_COLUMN]))


def test_empty_train_frame_rejected() -> None:
    with pytest.raises(TrainError, match="empty train frame"):
        Preprocessor(FEATS).fit(make_frame(0))


def test_all_null_train_column_rejected() -> None:
    """Coder-added behaviour, not in the spec (flag for the architect): fit raises."""
    t = make_frame(seed=1)
    t["b"] = np.nan
    with pytest.raises(TrainError, match="entirely NULL"):
        Preprocessor(FEATS).fit(t)


def test_use_before_fit_rejected() -> None:
    p = Preprocessor(FEATS)
    with pytest.raises(TrainError, match="before fit"):
        p.transform(make_frame(5))
    with pytest.raises(TrainError, match="before fit"):
        p.state_sha256()


def test_save_load_round_trip(tmp_path: Path, fitted: Preprocessor) -> None:
    path = tmp_path / "pre.pkl"
    save_preprocessor(fitted, path)
    loaded = load_preprocessor(path)
    assert loaded.state_sha256() == fitted.state_sha256()
    f = make_frame(25, seed=7)
    (xa, ca), (xb, cb) = fitted.transform(f), loaded.transform(f)
    assert np.array_equal(xa, xb) and np.array_equal(ca, cb)  # type: ignore[arg-type]
    assert loaded.categories == fitted.categories
    assert [p.name for p in tmp_path.iterdir()] == ["pre.pkl"]  # no tmp leftovers


def test_save_overwrites_existing_file(tmp_path: Path, fitted: Preprocessor) -> None:
    """Spec S3 says save/load via lgbm._atomic (os.replace) and is silent on refusal.

    Observed behaviour: the second save silently replaces the first.
    """
    path = tmp_path / "pre.pkl"
    save_preprocessor(fitted, path)
    other = Preprocessor(FEATS).fit(make_frame(seed=99))
    save_preprocessor(other, path)
    assert load_preprocessor(path).state_sha256() == other.state_sha256()


def test_load_missing_file(tmp_path: Path) -> None:
    with pytest.raises(TrainError, match="cannot load preprocessor"):
        load_preprocessor(tmp_path / "nope.pkl")


def test_load_corrupt_file(tmp_path: Path) -> None:
    path = tmp_path / "bad.pkl"
    path.write_bytes(b"not a pickle at all")
    with pytest.raises(TrainError, match="cannot load preprocessor"):
        load_preprocessor(path)
    path.write_bytes(b"")
    with pytest.raises(TrainError, match="cannot load preprocessor"):
        load_preprocessor(path)


def test_load_non_preprocessor_object(tmp_path: Path) -> None:
    import pickle

    path = tmp_path / "other.pkl"
    path.write_bytes(pickle.dumps({"a": 1}))
    with pytest.raises(TrainError, match="does not hold a Preprocessor"):
        load_preprocessor(path)
