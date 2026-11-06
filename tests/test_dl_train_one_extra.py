"""dl_train `train_one` extras: subsample seen by the Preprocessor, prior baseline flag, T bound flag."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from test_dl_train_one import SIZES, make_folds, run

torch = pytest.importorskip("torch")

from astro_triage.dataset.build import FEATURE_COLUMNS
from astro_triage.models import dl_train
from astro_triage.models.dl_data import subsample_index
from astro_triage.models.lgbm import TEMPERATURE_BOUNDS, Fold


def test_preprocessor_fit_sees_exactly_the_subsample(
    tmp_path: Path, monkeypatch
) -> None:
    seen: list[pd.DataFrame] = []
    orig = dl_train.Preprocessor.fit

    def spy(self, frame):
        seen.append(frame.copy())
        return orig(self, frame)

    monkeypatch.setattr(dl_train.Preprocessor, "fit", spy)
    folds = make_folds()
    run(tmp_path, folds, train_subset=90)
    want = folds["train"].frame.iloc[subsample_index(SIZES["train"], 90)]
    cols = list(FEATURE_COLUMNS)
    assert len(seen) == 1 and len(seen[0]) == 90
    assert np.array_equal(
        seen[0][cols].to_numpy(), want[cols].to_numpy(), equal_nan=True
    )


def test_beats_prior_logloss_true_on_separable_data(tmp_path: Path) -> None:
    out = run(tmp_path, make_folds(), max_epochs=15, patience=15)
    assert out["beats_prior_logloss"] is True


def test_beats_prior_logloss_false_with_shuffled_train_labels(tmp_path: Path) -> None:
    folds = make_folds()
    train = folds["train"]
    y = np.random.default_rng(1).permutation(train.labels)
    folds["train"] = Fold(train.source_id, y, train.frame)
    out = run(tmp_path, folds, max_epochs=15, patience=15)
    assert out["beats_prior_logloss"] is False


def test_temperature_at_bound_is_coherent_with_bounds(tmp_path: Path) -> None:
    out = run(tmp_path, make_folds())
    lo, hi = TEMPERATURE_BOUNDS
    t = out["temperature"]
    assert "temperature_at_bound" in out and isinstance(
        out["temperature_at_bound"], bool
    )
    assert out["temperature_at_bound"] == (t <= lo * 1.001 or t >= hi * 0.999)
