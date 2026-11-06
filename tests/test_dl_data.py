"""Tests for astro_triage.models.dl_data (spec S3, ADR 0006 section 3.3): behaviour of fit/transform.

Synthetic frames only. 1200 train rows keep sklearn above n_quantiles=1000 (no cap warning)
except in the one test that exercises the cap on purpose.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.preprocessing import QuantileTransformer

from astro_triage.models import dl_data
from astro_triage.models.dl_data import Preprocessor, subsample_index
from astro_triage.models.lgbm import (
    INDICATOR_COLUMN,
    MODEL_FULL,
    MODEL_NOIND,
    TrainError,
)

FEATS = ("a", "b", "c", INDICATOR_COLUMN)
N = 1200


def make_frame(
    features: tuple[str, ...], n: int = N, seed: int = 0, nan_cols: dict | None = None
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    cols: dict[str, np.ndarray] = {}
    for c in features:
        if c == INDICATOR_COLUMN:
            cols[c] = rng.choice([31.0, 95.0], n)
        else:
            cols[c] = rng.normal(size=n) * 5 + 2
    df = pd.DataFrame(cols)
    for c, rows in (nan_cols or {}).items():
        df.loc[rows, c] = np.nan
    return df


def test_fit_ignores_val_and_test_values() -> None:
    train = make_frame(FEATS, seed=1)
    base = Preprocessor(FEATS).fit(train)
    val = make_frame(FEATS, 300, seed=2)
    val[["a", "b"]] *= 10
    val.loc[:5, "c"] = np.nan  # would need an indicator; irrelevant to fit
    again = Preprocessor(FEATS).fit(train)
    assert base.state_sha256() == again.state_sha256()
    assert val is not train  # val never handed to fit


def test_fit_spy_sees_only_train_rows(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: list[int] = []
    orig = QuantileTransformer.fit

    def spy(self, X, y=None):
        seen.append(X.shape[0])
        return orig(self, X, y)

    monkeypatch.setattr(QuantileTransformer, "fit", spy)
    train = make_frame(FEATS, seed=1)
    p = Preprocessor(FEATS).fit(train)
    val = make_frame(FEATS, 77, seed=2) * 10
    val[INDICATOR_COLUMN] = 31.0
    p.transform(val)
    assert seen == [N]


def test_transform_does_not_change_state_and_is_deterministic() -> None:
    p = Preprocessor(FEATS).fit(make_frame(FEATS, seed=1))
    sha = p.state_sha256()
    frame = make_frame(FEATS, 50, seed=3)
    x1, c1 = p.transform(frame)
    x2, c2 = p.transform(frame)
    assert np.array_equal(x1, x2)
    assert c1 is not None and c2 is not None and np.array_equal(c1, c2)
    assert x1.dtype == np.float32 and c1.dtype == np.int64
    assert p.state_sha256() == sha


def test_noise_enters_fit_only() -> None:
    """Fit on noisy values: refitting is reproducible, and transform of train equals qt(clean)."""
    train = make_frame(("a",), seed=4)
    p = Preprocessor(("a",)).fit(train)
    clean = p._qt.transform(train[["a"]].to_numpy(np.float64))
    x, _ = p.transform(train)
    assert np.array_equal(x, clean.astype(np.float32))
    assert Preprocessor(("a",)).fit(train).state_sha256() == p.state_sha256()


def test_noise_changes_quantiles_vs_no_noise(monkeypatch: pytest.MonkeyPatch) -> None:
    train = make_frame(("a",), seed=4)
    with_noise = Preprocessor(("a",)).fit(train).state_sha256()
    monkeypatch.setattr(dl_data, "FIT_NOISE_SD", 0.0)
    assert Preprocessor(("a",)).fit(train).state_sha256() != with_noise


def test_indicators_nan_to_zero_and_counts() -> None:
    train = make_frame(FEATS, seed=1, nan_cols={"b": slice(0, 10)})
    p = Preprocessor(FEATS).fit(train)
    # Observation for the architect: ADR 3.3 names them '<name>__isna'; code stores source names.
    assert p.indicator_columns == ("b",)
    assert p.continuous_columns == ("a", "b", "c")
    assert p.n_continuous == 3 + 1  # coder's reading: continuous + indicators
    frame = make_frame(FEATS, 20, seed=5, nan_cols={"b": [0, 3]})
    x, cat = p.transform(frame)
    assert x.shape == (20, 4)
    assert (x[[0, 3], 1] == 0.0).all()
    assert x[:, 3].tolist() == [1.0 if i in (0, 3) else 0.0 for i in range(20)]
    assert np.isfinite(x).all() and cat is not None and cat.shape == (20,)


def test_indicator_free_train_has_no_extra_columns() -> None:
    p = Preprocessor(FEATS).fit(make_frame(FEATS, seed=1))
    assert p.indicator_columns == () and p.n_continuous == 3
    assert p.transform(make_frame(FEATS, 5, seed=2))[0].shape == (5, 3)


def test_noind_has_no_category_and_full_has() -> None:
    noind = Preprocessor(MODEL_NOIND.features).fit(make_frame(MODEL_NOIND.features))
    x, cat = noind.transform(make_frame(MODEL_NOIND.features, 10, seed=9))
    assert cat is None and noind.n_categories == 0 and noind.categories == ()
    assert x.shape == (10, len(MODEL_NOIND.features))
    full = Preprocessor(MODEL_FULL.features).fit(make_frame(MODEL_FULL.features))
    _, cat = full.transform(make_frame(MODEL_FULL.features, 10, seed=9))
    assert full.categories == (31, 95) and full.n_categories == 2
    assert cat is not None and set(cat.tolist()) <= {0, 1}
    assert full.n_continuous == len(MODEL_FULL.features) - 1


def test_category_index_maps_codes_to_sorted_vocab_positions() -> None:
    p = Preprocessor(FEATS).fit(make_frame(FEATS, seed=1))
    f = make_frame(FEATS, 4, seed=2)
    f[INDICATOR_COLUMN] = [95.0, 31.0, 95.0, 31.0]
    assert p.transform(f)[1].tolist() == [1, 0, 1, 0]  # type: ignore[union-attr]


def test_constant_train_column_stays_finite() -> None:
    train = make_frame(FEATS, seed=1)
    train["b"] = 3.0
    p = Preprocessor(FEATS).fit(train)
    x, _ = p.transform(make_frame(FEATS, 30, seed=2).assign(b=3.0))
    assert np.isfinite(x).all()
    far = make_frame(FEATS, 30, seed=2).assign(b=1e9)
    assert np.isfinite(p.transform(far)[0]).all()


def test_small_train_hits_quantile_cap_warning() -> None:
    train = make_frame(FEATS, 200, seed=1)
    with pytest.warns(UserWarning, match="n_quantiles"):
        p = Preprocessor(FEATS).fit(train)
    assert np.isfinite(p.transform(train)[0]).all()


def test_subsample_index_properties() -> None:
    idx = subsample_index(1000, 100)
    assert idx.dtype == np.int64 and len(idx) == 100
    assert (np.diff(idx) > 0).all()  # sorted and unique
    assert idx.min() >= 0 and idx.max() < 1000
    assert np.array_equal(idx, subsample_index(1000, 100))
    assert not np.array_equal(idx, subsample_index(1000, 100, seed=1))
    assert np.array_equal(subsample_index(7, 7), np.arange(7))
    assert subsample_index(5, 1).shape == (1,)


@pytest.mark.parametrize(("n_train", "n"), [(10, 0), (10, -1), (10, 11), (0, 1)])
def test_subsample_index_rejects_bad_size(n_train: int, n: int) -> None:
    with pytest.raises(TrainError, match="subsample size"):
        subsample_index(n_train, n)
