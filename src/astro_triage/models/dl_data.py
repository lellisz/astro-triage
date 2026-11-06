"""Train-only preprocessing for the deep-learning axis (ADR 0006 section 3.3, section 15).

No torch import. Every fitted quantity (quantiles, indicator set, solved-code vocabulary)
comes from the train frame handed to ``Preprocessor.fit``; ``transform`` never refits and
reads covariates only, so a fold's labels cannot reach this module.
"""

from __future__ import annotations

import hashlib
import json
import pickle
from pathlib import Path
from typing import Final

import numpy as np
import numpy.typing as npt
import pandas as pd
from sklearn.preprocessing import QuantileTransformer

from astro_triage.models.lgbm import INDICATOR_COLUMN, TrainError, _atomic

# ADR 0006 section 3.3 (continuous features): QuantileTransformer(output_distribution="normal",
# n_quantiles=1000, subsample=1_000_000, random_state=20261025).
QT_N_QUANTILES: Final = 1000
QT_SUBSAMPLE: Final = 1_000_000
QT_RANDOM_STATE: Final = 20261025
# ADR 0006 section 3.3 (Amended 2026-11-03, amendment A): noise N(0, 1e-3) as SD, default_rng(20261025),
# train non-null cells, quantile fit only.
FIT_NOISE_SD: Final = 1e-3
FIT_NOISE_SEED: Final = 20261025
# ADR 0006 section 15 (subsample_index): default seed, same as lgbm.SEED.
SUBSAMPLE_SEED: Final = 20261025

Floats32 = npt.NDArray[np.float32]
Ints64 = npt.NDArray[np.int64]


class Preprocessor:
    def __init__(self, features: tuple[str, ...]) -> None:
        self.features = tuple(features)
        self.continuous_columns = tuple(
            c for c in self.features if c != INDICATOR_COLUMN
        )
        self.has_category = INDICATOR_COLUMN in self.features
        self.indicator_columns: tuple[str, ...] = ()
        self.categories: tuple[int, ...] = ()
        self._qt: QuantileTransformer | None = None

    @property
    def n_continuous(self) -> int:
        """Width of ``X_cont``: continuous columns plus one indicator per train-NULL column."""
        return len(self.continuous_columns) + len(self.indicator_columns)

    @property
    def n_categories(self) -> int:
        return len(self.categories)

    def _matrix(self, frame: pd.DataFrame) -> npt.NDArray[np.float64]:
        absent = [c for c in self.features if c not in frame.columns]
        if absent:
            raise TrainError(f"frame lacks feature columns {absent}")
        return frame[list(self.continuous_columns)].to_numpy(dtype=np.float64)

    def fit(self, train: pd.DataFrame) -> Preprocessor:
        x = self._matrix(train)
        if len(x) == 0:
            raise TrainError("cannot fit the preprocessor on an empty train frame")
        null = np.isnan(x)
        dead = [
            c
            for c, d in zip(self.continuous_columns, null.all(axis=0), strict=True)
            if d
        ]
        if dead:
            raise TrainError(
                f"train columns {dead} are entirely NULL; drop them from the feature set"
            )
        self.indicator_columns = tuple(
            c
            for c, n in zip(self.continuous_columns, null.any(axis=0), strict=True)
            if n
        )
        noisy = x.copy()
        present = ~null
        noisy[present] += np.random.default_rng(FIT_NOISE_SEED).normal(
            0.0, FIT_NOISE_SD, int(present.sum())
        )
        self._qt = QuantileTransformer(
            output_distribution="normal",
            n_quantiles=QT_N_QUANTILES,
            subsample=QT_SUBSAMPLE,
            random_state=QT_RANDOM_STATE,
        ).fit(noisy)
        if self.has_category:
            codes = train[INDICATOR_COLUMN].to_numpy(dtype=np.float64)
            if np.isnan(codes).any():
                raise TrainError(
                    f"train {INDICATOR_COLUMN} has NULL codes; the solved code is required"
                )
            self.categories = tuple(int(c) for c in np.unique(codes))
        return self

    def transform(self, frame: pd.DataFrame) -> tuple[Floats32, Ints64 | None]:
        qt = self._qt
        if qt is None:
            raise TrainError("Preprocessor.transform called before fit")
        x = self._matrix(frame)
        null = np.isnan(x)
        index = {c: i for i, c in enumerate(self.continuous_columns)}
        for name in self.continuous_columns:
            if name not in self.indicator_columns and null[:, index[name]].any():
                raise TrainError(
                    f"column {name} has NULL cells but no NULL in train, so no indicator exists; "
                    "rebuild the release or fit on a train fold that covers the NULL pattern"
                )
        if np.isinf(x).any():
            raise TrainError("input features are not finite (inf); sklearn would reject them")
        z = qt.transform(x)
        z[null] = 0.0
        flags = null[:, [index[c] for c in self.indicator_columns]]
        x_cont = np.concatenate([z, flags.astype(np.float64)], axis=1).astype(
            np.float32
        )
        if not np.isfinite(x_cont).all():
            raise TrainError(
                "preprocessed features are not finite; check for inf in the input frame"
            )
        if not self.has_category:
            return x_cont, None
        codes = frame[INDICATOR_COLUMN].to_numpy(dtype=np.float64)
        if np.isnan(codes).any():
            raise TrainError(
                f"{INDICATOR_COLUMN} has NULL codes; the solved code is required"
            )
        vocab = np.asarray(self.categories, dtype=np.float64)
        pos = np.searchsorted(vocab, codes)
        known = (pos < len(vocab)) & (vocab[np.minimum(pos, len(vocab) - 1)] == codes)
        if not known.all():
            unseen = sorted({int(c) for c in codes[~known]})
            raise TrainError(
                f"{INDICATOR_COLUMN} codes {unseen} were not seen in train {self.categories}"
            )
        return x_cont, pos.astype(np.int64)

    def state_sha256(self) -> str:
        qt = self._qt
        if qt is None:
            raise TrainError("Preprocessor.state_sha256 called before fit")
        head = json.dumps(
            {
                "features": list(self.features),
                "continuous": list(self.continuous_columns),
                "indicators": list(self.indicator_columns),
                "categories": list(self.categories),
                "n_quantiles": int(qt.n_quantiles_),
                "noise_sd": FIT_NOISE_SD,
                "noise_seed": FIT_NOISE_SEED,
            },
            sort_keys=True,
        ).encode()
        digest = hashlib.sha256(head)
        digest.update(np.asarray(qt.quantiles_).astype("<f8").tobytes())
        digest.update(np.asarray(qt.references_).astype("<f8").tobytes())
        return digest.hexdigest()


def save_preprocessor(p: Preprocessor, path: Path) -> None:
    def write(tmp: str) -> None:
        with open(tmp, "wb") as fh:
            pickle.dump(p, fh, protocol=pickle.HIGHEST_PROTOCOL)

    _atomic(path, write)


def load_preprocessor(path: Path) -> Preprocessor:
    """Unpickle a preprocessor; pickle runs code, so only load files this project wrote under data/."""
    try:
        with open(path, "rb") as fh:
            obj = pickle.load(fh)
    except (OSError, pickle.UnpicklingError, EOFError) as exc:
        raise TrainError(
            f"cannot load preprocessor {path.name}: {exc}; refit it from the train fold"
        ) from exc
    if not isinstance(obj, Preprocessor):
        raise TrainError(f"{path.name} does not hold a Preprocessor")
    return obj


def subsample_index(n_train: int, n: int, seed: int = SUBSAMPLE_SEED) -> Ints64:
    if n <= 0 or n > n_train:
        raise TrainError(f"subsample size {n} must be in 1..{n_train}")
    perm = np.random.default_rng(seed).permutation(n_train)[:n]
    return np.sort(perm).astype(np.int64)
