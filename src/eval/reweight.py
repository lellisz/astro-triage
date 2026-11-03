"""Per-sample importance weights that map a test class mix onto a target prior.

Contract (ADR 0003, "Contract for reweight function"):

    reweight(y_true, prior) -> numpy.ndarray

Weights are ``w_i = pi[y_i] / n[y_i]`` where ``n[c]`` is the number of samples
of class ``c`` in ``y_true``, then normalised so that ``sum(w) == 1``.

Invariants:
  - Output is float64 with shape ``(len(y_true),)``, in the same order as
    ``y_true``. The input is not mutated. The function is deterministic.
  - ``sum(w) == 1`` within 1e-12, and the weights of class ``c`` sum to ``pi_c``.
  - ``prior`` must sum to 1 within 1e-6 (it is then renormalised exactly);
    a larger deviation raises ``ValueError``.
  - Any prior value that is negative, NaN or inf raises ``ValueError``.
  - A label in ``y_true`` that is absent from ``prior`` raises ``ValueError``
    (it is never silently dropped).
  - A class with ``pi_c > 0`` and ``n_c == 0`` raises ``EmptyClassError`` naming
    the class.
  - A class with ``pi_c == 0`` gives its samples weight 0 (allowed, whether or
    not the class is present in ``y_true``).
  - Empty ``y_true`` raises ``ValueError``.
  - The BALANCED regime equals ``reweight(y, uniform prior over the label set)``.

Allowed dependencies: numpy, collections.abc, typing.
"""

from collections.abc import Mapping, Sequence

import numpy as np


class EmptyClassError(ValueError):
    """A class with a positive prior has no samples in ``y_true``."""


def reweight(
    y_true: Sequence[str] | np.ndarray, prior: Mapping[str, float]
) -> np.ndarray:
    """Return per-sample weights mapping the class mix of ``y_true`` onto ``prior``.

    See the module docstring for the full contract.
    """
    labels = np.asarray(y_true)
    if labels.size == 0:
        raise ValueError("y_true is empty")

    values = np.array(list(prior.values()), dtype=np.float64)
    if not np.all(np.isfinite(values)) or np.any(values < 0):
        raise ValueError("prior values must be finite and non-negative")
    total = values.sum()
    if abs(total - 1.0) > 1e-6:
        raise ValueError(f"prior sums to {total!r}, expected 1 within 1e-6")
    pi = {cls: value / total for cls, value in zip(prior, values)}

    classes, inverse, counts = np.unique(
        labels, return_inverse=True, return_counts=True
    )
    present = classes.tolist()
    unknown = [cls for cls in present if cls not in pi]
    if unknown:
        raise ValueError(f"labels missing from prior: {unknown}")
    empty = [cls for cls, p in pi.items() if p > 0 and cls not in present]
    if empty:
        raise EmptyClassError(f"classes with positive prior and no samples: {empty}")

    per_class = np.array([pi[cls] / n for cls, n in zip(present, counts)])
    weights = per_class[inverse.reshape(-1)]
    return weights / weights.sum()
