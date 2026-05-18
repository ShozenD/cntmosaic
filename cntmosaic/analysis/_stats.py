"""Shared statistical utilities for model summarisers."""

from __future__ import annotations

import warnings
from typing import Literal, Tuple

import numpy as np
from numpy.typing import NDArray


def validate_alpha(alpha: float) -> None:
    """Validate that *alpha* lies strictly in (0, 1).

    Raises
    ------
    ValueError
        If *alpha* is not in the open interval (0, 1).
    """
    if not 0 < alpha < 1:
        raise ValueError(f"alpha must be in (0, 1), got {alpha}")


def compute_quantiles(
    samples: NDArray,
    probs: Tuple[float, ...],
    axis: int = 0,
) -> NDArray:
    """Compute quantiles along *axis*.

    Parameters
    ----------
    samples : NDArray
        Input array; the sample axis is given by *axis*.
    probs : tuple of float
        Quantile probabilities in [0, 1].
    axis : int, default=0
        Axis along which to compute quantiles.

    Returns
    -------
    NDArray
        Shape ``(len(probs), ...)`` with quantiles along axis 0.

    Raises
    ------
    ValueError
        If any probability is outside [0, 1].
    """
    if not all(0 <= p <= 1 for p in probs):
        raise ValueError(f"All probabilities must be in [0, 1], got {probs}")
    if list(probs) != sorted(probs):
        warnings.warn(
            "Probabilities are not sorted. Output will follow input order.",
            UserWarning,
        )
    return np.quantile(samples, probs, axis=axis)


def summarise_samples(
    samples: NDArray,
    probs: Tuple[float, ...],
    measure: Literal["mean", "median"] = "median",
    axis: int = 0,
) -> Tuple[NDArray, NDArray]:
    """Compute quantiles and a central-tendency measure in a single pass.

    For ``measure="median"``, augments *probs* with 0.5 so only one
    ``np.quantile`` call is needed; the median is extracted and stripped from
    the returned quantile array when it was not in the original *probs*.

    Parameters
    ----------
    samples : NDArray
        Input samples; the sample axis is given by *axis*.
    probs : tuple of float
        Quantile probabilities, not including the central-tendency quantile
        (e.g. ``(0.025, 0.975)`` for a 95 % CI).
    measure : {"mean", "median"}, default="median"
        Central-tendency measure to return alongside the quantiles.
    axis : int, default=0
        Axis along which to compute statistics.

    Returns
    -------
    quantiles : NDArray
        Shape ``(len(probs), ...)`` — quantiles in the order given by *probs*.
    central : NDArray
        Central-tendency array (mean or median); shape matches a single
        quantile slice.

    Raises
    ------
    ValueError
        If *measure* is not ``"mean"`` or ``"median"``.
    """
    if measure == "mean":
        central = samples.mean(axis=axis)
        quantiles = compute_quantiles(samples, probs, axis=axis)
    elif measure == "median":
        median_in_probs = 0.5 in probs
        augmented = tuple(sorted(set(probs) | {0.5}))
        all_q = np.quantile(samples, augmented, axis=axis)
        median_idx = list(augmented).index(0.5)
        central = all_q[median_idx]
        if median_in_probs:
            quantiles = all_q
        else:
            quantiles = np.delete(all_q, median_idx, axis=0)
    else:
        raise ValueError(f"measure must be 'mean' or 'median', got {measure!r}")
    return quantiles, central
