from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import scipy as sp
from numpy.typing import NDArray


def rbf_kernel(x1: float, x2: float, lenscale: float = 5.0) -> float:
    """Radial basis function kernel."""
    return np.exp(-((x1 - x2) ** 2) / (2 * lenscale**2))


def gram_matrix(xs: NDArray, lenscale: float = 5.0) -> NDArray:
    """Compute Gram matrix using RBF kernel."""
    return np.asarray([[rbf_kernel(x1, x2, lenscale) for x2 in xs] for x1 in xs])


@dataclass(eq=False)
class Stratification:
    """
    Defines a stratification of the population.

    This class defines a stratification of the population into different levels
    (e.g., socioeconomic status, geographic regions) and generates age distributions
    for each stratum using Gaussian Process-based perturbations of a reference age distribution.

    Attributes
    ----------
    name : str
                Name of the stratification (e.g., "SES", "Region").
    n_strata : int
                Number of strata in the stratification.
    ref_age_dist : NDArray
                Reference age distribution to base the stratification on.
    labels : list[str] | None, optional
                Labels for each level of the stratification. If None, default labels will be generated
                as "{name}_0", "{name}_1", ..., "{name}_{n_strata-1}", by default None.
    alpha : float | None, optional
                Scaling factor for the GP perturbations. If None, a random value between 0.01 and 0.2
                will be chosen, by default None.
    lenscale : float, optional
                Length scale of the RBF kernel used to generate smooth age-distribution perturbations,
                by default 5.0.
    seed : int | None, optional
                Random seed for reproducibility, by default None.
    """

    name: str
    n_strata: int
    ref_age_dist: NDArray
    labels: list[str] | None = None
    alpha: float | None = None
    lenscale: float = 5.0
    seed: int | None = None

    # Computed attributes — excluded from __init__, __repr__, and __eq__.
    # No default here: __post_init__ always assigns these before any method can run.
    codes: list[int] = field(init=False, repr=False)
    A: int = field(init=False, repr=False)
    _P: NDArray | None = field(default=None, init=False, repr=False)
    _Q: NDArray | None = field(default=None, init=False, repr=False)
    _df_P: pd.DataFrame | None = field(default=None, init=False, repr=False)
    _df_Q: pd.DataFrame | None = field(default=None, init=False, repr=False)

    def __post_init__(self):
        if self.labels is None:
            self.labels = [f"{self.name}_{i}" for i in range(self.n_strata)]
        self.codes = list(range(len(self.labels)))
        self.A = len(self.ref_age_dist)

        if self.alpha is None:
            self.alpha = np.random.default_rng(self.seed).uniform(0.01, 0.2)

    def _run_generate(self, seed: int | None = None) -> None:
        rng = np.random.default_rng(seed if seed is not None else self.seed)

        K = gram_matrix(np.arange(self.A), self.lenscale)
        L = sp.linalg.cholesky(K + 1e-6 * np.eye(self.A), lower=True)

        logits = np.array(
            [
                self.alpha * (L @ rng.standard_normal(self.A))
                for _ in range(self.n_strata)
            ]
        )

        probs = sp.special.softmax(logits, axis=0)
        self._P = np.round(probs * self.ref_age_dist[np.newaxis, :]).astype(int)

    def generate(self, seed: int | None = None) -> None:
        """
        Generate stratified age distributions using GP-based perturbations.

        Resets and recomputes all cached arrays and DataFrames. If a seed is
        provided it replaces the instance's stored seed for future calls.

        Parameters
        ----------
        seed : int | None, optional
            Random seed for reproducibility. Overwrites the instance seed when given.
        """
        if seed is not None:
            self.seed = seed

        self._P = None
        self._Q = None
        self._df_P = None
        self._df_Q = None

        self._run_generate(self.seed)

    @property
    def P(self) -> NDArray:
        """Population size matrix (n_strata × n_ages)."""
        if self._P is None:
            self._run_generate()
        return self._P

    @property
    def Q(self) -> NDArray:
        """Population proportion matrix (n_strata × n_ages)."""
        if self._Q is None:
            self._Q = self.P / self.P.sum(axis=0, keepdims=True)
        return self._Q

    @property
    def df_P(self) -> pd.DataFrame:
        """Population size as a long-form DataFrame."""
        if self._df_P is None:
            assert self.labels is not None
            self._df_P = pd.concat(
                [
                    pd.DataFrame(
                        {"age": np.arange(self.A), self.name: label, "P": self.P[k, :]}
                    )
                    for k, label in enumerate(self.labels)
                ],
                ignore_index=True,
            )
        return self._df_P

    @property
    def df_Q(self) -> pd.DataFrame:
        """Population proportion as a long-form DataFrame."""
        if self._df_Q is None:
            assert self.labels is not None
            self._df_Q = pd.concat(
                [
                    pd.DataFrame(
                        {"age": np.arange(self.A), self.name: label, "Q": self.Q[k, :]}
                    )
                    for k, label in enumerate(self.labels)
                ],
                ignore_index=True,
            )
        return self._df_Q
