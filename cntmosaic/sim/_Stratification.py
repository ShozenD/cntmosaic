from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import pandas as pd
import scipy as sp
from numpy.typing import NDArray


def rbf_kernel(x1: float, x2: float, lenscale: float = 5.0) -> float:
    """Radial basis function kernel."""
    return np.exp(-((x1 - x2) ** 2) / (2 * lenscale**2))


def gram_matrix(xs: NDArray) -> NDArray:
    """Compute Gram matrix using RBF kernel."""
    return np.asarray([[rbf_kernel(x1, x2) for x2 in xs] for x1 in xs])


@dataclass
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
    levels : int
                Number of levels in the stratification.
    ref_age_dist : NDArray
                Reference age distribution to base the stratification on.
    labels : Optional[List[str]], optional
                Labels for each level of the stratification. If None, default labels will be generated
                as "{name}_0", "{name}_1", ..., "{name}_{levels-1}", by default None.
    alpha : Optional[float], optional
                Scaling factor for the GP perturbations. If None, a random value between 0.01 and 0.2
                will be chosen, by default None.
    """

    name: str
    n_strata: int
    ref_age_dist: NDArray
    labels: Optional[List[str]] = None
    alpha: Optional[float] = None
    seed: Optional[int] = None

    # Computed attributes
    codes: List[int] = None
    _P: Optional[NDArray] = None
    _Q: Optional[NDArray] = None
    _df_P: Optional[pd.DataFrame] = None
    _df_Q: Optional[pd.DataFrame] = None

    def __post_init__(self):
        if self.labels is None:
            self.labels = [f"{self.name}_{i}" for i in range(self.n_strata)]
            self.codes = list(range(self.n_strata))
        else:
            self.codes = list(range(len(self.labels)))

        if self.alpha is None:
            self.alpha = np.random.uniform(0.01, 0.2)

        self.A = len(self.ref_age_dist)

        # Always generate during initialization
        self.generate(self.seed)

    def generate(self, seed: Optional[int] = None) -> None:
        """
        Generate stratified age distribution using GP-based perturbations.

        Parameters
        ----------
        seed : int, optional
            Random seed for reproducibility, by default None.

        Returns
        -------
            None
        """
        if seed is not None:
            np.random.seed(seed)

        # Generate GP samples for each stratum
        K = gram_matrix(np.arange(self.A))
        L = sp.linalg.cholesky(K + 1e-6 * np.eye(self.A), lower=True)

        logits = np.array(
            [
                self.alpha * (L @ np.random.normal(size=self.A))
                for _ in range(self.n_strata)
            ]
        )

        # Convert to probabilities and weight by reference distribution
        probs = sp.special.softmax(logits, axis=0)
        self._P = np.round(probs * self.ref_age_dist[np.newaxis, :]).astype(int)

    @property
    def P(self) -> NDArray:
        """
        Returns the population size matrix of the stratification.

        Returns
        -------
            NDArray
        """
        return self._P

    @property
    def Q(self) -> NDArray:
        """
        Returns the population proportion matrix of the stratification.

        Returns
        -------
            NDArray
        """
        if self._Q is None:
            self._Q = self._P / self._P.sum(axis=0, keepdims=True)

        return self._Q

    @property
    def df_P(self) -> pd.DataFrame:
        """
        Returns the population size DataFrame of the stratification.

        Returns
        -------
            pd.DataFrame
        """
        if self._df_P is None:
            self._df_P = pd.concat(
                [
                    pd.DataFrame(
                        {"age": np.arange(self.A), self.name: label, "P": self._P[k, :]}
                    )
                    for k, label in enumerate(self.labels)
                ],
                ignore_index=True,
            )

        return self._df_P

    @property
    def df_Q(self) -> pd.DataFrame:
        """
        Returns the population proportion DataFrame of the stratification.

        Returns
        -------
            pd.DataFrame
        """
        if self._df_Q is None:
            # Trigger Q computation if needed
            _ = self.Q

            self._df_Q = pd.concat(
                [
                    pd.DataFrame(
                        {"age": np.arange(self.A), self.name: label, "Q": self._Q[k, :]}
                    )
                    for k, label in enumerate(self.labels)
                ],
                ignore_index=True,
            )

        return self._df_Q
