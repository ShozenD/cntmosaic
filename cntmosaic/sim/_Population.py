from __future__ import annotations

import math
from dataclasses import dataclass, field
from itertools import product
from typing import List, Union

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from ._Stratification import Stratification


@dataclass
class Population:
    """
    Joint stratified population built from one or more Stratification objects.

    For a single Stratification this is a thin wrapper that delegates directly
    to it. For multiple Stratifications the joint population is formed by the
    element-wise product of the per-stratum proportion matrices across all
    combinations of strata (Kronecker structure).

    Attributes
    ----------
    strats : Stratification | list[Stratification]
        One or more stratification dimensions. When a list is supplied every
        Stratification must share the same ``ref_age_dist``.
    ref_age_dist : NDArray
        Reference age distribution (copied from the first stratification).
    coord_labels : list
        Stratum labels for each combination. A flat ``list[str]`` in the
        single-strat case; a ``list[tuple[str, ...]]`` in the multi-strat case.
    coord_codes : list
        Stratum integer codes for each combination. Matches the structure of
        ``coord_labels``.
    """

    strats: Union[Stratification, List[Stratification]]

    ref_age_dist: NDArray = field(init=False, repr=False)
    coord_labels: list = field(init=False, repr=False)
    coord_codes: list = field(init=False, repr=False)

    _Q: NDArray | None = field(default=None, init=False, repr=False)
    _P: NDArray | None = field(default=None, init=False, repr=False)
    _df_Q: pd.DataFrame | None = field(default=None, init=False, repr=False)
    _df_P: pd.DataFrame | None = field(default=None, init=False, repr=False)

    def __post_init__(self):
        if isinstance(self.strats, list):
            ref = self.strats[0].ref_age_dist
            for i, strat in enumerate(self.strats[1:], start=1):
                if not np.array_equal(strat.ref_age_dist, ref):
                    raise ValueError(
                        f"All stratifications must share the same ref_age_dist; "
                        f"strats[{i}] differs from strats[0]."
                    )
            self.ref_age_dist = ref
            self.coord_labels = list(product(*[strat.labels for strat in self.strats]))
            self.coord_codes = list(product(*[strat.codes for strat in self.strats]))
        else:
            self.ref_age_dist = self.strats.ref_age_dist
            self.coord_labels = self.strats.labels
            self.coord_codes = self.strats.codes

    def generate(self, seed: int | None = None) -> None:
        """
        Regenerate all stratified age distributions and clear cached arrays.

        Delegates to each child :class:`Stratification`'s ``generate()`` method,
        then invalidates this instance's cached Q, P, and DataFrame views.

        Parameters
        ----------
        seed : int | None, optional
            Random seed forwarded to every child Stratification.
        """
        strats = self.strats if isinstance(self.strats, list) else [self.strats]
        for strat in strats:
            strat.generate(seed)
        self._Q = None
        self._P = None
        self._df_Q = None
        self._df_P = None

    @property
    def Q(self) -> NDArray:
        """Population proportion matrix (n_joint_strata × n_ages)."""
        if isinstance(self.strats, Stratification):
            return self.strats.Q

        if self._Q is not None:
            return self._Q

        grid_size = math.prod(strat.n_strata for strat in self.strats)
        Q = np.ones((grid_size, self.ref_age_dist.shape[0]))
        for s, code_tuple in enumerate(self.coord_codes):
            for j, code in enumerate(code_tuple):
                Q[s, :] *= self.strats[j].Q[code, :]

        self._Q = Q
        return self._Q

    @property
    def df_Q(self) -> pd.DataFrame:
        """Population proportion as a long-form DataFrame."""
        if isinstance(self.strats, Stratification):
            return self.strats.df_Q

        if self._df_Q is not None:
            return self._df_Q

        n_combos = len(self.coord_labels)
        n_ages = self.Q.shape[1]
        strat_cols = {
            self.strats[j].name: np.repeat(
                [lbl[j] for lbl in self.coord_labels], n_ages
            )
            for j in range(len(self.strats))
        }
        self._df_Q = pd.DataFrame(
            {
                "age": np.tile(np.arange(n_ages), n_combos),
                **strat_cols,
                "Q": self.Q.ravel(),
            }
        )
        return self._df_Q

    @property
    def P(self) -> NDArray:
        """Population size matrix (n_joint_strata × n_ages)."""
        if isinstance(self.strats, Stratification):
            return self.strats.P

        if self._P is not None:
            return self._P

        self._P = np.round(self.Q * self.ref_age_dist[np.newaxis, :]).astype(int)
        return self._P

    @property
    def df_P(self) -> pd.DataFrame:
        """Population size as a long-form DataFrame."""
        if isinstance(self.strats, Stratification):
            return self.strats.df_P

        if self._df_P is not None:
            return self._df_P

        n_combos = len(self.coord_labels)
        n_ages = self.P.shape[1]
        strat_cols = {
            self.strats[j].name: np.repeat(
                [lbl[j] for lbl in self.coord_labels], n_ages
            )
            for j in range(len(self.strats))
        }
        self._df_P = pd.DataFrame(
            {
                "age": np.tile(np.arange(n_ages), n_combos),
                **strat_cols,
                "P": self.P.ravel(),
            }
        )
        return self._df_P
