from dataclasses import dataclass
from itertools import product
from typing import List, Optional, Union

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from ._Stratification import Stratification


@dataclass
class PopulationConstructor:
    strats: Union[Stratification, List[Stratification]]

    _Q: Optional[NDArray] = None
    _P: Optional[NDArray] = None
    _df_Q: Optional[pd.DataFrame] = None
    _df_P: Optional[pd.DataFrame] = None

    def __post_init__(self):
        if isinstance(self.strats, list):
            self.ref_age_dist = self.strats[0].ref_age_dist
            self.coord_labels = list(product(*[strat.labels for strat in self.strats]))
            self.coord_codes = list(product(*[strat.codes for strat in self.strats]))
        else:
            self.ref_age_dist = self.strats.ref_age_dist
            self.coord_labels = self.strats.labels
            self.coord_codes = self.strats.codes

    @property
    def Q(self):
        if isinstance(self.strats, Stratification):
            return self.strats.Q
        else:
            if self._Q is not None:
                return self._Q

            grid_size = np.prod([strat.n_strata for strat in self.strats])
            Q = np.ones((grid_size, self.strats[0].ref_age_dist.shape[0]))
            for s, code_tuple in enumerate(self.coord_codes):
                for j, code in enumerate(code_tuple):
                    Q[s, :] *= self.strats[j].Q[code, :]

            self._Q = Q

            return self._Q

    @property
    def df_Q(self) -> pd.DataFrame:
        if isinstance(self.strats, Stratification):
            return self.strats.df_Q

        if self._df_Q is not None:
            return self._df_Q

        data = []
        for s, label_tuple in enumerate(self.coord_labels):
            for age_idx in range(self.Q.shape[1]):
                data.append(
                    {
                        "age": int(age_idx),
                        "Q": self.Q[s, age_idx],
                        **{
                            f"{self.strats[j].name}": label_tuple[j]
                            for j in range(len(label_tuple))
                        },
                    }
                )

        self._df_Q = pd.DataFrame(data)

        return self._df_Q

    @property
    def P(self) -> NDArray:
        if isinstance(self.strats, Stratification):
            return self.strats.P

        if self._P is not None:
            return self._P

        P_ref = self.strats[0].ref_age_dist

        self._P = np.round(self.Q * P_ref[np.newaxis, :]).astype(int)

        return self._P

    @property
    def df_P(self) -> pd.DataFrame:
        if isinstance(self.strats, Stratification):
            return self.strats.df_P

        if self._df_P is not None:
            return self._df_P

        data = []
        for s, label_tuple in enumerate(self.coord_labels):
            for age_idx in range(self.P.shape[1]):
                data.append(
                    {
                        "age": int(age_idx),
                        "P": self.P[s, age_idx],
                        **{
                            f"{self.strats[j].name}": label_tuple[j]
                            for j in range(len(label_tuple))
                        },
                    }
                )

        self._df_P = pd.DataFrame(data)

        return self._df_P
