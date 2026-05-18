"""ContactSummary dataclass — the standard return type of all ModelSummariser methods."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    import pandas as pd
    import xarray as xr


@dataclass(frozen=True)
class ContactSummary:
    """Summary statistics for a single stratum's contact matrix posterior.

    Attributes
    ----------
    lower : NDArray
        Lower credible-interval bound.
        Shape ``(A, A)`` for rate/cint, ``(A,)`` for mcint.
    central : NDArray
        Central tendency (mean or median). Same shape as ``lower``.
    upper : NDArray
        Upper credible-interval bound. Same shape as ``lower``.
    alpha : float
        Significance level used to compute the interval (e.g. 0.05 for 95 % CI).
    measure : str
        Central-tendency measure: ``"mean"`` or ``"median"``.
    """

    lower: NDArray
    central: NDArray
    upper: NDArray
    alpha: float
    measure: str

    def to_array(self) -> NDArray:
        """Return shape ``(3, ...)`` array with ``[lower, central, upper]`` along axis 0."""
        return np.stack([self.lower, self.central, self.upper])

    def to_dataframe(
        self,
        participant_ages: Optional[list] = None,
        contact_ages: Optional[list] = None,
    ) -> "pd.DataFrame":
        """Flatten to a long-form DataFrame.

        Parameters
        ----------
        participant_ages : list, optional
            Labels for the participant-age axis. Defaults to integer indices.
        contact_ages : list, optional
            Labels for the contact-age axis. Defaults to integer indices.
            Ignored for mcint (1-D) results.

        Returns
        -------
        pd.DataFrame
            Columns: ``age_part``, [``age_cnt``], ``lower``, ``central``, ``upper``.
        """
        import pandas as pd

        if self.central.ndim == 1:
            A = self.central.shape[0]
            ages = participant_ages if participant_ages is not None else list(range(A))
            return pd.DataFrame(
                {
                    "age_part": ages,
                    "lower": self.lower,
                    "central": self.central,
                    "upper": self.upper,
                }
            )

        A_p, A_c = self.central.shape
        p_ages = participant_ages if participant_ages is not None else list(range(A_p))
        c_ages = contact_ages if contact_ages is not None else list(range(A_c))
        rows = [
            {
                "age_part": p_ages[i],
                "age_cnt": c_ages[j],
                "lower": self.lower[i, j],
                "central": self.central[i, j],
                "upper": self.upper[i, j],
            }
            for i in range(A_p)
            for j in range(A_c)
        ]
        return pd.DataFrame(rows)

    def to_xarray(
        self,
        participant_ages: Optional[list] = None,
        contact_ages: Optional[list] = None,
    ) -> "xr.DataArray":
        """Return an ``xarray.DataArray`` with named dimensions.

        Parameters
        ----------
        participant_ages : list, optional
            Coordinate values for the ``age_part`` dimension.
        contact_ages : list, optional
            Coordinate values for the ``age_cnt`` dimension.
            Ignored for mcint (1-D) results.

        Returns
        -------
        xr.DataArray
            Dimensions: ``statistic`` × ``age_part`` [× ``age_cnt``].
        """
        import xarray as xr

        arr = self.to_array()

        if self.central.ndim == 1:
            A = self.central.shape[0]
            p_ages = participant_ages if participant_ages is not None else list(range(A))
            return xr.DataArray(
                arr,
                dims=["statistic", "age_part"],
                coords={
                    "statistic": ["lower", "central", "upper"],
                    "age_part": p_ages,
                },
            )

        A_p, A_c = self.central.shape
        p_ages = participant_ages if participant_ages is not None else list(range(A_p))
        c_ages = contact_ages if contact_ages is not None else list(range(A_c))
        return xr.DataArray(
            arr,
            dims=["statistic", "age_part", "age_cnt"],
            coords={
                "statistic": ["lower", "central", "upper"],
                "age_part": p_ages,
                "age_cnt": c_ages,
            },
        )
