from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass
class Subgroup:
    """
    Configuration for a population subgroup (used by MatrixGenerator and ContactGenerator).

    Parameters
    ----------
    n : int
        The number of participants in this subgroup.
    age_dist : NDArray
        Age distribution array for this subgroup.
        Each element represents the count or proportion of individuals in that age group.
        Will be normalized to proportions internally.
    mean_cint_margin : float
        Average marginal contact intensity for this subgroup.
    label : str, optional
        Label for the subgroup (used when multiple subgroups are provided).
    """

    n: int
    age_dist: NDArray
    mean_cint_margin: float
    label: str = None
