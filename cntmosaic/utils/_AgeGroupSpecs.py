from __future__ import annotations

import numpy as np
import pandas as pd


class AgeGroupSpecs:
    """Age group specifications for contact matrix binning.

    Supports two construction modes:

    **Fixed-step / cut-point mode:**

        AgeGroupSpecs(min, max, step=5)
        AgeGroupSpecs(min, max, cuts=[18, 45, 65])

    **Explicit-bounds mode:**

        AgeGroupSpecs(age_min=[0, 5, 18], age_max=[4, 17, 80])

    In explicit-bounds mode ``age_min`` and ``age_max`` are parallel integer
    lists giving the *inclusive* lower and upper bound of each age group.
    They must be the same length, sorted ascending, and ``age_max[i] >=
    age_min[i]`` for every index.

    Parameters
    ----------
    min : int, optional
        Minimum age (inclusive). Required in fixed-step / cut-point mode.
    max : int, optional
        Maximum age (inclusive). Required in fixed-step / cut-point mode.
    step : int, optional
        Width of each age group when *cuts* is not provided. Default 5.
    cuts : list or np.ndarray, optional
        Internal boundary ages that start each new bin (exclusive of the
        first bin which starts at *min*).
    age_min : list of int, keyword-only, optional
        Sorted inclusive lower bounds for each age group.
    age_max : list of int, keyword-only, optional
        Sorted inclusive upper bounds for each age group.

    Attributes
    ----------
    left : list of int
        Inclusive lower bound of each bin.
    right : list of int
        Inclusive upper bound of each bin.
    bins : list of int
        Bin edges for use with ``pd.cut(..., bins=..., right=False)``.
        Equals ``left`` with the exclusive upper bound of the last bin appended.
    min, max : int
        Overall minimum and maximum age.
    range : int
        Total number of single-year ages covered (``max - min + 1``).
    bin_sizes : np.ndarray
        Number of single-year ages in each bin.
    cell_sizes : np.ndarray
        Outer product of ``bin_sizes`` (used for contact-rate weighting).
    """

    def __init__(
        self,
        min: int | None = None,
        max: int | None = None,
        step: int = 5,
        cuts: list | np.ndarray | None = None,
        *,
        age_min: list | None = None,
        age_max: list | None = None,
    ):
        if age_min is not None or age_max is not None:
            self._init_from_bounds(age_min, age_max)
        else:
            if min is None or max is None:
                raise ValueError(
                    "Either (min, max) or (age_min, age_max) must be provided."
                )
            self._init_from_step(min, max, step, cuts)

    # ------------------------------------------------------------------
    # Construction helpers
    # ------------------------------------------------------------------

    def _init_from_step(self, min, max, step, cuts):
        self.min = min
        self.max = max
        self.range = max - min + 1
        self.step = step
        self.cuts = cuts
        self.get_bounds_left()
        self.get_bounds_right()
        self.get_bin_sizes()
        self.get_cell_sizes()

    def _init_from_bounds(self, age_min, age_max):
        if age_min is None or age_max is None:
            raise ValueError("Both age_min and age_max must be provided together.")
        age_min = list(age_min)
        age_max = list(age_max)
        if len(age_min) != len(age_max):
            raise ValueError("age_min and age_max must have the same length.")
        if age_min != sorted(age_min):
            raise ValueError("age_min must be sorted in ascending order.")
        if age_max != sorted(age_max):
            raise ValueError("age_max must be sorted in ascending order.")
        for i, (lo, hi) in enumerate(zip(age_min, age_max)):
            if hi < lo:
                raise ValueError(
                    f"age_max[{i}]={hi} must be >= age_min[{i}]={lo}."
                )

        self.left = age_min
        self.right = age_max
        self.min = age_min[0]
        self.max = age_max[-1]
        self.range = self.max - self.min + 1
        self.step = None
        self.cuts = None
        self.bin_sizes = np.array([b - a + 1 for a, b in zip(age_min, age_max)])
        self.cell_sizes = np.outer(self.bin_sizes, self.bin_sizes)

    # ------------------------------------------------------------------
    # Bounds computation (fixed-step / cut-point mode only)
    # ------------------------------------------------------------------

    def get_bounds_left(self) -> list:
        """Return (and cache) the inclusive lower bound of each bin."""
        if not hasattr(self, "left"):
            if self.cuts is not None:
                self.left = [self.min] + list(self.cuts)
            else:
                self.left = list(range(self.min, self.max, self.step))
        return self.left

    def get_bounds_right(self) -> list:
        """Return (and cache) the inclusive upper bound of each bin."""
        if not hasattr(self, "right"):
            if self.cuts is not None:
                self.right = list(np.asarray(self.cuts) - 1) + [self.max]
            else:
                self.right = (
                    list(np.asarray(range(self.min + self.step, self.max, self.step)) - 1)
                    + [self.max]
                )
        return self.right

    def get_bin_sizes(self) -> np.ndarray:
        """Return (and cache) the number of single-year ages in each bin."""
        if not hasattr(self, "bin_sizes"):
            self.bin_sizes = np.diff(np.append(self.left, self.max + 1))
        return self.bin_sizes

    def get_cell_sizes(self) -> np.ndarray:
        """Return (and cache) the outer product of ``bin_sizes``."""
        if not hasattr(self, "cell_sizes"):
            self.cell_sizes = np.outer(self.bin_sizes, self.bin_sizes)
        return self.cell_sizes

    def get_cuts(self) -> list:
        """Return ``left`` with the exclusive upper bound of the last bin appended.

        The result is identical to the ``bins`` property and can be passed
        directly to ``pd.cut(..., bins=..., right=False)``.
        """
        return self.left + [self.right[-1] + 1]

    @property
    def bins(self) -> list:
        """Bin edges for ``pd.cut(..., bins=..., right=False)``.

        Equals ``left`` with ``right[-1] + 1`` appended so that every age up
        to and including ``max`` falls within a half-open interval ``[a, b)``.
        """
        return self.get_cuts()

    def cut(self, series: pd.Series) -> pd.Series:
        """Bin *series* into the age groups defined by this spec.

        Parameters
        ----------
        series : pd.Series
            Integer ages to bin.

        Returns
        -------
        pd.Series
            Categorical series of half-open intervals ``[a, b)``, one per
            element of *series*.

        Examples
        --------
        >>> specs = AgeGroupSpecs(0, 14, 5)
        >>> df["part_age_grp"] = specs.cut(df["part_age"])
        """
        return pd.cut(series, bins=self.bins, right=False)

    def replace(self, *, age_min: list, age_max: list) -> "AgeGroupSpecs":
        """Return a new instance with updated bounds, leaving this one unchanged.

        Parameters
        ----------
        age_min : list of int
            New inclusive lower bounds for each bin.
        age_max : list of int
            New inclusive upper bounds for each bin.

        Returns
        -------
        AgeGroupSpecs
            A fresh instance constructed from the provided bounds.
        """
        return AgeGroupSpecs(age_min=age_min, age_max=age_max)
