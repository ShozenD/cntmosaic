from __future__ import annotations

from itertools import product
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from ._PopulationConstructor import PopulationConstructor
from ._Stratification import Stratification


class ParticipantGenerator:
    """
    Generate synthetic participant data for a social contact survey.

    This class simulates a representative sample of participants from a stratified
    population. For each participant, an age is first sampled from the global
    population distribution, then a stratum is sampled conditional on that age
    using the population proportion matrix Q.

    The PopulationConstructor defines the joint population structure through:
    - Global age distribution (ref_age_dist)
    - Population proportion matrix Q[s, a]: probability of stratum s given age a
    - Stratification variables (e.g., gender, region, SES)

    Parameters
    ----------
    popcon : PopulationConstructor
        Population structure defining stratifications and age distributions.
        Contains the reference age distribution and proportion matrix Q.
    n_part : int
        Total number of participants to generate.

    Attributes
    ----------
    popcon : PopulationConstructor
        The population constructor with stratification information.
    n_part : int
        Total sample size.
    n_ages : int
        Number of age groups.
    n_strata : int
        Number of population strata.
    global_age_dist : NDArray
        Global population age distribution (proportions).
    Q : NDArray
        Population proportion matrix (n_strata, n_ages).
        Q[s, a] = P(stratum=s | age=a)
    strat_names : list of str
        Names of stratification variables.
    strat_labels : list
        Labels for each stratum (tuples for multiple stratifications).

    Examples
    --------
    **Example 1: Single stratification variable (Gender)**

    >>> import numpy as np
    >>> from cntmosaic.sim import Stratification, PopulationConstructor, ParticipantGenerator

    >>> # Define reference age distribution
    >>> ref_age_dist = np.array([1000, 1500, 2000, 1800, 1200])

    >>> # Create gender stratification
    >>> gender_strat = Stratification(
    ...     name='gender',
    ...     n_strata=2,
    ...     ref_age_dist=ref_age_dist,
    ...     labels=['Male', 'Female'],
    ...     seed=42
    ... )

    >>> # Build population constructor
    >>> popcon = PopulationConstructor(gender_strat)

    >>> # Generate 1000 participants
    >>> pg = ParticipantGenerator(popcon, n_part=1000)
    >>> df_participants = pg.generate(seed=123)

    >>> print(df_participants.head())
       id  age  gender
    0   1    2  Female
    1   2    4    Male
    2   3    3  Female
    3   4    1    Male
    4   5    3  Female

    >>> # Check age distribution reflects population
    >>> df_participants['age'].value_counts(normalize=True).sort_index()
    age
    0    0.12
    1    0.19
    2    0.27
    3    0.24
    4    0.18
    dtype: float64

    **Example 2: Multiple stratifications (Gender × Region)**

    >>> # Define stratifications
    >>> gender_strat = Stratification(
    ...     name='gender',
    ...     n_strata=2,
    ...     ref_age_dist=ref_age_dist,
    ...     labels=['Male', 'Female'],
    ...     seed=42
    ... )
    >>> region_strat = Stratification(
    ...     name='region',
    ...     n_strata=3,
    ...     ref_age_dist=ref_age_dist,
    ...     labels=['Urban', 'Suburban', 'Rural'],
    ...     seed=43
    ... )

    >>> # Build joint population
    >>> popcon = PopulationConstructor([gender_strat, region_strat])

    >>> # Generate participants
    >>> pg = ParticipantGenerator(popcon, n_part=2000)
    >>> df_participants = pg.generate(seed=456)

    >>> print(df_participants.head())
       id  age  gender     region
    0   1    3  Female      Urban
    1   2    2    Male  Suburban
    2   3    4  Female      Rural
    3   4    1  Female      Urban
    4   5    2    Male      Urban

    >>> # Check stratum distribution
    >>> df_participants.groupby(['gender', 'region']).size()
    gender  region
    Male    Urban       167
            Suburban    165
            Rural       168
    Female  Urban       333
            Suburban    334
            Rural       333
    dtype: int64

    Notes
    -----
    The sampling procedure is:
    1. Sample age a ~ Multinomial(global_age_dist)
    2. Sample stratum s ~ Multinomial(Q[:, a])
    3. Map stratum index to stratification variable values

    This ensures the generated sample is representative of the population
    structure defined by the PopulationConstructor.
    """

    def __init__(self, popcon: PopulationConstructor, n_part: int) -> None:
        """
        Initialize ParticipantGenerator with population structure.

        Parameters
        ----------
        popcon : PopulationConstructor
            Population structure defining stratifications and age distributions.
        n_part : int
            Total number of participants to generate. Must be positive.

        Raises
        ------
        ValueError
            If n_part is not positive.
        TypeError
            If popcon is not a PopulationConstructor instance.
        """
        if not isinstance(popcon, PopulationConstructor):
            raise TypeError(f"popcon must be PopulationConstructor, got {type(popcon)}")

        if n_part <= 0:
            raise ValueError(f"n_part must be positive, got {n_part}")

        self.popcon = popcon
        self.n_part = n_part

        # Extract population structure
        self._extract_population_structure()

    @classmethod
    def from_df(
        cls,
        df: pd.DataFrame,
        n_part: int,
        strat_var_cols: Optional[List[str]] = None,
        age_col: str = "age",
        pop_col: str = "P",
    ) -> ParticipantGenerator:
        """
        Create a ParticipantGenerator from a DataFrame with population sizes.

        This alternative constructor allows initialization directly from a
        DataFrame containing population counts by age and stratification
        variables, without needing to create Stratification and
        PopulationConstructor objects.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame containing population sizes. Must have:
            - An age column (default: "age")
            - A population size column (default: "P")
            - Zero or more stratification variable columns
        n_part : int
            Total number of participants to generate. Must be positive.
        strat_var_cols : list of str, optional
            Names of columns to use as stratification variables.
            If None, all columns except age_col and pop_col are used.
        age_col : str, default "age"
            Name of the column containing age values.
        pop_col : str, default "P"
            Name of the column containing population sizes.

        Returns
        -------
        ParticipantGenerator
            Initialized generator ready to produce participant samples.

        Raises
        ------
        ValueError
            If required columns are missing or if n_part is not positive.

        Examples
        --------
        >>> import pandas as pd
        >>> import numpy as np
        >>> from cntmosaic.sim import ParticipantGenerator

        >>> # Create population DataFrame
        >>> df_pop = pd.DataFrame({
        ...     'age': [0, 0, 1, 1, 2, 2],
        ...     'gender': ['Male', 'Female', 'Male', 'Female', 'Male', 'Female'],
        ...     'P': [100, 110, 150, 160, 120, 130]
        ... })

        >>> # Create generator from DataFrame
        >>> pg = ParticipantGenerator.from_df(df_pop, n_part=500, strat_var_cols=['gender'])
        >>> df_participants = pg.generate(seed=42)

        >>> print(df_participants.head())
           id  age  gender
        0   1    1  Female
        1   2    2  Female
        2   3    1    Male
        3   4    0  Female
        4   5    2    Male

        Notes
        -----
        The DataFrame should have one row per (age, stratum) combination.
        The population proportion matrix Q is computed as:
        Q[s, a] = P[s, a] / sum_s(P[s, a])

        This represents the probability of being in stratum s given age a.
        """
        if n_part <= 0:
            raise ValueError(f"n_part must be positive, got {n_part}")

        # Validate required columns
        if age_col not in df.columns:
            raise ValueError(f"Age column '{age_col}' not found in DataFrame")
        if pop_col not in df.columns:
            raise ValueError(f"Population column '{pop_col}' not found in DataFrame")

        # Determine stratification variables
        if strat_var_cols is None:
            strat_var_cols = [c for c in df.columns if c not in [age_col, pop_col]]

        for var in strat_var_cols:
            if var not in df.columns:
                raise ValueError(
                    f"Stratification variable '{var}' not found in DataFrame"
                )

        # Create instance without calling __init__
        instance = cls.__new__(cls)
        instance.popcon = None  # No PopulationConstructor in this mode
        instance.n_part = n_part

        # Extract population structure from DataFrame
        instance._extract_population_structure_from_df(
            df, strat_var_cols, age_col, pop_col
        )

        return instance

    def _extract_population_structure_from_df(
        self,
        df: pd.DataFrame,
        strat_var_cols: List[str],
        age_col: str,
        pop_col: str,
    ) -> None:
        """
        Extract population structure from a DataFrame.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame with population sizes.
        strat_var_cols : list of str
            Names of stratification variable columns.
        age_col : str
            Name of age column.
        pop_col : str
            Name of population size column.
        """
        # Get unique ages (sorted)
        ages = np.sort(df[age_col].unique())
        self.n_ages = len(ages)
        age_to_idx = {age: idx for idx, age in enumerate(ages)}

        # Handle case with no stratification variables
        if len(strat_var_cols) == 0:
            # Unstratified: single stratum
            self.n_strata = 1
            self.strat_names = []
            self.strat_labels = [()]
            self.is_multi_strat = False

            # Global age distribution
            ref_age_dist = np.zeros(self.n_ages)
            for _, row in df.iterrows():
                age_idx = age_to_idx[row[age_col]]
                ref_age_dist[age_idx] = row[pop_col]

            self.global_age_dist = ref_age_dist / ref_age_dist.sum()

            # Q matrix is trivially all ones (single stratum)
            self.Q = np.ones((1, self.n_ages))
            return

        # Get unique categories for each stratification variable
        strat_categories = {var: sorted(df[var].unique()) for var in strat_var_cols}

        # Build stratum labels as tuples (for multiple vars) or single values
        if len(strat_var_cols) == 1:
            self.is_multi_strat = False
            self.strat_names = strat_var_cols
            self.strat_labels = strat_categories[strat_var_cols[0]]
            strat_tuples = [(cat,) for cat in self.strat_labels]
        else:
            self.is_multi_strat = True
            self.strat_names = strat_var_cols
            # Generate all combinations of categories
            strat_tuples = list(
                product(*[strat_categories[var] for var in strat_var_cols])
            )
            self.strat_labels = strat_tuples

        self.n_strata = len(strat_tuples)

        # Create mapping from stratum tuple to index
        strat_to_idx = {tup: idx for idx, tup in enumerate(strat_tuples)}

        # Build population matrix P[s, a]
        P_matrix = np.zeros((self.n_strata, self.n_ages))

        for _, row in df.iterrows():
            age_idx = age_to_idx[row[age_col]]
            strat_tuple = tuple(row[var] for var in strat_var_cols)
            if strat_tuple in strat_to_idx:
                strat_idx = strat_to_idx[strat_tuple]
                P_matrix[strat_idx, age_idx] = row[pop_col]

        # Compute global age distribution (sum across strata)
        ref_age_dist = P_matrix.sum(axis=0)
        self.global_age_dist = ref_age_dist / ref_age_dist.sum()

        # Compute Q matrix: Q[s, a] = P[s, a] / sum_s(P[s, a])
        # Handle potential division by zero for ages with no population
        with np.errstate(divide="ignore", invalid="ignore"):
            self.Q = P_matrix / ref_age_dist[np.newaxis, :]
            # Set Q to uniform for ages with zero population
            zero_pop_ages = ref_age_dist == 0
            if zero_pop_ages.any():
                self.Q[:, zero_pop_ages] = 1.0 / self.n_strata

    def _extract_population_structure(self) -> None:
        """Extract and validate population structure from PopulationConstructor."""
        # Get reference age distribution and normalize to proportions
        ref_age_dist = self.popcon.ref_age_dist
        self.global_age_dist = ref_age_dist / ref_age_dist.sum()
        self.n_ages = len(self.global_age_dist)

        # Get population proportion matrix Q
        self.Q = self.popcon.Q  # Shape: (n_strata, n_ages)
        self.n_strata = self.Q.shape[0]

        # Extract stratification metadata
        self._extract_stratification_info()

    def _extract_stratification_info(self) -> None:
        """Extract stratification variable names and labels."""
        # Check if single or multiple stratifications
        if isinstance(self.popcon.strats, Stratification):
            # Single stratification
            self.strat_names = [self.popcon.strats.name]
            self.strat_labels = self.popcon.strats.labels
            self.is_multi_strat = False
        else:
            # Multiple stratifications
            self.strat_names = [strat.name for strat in self.popcon.strats]
            self.strat_labels = self.popcon.coord_labels
            self.is_multi_strat = True

    def _sample_ages(self, rng: np.random.Generator) -> NDArray:
        """
        Sample ages from global population distribution.

        Parameters
        ----------
        rng : np.random.Generator
            Random number generator.

        Returns
        -------
        NDArray
            Array of sampled ages (length n_part).
        """
        ages = rng.choice(self.n_ages, size=self.n_part, p=self.global_age_dist)
        return ages

    def _sample_strata_given_ages(
        self, ages: NDArray, rng: np.random.Generator
    ) -> NDArray:
        """
        Sample strata conditional on ages using Q matrix.

        For each age a, samples stratum from categorical distribution Q[:, a].

        Parameters
        ----------
        ages : NDArray
            Array of participant ages (length n_part).
        rng : np.random.Generator
            Random number generator.

        Returns
        -------
        NDArray
            Array of sampled stratum indices (length n_part).
        """
        strata = np.zeros(self.n_part, dtype=int)

        for i, age in enumerate(ages):
            # Get conditional distribution over strata given this age
            probs_given_age = self.Q[:, age]

            # Normalize to ensure valid probability distribution
            probs_given_age = probs_given_age / probs_given_age.sum()

            # Sample stratum
            strata[i] = rng.choice(self.n_strata, p=probs_given_age)

        return strata

    def _map_strata_to_labels(
        self, strata: NDArray
    ) -> Optional[Union[pd.Series, pd.DataFrame]]:
        """
        Map stratum indices to stratification variable labels.

        Parameters
        ----------
        strata : NDArray
            Array of stratum indices (length n_part).

        Returns
        -------
        pd.Series, pd.DataFrame, or None
            If no stratification: None.
            If single stratification: Series with stratification values.
            If multiple stratifications: DataFrame with columns for each variable.
        """
        # Handle unstratified case
        if len(self.strat_names) == 0:
            return None

        if not self.is_multi_strat:
            # Single stratification: return Series
            labels = [self.strat_labels[s] for s in strata]
            return pd.Series(labels, name=self.strat_names[0])
        else:
            # Multiple stratifications: return DataFrame
            data = {}
            for j, strat_name in enumerate(self.strat_names):
                labels = [self.strat_labels[s][j] for s in strata]
                data[strat_name] = labels
            return pd.DataFrame(data)

    def generate(self, seed: Optional[int] = None) -> pd.DataFrame:
        """
        Generate synthetic participant data.

        Sampling procedure:
        1. Sample age a from global population distribution
        2. Sample stratum s from Q[:, a] (conditional on age)
        3. Map stratum to stratification variable values

        Parameters
        ----------
        seed : int, optional
            Random seed for reproducibility.

        Returns
        -------
        pd.DataFrame
            DataFrame with columns:
            - 'id': Unique participant ID (1, 2, ..., n_part)
            - 'age': Age group index (0 to n_ages - 1)
            - One column per stratification variable (e.g., 'gender', 'region')

        Examples
        --------
        >>> pg = ParticipantGenerator(popcon, n_part=500)
        >>> df = pg.generate(seed=42)
        >>> df.columns
        Index(['id', 'age', 'gender'], dtype='object')
        >>> len(df)
        500
        """
        rng = np.random.default_rng(seed)

        # Step 1: Sample ages from global distribution
        ages = self._sample_ages(rng)

        # Step 2: Sample strata conditional on ages
        strata = self._sample_strata_given_ages(ages, rng)

        # Step 3: Map strata to labels
        strat_data = self._map_strata_to_labels(strata)

        # Build final DataFrame
        df = pd.DataFrame(
            {
                "id": np.arange(1, self.n_part + 1),
                "age": ages,
            }
        )

        # Append stratification columns (if any)
        if strat_data is None:
            pass  # No stratification variables
        elif isinstance(strat_data, pd.Series):
            df[strat_data.name] = strat_data.values
        else:
            for col in strat_data.columns:
                df[col] = strat_data[col].values

        return df
