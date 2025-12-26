from typing import Dict, Optional, Tuple, Union

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
    pop_constructor : PopulationConstructor
        Population structure defining stratifications and age distributions.
        Contains the reference age distribution and proportion matrix Q.
    n_participants : int
        Total number of participants to generate.

    Attributes
    ----------
    pop_constructor : PopulationConstructor
        The population constructor with stratification information.
    n_participants : int
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
    >>> pop_constructor = PopulationConstructor(gender_strat)

    >>> # Generate 1000 participants
    >>> pg = ParticipantGenerator(pop_constructor, n_participants=1000)
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
    >>> pop_constructor = PopulationConstructor([gender_strat, region_strat])

    >>> # Generate participants
    >>> pg = ParticipantGenerator(pop_constructor, n_participants=2000)
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

    def __init__(
        self, pop_constructor: PopulationConstructor, n_participants: int
    ) -> None:
        """
        Initialize ParticipantGenerator with population structure.

        Parameters
        ----------
        pop_constructor : PopulationConstructor
            Population structure defining stratifications and age distributions.
        n_participants : int
            Total number of participants to generate. Must be positive.

        Raises
        ------
        ValueError
            If n_participants is not positive.
        TypeError
            If pop_constructor is not a PopulationConstructor instance.
        """
        if not isinstance(pop_constructor, PopulationConstructor):
            raise TypeError(
                f"pop_constructor must be PopulationConstructor, got {type(pop_constructor)}"
            )

        if n_participants <= 0:
            raise ValueError(f"n_participants must be positive, got {n_participants}")

        self.pop_constructor = pop_constructor
        self.n_participants = n_participants

        # Extract population structure
        self._extract_population_structure()

    def _extract_population_structure(self) -> None:
        """Extract and validate population structure from PopulationConstructor."""
        # Get reference age distribution and normalize to proportions
        ref_age_dist = self.pop_constructor.ref_age_dist
        self.global_age_dist = ref_age_dist / ref_age_dist.sum()
        self.n_ages = len(self.global_age_dist)

        # Get population proportion matrix Q
        self.Q = self.pop_constructor.Q  # Shape: (n_strata, n_ages)
        self.n_strata = self.Q.shape[0]

        # Extract stratification metadata
        self._extract_stratification_info()

    def _extract_stratification_info(self) -> None:
        """Extract stratification variable names and labels."""
        # Check if single or multiple stratifications
        if isinstance(self.pop_constructor.strats, Stratification):
            # Single stratification
            self.strat_names = [self.pop_constructor.strats.name]
            self.strat_labels = self.pop_constructor.strats.labels
            self.is_multi_strat = False
        else:
            # Multiple stratifications
            self.strat_names = [strat.name for strat in self.pop_constructor.strats]
            self.strat_labels = self.pop_constructor.coord_labels
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
            Array of sampled ages (length n_participants).
        """
        ages = rng.choice(
            self.n_ages, size=self.n_participants, p=self.global_age_dist
        )
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
            Array of participant ages (length n_participants).
        rng : np.random.Generator
            Random number generator.

        Returns
        -------
        NDArray
            Array of sampled stratum indices (length n_participants).
        """
        strata = np.zeros(self.n_participants, dtype=int)

        for i, age in enumerate(ages):
            # Get conditional distribution over strata given this age
            probs_given_age = self.Q[:, age]

            # Normalize to ensure valid probability distribution
            probs_given_age = probs_given_age / probs_given_age.sum()

            # Sample stratum
            strata[i] = rng.choice(self.n_strata, p=probs_given_age)

        return strata

    def _map_strata_to_labels(self, strata: NDArray) -> Union[pd.Series, pd.DataFrame]:
        """
        Map stratum indices to stratification variable labels.

        Parameters
        ----------
        strata : NDArray
            Array of stratum indices (length n_participants).

        Returns
        -------
        pd.Series or pd.DataFrame
            If single stratification: Series with stratification values.
            If multiple stratifications: DataFrame with columns for each variable.
        """
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
            - 'id': Unique participant ID (1, 2, ..., n_participants)
            - 'age': Age group index (0 to n_ages - 1)
            - One column per stratification variable (e.g., 'gender', 'region')

        Examples
        --------
        >>> pg = ParticipantGenerator(pop_constructor, n_participants=500)
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
                "id": np.arange(1, self.n_participants + 1),
                "age": ages,
            }
        )

        # Append stratification columns
        if isinstance(strat_data, pd.Series):
            df[strat_data.name] = strat_data.values
        else:
            for col in strat_data.columns:
                df[col] = strat_data[col].values

        return df
