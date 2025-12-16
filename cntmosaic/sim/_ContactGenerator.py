from typing import Dict, Tuple, Union

import numpy as np
import pandas as pd
from numpy.typing import NDArray


class ContactGenerator:
    """
    Generate synthetic contact count data from contact intensity matrices and participant data.

    This class simulates contact counts for participants based on contact intensity matrices,
    supporting both Poisson and Negative Binomial sampling distributions with optional
    individual random effects.

    The generator works with outputs from:
    - ParticipantGenerator: provides participant demographics (df_part)
    - MatrixGenerator: provides contact intensity matrices (cint_matrices)

    Three modes of operation:
    1. Single population: One matrix for all participants
    2. Partial case: One matrix per subgroup (contacts with general population)
    3. Full case: All pairwise matrices (contacts between all subgroup pairs)

    Examples
    --------
    >>> import numpy as np
    >>> from cntmosaic.sim import MatrixGenerator, ParticipantGenerator, ContactGenerator, Subgroup
    >>> from cntmosaic.datasets import load_template_patterns

    **Example 1: Single population**

    >>> templates = load_template_patterns('USA')
    >>> subgroup = Subgroup(n=100, age_dist=np.array([100, 200, 300, 400, 500]), mean_cint_margin=15.0)
    >>>
    >>> # Generate participants and matrix
    >>> part_gen = ParticipantGenerator(subgroup)
    >>> participants = part_gen.generate(seed=42)
    >>>
    >>> mat_gen = MatrixGenerator(templates)
    >>> contact_matrix = mat_gen.generate_single(subgroup, seed=42)
    >>>
    >>> # Generate contacts
    >>> contact_gen = ContactGenerator(participants, contact_matrix)
    >>> contacts = contact_gen.generate(seed=42)
    >>> print(contacts.head())
       id  age_cnt   y
    0   1        2  12
    1   1        3   8
    2   1        4   3
    3   2        1   5
    4   2        3  15

    **Example 2: Multiple subgroups (Partial case)**

    >>> subgroups = [
    ...     Subgroup(n=100, age_dist=np.array([150, 250, 350, 250, 100]), mean_cint_margin=18.0, label='urban'),
    ...     Subgroup(n=50, age_dist=np.array([100, 150, 200, 300, 250]), mean_cint_margin=12.0, label='rural')
    ... ]
    >>>
    >>> part_gen = ParticipantGenerator(subgroups)
    >>> participants = part_gen.generate(seed=42)
    >>>
    >>> mat_gen = MatrixGenerator(templates)
    >>> matrices = mat_gen.generate_partial(subgroups, seed=42)
    >>>
    >>> contact_gen = ContactGenerator(participants, matrices)
    >>> contacts = contact_gen.generate(seed=42)
    >>> print(contacts.head())
       id  age_cnt   y
    0   1        2  15
    1   1        3   9
    2   2        1   6
    3   3        0  12
    4   3        2   8
    >>> # Note: No 'subgroup' column because contacts are with general population

    **Example 3: Full case (All subgroup interactions)**

    >>> subgroups = [
    ...     Subgroup(n=100, age_dist=np.array([150, 250, 350, 250, 100]), mean_cint_margin=18.0, label='urban'),
    ...     Subgroup(n=50, age_dist=np.array([100, 150, 200, 300, 250]), mean_cint_margin=12.0, label='rural')
    ... ]
    >>>
    >>> part_gen = ParticipantGenerator(subgroups)
    >>> participants = part_gen.generate(seed=42)
    >>>
    >>> mat_gen = MatrixGenerator(templates)
    >>> matrices = mat_gen.generate_full(subgroups, seed=42)  # Full case
    >>>
    >>> contact_gen = ContactGenerator(participants, matrices)
    >>> contacts = contact_gen.generate(seed=42)
    >>> print(contacts.head())
       id  age_cnt subgroup_cnt   y
    0   1        2        urban  12
    1   1        3        urban   8
    2   1        1        rural   3
    3   2        0        urban   7
    4   2        2        rural   5
    >>>
    >>> # Urban participants have contacts with both urban and rural subgroups
    >>> print(contacts[contacts['id'] == 1]['subgroup_cnt'].value_counts())
    subgroup_cnt
    urban    3
    rural    2
    """

    ALLOWED_MODELS = ["poisson", "negbin"]

    def __init__(
        self,
        df_part: pd.DataFrame,
        cint_matrices: Union[
            NDArray, Dict[Union[int, str], NDArray], Dict[Tuple, NDArray]
        ],
        model: str = "poisson",
        odisp: float = None,
        random_effects: bool = False,
        random_effects_shape: float = 5.0,
        random_effects_rate: float = 5.0,
    ):
        """
        Initialize ContactGenerator with participant data and contact matrices.

        Parameters
        ----------
        df_part : pd.DataFrame
                Participant data from ParticipantGenerator.generate().
                Must contain columns:
                - 'id': Unique participant identifier
                - 'age_group': Age group index
                - 'subgroup': Subgroup label (only if multiple subgroups)

        cint_matrices : NDArray or dict
                Contact intensity matrices from MatrixGenerator.
                Can be:
                - NDArray: Single matrix for homogeneous population (from generate_single)
                - dict mapping labels to NDArray: Subgroup-specific matrices (from generate_partial)
                  Keys must match the 'subgroup' values in df_part
                - dict mapping (source, target) tuples to NDArray: Full case matrices (from generate_full)
                  Only diagonal matrices (k, k) will be used for contact generation

        model : str, default='poisson'
                Contact count sampling distribution. Options:
                - 'poisson': Poisson distribution (equidispersed)
                - 'negbin': Negative binomial distribution (overdispersed)

        odisp : float, optional
                Overdispersion parameter for negative binomial model.
                Required if model='negbin'. Higher values = more overdispersion.
                Typical range: 0.1 to 2.0

        random_effects : bool, default=False
                Whether to include individual-level random effects.
                If True, multiplies contact intensities by Gamma(shape, rate) random variates
                to model heterogeneity in individual contact behavior.

        random_effects_shape : float, default=5.0
                Shape parameter for Gamma distribution of random effects.
                Mean of random effects = shape / rate.

        random_effects_rate : float, default=5.0
                Rate parameter for Gamma distribution of random effects.
                With shape=rate=5, mean=1 and variance=0.2.

        Raises
        ------
        ValueError
                If model is not in ALLOWED_MODELS
                If odisp is not provided when model='negbin'
                If cint_matrices structure doesn't match df_part subgroups

        Notes
        -----
        For the full case (generate_full output), only within-subgroup contacts are
        generated, using the diagonal matrices M_{k,k}. To generate between-subgroup
        contacts, use the matrices with different source and target indices.
        """
        if model not in self.ALLOWED_MODELS:
            raise ValueError(
                f"Model '{model}' is not supported. "
                f"Allowed models: {self.ALLOWED_MODELS}"
            )

        if model == "negbin" and odisp is None:
            raise ValueError(
                "Overdispersion parameter 'odisp' must be provided for negative binomial model."
            )

        self.df_part = df_part
        self.cint_matrices = cint_matrices
        self.model = model
        self.odisp = odisp
        self.random_effects = random_effects
        self.random_effects_shape = random_effects_shape
        self.random_effects_rate = random_effects_rate

        # Validate and parse matrix structure
        self._validate_matrices()

    def _validate_matrices(self) -> None:
        """Validate that matrices match participant data structure."""
        has_subgroups = "subgroup" in self.df_part.columns

        # Single population case
        if isinstance(self.cint_matrices, np.ndarray):
            if has_subgroups:
                raise ValueError(
                    "df_part contains multiple subgroups but cint_matrices is a single matrix. "
                    "Expected a dictionary of matrices."
                )
            self.is_full_case = False
            return

        # Multiple subgroups case
        if isinstance(self.cint_matrices, dict):
            if not has_subgroups:
                raise ValueError(
                    "cint_matrices is a dictionary but df_part has no 'subgroup' column. "
                    "Expected a single matrix."
                )

            # Check if this is a full case (tuple keys) or partial case (scalar keys)
            first_key = next(iter(self.cint_matrices.keys()))
            self.is_full_case = isinstance(first_key, tuple)

            if self.is_full_case:
                # Full case - validate all subgroups have all required matrices
                subgroups = list(self.df_part["subgroup"].unique())

                # Check that we have all (k, l) pairs
                for k in subgroups:
                    for l in subgroups:
                        if (k, l) not in self.cint_matrices:
                            raise ValueError(
                                f"Missing matrix for ({k}, {l}) in full case matrices"
                            )

                self.subgroup_labels = subgroups
            else:
                # Partial case - use matrices as-is
                subgroups = set(self.df_part["subgroup"].unique())
                matrix_labels = set(
                    [label.replace("->All", "") for label in self.cint_matrices.keys()]
                )

                if subgroups != matrix_labels:
                    raise ValueError(
                        f"Mismatch between subgroups in df_part {subgroups} "
                        f"and matrix labels {matrix_labels}"
                    )

                self.subgroup_labels = list(subgroups)
            return

        raise TypeError(
            f"cint_matrices must be NDArray or dict, got {type(self.cint_matrices)}"
        )

    def _generate_single(
        self,
        df_part: pd.DataFrame,
        cint_matrix: NDArray,
        rng: np.random.Generator,
        target_label: Union[str, int] = None,
    ) -> pd.DataFrame:
        """
        Generate contact counts for a single group of participants.

        Parameters
        ----------
        df_part : pd.DataFrame
                Participant data for this group
        cint_matrix : NDArray
                Contact intensity matrix (A×A)
        rng : np.random.Generator
                Random number generator
        target_label : str or int, optional
                Subgroup label of contacts (for full case only)
                If None, contacts are with general population (single or partial case)

        Returns
        -------
        pd.DataFrame
                Contact data with columns:
                - ['id', 'age_cnt', 'y'] for single or partial case
                - ['id', 'age_cnt', 'subgroup_cnt', 'y'] for full case
        """
        # Get age groups for all participants
        age_groups = df_part["age_group"].astype(int).values

        # Extract contact intensities for each participant's age group
        # lambda_[i, j] = expected contacts from participant i to age group j
        lambda_ = cint_matrix[age_groups, :]

        # Apply individual random effects if requested
        if self.random_effects:
            gamma_effects = rng.gamma(
                shape=self.random_effects_shape,
                scale=1.0 / self.random_effects_rate,
                size=lambda_.shape,
            )
            lambda_ = lambda_ * gamma_effects

        # Sample contact counts
        if self.model == "poisson":
            samples = rng.poisson(lambda_)

        elif self.model == "negbin":
            # Negative binomial parameterization: n_success, p_success
            # Mean = n * (1-p) / p, where we want Mean = lambda_
            # With overdispersion odisp, n = 1/odisp
            n_success = 1.0 / self.odisp
            p_success = n_success / (n_success + lambda_)
            samples = rng.negative_binomial(n_success, p_success)

        # Convert to long format: one row per (participant, age_group) pair with contact
        data = []
        participant_ids = df_part["id"].values

        for i, participant_id in enumerate(participant_ids):
            # Find age groups where this participant has contacts
            age_groups_with_contacts = np.where(samples[i, :] > 0)[0]

            for age_cnt in age_groups_with_contacts:
                contact_count = samples[i, age_cnt]

                # Determine columns based on case
                if target_label is not None:
                    # Full case: include subgroup_cnt
                    data.append([participant_id, age_cnt, target_label, contact_count])
                else:
                    # Single or partial case: no subgroup info
                    data.append([participant_id, age_cnt, contact_count])

        # Create DataFrame
        if target_label is not None:
            columns = ["id", "age_cnt", "subgroup_cnt", "y"]
        else:
            columns = ["id", "age_cnt", "y"]

        return pd.DataFrame(data, columns=columns)

    def generate(self, seed: int = None) -> pd.DataFrame:
        """
        Generate contact count data for all participants.

        Parameters
        ----------
        seed : int, optional
                Random seed for reproducibility

        Returns
        -------
        pd.DataFrame
                Contact data with columns:
                - 'id': Participant identifier (matches df_part)
                - 'age_cnt': Age group of contact
                - 'y': Number of contacts with this age group
                - 'subgroup_cnt': Subgroup of contact (only for full case)

                Each row represents contacts between a participant and an age group.
                Only non-zero contact counts are included.

                Note: For partial case, there is no 'subgroup' column because contacts
                are with the general population, not specific subgroups.

        Examples
        --------
        >>> # Single population
        >>> contacts = contact_gen.generate(seed=42)
        >>> print(contacts.head())
           id  age_cnt   y
        0   1        2  12
        1   1        3   8
        2   2        1   5

        >>> # Partial case (multiple subgroups, contacts with general population)
        >>> contacts = contact_gen.generate(seed=42)
        >>> print(contacts.head())
           id  age_cnt   y
        0   1        2  15
        1   1        3   9
        2   2        1   6
        3   3        0  12
        4   3        2   8

        >>> # Full case (all subgroup interactions)
        >>> contacts = contact_gen.generate(seed=42)
        >>> print(contacts.head())
           id  age_cnt subgroup_cnt   y
        0   1        2        urban  12
        1   1        3        urban   8
        2   1        1        rural   3
        3   2        0        urban   7
        4   2        2        rural   5

        Notes
        -----
        For participants with many contacts, the output can be large. Consider
        aggregating by age group or filtering rare contact patterns for analysis.

        In the full case, each participant can have contacts with all subgroups.
        For example, an urban participant will have contacts generated using both
        the ('urban', 'urban') and ('urban', 'rural') matrices.
        """
        rng = np.random.default_rng(seed)

        # Single population case
        if isinstance(self.cint_matrices, np.ndarray):
            return self._generate_single(self.df_part, self.cint_matrices, rng)

        # Multiple subgroups - Partial case
        if not self.is_full_case:
            dfs = []
            for label in self.subgroup_labels:
                # Filter participants for this subgroup
                df_part_sub = self.df_part[self.df_part["subgroup"] == label].copy()

                # Generate contacts for this subgroup (with general population)
                df_contacts = self._generate_single(
                    df_part_sub,
                    self.cint_matrices[label + "->All"],
                    rng,
                    target_label=None,  # No target label for partial case
                )
                dfs.append(df_contacts)

            # Combine all subgroups
            return pd.concat(dfs, ignore_index=True)

        # Multiple subgroups - Full case
        dfs = []
        for source_label in self.subgroup_labels:
            # Filter participants for this source subgroup
            df_part_source = self.df_part[
                self.df_part["subgroup"] == source_label
            ].copy()

            # Generate contacts with each target subgroup
            for target_label in self.subgroup_labels:
                # Get the appropriate contact matrix
                matrix = self.cint_matrices[(source_label, target_label)]

                # Generate contacts from source to target
                df_contacts = self._generate_single(
                    df_part_source, matrix, rng, target_label=target_label
                )

                dfs.append(df_contacts)

        # Combine all source-target pairs
        return pd.concat(dfs, ignore_index=True)

    def summarize_contacts(self, contacts: pd.DataFrame = None) -> pd.DataFrame:
        """
        Summarize contact patterns across participants.

        Parameters
        ----------
        contacts : pd.DataFrame, optional
                Contact data from generate(). If None, will call generate() with default seed.

        Returns
        -------
        pd.DataFrame
                Summary statistics with columns:
                - For single/partial case: participant-level statistics by subgroup
                - For full case: contact totals by source and target subgroup pairs

        Examples
        --------
        >>> # Single case
        >>> contacts = contact_gen.generate(seed=42)
        >>> summary = contact_gen.summarize_contacts(contacts)
        >>> print(summary)
          total_participants  total_contacts  mean_contacts  median_contacts
        0                100            1500          15.00            14.0

        >>> # Partial case
        >>> contacts = contact_gen.generate(seed=42)
        >>> summary = contact_gen.summarize_contacts(contacts)
        >>> print(summary)
          subgroup  total_participants  total_contacts  mean_contacts  median_contacts
        0    urban                 100            1800          18.00            17.0
        1    rural                  50             600          12.00            11.0

        >>> # Full case
        >>> contacts = contact_gen.generate(seed=42)
        >>> summary = contact_gen.summarize_contacts(contacts)
        >>> print(summary)
          subgroup subgroup_cnt  total_contacts  mean_contacts
        0    urban        urban            1200          12.00
        1    urban        rural             300           3.00
        2    rural        urban             200           4.00
        3    rural        rural             400           8.00
        """
        if contacts is None:
            contacts = self.generate()

        # Single population case
        if not hasattr(self, "is_full_case"):
            participant_totals = contacts.groupby("id")["y"].sum()

            summary = pd.DataFrame(
                {
                    "total_participants": [len(participant_totals)],
                    "total_contacts": [participant_totals.sum()],
                    "mean_contacts": [participant_totals.mean()],
                    "median_contacts": [participant_totals.median()],
                    "std_contacts": [participant_totals.std()],
                    "min_contacts": [participant_totals.min()],
                    "max_contacts": [participant_totals.max()],
                }
            )
            return summary

        # Full case - summarize by source-target pairs
        if self.is_full_case:
            summary_data = []
            for source_sg in self.subgroup_labels:
                for target_sg in self.subgroup_labels:
                    # Filter contacts from source to target
                    mask = contacts["subgroup_cnt"] == target_sg
                    contacts_filtered = contacts[mask]

                    # Get participants from source subgroup
                    source_participants = self.df_part[
                        self.df_part["subgroup"] == source_sg
                    ]["id"]
                    contacts_source_to_target = contacts_filtered[
                        contacts_filtered["id"].isin(source_participants)
                    ]

                    if len(contacts_source_to_target) > 0:
                        total_contacts = contacts_source_to_target["y"].sum()
                        n_participants = len(source_participants)
                        mean_contacts = (
                            total_contacts / n_participants if n_participants > 0 else 0
                        )
                    else:
                        total_contacts = 0
                        n_participants = len(source_participants)
                        mean_contacts = 0

                    summary_data.append(
                        {
                            "subgroup": source_sg,
                            "subgroup_cnt": target_sg,
                            "total_contacts": int(total_contacts),
                            "mean_contacts": mean_contacts,
                        }
                    )

            return pd.DataFrame(summary_data)

        # Partial case - summarize by subgroup
        # Calculate total contacts per participant
        participant_totals = contacts.groupby("id")["y"].sum()

        # Merge subgroup information
        participant_subgroups = self.df_part[["id", "subgroup"]].set_index("id")
        participant_totals = participant_totals.to_frame("total_contacts")
        participant_totals = participant_totals.join(participant_subgroups)

        # Calculate statistics by subgroup
        summary = (
            participant_totals.groupby("subgroup")["total_contacts"]
            .agg(
                [
                    ("total_participants", "count"),
                    ("total_contacts", "sum"),
                    ("mean_contacts", "mean"),
                    ("median_contacts", "median"),
                    ("std_contacts", "std"),
                    ("min_contacts", "min"),
                    ("max_contacts", "max"),
                ]
            )
            .reset_index()
        )

        return summary

    def get_contact_matrix_empirical(
        self, contacts: pd.DataFrame = None, normalize: bool = False
    ) -> Union[NDArray, Dict[Union[str, int], NDArray], Dict[Tuple, NDArray]]:
        """
        Calculate empirical contact matrix from generated contact data.

        This computes the actual contact patterns observed in the generated data,
        which can be compared to the input contact intensity matrices.

        Parameters
        ----------
        contacts : pd.DataFrame, optional
                Contact data from generate(). If None, will call generate() with default seed.
        normalize : bool, default=False
                If True, normalize by the number of participants in each age group

        Returns
        -------
        NDArray or dict of NDArray
                Empirical contact matrix/matrices where element [i,j] represents
                total (or average if normalized) contacts from age group i to age group j

                - For single case: returns NDArray
                - For partial case: returns dict mapping subgroup labels to NDArray
                - For full case: returns dict with (source, target) tuple keys

        Examples
        --------
        >>> # Single case
        >>> contacts = contact_gen.generate(seed=42)
        >>> empirical = contact_gen.get_contact_matrix_empirical(contacts, normalize=True)
        >>> print(f"Empirical contacts from age 0 to age 1: {empirical[0, 1]:.2f}")

        >>> # Partial case
        >>> contacts = contact_gen.generate(seed=42)
        >>> empirical = contact_gen.get_contact_matrix_empirical(contacts, normalize=True)
        >>> urban_matrix = empirical['urban']
        >>> print(f"Urban contacts (age 0→1): {urban_matrix[0, 1]:.2f}")

        >>> # Full case
        >>> contacts = contact_gen.generate(seed=42)
        >>> empirical = contact_gen.get_contact_matrix_empirical(contacts, normalize=True)
        >>> urban_to_rural = empirical[('urban', 'rural')]
        >>> print(f"Urban to rural contacts (age 0→1): {urban_to_rural[0, 1]:.2f}")
        """
        if contacts is None:
            contacts = self.generate()

        # Determine number of age groups from contact matrix
        if isinstance(self.cint_matrices, np.ndarray):
            n_age_groups = self.cint_matrices.shape[0]
        else:
            first_matrix = next(iter(self.cint_matrices.values()))
            n_age_groups = first_matrix.shape[0]

        # Single population case
        if isinstance(self.cint_matrices, np.ndarray):
            # Create empirical matrix
            empirical = np.zeros((n_age_groups, n_age_groups))

            # Merge with participant ages
            contacts_with_age = contacts.merge(
                self.df_part[["id", "age_group"]], on="id"
            )

            # Aggregate contacts
            for _, row in contacts_with_age.iterrows():
                age_from = int(row["age_group"])
                age_to = int(row["age_cnt"])
                empirical[age_from, age_to] += row["y"]

            # Normalize if requested
            if normalize:
                age_counts = self.df_part["age_group"].value_counts().sort_index()
                for i in range(n_age_groups):
                    if i in age_counts.index and age_counts[i] > 0:
                        empirical[i, :] /= age_counts[i]

            return empirical

        # Full case - create matrices for each (source, target) pair
        if self.is_full_case:
            empirical_matrices = {}

            for source_label in self.subgroup_labels:
                for target_label in self.subgroup_labels:
                    # Create empirical matrix for this pair
                    empirical = np.zeros((n_age_groups, n_age_groups))

                    # Filter participants from source subgroup
                    df_part_source = self.df_part[
                        self.df_part["subgroup"] == source_label
                    ]

                    # Filter contacts to target subgroup
                    contacts_pair = contacts[contacts["subgroup_cnt"] == target_label]

                    # Merge with participant ages
                    contacts_with_age = contacts_pair.merge(
                        df_part_source[["id", "age_group"]], on="id"
                    )

                    # Aggregate contacts
                    for _, row in contacts_with_age.iterrows():
                        age_from = int(row["age_group"])
                        age_to = int(row["age_cnt"])
                        empirical[age_from, age_to] += row["y"]

                    # Normalize if requested
                    if normalize:
                        age_counts = (
                            df_part_source["age_group"].value_counts().sort_index()
                        )
                        for i in range(n_age_groups):
                            if i in age_counts.index and age_counts[i] > 0:
                                empirical[i, :] /= age_counts[i]

                    empirical_matrices[(source_label, target_label)] = empirical

            return empirical_matrices

        # Partial case - create matrices for each subgroup
        empirical_matrices = {}

        for label in self.subgroup_labels:
            # Filter participants for this subgroup
            df_part_sub = self.df_part[self.df_part["subgroup"] == label]

            # Filter contacts for this subgroup
            # Note: In partial case, we need to get contacts for participants in this subgroup
            contacts_sub = contacts[contacts["id"].isin(df_part_sub["id"])]

            # Create empirical matrix
            empirical = np.zeros((n_age_groups, n_age_groups))

            # Merge with participant ages
            contacts_with_age = contacts_sub.merge(
                df_part_sub[["id", "age_group"]], on="id"
            )

            # Aggregate contacts
            for _, row in contacts_with_age.iterrows():
                age_from = int(row["age_group"])
                age_to = int(row["age_cnt"])
                empirical[age_from, age_to] += row["y"]

            # Normalize if requested
            if normalize:
                age_counts = df_part_sub["age_group"].value_counts().sort_index()
                for i in range(n_age_groups):
                    if i in age_counts.index and age_counts[i] > 0:
                        empirical[i, :] /= age_counts[i]

            empirical_matrices[label] = empirical

        return empirical_matrices
