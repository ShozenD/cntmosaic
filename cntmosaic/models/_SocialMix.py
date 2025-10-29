"""
Social Contact Matrix Estimation

This module implements the socialmixr algorithm for estimating age-structured
social contact matrices from survey data. Based on Funk et al. (2024).

Key Features:
- Contact intensity and rate matrix estimation
- Optional reciprocity (symmetry) adjustment
- Adaptive merging of zero-sample age groups
- Bootstrap uncertainty quantification
- Comprehensive input validation
"""

import warnings
from typing import Optional, Tuple, List
from dataclasses import dataclass

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from tqdm import tqdm

from ..utils import AgeBins, pixilate, depixilate


# ============================================================================
# Data Classes
# ============================================================================


@dataclass
class BootstrapResults:
    """
    Container for bootstrap uncertainty estimates.

    Attributes
    ----------
    intensity_samples : NDArray
                    Bootstrap samples of contact intensity matrices, shape (n_boot, B, B)
    rate_samples : NDArray
                    Bootstrap samples of contact rate matrices, shape (n_boot, B, B)
    n_successful : int
                    Number of successful bootstrap iterations
    n_requested : int
                    Number of requested bootstrap iterations
    """

    intensity_samples: NDArray[np.float64]
    rate_samples: NDArray[np.float64]
    n_successful: int
    n_requested: int

    def quantiles(self, q: List[float]) -> Tuple[NDArray, NDArray]:
        """
        Compute quantiles across bootstrap samples.

        Parameters
        ----------
        q : list of float
                        Quantile levels (e.g., [0.025, 0.975] for 95% CI)

        Returns
        -------
        intensity_quantiles : NDArray
                        Quantiles of intensity, shape (len(q), B, B)
        rate_quantiles : NDArray
                        Quantiles of rate, shape (len(q), B, B)
        """
        intensity_q = np.quantile(self.intensity_samples, q, axis=0)
        rate_q = np.quantile(self.rate_samples, q, axis=0)
        return intensity_q, rate_q

    def std(self) -> Tuple[NDArray, NDArray]:
        """
        Compute standard errors across bootstrap samples.

        Returns
        -------
        intensity_std : NDArray, shape (B, B)
        rate_std : NDArray, shape (B, B)
        """
        return self.intensity_samples.std(axis=0), self.rate_samples.std(axis=0)

    def mean(self) -> Tuple[NDArray, NDArray]:
        """
        Compute mean across bootstrap samples.

        Returns
        -------
        intensity_mean : NDArray, shape (B, B)
        rate_mean : NDArray, shape (B, B)
        """
        return self.intensity_samples.mean(axis=0), self.rate_samples.mean(axis=0)


# ============================================================================
# Validation
# ============================================================================


class InputValidator:
    """Validates input dataframes for SocialMix."""

    @staticmethod
    def validate_participants(df: pd.DataFrame) -> None:
        """
        Validate participant dataframe structure and content.

        Parameters
        ----------
        df : pd.DataFrame
            Participant dataframe

        Raises
        ------
        ValueError
            If validation fails
        """
        # Check required columns
        if "id" not in df.columns:
            raise ValueError("Missing required column 'id' in participants dataframe")

        # Must have age information
        has_age = "age_part" in df.columns
        has_age_grp = "age_grp_part" in df.columns

        if not (has_age or has_age_grp):
            raise ValueError(
                "Participants dataframe must have either 'age_part' or 'age_grp_part' column"
            )

        # Check for duplicate IDs
        if df["id"].duplicated().any():
            n_duplicates = df["id"].duplicated().sum()
            raise ValueError(
                f"Found {n_duplicates} duplicate participant IDs. Each participant must have unique ID."
            )

        # Validate age values if present
        if has_age:
            if df["age_part"].isna().any():
                raise ValueError("Participant ages contain missing values")

            if (df["age_part"] < 0).any():
                raise ValueError("Participant ages cannot be negative")

            if (df["age_part"] > 120).any():
                warnings.warn(
                    "Some participant ages exceed 120 years. Please verify data.",
                    UserWarning,
                )

        # Validate age group values if present
        if has_age_grp:
            if df["age_grp_part"].isna().any():
                raise ValueError("Participant age groups contain missing values")

            if not isinstance(df["age_grp_part"].dtype, pd.CategoricalDtype):
                raise ValueError(
                    "Participant age groups ('age_grp_part') must be categorical dtype"
                )

            # The values should be pd.Interval objects (closed='right')
            if not all(
                isinstance(interval, pd.Interval)
                for interval in df["age_grp_part"].cat.categories
            ):
                raise ValueError(
                    "Participant age groups ('age_grp_part') must be pd.Interval categories"
                )

    @staticmethod
    def validate_contacts(df: pd.DataFrame, participant_ids: set) -> None:
        """
        Validate contacts dataframe.

        Parameters
        ----------
        df : pd.DataFrame
            Contacts dataframe
        participant_ids : set
            Set of valid participant IDs

        Raises
        ------
        ValueError
            If validation fails
        """
        # Check required columns
        required_cols = {"id", "y"}
        missing = required_cols - set(df.columns)
        if missing:
            raise ValueError(
                f"Missing required columns in contacts dataframe: {missing}"
            )

        has_age = "age_cnt" in df.columns
        has_age_grp = "age_grp_cnt" in df.columns

        if not (has_age or has_age_grp):
            raise ValueError(
                "Contacts dataframe must have either 'age_cnt' or 'age_grp_cnt' column"
            )

        if has_age:
            if df["age_cnt"].isna().any():
                raise ValueError("Contact ages contain missing values")

            if (df["age_cnt"] < 0).any():
                raise ValueError("Contact ages cannot be negative")

            if (df["age_cnt"] > 120).any():
                warnings.warn(
                    "Some contact ages exceed 120 years. Please verify data.",
                    UserWarning,
                )

        if has_age_grp:
            if df["age_grp_cnt"].isna().any():
                raise ValueError("Contact age groups contain missing values")

            if not isinstance(df["age_grp_cnt"].dtype, pd.CategoricalDtype):
                raise ValueError(
                    "Contact age groups ('age_grp_cnt') must be categorical dtype"
                )

            if not all(
                isinstance(interval, pd.Interval)
                for interval in df["age_grp_cnt"].cat.categories
            ):
                raise ValueError(
                    "Contact age groups ('age_grp_cnt') must be pd.Interval categories"
                )

        # Check that contact IDs exist in participants
        invalid_ids = set(df["id"]) - participant_ids
        if invalid_ids:
            n_invalid = len(invalid_ids)
            raise ValueError(
                f"Found {n_invalid} contact records with IDs not in participants dataframe. "
                f"All contact IDs must correspond to valid participants."
            )

    @staticmethod
    def validate_age_distribution(df: pd.DataFrame) -> None:
        """
        Validate population age distribution dataframe.

        Parameters
        ----------
        df : pd.DataFrame
            Age distribution dataframe

        Raises
        ------
        ValueError
            If validation fails
        """
        # Check required columns
        required_cols = {"age", "P"}
        missing = required_cols - set(df.columns)
        if missing:
            raise ValueError(
                f"Missing required columns in age distribution dataframe: {missing}"
            )

        # Check P column
        if df["P"].isna().any():
            raise ValueError("Population sizes ('P') contain missing values")

        # Validate population sizes
        if (df["P"] <= 0).any():
            raise ValueError("Population sizes ('P') must be positive")

        if df["P"].sum() == 0:
            raise ValueError("Total population size is zero")

        # Validate ages
        if df["age"].isna().any():
            raise ValueError("Ages in distribution contain missing values")

        if (df["age"] < 0).any():
            raise ValueError("Ages in distribution cannot be negative")

        if (df["age"] > 120).any():
            warnings.warn(
                "Some ages in distribution exceed 120 years. Please verify data.",
                UserWarning,
            )


# ============================================================================
# Age Binning
# ============================================================================


class AgeBinProcessor:
    """Handles age binning and zero-group merging."""

    def __init__(self, age_bins: AgeBins):
        """
        Initialize age bin processor.

        Parameters
        ----------
        age_bins : AgeBins
            Age stratification bins
        """
        self.age_bins = age_bins

    def assign_age_groups(
        self, df: pd.DataFrame, age_col: str, group_col: str
    ) -> pd.DataFrame:
        """
        Assign age groups to dataframe.

        Parameters
        ----------
        df : pd.DataFrame
            Dataframe with age column
        age_col : str
             Name of age column
        group_col : str
            Name for new age group column

        Returns
        -------
        pd.DataFrame
            Dataframe with age group column added
        """
        df = df.copy()
        bins = self.age_bins.left + [self.age_bins.right[-1]]

        df[group_col] = pd.cut(df[age_col], bins=bins, right=False, include_lowest=True)

        return df

    @staticmethod
    def merge_zero_groups(
        intervals: List[pd.Interval], counts: NDArray[np.int64]
    ) -> List[pd.Interval]:
        """
        Merge zero-count intervals with adjacent non-zero intervals.

        Strategy:
        1. Zero-count groups are merged with the previous non-zero group if available
        2. Leading zero-count groups are merged with the first non-zero group
        3. If all groups are zero, raise an error

        Parameters
        ----------
        intervals : list of pd.Interval
            Sorted list of age intervals
        counts : NDArray
            Sample counts for each interval

        Returns
        -------
        list of pd.Interval
            Merged intervals

        Raises
        ------
        ValueError
            If all counts are zero

        Examples
        --------
        >>> intervals = [pd.Interval(0, 5), pd.Interval(5, 10),
        ...              pd.Interval(10, 15), pd.Interval(15, 20)]
        >>> counts = np.array([10, 0, 15, 0])
        >>> merged = AgeBinProcessor.merge_zero_groups(intervals, counts)
        >>> len(merged)
        2
        """
        if not np.any(counts > 0):
            raise ValueError(
                "All age groups have zero participants. Cannot create contact matrices."
            )

        merged = []
        n = len(intervals)
        i = 0

        while i < n:
            if counts[i] > 0:
                # Non-zero count: start new merged interval
                merged.append(intervals[i])
                i += 1
            else:
                # Zero count: merge with previous or look ahead
                if merged:
                    # Extend the last merged interval
                    prev = merged[-1]
                    merged[-1] = pd.Interval(
                        left=prev.left, right=intervals[i].right, closed="left"
                    )
                    i += 1
                else:
                    # No previous interval: find first non-zero and merge all leading zeros
                    j = i
                    while j < n and counts[j] == 0:
                        j += 1

                    if j >= n:
                        # Should never happen due to earlier check, but be defensive
                        raise ValueError("No non-zero age groups found")

                    # Create merged interval from first zero to first non-zero
                    merged.append(
                        pd.Interval(
                            left=intervals[i].left,
                            right=intervals[j].right,
                            closed="left",
                        )
                    )
                    i = j + 1

        return merged

    @staticmethod
    def reassign_age_groups(
        df: pd.DataFrame,
        age_grp_col: str,
        merged_intervals: List[pd.Interval],
        new_group_col: str = None,
    ) -> pd.DataFrame:
        """
        Reassign age groups based on merged intervals from merge_zero_groups.

        This method takes a dataframe with existing age group assignments and
        remaps them to the new merged age groups. Original intervals that were
        merged together will be assigned to the same new group.

        Parameters
        ----------
        df : pd.DataFrame
            Dataframe with existing age group column
        age_grp_col : str
            Name of existing age group column (contains pd.Interval objects)
        merged_intervals : list of pd.Interval
            Output from merge_zero_groups - the new merged intervals
        new_group_col : str, optional
            Name for new age group column. If None, overwrites age_grp_col

        Returns
        -------
        pd.DataFrame
            Dataframe with reassigned age groups

        Examples
        --------
        >>> # Original dataframe with fine-grained age groups
        >>> df = pd.DataFrame({
        ...     'id': [1, 2, 3, 4],
        ...     'age_grp': [pd.Interval(0, 5), pd.Interval(5, 10),
        ...                 pd.Interval(10, 15), pd.Interval(15, 20)]
        ... })
        >>>
        >>> # Merge zero groups
        >>> original_intervals = df['age_grp'].unique()
        >>> counts = np.array([2, 0, 1, 0])
        >>> merged = AgeBinProcessor.merge_zero_groups(original_intervals, counts)
        >>> # merged = [Interval(0, 10), Interval(10, 20)]
        >>>
        >>> # Reassign based on merged intervals
        >>> df_new = AgeBinProcessor.reassign_age_groups(
        ...     df, 'age_grp', merged, 'age_grp_merged'
        ... )
        >>> df_new['age_grp_merged'].unique()
        array([Interval(0, 10, closed='left'), Interval(10, 20, closed='left')])

        Notes
        -----
        The reassignment works by checking which merged interval contains each
        original interval. An original interval is contained in a merged interval
        if its left boundary is >= merged.left and its right boundary <= merged.right.
        """
        df = df.copy()

        # Determine output column name
        output_col = new_group_col if new_group_col is not None else age_grp_col

        # Get original intervals from the dataframe
        original_intervals = df[age_grp_col]

        # Create mapping from original intervals to merged intervals
        def find_merged_interval(original_interval):
            """
            Find which merged interval contains the original interval.

            An original interval is contained in a merged interval if:
            - original.left >= merged.left
            - original.right <= merged.right
            """
            if pd.isna(original_interval):
                return pd.NA

            for merged_interval in merged_intervals:
                # Check if original interval is contained within merged interval
                if (
                    original_interval.left >= merged_interval.left
                    and original_interval.right <= merged_interval.right
                ):
                    return merged_interval

            # If no match found, this shouldn't happen with valid data
            # but handle gracefully
            raise ValueError(
                f"Original interval {original_interval} does not fit within "
                f"any merged interval. This may indicate inconsistent age bins."
            )

        # Apply the mapping to create new column
        df[output_col] = original_intervals.apply(find_merged_interval)

        # Convert to categorical with ordered categories
        df[output_col] = pd.Categorical(
            df[output_col], categories=merged_intervals, ordered=True
        )

        return df

    @staticmethod
    def create_interval_mapping(
        original_intervals: List[pd.Interval], merged_intervals: List[pd.Interval]
    ) -> dict:
        """
        Create explicit mapping from original intervals to merged intervals.

        This is a helper method that can be useful for understanding how
        intervals were merged or for applying the same mapping to multiple
        dataframes.

        Parameters
        ----------
        original_intervals : list of pd.Interval
            Original fine-grained intervals
        merged_intervals : list of pd.Interval
            Merged intervals from merge_zero_groups

        Returns
        -------
        dict
            Mapping from original intervals to merged intervals

        Examples
        --------
        >>> original = [pd.Interval(0, 5), pd.Interval(5, 10),
        ...             pd.Interval(10, 15)]
        >>> merged = [pd.Interval(0, 10), pd.Interval(10, 15)]
        >>> mapping = AgeBinProcessor.create_interval_mapping(original, merged)
        >>> mapping[pd.Interval(0, 5)]
        Interval(0, 10, closed='left')
        >>> mapping[pd.Interval(5, 10)]
        Interval(0, 10, closed='left')
        """
        mapping = {}

        for orig_interval in original_intervals:
            for merged_interval in merged_intervals:
                if (
                    orig_interval.left >= merged_interval.left
                    and orig_interval.right <= merged_interval.right
                ):
                    mapping[orig_interval] = merged_interval
                    break
            else:
                raise ValueError(
                    f"Original interval {orig_interval} does not fit within "
                    f"any merged interval."
                )

        return mapping


# ============================================================================
# Matrix Computation
# ============================================================================


class ContactMatrixEstimator:
    """Computes contact matrices with numerical stability."""

    @staticmethod
    def compute_contact_counts(
        df_part: pd.DataFrame, df_cnt: pd.DataFrame, age_groups: List[str]
    ) -> NDArray[np.float64]:
        """
        Compute contact count matrix Y.

        Y[c,d] = sum of contacts by age group c with age group d

        Parameters
        ----------
        df_part : pd.DataFrame
                        Participants with 'id' and 'age_grp_part'
        df_cnt : pd.DataFrame
                        Contacts with 'id', 'age_grp_cnt', 'y'
        age_groups : list
                        Ordered list of age group labels

        Returns
        -------
        Y : NDArray, shape (B, B)
                        Contact count matrix
        """
        # Merge participants and contacts
        merged = df_cnt.merge(
            df_part[["id", "age_grp_part"]],
            on="id",
            how="inner",  # Only keep contacts with valid participants
            validate="m:1",  # Many contacts to one participant
        )

        if len(merged) == 0:
            raise ValueError(
                "No valid contacts after merging with participants. "
                "Check that contact IDs match participant IDs."
            )

        # Aggregate contact counts by age group pairs
        Y_agg = (
            merged.groupby(["age_grp_part", "age_grp_cnt"], observed=False)
            .agg({"y": "sum"})
            .unstack(fill_value=0)
        )

        # Extract values and reindex to ensure all age groups present
        Y_df = Y_agg["y"].reindex(index=age_groups, columns=age_groups, fill_value=0)

        return Y_df.values.astype(np.float64)

    @staticmethod
    def compute_sample_sizes(
        df_part: pd.DataFrame, age_groups: List[str]
    ) -> NDArray[np.int64]:
        """
        Compute sample sizes per age group.

        Parameters
        ----------
        df_part : pd.DataFrame
            Participants with 'age_grp_part'
        age_groups : list
            Ordered list of age group labels

        Returns
        -------
        N : NDArray, shape (B,)
            Sample size for each age group

        Raises
        ------
        ValueError
            If any age group has zero participants
        """
        counts = (
            df_part.groupby("age_grp_part", observed=False)
            .size()
            .reindex(age_groups, fill_value=0)
        )

        if (counts == 0).any():
            zero_groups = counts[counts == 0].index.tolist()
            raise ValueError(
                f"Age groups have zero participants after preprocessing: {zero_groups}. "
                f"This should not happen if adaptive_merge worked correctly."
            )

        return counts.values.astype(np.int64)

    @staticmethod
    def compute_population_sizes(
        df_age_dist: pd.DataFrame, age_groups: List[str]
    ) -> NDArray[np.int64]:
        """
        Compute population sizes per age group.

        Parameters
        ----------
        df_age_dist : pd.DataFrame
            Population distribution with 'age_grp' and 'P'
        age_groups : list
            Ordered list of age group labels

        Returns
        -------
        P : NDArray, shape (B,)
            Population size for each age group

        Raises
        ------
        ValueError
            If any age group has zero population
        """
        pop_sizes = (
            df_age_dist.groupby("age_grp", observed=False)
            .agg({"P": "sum"})
            .reindex(age_groups, fill_value=0)
        )

        if (pop_sizes["P"] == 0).any():
            zero_groups = pop_sizes[pop_sizes["P"] == 0].index.tolist()
            raise ValueError(
                f"Age groups have zero population: {zero_groups}. "
                f"Check that age_dist covers all age bins."
            )

        return pop_sizes["P"].values.astype(np.int64)

    @staticmethod
    def compute_intensity(
        Y: NDArray[np.float64],
        N: NDArray[np.int64],
        symmetric: bool = False,
        P: Optional[NDArray[np.int64]] = None,
    ) -> NDArray[np.float64]:
        """
        Compute contact intensity matrix.

        Basic: M[c,d] = Y[c,d] / N[c]
        Symmetric: M†[c,d] = 0.5 * (M[c,d] + P[d]/P[c] * M[d,c])

        This ensures reciprocity: M†[c,d] * P[c] = M†[d,c] * P[d]

        Parameters
        ----------
        Y : NDArray, shape (B, B)
            Contact count matrix
        N : NDArray, shape (B,)
            Sample sizes
        symmetric : bool
            Apply reciprocity adjustment
        P : NDArray, shape (B,), optional
            Population sizes (required if symmetric=True)

        Returns
        -------
        M : NDArray, shape (B, B)
            Contact intensity matrix

        Raises
        ------
        ValueError
            If inputs are invalid
        """
        # Convert to float for numerical stability
        N_float = N.astype(np.float64)

        if np.any(N_float <= 0):
            raise ValueError("Sample sizes must be positive")

        # Basic intensity: M = Y / N (broadcast over columns)
        M = Y / N_float[:, np.newaxis]

        if symmetric:
            P_float = P.astype(np.float64)

            # Symmetric adjustment: M† = 0.5 * (M + P^-1 * M^T * P)
            M = 0.5 * (M + (P_float[np.newaxis, :] / P_float[:, np.newaxis]) * M.T)

        return M

    @staticmethod
    def compute_rate(
        M: NDArray[np.float64], P: NDArray[np.int64]
    ) -> NDArray[np.float64]:
        """
        Compute contact rate matrix.

        ω[c,d] = M[c,d] / P[d]

        Parameters
        ----------
        M : NDArray, shape (B, B)
                        Contact intensity matrix
        P : NDArray, shape (B,)
                        Population sizes

        Returns
        -------
        omega : NDArray, shape (B, B)
                        Contact rate matrix
        """
        P_float = P.astype(np.float64)

        return M / P_float[np.newaxis, :]


# ============================================================================
# Bootstrap
# ============================================================================


class BootstrapEstimator:
    """Performs bootstrap uncertainty quantification."""

    def __init__(
        self,
        df_part: pd.DataFrame,
        df_cnt: pd.DataFrame,
        df_age_dist: pd.DataFrame,
        age_bins: AgeBins,
        symmetric: bool = False,
        n_boot: int = 1000,
        random_state: Optional[int] = None,
    ):
        """
        Initialize bootstrap estimator.

        Parameters
        ----------
        df_part : pd.DataFrame
            Participant data
        df_cnt : pd.DataFrame
            Contact data
        df_age_dist : pd.DataFrame
            Population age distribution
        age_bins : AgeBins
            Age stratification
        symmetric : bool
            Apply reciprocity adjustment
        n_boot : int
            Number of bootstrap samples
        random_state : int, optional
            Random seed for reproducibility
        """
        self.df_part = df_part
        self.df_cnt = df_cnt
        self.df_age_dist = df_age_dist
        self.age_bins = age_bins
        self.symmetric = symmetric
        self.n_boot = n_boot
        self.rng = np.random.default_rng(random_state)

    def run(
        self, progress: bool = True, min_success_rate: float = 0.5
    ) -> BootstrapResults:
        """
        Run bootstrap resampling.

        Parameters
        ----------
        progress : bool
                        Show progress bar
        min_success_rate : float
                        Minimum fraction of successful iterations required

        Returns
        -------
        BootstrapResults
                        Container with bootstrap samples and metadata

        Raises
        ------
        ValueError
                        If too many bootstrap iterations fail
        """
        intensity_samples = []
        rate_samples = []

        iterator = tqdm(range(self.n_boot), desc="Bootstrapping", disable=not progress)

        for i in iterator:
            try:
                M, omega = self._bootstrap_iteration()
                intensity_samples.append(M)
                rate_samples.append(omega)
            except (ValueError, ZeroDivisionError) as e:
                if progress:
                    iterator.write(f"Bootstrap iteration {i} failed: {str(e)[:100]}")
                continue
            except Exception as e:
                if progress:
                    iterator.write(
                        f"Bootstrap iteration {i} unexpected error: {str(e)[:100]}"
                    )
                continue

        n_successful = len(intensity_samples)
        success_rate = n_successful / self.n_boot

        if success_rate < min_success_rate:
            raise ValueError(
                f"Bootstrap failed: only {n_successful}/{self.n_boot} "
                f"iterations succeeded ({success_rate:.1%}). "
                f"This may indicate insufficient sample sizes in some age groups."
            )

        if progress and n_successful < self.n_boot:
            print(
                f"\nCompleted {n_successful}/{self.n_boot} bootstrap iterations "
                f"({success_rate:.1%} success rate)"
            )

        return BootstrapResults(
            intensity_samples=np.array(intensity_samples),
            rate_samples=np.array(rate_samples),
            n_successful=n_successful,
            n_requested=self.n_boot,
        )

    def _bootstrap_iteration(self) -> Tuple[NDArray, NDArray]:
        """
        Perform single bootstrap iteration.

        Resamples participants with replacement and filters corresponding contacts.

        Returns
        -------
        M : NDArray
                        Contact intensity matrix for this bootstrap sample
        omega : NDArray
                        Contact rate matrix for this bootstrap sample
        """
        # Resample participants with replacement
        n_part = len(self.df_part)
        boot_indices = self.rng.choice(n_part, size=n_part, replace=True)
        df_part_boot = self.df_part.iloc[boot_indices].reset_index(drop=True)

        # Create mapping: old_id -> list of new_ids
        id_mapping = {}
        for idx, old_id in enumerate(df_part_boot["id"]):
            new_id = f"boot_{idx}"  # or simply idx
            if old_id not in id_mapping:
                id_mapping[old_id] = []
            id_mapping[old_id].append(new_id)

        # Assign new unique IDs to participants
        df_part_boot["id"] = [f"boot_{i}" for i in range(len(df_part_boot))]

        # Duplicate contacts for each resampled participant
        df_cnt_boot_list = []
        for old_id, new_ids in id_mapping.items():
            contacts_for_id = self.df_cnt[self.df_cnt["id"] == old_id]
            for new_id in new_ids:
                contacts_copy = contacts_for_id.copy()
                contacts_copy["id"] = new_id
                df_cnt_boot_list.append(contacts_copy)

        df_cnt_boot = pd.concat(df_cnt_boot_list, ignore_index=True)

        if len(df_cnt_boot) == 0:
            raise ValueError("No contacts remain after resampling participants")

        # Create temporary SocialMix instance for this bootstrap sample
        sm = SocialMix(
            df_part_boot,
            df_cnt_boot,
            self.df_age_dist,
            self.age_bins,
            symmetric=self.symmetric,
            adaptive_merge=True,  # Handle edge cases in bootstrap
            verbose=False,  # Suppress warnings in bootstrap loop
        )

        return sm.compute_cint(), sm.compute_rate()


# ============================================================================
# Main Class
# ============================================================================


class SocialMix:
    """
    Estimate age-structured social contact matrices from survey data.

    Implements the socialmixr algorithm (Funk et al. 2024) for computing
    contact intensity and contact rate matrices from participant and contact
    data, with optional reciprocity adjustment and bootstrap uncertainty.

    Parameters
    ----------
    df_part : pd.DataFrame
        Participant data with columns:
        - 'id': unique participant identifier
        - 'age_part': participant age (numeric)
    df_cnt : pd.DataFrame
        Contact data with columns:
        - 'id': participant identifier (links to df_part)
        - 'age_cnt': contact age (numeric)
        - 'y': number of contacts (numeric, >= 0)
    df_age_dist : pd.DataFrame
        Population age distribution with columns:
        - 'age': age value (numeric)
        - 'P': population size at that age (numeric, > 0)
    age_bins : AgeBins
        Age stratification bins defining age groups
    symmetric : bool, default False
        Apply reciprocity adjustment to ensure M[c,d]*P[c] = M[d,c]*P[d]
    adaptive_merge : bool, default False
        Automatically merge age groups with zero participants
    verbose : bool, default True
        Print warnings and progress information

    Attributes
    ----------
    Y : NDArray
        Contact count matrix, shape (B, B)
    N : NDArray
        Sample sizes per age group, shape (B,)
    P : NDArray
        Population sizes per age group, shape (B,)
    effective_age_bins : AgeBins
        Age bins after any adaptive merging

    Methods
    -------
    compute_cint(recover_bins=False)
        Compute contact intensity matrix M
    compute_rate(recover_bins=False)
        Compute contact rate matrix ω
    run_bootstrap(n_boot=1000, random_state=None, progress=True)
        Estimate uncertainty via bootstrap

    Examples
    --------
    >>> # Create SocialMix instance
    >>> sm = SocialMix(df_part, df_cnt, df_age_dist, age_bins)
    >>>
    >>> # Get contact intensity matrix
    >>> M = sm.compute_cint()
    >>>
    >>> # Get contact rate matrix
    >>> omega = sm.compute_rate()
    >>>
    >>> # Bootstrap uncertainty
    >>> boot_results = sm.run_bootstrap(n_boot=1000, random_state=42)
    >>> M_std, omega_std = boot_results.std()
    >>> M_ci, omega_ci = boot_results.quantiles([0.025, 0.975])

    Notes
    -----
    Contact intensity M[c,d] represents the average number of contacts that
    individuals in age group c have with individuals in age group d.

    Contact rate ω[c,d] = M[c,d] / P[d] represents the per-capita rate at
    which individuals in age group c contact individuals in age group d.

    The reciprocity adjustment (symmetric=True) ensures that the total number
    of contacts from c to d equals the total from d to c: M[c,d]*P[c] = M[d,c]*P[d].
    """

    def __init__(
        self,
        df_part: pd.DataFrame,
        df_cnt: pd.DataFrame,
        df_age_dist: pd.DataFrame,
        age_bins: AgeBins,
        symmetric: bool = False,
        adaptive_merge: bool = False,
        verbose: bool = True,
    ):
        # Store parameters
        self.df_part = df_part.copy()
        self.df_cnt = df_cnt.copy()
        self.df_age_dist = df_age_dist.copy()
        self.age_bins = age_bins
        self.symmetric = symmetric
        self.adaptive_merge = adaptive_merge
        self.verbose = verbose

        # Initialize helper classes
        self.validator = InputValidator()
        self.age_processor = AgeBinProcessor(age_bins)
        self.estimator = ContactMatrixEstimator()

        # Computed attributes (initialized in pipeline)
        self.effective_age_bins: Optional[AgeBins] = None
        self._cint: Optional[NDArray] = None
        self._rate: Optional[NDArray] = None
        self._boot: Optional[BootstrapResults] = None
        self.Y: Optional[NDArray] = None
        self.N: Optional[NDArray] = None
        self.P: Optional[NDArray] = None

        # Run processing pipeline
        self._validate()
        self._preprocess()
        self._fit()

    def _validate(self) -> None:
        """Validate all input dataframes."""
        self.validator.validate_participants(self.df_part)
        self.validator.validate_contacts(self.df_cnt, set(self.df_part["id"]))
        self.validator.validate_age_distribution(self.df_age_dist)

    def _preprocess(self) -> None:
        """
        Assign age groups and handle zero-sample groups.

        This method:
        1. Assigns initial age groups to participants
        2. Checks for zero-sample groups
        3. Merges zero-sample groups if adaptive_merge=True
        4. Assigns age groups to contacts and population
        """
        # Reset indices to avoid potential issues
        self.df_part = self.df_part.reset_index(drop=True)
        self.df_cnt = self.df_cnt.reset_index(drop=True)
        self.df_age_dist = self.df_age_dist.reset_index(drop=True)

        # Assign initial age groups to participants if needed
        has_age_part = "age_part" in self.df_part.columns
        has_age_grp_part = "age_grp_part" in self.df_part.columns
        if not has_age_grp_part and has_age_part:
            self.df_part = self.age_processor.assign_age_groups(
                self.df_part, "age_part", "age_grp_part"
            )

        # Assign age groups to contacts and population
        has_age_cnt = "age_cnt" in self.df_cnt.columns
        has_age_grp_cnt = "age_grp_cnt" in self.df_cnt.columns
        if not has_age_grp_cnt and has_age_cnt:
            self.df_cnt = self.age_processor.assign_age_groups(
                self.df_cnt, "age_cnt", "age_grp_cnt"
            )

        has_age = "age" in self.df_age_dist.columns
        has_age_grp = "age_grp" in self.df_age_dist.columns
        if not has_age_grp and has_age:
            self.df_age_dist = self.age_processor.assign_age_groups(
                self.df_age_dist, "age", "age_grp"
            )

        # Check for zero-sample groups
        sample_sizes = self.df_part.groupby("age_grp_part", observed=False).size()

        if (sample_sizes == 0).any():
            if not self.adaptive_merge:
                zero_groups = sample_sizes[sample_sizes == 0].index.tolist()
                raise ValueError(
                    f"Some age groups have zero participants: {zero_groups}. "
                    f"Set adaptive_merge=True to merge zero-count groups automatically."
                )

            # Merge zero-count groups
            if self.verbose:
                warnings.warn(
                    "Some age groups have zero participants. Merging with adjacent groups.",
                    UserWarning,
                )

            merged_intervals = self.age_processor.merge_zero_groups(
                sample_sizes.index.tolist(), sample_sizes.values
            )

            # Create new age bins from merged intervals
            self.effective_age_bins = AgeBins(
                min=self.age_bins.min,
                max=self.age_bins.max,
                cuts=[interval.left for interval in merged_intervals[1:]],
            )

            if self.verbose:
                print(f"Original bins: {len(self.age_bins.left)} groups")
                print(f"Effective bins: {len(self.effective_age_bins.left)} groups")

            # Reassign age groups with effective bins
            self.age_processor = AgeBinProcessor(self.effective_age_bins)
            self.df_part = self.age_processor.reassign_age_groups(
                self.df_part, "age_grp_part", merged_intervals, "age_grp_part"
            )
            self.df_cnt = self.age_processor.reassign_age_groups(
                self.df_cnt, "age_grp_cnt", merged_intervals, "age_grp_cnt"
            )
            self.df_age_dist = self.age_processor.reassign_age_groups(
                self.df_age_dist, "age_grp", merged_intervals, "age_grp"
            )
        else:
            self.effective_age_bins = self.age_bins

    def _fit(self) -> None:
        """Compute core matrices Y, N, P."""
        # Get ordered list of age groups
        age_groups = self.df_part["age_grp_part"].cat.categories.tolist()

        # Compute aggregated matrices
        self.Y = self.estimator.compute_contact_counts(
            self.df_part, self.df_cnt, age_groups
        )
        self.N = self.estimator.compute_sample_sizes(self.df_part, age_groups)
        self.P = self.estimator.compute_population_sizes(self.df_age_dist, age_groups)

    def compute_cint(self, recover_bins: bool = False) -> NDArray[np.float64]:
        """
        Compute contact intensity matrix.

        M[c,d] represents the average number of contacts that individuals
        in age group c have with individuals in age group d.

        Parameters
        ----------
        recover_bins : bool, default False
                        If True and adaptive merging occurred, transform result back
                        to original age bins using pixilate/depixilate

        Returns
        -------
        M : NDArray, shape (B, B) or (B_original, B_original)
                        Contact intensity matrix
        """
        if self._cint is None:
            self._cint = self.estimator.compute_intensity(
                self.Y, self.N, self.symmetric, self.P
            )

        if recover_bins and self.effective_age_bins != self.age_bins:
            # Transform from effective bins back to original bins
            return pixilate(
                depixilate(
                    self._cint, self.effective_age_bins, self.df_age_dist["P"].values
                ),
                self.age_bins,
                self.df_age_dist["P"].values,
            )
        else:
            return self._cint

    def compute_rate(self, recover_bins: bool = False) -> NDArray[np.float64]:
        """
        Compute contact rate matrix.

        ω[c,d] represents the per-capita rate at which individuals in
        age group c contact individuals in age group d.

        Parameters
        ----------
        recover_bins : bool, default False
                        If True and adaptive merging occurred, transform result back
                        to original age bins using pixilate/depixilate

        Returns
        -------
        omega : NDArray, shape (B, B) or (B_original, B_original)
                        Contact rate matrix
        """
        if self._rate is None:
            M = self.compute_cint()
            self._rate = self.estimator.compute_rate(M, self.P)

        if recover_bins and self.effective_age_bins != self.age_bins:
            # Transform from effective bins back to original bins
            return pixilate(
                depixilate(
                    self._rate, self.effective_age_bins, self.df_age_dist["P"].values
                ),
                self.age_bins,
                self.df_age_dist["P"].values,
            )
        else:
            return self._rate

    def run_bootstrap(
        self,
        n_boot: int = 1000,
        random_state: Optional[int] = None,
        progress: bool = True,
        min_success_rate: float = 0.5,
    ) -> BootstrapResults:
        """
        Estimate uncertainty via bootstrap resampling.

        Resamples participants with replacement and recomputes contact
        matrices for each bootstrap sample. This provides estimates of
        sampling variability.

        Parameters
        ----------
        n_boot : int, default 1000
            Number of bootstrap samples
        random_state : int, optional
            Random seed for reproducibility
        progress : bool, default True
            Show progress bar
        min_success_rate : float, default 0.5
            Minimum fraction of successful iterations required

        Returns
        -------
        BootstrapResults
            Container with bootstrap samples of intensity and rate matrices,
            plus methods for computing statistics (std, quantiles, mean)

        Examples
        --------
        >>> boot_results = sm.run_bootstrap(n_boot=1000, random_state=42)
        >>> M_mean, omega_mean = boot_results.mean()
        >>> M_std, omega_std = boot_results.std()
        >>> M_ci, omega_ci = boot_results.quantiles([0.025, 0.975])
        """
        if self._boot is None:
            bootstrapper = BootstrapEstimator(
                self.df_part,
                self.df_cnt,
                self.df_age_dist,
                self.age_bins,  # Use original bins for bootstrap
                self.symmetric,
                n_boot,
                random_state,
            )

            self._boot = bootstrapper.run(
                progress=progress, min_success_rate=min_success_rate
            )

        return self._boot
