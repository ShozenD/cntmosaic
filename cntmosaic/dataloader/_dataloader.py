"""
Data loading and preprocessing for social contact matrix estimation.

This module provides classes for loading, validating, and preprocessing social contact
survey data for use in Bayesian contact matrix estimation models. It handles participant
data, contact data, and population distributions, ensuring data consistency and creating
properly formatted datasets for downstream analysis.

Classes
-------
ParticipantData : Dataclass for participant data with validation
CoordToColumns : Dataclass for column name mapping
StratPropData : Dataclass for stratified population proportions
BaseLoader : Abstract base class for data loading
DataLoader : Main data loader for contact survey data
"""

import itertools
import warnings
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union

import jax.numpy as jnp
import numpy as np
import pandas as pd
import xarray as xr
from numpy.typing import NDArray

from .._types import StratMode
from ._utils import make_idarrs_for_intervals
from .containers._ContactData import ContactData
from .containers._ModelData import ModelBaseData, ModelData, ModelStratData
from .containers._ParticipantData import ParticipantData
from .containers._PopulationData import PopulationData
from .containers._StratPropData import StratPropData


@dataclass
class CoordToColumns:
    """
    Column name mapping for contact survey data.

    This dataclass specifies the mapping between input dataframe columns and the
    variables required by contact matrix estimation models. It ensures consistent
    naming conventions and validates that required column mappings are provided.

    Attributes
    ----------
    age_part : str
        Column name for participant age in the participant dataframe.
        Should contain integer ages or age groups.
    age_grp_part : Optional[str], default=None
        Column name for participant age group in the participant dataframe.
        Must be pd.IntervalIndex if used. Use this OR age_part, not both.
    age_cnt : Optional[str], default=None
        Column name for contact age in the contact dataframe.
        Use this OR age_grp_cnt, not both.
    age_grp_cnt : Optional[str], default=None
        Column name for contact age group in the contact dataframe.
        Must be pd.IntervalIndex if used. Use this OR age_cnt, not both.
    id_col : str, default='id'
        Column name for unique participant identifiers.
        Must be present in both participant and contact dataframes.
    y : str, default='y'
        Column name for number of contacts in the contact dataframe.
        If not present, will be auto-created with value 1 per contact.
    z : str, default='z'
        Column name for number of group contacts in the participant dataframe.
        If not present, will be auto-created with value 0.
    strat_vars_part : Optional[Union[List[str], str]], default=None
        Stratification variable column name(s) in participant dataframe.
        Can be a single string or list of strings. Examples: 'gender', ['gender', 'setting']
    strat_vars_cnt : Optional[Union[List[str], str]], default=None
        Stratification variable column name(s) in contact dataframe.
        Can be a single string or list of strings.
    repeat_part : Optional[str], default=None
        Column name for repeat interview count in participant dataframe.
        Used to model repeat participation effects. If provided, automatically
        added to strat_vars_part during post-initialization.
    age_pop : Optional[str], default=None
        Column name for age in population dataframe.
        Required for population weighting; must be provided with P.
    age_grp_pop : Optional[str], default=None
        Column name for age group in population dataframe.
        Must be pd.IntervalIndex if used. Use this OR age_pop, not both.
    P : Optional[str], default=None
        Column name for population size/proportion in population dataframe.
        Required for population weighting; must be provided with age_pop or age_grp_pop.
    strat_vars_pop : Optional[Union[List[str], str]], default=None
        Stratification variable column name(s) in population dataframe.
        Can be a single string or list of strings.

    Raises
    ------
    ValueError
        If neither age_cnt nor age_grp_cnt is provided.
        If age_pop and P are not both set or both None.

    Warnings
    --------
    UserWarning
        If the same stratification variable appears in both strat_vars_part and
        strat_vars_cnt. The variable in strat_vars_cnt will be automatically removed
        to avoid ambiguity.

    Examples
    --------
    >>> # Basic usage with individual contact ages
    >>> col_map = CoordToColumns(
    ...     age_part="participant_age",
    ...     age_cnt="contact_age",
    ...     id_col="participant_id",
    ...     age_pop="age",
    ...     P="population_size"
    ... )
    >>>
    >>> # With age groups and stratification
    >>> col_map = CoordToColumns(
    ...     age_part="age_participant",
    ...     age_grp_cnt="age_group_contact",
    ...     strat_vars_part=["gender", "location"],
    ...     strat_vars_cnt="setting",
    ...     age_pop="age",
    ...     P="N"
    ... )
    >>>
    >>> # With repeat interview effects
    >>> col_map = CoordToColumns(
    ...     age_part="age",
    ...     age_cnt="contact_age",
    ...     repeat_part="interview_round",
    ...     age_pop="age",
    ...     P="pop_count"
    ... )

    Notes
    -----
    - The __post_init__ method automatically:
      * Converts single string strat_vars to lists
      * Resolves conflicts when same variable appears in both participant and contact data
      * Adds repeat_part to strat_vars_part if specified
    - For age groups (age_grp_cnt), the contact dataframe column must use pd.IntervalIndex
    - Population columns (age_pop, P) are required for most models
    """

    age_part: str
    age_grp_part: Optional[str] = None
    age_cnt: Optional[str] = None
    age_grp_cnt: Optional[str] = None
    id_col: str = "id"
    y: str = "y"
    z: str = "z"
    strat_vars_part: Optional[Union[List[str], str]] = None
    strat_vars_cnt: Optional[Union[List[str], str]] = None
    repeat_part: Optional[str] = None
    age_pop: Optional[str] = None
    age_grp_pop: Optional[str] = None
    P: Optional[str] = None
    strat_vars_pop: Optional[Union[List[str], str]] = None

    def age_vars(self) -> List[str]:
        """
        Get list of age variable names from contact and participant data.

        Returns
        -------
        List[str]
            Two-element list containing [contact_age_var, participant_age_var].

        Raises
        ------
        ValueError
            If neither age_cnt nor age_grp_cnt is provided.

        Examples
        --------
        >>> col_map = CoordToColumns(age_part="age_p", age_cnt="age_c")
        >>> col_map.age_vars()
        ['age_c', 'age_p']
        """
        if self.age_cnt:
            return [self.age_cnt, self.age_part]
        elif self.age_grp_cnt:
            return [self.age_grp_cnt, self.age_part]
        else:
            raise ValueError(
                "One of 'age_cnt' or 'age_grp_cnt' must be provided. "
                "Please specify either individual contact ages (age_cnt) or "
                "contact age groups (age_grp_cnt)."
            )

    def __post_init__(self) -> None:
        """
        Post-initialization processing and validation.

        Automatically called after dataclass initialization to:
        1. Convert string strat_vars to lists for consistent handling
        2. Validate that age_pop and P are provided together
        3. Validate that contact and population grouping variables match (comparing
           original variable names, i.e., strat_vars_cnt without _cnt suffix)
        4. Resolve naming conflicts between participant and contact stratification variables
        5. Add repeat_part to participant stratification variables if specified

        Raises
        ------
        ValueError
            If age_pop is provided without P, or vice versa.
            If contact grouping variables (without _cnt suffix) do not match
            population grouping variables.

        Warnings
        --------
        UserWarning
            If duplicate stratification variable names are found in both
            strat_vars_part and strat_vars_cnt. The duplicate in strat_vars_cnt
            will be removed.

        Notes
        -----
        The validation compares the original variable names between contact and
        population data. For example, if strat_vars_cnt=['gender_cnt'], it will
        be compared against strat_vars_pop=['gender'] (the _cnt suffix is stripped
        for comparison).
        """
        # Convert single strings to lists for consistent processing
        if isinstance(self.strat_vars_part, str):
            object.__setattr__(self, "strat_vars_part", [self.strat_vars_part])
        elif self.strat_vars_part is None:
            object.__setattr__(self, "strat_vars_part", [])

        if isinstance(self.strat_vars_cnt, str):
            object.__setattr__(self, "strat_vars_cnt", [self.strat_vars_cnt])
        elif self.strat_vars_cnt is None:
            object.__setattr__(self, "strat_vars_cnt", [])

        if isinstance(self.strat_vars_pop, str):
            object.__setattr__(self, "strat_vars_pop", [self.strat_vars_pop])
        elif self.strat_vars_pop is None:
            object.__setattr__(self, "strat_vars_pop", [])

        # Validate population column specifications
        age_pop_specified = self.age_pop is not None or self.age_grp_pop is not None
        if age_pop_specified != (self.P is not None):
            raise ValueError(
                "Both age column (age_pop or age_grp_pop) and 'P' must be specified together, or both left as None. "
                f"Currently: age_pop={self.age_pop}, age_grp_pop={self.age_grp_pop}, P={self.P}"
            )

        # Validate that contact and population grouping variables match
        # Contact variables have _cnt suffix, so we need to remove it for comparison
        strat_vars_cnt_original = [
            var.removesuffix("_cnt") for var in self.strat_vars_cnt
        ]
        if set(strat_vars_cnt_original) != set(self.strat_vars_pop):
            raise ValueError(
                "Contact grouping variables must match population grouping variables. "
                f"Contact variables (without _cnt suffix): {strat_vars_cnt_original}, "
                f"Population variables: {self.strat_vars_pop}"
            )

        # Handle naming conflicts between participant and contact stratification variables
        conflicting_vars = set(self.strat_vars_part).intersection(
            set(self.strat_vars_cnt)
        )
        if conflicting_vars:
            warnings.warn(
                f"Stratification variable(s) with identical names found in both "
                f"participant and contact data: {conflicting_vars}\n"
                f"Default behavior: keeping variable from participant data, "
                f"removing from contact data.\n"
                f"To treat them as separate variables, append suffixes to distinguish them "
                f"(e.g., 'setting_part' vs 'setting_cnt').",
                UserWarning,
                stacklevel=2,
            )
            for var in conflicting_vars:
                self.strat_vars_part.remove(var)

        # Add repeat interview column to participant grouping variables if specified
        # This is needed for data aggregation, but will be excluded from stratification analysis
        if self.repeat_part is not None:
            if self.repeat_part not in self.strat_vars_part:
                self.strat_vars_part.append(self.repeat_part)

    def infer_strat_modes(self) -> Dict[str, StratMode]:
        """
        Infer stratification mode for each variable.

        Determines whether each stratification variable uses PARTIAL or FULL mode
        based on whether the variable appears in both participant and contact data.

        Returns
        -------
        Dict[str, StratMode]
            Mapping from variable name (without suffix) to stratification mode.
            Empty dict if no stratification variables are specified.

        Logic
        -----
        - If var in strat_vars_part AND (var in strat_vars_cnt OR var_cnt in strat_vars_cnt) → FULL
        - If var in strat_vars_part only → PARTIAL

        Notes
        -----
        The variable names are normalized to remove the _part suffix for consistency.
        For example, if strat_vars_part=['gender_part'] and strat_vars_cnt=['gender_cnt'],
        the returned dict will be {'gender': StratMode.FULL}.

        Examples
        --------
        >>> col_map = CoordToColumns(
        ...     age_part='age',
        ...     age_cnt='contact_age',
        ...     strat_vars_part=['gender_part', 'setting_part'],
        ...     strat_vars_cnt=['gender_cnt']
        ... )
        >>> col_map.infer_strat_modes()
        {'gender': <StratMode.FULL: 'full'>, 'setting': <StratMode.PARTIAL: 'partial'>}
        """
        if self.strat_vars_part is None or len(self.strat_vars_part) == 0:
            return {}

        modes = {}

        # Normalize strat_vars_cnt to a set for faster lookup
        cnt_vars_set = set(self.strat_vars_cnt) if self.strat_vars_cnt else set()

        for var in self.strat_vars_part:
            # Remove _part suffix to get the base variable name
            base_var = var.removesuffix("_part")

            if base_var == "repeat":
                continue  # Skip repeat interview variable

            # Check if this variable also appears in contact data
            # It could be named 'var' or 'var_cnt' in strat_vars_cnt
            if base_var in cnt_vars_set or f"{base_var}_cnt" in cnt_vars_set:
                modes[base_var] = StratMode.FULL
            else:
                modes[base_var] = StratMode.PARTIAL

        return modes


class BaseLoader(ABC):
    """
    Abstract base class for loading and preprocessing contact survey data.

    This class provides the core functionality for validating, merging, and transforming
    contact survey data into formats suitable for statistical modeling. It handles:
    - Data validation and quality checks
    - Age range alignment between sample and population data
    - Calculation of sample sizes and contact frequencies
    - Group contact offset corrections
    - Creation of xarray datasets for model input

    Subclasses must implement specific validation methods for their data sources.

    Parameters
    ----------
    data : pd.DataFrame
        Combined dataframe containing both participant and contact information.
        Must include columns specified in col_map.
    pop : pd.DataFrame
        Population dataframe with age distribution and population sizes.
        Used for normalization and rate calculations.
    col_map : CoordToColumns
        Column mapping object specifying how to interpret dataframe columns.
    strat_prop_data : Optional[Union[StratPropData, List[StratPropData]]], default=None

    Attributes
    ----------
    data : pd.DataFrame
        Validated and cleaned input data.
    col_map : CoordToColumns
        Column mapping configuration.
    pop : pd.DataFrame
        Population dataframe.
    min_age : int
        Minimum age in the aligned age range.
    max_age : int
        Maximum age in the aligned age range.
    ds : Optional[xr.Dataset]
        Generated xarray dataset (created by load() method).

    Raises
    ------
    KeyError
        If required columns are missing from input dataframes.
    TypeError
        If age group column is not properly formatted as pd.IntervalIndex,
        or if pop_prop contains non-StratPropData objects.
    ValueError
        If age ranges are inconsistent or population proportions don't sum to 1.

    Notes
    -----
    - Age ranges are automatically aligned between sample and population data
    - Missing values in critical columns will raise errors
    - Object-type columns are automatically converted to categorical
    - Group contacts are handled with special offset corrections

    See Also
    --------
    DataLoader : Concrete implementation for separate participant/contact dataframes
    CoordToColumns : Column mapping specification
    StratPropData : Stratified population proportion specification
    """

    def __init__(
        self,
        data: pd.DataFrame,
        pop_data: pd.DataFrame,
        col_map: CoordToColumns,
        strat_prop_data: Optional[Union[StratPropData, List[StratPropData]]] = None,
    ) -> None:
        """
        Initialize base loader with data validation.

        Parameters
        ----------
        data : pd.DataFrame
            Combined input dataframe.
        pop_data : pd.DataFrame
            Population dataframe.
        col_map : CoordToColumns
            Column mapping specification.
        """
        self.data = data
        self.col_map = col_map
        self.pop_data = pop_data
        self._align_age_range()
        self.ds: Optional[xr.Dataset] = None

        # TODO: Update for new StratPropData structure

    def _align_age_range(self) -> None:
        """
        Align age ranges between sample and population data.

        Parameters
        ----------
        data : pd.DataFrame
            Input dataframe with contact and participant information.
        pop : pd.DataFrame
            Population dataframe with age distribution.
        col_map : CoordToColumns
            Column mapping specification.

        Warnings
        --------
        UserWarning
            - If age ranges between sample and population don't match
        """
        # Determine age ranges and ensure consistency
        part_min_age = int(self.data[self.col_map.age_part].min())
        part_max_age = int(self.data[self.col_map.age_part].max())

        if self.col_map.age_cnt:
            cnt_min_age = int(self.data[self.col_map.age_cnt].min())
            cnt_max_age = int(self.data[self.col_map.age_cnt].max())
        else:  # age_grp_cnt is specified
            cnt_min_age = int(self.data[self.col_map.age_grp_cnt].min().left)
            cnt_max_age = int(self.data[self.col_map.age_grp_cnt].max().right - 1)

        pop_min_age = int(self.pop_data[self.col_map.age_pop].min())
        pop_max_age = int(self.pop_data[self.col_map.age_pop].max())

        # Determine overall sample age range
        sample_min_age = min(part_min_age, cnt_min_age)
        sample_max_age = max(part_max_age, cnt_max_age)

        # Align minimum age with population data
        if sample_min_age != pop_min_age:
            warnings.warn(
                f"Sample minimum age ({sample_min_age}) differs from population minimum age ({pop_min_age}). "
                f"Filtering sample data to match population (age >= {pop_min_age}).",
                UserWarning,
                stacklevel=2,
            )
            self.data = self.data[
                self.data[self.col_map.age_part] >= pop_min_age
            ].copy()
            if self.data.empty:
                raise ValueError(
                    f"After filtering to age >= {pop_min_age}, no data remains. "
                    "Check age range compatibility."
                )
            min_age = pop_min_age
        else:
            min_age = sample_min_age

        # Align maximum age with population data
        if sample_max_age != pop_max_age:
            warnings.warn(
                f"Sample maximum age ({sample_max_age}) differs from population maximum age ({pop_max_age}). "
                f"Using population maximum age ({pop_max_age}) for analysis.",
                UserWarning,
                stacklevel=2,
            )
            max_age = pop_max_age
        else:
            max_age = sample_max_age

        # Store age bounds as instance attributes
        self.min_age: int = min_age
        self.max_age: int = max_age

    def construct_df_n(self) -> pd.DataFrame:
        """Construct dataframe of participant counts (N) stratified by age and grouping variables."""
        strat_vars_n = [self.col_map.age_part]
        if self.col_map.repeat_part:
            strat_vars_n.append(self.col_map.repeat_part)
        if self.col_map.strat_vars_part:
            # Add only variables that aren't already included
            for var in self.col_map.strat_vars_part:
                if var not in strat_vars_n:
                    strat_vars_n.append(var)
        df_n = (
            self.data.groupby(strat_vars_n, observed=False)
            .agg(N=(self.col_map.id_col, "nunique"))
            .reset_index()
        )
        return df_n, strat_vars_n

    def construct_df_S(self) -> pd.DataFrame:
        """Construct dataframe of group contact offsets (S) stratified by age and grouping variables."""
        # [Do] Calculate group contact offsets
        strat_vars_n = [self.col_map.age_part]
        if self.col_map.strat_vars_part:
            strat_vars_n += self.col_map.strat_vars_part
        if self.col_map.repeat_part and self.col_map.repeat_part not in strat_vars_n:
            strat_vars_n.append(self.col_map.repeat_part)

        df_z = (
            self.data[[self.col_map.id_col] + strat_vars_n + [self.col_map.z]]
            .drop_duplicates()
            .groupby(strat_vars_n, observed=True)["z"]
            .sum()
            .reset_index()
        )
        df_yz = (
            self.data[[self.col_map.id_col] + strat_vars_n + [self.col_map.y]]
            .drop_duplicates()
            .groupby(strat_vars_n, observed=True)["y"]
            .sum()
            .reset_index()
        )
        df_S = df_yz.merge(df_z, on=strat_vars_n, how="left")

        # For school age children (5-18) assume contacts are with other children
        mask = (
            (df_S[self.col_map.age_part] >= 5)
            & (df_S[self.col_map.age_part] <= 18)
            & (df_S[self.col_map.z] + df_S[self.col_map.y] > 0)
        )
        df_S["S"] = np.where(
            mask,
            1 - df_S[self.col_map.z] / (df_S[self.col_map.z] + df_S[self.col_map.y]),
            1.0,
        )

        # For adults (18+) assume contacts are random across population
        mask = (df_S[self.col_map.age_part] > 18) & (
            df_S[self.col_map.z] + df_S[self.col_map.y] > 0
        )
        df_S["S"] = np.where(
            mask,
            1
            - df_S[self.col_map.z]
            / (df_S[self.col_map.z] + df_S[self.col_map.y])
            / (self.max_age - self.min_age + 1),
            1.0,
        )

        df_S = df_S.drop(columns=[self.col_map.z, self.col_map.y])

        return df_S

    def construct_df_y(self) -> pd.DataFrame:
        """Construct dataframe of contact counts (y) stratified by age and grouping variables."""
        strat_vars = self.col_map.age_vars()
        if self.col_map.repeat_part:
            strat_vars.append(self.col_map.repeat_part)
        if self.col_map.strat_vars_part:
            # Add only variables that aren't already included
            for var in self.col_map.strat_vars_part:
                if var not in strat_vars:
                    strat_vars.append(var)
        if self.col_map.strat_vars_cnt:
            # Add only variables that aren't already included
            for var in self.col_map.strat_vars_cnt:
                if var not in strat_vars:
                    strat_vars.append(var)

        df_y = (
            self.data.groupby(strat_vars, observed=False)
            .agg({self.col_map.y: "sum"})
            .reset_index()
        )
        return df_y, strat_vars

    def construct_df_full(
        self, df_n: pd.DataFrame, df_S: pd.DataFrame, df_y: pd.DataFrame
    ) -> pd.DataFrame:
        """Construct full dataframe with all combinations of stratification variables."""
        df_n, strat_vars_n = self.construct_df_n()
        df_S = self.construct_df_S()
        df_y, strat_vars = self.construct_df_y()

        # Create a full Cartesian product of all stratification variable levels
        unique_coords = {var: self.data[var].unique() for var in strat_vars}
        unique_coords[self.col_map.age_part] = np.arange(
            self.min_age, self.max_age + 1, dtype=int
        )

        if self.col_map.age_cnt:
            unique_coords[self.col_map.age_cnt] = np.arange(
                self.min_age, self.max_age + 1, dtype=int
            )
        elif self.col_map.age_grp_cnt:
            unique_coords[self.col_map.age_grp_cnt] = self.data[
                self.col_map.age_grp_cnt
            ].cat.categories

        index = pd.MultiIndex.from_product(
            unique_coords.values(), names=unique_coords.keys()
        )
        df_full = pd.DataFrame(list(index), columns=unique_coords.keys())

        # Restore all category information for categorical columns
        if self.col_map.age_grp_cnt:
            df_full[self.col_map.age_grp_cnt] = pd.Categorical(
                df_full[self.col_map.age_grp_cnt],
                categories=self.data[self.col_map.age_grp_cnt].cat.categories,
                ordered=True,
            )
        if self.col_map.strat_vars_part:
            for var in self.col_map.strat_vars_part:
                if isinstance(self.data[var].dtype, pd.CategoricalDtype):
                    categories = self.data[var].cat.categories
                else:
                    categories = self.data[var].unique()
                df_full[var] = pd.Categorical(
                    df_full[var],
                    categories=categories,
                    ordered=True,
                )
        if self.col_map.strat_vars_cnt:
            for var in self.col_map.strat_vars_cnt:
                if isinstance(self.data[var].dtype, pd.CategoricalDtype):
                    categories = self.data[var].cat.categories
                else:
                    categories = self.data[var].unique()
                df_full[var] = pd.Categorical(
                    df_full[var],
                    categories=categories,
                    ordered=True,
                )

        df_full = pd.merge(df_full, df_y, on=strat_vars, how="left")
        df_full = pd.merge(df_full, df_n, on=strat_vars_n, how="left")
        df_full = pd.merge(df_full, df_S, on=strat_vars_n, how="left")

        # [Do] Finalise the data
        df_full = df_full.dropna(subset=["N"])
        df_full = df_full[df_full["N"] > 0]
        df_full["S"] = df_full["S"].fillna(1.0)
        df_full["log_S"] = np.where(df_full["S"] > 0, np.log(df_full["S"]), 0.0)
        df_full["y"] = df_full["y"].fillna(0)

        return df_full

    def construct_log_P(self) -> Union[NDArray, Dict[str, NDArray]]:
        """
        Construct log population proportions (log_P) stratified by age and grouping variables.

        Returns
        -------
        Union[NDArray, Dict[str, NDArray]]
            - If no stratification variables in population data: 1D array of log population sizes
                indexed by age.
            - If stratification variables present: Dict mapping variable name to 1D array of
                log population sizes indexed by age and that variable.
        """
        # Merge population data with stratification variables
        if self.col_map.strat_vars_pop:
            log_P = {}
            for var in self.col_map.strat_vars_pop:
                group_cols = [self.col_map.age_pop, var]
                # Calculate the marginal population size (P^t_a)
                df_strat_P = (
                    self.pop_data.groupby(group_cols, observed=False)[self.col_map.P]
                    .sum()
                    .reset_index()
                )
                log_P[var] = np.log(df_strat_P[self.col_map.P].to_numpy())
        else:
            # No stratification variables in population data only age (P_a)
            log_P = np.log(self.pop_data[self.col_map.P].to_numpy())

        return log_P

    def load(self) -> ModelData:
        """
        Load and transform data into a ModelData for model fitting.

        This is the main method that transforms raw contact survey data into a
        structured container with two components:

        1. **ModelBaseData**: Contains the numerical arrays needed for inference
           - Contact counts (y), age indices (aid, bid)
           - Sample sizes (log_N), population distribution (log_P)
           - Optional: offsets (log_S), repeat indicators (rid)

        2. **ModelStratData**: Contains hierarchical stratification information
           - strat_vars: Variable names and their categories
           - strat_modes: PARTIAL vs FULL stratification for each variable
           - strat_vars_full: Participant and contact categories for FULL mode
           - strat_ix: Categorical codes for indexing

        The method performs several key transformations:
        1. Aggregates participant counts by age and grouping variables
        2. Computes group contact offsets to adjust for household contacts
        3. Creates contact count matrix across all age combinations
        4. Builds full Cartesian product of all stratification variables
        5. Merges aggregated data with full grid (zero-filling missing cells)
        6. Constructs ModelBaseData and ModelStratData containers

        Returns
        -------
        ModelData
            Type-safe container with arrays and optional stratification metadata.
            Contains:
            - arrays: ModelBaseData (required fields: y, aid, bid, log_N, log_P, age_min, age_max)
            - strat_metadata: ModelStratData or None (stratification configuration)

        Notes
        -----
        **Group Contact Offset (S)**:
        The offset S adjusts for group contacts (e.g., "household members")
        that inflate contact counts. Different adjustments are applied:
        - Children (5-18): Assumes group contacts are with other children
        - Adults (18+): Assumes group contacts are random across population
        - S = 1 - z/(z+y) for children
        - S = 1 - z/(z+y)/(max_age - min_age + 1) for adults

        Where:
        - z = number of group contacts
        - y = number of individual contacts

        **Stratification Modes**:
        - PARTIAL: Variable recorded for participants only (e.g., setting)
        - FULL: Variable recorded for both participants AND contacts (e.g., gender)

        The method automatically:
        - Zero-fills for unobserved age combinations
        - Maintains categorical variable consistency
        - Expands age group intervals for coarse age data
        - Detects PARTIAL vs FULL stratification modes

        See Also
        --------
        ModelData : Type-safe container for model inputs
        ModelStratData : TypedDict for stratification configuration
        docs/full_stratification_integration_guide.md : Detailed integration diagrams

        Examples
        --------
        >>> # Basic usage
        >>> loader = BaseLoader(data, pop, col_map)
        >>> container = loader.load()
        >>> container.arrays['y'].shape
        (12345,)  # Number of unique strata
        >>>
        >>> # Access stratification metadata
        >>> if container.strat_metadata:
        ...     print(container.strat_metadata['strat_modes'])
        {'gender': 'full', 'setting': 'partial'}
        >>>
        >>> # Check for specific fields
        >>> container.has('log_S')  # Check if offset exists
        True
        >>> container.age_range
        (0, 75)
        """
        # No. of participants stratified by age and other grouping variables
        self.df_n = self.construct_df_n()

        # Offsets for group contacts
        self.df_S = self.construct_df_S()

        # No. of contacts stratified by age and other grouping variables
        self.df_y = self.construct_df_y()

        # Create a full grid of all combinations of the grouping variables via a cartesian product
        df_full = self.construct_df_full(self.df_n, self.df_S, self.df_y)

        # ========================
        # Construct ModelBaseData
        # ========================
        base_data = ModelBaseData(
            y=df_full["y"].to_numpy(),
            aid=df_full[self.col_map.age_part].to_numpy(),
            log_N=df_full["N"].to_numpy(),
            log_S=df_full["log_S"].to_numpy(),
            log_P=self.construct_log_P(),
            age_min=self.min_age,
            age_max=self.max_age,
        )

        if self.col_map.age_cnt:
            base_data["bid"] = df_full[self.col_map.age_cnt].to_numpy()
        elif self.col_map.age_grp_cnt:
            # [Do] Create indices for age aggregation
            aid_exp, bid_pad = make_idarrs_for_intervals(
                df_full, self.col_map.age_grp_cnt, base_data["aid"]
            )
            base_data["aid_exp"] = aid_exp
            base_data["bid_pad"] = bid_pad

        # If repeat effects are specified
        if self.col_map.repeat_part is not None:
            base_data["rid"] = df_full[self.col_map.repeat_part].astype(int).to_numpy()

        # ============================
        # Construct ModelStratData
        # Handles stratification logic for hierarchical models (HiBRCfine/HiBRCrefine)
        # ============================
        #
        # ModelStratData is a TypedDict that encapsulates all stratification information:
        #
        # 1. strat_vars: Dict[str, List[str]]
        #    Maps variable name → list of category names
        #    Example: {'gender': ['male', 'female'], 'setting': ['home', 'work', 'other']}
        #
        # 2. strat_modes: Dict[str, str]
        #    Maps variable name → mode ('partial' or 'full')
        #    - PARTIAL: Variable recorded for participants only
        #    - FULL: Variable recorded for both participants AND contacts
        #    Example: {'gender': 'full', 'setting': 'partial'}
        #
        # 3. strat_vars_full: Dict[str, Dict[str, List[str]]]
        #    Only populated for FULL mode variables
        #    Structure: {var_name: {'part': [categories], 'cnt': [categories]}}
        #    Allows validation that participant and contact categories match
        #    Example: {'gender': {'part': ['male', 'female'], 'cnt': ['male', 'female']}}
        #
        # 4. strat_ix: Dict[str, NDArray]
        #    Maps variable name → array of categorical codes (integers)
        #    For FULL mode, includes both 'var' and 'var_cnt' keys
        #    Example: {'gender': array([0,1,0,1,...]), 'gender_cnt': array([0,0,1,1,...])}
        #
        # See docs/full_stratification_integration_guide.md for detailed diagrams
        # ============================

        # Step 1: Infer stratification modes (PARTIAL vs FULL)
        strat_modes = self.col_map.infer_strat_modes()

        if len(strat_modes) == 0:  # No stratification variables
            strat_data = {}
        else:
            # Step 2: Build strat_vars - extract category names for each variable
            # Example: For gender with categories male/female, stores {'gender': ['male', 'female']}
            strat_vars = {}
            for var, mode in strat_modes.items():
                # Get unique categories from the dataframe column
                part_col = f"{var}_part"  # Assumption: There is always the strat var for participant
                if part_col in df_full.columns:
                    if isinstance(df_full[part_col].dtype, pd.CategoricalDtype):
                        categories = df_full[part_col].cat.categories.tolist()
                    else:
                        categories = df_full[part_col].unique().tolist()
                    strat_vars[var] = categories  # Use base variable name as key
                else:
                    raise ValueError(
                        f"Stratification variable '{part_col}' not found in processed data. "
                        f"Available columns: {df_full.columns.tolist()}"
                    )

            # Step 3: Build strat_vars_full - only for FULL mode variables
            # This provides participant and contact categories separately for validation
            # Example: {'gender': {'part': ['male', 'female'], 'cnt': ['male', 'female']}}
            strat_vars_full = {}
            for var, mode in strat_modes.items():
                if mode == StratMode.FULL:
                    # For FULL mode, extract categories from both participant and contact columns
                    # Participant column: 'gender'
                    # Contact column: 'gender_cnt'
                    part_col = f"{var}_part"
                    cnt_col = f"{var}_cnt"

                    if part_col not in df_full.columns:
                        raise ValueError(
                            f"FULL stratification for '{var}' requires participant column '{part_col}'"
                        )
                    if cnt_col not in df_full.columns:
                        raise ValueError(
                            f"FULL stratification for '{var}' requires contact column '{cnt_col}'"
                        )

                    if isinstance(df_full[part_col].dtype, pd.CategoricalDtype):
                        part_categories = df_full[part_col].cat.categories.tolist()
                    else:
                        part_categories = df_full[part_col].unique().tolist()

                    if isinstance(df_full[cnt_col].dtype, pd.CategoricalDtype):
                        cnt_categories = df_full[cnt_col].cat.categories.tolist()
                    else:
                        cnt_categories = df_full[cnt_col].unique().tolist()

                    strat_vars_full[var] = {
                        "part": part_categories,
                        "cnt": cnt_categories,
                    }

            # Step 4: Build strat_ix - categorical codes (integer indices) for each observation
            # These codes are used for indexing into prior samples during model inference
            #
            # Keys always use _part/_cnt suffixes to match DataFrame columns:
            #   For PARTIAL mode:
            #     strat_ix['setting_part'] = array([0, 1, 2, 1, ...])  # Participant setting codes
            #
            #   For FULL mode:
            #     strat_ix['gender_part'] = array([0, 1, 0, 1, ...])   # Participant gender codes
            #     strat_ix['gender_cnt'] = array([0, 0, 1, 1, ...])    # Contact gender codes
            #
            # Models use these to compute flat indices:
            #   PARTIAL: flat_idx = strat_ix['setting_part'][i]
            #   FULL:    flat_idx = strat_ix['gender_part'][i] * K + strat_ix['gender_cnt'][i]
            strat_ix = {}
            for var, mode in strat_modes.items():
                # Always include participant codes with _part suffix
                part_col = f"{var}_part"
                if part_col in df_full.columns:
                    if isinstance(df_full[part_col].dtype, pd.CategoricalDtype):
                        strat_ix[part_col] = df_full[part_col].cat.codes.to_numpy()
                    else:
                        # Convert to categorical first, then get codes
                        cat_col = pd.Categorical(df_full[part_col])
                        strat_ix[part_col] = cat_col.codes

                # For FULL mode, also include contact codes with _cnt suffix
                if mode == StratMode.FULL:
                    cnt_col = f"{var}_cnt"
                    if cnt_col in df_full.columns:
                        if isinstance(df_full[cnt_col].dtype, pd.CategoricalDtype):
                            strat_ix[cnt_col] = df_full[cnt_col].cat.codes.to_numpy()
                        else:
                            # Convert to categorical first, then get codes
                            cat_col = pd.Categorical(df_full[cnt_col])
                            strat_ix[cnt_col] = cat_col.codes

            # Step 5: Construct ModelStratData TypedDict
            # This object is passed to model classes (HiBRCfine/HiBRCrefine) which use it
            # to build StratConfig and StratIndexer objects for each variable
            strat_data = ModelStratData(
                strat_vars=strat_vars,
                strat_modes={
                    var: mode.value for var, mode in strat_modes.items()
                },  # Convert enum to string
                strat_vars_full=strat_vars_full,
                strat_ix=strat_ix,
            )

        # ============================
        # Construct ModelData
        # ============================
        self.model_data = ModelData(base_data, strat_data)

        return self.model_data


class DataLoader(BaseLoader):
    """
    Prepare contact survey data for Bayesian contact matrix estimation.

    This class handles the complete data preparation pipeline for contact matrix
    models, starting from validated ParticipantData, ContactData, and PopulationData
    objects. It:
    1. Merges contact and participant data
    2. Transforms data into xarray Dataset format via BaseLoader

    The DataLoader is the primary entry point for users working with standard
    contact survey data where participants and contacts are stored in separate
    dataframes (e.g., CoMix, POLYMOD surveys).

    Parameters
    ----------
    part_data : ParticipantData
        Validated participant data object containing preprocessed participant information.
        Already validated with standardized column names (id, age_part, {var}_part, z).
    cnt_data : ContactData
        Validated contact data object containing preprocessed contact information.
        Already validated with standardized column names (id, age_cnt, {var}_cnt, y).
    pop_data : PopulationData
        Validated population data object containing population age distribution.
        Already validated with standardized column names (age, P).
    strat_prop_data : Union[StratPropData, List[StratPropData], None], optional
        Population proportion specification(s) for demographic adjustment.
        Can be either:
        - A single StratPropData object
        - A list of StratPropData objects for multiple stratifications
        If None, no stratified population proportions are used.

        Example with single stratification:
            strat_prop_data = StratPropData.from_counts(
                data=df_gender, age_col='age', strat_col='gender', count_col='N'
            )
            DataLoader(part_data, cnt_data, pop_data, strat_prop_data=strat_prop_data)

        Example with multiple stratifications:
            strat_prop_gender = StratPropData.from_counts(...)
            strat_prop_region = StratPropData.from_counts(...)
            DataLoader(part_data, cnt_data, pop_data, strat_prop_data=[strat_prop_gender, strat_prop_region])

    Attributes
    ----------
    part_data : ParticipantData
        Validated participant data object.
    cnt_data : ContactData
        Validated contact data object.
    pop_data : PopulationData
        Validated population data object.
    col_map : CoordToColumns
        Generated column mapping object based on dataclass structures.
    data : pd.DataFrame
        Merged participant-contact dataframe passed to BaseLoader.

    Methods
    -------
    load()
        Inherited from BaseLoader - transforms data to xarray Dataset.

    Raises
    ------
    TypeError
        If inputs are not the correct dataclass types.
        If pop_prop contains non-StratPropData objects.

    Notes
    -----
    - All validation is performed by the dataclasses (ParticipantData, ContactData, PopulationData)
    - Column names are standardized by the dataclasses
    - Participant and contact data are merged on the 'id' column
    - No redundant validation is performed in DataLoader

    Examples
    --------
    >>> from cntmosaic.dataloader import (
    ...     DataLoader, ParticipantData, ContactData, PopulationData,
    ...     StratPropData
    ... )
    >>>
    >>> # Create validated data objects
    >>> part_data = ParticipantData(
    ...     df_part=part_df,
    ...     id_col='participant_id',
    ...     age_col='age',
    ...     strat_var_cols='gender'
    ... )
    >>>
    >>> cnt_data = ContactData(
    ...     df_cnt=cnt_df,
    ...     id_col='participant_id',
    ...     age_col='contact_age',
    ...     strat_vars='setting'
    ... )
    >>>
    >>> pop_data = PopulationData(
    ...     df_pop=pop_df,
    ...     age_col='age',
    ...     size_col='population'
    ... )
    >>>
    >>> # Create population proportion (single stratification)
    >>> pop_prop = StratPropData.from_counts(
    ...     data=df_gender,
    ...     age_col='age',
    ...     strat_col='gender',
    ...     count_col='population'
    ... )
    >>>
    >>> # Load data
    >>> loader = DataLoader(part_data, cnt_data, pop_data, pop_prop=pop_prop)
    >>> ds = loader.load()
    >>>
    >>> # Access contact matrix data
    >>> ds.y  # Contact counts
    >>> ds.log_N  # Log participant counts
    >>> ds.pop_prop_gender  # Stratified population proportions by gender
    """

    def __init__(
        self,
        part_data,  # ParticipantData type hint removed to avoid circular import
        cnt_data,  # ContactData type hint removed to avoid circular import
        pop_data,  # PopulationData type hint removed to avoid circular import
        strat_prop_data: Union[StratPropData, List[StratPropData], None] = None,
    ) -> None:
        # Import here to avoid circular dependency
        from .containers._ContactData import ContactData

        # Validate input types
        if not isinstance(part_data, ParticipantData):
            raise TypeError(
                f"part_data must be a ParticipantData object, got {type(part_data).__name__}"
            )
        if not isinstance(cnt_data, ContactData):
            raise TypeError(
                f"cnt_data must be a ContactData object, got {type(cnt_data).__name__}"
            )
        if not isinstance(pop_data, PopulationData):
            raise TypeError(
                f"pop_data must be a PopulationData object, got {type(pop_data).__name__}"
            )
        if not isinstance(strat_prop_data, (type(None), StratPropData, list)):
            raise TypeError(
                "strat_prop_data must be None, a StratPropData object, "
                "or a list of StratPropData objects."
            )

        # Store dataclass objects
        self.part_data = part_data
        self.cnt_data = cnt_data
        self.pop_data = pop_data
        self.strat_prop_data = strat_prop_data

        # Create CoordToColumns from dataclass structures
        col_map = self._create_col_map(part_data, cnt_data, pop_data)

        # Merge contact and participant data on 'id' column
        data = pd.merge(cnt_data.data, part_data.data, on="id")

        # Initialize parent class with merged data
        super().__init__(data, pop_data.data, col_map, strat_prop_data)

    def _create_col_map(self, part_data, cnt_data, pop_data) -> CoordToColumns:
        """
        Create CoordToColumns object from dataclass structures.

        Extracts column information from the standardized dataclass objects
        and builds a CoordToColumns configuration for BaseLoader.

        Parameters
        ----------
        part_data : ParticipantData
            Validated participant data object.
        cnt_data : ContactData
            Validated contact data object.
        pop_data : PopulationData
            Validated population data object.

        Returns
        -------
        CoordToColumns
            Column mapping configuration for BaseLoader.
        """
        if part_data.strat_var_cols:
            strat_vars_part = []
            for var in part_data.strat_var_cols:
                if var.endswith("_part"):
                    strat_vars_part.append(var)
                else:
                    strat_vars_part.append(f"{var}_part")
        else:
            strat_vars_part = None

        if cnt_data.strat_var_cols:
            strat_vars_cnt = []
            for var in cnt_data.strat_var_cols:
                if var.endswith("_cnt"):
                    strat_vars_cnt.append(var)
                else:
                    strat_vars_cnt.append(f"{var}_cnt")
        else:
            strat_vars_cnt = None

        # Create CoordToColumns
        col_map = CoordToColumns(
            age_part="age_part" if part_data.age_col else "age_grp_part",
            age_cnt="age_cnt" if cnt_data.age_col else None,
            age_grp_cnt="age_grp_cnt" if cnt_data.age_grp_col else None,
            id_col="id",
            y="y",
            z="z",
            strat_vars_part=strat_vars_part,
            strat_vars_cnt=strat_vars_cnt,
            repeat_part="repeat_part" if part_data.repeat_col else None,
            age_pop="age",
            P="P",
            strat_vars_pop=pop_data.strat_var_cols if pop_data.strat_var_cols else None,
        )

        return col_map
