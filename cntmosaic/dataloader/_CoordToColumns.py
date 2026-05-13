"""
Column name mapping for contact survey data.

Internal API — not exported from ``cntmosaic.dataloader``. Used exclusively by
``BaseLoader`` and ``DataLoader`` to specify the mapping between input dataframe
columns and the variables required by contact matrix estimation models.
"""

import warnings
from dataclasses import dataclass
from typing import List, Optional, Union


@dataclass
class CoordToColumns:
    """
    This dataclass helps manage the id, age, stratification, contact counts, group conntact counts,
    repeated interview counts, and population count column names for participant, contact, and population dataframes.

    It provides methods to retrieve columns used for group-then-summarise operations in the DataLoader preprocessing pipeline.
    Specifically, it provides the grouping variables needed to construct `df_n`, `df_y`, `df_S`, and `df_full` dataframes in the BaseLoader.

    Note: This class is used internally by DataLoader and is not typically instantiated directly by users.

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
    z: Optional[str] = None
    strat_vars_part: Optional[str] = None
    strat_vars_cnt: Optional[str] = None
    repeat_part: Optional[str] = None
    age_pop: Optional[str] = None
    age_grp_pop: Optional[str] = None
    P: Optional[str] = None
    strat_vars_pop: Optional[Union[List[str], str]] = None

    @property
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

    @property
    def strat_vars_n(self) -> List[str]:
        """
        Get list of grouping variable names for constructing `df_n`.

        Returns
        -------
        List[str]
            List of variable names used to group participant data for `df_n`.
        """
        strat_vars_n = [self.age_part]
        if self.repeat_part:
            strat_vars_n.append(self.repeat_part)
        if self.strat_vars_part:
            # Add only variables not already included
            for var in self.strat_vars_part:
                if var not in strat_vars_n:
                    strat_vars_n.append(var)

        return strat_vars_n

    @property
    def strat_vars_y(self) -> List[str]:
        """
        Get list of grouping variable names for constructing `df_y`.

        Returns
        -------
        List[str]
            List of variable names used to group contact data for `df_y`.
        """
        strat_vars = self.age_vars.copy()
        if self.repeat_part:
            strat_vars.append(self.repeat_part)
        if self.strat_vars_part:
            # Add only variables that aren't already included
            for var in self.strat_vars_part:
                if var not in strat_vars:
                    strat_vars.append(var)
        if self.strat_vars_cnt:
            # Add only variables that aren't already included
            for var in self.strat_vars_cnt:
                if var not in strat_vars:
                    strat_vars.append(var)

        return strat_vars

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
