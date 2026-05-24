"""
Column name mapping for contact survey data.

Internal API — not exported from ``cntmosaic.dataloader``. Used exclusively by
``BaseLoader`` and ``DataLoader`` to specify the mapping between input dataframe
columns and the variables required by contact matrix estimation models.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import TYPE_CHECKING, List, Optional, Union

if TYPE_CHECKING:
    from .containers._ContactData import ContactData
    from .containers._ParticipantData import ParticipantData
    from .containers._PopulationData import PopulationData


@dataclass(frozen=True)
class ColumnSpec:
    """
    This dataclass helps manage the id, age, stratification, contact counts, group conntact counts,
    repeated interview counts, and population count column names for participant, contact, and population dataframes.

    It provides methods to retrieve columns used for group-then-summarise operations in the DataLoader preprocessing pipeline.
    Specifically, it provides the grouping variables needed to construct `df_n`, `df_y`, `df_S`, and `df_full` dataframes in the BaseLoader.

    Note: This class is used internally by DataLoader and is not typically instantiated directly by users.

    Attributes
    ----------
    part_age : str
        Column name for participant age in the participant dataframe.
        Should contain integer ages or age groups.
    part_age_grp : Optional[str], default=None
        Column name for participant age group in the participant dataframe.
        Must be pd.IntervalIndex if used. Use this OR part_age, not both.
    cnt_age : Optional[str], default=None
        Column name for contact age in the contact dataframe.
        Use this OR cnt_age_grp, not both.
    cnt_age_grp : Optional[str], default=None
        Column name for contact age group in the contact dataframe.
        Must be pd.IntervalIndex if used. Use this OR cnt_age, not both.
    id_col : str, default='id'
        Column name for unique participant identifiers.
        Must be present in both participant and contact dataframes.
    y : str, default='y'
        Column name for number of contacts in the contact dataframe.
        If not present, will be auto-created with value 1 per contact.
    z : str, default='z'
        Column name for number of group contacts in the participant dataframe.
        If not present, will be auto-created with value 0.
    part_strat_vars : Optional[Union[List[str], str]], default=None
        Stratification variable column name(s) in participant dataframe.
        Can be a single string or list of strings. Examples: 'gender', ['gender', 'setting']
    cnt_strat_vars : Optional[Union[List[str], str]], default=None
        Stratification variable column name(s) in contact dataframe.
        Can be a single string or list of strings.
    part_repeat : Optional[str], default=None
        Column name for repeat interview count in participant dataframe.
        Used to model repeat participation effects. If provided, automatically
        added to part_strat_vars during post-initialization.
    pop_age : Optional[str], default=None
        Column name for age in population dataframe.
        Required for population weighting; must be provided with P.
    pop_age_grp : Optional[str], default=None
        Column name for age group in population dataframe.
        Must be pd.IntervalIndex if used. Use this OR pop_age, not both.
    P : Optional[str], default=None
        Column name for population size/proportion in population dataframe.
        Required for population weighting; must be provided with pop_age or pop_age_grp.
    pop_strat_vars : Optional[Union[List[str], str]], default=None
        Stratification variable column name(s) in population dataframe.
        Can be a single string or list of strings.

    Raises
    ------
    ValueError
        If neither cnt_age nor cnt_age_grp is provided.
        If pop_age and P are not both set or both None.

    Warnings
    --------
    UserWarning
        If the same stratification variable appears in both part_strat_vars and
        cnt_strat_vars. The variable in cnt_strat_vars will be automatically removed
        to avoid ambiguity.

    Examples
    --------
    >>> # Basic usage with individual contact ages
    >>> col_map = ColumnSpec(
    ...     part_age="participant_age",
    ...     cnt_age="contact_age",
    ...     id_col="participant_id",
    ...     pop_age="age",
    ...     P="population_size"
    ... )
    >>>
    >>> # With age groups and stratification
    >>> col_map = ColumnSpec(
    ...     part_age="part_ageicipant",
    ...     cnt_age_grp="age_group_contact",
    ...     part_strat_vars=["gender", "location"],
    ...     cnt_strat_vars="setting",
    ...     pop_age="age",
    ...     P="N"
    ... )
    >>>
    >>> # With repeat interview effects
    >>> col_map = ColumnSpec(
    ...     part_age="age",
    ...     cnt_age="contact_age",
    ...     part_repeat="interview_round",
    ...     pop_age="age",
    ...     P="pop_count"
    ... )

    Notes
    -----
    - The __post_init__ method automatically:
      * Converts single string strat_vars to lists
      * Resolves conflicts when same variable appears in both participant and contact data
      * Adds part_repeat to part_strat_vars if specified
    - For age groups (cnt_age_grp), the contact dataframe column must use pd.IntervalIndex
    - Population columns (pop_age, P) are required for most models
    """

    part_age: str
    part_age_grp: Optional[str] = None
    cnt_age: Optional[str] = None
    cnt_age_grp: Optional[str] = None
    id_col: str = "id"
    y: str = "y"
    z: Optional[str] = None
    part_strat_vars: Optional[str] = None
    cnt_strat_vars: Optional[str] = None
    part_repeat: Optional[str] = None
    pop_age: Optional[str] = None
    pop_age_grp: Optional[str] = None
    P: Optional[str] = None
    pop_strat_vars: Optional[Union[List[str], str]] = None

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
            If neither cnt_age nor cnt_age_grp is provided.

        Examples
        --------
        >>> col_map = ColumnSpec(part_age="age_p", cnt_age="age_c")
        >>> col_map.age_vars()
        ['age_c', 'age_p']
        """
        if self.cnt_age:
            return [self.cnt_age, self.part_age]
        elif self.cnt_age_grp:
            return [self.cnt_age_grp, self.part_age]
        else:
            raise ValueError(
                "One of 'cnt_age' or 'cnt_age_grp' must be provided. "
                "Please specify either individual contact ages (cnt_age) or "
                "contact age groups (cnt_age_grp)."
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
        strat_vars_n = [self.part_age]
        if self.part_repeat:
            strat_vars_n.append(self.part_repeat)
        if self.part_strat_vars:
            # Add only variables not already included
            for var in self.part_strat_vars:
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
        if self.part_repeat:
            strat_vars.append(self.part_repeat)
        if self.part_strat_vars:
            # Add only variables that aren't already included
            for var in self.part_strat_vars:
                if var not in strat_vars:
                    strat_vars.append(var)
        if self.cnt_strat_vars:
            # Add only variables that aren't already included
            for var in self.cnt_strat_vars:
                if var not in strat_vars:
                    strat_vars.append(var)

        return strat_vars

    def __post_init__(self) -> None:
        """
        Post-initialization processing and validation.

        Automatically called after dataclass initialization to:
        1. Convert string strat_vars to lists for consistent handling
        2. Validate that pop_age and P are provided together
        3. Validate that contact and population grouping variables match (comparing
           original variable names, i.e., cnt_strat_vars without _cnt suffix)
        4. Resolve naming conflicts between participant and contact stratification variables
        5. Add part_repeat to participant stratification variables if specified

        Raises
        ------
        ValueError
            If pop_age is provided without P, or vice versa.
            If contact grouping variables (without _cnt suffix) do not match
            population grouping variables.

        Warnings
        --------
        UserWarning
            If duplicate stratification variable names are found in both
            part_strat_vars and cnt_strat_vars. The duplicate in cnt_strat_vars
            will be removed.

        Notes
        -----
        The validation compares the original variable names between contact and
        population data. For example, if cnt_strat_vars=['gender_cnt'], it will
        be compared against pop_strat_vars=['gender'] (the _cnt suffix is stripped
        for comparison).
        """
        # Convert single strings to lists for consistent processing
        if isinstance(self.part_strat_vars, str):
            object.__setattr__(self, "part_strat_vars", [self.part_strat_vars])
        elif self.part_strat_vars is None:
            object.__setattr__(self, "part_strat_vars", [])

        if isinstance(self.cnt_strat_vars, str):
            object.__setattr__(self, "cnt_strat_vars", [self.cnt_strat_vars])
        elif self.cnt_strat_vars is None:
            object.__setattr__(self, "cnt_strat_vars", [])

        if isinstance(self.pop_strat_vars, str):
            object.__setattr__(self, "pop_strat_vars", [self.pop_strat_vars])
        elif self.pop_strat_vars is None:
            object.__setattr__(self, "pop_strat_vars", [])

    @classmethod
    def from_containers(
        cls,
        part_data: ParticipantData,
        cnt_data: ContactData,
        pop_data: PopulationData,
    ) -> ColumnSpec:
        """
        Build a ColumnSpec from validated container objects.

        This is the canonical factory used by DataFrameSurveySource and
        ContactSurveyLoader to avoid manually constructing column specs.
        """
        if part_data.strat_var_cols:
            part_strat_vars = [
                var if var.startswith("part_") else f"part_{var}"
                for var in part_data.strat_var_cols
            ]
        else:
            part_strat_vars = None

        if cnt_data.strat_var_cols:
            cnt_strat_vars = [
                var if var.startswith("cnt_") else f"cnt_{var}"
                for var in cnt_data.strat_var_cols
            ]
        else:
            cnt_strat_vars = None

        _part_has_coarse = bool(
            part_data.age_grp_col or (part_data.age_min_col and part_data.age_max_col)
        )
        _cnt_has_coarse = bool(
            cnt_data.age_grp_col or (cnt_data.age_min_col and cnt_data.age_max_col)
        )

        return cls(
            part_age="part_age" if part_data.age_col else "part_age_grp",
            part_age_grp="part_age_grp" if _part_has_coarse else None,
            cnt_age="cnt_age" if cnt_data.age_col else None,
            cnt_age_grp="cnt_age_grp" if _cnt_has_coarse else None,
            id_col="id",
            y="y",
            z=part_data.amb_cnt_col,
            part_strat_vars=part_strat_vars,
            cnt_strat_vars=cnt_strat_vars,
            part_repeat="part_repeat" if part_data.repeat_col else None,
            pop_age="age" if pop_data.age_col else None,
            pop_age_grp="pop_age_grp" if pop_data.age_grp_col else None,
            P="P",
            pop_strat_vars=pop_data.strat_var_cols if pop_data.strat_var_cols else None,
        )
