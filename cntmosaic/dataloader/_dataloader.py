from typing import List, Tuple, Union

import pandas as pd

from ._BaseLoader import BaseLoader
from ._CoordToColumns import CoordToColumns
from ._DataValidator import DataValidator
from .containers._ContactData import ContactData
from .containers._ParticipantData import ParticipantData
from .containers._PopulationData import PopulationData
from .containers._StratificationData import StratificationData


class DataLoader(BaseLoader):
    """
    Prepare contact survey data for Bayesian contact matrix estimation.

    This class handles the complete data preparation pipeline for contact matrix
    models.

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
    strat_data : Union[StratPropData, List[StratPropData], None], optional
        Population proportion specification(s) for demographic adjustment.
        Can be either:
        - A single StratPropData object
        - A list of StratPropData objects for multiple stratifications
        If None, no stratified population proportions are used.

        Example with single stratification:
            strat_data = StratPropData.from_counts(
                data=df_gender, age_col='age', strat_col='gender', count_col='N'
            )
            DataLoader(part_data, cnt_data, pop_data, strat_data=strat_data)

        Example with multiple stratifications:
            strat_prop_gender = StratPropData.from_counts(...)
            strat_prop_region = StratPropData.from_counts(...)
            DataLoader(part_data, cnt_data, pop_data, strat_data=[strat_prop_gender, strat_prop_region])

    Attributes
    ----------
    part_data : ParticipantData
        Validated participant data object.
    cnt_data : ContactData
        Validated contact data object.
    pop_data : PopulationData
        Validated population data object.
    strat_data : Union[StratPropData, None]
        Population proportion specification(s) for demographic adjustment.
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
    - **Composite Stratification**: If multiple stratification variables are specified
      (e.g., strat_var_cols=['gender', 'region']), DataLoader automatically combines them
      into a single composite variable (e.g., 'gender_region') with combined categories
      (e.g., 'M_North', 'F_South'). This simplifies downstream processing while preserving
      the full cross-classification structure.
    - **Consistency Requirement**: Stratification variables must be consistent across:
      * ParticipantData.strat_var_cols
      * ContactData.strat_var_cols (if FULL mode)
      * PopulationData.strat_var_cols (if stratified population)
      * StratPropData.var_name (must match composite name if multiple vars)
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
    >>>
    >>> # Composite stratification (gender + region)
    >>> part_data = ParticipantData(
    ...     df_part,
    ...     id_col='id',
    ...     age_col='age',
    ...     strat_var_cols=['gender', 'region']  # Will be merged into 'gender_region'
    ... )
    >>> # Create composite population proportions
    >>> df_composite = pd.DataFrame({
    ...     'age': [0, 0, 0, 0],
    ...     'gender_region': ['M_North', 'M_South', 'F_North', 'F_South'],
    ...     'count': [2600, 2400, 2500, 2500]
    ... })
    >>> pop_prop = StratPropData.from_counts(
    ...     data=df_composite,
    ...     age_col='age',
    ...     strat_var_cols='gender_region',  # Will auto-detect if omitted
    ...     count_col='count'
    ... )
    >>> loader = DataLoader(part_data, cnt_data, pop_data, strat_data=pop_prop)
    """

    def __init__(
        self,
        part_data: ParticipantData,
        cnt_data: ContactData,
        pop_data: PopulationData,
        strat_data: Union[StratificationData, None] = None,
    ) -> None:

        self.part_data, self.cnt_data, self.pop_data, self.strat_data = DataValidator(
            part_data=part_data,
            cnt_data=cnt_data,
            pop_data=pop_data,
            strat_data=strat_data,
        ).validate()

        # Create CoordToColumns from dataclass structures
        col_map = self._create_col_map(self.part_data, self.cnt_data, self.pop_data)

        # Merge contact and participant data on 'id' column
        data = pd.merge(self.cnt_data.data, self.part_data.data, on="id")

        # Restore categorical dtypes that may have been lost during merge
        # This ensures stratification variables remain categorical
        for col in data.columns:
            # Check if column exists in participant data and is categorical
            if col in self.part_data.data.columns:
                if isinstance(self.part_data.data[col].dtype, pd.CategoricalDtype):
                    data[col] = pd.Categorical(
                        data[col],
                        categories=self.part_data.data[col].cat.categories,
                        ordered=self.part_data.data[col].cat.ordered,
                    )
            # Check if column exists in contact data and is categorical
            elif col in self.cnt_data.data.columns:
                if isinstance(self.cnt_data.data[col].dtype, pd.CategoricalDtype):
                    data[col] = pd.Categorical(
                        data[col],
                        categories=self.cnt_data.data[col].cat.categories,
                        ordered=self.cnt_data.data[col].cat.ordered,
                    )

        # Initialize parent class with merged data
        super().__init__(data, self.pop_data.data, col_map, self.strat_data)

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
            strat_vars_part = part_data.get_strat_var_cols(suffix=True)
        else:
            strat_vars_part = None

        if cnt_data.strat_var_cols:
            strat_vars_cnt = cnt_data.get_strat_var_cols(suffix=True)
        else:
            strat_vars_cnt = None

        # Create CoordToColumns
        col_map = CoordToColumns(
            age_part="age_part" if part_data.age_col else "age_grp_part",
            age_cnt="age_cnt" if cnt_data.age_col else None,
            age_grp_cnt="age_grp_cnt" if cnt_data.age_grp_col else None,
            id_col="id",
            y="y",
            z=part_data.grp_cnt_col,  # Use actual group contact count column name
            strat_vars_part=strat_vars_part,
            strat_vars_cnt=strat_vars_cnt,
            repeat_part="repeat_part" if part_data.repeat_col else None,
            age_pop="age",
            P="P",
            strat_vars_pop=(
                pop_data.get_strat_var_cols() if pop_data.strat_var_cols else None
            ),
        )

        return col_map
