import warnings
from typing import Dict, List, Optional

import jax
import jax.numpy as jnp
import numpy as np
import numpyro
import pandas as pd
from jax import random
from jax.random import PRNGKey
from jax.typing import ArrayLike
from numpy.typing import NDArray
from numpyro import distributions as dist
from numpyro.handlers import plate, seed, trace
from numpyro.infer.autoguide import AutoNormal

from ..dataloader.containers import ContactData, ParticipantData
from ..distributions import IGMRF2D
from ..utils import AgeBins
from ._base import ContactModel
from ._numpyro import (
    get_samples_svi,
    posterior_predictive_mcmc,
    posterior_predictive_svi,
    run_inference_mcmc,
    run_inference_svi,
)


class Prem(ContactModel):
    """
    Estimate age-structured social contact matrices using the Prem et al. (2017) methodology.

    Implements a Bayesian model following the approach described in Prem et al. (2017)
    for inferring contact intensity matrices from social contact survey data. Prem uses
    completely independent priors for each stratum (e.g., gender, setting).
    The model does NOT adjust for reciprocity.

    Parameters
    ----------
    part_data : ParticipantData
        Validated participant data container. Should include age groups
        (age_grp_part) and optional stratification variables.
    cnt_data : ContactData
        Validated contact data container. Should include contact age groups
        (age_grp_cnt) and matching stratification variables.
    age_bins : AgeBins
        Age binning scheme to categorize ages into age groups.
        Used to assign age groups if raw ages are provided in the containers.
    random_effects : bool, default=False
        Whether to include participant-level random effects in the model.

    Attributes
    ----------
    strat_vars_part : List[str]
        Names of participant stratification variables.
    strat_vars_cnt : List[str]
        Names of contact stratification variables.
    strat_vars_shared : List[str]
        Stratification variables present in both participant and contact data.
    strat_vars_part_only : List[str]
        Stratification variables only in participant data.
    strat_vars_cnt_only : List[str]
        Stratification variables only in contact data.
    K : int
        Total number of strata. Calculated based on stratification mode:
        - No stratification: K=1 ("All->All")
        - Partial (participant only): K=product of participant categories
        - Full (same vars both sides): K=product of squares of categories
        - Mixed: K=product of (participant-only × shared² × contact-only)
    N : int
        Number of unique participants.
    C : int
        Number of contact age groups.
    D : int
        Number of participant age groups.

    Methods
    -------
    print_model_shape :
        Print the shapes of the model parameters.
    run_inference_mcmc :
        Run MCMC inference to estimate model parameters.
    run_inference_svi :
        Run stochastic variational inference to estimate model parameters.
    posterior_predictive_mcmc :
        Generate posterior predictive samples using MCMC results.
    posterior_predictive_svi :
        Generate posterior predictive samples using SVI results.

    Examples
    --------
    Basic usage without stratification:

    >>> from cntmosaic.dataloader.containers import ParticipantData, ContactData
    >>> from cntmosaic.utils import AgeBins
    >>> from jax.random import PRNGKey
    >>>
    >>> # Create validated data containers
    >>> part_data = ParticipantData(
    ...     df_part=df_part,
    ...     id_col='id',
    ...     age_col='age'
    ... )
    >>> cnt_data = ContactData(
    ...     df_cnt=df_cnt,
    ...     id_col='id',
    ...     age_col='age_cnt'
    ... )
    >>>
    >>> # Define age bins
    >>> age_bins = AgeBins.from_boundaries([0, 5, 10, 15, 20, 65, 100])
    >>>
    >>> # Initialize and run inference
    >>> model = Prem(part_data, cnt_data, age_bins)
    >>> model.run_inference_mcmc(PRNGKey(42), num_samples=1000)
    >>>
    >>> # Access posterior samples
    >>> samples = model._mcmc_result.get_samples()

    With stratification - Partial case (participant only):

    >>> # Stratify by participant gender only
    >>> part_data = ParticipantData(
    ...     df_part=df_part,
    ...     id_col='id',
    ...     age_col='age',
    ...     strat_var_cols='gender'  # M, F
    ... )
    >>> cnt_data = ContactData(
    ...     df_cnt=df_cnt,
    ...     id_col='id',
    ...     age_col='age_cnt'
    ...     # No stratification for contacts
    ... )
    >>>
    >>> # Model will fit 2 matrices: "M->All", "F->All"
    >>> model = Prem(part_data, cnt_data, age_bins)
    >>> print(f"Number of strata: {model.K}")  # 2

    With stratification - Full case (same vars both sides):

    >>> # Stratify by gender for both participants and contacts
    >>> part_data = ParticipantData(
    ...     df_part=df_part,
    ...     id_col='id',
    ...     age_col='age',
    ...     strat_var_cols='gender'
    ... )
    >>> cnt_data = ContactData(
    ...     df_cnt=df_cnt,
    ...     id_col='id',
    ...     age_col='age_cnt',
    ...     strat_var_cols='gender'
    ... )
    >>>
    >>> # Model will fit 4 matrices: "M->M", "M->F", "F->M", "F->F"
    >>> model = Prem(part_data, cnt_data, age_bins)
    >>> print(f"Number of strata: {model.K}")  # 4 (2²)
    >>>
    >>> # Each stratum gets independent beta0, tau, and beta_cd parameters
    >>> model.run_inference_mcmc(PRNGKey(42), num_samples=1000)

    Notes
    -----
    **Stratification Behavior**:
    - Each stratum receives completely independent priors (no hierarchical structure)
    - Stratification variables do NOT need to match between participants and contacts
    - Four stratification modes:
      1. No stratification: K=1, stratum="All->All"
      2. Partial (participant only): K=product of categories, e.g. "M->All", "F->All"
      3. Full (same vars both sides): K=product of squares, e.g. "M->M", "M->F", "F->M", "F->F"
      4. Mixed (some overlap): K combines partial and full modes
    - Multiple variables combined with underscore: "M_Urban->F_Rural"
    - Stratum names follow "participant->contact" format

    **Differences from Other Models**:
    - Unlike HiBRCfine: No hierarchical sharing across strata
    - Unlike BRCfine: Does not enforce reciprocity/rate consistency
    - Unlike vdKassteele: Uses IGMRF2D priors instead of van de Kassteele basis

    **Migration from Old API**:
    >>> # OLD (deprecated):
    >>> # model = Prem(df_part, df_cnt, age_bins)
    >>>
    >>> # NEW (current):
    >>> part_data = ParticipantData(df_part, id_col='id', age_col='age')
    >>> cnt_data = ContactData(df_cnt, id_col='id', age_col='age_cnt')
    >>> model = Prem(part_data, cnt_data, age_bins)

    References
    ----------
    Prem, K., Cook, A. R., & Jit, M. (2017).
    Projecting social contact matrices in 152 countries using contact surveys and demographic data.
    PLOS Computational Biology, 13(9), e1005697. https://doi.org/10.1371/journal.pcbi.1005697

    See Also
    --------
    ParticipantData : Validated participant data container
    ContactData : Validated contact data container
    HiBRCfine : Hierarchical model with shared priors across strata
    vdKassteele : Alternative model with van de Kassteele basis functions
    """

    def __init__(
        self,
        part_data: ParticipantData,
        cnt_data: ContactData,
        age_bins: AgeBins,
        random_effects: bool = False,
    ):
        # Store validated data containers
        self.part_data = part_data
        self.cnt_data = cnt_data
        self.age_bins = age_bins
        self.random_effects = random_effects

        # Stratification attributes (initialized in _preprocess)
        self.strat_vars_part: List[str] = []
        self.strat_vars_cnt: List[str] = []
        self.strat_vars_shared: List[str] = []
        self.strat_vars_part_only: List[str] = []
        self.strat_vars_cnt_only: List[str] = []
        self.strat_dims_part: Dict[str, int] = {}
        self.strat_dims_cnt: Dict[str, int] = {}
        self.K: int = 1  # Total number of strata

        # Computed attributes (initialized in pipeline)
        self.data: Optional[pd.DataFrame] = None
        self.y: Optional[np.ndarray] = None
        self.iix: Optional[NDArray[np.int64]] = None
        self.six: Optional[NDArray[np.int64]] = None  # Stratum indices
        self.N: Optional[int] = None
        self.C: Optional[int] = None
        self.D: Optional[int] = None
        self.cix: Optional[NDArray[np.int64]] = None
        self.dix: Optional[NDArray[np.int64]] = None
        self._mcmc_result: Optional[numpyro.infer.MCMC] = None
        self._svi_result: Optional[numpyro.infer.SVI] = None
        self._guide: Optional[callable] = None

        # Run processing pipeline (validation already done by containers)
        self._preprocess()
        self._load()

    def _preprocess(self) -> None:
        """
        Extract and validate stratification information from data containers.

        This method:
        1. Extracts stratification variables from both containers
        2. Identifies shared, participant-only, and contact-only variables
        3. Computes stratification dimensions and expected number of strata
        4. Assigns age groups to raw ages if needed

        Notes
        -----
        Stratification variables do NOT need to match between participants and contacts.
        Four modes are supported:
        - No stratification: Both empty → K=1
        - Partial: Only participant vars → K=product of participant categories
        - Full: Same vars on both sides → K=product of squares
        - Mixed: Some overlap → K=complex product
        """
        # Extract stratification variables from both containers
        self.strat_vars_part = self.part_data.get_strat_vars()

        cnt_vars = self.cnt_data.get_strat_vars()
        if cnt_vars:
            self.strat_vars_cnt = [var.removesuffix("_cnt") for var in cnt_vars]
        else:
            self.strat_vars_cnt = []

        # Identify shared and unique variables
        self.strat_vars_shared = sorted(
            list(set(self.strat_vars_part) & set(self.strat_vars_cnt))
        )
        self.strat_vars_part_only = sorted(
            list(set(self.strat_vars_part) - set(self.strat_vars_cnt))
        )
        self.strat_vars_cnt_only = sorted(
            list(set(self.strat_vars_cnt) - set(self.strat_vars_part))
        )

        # Validate shared stratification variables
        # For shared variables, categories and encoding order MUST match
        if self.strat_vars_shared:
            self._validate_shared_strat_vars()

        # For backward compatibility, maintain strat_vars attribute
        # (union of all stratification variables)
        self.strat_vars = sorted(
            list(set(self.strat_vars_part) | set(self.strat_vars_cnt))
        )

        # Calculate stratification dimensions for participant variables
        self.strat_dims_part = {}
        if self.strat_vars_part:
            for var in self.strat_vars_part:
                col_name = f"{var}_part"
                self.strat_dims_part[var] = self.part_data.data[col_name].nunique()

        # Calculate stratification dimensions for contact variables
        self.strat_dims_cnt = {}
        if self.strat_vars_cnt:
            for var in self.strat_vars_cnt:
                col_name = f"{var}_cnt"
                self.strat_dims_cnt[var] = self.cnt_data.data[col_name].nunique()

        # For backward compatibility, maintain strat_dims attribute
        # (merge of both sides, preferring participant dimensions for shared vars)
        self.strat_dims = {**self.strat_dims_cnt, **self.strat_dims_part}

        # Calculate expected number of strata
        self.K = self._calculate_expected_K()

        # Assign age groups if needed (containers may have raw ages)
        self._assign_age_groups()

    def _validate_shared_strat_vars(self) -> None:
        """
        Validate and align shared stratification variables.

        For variables that appear in both participant and contact data, this ensures:
        1. Both sides have the same unique categories
        2. Categories are encoded in the same order (using participant order as reference)

        If categories match but order differs, contact data is automatically reordered
        to match participant data encoding.

        Raises
        ------
        ValueError
            If categories don't match (different sets of values).
        """
        for var in self.strat_vars_shared:
            # Get column names
            col_part = f"{var}_part"
            col_cnt = f"{var}_cnt"

            # Get categories from both sides
            part_col = self.part_data.data[col_part]
            cnt_col = self.cnt_data.data[col_cnt]

            # Convert to categorical if not already
            if not hasattr(part_col, "cat"):
                part_col = part_col.astype("category")
                self.part_data.data[col_part] = part_col
            if not hasattr(cnt_col, "cat"):
                cnt_col = cnt_col.astype("category")
                self.cnt_data.data[col_cnt] = cnt_col

            # Get categories in encoding order
            part_cats = list(part_col.cat.categories)
            cnt_cats = list(cnt_col.cat.categories)

            # Check if categories match as sets
            part_set = set(part_cats)
            cnt_set = set(cnt_cats)

            if part_set != cnt_set:
                # Different categories - this is an error
                only_part = part_set - cnt_set
                only_cnt = cnt_set - part_set
                raise ValueError(
                    f"Shared stratification variable '{var}' has different categories:\n"
                    f"  Participant side: {part_cats}\n"
                    f"  Contact side: {cnt_cats}\n"
                    f"  Only in participants: {sorted(only_part) if only_part else 'None'}\n"
                    f"  Only in contacts: {sorted(only_cnt) if only_cnt else 'None'}\n"
                    f"For shared variables, both sides must have identical categories."
                )

            # Same categories but possibly different order
            # Use participant ordering as reference and reorder contact data
            if part_cats != cnt_cats:
                self.cnt_data.data[col_cnt] = self.cnt_data.data[
                    col_cnt
                ].cat.reorder_categories(part_cats, ordered=False)

    def _calculate_expected_K(self) -> int:
        """
        Calculate expected number of strata based on stratification mode.

        Returns
        -------
        int
            Expected number of unique strata:
            - Case 1 (no stratification): 1
            - Case 2 (partial): product of participant categories
            - Case 3 (mixed): product of (part-only x shared^2 x cnt-only)
            - Case 4 (full): product of squares of categories
        """
        if not self.strat_vars_part and not self.strat_vars_cnt:
            # Case 1: No stratification
            return 1

        K = 1

        # Participant-only variables (partial mode)
        for var in self.strat_vars_part_only:
            K *= self.strat_dims_part[var]

        # Contact-only variables
        for var in self.strat_vars_cnt_only:
            K *= self.strat_dims_cnt[var]

        # Shared variables (full mode for these variables)
        for var in self.strat_vars_shared:
            # For shared vars, we get all combinations: n_categories × n_categories
            K *= self.strat_dims_part[var] ** 2

        return int(K)

    def _assign_age_groups(self) -> None:
        """
        Assign age groups to participants and contacts if not already provided.

        Uses the age_bins provided during initialization to categorize
        raw ages into age groups.
        """
        # Construct bin edges from AgeBins left and right boundaries
        # age_bins.left gives [0, 5, 10, ...], age_bins.right gives [4, 9, 14, ..., max+1]
        # For pd.cut, we need the full edge sequence
        bin_edges = self.age_bins.left + [self.age_bins.right[-1]]

        # Create interval labels
        intervals = [
            pd.Interval(left=l, right=r, closed="left")
            for l, r in zip(self.age_bins.left, self.age_bins.right)
        ]

        # Assign age groups to participants if not present
        if "age_grp_part" not in self.part_data.data.columns:
            if "age_part" in self.part_data.data.columns:
                # Create age groups from raw ages
                ages = self.part_data.data["age_part"]
                age_grps = pd.cut(
                    ages,
                    bins=bin_edges,
                    right=False,
                    labels=intervals,
                )
                self.part_data.data["age_grp_part"] = age_grps
            else:
                raise ValueError(
                    "ParticipantData must have either 'age_part' or 'age_grp_part' column."
                )

        # Assign age groups to contacts if not present
        if "age_grp_cnt" not in self.cnt_data.data.columns:
            if "age_cnt" in self.cnt_data.data.columns:
                # Create age groups from raw ages
                ages = self.cnt_data.data["age_cnt"]
                age_grps = pd.cut(
                    ages,
                    bins=bin_edges,
                    right=False,
                    labels=intervals,
                )
                self.cnt_data.data["age_grp_cnt"] = age_grps
            else:
                raise ValueError(
                    "ContactData must have either 'age_cnt' or 'age_grp_cnt' column."
                )

    def _create_composite_stratum(
        self, df: pd.DataFrame, part_cols: List[str], cnt_cols: List[str]
    ) -> pd.Series:
        """
        Create composite stratification variable following "participant->contact" naming.

        Combines participant and contact stratification variables into a single
        identifier using the -> separator. Examples:
        - No stratification: "All->All"
        - Partial (participant only): "M_Urban->All"
        - Full (same vars): "M_Urban->M_Urban", "M_Urban->F_Rural"
        - Mixed: "M->Work", "F->Home"

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame containing stratification columns.
        part_cols : List[str]
            Participant stratification column names (e.g., ['gender_part', 'region_part']).
        cnt_cols : List[str]
            Contact stratification column names (e.g., ['gender_cnt', 'setting_cnt']).

        Returns
        -------
        pd.Series
            Composite stratification identifier for each row in "part->cnt" format.
        """
        # Build participant side
        if not part_cols:
            part_str = pd.Series(["All"] * len(df), index=df.index)
        else:
            # Combine multiple participant strat vars with underscore
            part_strs = []
            for col in part_cols:
                if col in df.columns:
                    part_strs.append(df[col].astype(str))

            if part_strs:
                part_str = part_strs[0]
                for s in part_strs[1:]:
                    part_str = part_str + "_" + s
            else:
                part_str = pd.Series(["All"] * len(df), index=df.index)

        # Build contact side
        if not cnt_cols:
            cnt_str = pd.Series(["All"] * len(df), index=df.index)
        else:
            # Combine multiple contact strat vars with underscore
            cnt_strs = []
            for col in cnt_cols:
                if col in df.columns:
                    cnt_strs.append(df[col].astype(str))

            if cnt_strs:
                cnt_str = cnt_strs[0]
                for s in cnt_strs[1:]:
                    cnt_str = cnt_str + "_" + s
            else:
                cnt_str = pd.Series(["All"] * len(df), index=df.index)

        # Combine with -> separator
        return part_str + "->" + cnt_str

    def _load(self) -> None:
        """
        Load and prepare data for modeling, including stratification indices.

        Creates a full cartesian product of (participant ID, contact age group),
        merges with actual contact data, and prepares indices for NumPyro model.
        For stratified models, also creates composite stratification variable and
        corresponding stratum indices.
        """
        # Access preprocessed data from containers
        df_part = self.part_data.data
        df_cnt = self.cnt_data.data

        # Ensure age_grp_cnt is categorical
        if not isinstance(df_cnt["age_grp_cnt"].dtype, pd.CategoricalDtype):
            df_cnt["age_grp_cnt"] = pd.Categorical(df_cnt["age_grp_cnt"], ordered=True)

        # Create complete contact matrix structure
        coords = {
            "id": df_cnt["id"].unique(),
            "age_grp_cnt": df_cnt["age_grp_cnt"].cat.categories,
        }

        # For stratified models, include contact stratification variables in Cartesian product
        # This ensures zero records for all (participant, contact_stratum, contact_age) combinations
        if self.strat_vars_cnt:
            for var in self.strat_vars_cnt:
                col_name = f"{var}_cnt"
                if col_name in df_cnt.columns:
                    # Ensure categorical
                    if not isinstance(df_cnt[col_name].dtype, pd.CategoricalDtype):
                        df_cnt[col_name] = pd.Categorical(df_cnt[col_name])
                    coords[col_name] = df_cnt[col_name].cat.categories

        # Create full cartesian product
        index = pd.MultiIndex.from_product(
            [list(coord) for coord in coords.values()], names=list(coords.keys())
        )

        df_cnt_full = pd.DataFrame(
            index.to_frame(index=False), columns=list(coords.keys())
        )

        # Determine merge keys (id, age_grp_cnt, and any contact strat vars)
        merge_keys = ["id", "age_grp_cnt"]
        if self.strat_vars_cnt:
            merge_keys.extend([f"{var}_cnt" for var in self.strat_vars_cnt])

        # Merge with actual contact data
        df_cnt_full = pd.merge(df_cnt_full, df_cnt, on=merge_keys, how="left")

        # Fill missing contacts with zeros
        df_cnt_full["y"] = df_cnt_full["y"].fillna(0).astype(int)

        # Restore categorical information for age groups
        df_cnt_full["age_grp_cnt"] = pd.Categorical(
            df_cnt_full["age_grp_cnt"],
            categories=df_cnt["age_grp_cnt"].cat.categories,
            ordered=True,
        )

        # Restore categorical information for contact stratification variables
        if self.strat_vars_cnt:
            for var in self.strat_vars_cnt:
                col_name = f"{var}_cnt"
                if col_name in df_cnt.columns and col_name in df_cnt_full.columns:
                    df_cnt_full[col_name] = pd.Categorical(
                        df_cnt_full[col_name],
                        categories=df_cnt[col_name].cat.categories,
                        ordered=getattr(df_cnt[col_name].cat, "ordered", False),
                    )

        # Merge with participant data
        self.data = pd.merge(df_cnt_full, df_part, on="id", how="left")

        # Check for missing participants
        if self.data["age_grp_part"].isnull().any():
            missing_ids = self.data[self.data["age_grp_part"].isna()]["id"].unique()
            raise ValueError(f"Missing participant data for IDs: {missing_ids}.")

        # Ensure age_grp_part is categorical
        if not isinstance(self.data["age_grp_part"].dtype, pd.CategoricalDtype):
            self.data["age_grp_part"] = pd.Categorical(
                self.data["age_grp_part"], ordered=True
            )

        # Build groupby columns based on stratification
        groupby_cols = ["id", "age_grp_part", "age_grp_cnt"]

        # Add stratification columns if present
        if self.strat_vars:
            strat_cols_part = [f"{v}_part" for v in self.strat_vars]
            strat_cols_cnt = [f"{v}_cnt" for v in self.strat_vars]
            # Only add columns that exist in the merged data
            for col in strat_cols_part + strat_cols_cnt:
                if col in self.data.columns and col not in groupby_cols:
                    groupby_cols.append(col)

        # Aggregate by age groups (and stratification if present)
        self.data = (
            self.data.groupby(groupby_cols, observed=False)["y"].sum().reset_index()
        )

        # Create stratification encoding
        if self.strat_vars_part or self.strat_vars_cnt:
            strat_cols_part = [f"{v}_part" for v in self.strat_vars_part]
            strat_cols_cnt = [f"{v}_cnt" for v in self.strat_vars_cnt]

            # Create composite stratification variable
            self.data["stratum"] = self._create_composite_stratum(
                self.data, strat_cols_part, strat_cols_cnt
            )

            # Encode stratum to integer index using categorical ordering (alphabetical)
            # This ensures deterministic encoding independent of data order
            self.data["stratum"] = pd.Categorical(self.data["stratum"], ordered=True)
            self.six = np.array(self.data["stratum"].cat.codes, dtype=np.int32)

            # Update K to actual number of unique strata (may differ from product if some combinations don't exist)
            self.K = int(self.data["stratum"].nunique())
        else:
            # No stratification - all observations in stratum 0
            self.six = np.zeros(len(self.data), dtype=np.int32)

        # Create index mappings
        # Convert ID to categorical for consistent encoding
        self.data["id_cat"] = pd.Categorical(self.data["id"])
        self.data["iix"] = self.data["id_cat"].cat.codes

        # Extract arrays
        self.y = np.array(self.data["y"].values)
        self.iix = np.array(self.data["iix"].values, dtype=np.int32)
        self.cix = np.array(self.data["age_grp_part"].cat.codes, dtype=np.int32)
        self.dix = np.array(self.data["age_grp_cnt"].cat.codes, dtype=np.int32)

        # Store dimensions and data sizes
        self.N = self.data["id"].nunique()
        self.C = self.data["age_grp_part"].cat.categories.size
        self.D = self.data["age_grp_cnt"].cat.categories.size

    def model(self, y: Optional[ArrayLike] = None) -> None:
        """
        NumPyro model definition with stratification support.

        For unstratified models (K=1), samples a single set of parameters.
        For stratified models (K>1), samples independent parameters for each
        stratum using plates, with no hierarchical sharing.

        Parameters
        ----------
        y : ArrayLike, optional
            Observed contact counts. If None, samples from the prior.

        Notes
        -----
        **Unstratified Model (K=1)**:
        - Single beta0 (intercept)
        - Single tau (precision)
        - Single beta_cd (2D IGMRF field)

        **Stratified Model (K>1)**:
        - Independent beta0 for each stratum
        - Independent tau for each stratum
        - Independent beta_cd (2D IGMRF) for each stratum
        - No hierarchical priors (unlike HiBRCfine)

        The model does NOT adjust for reciprocity or population weighting.
        """
        if self.K == 1:
            # Unstratified model - original Prem formulation
            # Prior on intercept with reasonable scale
            beta0 = numpyro.sample("beta0", dist.Normal(0.0, 2.5))

            # Precision parameter with informative prior
            tau = numpyro.sample("tau", dist.Gamma(2.0, 1.0))

            # 2D intrinsic Gaussian Markov random field
            beta_cd = numpyro.sample(
                "beta_cd",
                IGMRF2D(
                    num_nodes=(self.C, self.D),
                    order=(1, 1),
                    cond_prec1=tau,
                    cond_prec2=tau,
                ),
            ).reshape((self.C, self.D))

            # Log contact intensities
            log_cint = numpyro.deterministic("log_cint", beta0 + beta_cd)

            # Optional random effects
            if self.random_effects:
                mu_re = numpyro.sample("mu_re", dist.Normal(0.0, 1.0))
                tau_re = numpyro.sample("tau_re", dist.HalfNormal(1.0))

                with plate("random_effects", self.N):
                    sigma_re = numpyro.sample("sigma_re", dist.Normal(mu_re, tau_re))

                log_lambda = log_cint[self.cix, self.dix] + sigma_re[self.iix]
            else:
                log_lambda = log_cint[self.cix, self.dix]

        else:
            # Stratified model - independent priors per stratum
            # Sample independent intercepts for each stratum
            with plate("strata", self.K):
                beta0 = numpyro.sample("beta0", dist.Normal(0.0, 2.5))
                tau = numpyro.sample("tau", dist.Gamma(2.0, 1.0))

            # Sample independent 2D IGMRF fields for each stratum
            # Use expand to create K independent fields
            beta_cd = numpyro.sample(
                "beta_cd",
                IGMRF2D(
                    num_nodes=(self.C, self.D),
                    order=(1, 1),
                    cond_prec1=tau,
                    cond_prec2=tau,
                )
                .expand([self.K])
                .to_event(1),
            ).reshape((self.K, self.C, self.D))

            # Compute log contact intensities for each stratum
            # Reshape beta0 from (K,) to (K, 1, 1) for broadcasting with beta_cd (K, D, C)
            log_cint = numpyro.deterministic(
                "log_cint", beta0[:, jnp.newaxis, jnp.newaxis] + beta_cd
            )

            # Optional stratified random effects
            if self.random_effects:
                # Independent random effect parameters per stratum
                with plate("strata_re", self.K):
                    mu_re = numpyro.sample("mu_re", dist.Normal(0.0, 1.0))
                    tau_re = numpyro.sample("tau_re", dist.HalfNormal(1.0))

                # Participant-level random effects (within stratum)
                with plate("random_effects", self.N):
                    # Use stratum-specific hyperparameters
                    sigma_re = numpyro.sample(
                        "sigma_re",
                        dist.Normal(
                            mu_re[self.six[self.iix]], tau_re[self.six[self.iix]]
                        ),
                    )

                log_lambda = log_cint[self.six, self.cix, self.dix] + sigma_re[self.iix]
            else:
                log_lambda = log_cint[self.six, self.cix, self.dix]

        # Likelihood (same for both stratified and unstratified)
        lambda_param = jnp.exp(log_lambda)

        with plate("data", len(self.y)):
            numpyro.sample("obs", dist.Poisson(lambda_param), obs=y)

    def print_model_shape(self):
        """Print the shapes of the model parameters."""
        tr = trace(seed(self.model, random.PRNGKey(0))).get_trace()
        print(numpyro.util.format_shapes(tr))

    def run_inference_mcmc(
        self,
        prng_key: PRNGKey,
        num_samples: int = 500,
        num_warmup: int = 500,
        num_chains: int = 2,
        target_accept_prob: float = 0.8,
        max_tree_depth: int = 10,
        **kwargs,
    ) -> None:
        """Run full Bayesian inference using Hamiltonian Monte Carlo and NUT Sampler.

        Parameters
        ----------
        prng_key: jax.random.PRNGKey
            Random number generator key.
        num_samples: int, default=1000
            Number of samples to draw from the posterior.
        num_warmup: int, default=1000
            Number of warmup steps.
        num_chains: int, default=1
            Number of chains to run.
        target_accept_prob: float, default=0.8
            Target acceptance probability for NUTS.
        max_tree_depth: int, default=10
            Maximum tree depth for NUTS.
        **kwargs
            Additional keyword arguments to pass to the MCMC
        """
        try:
            self._mcmc_result = run_inference_mcmc(
                prng_key,
                self.model,
                num_samples=num_samples,
                num_warmup=num_warmup,
                num_chains=num_chains,
                target_accept_prob=target_accept_prob,
                max_tree_depth=max_tree_depth,
                y=self.y,
                **kwargs,
            )

            # Log diagnostics
            self._log_mcmc_diagnostics()

        except Exception as e:
            raise RuntimeError(f"MCMC inference failed: {e}")

    def _log_mcmc_diagnostics(self) -> None:
        """Log MCMC diagnostics information."""
        if self._mcmc_result is None:
            return

        try:
            extra_fields = self._mcmc_result.get_extra_fields()
            n_divergent = sum(extra_fields["diverging", 0])
            print(f"Number of divergent transitions: {n_divergent}")

            if n_divergent > 0:
                warnings.warn(
                    f"Found {n_divergent} divergent transitions. "
                    "Consider increasing target_accept_prob or max_tree_depth."
                )

        except Exception as e:
            warnings.warn(f"Failed to compute MCMC diagnostics: {e}")

    def run_inference_svi(
        self,
        prng_key: PRNGKey,
        guide: callable = None,
        num_steps: int = 5_000,
        peak_lr: float = 0.01,
    ) -> None:
        """
        Run stochastic variational inference.

        Parameters
        ----------
        prng_key : jax.random.PRNGKey
            Random number generator key.
        guide: callable
            Variational guide function.
        num_steps: int, default=5_000
            Number of optimization steps.
        peak_lr: float, default=0.01
            Peak learning rate.
        **model_kwargs
            Additional keyword arguments to pass to the SVI
        """
        if guide is None:
            # By default, use AutoNormal (mean-field) guide
            guide = AutoNormal(self.model)

        self._guide = guide

        try:
            self._svi_result = run_inference_svi(
                prng_key,
                self.model,
                self._guide,
                num_steps=num_steps,
                peak_lr=peak_lr,
                y=self.y,
            )

        except Exception as e:
            raise RuntimeError(f"SVI inference failed: {e}")

    def get_samples_svi(
        self,
        rng_key: PRNGKey,
        num_samples: int = 2000,
    ) -> Dict[str, jnp.ndarray]:
        """
        Sample parameters from the variational posterior (guide).

        This is the SVI equivalent of MCMC.get_samples() - it returns samples
        of the model parameters (e.g., beta0, beta_cd, tau) from the learned
        variational distribution.

        Parameters
        ----------
        rng_key : jax.random.PRNGKey
            Random number generator key.
        num_samples : int, default=2000
            Number of posterior samples to draw.

        Returns
        -------
        Dict[str, jax.Array]
            Dictionary of parameter samples (beta0, beta_cd, tau, etc.).
            Auto-guide internal variables are filtered out.

        Raises
        ------
        AttributeError
            If SVI inference has not been run.
        """
        if self._svi_result is None:
            raise AttributeError("run_inference_svi must be run first.")

        return get_samples_svi(
            rng_key,
            self._guide,
            self._svi_result.params,
            num_samples=num_samples,
        )

    def posterior_predictive_svi(
        self,
        rng_key: PRNGKey,
        num_samples: int = 5_000,
    ) -> Dict[str, jnp.ndarray]:
        """
        Generate posterior predictive samples using SVI results.

        Parameters
        ----------
        rng_key : jax.random.PRNGKey
            Random number generator key.
        num_samples: int, default=2000
            Number of samples to draw.

        Returns
        -------
        Dict[str, jax.Array]
            Posterior predictive samples.

        Raises
        ------
        AttributeError
            If SVI inference has not been run.

        **model_kwargs
            Additional keyword arguments to pass to the Predictive
        """
        if self._svi_result is None:
            raise AttributeError("run_inference_svi must be run first.")

        return posterior_predictive_svi(
            rng_key,
            self.model,
            self._guide,
            self._svi_result.params,
            num_samples=num_samples,
        )

    def posterior_predictive_mcmc(
        self,
        rng_key: PRNGKey,
        num_samples: int = 1000,
    ) -> Dict[str, jax.Array]:
        """Generate posterior predictive samples using MCMC.

        Parameters
        ----------
        rng_key : jax.random.PRNGKey
            Random number generator key.
        num_samples: int, default=1000
            Number of samples to generate.

        Returns
        -------
        dict[str, jax.Array]
            Posterior predictive samples.

        Raises
        ------
        AttributeError
            If MCMC inference has not been run.

        **model_kwargs
            Additional keyword arguments to pass to the Predictive
        """
        if self._mcmc_result is None:
            raise AttributeError("run_inference_mcmc must be run first.")

        return posterior_predictive_mcmc(
            rng_key,
            self.model,
            self._mcmc_result.get_samples(),
            num_samples=num_samples,
        )
