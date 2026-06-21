from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from jax.random import PRNGKey
from numpy.typing import NDArray

from ...dataloader.containers import ContactData, ParticipantData
from ...utils import AgeGroupSpecs
from .._base import ContactModel
from ..numpyro import PremNumPyroMixin
from ._prem_loader import PremDataLoader
from ._socialmix_helpers import assign_age_groups, validate_shared_strat_vars


class Prem(PremNumPyroMixin, ContactModel):
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
        (part_age_grp) and optional stratification variables.
    cnt_data : ContactData
        Validated contact data container. Should include contact age groups
        (cnt_age_grp) and matching stratification variables.
    age_group_specs : AgeGroupSpecs
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
        Number of participant age groups.
    D : int
        Number of contact age groups.

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
    >>> from cntmosaic.utils import AgeGroupSpecs
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
    >>> age_bins = AgeGroupSpecs.from_boundaries([0, 5, 10, 15, 20, 65, 100])
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
    - Unlike GenMixFF: No hierarchical sharing across strata
    - Unlike AgeMixFF: Does not enforce reciprocity/rate consistency
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
    GenMixFF : Hierarchical model with shared priors across strata
    vdKassteele : Alternative model with van de Kassteele basis functions
    """

    def __init__(
        self,
        part_data: ParticipantData,
        cnt_data: ContactData,
        age_group_specs: AgeGroupSpecs,
        random_effects: bool = False,
        backend: Optional[Any] = None,
    ):
        super().__init__(backend=backend)

        # Store validated data containers
        self.part_data = part_data
        self.cnt_data = cnt_data
        self.age_group_specs = age_group_specs
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

        # Run processing pipeline (validation already done by containers)
        self._preprocess()
        self._load()

    # ------------------------------------------------------------------
    # Preprocessing
    # ------------------------------------------------------------------

    def _preprocess(self) -> None:
        """Extract stratification metadata and assign age groups."""
        # Extract stratification variables from both containers
        self.strat_vars_part = self.part_data.get_strat_vars()

        cnt_vars = self.cnt_data.get_strat_vars()
        self.strat_vars_cnt = (
            [var.removeprefix("cnt_") for var in cnt_vars] if cnt_vars else []
        )

        # Partition into shared vs side-only sets
        self.strat_vars_shared = sorted(
            set(self.strat_vars_part) & set(self.strat_vars_cnt)
        )
        self.strat_vars_part_only = sorted(
            set(self.strat_vars_part) - set(self.strat_vars_cnt)
        )
        self.strat_vars_cnt_only = sorted(
            set(self.strat_vars_cnt) - set(self.strat_vars_part)
        )

        if self.strat_vars_shared:
            validate_shared_strat_vars(
                self.strat_vars_shared, self.part_data, self.cnt_data
            )

        # Union of all stratification variables (backward-compatibility attribute)
        self.strat_vars = sorted(set(self.strat_vars_part) | set(self.strat_vars_cnt))

        # Stratification dimension counts
        self.strat_dims_part = {
            var: self.part_data.data[f"part_{var}"].nunique()
            for var in self.strat_vars_part
        }
        self.strat_dims_cnt = {
            var: self.cnt_data.data[f"cnt_{var}"].nunique()
            for var in self.strat_vars_cnt
        }
        # Merged dims (participant side wins for shared vars) — backward-compatibility
        self.strat_dims = {**self.strat_dims_cnt, **self.strat_dims_part}

        self.K = self._calculate_K()

        # Assign age groups if the containers hold raw ages rather than bins
        if "part_age_grp" not in self.part_data.data.columns:
            if "part_age" not in self.part_data.data.columns:
                raise ValueError(
                    "ParticipantData must have either 'part_age' or 'part_age_grp' column."
                )
            self.part_data.data = assign_age_groups(
                self.part_data.data, "part_age", self.age_group_specs, "part_age_grp"
            )

        if "cnt_age_grp" not in self.cnt_data.data.columns:
            if "cnt_age" not in self.cnt_data.data.columns:
                raise ValueError(
                    "ContactData must have either 'cnt_age' or 'cnt_age_grp' column."
                )
            self.cnt_data.data = assign_age_groups(
                self.cnt_data.data, "cnt_age", self.age_group_specs, "cnt_age_grp"
            )

    def _calculate_K(self) -> int:
        """Return the expected number of strata from the stratification configuration."""
        if not self.strat_vars_part and not self.strat_vars_cnt:
            return 1
        K = 1
        for var in self.strat_vars_part_only:
            K *= self.strat_dims_part[var]
        for var in self.strat_vars_cnt_only:
            K *= self.strat_dims_cnt[var]
        for var in self.strat_vars_shared:
            K *= self.strat_dims_part[var] ** 2
        return int(K)

    # ------------------------------------------------------------------
    # Data loading
    # ------------------------------------------------------------------

    def _load(self) -> None:
        """Build flat index arrays for the NumPyro model via :class:`PremDataLoader`."""
        PremDataLoader(self).load_data()

    # ------------------------------------------------------------------
    # SVI helpers
    # ------------------------------------------------------------------

    def get_samples_svi(
        self,
        rng_key: PRNGKey,
        num_samples: int = 2000,
    ) -> Dict[str, Any]:
        """Sample parameters directly from the variational posterior (guide).

        Parameters
        ----------
        rng_key : jax.random.PRNGKey
            Random number generator key.
        num_samples : int, default=2000
            Number of posterior samples to draw.

        Returns
        -------
        Dict[str, Any]
            Dictionary of parameter samples (beta0, beta_cd, tau, etc.).

        Raises
        ------
        AttributeError
            If SVI inference has not been run.
        """
        if self._svi_result is None:
            raise AttributeError("run_inference_svi must be run first.")

        return self._get_backend().get_svi_samples(
            rng_key, self._guide, self._svi_result, num_samples=num_samples
        )
