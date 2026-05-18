"""
Generalised mixing model with coarse-age resolution for both participant and contact ages.

This module implements the GenMixCC model, which extends AgeMixCC to support
stratified populations (e.g., by gender, setting) using hierarchical priors
with coarse age group resolution for both participants and contacts.
"""

from typing import Any, Dict, Optional

import jax.numpy as jnp
import numpy as np

from .._types import StratMode
from ..dataloader import ContactSurveyLoader
from ..utils import AgeGroupSpecs
from ._AgeMixCC import AgeMixCC
from ._math import clr
from .numpyro import GenMixCCNumPyroMixin
from .numpyro.priors import Hill, vdKassteele2D


class GenMixCC(GenMixCCNumPyroMixin, AgeMixCC):
    """
    Generalised mixing model with coarse-age resolution for both participant and contact ages.

    GenMixCC (Generalised Mixing, Coarse-Coarse) extends AgeMixCC to handle stratified
    contact data (e.g., by gender, setting, region) using hierarchical priors. Both
    participant ages and contact ages are at coarse group resolution (e.g., 5-year bands).

    The model combines:
    1. Coarse age resolution: Both participant and contact ages are recorded as groups
    2. Hierarchical structure: Models population subgroups with shared smooth patterns
    3. Rate consistency: Ensures bidirectional contact balance via population weights

    All priors must be ``vdKassteele2D``, which is the statistically appropriate choice
    for discrete coarse age groups (unlike PSpline or HSGP which assume a continuous domain).

    Mathematical Model
    ------------------
    For each observed contact y_i from participant age group c_i to contact age group d_i
    in stratum s_i:

        log(rate[c, d]) = β₀ + f(c, d)   (shared smooth baseline, B×B)
        log(δ_s[c, d]) = log(prior_s) - log(P_s[c,d])  (stratum-specific deviation)
        log(cint[c_i, d_i]) = log(rate[c_i, d_i]) + log(δ_s[c_i, d_i]) + log(P[d_i])
        μ_i = exp(log(cint[c_i, d_i]) + log(N_i) + log(S_i) + η_i)
        y_i ~ Poisson(μ_i) or NegativeBinomial2(μ_i, φ)

    where:
    - β₀: baseline contact rate
    - f(c, d): 2D age-age function (B×B, from 'rate' prior)
    - δ_s: stratum-specific multiplicative adjustments
    - P[d]: overall population proportion in coarse group d
    - P_s[c,d]: stratum-specific population proportions (for centering δ_s)
    - N_i: survey sample size
    - S_i: setting-specific offset
    - η_i: repeat interview effect
    - φ: overdispersion parameter

    Parameters
    ----------
    dataloader : ContactSurveyLoader
        ContactSurveyLoader object containing stratified contact data with columns:
        - y: observed contact counts
        - cid: participant age indices (coarse age groups)
        - did: contact age indices (coarse age groups)
        - log_N: log of survey sample sizes
        - log_P: log of population age distribution, shape (K, B)
        - log_V: log of setting-specific offsets (optional)
        - rid: repeat interview indicators (optional)
        - strat_vars: stratification variables (e.g., gender, setting)
    priors : dict[str, vdKassteele2D]
        Dictionary of prior specifications. Must contain:
        - 'rate': vdKassteele2D(prior_type='global') for baseline rates
        - One vdKassteele2D per stratification variable in strat_vars
    likelihood : str, default='negbin'
        Observation likelihood:
        - 'negbin': Negative binomial (recommended for overdispersed counts)
        - 'poisson': Poisson (assumes mean = variance)
    inv_odist : float, default=1.0
        Prior mean for inverse overdispersion (negbin only).

    Examples
    --------
    >>> from cntmosaic.dataloader import ContactSurveyLoader, StratificationData
    >>> from cntmosaic.models import GenMixCC
    >>> from cntmosaic.models.numpyro.priors import vdKassteele2D
    >>> from jax.random import PRNGKey
    >>>
    >>> priors = {
    ...     "rate": vdKassteele2D(prior_type="global"),
    ...     "sex": vdKassteele2D(prior_type="partial"),
    ... }
    >>> model = GenMixCC(dataloader, priors, likelihood="negbin")
    >>> model.run_inference_mcmc(PRNGKey(42), num_samples=1000, num_warmup=1000, num_chains=4)

    See Also
    --------
    AgeMixCC : Age-only model with coarse-coarse age resolution (no stratification)
    GenMixFF : Generalised mixing model with fine-age resolution for both ages
    GenMixFC : Generalised mixing model with fine participant / coarse contact age
    vdKassteele2D : IGMRF prior for coarse age grids
    """

    default_priors = {"rate": vdKassteele2D(prior_type="global")}

    def __init__(
        self,
        dataloader: ContactSurveyLoader,
        priors: Dict[str, vdKassteele2D],
        likelihood: str = "negbin",
        inv_odist: float = 1.0,
        backend: Optional[Any] = None,
        age_group_specs: Optional[AgeGroupSpecs] = None,
    ) -> None:
        """
        Initialize GenMixCC with hierarchical structure and coarse-coarse age resolution.

        Parameters
        ----------
        dataloader : ContactSurveyLoader
            Preprocessed stratified contact data with coarse age groups for both ages.
        priors : Dict[str, vdKassteele2D]
            Prior specifications. Must contain 'rate' (prior_type='global') and one
            vdKassteele2D prior per stratification variable.
        likelihood : str, default='negbin'
            Observation likelihood ('negbin' or 'poisson').
        inv_odist : float, default=1.0
            Prior mean for inverse overdispersion (negbin only).
        backend : InferenceBackend, optional
            Pluggable inference engine (default: NumPyroBackend).
        age_group_specs : AgeGroupSpecs, optional
            Age group specification object encoding bin boundaries. Propagated into
            ContactSummary so that plot_mosaic_pixilated can be called directly on a
            summary without a separate AgeGroupSpecs argument.
        """
        effective_priors = self.default_priors.copy()
        effective_priors.update(priors)

        # AgeMixCC.__init__ sets: cid, did, log_P (1,B), B, log_V, and optionally rid/hill.
        # It also calls prior.set_age_bounds(0, B-1) for all priors.
        super().__init__(dataloader, effective_priors, likelihood, inv_odist=inv_odist, backend=backend, age_group_specs=age_group_specs)

        # Override log_P: stratified case is (K, B) not (1, B)
        self.log_P = jnp.array(self.data.log_P)

        self._validate_hierarchical_inputs()
        self.set_prior_event_dim()
        self.set_prior_loc()

        # Restore rid/hill in case super().__init__() set them before log_P override
        if self.data.rid is not None:
            self.rid = jnp.array(self.data.rid, dtype=jnp.int32)
            self.hill = Hill(max_value=int(self.data.rid.max()))

    def _validate_hierarchical_inputs(self) -> None:
        """Validate that priors and data strat_vars are consistent."""
        if "rate" not in self.priors:
            raise ValueError("'rate' prior must be provided in the priors dictionary.")

        if self.priors["rate"].prior_type != "global":
            raise ValueError(
                f"'rate' prior_type must be 'global', got '{self.priors['rate'].prior_type}'."
            )

        data_strat_vars = set(self.data.strat_modes.keys())
        prior_strat_vars = set(var for var in self.priors.keys() if var != "rate")

        if prior_strat_vars != data_strat_vars:
            raise ValueError(
                f"Mismatch between stratification variables in priors and dataset.\n"
                f"Priors contain: {sorted(prior_strat_vars)}\n"
                f"Data strat_vars contain: {sorted(data_strat_vars)}\n"
                f"They must match exactly."
            )

        for var, mode in self.data.strat_modes.items():
            prior_type = self.priors[var].prior_type
            if mode == StratMode.PARTIAL and prior_type not in ["partial", "full"]:
                raise ValueError(
                    f"Stratification variable '{var}' is PARTIAL but prior_type is "
                    f"'{prior_type}'. Must be 'partial' or 'full'."
                )
            if mode == StratMode.FULL and prior_type != "full":
                raise ValueError(
                    f"Stratification variable '{var}' is FULL but prior_type is "
                    f"'{prior_type}'. Must be 'full'."
                )

    def set_prior_event_dim(self) -> None:
        """Configure event dimensions for each prior based on dataset structure."""
        for var, prior in self.priors.items():
            if var == "rate":
                prior.set_event_dim(1)
            else:
                if self.data.strat_modes[var] == StratMode.PARTIAL:
                    prior.set_event_dim(self.data.strat_dims[var])
                elif self.data.strat_modes[var] == StratMode.FULL:
                    prior.set_event_dim(int(np.sqrt(self.data.strat_dims[var])))
                else:
                    raise ValueError(
                        f"Unknown stratification mode for variable '{var}': "
                        f"{self.data.strat_modes[var]}"
                    )

    def set_prior_loc(self) -> None:
        """Center stratification priors on CLR of marginal population proportions."""
        for var, prior in self.priors.items():
            if var != "rate":
                prior.set_loc(clr(self.data.marginal_multipliers[var], axis=0))
