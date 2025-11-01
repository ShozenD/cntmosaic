import warnings
from typing import Optional, Dict

import pandas as pd
import numpy as np
from numpy.typing import NDArray

import jax
import jax.numpy as jnp
from jax.typing import ArrayLike
from jax import random
from jax.random import PRNGKey

import numpyro
from numpyro import distributions as dist
from numpyro.handlers import seed, trace
from ._numpyro import (
    run_inference_mcmc,
    run_inference_svi,
    posterior_predictive_mcmc,
    posterior_predictive_svi,
)

from ..utils import AgeBins
from ..distributions import IGMRF2D
from ._SocialMix import (
    InputValidator,
    AgeBinProcessor,
)  # The same validator and processor can be used here


class Prem:
    """
    Estimate age-structured social contact matrices using the Prem et al. (2017) methodology.

    Implements a Bayesian hierarchical model following the approach described in Prem et al. (2017)
    for inferring contact intensities and contact rates from participant and contact data.

    Parameters
    ----------
    df_part: pd.DataFrame
        Participant data with columns:
        - 'id': unique participant identifier
        - 'age_part': participant age (numeric), optional
        - 'age_grp_part': participant age group (pd.Interval, categorical), optional
    df_cnt: pd.DataFrame
        Contact data with columns:
        - 'id': participant identifier (links to df_part)
        - 'age_cnt': contact age (numeric), optional
        - 'age_grp_cnt': contact age group (pd.Interval, categorical), optional
        - 'y': number of contacts (numeric, >= 0)
    age_bins: AgeBins
        Age binning scheme to categorize ages into age groups.
    random_effects: bool, default=False
        Whether to include participant-level random effects in the model.

    Methods
    -------
    print_model_shape:
        Print the shapes of the model parameters.
    run_inference_mcmc:
        Run MCMC inference to estimate model parameters.
    run_inference_svi:
        Run stochastic variational inference to estimate model parameters.
    posterior_predictive_mcmc:
        Generate posterior predictive samples using MCMC results.
    posterior_predictive_svi:
        Generate posterior predictive samples using SVI results.

    References
    ----------
    Prem, K., Cook, A. R., & Jit, M. (2017).
    Projecting social contact matrices in 152 countries using contact surveys and demographic data.
    PLOS Computational Biology, 13(9), e1005697. https://doi.org/10.1371/journal.pcbi.1005697
    """

    def __init__(
        self,
        df_part: pd.DataFrame,
        df_cnt: pd.DataFrame,
        age_bins: AgeBins,
        random_effects: bool = False,
    ):
        # Store parameters
        self.df_part = df_part.copy(deep=True)
        self.df_cnt = df_cnt.copy(deep=True)
        self.age_bins = age_bins
        self.random_effects = random_effects

        # Initialize helper classes
        self.validator = InputValidator()
        self.age_processor = AgeBinProcessor(age_bins)

        # Computed attributes (initialized in pipeline)
        self.data: Optional[pd.DataFrame] = None
        self.y: Optional[np.ndarray] = None
        self.iix: Optional[NDArray[np.int64]] = None
        self.N: Optional[int] = None
        self.C: Optional[int] = None
        self.D: Optional[int] = None
        self.cix: Optional[NDArray[np.int64]] = None
        self.dix: Optional[NDArray[np.int64]] = None
        self._mcmc_result: Optional[numpyro.infer.MCMC] = None
        self._svi_result: Optional[numpyro.infer.SVI] = None
        self._guide: Optional[callable] = None

        # Run processing pipeline
        self._validate()
        self._preprocess()
        self._load()

    def _validate(self) -> None:
        """Validate input dataframes."""
        self.validator.validate_participants(self.df_part)
        self.validator.validate_contacts(self.df_cnt, set(self.df_part["id"]))

    def _preprocess(self) -> None:
        """
        Assign age groups to participants and contacts if not already provided.
        """
        # Reset indices to avoid potential issues
        self.df_part = self.df_part.reset_index(drop=True)
        self.df_cnt = self.df_cnt.reset_index(drop=True)

        # Assign age groups to participants if not present
        has_age_part = "age_part" in self.df_part.columns
        has_age_grp_part = "age_grp_part" in self.df_part.columns
        if not has_age_grp_part and has_age_part:
            self.df_part = self.age_processor.assign_age_groups(
                self.df_part, "age_part", "age_grp_part"
            )

        # Assign age groups to contacts if not present
        has_age_cnt = "age_cnt" in self.df_cnt.columns
        has_age_grp_cnt = "age_grp_cnt" in self.df_cnt.columns
        if not has_age_grp_cnt and has_age_cnt:
            self.df_cnt = self.age_processor.assign_age_groups(
                self.df_cnt, "age_cnt", "age_grp_cnt"
            )

    def _load(self):
        """Load and prepare data for modeling."""
        # Create complete contact matrix structure
        coords = {
            "id": self.df_cnt["id"].unique(),
            "age_grp_cnt": self.df_cnt["age_grp_cnt"].cat.categories,
        }

        # Create full cartesian product
        index = pd.MultiIndex.from_product(
            [list(coord) for coord in coords.values()], names=list(coords.keys())
        )

        df_cnt_full = pd.DataFrame(
            index.to_frame(index=False), columns=list(coords.keys())
        )

        # Merge with actual contact data
        df_cnt_full = pd.merge(
            df_cnt_full, self.df_cnt, on=["id", "age_grp_cnt"], how="left"
        )

        # Fille missing contacts with zeros
        df_cnt_full["y"] = df_cnt_full["y"].fillna(0).astype(int)

        # Restore categorical information
        df_cnt_full["age_grp_cnt"] = pd.Categorical(
            df_cnt_full["age_grp_cnt"],
            categories=self.df_cnt["age_grp_cnt"].cat.categories,
            ordered=True,
        )

        # Merge with participant data
        self.data = pd.merge(df_cnt_full, self.df_part, on="id", how="left")

        # Check for missing participants
        if self.data["age_grp_part"].isnull().any():
            missing_ids = self.data[self.data["age_grp_part"].isna()]["id"].unique()
            raise ValueError(f"Missing participant data for IDs: {missing_ids}. ")

        # Aggregate by age groups
        self.data = (
            self.data.groupby(["id", "age_grp_part", "age_grp_cnt"], observed=True)["y"]
            .sum()
            .reset_index()
        )

        # Create index mappings
        self.data["iix"] = pd.factorize(self.data["id"])[0]

        # Extract arrays
        self.y = np.array(self.data["y"].values)
        self.iix = np.array(self.data["iix"].values)
        self.cix = np.array(self.data["age_grp_part"].cat.codes)
        self.dix = np.array(self.data["age_grp_cnt"].cat.codes)

        # Store dimensions and data sizes
        self.N = self.data["id"].nunique()
        self.C = self.data["age_grp_cnt"].cat.categories.size
        self.D = self.data["age_grp_part"].cat.categories.size

    def model(self, y: Optional[ArrayLike] = None) -> None:
        """
        NumPyro model definition

        Parameters
        ----------
        y : ArrayLike, optional
            Observed contact counts.
        """
        # Prior on intercept with reasonable scale
        beta0 = numpyro.sample("beta0", dist.Normal(0.0, 2.5))

        # Precision parameter with informative prior
        tau = numpyro.sample("tau", dist.Gamma(2.0, 1.0))

        # 2D intrinsic Gaussian Markov random field
        beta_cd = numpyro.sample(
            "beta_cd",
            IGMRF2D(
                num_nodes=(self.D, self.C),
                order=(1, 1),
                cond_prec1=tau,
                cond_prec2=tau,
            ),
        ).reshape((self.D, self.C))

        # Log contact intensities
        log_cint = numpyro.deterministic("log_cint", beta0 + beta_cd)

        # Optional random effects
        if self.random_effects:
            mu_re = numpyro.sample("mu_re", dist.Normal(0.0, 1.0))
            tau_re = numpyro.sample("tau_re", dist.HalfNormal(1.0))

            with numpyro.plate("random_effects", self.N):
                sigma_re = numpyro.sample("sigma_re", dist.Normal(mu_re, tau_re))

            log_lambda = log_cint[self.cix, self.dix] + sigma_re[self.iix]
        else:
            log_lambda = log_cint[self.cix, self.dix]

        lambda_param = jnp.exp(log_lambda)

        with numpyro.plate("data", len(self.y)):
            numpyro.sample("obs", dist.Poisson(lambda_param), obs=y)

    def print_model_shape(self):
        """Print the shapes of the model parameters."""
        tr = trace(seed(self.model, random.PRNGKey(0))).get_trace()
        print(numpyro.util.format_shapes(tr))

    def run_inference_mcmc(
        self,
        rng_key: PRNGKey,
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
        rng_key: jax.random.PRNGKey
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
                rng_key,
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
        guide: callable,
        num_steps: int = 5_000,
        peak_lr: float = 0.01,
        **kwargs,
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
        self._guide = guide

        try:
            self._svi_result = run_inference_svi(
                prng_key,
                self.model,
                self._guide,
                num_steps=num_steps,
                peak_lr=peak_lr,
                y=self.y,
                **kwargs,
            )

        except Exception as e:
            raise RuntimeError(f"SVI inference failed: {e}")

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
