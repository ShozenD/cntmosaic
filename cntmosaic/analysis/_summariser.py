import warnings

import numpy as np
from numpy.typing import NDArray
from typing import Optional, Tuple, Dict, Literal

import jax
from ..models import SocialMix, BRCfine, BRCrefine, HiBRCfine, HiBRCrefine, Prem
from ..utils import pixilate, depixilate, AgeBins


class ModelSummariserSVI:
    def __init__(self, model):
        self.model = model
        self.prng_key = jax.random.PRNGKey(0)
        self.get_post_predictive()
        self.get_post_predictive_cint()

    def get_post_predictive(self):
        """
        Get the posterior predictive distribution of the model.
        This is a wrapper around the model's posterior_predictive_svi method.
        It uses the model's guide to sample from the posterior predictive distribution.
        The guide is a variational approximation to the posterior distribution.
        """
        self.post_pred = self.model.posterior_predictive_svi(
            self.prng_key, self.model.guide
        )

    def get_post_predictive_cint(self):
        if isinstance(self.model, (HiBRCfine, HiBRCrefine)):
            # For HiBRC models, the contact intensity needs to be computed
            log_rate = self.post_pred["log_rate"].astype(
                np.float32
            )  # Convert early to save memory
            log_P = self.model.log_P.astype(np.float32)  # Convert early to save memory
            post_pred_cint = {}

            for name, site in self.post_pred.items():
                if "log_delta" in name:
                    var = name.split("/")[0]
                    cat = self.model.ds.attrs["grp_vars"][var]
                    site = site.astype(np.float32)  # Convert early to save memory

                    # Initialize dict for this variable
                    post_pred_cint[var] = {}

                    # Process each category separately to avoid memory explosion
                    for i, c in enumerate(cat):
                        # Compute contact intensity for this category only
                        # Add dimensions efficiently without creating large intermediate arrays
                        log_rate_expanded = log_rate[
                            :, np.newaxis, :, :
                        ]  # shape: (n_samples, 1, A, A)
                        site_cat = site[
                            :, i : i + 1, :, :
                        ]  # shape: (n_samples, 1, A, A) - slice to keep dims
                        log_P_expanded = log_P[
                            np.newaxis, np.newaxis, :, :
                        ]  # shape: (1, 1, A, A)

                        # Compute log sum and exp in one operation to minimize memory
                        log_sum = log_rate_expanded + site_cat + log_P_expanded

                        # Use np.exp with out parameter to avoid creating intermediate arrays
                        cint = np.exp(log_sum, dtype=np.float32).squeeze(
                            axis=1
                        )  # Remove singleton dimension
                        post_pred_cint[var][c] = cint

                        # Clean up intermediate arrays to free memory immediately
                        del log_rate_expanded, site_cat, log_sum, cint

            self.post_pred_cint = post_pred_cint
        elif isinstance(self.model, (BRCfine, BRCrefine)):
            pass

    def summarise_rate(self, probs: tuple = (0.025, 0.5, 0.975)):
        """
        Summarise the rate parameter of the model.
        This is a wrapper around the model's summarise_rate method.
        It uses the model's posterior predictive distribution to compute the summary statistics.
        """
        if "sum_rate" not in self.__dict__:
            self.sum_rate = np.quantile(
                np.exp(self.post_pred["log_rate"]), probs, axis=0
            )

        return self.sum_rate

    def summarise_cint(self, probs: tuple = (0.025, 0.5, 0.975)):
        """
        Summarise the contact intensity matrix of the model.
        It uses the model's posterior predictive distribution to compute the summary statistics.
        """
        if "sum_cint" not in self.__dict__:
            if type(self.model) in (BRCfine, BRCrefine):
                # For BRC models, the contact intensity is stored in 'log_cint'
                self.sum_cint = np.quantile(self.post_pred["log_cint"], probs, axis=0)
                self.sum_cint = np.exp(self.sum_cint)

            elif type(self.model) in (HiBRCfine, HiBRCrefine):
                self.sum_cint = {
                    var: {
                        name: np.quantile(value, probs, axis=0)
                        for name, value in cat.items()
                    }
                    for var, cat in self.post_pred_cint.items()
                }

        return self.sum_cint

    def summarise_mcint(self, probs: tuple = (0.025, 0.5, 0.975)):
        """
        Summarise the marginal contact intensity of the model.
        It uses the model's posterior predictive distribution to compute the summary statistics.
        """
        if "sum_mcint" not in self.__dict__:
            if type(self.model) in (BRCfine, BRCrefine):
                mcint = np.exp(self.post_pred["log_cint"]).sum(axis=2)
                self.sum_mcint = np.quantile(mcint, probs, axis=0)

            elif type(self.model) in (HiBRCfine, HiBRCrefine):
                self.sum_mcint = {
                    var: {
                        name: np.quantile(value.sum(axis=2), probs, axis=0)
                        for name, value in cat.items()
                    }
                    for var, cat in self.post_pred_cint.items()
                }

        return self.sum_mcint


class ModelSummariserSocialMix:
    """
    Statistical summariser for SocialMix bootstrap results.

    Computes quantiles and confidence intervals for contact matrices
    from bootstrap samples, with proper handling of adaptive age binning.

    Parameters
    ----------
    sm : SocialMix
        Fitted SocialMix model with bootstrap results

    Attributes
    ----------
    sm : SocialMix
        Reference to the SocialMix model
    age_bins : AgeBins
        Original age bins before any adaptive merging
    effective_age_bins : AgeBins
        Age bins after adaptive merging (may equal age_bins)
    age_dist : NDArray
        Population age distribution

    Examples
    --------
    >>> sm = SocialMix(df_part, df_cnt, df_age_dist, age_bins)
    >>> sm.run_bootstrap(n_boot=1000, random_state=42)
    >>>
    >>> summariser = ModelSummariserSocialMix(sm)
    >>>
    >>> # Get 95% confidence intervals for contact intensity
    >>> summary = summariser.summarise_cint(alpha=0.05)
    >>> lower, median, upper = summary['lower'], summary['median'], summary['upper']
    >>>
    >>> # Get different confidence level
    >>> summary_90 = summariser.summarise_rate(alpha=0.10)
    """

    def __init__(self, sm: SocialMix):
        """
        Initialize summariser with a fitted SocialMix model.

        Parameters
        ----------
        sm : SocialMix
            SocialMix model that has been fitted and bootstrapped

        Raises
        ------
        ValueError
            If model has not been fitted or bootstrapped
        """
        # Validate model has been fitted
        if sm.Y is None:
            raise ValueError("SocialMix model has not been fitted")

        # Store reference to model
        self.sm = sm

        # Reference key attributes (no copies)
        self.age_bins = sm.age_bins
        self.effective_age_bins = sm.effective_age_bins
        self.age_dist = sm.df_age_dist["P"].values

        # Simple cache: {cache_key: result_dict}
        self._cache: Dict[str, Dict[str, NDArray]] = {}

    def _validate_bootstrap(self) -> None:
        """
        Validate that bootstrap results exist and are valid.

        Raises
        ------
        ValueError
            If bootstrap has not been run or results are invalid
        """
        if not hasattr(self.sm, "_boot") or self.sm._boot is None:
            raise ValueError(
                "Bootstrap results not found. "
                "Run sm.run_bootstrap() before computing summaries."
            )

        boot = self.sm._boot

        # Check that we have successful bootstrap samples
        if boot.n_successful == 0:
            raise ValueError("No successful bootstrap iterations")

        # Check for invalid values
        if np.any(~np.isfinite(boot.intensity_samples)):
            raise ValueError("Bootstrap intensity samples contain NaN or Inf values")

        if np.any(~np.isfinite(boot.rate_samples)):
            raise ValueError("Bootstrap rate samples contain NaN or Inf values")

        # Warn if success rate is low
        success_rate = boot.n_successful / boot.n_requested
        if success_rate < 0.9:
            warnings.warn(
                f"Only {boot.n_successful}/{boot.n_requested} bootstrap iterations "
                f"succeeded ({success_rate:.1%}). Results may be unreliable.",
                UserWarning,
            )

    def _validate_alpha(self, alpha: float) -> None:
        """Validate alpha parameter."""
        if not 0 < alpha < 1:
            raise ValueError(f"alpha must be in (0, 1), got {alpha}")

    def _compute_quantiles(
        self, data: NDArray, probs: Tuple[float, ...], axis: int = 0
    ) -> NDArray:
        """
        Compute quantiles with validation and R-compatible method.

        Parameters
        ----------
        data : NDArray
            Input data, shape (n_samples, ...)
        probs : tuple of float
            Quantile probabilities in [0, 1]
        axis : int, default=0
            Axis along which to compute quantiles

        Returns
        -------
        quantiles : NDArray
            Shape (len(probs), ...) with quantiles along axis 0
        """
        # Validate probabilities
        if not all(0 <= p <= 1 for p in probs):
            raise ValueError(f"All probabilities must be in [0, 1], got {probs}")

        # Sort probabilities to ensure correct ordering in output
        if list(probs) != sorted(probs):
            warnings.warn(
                "Probabilities are not sorted. Output will follow input order.",
                UserWarning,
            )

        result = np.quantile(data, probs, axis=axis)

        return result

    def _depixilate_samples(self, samples: NDArray, needs_age_dist: bool) -> NDArray:
        """
        Depixilate all bootstrap samples efficiently.

        CRITICAL: This must be done BEFORE computing quantiles, because
        depixilation is a nonlinear transformation that doesn't commute
        with quantile operations.

        Parameters
        ----------
        samples : NDArray
            Bootstrap samples, shape (n_boots, B, B)
        needs_age_dist : bool
            Whether depixilate requires age_dist
            - True for intensity (contact counts per person)
            - False for rate (per-capita contact rates)

        Returns
        -------
        depix_samples : NDArray
            Depixilated samples, shape (n_boots, age_bins.range, age_bins.range)

        Notes
        -----
        For efficiency, we preallocate the output array and use
        in-place operations where possible.
        """
        n_boots = samples.shape[0]
        A = self.age_bins.range

        # Preallocate output
        depix_samples = np.empty((n_boots, A, A), dtype=np.float64)

        # Depixilate each sample
        for i in range(n_boots):
            if needs_age_dist:
                depix_samples[i] = depixilate(
                    samples[i], self.effective_age_bins, self.age_dist
                )
            else:
                depix_samples[i] = depixilate(samples[i], self.effective_age_bins)

        return depix_samples

    def _get_probs_from_alpha(self, alpha: float) -> Tuple[float, float, float]:
        """Convert alpha to (lower, median, upper) probabilities."""
        return (alpha / 2, 0.5, 1 - alpha / 2)

    def summarise_cint(
        self,
        alpha: float = 0.05,
        return_depixilated: bool = False,
        force_recompute: bool = False,
    ) -> Dict[str, NDArray]:
        """
        Compute summary statistics for contact intensity matrix.

        Contact intensity M[c,d] represents the average number of contacts
        that individuals in age group c have with individuals in age group d.

        Parameters
        ----------
        alpha : float, default=0.05
            Significance level for confidence intervals (e.g., 0.05 for 95% CI)
        return_depixilated : bool, default=True
            If True and adaptive merging occurred, return results in original
            age bins. If False or no merging, return in effective bins.
        force_recompute : bool, default=False
            Force recomputation even if cached

        Returns
        -------
        summary : dict
            Dictionary containing:
            - 'lower': Lower confidence bound, shape (B, B)
            - 'median': Median estimate, shape (B, B)
            - 'upper': Upper confidence bound, shape (B, B)
            - 'alpha': Significance level used

        Examples
        --------
        >>> summary = summariser.summarise_cint(alpha=0.05)
        >>> ci_95_lower = summary['lower']
        >>> ci_95_upper = summary['upper']
        >>> median_estimate = summary['median']
        """
        self._validate_alpha(alpha)
        self._validate_bootstrap()

        # Check cache
        cache_key = f"cint_alpha{alpha}_depix{return_depixilated}"

        if not force_recompute and cache_key in self._cache:
            return self._cache[cache_key]

        # Get bootstrap samples
        samples = self.sm._boot.intensity_samples
        probs = self._get_probs_from_alpha(alpha)

        # Depixilate BEFORE computing quantiles if needed
        if return_depixilated:
            samples = self._depixilate_samples(samples, needs_age_dist=True)

        # Compute quantiles
        quantiles = self._compute_quantiles(samples, probs, axis=0)

        # Prepare result
        result = {
            "lower": quantiles[0],
            "median": quantiles[1],
            "upper": quantiles[2],
            "alpha": alpha,
        }

        # Cache and return
        self._cache[cache_key] = result
        return result

    def summarise_rate(
        self,
        alpha: float = 0.05,
        return_depixilated: bool = False,
        force_recompute: bool = False,
    ) -> Dict[str, NDArray]:
        """
        Compute summary statistics for contact rate matrix.

        Contact rate ω[c,d] represents the per-capita rate at which
        individuals in age group c contact individuals in age group d.

        Parameters
        ----------
        alpha : float, default=0.05
            Significance level for confidence intervals
        return_depixilated : bool, default=False
            If True and adaptive merging occurred, return results in original
            age bins. If False or no merging, return in effective bins.
        force_recompute : bool, default=False
            Force recomputation even if cached

        Returns
        -------
        summary : dict
            Dictionary containing:
            - 'lower': Lower confidence bound, shape (B, B)
            - 'median': Median estimate, shape (B, B)
            - 'upper': Upper confidence bound, shape (B, B)
            - 'alpha': Significance level used
        """
        self._validate_alpha(alpha)
        self._validate_bootstrap()

        # Check cache
        cache_key = f"rate_alpha{alpha}_depix{return_depixilated}"

        if not force_recompute and cache_key in self._cache:
            return self._cache[cache_key]

        # Get bootstrap samples
        samples = self.sm._boot.rate_samples
        probs = self._get_probs_from_alpha(alpha)

        # Depixilate BEFORE computing quantiles if needed
        if return_depixilated:
            samples = self._depixilate_samples(samples, needs_age_dist=False)

        # Compute quantiles
        quantiles = self._compute_quantiles(samples, probs, axis=0)

        # Prepare result
        result = {
            "lower": quantiles[0],
            "median": quantiles[1],
            "upper": quantiles[2],
            "alpha": alpha,
        }

        # Cache and return
        self._cache[cache_key] = result
        return result

    def summarise_mcint(
        self,
        alpha: float = 0.05,
        return_depixilated: bool = False,
        force_recompute: bool = False,
    ) -> Dict[str, NDArray]:
        """
        Compute summary statistics for marginal contact intensity.

        Marginal contact intensity m_c = Σ_d M[c,d] represents the total
        average number of contacts made by individuals in age group c
        across all age groups.

        Parameters
        ----------
        alpha : float, default=0.05
            Significance level for confidence intervals
        return_depixilated : bool, default=False
            If True and adaptive merging occurred, return results in original
            age bins. If False or no merging, return in effective bins.
        force_recompute : bool, default=False
            Force recomputation even if cached

        Returns
        -------
        summary : dict
            Dictionary containing:
            - 'lower': Lower confidence bound, shape (B,)
            - 'median': Median estimate, shape (B,)
            - 'upper': Upper confidence bound, shape (B,)
            - 'alpha': Significance level used
            - 'quantiles': All three as array, shape (3, B)

        Notes
        -----
        Marginal intensity is computed by summing the intensity matrix
        over the contact age dimension. When depixilation is needed,
        we must:
        1. Depixilate each full intensity matrix sample
        2. Compute marginals from depixilated matrices
        3. Then compute quantiles

        This ensures mathematical correctness, as marginals and depixilation
        don't commute in general.
        """
        self._validate_alpha(alpha)
        self._validate_bootstrap()

        # Check cache
        needs_depix = return_depixilated and self.effective_age_bins != self.age_bins
        cache_key = f"mcint_alpha{alpha}_depix{needs_depix}"

        if not force_recompute and cache_key in self._cache:
            return self._cache[cache_key]

        # Get bootstrap intensity samples
        intensity_samples = self.sm._boot.intensity_samples
        probs = self._get_probs_from_alpha(alpha)

        # Handle depixilation if needed
        if needs_depix:
            # IMPORTANT: Depixilate FULL matrices first
            intensity_samples = self._depixilate_samples(
                intensity_samples, needs_age_dist=True
            )

        # Compute marginals by summing over contact age (last axis)
        marginal_samples = intensity_samples.sum(axis=-1)  # Shape: (n_boots, B)

        # Compute quantiles
        quantiles = self._compute_quantiles(marginal_samples, probs, axis=0)

        # Prepare result
        result = {
            "lower": quantiles[0],
            "median": quantiles[1],
            "upper": quantiles[2],
            "alpha": alpha,
        }

        # Cache and return
        self._cache[cache_key] = result
        return result

    def get_point_estimates(
        self, return_depixilated: bool = False
    ) -> Dict[str, Dict[str, NDArray]]:
        """
        Get point estimates (mean and std) for all statistics.

        Parameters
        ----------
        return_depixilated : bool, default=False
            Whether to return depixilated results if adaptive merging occurred

        Returns
        -------
        estimates : dict
            Nested dictionary with structure:
            {
                'cint': {'mean': array, 'std': array},
                'rate': {'mean': array, 'std': array},
                'mcint': {'mean': array, 'std': array}
            }

        Examples
        --------
        >>> estimates = summariser.get_point_estimates()
        >>> cint_mean = estimates['cint']['mean']
        >>> cint_std = estimates['cint']['std']
        """
        self._validate_bootstrap()

        needs_depix = return_depixilated and self.effective_age_bins != self.age_bins

        boot = self.sm._boot

        # Get samples (potentially depixilated)
        if needs_depix:
            cint_samples = self._depixilate_samples(
                boot.intensity_samples, needs_age_dist=True
            )
            rate_samples = self._depixilate_samples(
                boot.rate_samples, needs_age_dist=False
            )
        else:
            cint_samples = boot.intensity_samples
            rate_samples = boot.rate_samples

        # Compute marginals
        mcint_samples = cint_samples.sum(axis=2)

        return {
            "cint": {
                "mean": cint_samples.mean(axis=0),
                "std": cint_samples.std(axis=0, ddof=1),  # Sample std
            },
            "rate": {
                "mean": rate_samples.mean(axis=0),
                "std": rate_samples.std(axis=0, ddof=1),
            },
            "mcint": {
                "mean": mcint_samples.mean(axis=0),
                "std": mcint_samples.std(axis=0, ddof=1),
            },
        }

    def clear_cache(self) -> None:
        """Clear all cached computations."""
        self._cache.clear()

    def get_cache_info(self) -> Dict[str, int]:
        """
        Get information about cached results.

        Returns
        -------
        info : dict
            Dictionary with cache statistics
        """
        return {"n_cached": len(self._cache), "cache_keys": list(self._cache.keys())}


class ModelSummariserMCMC:
    """
    Basic Model implementation only
    """

    def __init__(self, model):
        self.model = model
        self.prng_key = jax.random.PRNGKey(0)

    def get_posterior(self):
        """Get posterior samples from the MCMC run."""
        self.post = self.model.mcmc.get_samples()

    def get_post_cint(self):
        """Calculate posterior contact intensity from MCMC samples"""
        if not hasattr(self, "post"):
            self.get_posterior()
        self.post_cint = {"general": np.exp(self.post["log_cint"])}
        return self.post_cint

    def summarise_rate(self, probs: tuple = (0.025, 0.5, 0.975)):
        """Summarise the posterior contact rate

        Parameters
        ----------
        probs : tuple
            The quantiles to compute

        Returns
        -------
        dict
            A dictionary containing the quantiles of the posterior contact rate
        """
        if not hasattr(self, "post"):
            self.post = self.get_posterior()

        if not hasattr(self, "sum_post_rate"):
            self.sum_post_rate = np.quantile(
                np.exp(self.post["log_rate"]), probs, axis=0
            )

        return self.sum_post_rate

    def summarise_cint(self, probs: tuple = (0.025, 0.5, 0.975)):
        """Summarise the posterior contact intensity

        Parameters
        ----------
        probs : tuple
            The quantiles to compute

        Returns
        -------
        dict
            A dictionary containing the quantiles of the posterior contact intensity
        """
        if not hasattr(self, "post_cint"):
            self.get_post_cint()

        if not hasattr(self, "sum_post_cint"):
            self.sum_post_cint = {
                name: np.quantile(value, probs, axis=0)
                for name, value in self.post_cint.items()
            }

        return self.sum_post_cint

    def summarise_mcint(self, probs: tuple = (0.025, 0.5, 0.975)):
        """Summarise the posterior marginal contact intensity

        Parameters
        ----------
        probs : tuple
            The quantiles to compute

        Returns
        -------
        dict
            A dictionary containing the quantiles of the posterior marginal contact intensity
        """
        if not hasattr(self, "post_cint"):
            self.get_post_cint()

        if not hasattr(self, "sum_post_mcint"):
            self.sum_post_mcint = {
                var: {
                    name: np.quantile(value.sum(axis=-1), probs, axis=0)
                    for name, value in cat.items()
                }
                for var, cat in self.post_cint.items()
            }

        return self.sum_post_mcint


class ModelSummariserPrem:
    def __init__(
        self,
        model: Prem,
        age_bins: AgeBins = None,
        age_dist: NDArray | None = None,
        age_grp_dist: NDArray | None = None,
        alpha=0.05,
    ):
        """Summarises the inference results for Prem et al. style models.

        Parameters
        ----------
        model: Prem
                A object of class Prem. SVI or MCMC must have been run.
        age_bins: AgeBins, optional
                AgeBins object that defines the age bins used in the model.
                This data is used for pixilating and depixilating the contact intensity matrix.
        age_dist: NDArray, optional
                An array of population sizes for each fine age (1-year age).
                This data is used for depixilating the contact intensity matrix.
                Must be an NDarray of shape (num_fine_age,).
        age_grp_dist: NDArray, optional
                An array of population sizes for each age group.
                Must be of shape (num_age_grps,).
        alpha: float, default=0.05
                Significance level for credible intervals.
        """

        self.model = model
        self.age_bins = age_bins
        self.age_dist = age_dist
        self.age_grp_dist = age_grp_dist
        self.alpha = alpha
        self.prng_key = jax.random.PRNGKey(0)

        if hasattr(model, "mcmc"):
            self.post = model.mcmc.get_samples()
        elif hasattr(model, "svi"):
            self.post = model.posterior_predictive_svi(self.prng_key, model.guide)
        else:
            raise ValueError("Model must have either mcmc or svi attributes.")
        self.post_cint = np.exp(self.post["log_cint"])

        if self.age_grp_dist is None:
            if self.age_bins is not None and self.age_dist is not None:
                # Calculate age_grp_dist from age_dist and age_bins
                age_grp_dist = []
                age_edges = self.age_bins.left + [self.age_bins.max + 1]
                for i in range(len(age_edges) - 1):
                    start_age = age_edges[i]
                    end_age = age_edges[i + 1]
                    age_grp_dist.append(self.age_dist[start_age:end_age].sum())

                self.age_grp_dist = np.array(age_grp_dist)

    def symmetrize_cint(self):
        """Symmetrize the contact intensity matrix using the reciprocity adjustment."""

        if self.age_grp_dist is None:
            raise ValueError("age_grp_dist must be provided for symmetrization.")

        # Symmetrize the contact intensity matrix
        M = self.post_cint
        print()
        P = np.diag(self.age_grp_dist)[np.newaxis, ...]
        P_inv = np.diag(1 / self.age_grp_dist)[np.newaxis, ...]
        self.post_cint = 0.5 * (M + P_inv @ np.transpose(M, (0, 2, 1)) @ P)

    def summarise_cint(
        self,
        depix: bool = False,
        symmetrize: bool = False,
        probs: tuple = None,
        alpha: float = 0.05,
    ):
        """
        Summarise the posterior contact intensity matrix.

        Parameters
        ----------
        depix: bool, default=False
                Whether to depixilate the contact intensity matrix.
        symmetrize: bool, default=False
                Whether to apply the reciprocity adjustment to the contact intensity matrix.
        probs: tuple, optional
                The quantiles to compute. If None, uses (alpha/2, 0.5, 1-alpha/2).
        alpha: float, default=0.05
                Significance level for credible intervals.

        Returns
        -------
        NDArray
                The quantiles of the contact intensity matrix.
        """
        if probs is None:
            probs = (alpha / 2, 0.5, 1 - alpha / 2)

        if symmetrize:
            self.symmetrize_cint()

        self.sum_cint = np.quantile(self.post_cint, probs, axis=0)

        if depix:
            if self.age_bins is None:
                raise ValueError("age_bins must be provided for depixilation.")
            if self.age_dist is None:
                raise ValueError("age_dist must be provided for depixilation.")

            # Depixilate the summed contact intensity
            if not hasattr(self, "depix_sum_cint"):
                self.depix_sum_cint = np.array(
                    [
                        depixilate(self.sum_cint[i, :, :], self.age_bins, self.age_dist)
                        for i in range(self.sum_cint.shape[0])
                    ]
                )

            return self.depix_sum_cint
        else:
            return self.sum_cint

    def summarise_mcint(
        self,
        depixilate: bool = False,
        symmetrize: bool = False,
        probs: tuple = None,
        alpha: float = 0.05,
    ):
        """
                Summarise the marginal contact intensity of the model.

                Parameters
                ----------
                depixilate: bool, default=False
                        Whether to depixilate the contact intensity matrix before calculating marginal contact intensity.
                symmetrize: bool, default=False
                        Whether to apply the reciprocity adjustment to the contact intensity matrix.
                probs: tuple, optional
                        The quantiles to compute. If None, uses (alpha/2, 0.5, 1-alpha/2).
                alpha: float, default=0.05
                        Significance level for credible intervals.

        Returns
                -------
                NDArray
                        The quantiles of the marginal contact intensity.
        """
        if probs is None:
            probs = (alpha / 2, 0.5, 1 - alpha / 2)

        if not hasattr(self, "sum_mcint"):
            if symmetrize:
                self.symmetrize_cint()
            mcint = self.post_cint.sum(axis=1)
            self.sum_mcint = np.quantile(mcint, probs, axis=0)

        if depixilate and self.age_bins is not None and self.age_dist is not None:
            if not hasattr(self, "depix_sum_mcint"):
                # Depixilate the summed marginal contact intensity
                depix_cint = np.array(
                    [
                        depixilate(
                            self.post_cint[i, :, :], self.age_bins, self.age_dist
                        )
                        for i in range(self.post_cint.shape[0])
                    ]
                )
                depix_mcint = depix_cint.sum(axis=1)
                self.depix_sum_mcint = np.quantile(depix_mcint, probs, axis=0)

            return self.depix_sum_mcint
        else:
            return self.sum_mcint
