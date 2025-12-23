import warnings
from typing import Any, Dict, Literal, Optional, Tuple

import jax
import numpy as np
import pandas as pd
from jax.random import PRNGKey
from numpy.typing import NDArray

from ..models import BRCfine, BRCrefine, HiBRCfine, HiBRCrefine, Prem, SocialMix
from ..models._SocialMix import AgeBinProcessor, InputValidator
from ..models._vdKassteele import vdKassteele
from ..utils import AgeBins, depixilate, pixilate


def validate_alpha(alpha: float) -> None:
    """Validate alpha parameter."""
    if not 0 < alpha < 1:
        raise ValueError(f"alpha must be in (0, 1), got {alpha}")


def get_probs_from_alpha(alpha: float) -> Tuple[float, float, float]:
    """Convert alpha to (lower, median, upper) probabilities."""
    return (alpha / 2, 0.5, 1 - alpha / 2)


def compute_quantiles(
    samples: NDArray, probs: Tuple[float, ...], axis: int = 0
) -> NDArray:
    """
    Compute quantiles with validation and R-compatible method.

    Parameters
    ----------
    samples : NDArray
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

    result = np.quantile(samples, probs, axis=axis)

    return result


class ModelSummariserBRC:
    """
    Statistical summariser for BRC model inference results (MCMC or SVI).

    Unified summariser for BRCfine, BRCrefine, HiBRCfine, HiBRCrefine, and vdKassteele models.
    Computes quantiles and credible intervals for contact matrices from MCMC or SVI
    posterior samples, with automatic detection of inference method and model type.

    Parameters
    ----------
    model : BRCfine | BRCrefine | HiBRCfine | HiBRCrefine | vdKassteele
        Fitted BRC or vdKassteele model with MCMC or SVI results.
    num_samples : int, default=3000
        Number of posterior samples to draw if using SVI.

    Attributes
    ----------
    model : BRC
        Reference to the BRC model.
    inference_method : Literal["mcmc", "svi"]
        Detected inference method ("mcmc" or "svi").
    model_type : Literal["brc", "hibrc"]
        Detected model type ("brc" for BRCfine/BRCrefine, "hibrc" for HiBRCfine/HiBRCrefine).
    num_samples : int
        Number of samples for SVI posterior.
    post_samples : Dict[str, NDArray]
        Posterior samples from MCMC or SVI.
    post_cint_samples : NDArray | Dict
        Posterior contact intensity samples.
        - For BRC: NDArray of shape (n_samples, A, A)
        - For HiBRC: Dict[str, NDArray] mapping full_labels to samples
          where full_label is like "M_A->All" (shape: (n_samples, A, A) each)

    Raises
    ------
    ValueError
        If neither MCMC nor SVI has been run on the model.
        If model type is not supported.
    TypeError
        If model is not a BRC-family model.

    Examples
    --------
    >>> from cntmosaic.models import BRCfine
    >>> from cntmosaic.models.priors import Spline2D
    >>> from cntmosaic.analysis import ModelSummariserBRC
    >>>
    >>> # Fit BRC model with MCMC
    >>> model = BRCfine(dataloader, priors={"rate": Spline2D()})
    >>> model.run_inference_mcmc(PRNGKey(0), num_samples=1000)
    >>>
    >>> # Create summariser (auto-detects MCMC)
    >>> summariser = ModelSummariserBRC(model)
    >>>
    >>> # Get 95% credible intervals for contact intensity
    >>> summary = summariser.summarise_cint(alpha=0.05)
    >>> lower, median, upper = summary['lower'], summary['median'], summary['upper']
    >>>
    >>> # Also works with SVI
    >>> model_svi = BRCfine(dataloader, priors={"rate": Spline2D()})
    >>> model_svi.run_inference_svi(PRNGKey(0), guide, num_steps=10000)
    >>> summariser_svi = ModelSummariserBRC(model_svi, num_samples=3000)
    >>>
    >>> # Works with hierarchical models too
    >>> hibrc = HiBRCfine(dataloader, priors)
    >>> hibrc.run_inference_mcmc(PRNGKey(0), num_samples=1000)
    >>> summariser_hibrc = ModelSummariserBRC(hibrc)
    >>> summary_by_group = summariser_hibrc.summarise_cint(alpha=0.05)

    Notes
    -----
    - Automatically detects inference method (MCMC vs SVI)
    - Automatically detects model type (BRC vs HiBRC)
    - Unified API across all BRC-family models
    - Efficient memory management for large posterior samples
    - Results are cached to avoid redundant computation

    See Also
    --------
    ModelSummariserPrem : Summariser for Prem models
    ModelSummariserSocialMix : Summariser for SocialMix models
    """

    def __init__(
        self,
        model: "BRCfine | BRCrefine | HiBRCfine | HiBRCrefine",
        num_samples: int = 3000,
    ) -> None:
        """
        Initialize summariser with a BRC model.

        Parameters
        ----------
        model : BRCfine | BRCrefine | HiBRCfine | HiBRCrefine | vdKassteele
            BRC or vdKassteele model with completed MCMC or SVI inference.
        num_samples : int, default=3000
            Number of posterior samples to draw if using SVI.

        Raises
        ------
        ValueError
            If neither MCMC nor SVI has been run on the model.
        TypeError
            If model is not a BRC-family model.
        """
        # Validate model type
        from ..models import BRCfine, BRCrefine, HiBRCfine, HiBRCrefine, vdKassteele

        if not isinstance(
            model, (BRCfine, BRCrefine, HiBRCfine, HiBRCrefine, vdKassteele)
        ):
            raise TypeError(
                f"Model must be a BRC-family model (BRCfine, BRCrefine, HiBRCfine, HiBRCrefine) or"
                "vdKassteele, "
                f"got {type(model).__name__}"
            )

        # Detect inference method
        has_mcmc = hasattr(model, "_mcmc_result") and model._mcmc_result is not None
        has_svi = hasattr(model, "_svi_result") and model._svi_result is not None

        if not (has_mcmc or has_svi):
            raise ValueError(
                "Neither MCMC nor SVI has been run on the model. "
                "Call model.run_inference_mcmc() or model.run_inference_svi() first."
            )

        # Store configuration
        self.model = model
        self.inference_method: Literal["mcmc", "svi"] = "mcmc" if has_mcmc else "svi"
        self.num_samples = num_samples

        # Detect model type
        # vdKassteele can be either BRC or HIBRC depending on prior_type
        if isinstance(model, vdKassteele):
            if model.prior_type == "global":
                self.model_type: Literal["brc", "hibrc"] = "brc"
            else:
                self.model_type: Literal["brc", "hibrc"] = "hibrc"
        elif isinstance(model, (HiBRCfine, HiBRCrefine)):
            self.model_type: Literal["brc", "hibrc"] = "hibrc"
        else:
            self.model_type: Literal["brc", "hibrc"] = "brc"

        # Initialize cache
        self._cache: Dict[str, Dict[str, NDArray]] = {}

        # Load posterior samples and compute contact intensities
        self._load_posterior()
        self._compute_contact_intensities()

    def _load_posterior(self) -> None:
        """Load posterior samples from MCMC or SVI."""
        if self.inference_method == "mcmc":
            self.post_samples = self.model._mcmc_result.get_samples()
        else:  # svi
            # Generate samples from variational posterior
            self.post_samples = self.model.posterior_predictive_svi(
                PRNGKey(0), self.model._guide, num_samples=self.num_samples
            )

    def _compute_contact_intensities(self) -> None:
        """
        Compute contact intensity samples from posterior.

        For BRC models: contact intensity = exp(log_cint)
        For HiBRC models: contact intensity = exp(log_rate + log_delta + log_P)
        For vdKassteele models (stratified): contact intensity = exp(log_rate + log_P)
        """
        if self.model_type == "brc":
            # Simple case: contact intensity is directly available
            if "log_cint" in self.post_samples:
                self.post_cint_samples = np.exp(self.post_samples["log_cint"])
            else:
                raise ValueError(
                    "Posterior samples must contain 'log_cint' field for BRC models"
                )

        else:  # hibrc or vdKassteele
            # Complex case: compute contact intensity for each stratification category
            if isinstance(self.model, vdKassteele):
                self.post_cint_samples = self._compute_vdKassteele_contact_intensities()
            else:
                self.post_cint_samples = self._compute_hibrc_contact_intensities()

    def _compute_vdKassteele_contact_intensities(self) -> Dict[str, NDArray]:
        """
        Compute stratified contact intensities for vdKassteele models.

        Returns
        -------
        Dict[str, NDArray]
            Dictionary mapping full_labels to intensity samples:
            {label: intensity_samples} where intensity_samples has shape (n_samples, A, A)

            For example: {"M_A->All": array(...), "F_B->All": array(...)}

        Notes
        -----
        vdKassteele differs from HiBRC models:
        - log_rate is 4D (n_samples, n_strata, A, A) when stratified
        - No separate log_delta parameter
        - Stratification is encoded directly in log_rate
        """
        log_rate = self.post_samples["log_rate"].astype(np.float32)
        log_P = self.model.log_P.astype(np.float32)  # Shape (1, A) or (K, A)

        # Get full labels sorted by flat_ix
        full_labels = self.model.data.strat_data["full_labels"]

        # Map flat_ix to flat_pixs for population stratification
        flat_ix_to_flat_pixs = {}
        flat_ix_array = self.model.data.strat_data["flat_ix"]
        flat_pixs_array = self.model.data.strat_data["flat_pixs"]
        for flat_idx in np.unique(flat_ix_array):
            mask = flat_ix_array == flat_idx
            flat_pixs_val = flat_pixs_array[mask][0]
            flat_ix_to_flat_pixs[flat_idx] = flat_pixs_val

        post_cint = {}
        flat_ix_values = np.unique(flat_ix_array)

        # log_rate has shape (n_samples, n_strata, A, A)
        A = log_rate.shape[2]

        # Process each stratum separately
        for flat_idx in flat_ix_values:
            label = full_labels[flat_idx]

            # Extract log_rate for this stratum: (n_samples, A, A)
            log_rate_matrix = log_rate[:, flat_idx, :, :]

            # Get the appropriate row of log_P for this stratum
            flat_pixs_idx = flat_ix_to_flat_pixs[flat_idx]
            log_P_vector = log_P[flat_pixs_idx, :]  # Shape (A,)

            # Broadcast log_P along contact age dimension
            log_P_matrix = log_P_vector[np.newaxis, :]  # Shape (1, A)

            # Compute contact intensity: exp(log_rate + log_P)
            # vdKassteele does NOT use log_delta
            log_sum = log_rate_matrix + log_P_matrix
            cint = np.exp(log_sum, dtype=np.float32)  # (n_samples, A, A)

            post_cint[label] = cint

            # Free memory
            del log_rate_matrix, log_sum, cint

        return post_cint

    def _compute_hibrc_contact_intensities(self) -> Dict[str, NDArray]:
        """
        Compute stratified contact intensities for HiBRC models.

        Returns
        -------
        Dict[str, NDArray]
            Dictionary mapping full_labels to intensity samples:
            {label: intensity_samples} where intensity_samples has shape (n_samples, A, A)

            For example: {"M_A->All": array(...), "F_B->All": array(...)}

        Notes
        -----
        Memory-efficient implementation:
        - Processes each category separately to avoid memory explosion
        - Uses float32 for intermediate computations
        - Maps flat_ix categories to full_labels for intuitive access
        - Handles log_P correctly for PARTIAL (shape (1, A)) vs FULL (shape (K, A)) stratification
        """
        log_rate = self.post_samples["log_rate"].astype(np.float32)
        log_delta = self.post_samples["log_delta"].astype(np.float32)
        log_P = self.model.log_P.astype(np.float32)  # Shape (1, A) or (K, A)

        # Get full labels sorted by flat_ix
        full_labels = self.model.data.strat_data["full_labels"]

        # Map flat_ix to flat_pixs for population stratification
        # For each flat_ix, find the corresponding flat_pixs value
        flat_ix_to_flat_pixs = {}
        flat_ix_array = self.model.data.strat_data["flat_ix"]
        flat_pixs_array = self.model.data.strat_data["flat_pixs"]
        for flat_idx in np.unique(flat_ix_array):
            # Get first occurrence of this flat_idx
            mask = flat_ix_array == flat_idx
            flat_pixs_val = flat_pixs_array[mask][0]
            flat_ix_to_flat_pixs[flat_idx] = flat_pixs_val

        post_cint = {}

        # Get unique flat_ix values
        flat_ix_values = np.unique(flat_ix_array)

        # Process each stratum separately to save memory
        for flat_idx in flat_ix_values:
            label = full_labels[flat_idx]

            # Extract this stratum's offset: (n_samples, A, A)
            log_delta_stratum = log_delta[:, flat_idx, :, :]

            # Reshape log_rate from (n_samples, A*A) to (n_samples, A, A)
            A = log_delta_stratum.shape[1]
            log_rate_matrix = log_rate.reshape(-1, A, A)

            # Get the appropriate row of log_P for this stratum
            # log_P shape is (1, A) for PARTIAL or (K, A) for FULL stratification
            flat_pixs_idx = flat_ix_to_flat_pixs[flat_idx]
            log_P_vector = log_P[flat_pixs_idx, :]  # Shape (A,)

            # Broadcast log_P along contact age dimension: (A,) -> (A, A)
            # by adding along axis 0 (participant age dimension)
            log_P_matrix = log_P_vector[np.newaxis, :]  # Shape (1, A)

            # Compute contact intensity: exp(log_rate + log_delta + log_P)
            # Broadcasting: (n_samples, A, A) + (n_samples, A, A) + (1, A)
            log_sum = log_rate_matrix + log_delta_stratum + log_P_matrix
            cint = np.exp(log_sum, dtype=np.float32)  # (n_samples, A, A)

            post_cint[label] = cint

            # Free memory immediately
            del log_delta_stratum, log_rate_matrix, log_sum, cint

        return post_cint

    def summarise_rate(
        self,
        alpha: float = 0.05,
        probs: Optional[Tuple[float, ...]] = None,
    ) -> NDArray | Dict[str, NDArray]:
        """
        Compute summary statistics for contact rate matrix.

        Contact rate R[c,d] represents the per-capita rate at which
        individuals in age c contact individuals in age d.

        Parameters
        ----------
        alpha : float, default=0.05
            Significance level for credible intervals (e.g., 0.05 for 95% CI).
            Ignored if probs is provided.
        probs : Tuple[float, ...], optional
            Specific quantile probabilities to compute.
            If None, uses (alpha/2, 0.5, 1-alpha/2).

        Returns
        -------
        NDArray | Dict
            - For BRC models: NDArray of shape (3, A, A) or (len(probs), A, A)
              containing [lower, median, upper] quantiles (or custom quantiles)
            - For HiBRC models: Dict[str, NDArray] with structure
              {full_label: quantiles} where full_label is like "M_A->All"

        Raises
        ------
        ValueError
            If alpha not in (0, 1).

        Examples
        --------
        >>> # BRC model
        >>> summary = summariser.summarise_rate(alpha=0.05)
        >>> lower, median, upper = summary[0], summary[1], summary[2]
        >>>
        >>> # Custom quantiles
        >>> quantiles = summariser.summarise_rate(probs=(0.1, 0.5, 0.9))
        >>>
        >>> # HiBRC model
        >>> summary = summariser_hibrc.summarise_rate(alpha=0.05)
        >>> male_a_median = summary['M_A->All'][1]

        Notes
        -----
        For BRC models, rate = exp(log_rate) from posterior samples.
        For HiBRC models, rate = exp(log_rate + log_delta) for each category.
        """
        if probs is None:
            validate_alpha(alpha)
            probs = get_probs_from_alpha(alpha)

        cache_key = f"rate_probs{probs}"
        if cache_key in self._cache:
            return self._cache[cache_key]

        if self.model_type == "brc":
            # Simple case: rate is directly available
            rate_samples = np.exp(self.post_samples["log_rate"])
            result = compute_quantiles(rate_samples, probs, axis=0)

        else:  # hibrc
            # Compute rate for each stratum
            log_rate = self.post_samples["log_rate"]
            log_delta = self.post_samples["log_delta"]
            full_labels = self.model.data.strat_data["full_labels"]
            flat_ix_values = np.unique(self.model.data.strat_data["flat_ix"])

            result = {}
            for flat_idx in flat_ix_values:
                label = full_labels[flat_idx]
                # rate = exp(log_rate + log_delta[flat_idx])
                log_rate_stratum = log_rate + log_delta[:, flat_idx, :, :]
                rate_samples = np.exp(log_rate_stratum)
                result[label] = compute_quantiles(rate_samples, probs, axis=0)

        self._cache[cache_key] = result
        return result

    def summarise_cint(
        self,
        alpha: float = 0.05,
        probs: Optional[Tuple[float, ...]] = None,
    ) -> NDArray | Dict[str, NDArray]:
        """
        Compute summary statistics for contact intensity matrix.

        Contact intensity M[c,d] represents the average number of contacts
        that individuals in age c have with individuals in age d.

        Parameters
        ----------
        alpha : float, default=0.05
            Significance level for credible intervals (e.g., 0.05 for 95% CI).
            Ignored if probs is provided.
        probs : Tuple[float, ...], optional
            Specific quantile probabilities to compute.
            If None, uses (alpha/2, 0.5, 1-alpha/2).

        Returns
        -------
        NDArray | Dict
            - For BRC models: NDArray of shape (3, A, A) or (len(probs), A, A)
            - For HiBRC models: Dict[str, NDArray] with structure
              {full_label: quantiles} where full_label is like "M_A->All"

        Raises
        ------
        ValueError
            If alpha not in (0, 1).

        Examples
        --------
        >>> summary = summariser.summarise_cint(alpha=0.05)
        >>> lower, median, upper = summary[0], summary[1], summary[2]
        >>>
        >>> # For HiBRC with gender stratification
        >>> summary = summariser_hibrc.summarise_cint(alpha=0.05)
        >>> male_median = summary['M_A->All'][1]
        >>> female_median = summary['F_B->All'][1]

        Notes
        -----
        Contact intensity incorporates population structure through:
        M[c,d] = exp(log_rate[c,d] + log_P[d])
        """
        if probs is None:
            validate_alpha(alpha)
            probs = get_probs_from_alpha(alpha)

        cache_key = f"cint_probs{probs}"
        if cache_key in self._cache:
            return self._cache[cache_key]

        if self.model_type == "brc":
            # Simple case: direct quantile computation
            result = compute_quantiles(self.post_cint_samples, probs, axis=0)

        else:  # hibrc
            # Compute quantiles for each stratum
            result = {}
            for label, samples in self.post_cint_samples.items():
                result[label] = compute_quantiles(samples, probs, axis=0)

        self._cache[cache_key] = result
        return result

    def summarise_mcint(
        self,
        alpha: float = 0.05,
        probs: Optional[Tuple[float, ...]] = None,
    ) -> NDArray | Dict[str, NDArray]:
        """
        Compute summary statistics for marginal contact intensity.

        Marginal contact intensity m[c] = Σ_d M[c,d] represents the total
        average number of contacts made by individuals in age c across all ages.

        Parameters
        ----------
        alpha : float, default=0.05
            Significance level for credible intervals (e.g., 0.05 for 95% CI).
            Ignored if probs is provided.
        probs : Tuple[float, ...], optional
            Specific quantile probabilities to compute.
            If None, uses (alpha/2, 0.5, 1-alpha/2).

        Returns
        -------
        NDArray | Dict
            - For BRC models: NDArray of shape (3, A) or (len(probs), A)
            - For HiBRC models: Dict[str, NDArray] with structure
              {full_label: quantiles} where full_label is like "M_A->All"

        Examples
        --------
        >>> summary = summariser.summarise_mcint(alpha=0.05)
        >>> lower, median, upper = summary[0], summary[1], summary[2]
        >>> # median[age] gives median total contacts for that age
        >>>
        >>> # For HiBRC
        >>> summary = summariser_hibrc.summarise_mcint(alpha=0.05)
        >>> male_median = summary['M_A->All'][1]  # shape (A,)

        Notes
        -----
        Marginal contact intensity is computed by summing the contact intensity
        matrix over the contact age dimension: m[c] = Σ_d M[c,d]
        """
        if probs is None:
            validate_alpha(alpha)
            probs = get_probs_from_alpha(alpha)

        cache_key = f"mcint_probs{probs}"
        if cache_key in self._cache:
            return self._cache[cache_key]

        if self.model_type == "brc":
            # Sum over contact age (last axis)
            mcint_samples = self.post_cint_samples.sum(axis=-1)
            result = compute_quantiles(mcint_samples, probs, axis=0)

        else:  # hibrc
            # Compute marginal for each stratum
            result = {}
            for label, samples in self.post_cint_samples.items():
                # Sum over contact age dimension
                mcint_samples = samples.sum(axis=-1)
                result[label] = compute_quantiles(mcint_samples, probs, axis=0)

        self._cache[cache_key] = result
        return result

    def get_posterior_samples(
        self,
        quantity: Literal["rate", "cint", "mcint"] = "cint",
    ) -> NDArray | Dict[str, NDArray]:
        """
        Get raw posterior samples for specified quantity.

        Useful for custom post-processing or plotting.

        Parameters
        ----------
        quantity : {"rate", "cint", "mcint"}, default="cint"
            Which quantity to return samples for:
            - "rate": Contact rate matrix R[c,d]
            - "cint": Contact intensity matrix M[c,d]
            - "mcint": Marginal contact intensity m[c]

        Returns
        -------
        NDArray | Dict
            - For BRC models: NDArray of shape (n_samples, A, A) or (n_samples, A)
            - For HiBRC models: Dict[str, NDArray] with structure
              {full_label: samples} where full_label is like "M_A->All"

        Examples
        --------
        >>> # Get raw samples for custom analysis
        >>> samples = summariser.get_posterior_samples("cint")
        >>> # Custom statistic: 90th percentile
        >>> custom_stat = np.percentile(samples, 90, axis=0)
        >>>
        >>> # For plotting posterior distributions
        >>> import matplotlib.pyplot as plt
        >>> samples = summariser.get_posterior_samples("mcint")
        >>> plt.hist(samples[:, 25])  # Distribution for age 25
        >>>
        >>> # For HiBRC
        >>> samples = summariser_hibrc.get_posterior_samples("cint")
        >>> male_samples = samples['M_A->All']  # shape (n_samples, A, A)
        """
        if quantity == "rate":
            if self.model_type == "brc":
                return np.exp(self.post_samples["log_rate"])
            else:  # hibrc
                result = {}
                log_rate = self.post_samples["log_rate"]
                log_delta = self.post_samples["log_delta"]
                full_labels = self.model.data.strat_data["full_labels"]
                flat_ix_values = np.unique(self.model.data.strat_data["flat_ix"])

                for flat_idx in flat_ix_values:
                    label = full_labels[flat_idx]
                    log_rate_stratum = log_rate + log_delta[:, flat_idx, :, :]
                    result[label] = np.exp(log_rate_stratum)
                return result

        elif quantity == "cint":
            return self.post_cint_samples

        elif quantity == "mcint":
            if self.model_type == "brc":
                return self.post_cint_samples.sum(axis=-1)
            else:  # hibrc
                result = {}
                for label, samples in self.post_cint_samples.items():
                    result[label] = samples.sum(axis=-1)
                return result

        else:
            raise ValueError(
                f"Unknown quantity: {quantity}. Must be 'rate', 'cint', or 'mcint'"
            )

    def get_point_estimates(
        self,
        quantity: Literal["rate", "cint", "mcint"] = "cint",
    ) -> Dict[str, NDArray] | Dict[str, Dict[str, NDArray]]:
        """
        Get point estimates (mean and std) for specified quantity.

        Parameters
        ----------
        quantity : {"rate", "cint", "mcint"}, default="cint"
            Which quantity to compute estimates for.

        Returns
        -------
        Dict
            - For BRC models: {'mean': array, 'std': array}
            - For HiBRC models: {full_label: {'mean': array, 'std': array}}
              where full_label is like "M_A->All"

        Examples
        --------
        >>> estimates = summariser.get_point_estimates("cint")
        >>> cint_mean = estimates['mean']
        >>> cint_std = estimates['std']
        >>>
        >>> # For HiBRC
        >>> estimates = summariser_hibrc.get_point_estimates("cint")
        >>> male_mean = estimates['M_A->All']['mean']
        >>> male_std = estimates['M_A->All']['std']
        """
        samples = self.get_posterior_samples(quantity)

        if self.model_type == "brc":
            return {
                "mean": samples.mean(axis=0),
                "std": samples.std(axis=0, ddof=1),
            }
        else:  # hibrc
            result = {}
            for label, label_samples in samples.items():
                result[label] = {
                    "mean": label_samples.mean(axis=0),
                    "std": label_samples.std(axis=0, ddof=1),
                }
            return result

    def clear_cache(self) -> None:
        """Clear all cached computations."""
        self._cache.clear()

    def get_cache_info(self) -> Dict[str, Any]:
        """
        Get information about cached results.

        Returns
        -------
        info : Dict[str, Any]
            Dictionary with cache statistics.
        """
        return {
            "n_cached": len(self._cache),
            "cache_keys": list(self._cache.keys()),
            "inference_method": self.inference_method,
            "model_type": self.model_type,
        }


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
        validate_alpha(alpha)
        self._validate_bootstrap()

        # Check cache
        cache_key = f"cint_alpha{alpha}_depix{return_depixilated}"

        if not force_recompute and cache_key in self._cache:
            return self._cache[cache_key]

        # Get bootstrap samples
        samples = self.sm._boot.intensity_samples
        probs = get_probs_from_alpha(alpha)

        # Depixilate BEFORE computing quantiles if needed
        if return_depixilated:
            samples = self._depixilate_samples(samples, needs_age_dist=True)

        # Compute quantiles
        quantiles = compute_quantiles(samples, probs, axis=0)

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

        Contact rate R[c,d] represents the per-capita rate at which
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
        validate_alpha(alpha)
        self._validate_bootstrap()

        # Check cache
        cache_key = f"rate_alpha{alpha}_depix{return_depixilated}"

        if not force_recompute and cache_key in self._cache:
            return self._cache[cache_key]

        # Get bootstrap samples
        samples = self.sm._boot.rate_samples
        probs = get_probs_from_alpha(alpha)

        # Depixilate BEFORE computing quantiles if needed
        if return_depixilated:
            samples = self._depixilate_samples(samples, needs_age_dist=False)

        # Compute quantiles
        quantiles = compute_quantiles(samples, probs, axis=0)

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
        validate_alpha(alpha)
        self._validate_bootstrap()

        # Check cache
        needs_depix = return_depixilated and self.effective_age_bins != self.age_bins
        cache_key = f"mcint_alpha{alpha}_depix{needs_depix}"

        if not force_recompute and cache_key in self._cache:
            return self._cache[cache_key]

        # Get bootstrap intensity samples
        intensity_samples = self.sm._boot.intensity_samples
        probs = get_probs_from_alpha(alpha)

        # Handle depixilation if needed
        if needs_depix:
            # IMPORTANT: Depixilate FULL matrices first
            intensity_samples = self._depixilate_samples(
                intensity_samples, needs_age_dist=True
            )

        # Compute marginals by summing over contact age (last axis)
        marginal_samples = intensity_samples.sum(axis=-1)  # Shape: (n_boots, B)

        # Compute quantiles
        quantiles = compute_quantiles(marginal_samples, probs, axis=0)

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


class ModelSummariserPrem:
    """
    Statistical summariser for Prem model inference results.

    Computes quantiles and credible intervals for contact matrices from MCMC or SVI
    posterior samples, with proper handling of symmetrization and depixilation.

    Parameters
    ----------
    prem : Prem
        Fitted Prem model with MCMC or SVI results.
    df_age_dist : pd.DataFrame, optional
        Population age distribution dataframe with columns:
        - 'age': integer age (0 to max_age)
        - 'P': population count for each age
        Used for depixilation operations.
    df_age_grp_dist : pd.DataFrame, optional
        Age group-level population distribution.

    Attributes
    ----------
    prem : Prem
        Reference to the Prem model
    age_bins : AgeBins
        Age bins used in the model
    age_dist : NDArray, optional
        Fine-grained (1-year) population distribution
    age_grp_dist : NDArray, optional
        Coarse-grained (age group) population distribution
    post_samples : Dict[str, NDArray]
        Posterior samples from MCMC or SVI
    post_cint_samples : NDArray
        Posterior contact intensity samples (exponential of log_cint)

    Examples
    --------
    >>> prem = Prem(df_part, df_cnt, age_bins)
    >>> prem.run_inference_mcmc(rng_key, num_samples=1000)
    >>>
    >>> summariser = ModelSummariserPrem(prem, df_age_dist)
    >>>
    >>> # Get 95% credible intervals for contact intensity
    >>> summary = summariser.summarise_cint(alpha=0.05)
    >>> lower, median, upper = summary['lower'], summary['median'], summary['upper']
    >>>
    >>> # Get symmetrized and depixilated results
    >>> summary_full = summariser.summarise_cint(
    ...     alpha=0.05,
    ...     return_symmetrized=True,
    ...     return_depixilated=True
    ... )
    """

    def __init__(
        self,
        prem: Prem,
        df_age_dist: Optional[pd.DataFrame] = None,
        df_age_grp_dist: Optional[pd.DataFrame] = None,
        num_samples: int = 3000,
    ) -> None:
        """
        Initialize summariser with a Prem model.

        Parameters
        ----------
        prem : Prem
            Prem model with completed MCMC or SVI inference.
        df_age_dist : pd.DataFrame, optional
            Fine-grained age distribution for depixilation.
        df_age_grp_dist : pd.DataFrame, optional
            Age group-level population distribution.
        num_samples : int, default=3000
            Number of posterior samples to draw if using SVI.

        Raises
        ------
        ValueError
            If neither MCMC nor SVI has been run on the model.
            If model has not been properly initialized.
        """
        # Validate that either MCMC or SVI has been run
        has_mcmc = prem._mcmc_result is not None
        has_svi = prem._svi_result is not None
        if not (has_mcmc or has_svi):
            raise ValueError(
                "Either MCMC or SVI must have been run on the model. "
                "Call prem.run_inference_mcmc() or prem.run_inference_svi() first."
            )

        # Validate model data is loaded
        if prem.data is None:
            raise ValueError("Prem model data not initialized")

        # Store reference to model
        self.prem = prem

        # Reference key attributes
        self.age_bins = prem.age_bins
        self.df_age_dist = df_age_dist
        self.df_age_grp_dist = df_age_grp_dist
        self.num_samples = num_samples

        # Initialize helper classes
        self.validator = InputValidator()
        self.age_processor = AgeBinProcessor(self.age_bins)

        # Computed attributes (initialized in pipeline)
        self.age_dist: Optional[NDArray] = None
        self.age_grp_dist: Optional[NDArray] = None
        self.post_samples: Optional[Dict[str, NDArray]] = None
        self.post_cint_samples: Optional[NDArray] = None

        # Simple cache: {cache_key: result_dict}
        self._cache: Dict[str, Dict[str, NDArray]] = {}

        # Run processing pipeline
        self._validate()
        self._preprocess()
        self._load()

        # Derive age_grp_dist from age_dist if not provided
        if (
            self.age_grp_dist is None
            and self.age_bins is not None
            and self.age_dist is not None
        ):
            self.age_grp_dist = self._compute_age_grp_dist()

    def _validate(self) -> None:
        """Validate age distribution data if provided."""
        if self.df_age_dist is not None:
            self.validator.validate_age_distribution(self.df_age_dist)

    def _preprocess(self) -> None:
        """Preprocess age distribution data."""
        if self.df_age_dist is not None:
            # Assign age groups to population if needed
            has_age = "age" in self.df_age_dist.columns
            has_age_grp = "age_grp" in self.df_age_dist.columns
            if not has_age_grp and has_age:
                self.df_age_dist = self.age_processor.assign_age_groups(
                    self.df_age_dist, "age", "age_grp"
                )

    def _load(self) -> None:
        """Load age distributions and posterior samples."""
        # Load fine age distribution
        if self.df_age_dist is not None:
            self.age_dist = self.df_age_dist.sort_values("age")["P"].values

        # Load age group distribution
        if self.df_age_grp_dist is not None:
            self.age_grp_dist = self.df_age_grp_dist["P"].values
        elif self.df_age_dist is not None:
            # Aggregate age distribution to age groups
            age_grp_dist = (
                self.df_age_dist.groupby("age_grp", observed=True)["P"]
                .sum()
                .reset_index()
            )
            self.age_grp_dist = age_grp_dist["P"].values
        else:
            warnings.warn(
                "Age distribution not provided. "
                "Symmetrization and depixilation will not be possible.",
                UserWarning,
            )

        # Load posterior samples
        if self.prem._mcmc_result is not None:
            self.post_samples = self.prem._mcmc_result.get_samples()
        elif self.prem._svi_result is not None:
            # For SVI, generate samples from variational posterior
            self.post_samples = self.prem.posterior_predictive_svi(
                PRNGKey(0), num_samples=self.num_samples
            )

        # Compute contact intensity samples from log_cint
        if self.post_samples is not None and "log_cint" in self.post_samples:
            self.post_cint_samples = np.exp(self.post_samples["log_cint"])
        else:
            raise ValueError("Posterior samples must contain 'log_cint' field")

    def _compute_age_grp_dist(self) -> NDArray:
        """Compute age group distribution from fine-grained age distribution."""
        age_grp_dist = []
        age_edges = self.age_bins.left + [self.age_bins.max + 1]

        for i in range(len(age_edges) - 1):
            start_age = int(age_edges[i])
            end_age = int(age_edges[i + 1])
            age_grp_dist.append(self.age_dist[start_age:end_age].sum())

        return np.array(age_grp_dist)

    @staticmethod
    def symmetrize_cint_samples(
        cint_samples: NDArray, age_grp_dist: NDArray
    ) -> NDArray:
        """
        Symmetrize contact intensity matrix using reciprocity adjustment.

        Applies the reciprocity constraint to ensure that the expected number
        of contacts is balanced across age groups, weighted by population size:

        M_sym[c,d] = 0.5 * (M[c,d] + P[d]/P[c] * M[d,c])

        Parameters
        ----------
        cint_samples : NDArray
            Contact intensity samples, shape (n_samples, B, B) where B is
            number of age groups.
        age_grp_dist : NDArray
            Population distribution by age group, shape (B,).

        Returns
        -------
        NDArray
            Symmetrized contact intensity samples, same shape as input.

        Notes
        -----
        The reciprocity adjustment ensures that the total number of contacts
        from group c to group d equals the total from d to c when weighted
        by population sizes: P[c] * M[c,d] = P[d] * M[d,c]

        This is mathematically equivalent to:
        M_sym = 0.5 * (M + P_inv @ M.T @ P)
        where P = diag(age_grp_dist).
        """
        # Validate inputs
        if cint_samples.ndim != 3:
            raise ValueError(
                f"cint_samples must be 3D (n_samples, B, B), got shape {cint_samples.shape}"
            )

        n_age_groups = cint_samples.shape[1]
        if len(age_grp_dist) != n_age_groups:
            raise ValueError(
                f"age_grp_dist length ({len(age_grp_dist)}) must match "
                f"number of age groups ({n_age_groups})"
            )

        if np.any(age_grp_dist <= 0):
            raise ValueError("age_grp_dist must contain positive values")

        # Symmetrize using reciprocity
        M = cint_samples
        P = np.diag(age_grp_dist)[np.newaxis, ...]
        P_inv = np.diag(1 / age_grp_dist)[np.newaxis, ...]

        return 0.5 * (M + P_inv @ np.transpose(M, (0, 2, 1)) @ P)

    def _depixilate_samples(self, samples: NDArray, needs_age_dist: bool) -> NDArray:
        """
        Depixilate posterior samples efficiently.

        CRITICAL: This must be done BEFORE computing quantiles, because
        depixilation is a nonlinear transformation that doesn't commute
        with quantile operations.

        Parameters
        ----------
        samples : NDArray
            Posterior samples at age group resolution, shape (n_samples, B, B)
            where B is the number of age groups.
        needs_age_dist : bool
            Whether depixilate requires age_dist parameter.
            - True for intensity (requires population weighting)
            - False for rate (no population weighting needed)

        Returns
        -------
        depix_samples : NDArray
            Depixilated samples at 1-year age resolution,
            shape (n_samples, A, A) where A is the max age.

        Raises
        ------
        ValueError
            If age_bins or age_dist not available when required.

        Notes
        -----
        For computational efficiency:
        1. Preallocate the output array
        2. Use vectorized operations where possible
        3. Depixilate each sample independently to avoid memory issues
        """
        if self.age_bins is None:
            raise ValueError("age_bins must be provided for depixilation")

        if needs_age_dist and self.age_dist is None:
            raise ValueError("age_dist must be provided for intensity depixilation")

        n_samples = samples.shape[0]
        A = self.age_bins.range

        # Preallocate output
        depix_samples = np.empty((n_samples, A, A), dtype=np.float64)

        # Depixilate each sample
        for i in range(n_samples):
            if needs_age_dist:
                depix_samples[i] = depixilate(samples[i], self.age_bins, self.age_dist)
            else:
                depix_samples[i] = depixilate(samples[i], self.age_bins)

        return depix_samples

    def summarise_cint(
        self,
        alpha: float = 0.05,
        return_symmetrized: bool = False,
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
            Significance level for credible intervals (e.g., 0.05 for 95% CI).
        return_symmetrized : bool, default=False
            If True, apply reciprocity adjustment to enforce demographic symmetry.
            Requires age_grp_dist to be available.
        return_depixilated : bool, default=False
            If True, return results at 1-year age resolution instead of age groups.
            Requires age_bins and age_dist to be available.
        force_recompute : bool, default=False
            Force recomputation even if cached.

        Returns
        -------
        summary : Dict[str, NDArray]
            Dictionary containing:
            - 'lower': Lower credible bound, shape (B, B) or (A, A)
            - 'median': Median estimate, shape (B, B) or (A, A)
            - 'upper': Upper credible bound, shape (B, B) or (A, A)
            - 'alpha': Significance level used

        Raises
        ------
        ValueError
            If alpha not in (0, 1), or required data not available for
            symmetrization/depixilation.

        Examples
        --------
        >>> summary = summariser.summarise_cint(alpha=0.05)
        >>> ci_95_lower = summary['lower']
        >>> ci_95_upper = summary['upper']
        >>> median_estimate = summary['median']
        >>>
        >>> # Get symmetrized and depixilated results
        >>> summary_full = summariser.summarise_cint(
        ...     alpha=0.05,
        ...     return_symmetrized=True,
        ...     return_depixilated=True
        ... )

        Notes
        -----
        Order of operations:
        1. Symmetrization (if requested)
        2. Depixilation (if requested)
        3. Quantile computation

        This order is critical because depixilation and quantiles don't commute.
        """
        validate_alpha(alpha)
        probs = get_probs_from_alpha(alpha)

        # Check cache
        cache_key = (
            f"cint_alpha{alpha}_sym{return_symmetrized}_depix{return_depixilated}"
        )
        if not force_recompute and cache_key in self._cache:
            return self._cache[cache_key]

        # Validate requirements for symmetrization
        if return_symmetrized and self.age_grp_dist is None:
            raise ValueError(
                "Age group distribution required for symmetrization. "
                "Provide df_age_dist or df_age_grp_dist to constructor."
            )

        # Validate requirements for depixilation
        if return_depixilated:
            if self.age_bins is None:
                raise ValueError("age_bins required for depixilation")
            if self.age_dist is None:
                raise ValueError(
                    "Fine-grained age distribution required for depixilation. "
                    "Provide df_age_dist to constructor."
                )

        # Start with posterior samples
        samples = self.post_cint_samples.copy()

        # Apply symmetrization if requested
        if return_symmetrized:
            samples = self.symmetrize_cint_samples(samples, self.age_grp_dist)

        # Apply depixilation if requested
        if return_depixilated:
            samples = self._depixilate_samples(samples, needs_age_dist=True)

        # Compute quantiles
        quantiles = compute_quantiles(samples, probs, axis=0)

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
        return_symmetrized: bool = False,
        return_depixilated: bool = False,
        force_recompute: bool = False,
    ) -> Dict[str, NDArray]:
        """
        Compute summary statistics for contact rate matrix.

        Contact rate R[c,d] represents the per-capita rate at which
        individuals in age group c contact individuals in age group d.
        Computed as: R[c,d] = M[c,d] / P[d]

        Parameters
        ----------
        alpha : float, default=0.05
            Significance level for credible intervals.
        return_symmetrized : bool, default=False
            If True, symmetrize before computing rates.
        return_depixilated : bool, default=False
            If True, return at 1-year age resolution.
        force_recompute : bool, default=False
            Force recomputation even if cached.

        Returns
        -------
        summary : Dict[str, NDArray]
            Dictionary containing:
            - 'lower': Lower credible bound
            - 'median': Median estimate
            - 'upper': Upper credible bound
            - 'alpha': Significance level used

        Raises
        ------
        ValueError
            If age_grp_dist not available (required for rate computation).

        Examples
        --------
        >>> summary = summariser.summarise_rate(alpha=0.05)
        >>> ci_95_lower = summary['lower']
        >>> ci_95_upper = summary['upper']
        >>> median_estimate = summary['median']

        Notes
        -----
        Rates are computed by dividing intensity by contacted population.
        This transformation is applied AFTER symmetrization/depixilation
        to ensure proper handling of population weights.
        """
        validate_alpha(alpha)
        probs = get_probs_from_alpha(alpha)

        # Check cache
        cache_key = (
            f"rate_alpha{alpha}_sym{return_symmetrized}_depix{return_depixilated}"
        )
        if not force_recompute and cache_key in self._cache:
            return self._cache[cache_key]

        # Validate age_grp_dist is available
        if self.age_grp_dist is None:
            raise ValueError(
                "Age group distribution required for rate computation. "
                "Provide df_age_dist or df_age_grp_dist to constructor."
            )

        # Validate requirements for symmetrization
        if return_symmetrized and self.age_grp_dist is None:
            raise ValueError("Age group distribution required for symmetrization")

        # Validate requirements for depixilation
        if return_depixilated:
            if self.age_bins is None:
                raise ValueError("age_bins required for depixilation")
            if self.age_dist is None:
                raise ValueError(
                    "Fine-grained age distribution required for depixilation"
                )

        # Start with posterior intensity samples
        samples = self.post_cint_samples.copy()

        # Apply symmetrization if requested (before converting to rate)
        if return_symmetrized:
            samples = self.symmetrize_cint_samples(samples, self.age_grp_dist)

        # Apply depixilation if requested (before converting to rate)
        if return_depixilated:
            samples = self._depixilate_samples(samples, needs_age_dist=True)
            # Use fine-grained age distribution for rate computation
            pop_dist = self.age_dist
        else:
            # Use age group distribution
            pop_dist = self.age_grp_dist

        # Convert intensity to rate: R[c,d] = M[c,d] / P[d]
        # Broadcasting: samples is (n_samples, B, B), pop_dist is (B,)
        rate_samples = samples / pop_dist[np.newaxis, np.newaxis, :]

        # Compute quantiles
        quantiles = compute_quantiles(rate_samples, probs, axis=0)

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
        return_symmetrized: bool = False,
        return_depixilated: bool = False,
        force_recompute: bool = False,
    ) -> Dict[str, NDArray]:
        """
        Compute summary statistics for marginal contact intensity.

        Marginal contact intensity m[c] = Σ_d M[c,d] represents the total
        average number of contacts made by individuals in age group c
        across all age groups.

        Parameters
        ----------
        alpha : float, default=0.05
            Significance level for credible intervals.
        return_symmetrized : bool, default=False
            If True, symmetrize before computing marginals.
        return_depixilated : bool, default=False
            If True, return at 1-year age resolution.
        force_recompute : bool, default=False
            Force recomputation even if cached.

        Returns
        -------
        summary : Dict[str, NDArray]
            Dictionary containing:
            - 'lower': Lower credible bound, shape (B,) or (A,)
            - 'median': Median estimate, shape (B,) or (A,)
            - 'upper': Upper credible bound, shape (B,) or (A,)
            - 'alpha': Significance level used

        Examples
        --------
        >>> summary = summariser.summarise_mcint(alpha=0.05)
        >>> ci_95_lower = summary['lower']
        >>> ci_95_upper = summary['upper']
        >>> median_estimate = summary['median']

        Notes
        -----
        Marginal intensity is computed by summing the intensity matrix
        over the contact age dimension. When depixilation is requested:
        1. Depixilate each full intensity matrix sample
        2. Compute marginals from depixilated matrices
        3. Then compute quantiles

        This ordering is critical because marginals and depixilation
        don't commute in general.
        """
        validate_alpha(alpha)
        probs = get_probs_from_alpha(alpha)

        # Check cache
        cache_key = (
            f"mcint_alpha{alpha}_sym{return_symmetrized}_depix{return_depixilated}"
        )
        if not force_recompute and cache_key in self._cache:
            return self._cache[cache_key]

        # Validate requirements
        if return_symmetrized and self.age_grp_dist is None:
            raise ValueError("Age group distribution required for symmetrization")

        if return_depixilated:
            if self.age_bins is None:
                raise ValueError("age_bins required for depixilation")
            if self.age_dist is None:
                raise ValueError(
                    "Fine-grained age distribution required for depixilation"
                )

        # Start with posterior samples
        samples = self.post_cint_samples.copy()

        # Apply symmetrization if requested
        if return_symmetrized:
            samples = self.symmetrize_cint_samples(samples, self.age_grp_dist)

        # Apply depixilation if requested
        if return_depixilated:
            samples = self._depixilate_samples(samples, needs_age_dist=True)

        # Compute marginals by summing over contact age (last axis)
        mcint_samples = samples.sum(axis=-1)  # Shape: (n_samples, B) or (n_samples, A)

        # Compute quantiles
        quantiles = compute_quantiles(mcint_samples, probs, axis=0)

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
        self,
        return_symmetrized: bool = False,
        return_depixilated: bool = False,
    ) -> Dict[str, Dict[str, NDArray]]:
        """
        Get point estimates (mean and std) for all statistics.

        Parameters
        ----------
        return_symmetrized : bool, default=False
            Whether to apply symmetrization.
        return_depixilated : bool, default=False
            Whether to return depixilated results.

        Returns
        -------
        estimates : Dict[str, Dict[str, NDArray]]
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
        # Get samples (potentially symmetrized/depixilated)
        samples = self.post_cint_samples.copy()

        if return_symmetrized:
            if self.age_grp_dist is None:
                raise ValueError("Age group distribution required for symmetrization")
            samples = self.symmetrize_cint_samples(samples, self.age_grp_dist)

        if return_depixilated:
            if self.age_bins is None or self.age_dist is None:
                raise ValueError("age_bins and age_dist required for depixilation")
            cint_samples = self._depixilate_samples(samples, needs_age_dist=True)
            pop_dist = self.age_dist
        else:
            cint_samples = samples
            pop_dist = self.age_grp_dist

        # Compute rate samples
        if pop_dist is not None:
            rate_samples = cint_samples / pop_dist[np.newaxis, np.newaxis, :]
        else:
            rate_samples = None

        # Compute marginals
        mcint_samples = cint_samples.sum(axis=2)

        # Prepare results
        result = {
            "cint": {
                "mean": cint_samples.mean(axis=0),
                "std": cint_samples.std(axis=0, ddof=1),
            },
        }

        if rate_samples is not None:
            result["rate"] = {
                "mean": rate_samples.mean(axis=0),
                "std": rate_samples.std(axis=0, ddof=1),
            }

        result["mcint"] = {
            "mean": mcint_samples.mean(axis=0),
            "std": mcint_samples.std(axis=0, ddof=1),
        }

        return result

    def clear_cache(self) -> None:
        """Clear all cached computations."""
        self._cache.clear()

    def get_cache_info(self) -> Dict[str, Any]:
        """
        Get information about cached results.

        Returns
        -------
        info : Dict[str, Any]
            Dictionary with cache statistics including number of cached items
            and their keys.
        """
        return {
            "n_cached": len(self._cache),
            "cache_keys": list(self._cache.keys()),
        }
