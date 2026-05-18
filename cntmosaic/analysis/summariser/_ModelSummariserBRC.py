import warnings
from typing import Any, Dict, Literal, Optional, Tuple

import jax
import numpy as np
import pandas as pd
from jax.random import PRNGKey
from numpy.typing import NDArray

from ...dataloader.containers import PopulationData
from ...models import AgeMixCC, AgeMixFF, AgeMixFC, GenMixFF, GenMixFC
from ...models._vdKassteele import vdKassteele


def validate_alpha(alpha: float) -> None:
    """Validate alpha parameter."""
    if not 0 < alpha < 1:
        raise ValueError(f"alpha must be in (0, 1), got {alpha}")


def get_probs_from_alpha(alpha: float) -> Tuple[float, float]:
    """Convert alpha to (lower, upper) probabilities."""
    return (alpha / 2, 1 - alpha / 2)


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

    Unified summariser for AgeMixCC, AgeMixFF, AgeMixFC, GenMixFF, GenMixFC, and vdKassteele models.
    Computes quantiles and credible intervals for contact matrices from MCMC or SVI
    posterior samples, with automatic detection of inference method and model type.

    Parameters
    ----------
    model : AgeMixCC | AgeMixFF | AgeMixFC | GenMixFF | GenMixFC | vdKassteele
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
        Detected model type ("brc" for AgeMixFF/AgeMixFC, "hibrc" for GenMixFF/GenMixFC).
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
        If model is not a GenMix-family model.

    Examples
    --------
    >>> from cntmosaic.models import AgeMixFF
    >>> from cntmosaic.models.numpyro.priors import Spline2D
    >>> from cntmosaic.analysis import ModelSummariserBRC
    >>>
    >>> # Fit model with MCMC
    >>> model = AgeMixFF(dataloader, priors={"rate": Spline2D()})
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
    >>> model_svi = AgeMixFF(dataloader, priors={"rate": Spline2D()})
    >>> model_svi.run_inference_svi(PRNGKey(0), guide, num_steps=10000)
    >>> summariser_svi = ModelSummariserBRC(model_svi, num_samples=3000)
    >>>
    >>> # Works with hierarchical models too
    >>> gen_mix_ff = GenMixFF(dataloader, priors)
    >>> gen_mix_ff.run_inference_mcmc(PRNGKey(0), num_samples=1000)
    >>> summariser_hi = ModelSummariserBRC(gen_mix_ff)
    >>> summary_by_group = summariser_hi.summarise_cint(alpha=0.05)

    Notes
    -----
    - Automatically detects inference method (MCMC vs SVI)
    - Automatically detects model type (AgeMix vs GenMix)
    - Unified API across all GenMix-family models
    - Efficient memory management for large posterior samples
    - Results are cached to avoid redundant computation

    See Also
    --------
    ModelSummariserPrem : Summariser for Prem models
    ModelSummariserSocialMix : Summariser for SocialMix models
    """

    def __init__(
        self,
        model: "AgeMixCC | AgeMixFF | AgeMixFC | GenMixFF | GenMixFC",
        num_samples: int = 3000,
    ) -> None:
        """
        Initialize summariser with a BRC model.

        Parameters
        ----------
        model : AgeMixFF | AgeMixFC | GenMixFF | GenMixFC | vdKassteele
            GenMix-family or vdKassteele model with completed MCMC or SVI inference.
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
        from ...models import (
            AgeMixCC,
            AgeMixFF,
            AgeMixFC,
            GenMixFF,
            GenMixFC,
            vdKassteele,
        )

        if not isinstance(
            model, (AgeMixCC, AgeMixFF, AgeMixFC, GenMixFF, GenMixFC, vdKassteele)
        ):
            raise TypeError(
                f"Model must be a GenMix-family model (AgeMixCC, AgeMixFF, AgeMixFC, GenMixFF, GenMixFC) or "
                "vdKassteele, "
                f"got {type(model).__name__}"
            )

        # Detect inference method via ContactModel property
        _method = model.inference_method
        if _method is None:
            raise ValueError(
                "Neither MCMC nor SVI has been run on the model. "
                "Call model.run_inference_mcmc() or model.run_inference_svi() first."
            )

        # Store configuration
        self.model = model
        self.inference_method: Literal["mcmc", "svi"] = _method
        self.num_samples = num_samples
        # Backend name for dispatch (e.g. "numpyro")
        self._backend_name: str = model._get_backend().backend_name()

        # Detect model type
        # vdKassteele can be either BRC or HIBRC depending on prior_type
        if isinstance(model, vdKassteele):
            if model.prior_type == "global":
                self.model_type: Literal["brc", "hibrc"] = "brc"
            else:
                self.model_type: Literal["brc", "hibrc"] = "hibrc"
        elif isinstance(model, (GenMixFF, GenMixFC)):
            self.model_type: Literal["brc", "hibrc"] = "hibrc"
        else:
            self.model_type: Literal["brc", "hibrc"] = "brc"

        # Initialize cache
        self._cache: Dict[str, Dict[str, NDArray]] = {}

        # Load posterior samples and compute contact intensities
        self._load_posterior()
        self._compute_contact_intensities()

    def _load_posterior(self) -> None:
        """Load posterior samples, routing through the model's inference backend."""
        self.post_samples = self.model.draw_posterior_samples(
            PRNGKey(0), num_samples=self.num_samples
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
        full_labels = self.model.data.full_labels

        # Map flat_ix to flat_pixs for population stratification
        flat_ix_to_flat_pixs = {}
        flat_ix_array = self.model.data.flat_ix
        flat_pixs_array = self.model.data.flat_pixs
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
        full_labels = self.model.data.full_labels

        # Map flat_ix to flat_pixs for population stratification
        # For each flat_ix, find the corresponding flat_pixs value
        flat_ix_to_flat_pixs = {}
        flat_ix_array = self.model.data.flat_ix
        flat_pixs_array = self.model.data.flat_pixs
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
        measure: Literal["mean", "median"] = "median",
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

        cache_key = f"rate_{measure}_probs{probs}"
        if cache_key in self._cache:
            return self._cache[cache_key]

        if self.model_type == "brc":
            # Simple case: rate is directly available
            rate_samples = np.exp(self.post_samples["log_rate"])

            if measure == "mean":
                rate_central = rate_samples.mean(axis=0)
            if measure == "median":
                rate_central = np.median(rate_samples, axis=0)

            rate_summary = compute_quantiles(rate_samples, probs, axis=0)
            rate_summary = np.insert(rate_summary, 1, rate_central, axis=0)

            result = rate_summary

        else:  # hibrc
            # Compute rate for each stratum
            log_rate = self.post_samples["log_rate"]
            log_delta = self.post_samples["log_delta"]
            full_labels = self.model.data.full_labels
            flat_ix_values = np.unique(self.model.data.flat_ix)

            result = {}
            for flat_idx in flat_ix_values:
                label = full_labels[flat_idx]
                # rate = exp(log_rate + log_delta[flat_idx])
                log_rate_stratum = log_rate + log_delta[:, flat_idx, :, :]
                rate_samples = np.exp(log_rate_stratum)

                if measure == "mean":
                    rate_central = rate_samples.mean(axis=0)
                if measure == "median":
                    rate_central = np.median(rate_samples, axis=0)

                rate_summary = compute_quantiles(rate_samples, probs, axis=0)
                rate_summary = np.insert(rate_summary, 1, rate_central, axis=0)
                result[label] = rate_summary

        self._cache[cache_key] = result
        return result

    def summarise_cint(
        self,
        alpha: float = 0.05,
        measure: Literal["mean", "median"] = "median",
        probs: Optional[Tuple[float, ...]] = None,
    ) -> Dict[str, NDArray]:
        """
        Compute summary statistics for contact intensity matrix.

        Contact intensity M[c,d] represents the average number of contacts
        that individuals in age c have with individuals in age d.

        Parameters
        ----------
        alpha : float, default=0.05
            Significance level for credible intervals (e.g., 0.05 for 95% CI).
            Ignored if probs is provided.
        measure: Literal["mean", "median"] = "median",
            Summary measure of central tendency to use. Defaults to "median".

        probs : Tuple[float, ...], optional
            Specific quantile probabilities to compute.
            If None, uses (alpha/2, 0.5, 1-alpha/2).

        Returns
        -------
        Dict[str, NDArray]
            Dict[str, NDArray] with structure
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

        cache_key = f"cint_{measure}_probs{probs}"
        if cache_key in self._cache:
            return self._cache[cache_key]

        if self.model_type == "brc":
            # Simple case: direct quantile computation
            result = {}

            cint_summary = compute_quantiles(self.post_cint_samples, probs, axis=0)

            # Measures of central tendency
            if measure == "mean":
                cint_central = self.post_cint_samples.mean(axis=0)

            if measure == "median":
                cint_central = np.median(self.post_cint_samples, axis=0)

            cint_summary = np.insert(cint_summary, 1, cint_central, axis=0)
            result["All->All"] = cint_summary

        else:  # hibrc
            # Compute quantiles for each stratum
            result = {}
            for label, samples in self.post_cint_samples.items():
                cint_summary = compute_quantiles(samples, probs, axis=0)

                # Measures of central tendency
                if measure == "mean":
                    cint_central = samples.mean(axis=0)

                if measure == "median":
                    cint_central = np.median(samples, axis=0)

                cint_summary = np.insert(cint_summary, 1, cint_central, axis=0)
                result[label] = cint_summary

        self._cache[cache_key] = result
        return result

    def summarise_mcint(
        self,
        alpha: float = 0.05,
        measure: Literal["mean", "median"] = "median",
        probs: Optional[Tuple[float, ...]] = None,
    ) -> Dict[str, NDArray]:
        """
        Compute summary statistics for marginal contact intensity.

        Marginal contact intensity m[c] = Σ_d M[c,d] represents the total
        average number of contacts made by individuals in age c across all ages.

        Parameters
        ----------
        alpha : float, default=0.05
            Significance level for credible intervals (e.g., 0.05 for 95% CI).
            Ignored if probs is provided.
        measure: Literal["mean", "median"] = "median",
            Summary measure of central tendency to use. Defaults to "median".
        probs : Tuple[float, ...], optional
            Specific quantile probabilities to compute.
            If None, uses (alpha/2, 0.5, 1-alpha/2).

        Returns
        -------
        Dict[str, NDArray]
            - Dict[str, NDArray] with structure
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

        cache_key = f"mcint_{measure}_probs{probs}"
        if cache_key in self._cache:
            return self._cache[cache_key]

        result = {}
        if self.model_type == "brc":
            # Sum over contact age (last axis)
            mcint_samples = self.post_cint_samples.sum(axis=-1)

            mcint_summary = compute_quantiles(mcint_samples, probs, axis=0)

            if measure == "mean":
                mcint_central = mcint_samples.mean(axis=0)
            if measure == "median":
                mcint_central = np.median(mcint_samples, axis=0)

            mcint_summary = np.insert(mcint_summary, 1, mcint_central, axis=0)
            result["All->All"] = mcint_summary

        else:  # hibrc
            # Compute marginal for each stratum
            for label, samples in self.post_cint_samples.items():
                # Sum over contact age dimension
                mcint_samples = samples.sum(axis=-1)
                mcint_summary = compute_quantiles(mcint_samples, probs, axis=0)

                if measure == "mean":
                    mcint_central = mcint_samples.mean(axis=0)
                if measure == "median":
                    mcint_central = np.median(mcint_samples, axis=0)

                mcint_summary = np.insert(mcint_summary, 1, mcint_central, axis=0)
                result[label] = mcint_summary

        self._cache[cache_key] = result
        return result

    def get_posterior_samples(
        self,
        quantity: Literal["rate", "cint", "mcint", "delta"] = "cint",
    ) -> NDArray | Dict[str, NDArray]:
        """
        Get raw posterior samples for specified quantity.

        Useful for custom post-processing or plotting.

        Parameters
        ----------
        quantity : {"rate", "cint", "mcint", "delta"}, default="cint"
            Which quantity to return samples for:
            - "rate": Contact rate matrix r[a,b]
            - "cint": Contact intensity matrix m[a,b]
            - "mcint": Marginal contact intensity m[a] = Σ_b m[a,b]
            - "delta": Contact rate modifiers[a,b] (HiBRC only)

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
                full_labels = self.model.data.full_labels
                flat_ix_values = np.unique(self.model.data.flat_ix)

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
        elif quantity == "delta":
            if self.model_type == "hibrc":
                return np.exp(self.post_samples["log_delta"])

        else:
            raise ValueError(
                f"Unknown quantity: {quantity}. Must be 'rate', 'cint', 'mcint', or 'delta'"
            )

    def get_point_estimates(
        self,
        quantity: Literal["rate", "cint", "mcint", "delta"] = "cint",
    ) -> Dict[str, NDArray] | Dict[str, Dict[str, NDArray]]:
        """
        Get point estimates (mean and std) for specified quantity.

        Parameters
        ----------
        quantity : {"rate", "cint", "mcint", "delta"}, default="cint"
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
