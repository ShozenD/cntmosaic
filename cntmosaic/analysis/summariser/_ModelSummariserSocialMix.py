import warnings
from typing import Dict, Optional, Tuple, Union

import numpy as np
from numpy.typing import NDArray

from ...models import SocialMix
from ...utils import AgeBins, depixilate


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


class ModelSummariserSocialMix:
    """
    Statistical summariser for SocialMix bootstrap results.

    Computes quantiles and confidence intervals for contact matrices
    from bootstrap samples, with proper handling of adaptive age binning
    and stratification.

    Parameters
    ----------
    sm : SocialMix
        Fitted SocialMix model with bootstrap results

    Attributes
    ----------
    sm : SocialMix
        Reference to the SocialMix model
    age_bins : AgeBins
        Age bins used in the model
    pop_data : PopulationData or None
        Population data container from the model
    age_dist : NDArray or None
        Population age distribution for depixilation

    Examples
    --------
    >>> # Unstratified model
    >>> sm = SocialMix(part_data, cnt_data, age_bins, pop_data)
    >>> sm.run_inference_bootstrap(n_boot=1000, random_state=42)
    >>>
    >>> summariser = ModelSummariserSocialMix(sm)
    >>>
    >>> # Get 95% confidence intervals for contact intensity
    >>> summary = summariser.summarise_cint(alpha=0.05)
    >>> # For unstratified: summary is NDArray of shape (3, C, D)
    >>> lower, median, upper = summary[0], summary[1], summary[2]
    >>>
    >>> # For stratified: summary is Dict[str, NDArray]
    >>> lower_MF = summary['M->F'][0]  # Lower bound for M->F
    >>> median_MF = summary['M->F'][1]  # Median for M->F
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
            If model has not been fitted
        """
        # Validate model has been fitted
        if sm.Y is None:
            raise ValueError("SocialMix model has not been fitted")

        # Store reference to model
        self.sm = sm

        # Reference key attributes
        self.age_bins = sm.age_bins
        self.pop_data = sm.pop_data

        # Extract population age distribution if available
        if self.pop_data is not None:
            # Get fine-grained (1-year) age distribution
            age_dist_df = self.pop_data.get_age_distribution(by_group=False)
            self.age_dist = age_dist_df.values
        else:
            self.age_dist = None

        # Simple cache: {cache_key: result}
        self._cache: Dict[str, Union[NDArray, Dict[str, NDArray]]] = {}

    def _validate_bootstrap(self) -> None:
        """Validate that bootstrap has been run."""
        if self.sm._boot is None:
            raise ValueError(
                "Bootstrap has not been run. Call sm.run_inference_bootstrap() first."
            )

    def _is_stratified(self) -> bool:
        """Check if model uses stratification."""
        if self.sm.strat_mode != "single":
            return True
        else:
            return False

    def _stack_samples(self, samples_list) -> Union[NDArray, Dict[str, NDArray]]:
        """
        Stack bootstrap samples from List[Dict] to NDArray or Dict[str, NDArray].

        Parameters
        ----------
        samples_list : List[Dict[str, NDArray]]
            List of dictionaries from bootstrap iterations

        Returns
        -------
        Union[NDArray, Dict[str, NDArray]]
            For unstratified: NDArray of shape (n_boot, C, D)
            For stratified: Dict mapping labels to NDArray of shape (n_boot, C, D)
        """
        # Get keys from first sample
        keys = list(samples_list[0].keys())

        # If single "All->All" key, return as plain array
        if len(keys) == 1 and keys[0] == "All->All":
            return np.stack([s["All->All"] for s in samples_list])

        # Otherwise return as dict
        stacked = {}
        for key in keys:
            stacked[key] = np.stack([s[key] for s in samples_list])

        return stacked

    def _depixilate_unstratified(self, samples: NDArray, use_age_dist: bool) -> NDArray:
        """Depixilate unstratified bootstrap samples."""
        n_boot = samples.shape[0]
        A = self.age_bins.range
        depix_samples = np.empty((n_boot, A, A), dtype=np.float64)

        for i in range(n_boot):
            if use_age_dist:
                depix_samples[i] = depixilate(samples[i], self.age_bins, self.age_dist)
            else:
                depix_samples[i] = depixilate(samples[i], self.age_bins)

        return depix_samples

    def _depixilate_stratified(
        self, samples_dict: Dict[str, NDArray], use_age_dist: bool
    ) -> Dict[str, NDArray]:
        """Depixilate stratified bootstrap samples."""
        A = self.age_bins.range
        depix_samples = {}

        for label, samples in samples_dict.items():
            n_boot = samples.shape[0]
            depix_label = np.empty((n_boot, A, A), dtype=np.float64)

            for i in range(n_boot):
                if use_age_dist:
                    depix_label[i] = depixilate(
                        samples[i], self.age_bins, self.age_dist
                    )
                else:
                    depix_label[i] = depixilate(samples[i], self.age_bins)

            depix_samples[label] = depix_label

        return depix_samples

    def _compute_quantiles_unstratified(
        self, samples: NDArray, probs: Tuple[float, ...]
    ) -> NDArray:
        """Compute quantiles for unstratified samples."""
        return compute_quantiles(samples, probs, axis=0)

    def _compute_quantiles_stratified(
        self, samples_dict: Dict[str, NDArray], probs: Tuple[float, ...]
    ) -> Dict[str, NDArray]:
        """Compute quantiles for stratified samples."""
        quantiles = {}
        for label, samples in samples_dict.items():
            quantiles[label] = compute_quantiles(samples, probs, axis=0)
        return quantiles

    def summarise_cint(
        self,
        alpha: float = 0.05,
        return_depixilated: bool = False,
        force_recompute: bool = False,
    ) -> Union[NDArray, Dict[str, NDArray]]:
        """
        Compute summary statistics for contact intensity matrix.

        Contact intensity M[c,d] represents the average number of contacts
        that individuals in age group c have with individuals in age group d.

        Parameters
        ----------
        alpha : float, default=0.05
            Significance level for confidence intervals (e.g., 0.05 for 95% CI)
        return_depixilated : bool, default=False
            If True and adaptive merging occurred, return results in original
            age bins. If False or no merging, return in effective bins.
        force_recompute : bool, default=False
            Force recomputation even if cached

        Returns
        -------
        Union[NDArray, Dict[str, NDArray]]
            For unstratified: NDArray of shape (3, C, D) with [lower, median, upper]
            For stratified: Dict mapping stratum labels to NDArray of shape (3, C, D)

        Examples
        --------
        >>> # Unstratified
        >>> summary = summariser.summarise_cint(alpha=0.05)
        >>> lower, median, upper = summary[0], summary[1], summary[2]
        >>>
        >>> # Stratified
        >>> summary = summariser.summarise_cint(alpha=0.05)
        >>> lower_MF = summary['M->F'][0]
        >>> median_MF = summary['M->F'][1]
        """
        validate_alpha(alpha)
        self._validate_bootstrap()

        # Check cache
        cache_key = f"cint_alpha{alpha}_depix{return_depixilated}"
        if not force_recompute and cache_key in self._cache:
            return self._cache[cache_key]

        # Stack bootstrap samples
        samples = self._stack_samples(self.sm._boot.cint_samples)
        probs = get_probs_from_alpha(alpha)

        # Handle depixilation if needed
        if return_depixilated:
            if self.age_dist is None:
                raise ValueError(
                    "Population data required for depixilation. "
                    "Provide pop_data when initializing SocialMix."
                )

            if isinstance(samples, dict):
                samples = self._depixilate_stratified(samples, use_age_dist=True)
            else:
                samples = self._depixilate_unstratified(samples, use_age_dist=True)

        # Compute quantiles
        if isinstance(samples, dict):
            result = self._compute_quantiles_stratified(samples, probs)
        else:
            result = self._compute_quantiles_unstratified(samples, probs)

        # Cache and return
        self._cache[cache_key] = result
        return result

    def summarise_rate(
        self,
        alpha: float = 0.05,
        return_depixilated: bool = False,
        force_recompute: bool = False,
    ) -> Union[NDArray, Dict[str, NDArray]]:
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
        Union[NDArray, Dict[str, NDArray]]
            For unstratified: NDArray of shape (3, C, D) with [lower, median, upper]
            For stratified: Dict mapping stratum labels to NDArray of shape (3, C, D)
        """
        validate_alpha(alpha)
        self._validate_bootstrap()

        # Check cache
        cache_key = f"rate_alpha{alpha}_depix{return_depixilated}"
        if not force_recompute and cache_key in self._cache:
            return self._cache[cache_key]

        # Stack bootstrap samples
        samples = self._stack_samples(self.sm._boot.rate_samples)
        probs = get_probs_from_alpha(alpha)

        # Handle depixilation if needed
        if return_depixilated:
            if isinstance(samples, dict):
                samples = self._depixilate_stratified(samples, use_age_dist=False)
            else:
                samples = self._depixilate_unstratified(samples, use_age_dist=False)

        # Compute quantiles
        if isinstance(samples, dict):
            result = self._compute_quantiles_stratified(samples, probs)
        else:
            result = self._compute_quantiles_unstratified(samples, probs)

        # Cache and return
        self._cache[cache_key] = result
        return result

    def summarise_mcint(
        self,
        alpha: float = 0.05,
        return_depixilated: bool = False,
        force_recompute: bool = False,
    ) -> Union[NDArray, Dict[str, NDArray]]:
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
        Union[NDArray, Dict[str, NDArray]]
            For unstratified: NDArray of shape (3, C) with [lower, median, upper]
            For stratified: Dict mapping stratum labels to NDArray of shape (3, C)

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
        cache_key = f"mcint_alpha{alpha}_depix{return_depixilated}"
        if not force_recompute and cache_key in self._cache:
            return self._cache[cache_key]

        # Stack bootstrap samples
        samples = self._stack_samples(self.sm._boot.cint_samples)
        probs = get_probs_from_alpha(alpha)

        # Handle depixilation if needed
        if return_depixilated:
            if self.age_dist is None:
                raise ValueError(
                    "Population data required for depixilation. "
                    "Provide pop_data when initializing SocialMix."
                )

            if isinstance(samples, dict):
                samples = self._depixilate_stratified(samples, use_age_dist=True)
            else:
                samples = self._depixilate_unstratified(samples, use_age_dist=True)

        # Compute marginals by summing over contact age (last axis)
        if isinstance(samples, dict):
            marginal_samples = {label: s.sum(axis=-1) for label, s in samples.items()}
            result = self._compute_quantiles_stratified(marginal_samples, probs)
        else:
            marginal_samples = samples.sum(axis=-1)
            result = self._compute_quantiles_unstratified(marginal_samples, probs)

        # Cache and return
        self._cache[cache_key] = result
        return result

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

    def _validate(self) -> None:
        """Validate model and bootstrap state."""
        self._validate_bootstrap()
