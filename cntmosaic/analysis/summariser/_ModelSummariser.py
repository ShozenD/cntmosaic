from __future__ import annotations

from typing import Any, Dict, Literal, Optional, Tuple

import numpy as np
from jax.random import PRNGKey
from numpy.typing import NDArray

from .._stats import summarise_samples, validate_alpha
from ._summary import ContactSummary


def _get_probs_from_alpha(alpha: float) -> Tuple[float, float]:
    return (alpha / 2, 1 - alpha / 2)


def _normalize_cache_key(
    quantity: str,
    measure: str,
    probs: Tuple[float, ...],
) -> Tuple[str, str, Tuple[float, ...]]:
    return (quantity, measure, tuple(round(p, 10) for p in probs))


class ModelSummariser:
    """
    Statistical summariser for AgeMix and GenMix model inference results (MCMC or SVI).

    Unified summariser for AgeMixCC, AgeMixFF, AgeMixFC, GenMixFF, GenMixFC, and
    vdKassteele models. Computes credible intervals and posterior summaries for
    contact matrices, with automatic detection of the inference method and model type.

    Posterior loading and intensity computation are **lazy**: no samples are drawn
    until the first call to a ``summarise_*`` or ``get_posterior_samples`` method.

    Parameters
    ----------
    model : AgeMixCC | AgeMixFF | AgeMixFC | GenMixFF | GenMixFC | vdKassteele
        Fitted model with completed MCMC or SVI inference.
    num_samples : int, default=3000
        Number of posterior samples to draw when using SVI.
    rng_key : PRNGKey, optional
        JAX random key used to draw SVI posterior samples.
        Ignored for MCMC models (samples are stored deterministically).
        Defaults to ``PRNGKey(0)`` when not provided.

    Attributes
    ----------
    model : ContactModel
        Reference to the fitted model.
    inference_method : {"mcmc", "svi"}
        Detected inference method.
    model_type : {"agemix", "genmix"}
        Detected model type.
        ``"agemix"`` — AgeMixCC, AgeMixFF, AgeMixFC, unstratified vdKassteele.
        ``"genmix"`` — GenMixFF, GenMixFC, stratified vdKassteele.
    num_samples : int
        Number of SVI posterior samples.

    Raises
    ------
    ValueError
        If neither MCMC nor SVI has been run on the model.
    TypeError
        If *model* is not a supported model type.

    Examples
    --------
    >>> from cntmosaic.models import AgeMixFF
    >>> from cntmosaic.models.numpyro.priors import Spline2D
    >>> from cntmosaic.analysis import ModelSummariser
    >>>
    >>> model = AgeMixFF(dataloader, priors={"rate": Spline2D()})
    >>> model.run_inference_mcmc(PRNGKey(0), num_samples=1000)
    >>>
    >>> summariser = ModelSummariser(model)
    >>> summary = summariser.summarise_cint(alpha=0.05)
    >>> summary["All->All"].lower   # shape (A, A)
    >>> summary["All->All"].central
    >>> summary["All->All"].upper

    See Also
    --------
    ModelSummariserPrem : Summariser for Prem models.
    ModelSummariserSocialMix : Summariser for SocialMix models.
    """

    def __init__(
        self,
        model: "AgeMixCC | AgeMixFF | AgeMixFC | GenMixFF | GenMixFC | GenMixCC | vdKassteele",
        num_samples: int = 3000,
        rng_key: Optional[PRNGKey] = None,
    ) -> None:
        from ...models import AgeMixCC, AgeMixFF, AgeMixFC, GenMixFF, GenMixFC, GenMixCC
        from ...models._vdKassteele import vdKassteele

        if not isinstance(
            model, (AgeMixCC, AgeMixFF, AgeMixFC, GenMixFF, GenMixFC, GenMixCC, vdKassteele)
        ):
            raise TypeError(
                f"model must be one of AgeMixCC, AgeMixFF, AgeMixFC, GenMixFF, GenMixFC, "
                f"GenMixCC, or vdKassteele, got {type(model).__name__}"
            )

        _method = model.inference_method
        if _method is None:
            raise ValueError(
                "Neither MCMC nor SVI has been run on the model. "
                "Call model.run_inference_mcmc() or model.run_inference_svi() first."
            )

        self.model = model
        self.inference_method: Literal["mcmc", "svi"] = _method
        self.num_samples = num_samples
        self._rng_key: PRNGKey = rng_key if rng_key is not None else PRNGKey(0)

        # Detect model type
        if isinstance(model, vdKassteele):
            self.model_type: Literal["agemix", "genmix"] = (
                "agemix" if model.prior_type == "global" else "genmix"
            )
        elif isinstance(model, (GenMixFF, GenMixFC, GenMixCC)):
            self.model_type = "genmix"
        else:
            self.model_type = "agemix"

        # Lazy-loaded state
        self._cache: Dict[Tuple, Any] = {}
        self._post_samples: Optional[Dict[str, NDArray]] = None
        self._post_cint_samples: Optional[NDArray | Dict[str, NDArray]] = None
        self._post_rate_samples: Optional[NDArray | Dict[str, NDArray]] = None
        self._flat_ix_values: Optional[NDArray] = None
        self._flat_ix_to_flat_pixs: Optional[Dict[int, int]] = None

    # ------------------------------------------------------------------
    # Lazy loading
    # ------------------------------------------------------------------

    def _ensure_loaded(self) -> None:
        """Load posterior samples and compute intensities on first access."""
        if self._post_samples is not None:
            return

        self._post_samples = self.model.draw_posterior_samples(
            self._rng_key, num_samples=self.num_samples
        )

        from ...models._vdKassteele import vdKassteele

        if self.model_type == "genmix" and not isinstance(self.model, vdKassteele):
            self._flat_ix_values, self._flat_ix_to_flat_pixs = (
                self._build_flat_ix_map()
            )

        self._compute_contact_intensities()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_flat_ix_map(self) -> Tuple[NDArray, Dict[int, int]]:
        """Build flat_ix → flat_pixs mapping in O(n_obs) using np.unique."""
        flat_ix_array = self.model.data.flat_ix
        flat_pixs_array = self.model.data.flat_pixs
        unique_vals, first_occ = np.unique(flat_ix_array, return_index=True)
        mapping = dict(
            zip(unique_vals.tolist(), flat_pixs_array[first_occ].tolist())
        )
        return unique_vals, mapping

    def _compute_contact_intensities(self) -> None:
        from ...models._vdKassteele import vdKassteele

        if self.model_type == "agemix":
            if "log_cint" in self._post_samples:
                self._post_cint_samples = np.exp(self._post_samples["log_cint"])
                self._post_rate_samples = np.exp(self._post_samples["log_rate"])
            else:
                raise ValueError(
                    "Posterior samples must contain 'log_cint' for agemix models."
                )
        elif isinstance(self.model, vdKassteele):
            cint, rate = self._compute_vdkassteele_intensities()
            self._post_cint_samples = cint
            self._post_rate_samples = rate
        else:
            cint, rate = self._compute_genmix_intensities()
            self._post_cint_samples = cint
            self._post_rate_samples = rate

    def _compute_vdkassteele_intensities(
        self,
    ) -> Tuple[Dict[str, NDArray], Dict[str, NDArray]]:
        """Compute per-stratum contact intensities for vdKassteele models."""
        log_rate = np.asarray(self._post_samples["log_rate"], dtype=np.float32)
        log_P = np.asarray(self.model.log_P, dtype=np.float32)
        full_labels = self.model.data.full_labels
        flat_ix_array = self.model.data.flat_ix
        flat_pixs_array = self.model.data.flat_pixs

        unique_vals, first_occ = np.unique(flat_ix_array, return_index=True)
        flat_ix_to_pixs = dict(
            zip(unique_vals.tolist(), flat_pixs_array[first_occ].tolist())
        )

        post_cint: Dict[str, NDArray] = {}
        post_rate: Dict[str, NDArray] = {}

        for flat_idx in unique_vals:
            label = full_labels[flat_idx]
            log_rate_stratum = log_rate[:, flat_idx, :, :]
            log_P_vector = log_P[flat_ix_to_pixs[flat_idx], :]
            log_P_matrix = log_P_vector[np.newaxis, :]

            rate = np.exp(log_rate_stratum.astype(np.float32, copy=False))
            cint = np.exp(
                (log_rate_stratum + log_P_matrix).astype(np.float32, copy=False)
            )
            post_rate[label] = rate
            post_cint[label] = cint

        return post_cint, post_rate

    def _compute_genmix_intensities(
        self,
    ) -> Tuple[Dict[str, NDArray], Dict[str, NDArray]]:
        """Compute per-stratum contact intensities for GenMix models.

        Computes rate and cint together to avoid recomputing log_rate + log_delta.
        """
        log_rate_raw = np.asarray(self._post_samples["log_rate"], dtype=np.float32)
        log_delta = np.asarray(self._post_samples["log_delta"], dtype=np.float32)
        log_P = np.asarray(self.model.log_P, dtype=np.float32)
        full_labels = self.model.data.full_labels

        # Hoist reshape above the loop — same result every iteration
        A = log_delta.shape[2]
        log_rate = log_rate_raw.reshape(-1, A, A)

        post_cint: Dict[str, NDArray] = {}
        post_rate: Dict[str, NDArray] = {}

        for flat_idx in self._flat_ix_values:
            label = full_labels[flat_idx]
            log_delta_stratum = log_delta[:, flat_idx, :, :]
            log_P_vector = log_P[self._flat_ix_to_flat_pixs[flat_idx], :]
            log_P_matrix = log_P_vector[np.newaxis, :]  # (1, A)

            log_sum_rate = log_rate + log_delta_stratum
            log_sum_cint = log_sum_rate + log_P_matrix

            post_rate[label] = np.exp(log_sum_rate.astype(np.float32, copy=False))
            post_cint[label] = np.exp(log_sum_cint.astype(np.float32, copy=False))

        return post_cint, post_rate

    def _make_summary(
        self,
        samples: NDArray,
        probs: Tuple[float, ...],
        measure: str,
        alpha: float,
    ) -> ContactSummary:
        quantiles, central = summarise_samples(samples, probs, measure)
        return ContactSummary(
            lower=quantiles[0],
            central=central,
            upper=quantiles[-1],
            alpha=alpha,
            measure=measure,
            age_group_specs=getattr(self.model, "age_group_specs", None),
        )

    # ------------------------------------------------------------------
    # Public summarise methods
    # ------------------------------------------------------------------

    def summarise_rate(
        self,
        alpha: float = 0.05,
        measure: Literal["mean", "median"] = "median",
        probs: Optional[Tuple[float, ...]] = None,
    ) -> Dict[str, ContactSummary]:
        """Compute credible-interval summaries for the contact rate matrix.

        Contact rate R[c, d] is the per-capita rate at which individuals in
        age group c contact individuals in age group d.

        Parameters
        ----------
        alpha : float, default=0.05
            Significance level for credible intervals (e.g. 0.05 for 95 % CI).
            Ignored when *probs* is provided.
        measure : {"mean", "median"}, default="median"
            Central-tendency measure.
        probs : tuple of float, optional
            Custom lower/upper quantile probabilities. When provided, *alpha*
            is ignored. Must have exactly two elements for the ``lower`` and
            ``upper`` fields of :class:`ContactSummary`.

        Returns
        -------
        Dict[str, ContactSummary]
            One entry per stratum.
            For agemix models the single key is ``"All->All"``.
            For genmix models the keys are stratum labels such as ``"M_A->All"``.

        Raises
        ------
        ValueError
            If *alpha* is not in (0, 1), or if *measure* is invalid.
        """
        self._ensure_loaded()
        if probs is None:
            validate_alpha(alpha)
            probs = _get_probs_from_alpha(alpha)

        cache_key = _normalize_cache_key("rate", measure, probs)
        if cache_key in self._cache:
            return self._cache[cache_key]

        result: Dict[str, ContactSummary] = {}

        if self.model_type == "agemix":
            result["All->All"] = self._make_summary(
                self._post_rate_samples, probs, measure, alpha
            )
        else:
            for label, samples in self._post_rate_samples.items():
                result[label] = self._make_summary(samples, probs, measure, alpha)

        self._cache[cache_key] = result
        return result

    def summarise_cint(
        self,
        alpha: float = 0.05,
        measure: Literal["mean", "median"] = "median",
        probs: Optional[Tuple[float, ...]] = None,
    ) -> Dict[str, ContactSummary]:
        """Compute credible-interval summaries for the contact intensity matrix.

        Contact intensity M[c, d] is the average number of contacts that
        individuals in age group c have with individuals in age group d.

        Parameters
        ----------
        alpha : float, default=0.05
            Significance level for credible intervals.
        measure : {"mean", "median"}, default="median"
            Central-tendency measure.
        probs : tuple of float, optional
            Custom lower/upper quantile probabilities.

        Returns
        -------
        Dict[str, ContactSummary]
            One entry per stratum (key ``"All->All"`` for agemix models).
        """
        self._ensure_loaded()
        if probs is None:
            validate_alpha(alpha)
            probs = _get_probs_from_alpha(alpha)

        cache_key = _normalize_cache_key("cint", measure, probs)
        if cache_key in self._cache:
            return self._cache[cache_key]

        result: Dict[str, ContactSummary] = {}

        if self.model_type == "agemix":
            result["All->All"] = self._make_summary(
                self._post_cint_samples, probs, measure, alpha
            )
        else:
            for label, samples in self._post_cint_samples.items():
                result[label] = self._make_summary(samples, probs, measure, alpha)

        self._cache[cache_key] = result
        return result

    def summarise_mcint(
        self,
        alpha: float = 0.05,
        measure: Literal["mean", "median"] = "median",
        probs: Optional[Tuple[float, ...]] = None,
    ) -> Dict[str, ContactSummary]:
        """Compute credible-interval summaries for the marginal contact intensity.

        Marginal contact intensity m[c] = Σ_d M[c, d] is the total average
        number of contacts made by individuals in age group c.

        Parameters
        ----------
        alpha : float, default=0.05
            Significance level for credible intervals.
        measure : {"mean", "median"}, default="median"
            Central-tendency measure.
        probs : tuple of float, optional
            Custom lower/upper quantile probabilities.

        Returns
        -------
        Dict[str, ContactSummary]
            One entry per stratum; each ``ContactSummary`` has 1-D arrays of
            shape (A,) instead of (A, A).
        """
        self._ensure_loaded()
        if probs is None:
            validate_alpha(alpha)
            probs = _get_probs_from_alpha(alpha)

        cache_key = _normalize_cache_key("mcint", measure, probs)
        if cache_key in self._cache:
            return self._cache[cache_key]

        result: Dict[str, ContactSummary] = {}

        if self.model_type == "agemix":
            mcint_samples = self._post_cint_samples.sum(axis=-1)
            result["All->All"] = self._make_summary(
                mcint_samples, probs, measure, alpha
            )
        else:
            for label, samples in self._post_cint_samples.items():
                mcint_samples = samples.sum(axis=-1)
                result[label] = self._make_summary(
                    mcint_samples, probs, measure, alpha
                )

        self._cache[cache_key] = result
        return result

    # ------------------------------------------------------------------
    # Raw posterior access
    # ------------------------------------------------------------------

    def get_posterior_samples(
        self,
        quantity: Literal["rate", "cint", "mcint", "delta"] = "cint",
    ) -> NDArray | Dict[str, NDArray]:
        """Return raw posterior samples for the requested quantity.

        Parameters
        ----------
        quantity : {"rate", "cint", "mcint", "delta"}, default="cint"
            ``"rate"``  — contact rate matrix R[a, b]
            ``"cint"``  — contact intensity matrix M[a, b]
            ``"mcint"`` — marginal contact intensity m[a] = Σ_b M[a, b]
            ``"delta"`` — stratum-specific rate modifiers (genmix only)

        Returns
        -------
        NDArray or Dict[str, NDArray]
            For agemix models: NDArray of shape (n_samples, A, A) or (n_samples, A).
            For genmix models: Dict mapping stratum labels to NDArray.

        Raises
        ------
        ValueError
            If *quantity* is unknown, or if ``"delta"`` is requested from an
            agemix model.
        """
        self._ensure_loaded()

        cache_key = ("raw", quantity)
        if cache_key in self._cache:
            return self._cache[cache_key]

        if quantity == "rate":
            result = self._post_rate_samples

        elif quantity == "cint":
            result = self._post_cint_samples

        elif quantity == "mcint":
            if self.model_type == "agemix":
                result = self._post_cint_samples.sum(axis=-1)
            else:
                result = {
                    label: samples.sum(axis=-1)
                    for label, samples in self._post_cint_samples.items()
                }

        elif quantity == "delta":
            if self.model_type != "genmix":
                raise ValueError(
                    "'delta' posterior samples are only available for genmix models "
                    "(GenMixFF, GenMixFC, or stratified vdKassteele)."
                )
            result = np.exp(self._post_samples["log_delta"])

        else:
            raise ValueError(
                f"Unknown quantity {quantity!r}. "
                "Must be 'rate', 'cint', 'mcint', or 'delta'."
            )

        self._cache[cache_key] = result
        return result

    def get_point_estimates(
        self,
        quantity: Literal["rate", "cint", "mcint", "delta"] = "cint",
    ) -> Dict[str, NDArray] | Dict[str, Dict[str, NDArray]]:
        """Return posterior mean and standard deviation for the requested quantity.

        .. note::
            Posterior standard deviation assumes symmetry and may be misleading for
            right-skewed posteriors (e.g. contact intensity after the ``exp``
            transform). Consider using :meth:`summarise_cint` with asymmetric
            credible intervals for uncertainty quantification.

        Parameters
        ----------
        quantity : {"rate", "cint", "mcint", "delta"}, default="cint"
            Quantity to compute point estimates for.

        Returns
        -------
        Dict
            For agemix models: ``{"mean": NDArray, "std": NDArray}``.
            For genmix models: ``{stratum_label: {"mean": NDArray, "std": NDArray}}``.
        """
        samples = self.get_posterior_samples(quantity)

        if self.model_type == "agemix":
            return {
                "mean": samples.mean(axis=0),
                "std": samples.std(axis=0, ddof=1),
            }

        result: Dict[str, Dict[str, NDArray]] = {}
        for label, label_samples in samples.items():
            result[label] = {
                "mean": label_samples.mean(axis=0),
                "std": label_samples.std(axis=0, ddof=1),
            }
        return result

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def strata(self) -> list[str]:
        """Ordered list of stratum labels.

        Returns ``["All->All"]`` for agemix models, or the actual stratum
        labels (e.g. ``["M_A->All", "F_B->All"]``) for genmix models.
        """
        self._ensure_loaded()
        if self.model_type == "agemix":
            return ["All->All"]
        return list(self._post_cint_samples.keys())

    # ------------------------------------------------------------------
    # Cache management
    # ------------------------------------------------------------------

    def clear_cache(self) -> None:
        """Clear all cached summarisation results."""
        self._cache.clear()

    def get_cache_info(self) -> Dict[str, Any]:
        """Return information about cached results.

        Returns
        -------
        dict
            ``n_cached``         — number of cached entries.
            ``cache_keys``       — list of cache keys.
            ``inference_method`` — "mcmc" or "svi".
            ``model_type``       — "agemix" or "genmix".
        """
        return {
            "n_cached": len(self._cache),
            "cache_keys": list(self._cache.keys()),
            "inference_method": self.inference_method,
            "model_type": self.model_type,
        }

    def release_raw_samples(self) -> None:
        """Release raw posterior samples from memory.

        Useful in memory-constrained pipelines after summaries have been
        computed and cached. Raw samples can be reloaded by calling
        ``clear_cache()`` followed by any ``summarise_*`` method.
        """
        self._post_samples = None

    # ------------------------------------------------------------------
    # Dunder helpers
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        loaded = self._post_samples is not None
        strata_repr = str(self.strata) if loaded else "not loaded"
        return (
            f"ModelSummariser("
            f"model_type={self.model_type!r}, "
            f"inference_method={self.inference_method!r}, "
            f"strata={strata_repr}, "
            f"num_samples={self.num_samples})"
        )
