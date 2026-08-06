"""Statistical summariser for Prem2 model inference results."""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd
from jax.random import PRNGKey
from numpy.typing import NDArray

from ...dataloader.containers import PopulationData
from ...models.classical._socialmix_helpers import (
    apply_reciprocity as _sm_apply_reciprocity,
)
from ...models.classical._socialmix_helpers import contact_labels
from ...utils import AgeGroupSpecs, depixilate
from .._stats import compute_quantiles, validate_alpha
from ._summary import ContactSummary

if TYPE_CHECKING:
    from ...models.classical._Prem2 import Prem2


def _get_ci_probs(alpha: float) -> Tuple[float, float, float]:
    """Return (lower, median, upper) quantile probabilities for the given alpha."""
    return (alpha / 2, 0.5, 1 - alpha / 2)


class ModelSummariserPrem2:
    """
    Statistical summariser for Prem2 model inference results.

    Computes quantiles and credible intervals for aggregated-cell contact
    matrices from MCMC or SVI posterior samples, with proper handling of
    reciprocity and depixilation. Unlike :class:`ModelSummariserPrem`, which
    special-cases unstratified (``K == 1``) models as raw arrays, this
    summariser always represents posterior samples as a
    ``Dict[str, NDArray]`` keyed by ``"source->target"`` stratum label
    (``"All->All"`` for unstratified models) — the same convention used by
    :class:`ModelSummariserSocialMix`, matching Prem2's own
    ``K_part``/``K_cnt`` stratification bookkeeping.

    Parameters
    ----------
    prem2 : Prem2
        Fitted Prem2 model with MCMC or SVI results.
    pop_data : PopulationData, optional
        Population data container with fine-grained (1-year) age
        distribution. Prem2 has no population data of its own, so this must
        be supplied externally for reciprocity adjustment, depixilation, and
        rate computation.
    num_samples : int, default=3000
        Number of posterior samples to draw if using SVI.

    Attributes
    ----------
    prem2 : Prem2
        Reference to the Prem2 model.
    age_group_specs : AgeGroupSpecs
        Age bins used by the model.
    strata_labels : List[str]
        Ordered ``"source->target"`` stratum labels.
    post_cint_samples : Dict[str, NDArray]
        Posterior contact intensity samples (``exp(log_cint)``), one entry
        per stratum, shape ``(n_samples, C, D)``.

    Examples
    --------
    >>> prem2 = Prem2(part_data, cnt_data, age_bins)
    >>> prem2.run_inference_svi(PRNGKey(0), num_steps=2000)
    >>>
    >>> summariser = ModelSummariserPrem2(prem2, pop_data=pop_data)
    >>> summary = summariser.summarise_cint(alpha=0.05)
    >>> summary["All->All"].central
    """

    def __init__(
        self,
        prem2: "Prem2",
        pop_data: Optional[PopulationData] = None,
        num_samples: int = 3000,
    ) -> None:
        if prem2.inference_method is None:
            raise ValueError(
                "Either MCMC or SVI must have been run on the model. "
                "Call prem2.run_inference_mcmc() or prem2.run_inference_svi() first."
            )

        self.prem2 = prem2

        # Reference key attributes directly off the model — Prem2 already
        # computes these during _preprocess(), so there is no need to
        # re-derive strat_mode/labels the way ModelSummariserPrem does.
        self.age_group_specs = prem2.age_group_specs
        self.K_part = prem2.K_part
        self.K_cnt = prem2.K_cnt
        self.C = prem2.C
        self.D = prem2.D
        self.strat_vars_part = prem2.strat_vars_part
        self.strat_vars_cnt = prem2.strat_vars_cnt
        self.strat_vars_shared = prem2.strat_vars_shared
        self.strat_mode = prem2.strat_mode
        self.strata_labels = prem2._create_stratum_labels()

        self.pop_data = pop_data
        self.num_samples = num_samples

        self.age_dist: Optional[NDArray] = None
        self.age_grp_dist: Optional[NDArray] = None
        self.post_samples: Optional[Dict[str, Any]] = None
        self.post_cint_samples: Optional[Dict[str, NDArray]] = None

        # Simple cache: {cache_key: result_dict}
        self._cache: Dict[str, Dict[str, ContactSummary]] = {}

        self._validate()
        self._load()

        if (
            self.age_grp_dist is None
            and self.age_group_specs is not None
            and self.age_dist is not None
        ):
            self.age_grp_dist = self._compute_age_grp_dist()

    # ------------------------------------------------------------------
    # Validation / loading
    # ------------------------------------------------------------------

    def _validate(self) -> None:
        """Validate population data stratification against participant data."""
        if self.pop_data is None:
            return

        if self.strat_vars_part and self.pop_data.strat_vars:
            self._validate_population_stratification()

    def _validate_population_stratification(self) -> None:
        """
        Validate that population stratification variables match participant data.

        Ensures the same stratification variables and categories exist on
        both sides (using participant category order as reference). If
        categories match but order differs, population data is reordered.

        Raises
        ------
        ValueError
            If stratification variables or categories don't match.
        """
        part_strat_vars = self.strat_vars_part
        pop_strat_vars = self.pop_data.strat_vars

        if set(part_strat_vars) != set(pop_strat_vars):
            raise ValueError(
                f"Population stratification variables {pop_strat_vars} don't match "
                f"participant stratification variables {part_strat_vars}. "
                f"For stratified models, PopulationData must have the same stratification variables."
            )

        for var in part_strat_vars:
            col_part = f"part_{var}"
            col_pop = var

            part_col = self.prem2.part_data.data[col_part]
            pop_col = self.pop_data.data[col_pop]

            if not hasattr(part_col, "cat"):
                part_col = part_col.astype("category")
            if not hasattr(pop_col, "cat"):
                pop_col = pop_col.astype("category")
                self.pop_data.data[col_pop] = pop_col

            part_cats = list(part_col.cat.categories)
            pop_cats = list(pop_col.cat.categories)

            part_set = set(part_cats)
            pop_set = set(pop_cats)

            if part_set != pop_set:
                only_part = part_set - pop_set
                only_pop = pop_set - part_set
                raise ValueError(
                    f"Population stratification variable '{var}' has different categories:\n"
                    f"  Participant side: {part_cats}\n"
                    f"  Population side: {pop_cats}\n"
                    f"  Only in participants: {sorted(only_part) if only_part else 'None'}\n"
                    f"  Only in population: {sorted(only_pop) if only_pop else 'None'}\n"
                    f"For stratified models, population must have the same categories as participants."
                )

            if part_cats != pop_cats:
                self.pop_data.data[col_pop] = self.pop_data.data[
                    col_pop
                ].cat.reorder_categories(part_cats, ordered=False)

    def _load(self) -> None:
        """Load age distributions and posterior samples."""
        if self.pop_data is not None:
            self.age_dist = self.pop_data.data.groupby("age")["P"].sum().values
        else:
            warnings.warn(
                "PopulationData not provided. "
                "Reciprocity adjustment and depixilation will not be possible.",
                UserWarning,
            )

        self.post_samples = self.prem2.draw_posterior_samples(
            PRNGKey(0), num_samples=self.num_samples
        )

        if "log_cint" not in self.post_samples:
            available_fields = list(self.post_samples.keys())
            raise ValueError(
                f"Posterior samples must contain 'log_cint' field. "
                f"Available fields: {available_fields}."
            )

        self.post_cint_samples = self._reshape_cint_samples(
            np.asarray(self.post_samples["log_cint"])
        )

    def _compute_age_grp_dist(self) -> NDArray:
        """Compute age group distribution from fine-grained age distribution."""
        age_grp_dist = []
        age_edges = self.age_group_specs.left + [self.age_group_specs.max + 1]

        for i in range(len(age_edges) - 1):
            start_age = int(age_edges[i])
            end_age = int(age_edges[i + 1])
            age_grp_dist.append(self.age_dist[start_age:end_age].sum())

        return np.array(age_grp_dist)

    def _reshape_cint_samples(self, log_cint: NDArray) -> Dict[str, NDArray]:
        """
        Split posterior ``log_cint`` samples into a per-stratum intensity dict.

        ``Prem2NumPyroMixin.model()`` produces ``log_cint`` with shape
        ``(n, C, D)``, ``(n, K_part, C, D)``, or ``(n, K_part, K_cnt, C, D)``
        depending on stratification (see ``cntmosaic/models/numpyro/_Prem2.py``).
        This always returns a ``Dict[str, NDArray]`` — unlike
        :class:`ModelSummariserPrem`, there is no raw-NDArray special case for
        the unstratified model.

        Returns
        -------
        Dict[str, NDArray]
            Mapping stratum label -> contact intensity samples, shape
            ``(n_samples, C, D)``.
        """
        labels = self.strata_labels

        if self.K_part == 1 and self.K_cnt == 1:
            return {labels[0]: np.exp(log_cint)}

        if self.K_part > 1 and self.K_cnt == 1:
            arr = np.exp(log_cint)  # (n, K_part, C, D)
            return {label: arr[:, k] for k, label in enumerate(labels)}

        # Full/mixed: flatten (K_part, K_cnt) into a single axis matching
        # create_stratum_labels's itertools.product(part, cnt) order.
        n = log_cint.shape[0]
        arr = np.exp(log_cint).reshape(n, self.K_part * self.K_cnt, self.C, self.D)
        return {label: arr[:, k] for k, label in enumerate(labels)}

    # ------------------------------------------------------------------
    # Population helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _aggregate_population_to_bins(
        pop_data: PopulationData, age_bins: AgeGroupSpecs
    ) -> PopulationData:
        """
        Aggregate fine-grained population to match model age bins.

        Duplicated from ``ModelSummariserPrem._aggregate_population_to_bins``
        (generic pop_data -> age_bins aggregation, no Prem-specific coupling).

        Parameters
        ----------
        pop_data : PopulationData
            Fine-grained population data.
        age_bins : AgeGroupSpecs
            Age bins from the model.

        Returns
        -------
        PopulationData
            Aggregated population matching age bins.
        """
        df_pop = pop_data.data.copy()

        age_edges = list(age_bins.left) + [age_bins.max + 1]
        age_labels = age_bins.left

        df_pop["age_grp"] = pd.cut(
            df_pop["age"],
            bins=age_edges,
            labels=age_labels,
            right=False,
            include_lowest=True,
        )

        group_cols = ["age_grp"]
        if pop_data.strat_var_cols:
            group_cols.extend(pop_data.strat_var_cols)

        df_pop_agg = df_pop.groupby(group_cols, observed=False)["P"].sum().reset_index()
        df_pop_agg = df_pop_agg.rename(columns={"age_grp": "age"})
        df_pop_agg["age"] = df_pop_agg["age"].astype(int)

        return PopulationData(
            df_pop_agg,
            age_col="age",
            size_col="P",
            strat_var_cols=pop_data.strat_var_cols,
        )

    def _build_reciprocity_P(self) -> Optional[NDArray]:
        """
        Build the population array required by ``apply_reciprocity``.

        Returns
        -------
        NDArray or None
            Shape ``(D,)`` for single mode, ``(K_cnt, D)`` for full mode.
            ``None`` for "partial"/"mixed" (reciprocity is a no-op there).

        Raises
        ------
        ValueError
            If ``pop_data`` was not supplied but reciprocity is applicable.
        """
        if self.strat_mode not in ("single", "full"):
            return None

        if self.pop_data is None:
            raise ValueError(
                "PopulationData required for reciprocity adjustment. "
                f"Provide pop_data when constructing ModelSummariserPrem2 for "
                f"{self.strat_mode!r} stratification mode."
            )

        if self.strat_mode == "single":
            age_dist = self.pop_data.data.groupby("age")["P"].sum().values
            if len(age_dist) != self.D:
                pop_agg = self._aggregate_population_to_bins(
                    self.pop_data, self.age_group_specs
                )
                age_dist = pop_agg.data.groupby("age")["P"].sum().values
            return age_dist

        # Full mode: stack per-target-stratum population in contact_labels
        # order, matching the k_t = idx % K_cnt indexing inside the shared
        # _socialmix_helpers.apply_reciprocity.
        cnt_order = contact_labels(self.strat_vars_cnt, self.prem2.cnt_data)
        pop_sizes = self.pop_data.get_stratified_pop_sizes(
            strat_var_cols=self.strat_vars_cnt
        )
        first = pop_sizes[cnt_order[0]]
        if len(first) != self.D:
            pop_agg = self._aggregate_population_to_bins(
                self.pop_data, self.age_group_specs
            )
            pop_sizes = pop_agg.get_stratified_pop_sizes(
                strat_var_cols=self.strat_vars_cnt
            )
        return np.stack([pop_sizes[label] for label in cnt_order])

    def _get_rate_denominator(
        self, label: str, depixilated: bool
    ) -> Optional[NDArray]:
        """
        Return the population denominator used for ``rate = cint / P``.

        Uses the target-side (contact) stratum population when available,
        falling back to the unstratified ``age_grp_dist``/``age_dist``
        otherwise — a refinement over :class:`ModelSummariserPrem`, which
        always divides by one unstratified vector regardless of stratum.
        """
        pop_dist = self.age_dist if depixilated else self.age_grp_dist
        if pop_dist is None or self.pop_data is None or not self.strat_vars_cnt:
            return pop_dist

        target = label.split("->")[1] if "->" in label else None
        if target is None:
            return pop_dist

        try:
            pop_sizes = self.pop_data.get_stratified_pop_sizes(
                strat_var_cols=self.strat_vars_cnt
            )
        except (ValueError, KeyError):
            return pop_dist

        if target not in pop_sizes:
            return pop_dist

        target_dist = pop_sizes[target]
        expected_len = self.D if depixilated is False else self.age_group_specs.range
        if len(target_dist) != expected_len:
            return pop_dist
        return target_dist

    # ------------------------------------------------------------------
    # Reciprocity
    # ------------------------------------------------------------------

    @staticmethod
    def apply_reciprocity(
        cint_samples: Dict[str, NDArray],
        P: Optional[NDArray],
        strat_mode: str,
        K_cnt: int,
    ) -> Dict[str, NDArray]:
        """
        Batch-apply population-weighted reciprocity across posterior draws.

        Reuses :func:`cntmosaic.models.classical._socialmix_helpers.apply_reciprocity`,
        which operates on a single draw (a dict of raw ``(C, D)`` matrices).
        This loops over the sample axis, calling it once per draw, rather
        than reimplementing the reciprocity math for a batched axis — the
        same per-draw cost pattern already accepted for depixilation.

        Parameters
        ----------
        cint_samples : Dict[str, NDArray]
            Contact intensity samples keyed by stratum label, shape
            ``(n_samples, C, D)``.
        P : NDArray or None
            Population sizes — shape ``(D,)`` for single mode, ``(K_cnt, D)``
            for full mode. Required (non-None) for "single"/"full" modes.
        strat_mode : str
            One of ``"single"``, ``"partial"``, ``"full"``, ``"mixed"``.
        K_cnt : int
            Number of contact strata (used for index math in full mode).

        Returns
        -------
        Dict[str, NDArray]
            Reciprocity-adjusted samples, same structure as input.
        """
        if strat_mode not in ("single", "full"):
            warnings.warn(
                f"Reciprocity not applied for {strat_mode!r} stratification mode. "
                "Contact rates have no inherent symmetry when only one side is "
                "stratified or different variables are used.",
                UserWarning,
            )
            return cint_samples

        if P is None:
            raise ValueError("P (population sizes) required for reciprocity adjustment.")

        labels = list(cint_samples.keys())
        n_samples = cint_samples[labels[0]].shape[0]
        result = {label: np.empty_like(arr) for label, arr in cint_samples.items()}

        for i in range(n_samples):
            draw = {label: arr[i] for label, arr in cint_samples.items()}
            adjusted = _sm_apply_reciprocity(draw, strat_mode, K_cnt, P)
            for label, m in adjusted.items():
                result[label][i] = m

        return result

    # ------------------------------------------------------------------
    # Depixilation
    # ------------------------------------------------------------------

    def _depixilate_samples(
        self, samples: Dict[str, NDArray], pop_data: PopulationData
    ) -> Dict[str, NDArray]:
        """
        Depixilate posterior samples to 1-year age resolution.

        CRITICAL: This must be done BEFORE computing quantiles, because
        depixilation is a nonlinear transformation that doesn't commute with
        quantile operations.

        Uses the *source* (participant-side) stratum population for
        disaggregation weights, looked up via
        :meth:`PopulationData.get_stratified_pop_sizes`.

        Parameters
        ----------
        samples : Dict[str, NDArray]
            Posterior samples at age-group resolution, shape
            ``(n_samples, C, D)`` per stratum.
        pop_data : PopulationData
            Population data with fine-grained (1-year) age distribution.

        Returns
        -------
        Dict[str, NDArray]
            Depixilated samples, shape ``(n_samples, A, A)`` per stratum,
            where ``A = age_group_specs.range``.
        """
        if self.age_group_specs is None:
            raise ValueError("age_group_specs must be available for depixilation.")

        if pop_data.n_ages < self.age_group_specs.range:
            raise ValueError(
                f"PopulationData has only {pop_data.n_ages} ages, "
                f"but need {self.age_group_specs.range} for fine-grained depixilation"
            )

        A = self.age_group_specs.range

        source_pop: Optional[Dict[str, NDArray]] = None
        if self.strat_vars_part:
            try:
                source_pop = pop_data.get_stratified_pop_sizes(
                    strat_var_cols=self.strat_vars_part
                )
            except (ValueError, KeyError):
                source_pop = None

        total_age_dist = pop_data.data.groupby("age")["P"].sum().values

        result: Dict[str, NDArray] = {}
        for label, arr in samples.items():
            source = label.split("->")[0] if "->" in label else None
            if source_pop is not None and source in source_pop:
                age_dist = source_pop[source]
            else:
                age_dist = total_age_dist

            n_samples = arr.shape[0]
            depix = np.empty((n_samples, A, A), dtype=np.float64)
            for i in range(n_samples):
                depix[i] = depixilate(arr[i], self.age_group_specs, age_dist)
            result[label] = depix

        return result

    # ------------------------------------------------------------------
    # Public summarisation API
    # ------------------------------------------------------------------

    def summarise_cint(
        self,
        alpha: float = 0.05,
        apply_reciprocity: bool = False,
        return_depixilated: bool = False,
        force_recompute: bool = False,
    ) -> Dict[str, ContactSummary]:
        """
        Compute credible-interval summaries for the contact intensity matrix.

        Contact intensity M[c,d] represents the average number of contacts
        that individuals in age group c have with individuals in age group d.

        Parameters
        ----------
        alpha : float, default=0.05
            Significance level for credible intervals (e.g., 0.05 for 95% CI).
        apply_reciprocity : bool, default=False
            If True, apply reciprocity adjustment to enforce demographic
            symmetry. Only applied for "single" and "full" stratification
            modes. Requires pop_data to be provided.
        return_depixilated : bool, default=False
            If True, return results at 1-year age resolution instead of age
            groups. Requires pop_data to be available.
        force_recompute : bool, default=False
            Force recomputation even if cached.

        Returns
        -------
        Dict[str, ContactSummary]
            One entry per stratum, keyed by ``"source->target"`` label
            (``"All->All"`` for unstratified models).

        Notes
        -----
        Order of operations:
        1. Reciprocity adjustment (if requested and applicable)
        2. Depixilation (if requested)
        3. Quantile computation

        This order is critical because depixilation and quantiles don't
        commute.
        """
        validate_alpha(alpha)
        probs = _get_ci_probs(alpha)

        cache_key = f"cint_alpha{alpha}_recip{apply_reciprocity}_depix{return_depixilated}"
        if not force_recompute and cache_key in self._cache:
            return self._cache[cache_key]

        samples = {k: v.copy() for k, v in self.post_cint_samples.items()}

        if apply_reciprocity:
            P = self._build_reciprocity_P()
            samples = self.apply_reciprocity(samples, P, self.strat_mode, self.K_cnt)

        if return_depixilated:
            if self.pop_data is None:
                raise ValueError("pop_data must be provided for depixilation.")
            samples = self._depixilate_samples(samples, self.pop_data)

        result: Dict[str, ContactSummary] = {}
        for label, arr in samples.items():
            q = compute_quantiles(arr, probs, axis=0)
            result[label] = ContactSummary(
                lower=q[0],
                central=q[1],
                upper=q[2],
                alpha=alpha,
                measure="median",
                age_group_specs=self.age_group_specs,
            )

        self._cache[cache_key] = result
        return result

    def summarise_rate(
        self,
        alpha: float = 0.05,
        apply_reciprocity: bool = False,
        return_depixilated: bool = False,
        force_recompute: bool = False,
    ) -> Dict[str, ContactSummary]:
        """
        Compute credible-interval summaries for the contact rate matrix.

        Contact rate R[c,d] represents the per-capita rate at which
        individuals in age group c contact individuals in age group d.
        Computed as: R[c,d] = M[c,d] / P[d].

        Uses the target-side (contact) stratum population as the
        denominator when available, falling back to the unstratified
        population otherwise.

        Parameters
        ----------
        alpha : float, default=0.05
            Significance level for credible intervals.
        apply_reciprocity : bool, default=False
            If True, apply reciprocity adjustment before computing rates.
        return_depixilated : bool, default=False
            If True, return at 1-year age resolution.
        force_recompute : bool, default=False
            Force recomputation even if cached.

        Returns
        -------
        Dict[str, ContactSummary]
            One entry per stratum.

        Raises
        ------
        ValueError
            If no population data is available for rate computation.
        """
        validate_alpha(alpha)
        probs = _get_ci_probs(alpha)

        cache_key = f"rate_alpha{alpha}_recip{apply_reciprocity}_depix{return_depixilated}"
        if not force_recompute and cache_key in self._cache:
            return self._cache[cache_key]

        if self.age_grp_dist is None and self.age_dist is None:
            raise ValueError(
                "Population data required for rate computation. "
                "Provide pop_data to the ModelSummariserPrem2 constructor."
            )

        samples = {k: v.copy() for k, v in self.post_cint_samples.items()}

        if apply_reciprocity:
            P = self._build_reciprocity_P()
            samples = self.apply_reciprocity(samples, P, self.strat_mode, self.K_cnt)

        if return_depixilated:
            if self.pop_data is None:
                raise ValueError("pop_data must be provided for depixilation.")
            samples = self._depixilate_samples(samples, self.pop_data)

        result: Dict[str, ContactSummary] = {}
        for label, cint_samples in samples.items():
            pop_dist = self._get_rate_denominator(label, return_depixilated)
            if pop_dist is None:
                raise ValueError(
                    "Population data required for rate computation. "
                    "Provide pop_data to the ModelSummariserPrem2 constructor."
                )
            rate_samples = cint_samples / pop_dist[np.newaxis, np.newaxis, :]
            q = compute_quantiles(rate_samples, probs, axis=0)
            result[label] = ContactSummary(
                lower=q[0],
                central=q[1],
                upper=q[2],
                alpha=alpha,
                measure="median",
                age_group_specs=self.age_group_specs,
            )

        self._cache[cache_key] = result
        return result

    def summarise_mcint(
        self,
        alpha: float = 0.05,
        apply_reciprocity: bool = False,
        return_depixilated: bool = False,
        force_recompute: bool = False,
    ) -> Dict[str, ContactSummary]:
        """
        Compute credible-interval summaries for the marginal contact intensity.

        Marginal contact intensity m[c] = Sum_d M[c,d] represents the total
        average number of contacts made by individuals in age group c across
        all age groups.

        Parameters
        ----------
        alpha : float, default=0.05
            Significance level for credible intervals.
        apply_reciprocity : bool, default=False
            If True, apply reciprocity adjustment before computing marginals.
        return_depixilated : bool, default=False
            If True, return at 1-year age resolution.
        force_recompute : bool, default=False
            Force recomputation even if cached.

        Returns
        -------
        Dict[str, ContactSummary]
            One entry per stratum; each ``ContactSummary`` has 1-D arrays of
            shape ``(A,)`` instead of ``(A, A)``.

        Notes
        -----
        When depixilation is requested: depixilate each full intensity
        matrix sample, then compute marginals, then quantiles — this
        ordering is critical because marginals and depixilation don't
        commute in general.
        """
        validate_alpha(alpha)
        probs = _get_ci_probs(alpha)

        cache_key = f"mcint_alpha{alpha}_recip{apply_reciprocity}_depix{return_depixilated}"
        if not force_recompute and cache_key in self._cache:
            return self._cache[cache_key]

        samples = {k: v.copy() for k, v in self.post_cint_samples.items()}

        if apply_reciprocity:
            P = self._build_reciprocity_P()
            samples = self.apply_reciprocity(samples, P, self.strat_mode, self.K_cnt)

        if return_depixilated:
            if self.pop_data is None:
                raise ValueError("pop_data must be provided for depixilation.")
            samples = self._depixilate_samples(samples, self.pop_data)

        result: Dict[str, ContactSummary] = {}
        for label, cint_samples in samples.items():
            marginal = cint_samples.sum(axis=-1)
            q = compute_quantiles(marginal, probs, axis=0)
            result[label] = ContactSummary(
                lower=q[0],
                central=q[1],
                upper=q[2],
                alpha=alpha,
                measure="median",
                age_group_specs=self.age_group_specs,
            )

        self._cache[cache_key] = result
        return result

    def get_point_estimates(
        self,
        apply_reciprocity: bool = False,
        return_depixilated: bool = False,
    ) -> Dict[str, Dict[str, Dict[str, NDArray]]]:
        """
        Get point estimates (mean and std) for all statistics, per stratum.

        Unlike :meth:`ModelSummariserPrem.get_point_estimates`, which returns
        an un-nested ``{"cint": ..., "rate": ..., "mcint": ...}`` dict (and is
        only correct for unstratified models), this always nests one level
        deeper by stratum label, since Prem2 has no unstratified special case.

        Parameters
        ----------
        apply_reciprocity : bool, default=False
            Whether to apply reciprocity adjustment.
        return_depixilated : bool, default=False
            Whether to return depixilated results.

        Returns
        -------
        Dict[str, Dict[str, Dict[str, NDArray]]]
            ``{stratum_label: {"cint": {"mean","std"}, "rate": {...}, "mcint": {...}}}``.
            ``"rate"`` is omitted for a stratum if no population denominator
            is available.
        """
        samples = {k: v.copy() for k, v in self.post_cint_samples.items()}

        if apply_reciprocity:
            P = self._build_reciprocity_P()
            samples = self.apply_reciprocity(samples, P, self.strat_mode, self.K_cnt)

        if return_depixilated:
            if self.pop_data is None:
                raise ValueError("pop_data must be provided for depixilation.")
            samples = self._depixilate_samples(samples, self.pop_data)

        result: Dict[str, Dict[str, Dict[str, NDArray]]] = {}
        for label, cint_samples in samples.items():
            entry: Dict[str, Dict[str, NDArray]] = {
                "cint": {
                    "mean": cint_samples.mean(axis=0),
                    "std": cint_samples.std(axis=0, ddof=1),
                },
            }

            pop_dist = self._get_rate_denominator(label, return_depixilated)
            if pop_dist is not None:
                rate_samples = cint_samples / pop_dist[np.newaxis, np.newaxis, :]
                entry["rate"] = {
                    "mean": rate_samples.mean(axis=0),
                    "std": rate_samples.std(axis=0, ddof=1),
                }

            mcint_samples = cint_samples.sum(axis=-1)
            entry["mcint"] = {
                "mean": mcint_samples.mean(axis=0),
                "std": mcint_samples.std(axis=0, ddof=1),
            }

            result[label] = entry

        return result

    def clear_cache(self) -> None:
        """Clear all cached computations."""
        self._cache.clear()

    def get_cache_info(self) -> Dict[str, Any]:
        """
        Get information about cached results.

        Returns
        -------
        Dict[str, Any]
            Dictionary with cache statistics including number of cached items
            and their keys.
        """
        return {
            "n_cached": len(self._cache),
            "cache_keys": list(self._cache.keys()),
        }
