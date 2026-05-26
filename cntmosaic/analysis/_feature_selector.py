"""
Feature selection via LOO-CV ELPD model comparison.

Provides ``FeatureSelector``, which automates fitting and LOO-CV comparison of
contact survey models that differ in their stratification variables.  The
reference model (highest complexity) anchors the comparison; every simpler
candidate is evaluated on the reference model's observation grid using
projected stratification indices.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Type

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    import pandas as pd
    from jax import Array

from .._types import StratMode
from ..dataloader._ContactSurveyLoader import ContactSurveyLoader
from ..dataloader._stratification import (
    infer_strat_dims,
    infer_strat_ixs,
    infer_strat_pixs,
    make_flat_ix,
)
from ._arviz import svi_to_inference_data


@dataclass
class ModelConfig:
    """
    Configuration for a single model in a :class:`FeatureSelector` run.

    Parameters
    ----------
    name : str
        Unique label used as the key in the comparison table.
    model_cls : type
        ContactModel subclass to instantiate (e.g. ``GenMixFF``, ``AgeMixFF``).
    dataloader : ContactSurveyLoader
        Pre-built loader for this model's stratification configuration.
    priors : dict
        Prior dict passed to ``model_cls.__init__``.
    likelihood : str, default "negbin"
        Likelihood string passed to ``model_cls.__init__``.
    guide_factory : callable, optional
        ``guide_factory(model) -> guide``.  When supplied, overrides the
        ``FeatureSelector``-level ``guide_factory`` for this model only.
    """

    name: str
    model_cls: type
    dataloader: ContactSurveyLoader
    priors: Dict[str, Any]
    likelihood: str = "negbin"
    guide_factory: Optional[Callable] = field(default=None, repr=False)


@dataclass
class FeatureSelectionResult:
    """
    Results returned by :meth:`FeatureSelector.run`.

    Attributes
    ----------
    comparison : pd.DataFrame
        ``arviz.compare()`` output, sorted by ELPD-LOO (best model first).
    idatas : dict
        ``InferenceData`` objects keyed by model name.
    models : dict
        Fitted ``ContactModel`` instances keyed by model name.
    reference_name : str
        Name of the reference (most complex) model.
    """

    comparison: "pd.DataFrame"
    idatas: Dict[str, Any]
    models: Dict[str, Any]
    reference_name: str

    @property
    def best_model(self):
        """Fitted model with the highest ELPD-LOO."""
        return self.models[self.comparison.index[0]]


class FeatureSelector:
    """
    Automated LOO-CV ELPD feature selection over stratification variables.

    Fits a reference model and a set of simpler candidate models, evaluates
    every candidate on the reference model's observation grid, and returns a
    ranked LOO-CV comparison via ``arviz.compare``.

    Parameters
    ----------
    reference_config : ModelConfig
        Configuration for the most complex (reference) model.  All candidate
        models must use a strict subset of its stratification variables.
    candidate_configs : list of ModelConfig
        Simpler model configurations.  Each candidate's ``part_strat_vars``
        must be a subset of the reference's, and its ``cnt_strat_vars`` must be
        a subset of the reference's.  FULL-to-PARTIAL demotion (keeping a
        participant-side variable but dropping the contact side) is not
        supported.
    guide_factory : callable, optional
        Default ``guide_factory(model) -> guide`` used for any config that does
        not supply its own.  Required unless every ``ModelConfig`` sets its own.
    num_steps : int, default 5 000
        SVI optimisation steps, shared across all models.
    peak_lr : float, default 0.01
        Peak learning rate for the cosine-annealing schedule.
    num_samples : int, default 1 000
        Posterior samples drawn per model by ``svi_to_inference_data``.

    Examples
    --------
    >>> selector = FeatureSelector(
    ...     reference_config=ModelConfig("sex_educ", GenMixFF, loader_full, priors_full),
    ...     candidate_configs=[
    ...         ModelConfig("sex_only", GenMixFF, loader_sex, priors_sex),
    ...         ModelConfig("no_strat", AgeMixFF, loader_simple, priors_simple),
    ...     ],
    ...     guide_factory=make_guide,
    ...     num_steps=5_000,
    ... )
    >>> result = selector.run(PRNGKey(0))
    >>> result.comparison
    >>> result.best_model
    """

    def __init__(
        self,
        reference_config: ModelConfig,
        candidate_configs: List[ModelConfig],
        guide_factory: Optional[Callable] = None,
        num_steps: int = 5_000,
        peak_lr: float = 0.01,
        num_samples: int = 1_000,
    ) -> None:
        self.reference_config = reference_config
        self.candidate_configs = list(candidate_configs)
        self.guide_factory = guide_factory
        self.num_steps = num_steps
        self.peak_lr = peak_lr
        self.num_samples = num_samples

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self, prng_key: "Array") -> FeatureSelectionResult:
        """
        Fit all models and return a ranked LOO-CV comparison.

        Parameters
        ----------
        prng_key : PRNGKey
            Base JAX random key.  Each model receives an independent split.

        Returns
        -------
        FeatureSelectionResult
        """
        from arviz import compare
        from jax import random

        self._validate()
        self._ensure_loaded()

        all_configs = [self.reference_config] + self.candidate_configs
        keys = self._split_keys(prng_key, len(all_configs))

        fitted_models: Dict[str, Any] = {}
        for cfg, key in zip(all_configs, keys):
            fitted_models[cfg.name] = self._fit_model(cfg, key)

        ref_name = self.reference_config.name
        ref_loader = self.reference_config.dataloader

        idatas: Dict[str, Any] = {}
        idatas[ref_name] = svi_to_inference_data(
            fitted_models[ref_name],
            num_samples=self.num_samples,
        )
        for cfg in self.candidate_configs:
            kwargs_dict = self._build_kwargs_dict(cfg, ref_loader)
            idatas[cfg.name] = svi_to_inference_data(
                fitted_models[cfg.name],
                num_samples=self.num_samples,
                kwargs_dict=kwargs_dict,
            )

        comparison = compare(idatas, ic="loo")
        return FeatureSelectionResult(
            comparison=comparison,
            idatas=idatas,
            models=fitted_models,
            reference_name=ref_name,
        )

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def _validate(self) -> None:
        """Check structural constraints before any computation runs."""
        if not self.candidate_configs:
            raise ValueError("At least one candidate_config is required.")

        all_configs = [self.reference_config] + self.candidate_configs

        # Unique names
        names = [c.name for c in all_configs]
        if len(names) != len(set(names)):
            duplicates = sorted({n for n in names if names.count(n) > 1})
            raise ValueError(f"Duplicate model names: {duplicates}")

        ref_part = self._part_vars(self.reference_config)
        ref_cnt = self._cnt_vars(self.reference_config)

        for cfg in self.candidate_configs:
            cand_part = self._part_vars(cfg)
            cand_cnt = self._cnt_vars(cfg)

            # Subset check — participant side
            extra_part = cand_part - ref_part
            if extra_part:
                raise ValueError(
                    f"Candidate '{cfg.name}' has participant stratification variables "
                    f"{sorted(extra_part)} not present in the reference model "
                    f"(reference has: {sorted(ref_part)})."
                )

            # Subset check — contact side
            extra_cnt = cand_cnt - ref_cnt
            if extra_cnt:
                raise ValueError(
                    f"Candidate '{cfg.name}' has contact stratification variables "
                    f"{sorted(extra_cnt)} not present in the reference model "
                    f"(reference has: {sorted(ref_cnt)})."
                )

            # No FULL-to-PARTIAL demotion: if a variable is FULL in the reference
            # (present in both part and cnt), it must stay FULL or be dropped
            # entirely — it cannot appear only on the participant side.
            for var in cand_part:
                if var in ref_cnt and var not in cand_cnt:
                    raise ValueError(
                        f"Candidate '{cfg.name}': variable '{var}' is FULL "
                        f"(participant + contact) in the reference but only on the "
                        f"participant side in the candidate. FULL-to-PARTIAL demotion "
                        f"is not supported. Either keep '{var}' as FULL or drop it "
                        f"entirely from the candidate."
                    )

        if not ref_part and not ref_cnt:
            warnings.warn(
                "Reference model has no stratification variables. "
                "All models will be structurally identical.",
                UserWarning,
                stacklevel=3,
            )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _part_vars(cfg: ModelConfig) -> set:
        """Canonical (no prefix) participant strat var names for a config."""
        return {
            v.removeprefix("part_")
            for v in (cfg.dataloader.col_map.part_strat_vars or [])
        }

    @staticmethod
    def _cnt_vars(cfg: ModelConfig) -> set:
        """Canonical (no prefix) contact strat var names for a config."""
        return {
            v.removeprefix("cnt_")
            for v in (cfg.dataloader.col_map.cnt_strat_vars or [])
        }

    def _ensure_loaded(self) -> None:
        """Call .load() on any loader that hasn't been loaded yet."""
        for cfg in [self.reference_config] + self.candidate_configs:
            if cfg.dataloader.model_data is None:
                cfg.dataloader.load()

    def _fit_model(self, cfg: ModelConfig, key: "Array"):
        """Instantiate and fit a model via SVI, returning the fitted instance."""
        model = cfg.model_cls(cfg.dataloader, cfg.priors, cfg.likelihood)

        factory = cfg.guide_factory or self.guide_factory
        if factory is None:
            raise ValueError(
                f"No guide_factory provided for model '{cfg.name}'. "
                "Supply one via ModelConfig.guide_factory or "
                "FeatureSelector.guide_factory."
            )
        guide = factory(model)

        model.run_inference_svi(key, guide, num_steps=self.num_steps, peak_lr=self.peak_lr)

        self._warn_if_not_converged(cfg.name, model)
        return model

    @staticmethod
    def _warn_if_not_converged(name: str, model) -> None:
        """Emit a warning when the SVI loss curve looks unconverged."""
        try:
            losses = np.asarray(model._svi_result.losses)
            if len(losses) < 2:
                return
            # Compare last 1 % of steps to first 1 % — still decreasing if ratio > 0.99
            n = max(1, len(losses) // 100)
            ratio = float(losses[-n:].mean()) / float(losses[:n].mean())
            if ratio > 0.99:
                warnings.warn(
                    f"Model '{name}': SVI loss at the end of training is still close "
                    "to its initial value (final/initial ratio "
                    f"{ratio:.3f} > 0.99). The model may not have converged. "
                    "Consider increasing num_steps.",
                    UserWarning,
                    stacklevel=4,
                )
        except Exception:
            pass  # convergence check is best-effort

    def _build_kwargs_dict(
        self,
        cfg: ModelConfig,
        ref_loader: ContactSurveyLoader,
    ) -> Dict[str, NDArray]:
        """
        Build the kwargs_dict that evaluates the candidate on the reference data.

        For an unstratified candidate, only the base observation arrays are
        returned (y, aid/bid, log_N, log_V).  For a stratified candidate,
        flat_ix and flat_pixs are recomputed by projecting the reference
        loader's df_full down to the candidate's subset of variables.
        """
        ref_md = ref_loader.model_data
        df_full = ref_loader.df_full

        # Base observation arrays — always included
        kwargs: Dict[str, NDArray] = {
            "y": ref_md.y,
            "log_N": ref_md.log_N,
        }
        for field_name in ("aid", "bid", "cid", "did", "log_V"):
            val = getattr(ref_md, field_name, None)
            if val is not None:
                kwargs[field_name] = val

        cand_part_vars = list(self._part_vars(cfg))
        if not cand_part_vars:
            return kwargs

        # Project reference strat_modes to candidate's subset (preserving order)
        ref_strat_modes: Dict[str, StratMode] = ref_md.strat_modes  # type: ignore[assignment]
        # Preserve insertion order: use the order from the reference col_map
        ref_part_ordered = [
            v.removeprefix("part_")
            for v in ref_loader.col_map.part_strat_vars
            if v.removeprefix("part_") in cand_part_vars
        ]
        subset_modes = {v: ref_strat_modes[v] for v in ref_part_ordered}

        subset_dims = infer_strat_dims(df_full, subset_modes)
        subset_ixs = infer_strat_ixs(df_full, subset_modes)
        kwargs["flat_ix"] = make_flat_ix(subset_ixs, subset_dims)
        kwargs["flat_pixs"] = infer_strat_pixs(df_full, subset_modes, subset_dims)
        return kwargs

    @staticmethod
    def _split_keys(base_key: "Array", n: int) -> List["Array"]:
        """Split base_key into n independent JAX random keys."""
        from jax import random

        keys = []
        key = base_key
        for _ in range(n):
            key, subkey = random.split(key)
            keys.append(subkey)
        return keys
