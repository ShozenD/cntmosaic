import warnings
from typing import Dict, Literal, Optional

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from ..summariser._ModelSummariserBRC import ModelSummariserBRC
from ._base import (
    BaseModelEvaluator,
    aggregate_metrics,
    compute_metrics,
    validate_alpha,
)


class ModelEvaluatorBRC(BaseModelEvaluator):
    """
    Evaluator for BRC-family model performance against ground truth.

    Computes error metrics and uncertainty quantification statistics for estimated
    contact intensity matrices from Bayesian posterior samples. Supports both
    standard BRC models (BRCfine, BRCrefine) and hierarchical models (HiBRCfine, HiBRCrefine).

    This evaluator follows the same design patterns as ModelEvaluatorPrem for consistency,
    with automatic detection of model type and appropriate metric aggregation.

    Parameters
    ----------
    summariser : ModelSummariserBRC
        Summariser containing posterior samples and point estimates from a fitted
        BRC-family model (MCMC or SVI inference).
    cint_matrix_true : NDArray or dict
        Ground truth contact intensity matrix.
        - For BRC models: NDArray of shape (A, A) where A is number of age groups
        - For HiBRC models: dict with structure {var1: {cat1: matrix, ...}, ...}
    alpha : float, default=0.05
        Significance level for interval score and coverage computations.
        Must be in (0, 1).

    Attributes
    ----------
    summariser : ModelSummariserBRC
        Reference to the model summariser
    cint_true : NDArray or dict
        Ground truth contact intensity matrices
    mcint_true : NDArray or dict
        Ground truth marginal contact intensities (computed from cint_true)
    alpha : float
        Significance level for credible intervals
    model_type : str
        Either 'brc' or 'hibrc', automatically detected

    Examples
    --------
    **Standard BRC model evaluation:**

    >>> # Fit model
    >>> model = BRCfine(dataloader, priors, likelihood="poisson")
    >>> model.run_inference_mcmc(rng_key, num_samples=1000)
    >>>
    >>> # Create summariser and evaluator
    >>> summariser = ModelSummariserBRC(model)
    >>> evaluator = ModelEvaluatorBRC(summariser, cint_matrix_true, alpha=0.05)
    >>>
    >>> # Evaluate contact intensity
    >>> cint_metrics = evaluator.evaluate_cint()
    >>> print(cint_metrics)
    >>>
    >>> # Evaluate marginal contact intensity
    >>> mcint_metrics = evaluator.evaluate_mcint()
    >>> print(mcint_metrics)
    >>>
    >>> # Get all metrics at once
    >>> all_metrics = evaluator.evaluate()

    **Hierarchical BRC model evaluation:**

    >>> # For HiBRC models with stratification
    >>> model = HiBRCfine(dataloader, priors, likelihood="poisson")
    >>> model.run_inference_svi(rng_key, guide, num_steps=5000)
    >>>
    >>> # True matrices organized by stratification
    >>> cint_true = {
    ...     "gender": {"male": matrix_m, "female": matrix_f},
    ...     "location": {"urban": matrix_u, "rural": matrix_r}
    ... }
    >>>
    >>> summariser = ModelSummariserBRC(model, num_samples=500)
    >>> evaluator = ModelEvaluatorBRC(summariser, cint_true, alpha=0.05)
    >>>
    >>> # Returns metrics aggregated by variable and category
    >>> metrics = evaluator.evaluate()

    Notes
    -----
    - Caching is used to avoid redundant computation
    - For HiBRC models, metrics are computed for each category and aggregated
    - All metrics are returned as pandas DataFrames for easy analysis
    - The interval score penalizes both interval width and coverage failures

    See Also
    --------
    ModelSummariserBRC : Summarises posterior distributions from BRC models
    ModelEvaluatorPrem : Similar evaluator for Prem models
    """

    def __init__(
        self,
        summariser: ModelSummariserBRC,
        cint_matrix_true: NDArray[np.float64] | Dict[str, NDArray],
        alpha: float = 0.05,
    ) -> None:
        """
        Initialize ModelEvaluatorBRC.

        Parameters
        ----------
        summariser : ModelSummariserBRC
            Summariser with posterior samples from fitted BRC model
        cint_matrix_true : NDArray or dict[str, NDArray]
            True contact intensity matrix (for BRC) or dict of matrices (for HiBRC)
        alpha : float, default=0.05
            Significance level for interval metrics (must be in (0, 1))

        Raises
        ------
        ValueError
            If alpha not in (0, 1), or if model hasn't been fitted
        TypeError
            If summariser is not ModelSummariserBRC instance, or incompatible types
        """
        super().__init__(summariser, cint_matrix_true, alpha)

        # BRC-specific attribute
        self.model_type = self.summariser.model_type

    def _validate_summariser(self, summariser: ModelSummariserBRC) -> None:
        """Validate that summariser has required posterior samples."""
        if not isinstance(summariser, ModelSummariserBRC):
            raise TypeError(
                f"Expected ModelSummariserBRC instance, got {type(summariser).__name__}. "
                f"Usage: summariser = ModelSummariserBRC(model); "
                f"evaluator = ModelEvaluatorBRC(summariser, cint_true)"
            )

        if summariser.post_cint_samples is None:
            raise ValueError(
                "Summariser must have posterior contact intensity samples. "
                "Ensure MCMC or SVI inference was run on the BRC model."
            )

    def evaluate_cint(self, alpha: Optional[float] = None) -> pd.DataFrame:
        """
        Evaluate the posterior contact intensity matrix.

        Computes RMSE, MAE, MAPE, interval score, and coverage for the
        estimated contact intensity matrix against ground truth.

        Parameters
        ----------
        alpha : float, optional
            Significance level for interval metrics. If None, uses self.alpha

        Returns
        -------
        pd.DataFrame
            Evaluation metrics:
            - For BRC models: Single row DataFrame
            - For HiBRC models: One row per variable-category combination,
              plus aggregated rows per variable and overall

        Notes
        -----
        Results are cached to avoid redundant computation when called multiple times
        with the same alpha value.

        Examples
        --------
        >>> evaluator = ModelEvaluatorBRC(summariser, cint_true)
        >>> cint_metrics = evaluator.evaluate_cint(alpha=0.05)
        >>> print(f"RMSE: {cint_metrics['rmse'].values[0]:.3f}")
        >>> print(f"Coverage: {cint_metrics['coverage'].values[0]:.1f}%")
        """
        if alpha is None:
            alpha = self.alpha
        else:
            validate_alpha(alpha)

        cache_key = f"cint_alpha{alpha}"

        if cache_key not in self._metrics_cache:
            if self.model_type == "brc":
                # Standard BRC model
                summary = self.summariser.summarise_cint(alpha=alpha)
                y_true = self.cint_true["All->All"]
                y_est = summary["All->All"][1]  # median
                y_low = summary["All->All"][0]  # lower bound
                y_high = summary["All->All"][2]  # upper bound

                rmse, mae, mape, int_score, coverage = compute_metrics(
                    y_true, y_est, y_low, y_high
                )

                metrics_df = pd.DataFrame(
                    {
                        "var": ["all"],
                        "cat": ["all"],
                        "rmse": [rmse],
                        "mae": [mae],
                        "mape": [mape],
                        "interval_score": [int_score],
                        "coverage": [coverage],
                    }
                )

            else:  # model_type == "hibrc"
                # Hierarchical BRC model
                summary_dict = self.summariser.summarise_cint(alpha=alpha)
                metrics_df = aggregate_metrics(self.cint_true, summary_dict)

            self._metrics_cache[cache_key] = metrics_df

        return self._metrics_cache[cache_key].copy()

    def evaluate_mcint(self, alpha: Optional[float] = None) -> pd.DataFrame:
        """
        Evaluate the posterior marginal contact intensity.

        Computes RMSE, MAE, MAPE, interval score, and coverage for the
        estimated marginal contact intensity (sum over contact ages) against
        ground truth.

        Parameters
        ----------
        alpha : float, optional
            Significance level for interval metrics. If None, uses self.alpha

        Returns
        -------
        pd.DataFrame
            Evaluation metrics with same structure as evaluate_cint()

        Notes
        -----
        Marginal contact intensity represents the total contacts made by each
        age group, summed across all contact ages. This is often a key quantity
        of interest in epidemiological applications.

        Results are cached to avoid redundant computation.

        Examples
        --------
        >>> evaluator = ModelEvaluatorBRC(summariser, cint_true)
        >>> mcint_metrics = evaluator.evaluate_mcint(alpha=0.05)
        >>> print(f"Marginal RMSE: {mcint_metrics['rmse'].values[0]:.3f}")
        """
        if alpha is None:
            alpha = self.alpha
        else:
            validate_alpha(alpha)

        cache_key = f"mcint_alpha{alpha}"

        if cache_key not in self._metrics_cache:
            if self.model_type == "brc":
                # Standard BRC model
                summary = self.summariser.summarise_mcint(alpha=alpha)
                y_true = self.mcint_true["All->All"]
                y_est = summary["All->All"][1]  # median
                y_low = summary["All->All"][0]  # lower bound
                y_high = summary["All->All"][2]  # upper bound

                rmse, mae, mape, int_score, coverage = compute_metrics(
                    y_true, y_est, y_low, y_high
                )

                metrics_df = pd.DataFrame(
                    {
                        "var": ["all"],
                        "cat": ["all"],
                        "rmse": [rmse],
                        "mae": [mae],
                        "mape": [mape],
                        "interval_score": [int_score],
                        "coverage": [coverage],
                    }
                )

            else:  # model_type == "hibrc"
                # Hierarchical BRC model
                summary_dict = self.summariser.summarise_mcint(alpha=alpha)
                metrics_df = aggregate_metrics(self.mcint_true, summary_dict)

            self._metrics_cache[cache_key] = metrics_df

        return self._metrics_cache[cache_key].copy()

    def get_point_estimate_error(
        self, quantity: Literal["cint", "mcint"] = "cint"
    ) -> Dict[str, float]:
        """
        Compute error metrics for point estimates (mean from posterior).

        Parameters
        ----------
        quantity : {'cint', 'mcint'}, default='cint'
            Which quantity to compute errors for

        Returns
        -------
        Dict[str, float]
            Dictionary containing:
            - 'rmse': Root mean squared error
            - 'mae': Mean absolute error
            - 'mape': Mean absolute percentage error
            - 'relative_error': Relative Frobenius norm error

        Notes
        -----
        This method uses the posterior mean instead of median, which may differ
        slightly from the metrics in evaluate_cint() and evaluate_mcint().

        For HiBRC models, errors are aggregated across all categories.

        Examples
        --------
        >>> evaluator = ModelEvaluatorBRC(summariser, cint_true)
        >>> point_errors = evaluator.get_point_estimate_error("cint")
        >>> print(f"Point estimate RMSE: {point_errors['rmse']:.3f}")
        """
        if quantity not in ["cint", "mcint"]:
            raise ValueError(f"quantity must be 'cint' or 'mcint', got {quantity}")

        cache_key = f"point_{quantity}"

        if cache_key not in self._metrics_cache:
            # Get point estimates from summariser
            point_est = self.summariser.get_point_estimates(quantity)
            y_hat = point_est["mean"]

            # Get true values
            if quantity == "cint":
                y_true = self.cint_true
            else:
                y_true = self.mcint_true

            # Compute errors
            if self.model_type == "brc":
                diff = y_hat - y_true
                rmse = float(np.sqrt(np.mean(diff**2)))
                mae = float(np.mean(np.abs(diff)))

                # Avoid division by zero in MAPE
                mask = y_true != 0
                if np.any(mask):
                    mape = float(np.mean(np.abs(diff[mask] / y_true[mask])) * 100)
                else:
                    mape = np.inf

                # Relative error - use Frobenius norm for matrices, L2 norm for vectors
                if diff.ndim == 2:
                    norm_diff = np.linalg.norm(diff, "fro")
                    norm_true = np.linalg.norm(y_true, "fro")
                else:
                    norm_diff = np.linalg.norm(diff)
                    norm_true = np.linalg.norm(y_true)
                relative_error = float(
                    norm_diff / norm_true if norm_true > 0 else np.inf
                )

            else:  # hibrc
                # Aggregate across all categories
                all_diffs = []
                all_true = []

                for var in y_true.keys():
                    for cat in y_true[var].keys():
                        diff = y_hat[var][cat] - y_true[var][cat]
                        all_diffs.append(diff.flatten())
                        all_true.append(y_true[var][cat].flatten())

                all_diffs = np.concatenate(all_diffs)
                all_true = np.concatenate(all_true)

                rmse = float(np.sqrt(np.mean(all_diffs**2)))
                mae = float(np.mean(np.abs(all_diffs)))

                mask = all_true != 0
                if np.any(mask):
                    mape = float(
                        np.mean(np.abs(all_diffs[mask] / all_true[mask])) * 100
                    )
                else:
                    mape = np.inf

                norm_diff = np.linalg.norm(all_diffs)
                norm_true = np.linalg.norm(all_true)
                relative_error = float(
                    norm_diff / norm_true if norm_true > 0 else np.inf
                )

            result = {
                "rmse": rmse,
                "mae": mae,
                "mape": mape,
                "relative_error": relative_error,
            }

            self._metrics_cache[cache_key] = pd.DataFrame([result])

        return self._metrics_cache[cache_key].iloc[0].to_dict()

    def get_cache_info(self) -> Dict[str, any]:
        """
        Get information about cached metrics.

        Returns
        -------
        Dict
            Cache statistics including:
            - 'n_cached': Number of cached results
            - 'cached_metrics': List of cached metric keys
            - 'model_type': Type of model being evaluated
            - 'alpha': Current significance level
        """
        return {
            "n_cached": len(self._metrics_cache),
            "cached_metrics": list(self._metrics_cache.keys()),
            "model_type": self.model_type,
            "alpha": self.alpha,
        }
