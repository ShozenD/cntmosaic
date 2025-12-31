import warnings
from typing import Dict, Literal, Optional

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from sklearn.metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    root_mean_squared_error,
)

from ..summariser._ModelSummariserSocialMix import ModelSummariserSocialMix


def validate_alpha(alpha: float) -> None:
    """Validate alpha parameter."""
    if not 0 < alpha < 1:
        raise ValueError(f"alpha must be in (0, 1), got {alpha}")


def interval_score(y_true, y_low, y_high, alpha):
    """
    Compute the interval score for given true values and interval bounds.
    """
    return np.mean(
        (y_high - y_low)
        + 2 / alpha * (y_low - y_true) * np.maximum(0, y_low - y_true)
        + 2 / alpha * (y_high - y_true) * np.maximum(0, y_high - y_true)
    )


def compute_metrics(y_true, y_est, y_low, y_high):
    """
    Compute RMSE, MAE, and coverage for given true values, estimates, and interval bounds.
    """
    rmse = root_mean_squared_error(y_true, y_est)
    mae = mean_absolute_error(y_true, y_est)
    mape = mean_absolute_percentage_error(y_true, y_est) * 100
    int_score = interval_score(y_true, y_low, y_high, alpha=0.05)
    coverage = np.mean((y_true >= y_low) & (y_true <= y_high)) * 100
    return rmse, mae, mape, int_score, coverage


def process_variable_metrics(
    data_eval: Dict[str, NDArray], data_est: Dict[str, NDArray]
):
    """
    Compute metrics for a single variable across its categories and overall.
    """
    metrics = []
    for key, values in data_est.items():
        rmse, mae, mape, int_score, coverage = compute_metrics(
            data_eval[key], values[1], values[0], values[2]
        )
        metrics.append(
            {
                "cat": key,
                "rmse": rmse,
                "mae": mae,
                "mape": mape,
                "interval_score": int_score,
                "coverage": coverage,
            }
        )

    # Compute overall metrics for the variable
    y_true = np.vstack([data_eval[key] for key in data_est.keys()])
    y_est = np.vstack([values[1] for values in data_est.values()])
    y_low = np.vstack([values[0] for values in data_est.values()])
    y_high = np.vstack([values[2] for values in data_est.values()])

    rmse, mae, mape, int_score, coverage = compute_metrics(y_true, y_est, y_low, y_high)

    metrics.append(
        {
            "cat": "all",
            "rmse": rmse,
            "mae": mae,
            "mape": mape,
            "interval_score": int_score,
            "coverage": coverage,
        }
    )

    return metrics


def aggregate_metrics(
    data_eval: Dict[str, NDArray], data_est: Dict[str, NDArray]
) -> pd.DataFrame:
    """
    Aggregate metrics for all variables and categories, and compute overall metrics.
    """
    all_metrics = []
    all_metrics.extend(process_variable_metrics(data_eval, data_est))

    # Compute overall metrics across all variables and categories
    y_true = np.vstack([data_eval[cat] for cat in data_est.keys()])
    y_est = np.vstack([values[1] for values in data_est.values()])
    y_low = np.vstack([values[0] for values in data_est.values()])
    y_high = np.vstack([values[2] for values in data_est.values()])

    rmse, mae, mape, int_score, coverage = compute_metrics(y_true, y_est, y_low, y_high)
    all_metrics.append(
        {
            "cat": "all",
            "rmse": rmse,
            "mae": mae,
            "mape": mape,
            "interval_score": int_score,
            "coverage": coverage,
        }
    )

    # Combine into a DataFrame
    return pd.DataFrame(all_metrics)


class ModelEvaluatorSocialMix:
    """
    Evaluator for SocialMix model performance against ground truth.

    Computes error metrics and uncertainty quantification statistics for estimated
    contact intensity matrices from bootstrap samples. Supports both unstratified
    and stratified SocialMix models.

    **Evaluation Strategy**: The SocialMix model estimates contact matrices at
    age-group resolution (e.g., 6 coarse bins). To evaluate against fine-grained
    (1-year) ground truth, this evaluator uses **depixilation** to project binned
    bootstrap samples to fine-grained resolution before computing metrics.

    This evaluator follows the same design patterns as ModelEvaluatorPrem for
    consistency, with automatic detection of stratification mode and appropriate
    metric aggregation.

    Parameters
    ----------
    summariser : ModelSummariserSocialMix
        Summariser containing bootstrap results from a fitted SocialMix model.
        Must have pop_data or age_dist for depixilation.
    cint_matrix_true : NDArray or Dict[str, NDArray]
        Ground truth contact intensity matrix at fine-grained (1-year) resolution.
        - For unstratified: NDArray of shape (A_fine, A_fine) where A_fine is max age
        - For stratified: Dict mapping stratum labels to NDArray of shape (A_fine, A_fine)
    alpha : float, default=0.05
        Significance level for interval score and coverage computations.
        Must be in (0, 1).

    Attributes
    ----------
    summariser : ModelSummariserSocialMix
        Reference to the model summariser
    cint_true : NDArray or Dict[str, NDArray]
        Ground truth contact intensity matrices
    mcint_true : NDArray or Dict[str, NDArray]
        Ground truth marginal contact intensities (computed from cint_true)
    alpha : float
        Significance level for credible intervals
    age_bins : AgeBins
        Age bin definition from summariser
    pop_data : PopulationData or None
        Population data from summariser (may be None for legacy usage)
    age_dist : NDArray or None
        Fine-grained age distribution for depixilation

    Examples
    --------
    **Unstratified SocialMix model evaluation:**

    >>> # Fit model
    >>> sm = SocialMix(part_data, cnt_data, age_bins, pop_data)
    >>> sm.run_inference_bootstrap(n_boot=1000, random_state=42)
    >>>
    >>> # Create summariser and evaluator
    >>> summariser = ModelSummariserSocialMix(sm)
    >>> evaluator = ModelEvaluatorSocialMix(summariser, cint_true, alpha=0.05)
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

    **Stratified SocialMix model evaluation:**

    >>> # For stratified models
    >>> sm = SocialMix(part_data, cnt_data, age_bins, pop_data, strat_var='gender')
    >>> sm.run_inference_bootstrap(n_boot=1000, random_state=42)
    >>>
    >>> # True matrices organized by stratification
    >>> cint_true = {
    ...     "M->M": matrix_mm,
    ...     "M->F": matrix_mf,
    ...     "F->M": matrix_fm,
    ...     "F->F": matrix_ff
    ... }
    >>>
    >>> summariser = ModelSummariserSocialMix(sm)
    >>> evaluator = ModelEvaluatorSocialMix(summariser, cint_true, alpha=0.05)
    >>>
    >>> # Returns metrics aggregated by stratum label
    >>> metrics = evaluator.evaluate()

    Notes
    -----
    **Depixilation for Evaluation**:
    The SocialMix model estimates at age-group resolution (e.g., 6 bins), but
    ground truth is often available at fine-grained (1-year) resolution. This
    evaluator automatically applies depixilation to project bootstrap samples
    from age-group to fine-grained resolution before computing metrics.

    **Metrics Computed**:
    - RMSE: Root mean squared error
    - MAE: Mean absolute error
    - MAPE: Mean absolute percentage error
    - Interval score: Uncertainty calibration (penalizes width + coverage failures)
    - Coverage: Proportion of true values within credible intervals

    **Implementation Notes**:
    - Caching is used to avoid redundant computation
    - For stratified models, metrics are computed for each stratum pair and aggregated
    - All metrics are returned as pandas DataFrames for easy analysis
    - Requires PopulationData or age_dist in summariser for depixilation

    See Also
    --------
    ModelSummariserSocialMix : Summarises bootstrap distributions from SocialMix models
    ModelEvaluatorPrem : Similar evaluator for Prem models
    """

    def __init__(
        self,
        summariser: ModelSummariserSocialMix,
        cint_matrix_true: NDArray[np.float64] | Dict[str, NDArray],
        alpha: float = 0.05,
    ) -> None:
        """
        Initialize ModelEvaluatorSocialMix.

        Parameters
        ----------
        summariser : ModelSummariserSocialMix
            Summariser with bootstrap samples from fitted SocialMix model
        cint_matrix_true : NDArray or Dict[str, NDArray]
            True contact intensity matrix (unstratified) or dict of matrices (stratified)
            - Unstratified: NDArray of shape (A, A) at 1-year resolution
            - Stratified: Dict mapping stratum labels to NDArray of shape (A, A)
        alpha : float, default=0.05
            Significance level for interval metrics (must be in (0, 1))

        Raises
        ------
        ValueError
            If alpha not in (0, 1), or if model hasn't been fitted
        TypeError
            If summariser is not ModelSummariserSocialMix instance, or incompatible types
        """
        # Validate inputs
        validate_alpha(alpha)
        self._validate_summariser(summariser)
        self._validate_true_matrix(cint_matrix_true)

        # Store references
        self.summariser = summariser
        self.alpha = alpha

        # Extract attributes from summariser
        self.age_bins = self.summariser.age_bins
        self.pop_data = self.summariser.pop_data
        self.age_dist = self.summariser.age_dist
        self._is_stratified = self.summariser._is_stratified()

        # Normalize cint_true format based on stratification
        if isinstance(cint_matrix_true, dict):
            if not self._is_stratified and "All->All" in cint_matrix_true:
                # Unstratified model with dict input - extract the matrix
                self.cint_true = cint_matrix_true["All->All"]
            else:
                # Stratified model - keep as dict
                self.cint_true = cint_matrix_true
        else:
            # Already an NDArray
            self.cint_true = cint_matrix_true

        # Compute marginal contact intensities from true matrices
        self.mcint_true = self._compute_marginals(self.cint_true)

        # Cache for computed metrics
        self._metrics_cache: Dict[str, pd.DataFrame] = {}

        # Cache for computed metrics
        self._metrics_cache: Dict[str, pd.DataFrame] = {}

    def _validate_summariser(self, summariser: ModelSummariserSocialMix) -> None:
        """Validate that summariser has required bootstrap results."""
        # Check for required attributes
        required_attrs = [
            "sm",
            "age_bins",
            "summarise_cint",
            "summarise_mcint",
        ]
        missing_attrs = [
            attr for attr in required_attrs if not hasattr(summariser, attr)
        ]

        if missing_attrs:
            raise TypeError(
                f"Summariser missing required attributes: {missing_attrs}. "
                f"Expected ModelSummariserSocialMix instance or compatible object."
            )

        if summariser.sm._boot is None:
            raise ValueError(
                "Summariser must have bootstrap results. "
                "Ensure run_inference_bootstrap() was run on the SocialMix model."
            )

    def _validate_true_matrix(self, cint_true: NDArray | Dict[str, NDArray]) -> None:
        """Validate true matrix dimensions and values."""
        if isinstance(cint_true, dict):
            # Stratified case: Dictionary of NDArrays
            for label, matrix in cint_true.items():
                self._validate_single_matrix(matrix, f"cint_true['{label}']")
        else:
            # Unstratified case: validate single matrix
            self._validate_single_matrix(cint_true, "cint_true")

    def _validate_single_matrix(self, matrix: NDArray, name: str) -> None:
        """Validate a single contact intensity matrix."""
        if not isinstance(matrix, np.ndarray):
            raise TypeError(
                f"{name} must be a numpy array, got {type(matrix).__name__}"
            )

        if matrix.ndim != 2:
            raise ValueError(f"{name} must be 2D, got shape {matrix.shape}")

        if matrix.shape[0] != matrix.shape[1]:
            raise ValueError(f"{name} must be square, got shape {matrix.shape}")

        if not np.all(np.isfinite(matrix)):
            raise ValueError(f"{name} contains NaN or Inf values")

        if np.any(matrix < 0):
            raise ValueError(f"{name} contains negative values")

    def _compute_marginals(
        self, cint: NDArray | Dict[str, NDArray]
    ) -> NDArray | Dict[str, NDArray]:
        """
        Compute marginal contact intensities from contact intensity matrices.

        Marginal contact intensity is the sum over contact ages (axis=1).

        Parameters
        ----------
        cint : NDArray or Dict[str, NDArray]
            Contact intensity matrix(ces)

        Returns
        -------
        NDArray or Dict[str, NDArray]
            Marginal contact intensities
        """
        if isinstance(cint, dict):
            # Stratified case
            return {label: matrix.sum(axis=1) for label, matrix in cint.items()}
        else:
            # Unstratified case
            return cint.sum(axis=1)

    def evaluate(self, alpha: Optional[float] = None) -> pd.DataFrame:
        """
        Compute all evaluation metrics.

        This is a convenience method that calls both evaluate_cint() and
        evaluate_mcint() and combines their results.

        Parameters
        ----------
        alpha : float, optional
            Significance level for interval metrics. If None, uses self.alpha

        Returns
        -------
        pd.DataFrame
            DataFrame containing all metrics:
            - For unstratified: Rows with cint and mcint metrics
            - For stratified: Multiple rows with metrics per stratum label,
              plus overall aggregated metrics

        Notes
        -----
        Metrics computed include:
        - RMSE: Root mean squared error
        - MAE: Mean absolute error
        - MAPE: Mean absolute percentage error
        - interval_score: Negatively oriented interval score (lower is better)
        - coverage: Percentage of true values within credible intervals

        Examples
        --------
        >>> evaluator = ModelEvaluatorSocialMix(summariser, cint_true, alpha=0.05)
        >>> metrics = evaluator.evaluate()
        >>> print(metrics)
        """
        if alpha is None:
            alpha = self.alpha

        cint_metrics = self.evaluate_cint(alpha=alpha)
        mcint_metrics = self.evaluate_mcint(alpha=alpha)

        # Add metric type column for clarity
        cint_metrics["metric_type"] = "cint"
        mcint_metrics["metric_type"] = "mcint"

        # Combine
        all_metrics = pd.concat([cint_metrics, mcint_metrics], ignore_index=True)

        return all_metrics

    def evaluate_cint(self, alpha: Optional[float] = None) -> pd.DataFrame:
        """
        Evaluate the bootstrap contact intensity matrix.

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
            - For unstratified: Single row DataFrame
            - For stratified: One row per stratum label, plus aggregated "all" row

        Notes
        -----
        Results are cached to avoid redundant computation when called multiple times
        with the same alpha value.

        Examples
        --------
        >>> evaluator = ModelEvaluatorSocialMix(summariser, cint_true)
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
            if not self._is_stratified:
                # Unstratified SocialMix model
                # Depixilate binned estimates to fine-grained resolution for comparison
                summary = self.summariser.summarise_cint(
                    alpha=alpha, return_depixilated=True
                )

                y_true = self.cint_true
                y_est = summary[1]  # median
                y_low = summary[0]  # lower bound
                y_high = summary[2]  # upper bound

                rmse, mae, mape, int_score, coverage = compute_metrics(
                    y_true, y_est, y_low, y_high
                )

                metrics_df = pd.DataFrame(
                    {
                        "cat": ["all"],
                        "rmse": [rmse],
                        "mae": [mae],
                        "mape": [mape],
                        "interval_score": [int_score],
                        "coverage": [coverage],
                    }
                )

            else:  # Stratified
                # Stratified SocialMix model
                # Depixilate binned estimates to fine-grained resolution for comparison
                summary_dict = self.summariser.summarise_cint(
                    alpha=alpha, return_depixilated=True
                )

                # summary_dict: {label: NDArray(3, A, A)}
                metrics_df = aggregate_metrics(self.cint_true, summary_dict)

            self._metrics_cache[cache_key] = metrics_df

        return self._metrics_cache[cache_key].copy()

    def evaluate_mcint(self, alpha: Optional[float] = None) -> pd.DataFrame:
        """
        Evaluate the bootstrap marginal contact intensity.

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
        >>> evaluator = ModelEvaluatorSocialMix(summariser, cint_true)
        >>> mcint_metrics = evaluator.evaluate_mcint(alpha=0.05)
        >>> print(f"Marginal RMSE: {mcint_metrics['rmse'].values[0]:.3f}")
        """
        if alpha is None:
            alpha = self.alpha
        else:
            validate_alpha(alpha)

        cache_key = f"mcint_alpha{alpha}"

        if cache_key not in self._metrics_cache:
            if not self._is_stratified:
                # Unstratified SocialMix model
                # Use depixilation to project binned estimates to fine-grained resolution
                summary = self.summariser.summarise_mcint(
                    alpha=alpha, return_depixilated=True
                )
                y_true = self.mcint_true
                y_est = summary[1]  # median
                y_low = summary[0]  # lower bound
                y_high = summary[2]  # upper bound

                rmse, mae, mape, int_score, coverage = compute_metrics(
                    y_true, y_est, y_low, y_high
                )

                metrics_df = pd.DataFrame(
                    {
                        "cat": ["all"],
                        "rmse": [rmse],
                        "mae": [mae],
                        "mape": [mape],
                        "interval_score": [int_score],
                        "coverage": [coverage],
                    }
                )

            else:  # Stratified
                # Stratified SocialMix model
                # Use depixilation to project binned estimates to fine-grained resolution
                summary_dict = self.summariser.summarise_mcint(
                    alpha=alpha, return_depixilated=True
                )
                metrics_df = aggregate_metrics(self.mcint_true, summary_dict)

            self._metrics_cache[cache_key] = metrics_df

        return self._metrics_cache[cache_key].copy()

    def clear_cache(self) -> None:
        """Clear the metrics cache."""
        self._metrics_cache.clear()

    def get_cache_info(self) -> Dict[str, int]:
        """
        Get information about cached metrics.

        Returns
        -------
        Dict[str, int]
            Dictionary with cache statistics
        """
        return {"cached_items": len(self._metrics_cache)}
