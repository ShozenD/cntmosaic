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

from ..models import BRCfine, BRCrefine, HiBRCfine, HiBRCrefine
from ..utils import AgeBins, depixilate, pixilate
from ._summariser import (
    ModelSummariserBRC,
    ModelSummariserPrem,
    ModelSummariserSocialMix,
)


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


class ModelEvaluatorBRC:
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
        # Validate inputs
        validate_alpha(alpha)
        self._validate_summariser(summariser)
        self._validate_true_matrix(cint_matrix_true)

        # Store references
        self.summariser = summariser
        self.cint_true = cint_matrix_true
        self.alpha = alpha

        # Detect model type
        self.model_type = self.summariser.model_type

        # Compute marginal contact intensities from true matrices
        self.mcint_true = self._compute_marginals(cint_matrix_true)

        # Cache for computed metrics
        self._metrics_cache: Dict[str, pd.DataFrame] = {}

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

    def _validate_true_matrix(self, cint_true: NDArray | Dict[str, NDArray]) -> None:
        """Validate true matrix dimensions and values."""
        if isinstance(cint_true, dict):
            # HiBRC case: Dictionary of NDArrays
            for key, values in cint_true.items():
                if not isinstance(values, np.ndarray):
                    raise TypeError(
                        f"For HiBRC models, cint_true must be dict of dicts. "
                        f"Variable '{key}' is not a dict."
                    )
                self._validate_single_matrix(values, f"{key}")
        else:
            # BRC case: validate single matrix
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
        """Compute marginal contact intensities by summing over contact age."""
        if isinstance(cint, dict):
            # HiBRC case: compute for each category
            mcint = {}
            for key in cint.keys():
                mcint[key] = cint[key].sum(axis=1)
            return mcint
        else:
            # BRC case: simple sum
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
            - For BRC models: Single row with cint and mcint metrics
            - For HiBRC models: Multiple rows with metrics per variable/category,
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
        >>> evaluator = ModelEvaluatorBRC(summariser, cint_true, alpha=0.05)
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
                y_true = self.cint_true
                y_est = summary[1]  # median
                y_low = summary[0]  # lower bound
                y_high = summary[2]  # upper bound

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
                y_true = self.mcint_true
                y_est = summary[1]  # median
                y_low = summary[0]  # lower bound
                y_high = summary[2]  # upper bound

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

    def clear_cache(self) -> None:
        """Clear all cached metric computations."""
        self._metrics_cache.clear()

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


class ModelEvaluatorSocialMix:
    """
    Evaluator for SocialMix model performance against ground truth.

    Computes various error metrics and uncertainty quantification statistics
    for estimated contact intensity matrices compared to a known true matrix.

    Parameters
    ----------
    summariser : ModelSummariserSocialMix
        Summariser containing bootstrap results and point estimates
    cint_matrix_true : NDArray
        True contact intensity matrix at 1-year age resolution.
        Shape should be (max_age, max_age) where max_age matches the
        population age distribution.
    alpha : float, default=0.05
        Significance level for interval score and coverage computations

    Attributes
    ----------
    m_true : NDArray
        True contact intensity matrix (1-year resolution)
    pix_m_true : NDArray
        True matrix aggregated to effective age bins
    depix_m_true : NDArray
        Reconstructed true matrix from aggregated version (for discretization error)
    m_hat : NDArray
        Estimated contact intensity matrix (effective age bins)

    Examples
    --------
    >>> sm = SocialMix(...)
    >>> sm.run_bootstrap(n_boot=1000)
    >>> summariser = ModelSummariserSocialMix(sm)
    >>>
    >>> # Assume we have ground truth from simulation
    >>> evaluator = ModelEvaluatorSocialMix(summariser, m_true, alpha=0.05)
    >>>
    >>> # Get all metrics
    >>> metrics = evaluator.evaluate()
    >>> print(metrics)
    >>>
    >>> # Get individual metrics
    >>> disc_err = evaluator.compute_discretization_error()
    >>> est_err = evaluator.compute_estimation_error()
    >>> coverage = evaluator.compute_coverage()
    """

    def __init__(
        self,
        summariser: ModelSummariserSocialMix,
        cint_matrix_true: NDArray[np.float64],
        alpha: float = 0.05,
    ):
        """
        Initialize evaluator with summariser and ground truth.

        Parameters
        ----------
        summariser : ModelSummariserSocialMix
            Summariser with bootstrap results
        cint_matrix_true : NDArray
            True contact intensity matrix (1-year resolution)
        alpha : float, default=0.05
            Significance level for interval metrics

        Raises
        ------
        ValueError
            If dimensions are incompatible or inputs invalid
        """
        # Validate inputs
        validate_alpha(alpha)
        self._validate_summariser(summariser)
        self._validate_true_matrix(cint_matrix_true, summariser)

        # Store references
        self.summariser = summariser
        self.alpha = alpha
        self.age_bins = summariser.effective_age_bins
        self.age_dist = summariser.age_dist

        # Store true matrix and transformations
        self.m_true = cint_matrix_true.astype(np.float64)

        # Aggregate true matrix to effective bins (pixilate)
        self.pix_m_true = pixilate(self.m_true, self.age_bins, self.age_dist)

        # Reconstruct from aggregated (for discretization error computation)
        self.depix_m_true = depixilate(self.pix_m_true, self.age_bins, self.age_dist)

        # Get point estimate from model
        self.m_hat = summariser.sm.compute_cint(recover_bins=False)

        # Cache for computed metrics
        self._metrics_cache: Dict[str, float] = {}

    def _validate_summariser(self, summariser: ModelSummariserSocialMix) -> None:
        """Validate that summariser has required bootstrap results."""
        if not hasattr(summariser.sm, "_boot") or summariser.sm._boot is None:
            raise ValueError(
                "Summariser must have bootstrap results. "
                "Run sm.run_bootstrap() before creating evaluator."
            )

    def _validate_true_matrix(
        self, m_true: NDArray, summariser: ModelSummariserSocialMix
    ) -> None:
        """Validate true matrix dimensions and values."""
        # Check dimensionality
        if m_true.ndim != 2:
            raise ValueError(f"True matrix must be 2D, got shape {m_true.shape}")

        if m_true.shape[0] != m_true.shape[1]:
            raise ValueError(f"True matrix must be square, got shape {m_true.shape}")

        # Check compatibility with age distribution
        expected_size = len(summariser.age_dist)
        if m_true.shape[0] != expected_size:
            raise ValueError(
                f"True matrix size ({m_true.shape[0]}) must match "
                f"age distribution length ({expected_size})"
            )

        # Check for valid values
        if not np.all(np.isfinite(m_true)):
            raise ValueError("True matrix contains NaN or Inf values")

        if np.any(m_true < 0):
            raise ValueError("True matrix contains negative values")

    def compute_discretization_error(self) -> float:
        """
        Compute discretization error (information loss from aggregation).

        Measures MSE between true fine-grained matrix and its reconstruction
        after aggregating to coarse bins and depixilating back:

        disc_error = MSE(depix(pix(M_true)), M_true)

        This quantifies the inherent information loss from using coarser
        age bins, independent of estimation quality.

        Returns
        -------
        error : float
            Mean squared discretization error

        Notes
        -----
        This error is deterministic (no bootstrap uncertainty) and represents
        a lower bound on achievable error for the given age binning.
        """
        if "discretization" not in self._metrics_cache:
            error = np.mean(np.square(self.depix_m_true - self.m_true))
            self._metrics_cache["discretization"] = float(error)

        return self._metrics_cache["discretization"]

    def compute_estimation_error(self) -> float:
        """
        Compute estimation error at the effective bin level.

        Measures MSE between estimated matrix and true matrix, both at
        the effective (possibly merged) bin resolution:

        est_error = MSE(M_hat, pix(M_true))

        This isolates the error due to finite sampling and estimation,
        excluding discretization effects.

        Returns
        -------
        error : float
            Mean squared estimation error

        Notes
        -----
        This error reflects the quality of the estimation procedure
        (survey sampling, bootstrap, etc.) at the working resolution.
        """
        if "estimation" not in self._metrics_cache:
            error = np.mean(np.square(self.m_hat - self.pix_m_true))
            self._metrics_cache["estimation"] = float(error)

        return self._metrics_cache["estimation"]

    def compute_total_error(self) -> float:
        """
        Compute total error at fine-grained resolution.

        Measures MSE between depixilated estimate and true matrix:

        total_error = MSE(depix(M_hat), M_true)

        Returns
        -------
        error : float
            Mean squared total error

        Notes
        -----
        Due to nonlinearity of depixilation:
        total_error ≈ discretization_error + estimation_error

        The approximation improves when:
        - Age bins are not too coarse
        - Population distribution is smooth
        - Estimation error is small

        For exact decomposition, use the individual error components.
        """
        if "total" not in self._metrics_cache:
            # Depixilate estimate to fine-grained resolution
            depix_m_hat = depixilate(self.m_hat, self.age_bins, self.age_dist)
            error = np.mean(np.square(depix_m_hat - self.m_true))
            self._metrics_cache["total"] = float(error)

        return self._metrics_cache["total"]

    def compute_interval_score(self, alpha: Optional[float] = None) -> float:
        """
        Compute interval score for uncertainty quantification.

        The interval score is a proper scoring rule for interval forecasts
        that balances width and coverage. Lower scores are better.

        IS = (U - L) + (2/α) * (L - y) * I(y < L) + (2/α) * (y - U) * I(y > U)

        where:
        - U, L: upper and lower bounds of (1-α) prediction interval
        - y: true value
        - α: significance level

        Parameters
        ----------
        alpha : float, optional
            Significance level. If None, uses self.alpha

        Returns
        -------
        score : float
            Mean interval score across all matrix elements

        Notes
        -----
        The interval score penalizes:
        1. Wide intervals (U - L term)
        2. True values outside interval (penalty terms)

        For perfect calibration at level α, the penalty terms should
        contribute approximately α * mean_width to the total score.

        References
        ----------
        Gneiting, T., & Raftery, A. E. (2007). Strictly proper scoring rules,
        prediction, and estimation. Journal of the American Statistical
        Association, 102(477), 359-378.
        """
        if alpha is None:
            alpha = self.alpha
        else:
            validate_alpha(alpha)

        cache_key = f"interval_score_{alpha}"

        if cache_key not in self._metrics_cache:
            # Get confidence bounds (depixilated to match m_true resolution)
            summary = self.summariser.summarise_cint(
                alpha=alpha, return_depixilated=True
            )
            lower = summary["lower"]
            upper = summary["upper"]

            # Compute interval score components
            width = upper - lower

            # Lower penalty: true value below interval
            lower_penalty = (2 / alpha) * np.maximum(0, lower - self.m_true)

            # Upper penalty: true value above interval
            upper_penalty = (2 / alpha) * np.maximum(0, self.m_true - upper)

            # Total score (mean over all elements)
            score = np.mean(width + lower_penalty + upper_penalty)

            self._metrics_cache[cache_key] = float(score)

        return self._metrics_cache[cache_key]

    def compute_coverage(self, alpha: Optional[float] = None) -> float:
        """
        Compute empirical coverage of confidence intervals.

        Computes the proportion of true matrix elements that fall within
        their corresponding (1-α) confidence intervals.

        Parameters
        ----------
        alpha : float, optional
            Significance level. If None, uses self.alpha

        Returns
        -------
        coverage : float
            Empirical coverage as percentage (0-100)

        Notes
        -----
        For well-calibrated intervals at level α:
        - Expected coverage ≈ (1 - α) * 100%
        - For α=0.05, expect ≈ 95% coverage

        Undercoverage (< expected) suggests:
        - Underestimated uncertainty
        - Optimistic confidence intervals

        Overcoverage (> expected) suggests:
        - Overestimated uncertainty
        - Conservative confidence intervals
        """
        if alpha is None:
            alpha = self.alpha
        else:
            validate_alpha(alpha)

        cache_key = f"coverage_{alpha}"

        if cache_key not in self._metrics_cache:
            # Get confidence bounds (depixilated)
            summary = self.summariser.summarise_cint(
                alpha=alpha, return_depixilated=True
            )
            lower = summary["lower"]
            upper = summary["upper"]

            # Check coverage element-wise
            covered = (self.m_true >= lower) & (self.m_true <= upper)
            coverage = np.mean(covered) * 100  # Convert to percentage

            self._metrics_cache[cache_key] = float(coverage)

        return self._metrics_cache[cache_key]

    def compute_rmse(
        self, error_type: Literal["total", "estimation", "discretization"] = "total"
    ) -> float:
        """
        Compute root mean squared error.

        Parameters
        ----------
        error_type : {'total', 'estimation', 'discretization'}
            Which error type to compute RMSE for

        Returns
        -------
        rmse : float
            Root mean squared error
        """
        if error_type == "total":
            mse = self.compute_total_error()
        elif error_type == "estimation":
            mse = self.compute_estimation_error()
        elif error_type == "discretization":
            mse = self.compute_discretization_error()
        else:
            raise ValueError(f"Unknown error_type: {error_type}")

        return np.sqrt(mse)

    def compute_mae(
        self, error_type: Literal["total", "estimation", "discretization"] = "total"
    ) -> float:
        """
        Compute mean absolute error.

        Parameters
        ----------
        error_type : {'total', 'estimation', 'discretization'}
            Which error to compute MAE for

        Returns
        -------
        mae : float
            Mean absolute error
        """
        cache_key = f"mae_{error_type}"

        if cache_key not in self._metrics_cache:
            if error_type == "total":
                depix_m_hat = depixilate(self.m_hat, self.age_bins, self.age_dist)
                diff = depix_m_hat - self.m_true
            elif error_type == "estimation":
                diff = self.m_hat - self.pix_m_true
            elif error_type == "discretization":
                diff = self.depix_m_true - self.m_true
            else:
                raise ValueError(f"Unknown error_type: {error_type}")

            mae = np.mean(np.abs(diff))
            self._metrics_cache[cache_key] = float(mae)

        return self._metrics_cache[cache_key]

    def compute_relative_error(self) -> float:
        """
        Compute relative error (normalized by true matrix Frobenius norm).

        relative_error = ||M_hat - M_true||_F / ||M_true||_F

        Returns
        -------
        rel_error : float
            Relative Frobenius norm error

        Notes
        -----
        This metric is scale-invariant and useful for comparing
        performance across different contact matrices with different
        magnitudes.
        """
        if "relative" not in self._metrics_cache:
            depix_m_hat = depixilate(self.m_hat, self.age_bins, self.age_dist)

            numerator = np.linalg.norm(depix_m_hat - self.m_true, "fro")
            denominator = np.linalg.norm(self.m_true, "fro")

            if denominator == 0:
                warnings.warn(
                    "True matrix has zero norm. Relative error undefined.", UserWarning
                )
                rel_error = np.inf
            else:
                rel_error = numerator / denominator

            self._metrics_cache["relative"] = float(rel_error)

        return self._metrics_cache["relative"]

    def evaluate(
        self, alpha: Optional[float] = None, include_extended: bool = False
    ) -> pd.DataFrame:
        """
        Compute all evaluation metrics and return as DataFrame.

        Parameters
        ----------
        alpha : float, optional
            Significance level for interval metrics. If None, uses self.alpha
        include_extended : bool, default=False
            Whether to include extended metrics (RMSE, MAE, relative error)

        Returns
        -------
        metrics : pd.DataFrame
            Single-row DataFrame with all metrics

        Examples
        --------
        >>> metrics = evaluator.evaluate(alpha=0.05)
        >>> print(metrics)
           discretization_error  estimation_error  total_error  interval_score  coverage
        0              0.123              0.456         0.579            2.34      94.2

        >>> # With extended metrics
        >>> metrics_ext = evaluator.evaluate(include_extended=True)
        >>> print(metrics_ext.columns)
        Index(['discretization_error', 'estimation_error', 'total_error',
               'interval_score', 'coverage', 'rmse_total', 'rmse_estimation',
               'mae_total', 'relative_error'], dtype='object')
        """
        if alpha is None:
            alpha = self.alpha

        # Core metrics
        metrics = {
            "alpha": alpha,
            "discretization_error": self.compute_discretization_error(),
            "estimation_error": self.compute_estimation_error(),
            "total_error": self.compute_total_error(),
            "interval_score": self.compute_interval_score(alpha),
            "coverage": self.compute_coverage(alpha),
        }

        # Extended metrics
        if include_extended:
            metrics.update(
                {
                    "rmse_total": self.compute_rmse("total"),
                    "rmse_estimation": self.compute_rmse("estimation"),
                    "rmse_discretization": self.compute_rmse("discretization"),
                    "mae_total": self.compute_mae("total"),
                    "mae_estimation": self.compute_mae("estimation"),
                    "relative_error": self.compute_relative_error(),
                }
            )

        return pd.DataFrame([metrics])

    def get_error_decomposition(self) -> Dict[str, float]:
        """
        Get error decomposition with approximation quality check.

        Returns
        -------
        decomposition : dict
            Dictionary containing:
            - 'discretization': Discretization error
            - 'estimation': Estimation error
            - 'total': Total error
            - 'sum_components': Sum of discretization + estimation
            - 'decomp_residual': total - sum_components
            - 'decomp_quality': 1 - |residual| / total (closer to 1 is better)

        Notes
        -----
        The decomposition residual quantifies the nonlinear interaction
        between discretization and estimation errors. Small residuals
        (<10% of total) indicate good linear approximation.
        """
        disc = self.compute_discretization_error()
        est = self.compute_estimation_error()
        total = self.compute_total_error()
        sum_components = disc + est
        residual = total - sum_components

        quality = 1 - abs(residual) / total if total > 0 else 0

        return {
            "discretization": disc,
            "estimation": est,
            "total": total,
            "sum_components": sum_components,
            "decomp_residual": residual,
            "decomp_quality": quality,
        }

    def clear_cache(self) -> None:
        """Clear all cached metric computations."""
        self._metrics_cache.clear()

    def get_cache_info(self) -> Dict[str, int]:
        """
        Get information about cached metrics.

        Returns
        -------
        info : dict
            Cache statistics
        """
        return {
            "n_cached": len(self._metrics_cache),
            "cached_metrics": list(self._metrics_cache.keys()),
        }


class ModelEvaluatorPrem:
    """
    Evaluator for Prem model performance against ground truth.

    Computes various error metrics and uncertainty quantification statistics
    for estimated contact intensity matrices from Bayesian posterior samples.

    Parameters
    ----------
    summariser : ModelSummariserPrem
        Summariser containing posterior samples and point estimates.
    cint_matrix_true : NDArray
        True contact intensity matrix at 1-year age resolution.
        Shape should be (max_age, max_age) where max_age matches the
        population age distribution.
    alpha : float, default=0.05
        Significance level for interval score and coverage computations
    symmetric : bool, default=False
        Whether to symmetrize the estimated contact matrix before evaluation.

    Attributes
    ----------
    m_true : NDArray
        True contact intensity matrix (1-year resolution)
    pix_m_true : NDArray
        True matrix aggregated to age group bins
    depix_m_true : NDArray
        Reconstructed true matrix from aggregated version
    m_hat : NDArray
        Estimated contact intensity matrix (median from posterior)

    Examples
    --------
    >>> prem = Prem(df_part, df_cnt, age_bins)
    >>> prem.run_inference_mcmc(rng_key, num_samples=1000)
    >>> summariser = ModelSummariserPrem(prem, df_age_dist)
    >>>
    >>> # Evaluate against ground truth
    >>> evaluator = ModelEvaluatorPrem(summariser, m_true, alpha=0.05)
    >>>
    >>> # Get all metrics
    >>> metrics = evaluator.evaluate()
    >>> print(metrics)
    >>>
    >>> # Get individual metrics
    >>> disc_err = evaluator.compute_discretization_error()
    >>> est_err = evaluator.compute_estimation_error()
    >>> coverage = evaluator.compute_coverage()
    """

    def __init__(
        self,
        summariser: ModelSummariserPrem,
        cint_matrix_true: NDArray[np.float64],
        alpha: float = 0.05,
        symmetric: bool = False,
    ) -> None:
        """
        Initialize ModelEvaluatorPrem.

        Parameters
        ----------
        summariser : ModelSummariserPrem
            Summariser with MCMC/SVI posterior samples
        cint_matrix_true : NDArray
            True contact intensity matrix (1-year resolution)
        alpha : float, default=0.05
            Significance level for interval metrics
        symmetric : bool, default=False
            Whether to apply reciprocity adjustment to estimates

        Raises
        ------
        ValueError
            If alpha not in (0, 1), or if matrix dimensions incompatible
        TypeError
            If summariser is not ModelSummariserPrem instance
        """
        # Validate inputs
        validate_alpha(alpha)
        self._validate_summariser(summariser)
        self._validate_true_matrix(cint_matrix_true, summariser)

        # Store references
        self.summariser = summariser
        self.alpha = alpha
        self.symmetric = symmetric
        self.age_bins = summariser.age_bins

        # Get age_dist - handle the case where it might be None
        if summariser.age_dist is None:
            raise ValueError(
                "Summariser must have age_dist (1-year population distribution). "
                "Pass df_age_dist when creating ModelSummariserPrem."
            )
        self.age_dist = summariser.age_dist

        # Store true matrix and transformations
        self.m_true = cint_matrix_true.astype(np.float64)

        # Aggregate true matrix to age group bins (pixilate)
        self.pix_m_true = pixilate(self.m_true, self.age_bins, self.age_dist)

        # Reconstruct from age groups back to 1-year (depixilate)
        self.depix_m_true = depixilate(self.pix_m_true, self.age_bins, self.age_dist)

        # Get point estimate from summariser
        summary_cint = self.summariser.summarise_cint(return_symmetrized=self.symmetric)
        self.m_hat = summary_cint["median"]

        # Cache for computed metrics
        self._metrics_cache: Dict[str, float] = {}

    def _validate_summariser(self, summariser: ModelSummariserPrem) -> None:
        """Validate that summariser has required posterior samples."""
        if not isinstance(summariser, ModelSummariserPrem):
            raise TypeError(
                f"Expected ModelSummariserPrem instance, got {type(summariser).__name__}. "
                f"Usage: summariser = ModelSummariserPrem(prem, df_age_dist); "
                f"evaluator = ModelEvaluatorPrem(summariser, m_true)"
            )

        if summariser.post_cint_samples is None:
            raise ValueError(
                "Summariser must have posterior samples. "
                "Ensure MCMC or SVI inference was run on the Prem model."
            )

    def _validate_true_matrix(
        self, m_true: NDArray, summariser: ModelSummariserPrem
    ) -> None:
        """Validate true matrix dimensions and values."""
        # Check dimensionality
        if m_true.ndim != 2:
            raise ValueError(f"True matrix must be 2D, got shape {m_true.shape}")

        if m_true.shape[0] != m_true.shape[1]:
            raise ValueError(f"True matrix must be square, got shape {m_true.shape}")

        # Check compatibility with age distribution
        if summariser.age_dist is not None:
            expected_size = len(summariser.age_dist)
            if m_true.shape[0] != expected_size:
                raise ValueError(
                    f"True matrix size ({m_true.shape[0]}) must match "
                    f"age distribution length ({expected_size}). "
                    f"Ensure true matrix is at 1-year age resolution."
                )

        # Check for valid values
        if not np.all(np.isfinite(m_true)):
            raise ValueError("True matrix contains NaN or Inf values")

        if np.any(m_true < 0):
            raise ValueError("True matrix contains negative values")

    def evaluate(
        self,
        alpha: Optional[float] = None,
        symmetric: Optional[bool] = None,
        include_extended: bool = False,
    ) -> Dict[str, float]:
        """
        Compute all evaluation metrics.

        Parameters
        ----------
        alpha : float, optional
            Significance level for interval metrics. If None, uses self.alpha
        symmetric : bool, optional
            Whether to symmetrize the estimated contact matrix before evaluation.
            If None, uses self.symmetric
        include_extended : bool, default=False
            Whether to include extended metrics (RMSE, MAE, relative error)

        Returns
        -------
        Dict[str, float]
            Dictionary containing:
            - 'discretization_error': MSE between true matrix and depixilated version
            - 'estimation_error': MSE between estimate and pixilated true matrix
            - 'total_error': Sum of discretization and estimation errors
            - 'interval_score': Negatively oriented interval score (lower is better)
            - 'coverage': Percentage of true values within credible intervals

        Notes
        -----
        Total error = discretization error + estimation error
        This equals MSE(depixilate(m_hat), m_true)
        """
        if alpha is None:
            alpha = self.alpha

        if symmetric is None:
            symmetric = self.symmetric

        metrics = {
            "discretization_error": self.compute_discretization_error(),
            "estimation_error": self.compute_estimation_error(symmetric=symmetric),
            "total_error": self.compute_total_error(symmetric=symmetric),
            "interval_score": self.compute_interval_score(
                alpha=alpha, symmetric=symmetric
            ),
            "coverage": self.compute_coverage(alpha=alpha, symmetric=symmetric),
        }

        # Extended metrics
        if include_extended:
            metrics.update(
                {
                    "rmse_total": self.compute_rmse("total", symmetric=symmetric),
                    "rmse_estimation": self.compute_rmse(
                        "estimation", symmetric=symmetric
                    ),
                    "rmse_discretization": self.compute_rmse("discretization"),
                    "mae_total": self.compute_mae("total", symmetric=symmetric),
                    "mae_estimation": self.compute_mae(
                        "estimation", symmetric=symmetric
                    ),
                    "relative_error": self.compute_relative_error(symmetric=symmetric),
                }
            )

        return pd.DataFrame([metrics])

    def compute_discretization_error(self) -> float:
        """
        Compute discretization error (MSE from aggregation/disaggregation).

        Returns
        -------
        float
            Mean squared error: MSE(depixilate(pixilate(m_true)), m_true)

        Notes
        -----
        This quantifies information loss from the pixilation-depixilation process.
        Lower values indicate less loss from age grouping.
        """
        return float(np.mean(np.square(self.depix_m_true - self.m_true)))

    def compute_estimation_error(self, symmetric: Optional[bool] = None) -> float:
        """
        Compute estimation error (MSE at age group level).

        Parameters
        ----------
        symmetric : bool, optional
            Whether to use symmetrized estimates. If None, uses self.symmetric

        Returns
        -------
        float
            Mean squared error: MSE(m_hat, pixilate(m_true))

        Notes
        -----
        This quantifies the model's ability to estimate the aggregated
        contact matrix at the age group level, independent of resolution loss.
        """
        if symmetric is None:
            symmetric = self.symmetric

        cache_key = f"estimation_sym{symmetric}"

        if cache_key not in self._metrics_cache:
            # Get estimate with appropriate symmetrization
            summary_cint = self.summariser.summarise_cint(return_symmetrized=symmetric)
            m_hat = summary_cint["median"]

            error = np.mean(np.square(m_hat - self.pix_m_true))
            self._metrics_cache[cache_key] = float(error)

        return self._metrics_cache[cache_key]

    def compute_total_error(self, symmetric: Optional[bool] = None) -> float:
        """
        Compute total error (discretization + estimation).

        Parameters
        ----------
        symmetric : bool, optional
            Whether to use symmetrized estimates. If None, uses self.symmetric

        Returns
        -------
        float
            Sum of discretization and estimation errors.
            Equivalent to MSE(depixilate(m_hat), m_true).

        Notes
        -----
        This represents the overall MSE between the reconstructed estimate
        and the true matrix at 1-year age resolution.
        """
        if symmetric is None:
            symmetric = self.symmetric

        cache_key = f"total_sym{symmetric}"

        if cache_key not in self._metrics_cache:
            # Get estimate with appropriate symmetrization
            summary_cint = self.summariser.summarise_cint(return_symmetrized=symmetric)
            m_hat = summary_cint["median"]

            # Depixilate estimate to fine-grained resolution
            depix_m_hat = depixilate(m_hat, self.age_bins, self.age_dist)
            error = np.mean(np.square(depix_m_hat - self.m_true))
            self._metrics_cache[cache_key] = float(error)

        return self._metrics_cache[cache_key]

    def compute_interval_score(
        self,
        alpha: Optional[float] = None,
        symmetric: bool = False,
    ) -> float:
        """
        Compute negatively oriented interval score.

        Parameters
        ----------
        alpha : float, optional
            Significance level for credible intervals if none uses self.alpha
        symmetric : bool, default=False
            Whether to symmetrize the matrix estimates before evaluation.

        Returns
        -------
        float
            Interval score averaged over all matrix elements (lower is better)

        Notes
        -----
        The interval score is defined as:
            IS = (u - l) + (2/α) * (l - y) * I(y < l) + (2/α) * (y - u) * I(y > u)

        where:
        - l, u are the lower and upper α-level credible interval bounds
        - y is the true value
        - α is the significance level
        - I() is the indicator function

        This score rewards narrow intervals while penalizing coverage failures.
        """
        if alpha is None:
            alpha = self.alpha
        else:
            validate_alpha(alpha)

        if symmetric is None:
            symmetric = self.symmetric

        # Get depixilated summaries for interval score computation
        summary_depix = self.summariser.summarise_cint(
            alpha=alpha, return_depixilated=True, return_symmetrized=symmetric
        )
        depix_lower = summary_depix["lower"]
        depix_upper = summary_depix["upper"]

        penalty_factor = 2.0 / alpha
        score = (
            (depix_upper - depix_lower)
            + penalty_factor
            * (depix_lower - self.m_true)
            * np.maximum(0, depix_lower - self.m_true)
            + penalty_factor
            * (self.m_true - depix_upper)
            * np.maximum(0, self.m_true - depix_upper)
        )

        return float(np.mean(score))

    def compute_coverage(
        self, alpha: Optional[float] = None, symmetric: bool = False
    ) -> float:
        """
        Compute empirical coverage of credible intervals.

        Parameters
        ----------
        alpha : float, optional
            Significance level for credible intervals if none uses self.alpha
        symmetric : bool, default=False
            Whether to symmetrize the matrix estimates before evaluation.

        Returns
        -------
        float
            Percentage of true matrix elements falling within credible intervals

        Notes
        -----
        For well-calibrated intervals, coverage should be close to (1-α)*100%.
        For α=0.05, expect approximately 95% coverage.
        """
        if alpha is None:
            alpha = self.alpha
        else:
            validate_alpha(alpha)

        if symmetric is None:
            symmetric = self.symmetric

        cache_key = f"coverage_{alpha}"

        if cache_key not in self._metrics_cache:
            # Get confidence bounds (depixilated)
            summary_depix = self.summariser.summarise_cint(
                alpha=alpha, return_depixilated=True, return_symmetrized=symmetric
            )
            depix_lower = summary_depix["lower"]
            depix_upper = summary_depix["upper"]

            in_interval = (self.m_true >= depix_lower) & (self.m_true <= depix_upper)

            self._metrics_cache[cache_key] = float(np.mean(in_interval) * 100.0)

        return self._metrics_cache[cache_key]

    def compute_rmse(
        self,
        error_type: Literal["total", "estimation", "discretization"] = "total",
        symmetric: Optional[bool] = None,
    ) -> float:
        """
        Compute root mean squared error.

        Parameters
        ----------
        error_type : {'total', 'estimation', 'discretization'}
            Which error type to compute RMSE for
        symmetric : bool, optional
            Whether to use symmetrized estimates. If None, uses self.symmetric

        Returns
        -------
        rmse : float
            Root mean squared error
        """
        if error_type == "total":
            mse = self.compute_total_error(symmetric=symmetric)
        elif error_type == "estimation":
            mse = self.compute_estimation_error(symmetric=symmetric)
        elif error_type == "discretization":
            mse = self.compute_discretization_error()
        else:
            raise ValueError(f"Unknown error_type: {error_type}")

        return np.sqrt(mse)

    def compute_mae(
        self,
        error_type: Literal["total", "estimation", "discretization"] = "total",
        symmetric: Optional[bool] = None,
    ) -> float:
        """
        Compute mean absolute error.

        Parameters
        ----------
        error_type : {'total', 'estimation', 'discretization'}
            Which error to compute MAE for
        symmetric : bool, optional
            Whether to use symmetrized estimates. If None, uses self.symmetric

        Returns
        -------
        mae : float
            Mean absolute error
        """
        if symmetric is None:
            symmetric = self.symmetric

        cache_key = f"mae_{error_type}_sym{symmetric}"

        if cache_key not in self._metrics_cache:
            # Get estimate with appropriate symmetrization for total and estimation
            if error_type in ["total", "estimation"]:
                summary_cint = self.summariser.summarise_cint(
                    return_symmetrized=symmetric
                )
                m_hat = summary_cint["median"]

            if error_type == "total":
                depix_m_hat = depixilate(m_hat, self.age_bins, self.age_dist)
                diff = depix_m_hat - self.m_true
            elif error_type == "estimation":
                diff = m_hat - self.pix_m_true
            elif error_type == "discretization":
                diff = self.depix_m_true - self.m_true
            else:
                raise ValueError(f"Unknown error_type: {error_type}")

            mae = np.mean(np.abs(diff))
            self._metrics_cache[cache_key] = float(mae)

        return self._metrics_cache[cache_key]

    def compute_relative_error(self, symmetric: Optional[bool] = None) -> float:
        """
        Compute relative error (normalized by true matrix Frobenius norm).

        Parameters
        ----------
        symmetric : bool, optional
            Whether to use symmetrized estimates. If None, uses self.symmetric

        Returns
        -------
        rel_error : float
            Relative Frobenius norm error

        Notes
        -----
        relative_error = ||M_hat - M_true||_F / ||M_true||_F

        This metric is scale-invariant and useful for comparing
        performance across different contact matrices with different
        magnitudes.
        """
        if symmetric is None:
            symmetric = self.symmetric

        cache_key = f"relative_sym{symmetric}"

        if cache_key not in self._metrics_cache:
            # Get estimate with appropriate symmetrization
            summary_cint = self.summariser.summarise_cint(return_symmetrized=symmetric)
            m_hat = summary_cint["median"]

            depix_m_hat = depixilate(m_hat, self.age_bins, self.age_dist)

            numerator = np.linalg.norm(depix_m_hat - self.m_true, "fro")
            denominator = np.linalg.norm(self.m_true, "fro")

            if denominator == 0:
                warnings.warn(
                    "True matrix has zero norm. Relative error undefined.", UserWarning
                )
                rel_error = np.inf
            else:
                rel_error = numerator / denominator

            self._metrics_cache[cache_key] = float(rel_error)

        return self._metrics_cache[cache_key]

    def clear_cache(self) -> None:
        """Clear all cached metric computations."""
        self._metrics_cache.clear()

    def get_cache_info(self) -> Dict[str, int]:
        """
        Get information about cached metrics.

        Returns
        -------
        info : dict
            Cache statistics
        """
        return {
            "n_cached": len(self._metrics_cache),
            "cached_metrics": list(self._metrics_cache.keys()),
        }
