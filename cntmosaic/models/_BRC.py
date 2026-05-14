import warnings
from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, Optional

import jax
import jax.numpy as jnp
import numpy as np
from jax.random import PRNGKey
from jax.typing import ArrayLike
from numpy.typing import NDArray

from ..dataloader import DataLoader
from ..dataloader.containers._ModelData import ModelData
from ._base import ContactModel
from .numpyro.priors import Prior2D


class BRC(ContactModel, ABC):
    """
    Base class for the Bayesian Rate Consistency model.

    The BRC model provides a Bayesian framework for estimating social contact matrices
    from contact survey data. It ensures consistency between forward and reciprocal
    contact rates by incorporating population age distribution constraints.

    Parameters
    ----------
    dataloader : DataLoader
        DataLoader object containing the processed participant and contact data.
        Must be properly initialized with participant data, contact data, and
        population age distribution. See `DataLoader` for more details.
    priors : dict
        Dictionary containing prior specifications for model parameters.
        Must contain at least a 'rate' key specifying the prior for contact rates.
        Additional priors depend on the specific model subclass.
    likelihood : str, default='negbin'
        Likelihood function to use for the observation model.
        Options are:
            - 'negbin': Negative binomial distribution (handles overdispersion)
            - 'poisson': Poisson distribution (assumes equidispersion)

    Attributes
    ----------
    ds : xarray.Dataset
        Loaded dataset from the dataloader containing processed contact data.
    priors : dict
        Dictionary of prior specifications.
    likelihood : str
        Selected likelihood function.
    age_min : int
        Minimum age in the population (set via `set_age_dims`).
    age_max : int
        Maximum age in the population (set via `set_age_dims`).
    A : int
        Number of age groups (age_max - age_min + 1).
    y : ArrayLike
        Observed contact counts (set by subclasses).
    age_dist : NDArray, optional
        Population age distribution (set via `set_age_dist`).

    Raises
    ------
    ValueError
        If likelihood is not one of the allowed values.
        If priors is not a dictionary.
        If priors does not contain 'rate' specification.

    References
    ----------
    Shozen Dan et al., "Estimating fine age structure and time trends in
    human contact patterns from coarse contact data: The Bayesian rate consistency model",
    PLoS Computational Biology. 2023

    Examples
    --------
    >>> from cntmosaic.dataloader import DataLoader
    >>> from cntmosaic.models.numpyro.priors import Spline2D
    >>> from cntmosaic.models import BRCfine
    >>>
    >>> # Create dataloader with contact data
    >>> dataloader = DataLoader(df_part, df_cnt, df_age_dist)
    >>>
    >>> # Specify priors
    >>> priors = {"rate": Spline2D(prior_type="global", M=30, degree=3)}
    >>>
    >>> # Initialize model
    >>> model = BRCfine(dataloader, priors, likelihood="negbin")
    >>>
    >>> # Run inference
    >>> from jax.random import PRNGKey
    >>> model.run_inference_mcmc(PRNGKey(0), num_samples=1000)

    See Also
    --------
    BRCfine : Fine-grained age resolution BRC model
    BRCrefine : Coarse-to-fine age refinement BRC model
    DataLoader : Data preprocessing and loading utilities
    """

    ALLOWED_LIKELIHOODS = ["negbin", "poisson"]

    def __init__(
        self,
        dataloader: DataLoader,
        priors: Dict[str, Any],
        likelihood: str = "negbin",
        backend: Optional[Any] = None,
    ) -> None:
        super().__init__(backend=backend)
        self._validate_common_inputs(dataloader, priors, likelihood)

        self.data: ModelData = dataloader.load()
        self.priors: Dict[str, Prior2D] = priors
        self.likelihood: str = likelihood

        # Computed attributes
        self.age_dist: Optional[NDArray] = None
        self.age_min: Optional[int] = None
        self.age_max: Optional[int] = None
        self.A: Optional[int] = None

        self.set_age_dims(self.data.age_min, self.data.age_max)

    def _validate_common_inputs(
        self, dataloader: DataLoader, priors: Dict[str, Any], likelihood: str
    ) -> None:
        """
        Validate inputs common to all BRC model variants.

        This method performs essential validation checks on the core model inputs:
        1. Ensures the likelihood function is supported
        2. Verifies priors is a dictionary
        3. Confirms that rate prior specification is present

        Parameters
        ----------
        dataloader : DataLoader
            DataLoader object to validate.
        priors : dict
            Dictionary of prior specifications to validate.
        likelihood : str
            Likelihood function name to validate.

        Raises
        ------
        ValueError
            If likelihood is not in ALLOWED_LIKELIHOODS.
            If priors is not a dictionary.
            If 'rate' key is missing from priors.

        Notes
        -----
        Subclasses may implement additional validation via their own
        `_validate_inputs()` method for model-specific requirements.
        """
        if likelihood not in self.ALLOWED_LIKELIHOODS:
            raise ValueError(
                f"likelihood must be one of: {self.ALLOWED_LIKELIHOODS}, "
                f"got '{likelihood}'"
            )

        if not isinstance(priors, dict):
            raise ValueError("priors must be a dictionary")

        if "rate" not in priors.keys():
            raise ValueError("priors must contain the specifications for 'rate'")

    def _make_guide(self, guide: Optional[Callable]) -> Callable:
        """BRC-family models require an explicit guide."""
        if guide is None:
            raise ValueError(
                "BRC-family models require an explicit guide. "
                "Pass a guide to run_inference_svi(), e.g. AutoNormal(model.model)."
            )
        return guide

    def set_age_dims(self, age_min: int, age_max: int) -> None:
        """
        Set the age dimensions of the model and configure priors accordingly.

        This method establishes the age range for the contact matrix and communicates
        these bounds to all prior objects. It computes the number of age groups (A)
        and ensures all priors are properly initialized for the specified age range.

        Parameters
        ----------
        age_min : int
            Minimum age in years (inclusive). Must be non-negative.
        age_max : int
            Maximum age in years (inclusive). Must be greater than age_min.

        Raises
        ------
        ValueError
            If age_min is negative.
            If age_max <= age_min.

        Notes
        -----
        The number of age groups is computed as A = age_max - age_min + 1.
        For example, if age_min=0 and age_max=80, then A=81 age groups.

        This method should be called automatically during initialization, but can
        be manually invoked to reconfigure the model for a different age range.

        Examples
        --------
        >>> model = BRCfine(dataloader, priors)
        >>> model.set_age_dims(0, 80)  # 81 age groups (0-80)
        >>> print(model.A)
        81
        """
        if age_min < 0:
            raise ValueError(f"age_min must be non-negative, got {age_min}")
        if age_max <= age_min:
            raise ValueError(
                f"age_max must be greater than age_min, got age_max={age_max}, "
                f"age_min={age_min}"
            )

        self.age_min = age_min
        self.age_max = age_max
        self.A = age_max - age_min + 1

        for prior in self.priors.values():
            prior.set_age_bounds(age_min, age_max)

    def set_age_dist(self, age_dist: NDArray) -> None:
        """
        Set the population age distribution for the model.

        The population age distribution is used to ensure rate consistency in the
        contact matrix estimation. It represents the proportion of the population
        in each age group and is typically obtained from census or demographic data.

        Parameters
        ----------
        age_dist : NDArray
            Population age distribution as a 1D array of length A, where A is the
            number of age groups. Should contain non-negative values that sum to 1.
            Element i represents the proportion of the population in age group i.

        Raises
        ------
        ValueError
            If age_dist length doesn't match the number of age groups (A).
            If age_dist contains negative values.
            If age_dist doesn't sum to approximately 1.0.

        Notes
        -----
        The age distribution is used in the model to weight contact rates and ensure
        demographic consistency in the estimated contact matrix. It's essential that
        the age_dist aligns with the age grouping defined by age_min and age_max.

        Examples
        --------
        >>> import numpy as np
        >>> # Uniform age distribution for 81 age groups
        >>> age_dist = np.ones(81) / 81
        >>> model.set_age_dist(age_dist)
        >>>
        >>> # Or load from demographic data
        >>> from cntmosaic.datasets import load_age_distribution
        >>> df_age = load_age_distribution("United_States")
        >>> model.set_age_dist(df_age.P.values)
        """
        if not hasattr(self, "A"):
            raise AttributeError("Age dimensions not set. Call set_age_dims() first.")

        age_dist = np.asarray(age_dist)

        if age_dist.ndim != 1:
            raise ValueError(
                f"age_dist must be 1-dimensional, got shape {age_dist.shape}"
            )

        if len(age_dist) != self.A:
            raise ValueError(
                f"age_dist length ({len(age_dist)}) must match number of age "
                f"groups (A={self.A})"
            )

        if np.any(age_dist < 0):
            raise ValueError("age_dist must contain only non-negative values")

        age_dist_sum = np.sum(age_dist)
        if not np.isclose(age_dist_sum, 1.0, rtol=1e-3):
            warnings.warn(
                f"age_dist sums to {age_dist_sum:.6f}, not 1.0. "
                "Consider normalizing the distribution.",
                stacklevel=2,
            )

        self.age_dist = age_dist

    @abstractmethod
    def model(self, y: Optional[ArrayLike] = None) -> None:
        """
        Define the probabilistic model for contact matrix estimation.

        This abstract method must be implemented by all subclasses to specify the
        generative model structure, including priors, likelihood, and any deterministic
        transformations. The model should be compatible with NumPyro's primitives.

        Parameters
        ----------
        y : ArrayLike, optional
            Observed contact counts.  When ``None`` the model samples from
            the prior (prior predictive mode).

        Raises
        ------
        NotImplementedError
            If a subclass doesn't implement this method.

        Notes
        -----
        Subclass implementations should:
        1. Sample from priors for all latent variables
        2. Apply necessary transformations to compute contact intensities
        3. Define the observation likelihood given the model parameters
        4. Use numpyro.sample() and numpyro.deterministic() appropriately

        Examples
        --------
        >>> def model(self, y=None):
        ...     beta0 = numpyro.sample("baseline", dist.Normal(0.0, 2.5))
        ...     with scope(prefix="rate"):
        ...         f = self.priors["rate"].sample()
        ...     # ... rest of model specification
        """
        raise NotImplementedError

    def prior_sampler(
        self, para_name: str, num_samples: int = 1, seed: int = 0
    ) -> jax.Array:
        """
        Sample from the prior distribution of a specified parameter.

        This method is useful for:
        - Prior predictive checks: Assessing if priors are reasonable before seeing data
        - Visualization: Understanding the prior distribution's shape and range
        - Sensitivity analysis: Testing how prior choices affect inference
        - Debugging: Verifying prior specifications are correct

        Parameters
        ----------
        para_name : str
            Name of the parameter to sample from. Must be a key in `self.priors`.
            Common parameter names include 'rate', 'baseline', 'inv_disp', etc.
        num_samples : int, default=1
            Number of samples to draw from the prior distribution.
        seed : int, default=0
            Seed for random number generation to ensure reproducibility.

        Returns
        -------
        jax.Array
            Array of shape `(num_samples, ...)` containing the sampled values.
            The trailing dimensions depend on the parameter's shape in the model.

        Raises
        ------
        KeyError
            If `para_name` is not found in `self.priors`.
        ValueError
            If `num_samples` is not positive.

        Notes
        -----
        - This samples directly from the prior, ignoring any observed data
        - For compositional priors (e.g., with ALR/CLR/ILR transforms),
          samples are returned in the transformed space
        - The prior must have a `sample()` method compatible with JAX

        Examples
        --------
        >>> from cntmosaic.models.numpyro.priors import Spline2D
        >>> priors = {"rate": Spline2D(prior_type="global", M=30, degree=3)}
        >>> model = BRCfine(dataloader, priors)
        >>>
        >>> # Sample from the rate prior
        >>> rate_samples = model.prior_sampler("rate", num_samples=100, seed=42)
        >>> print(rate_samples.shape)  # (100, A, A) where A is number of age groups
        >>>
        >>> # Visualize prior samples
        >>> import matplotlib.pyplot as plt
        >>> plt.imshow(rate_samples[0])  # Show first sample
        >>> plt.colorbar()
        >>> plt.title("Prior sample for contact rate matrix")

        See Also
        --------
        print_model_shape : Display model structure and parameter shapes
        """
        if num_samples <= 0:
            raise ValueError(f"num_samples must be positive, got {num_samples}")

        if para_name not in self.priors:
            raise KeyError(
                f"Parameter '{para_name}' not found in priors. "
                f"Available parameters: {list(self.priors.keys())}"
            )

        prng_key = jax.random.PRNGKey(seed)
        _, subkey = jax.random.split(prng_key)

        samples = self.priors[para_name].sample(subkey, sample_shape=(num_samples,))

        return samples
