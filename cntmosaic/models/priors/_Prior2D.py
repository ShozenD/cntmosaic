from abc import ABC, abstractmethod
from typing import Optional, Union
import numpy as np
import warnings
import jax.numpy as jnp
from .._math import alr, clr, ilr


class Prior2D(ABC):
    """
    Abstract base class for 2D prior distributions used in contact matrix estimation.

    This class provides a common interface for various prior distributions over 2D grids,
    supporting different grid types (age-age, age difference), transformations (ALR, CLR, ILR),
    and prior structures (global, partial, full).

    The class handles:
    - Grid configuration for age-structured contact matrices
    - Compositional data transformations (log-ratio transformations)
    - Prior location (mean) parameter management with shape validation
    - Event dimension calculation based on transformation and prior type

    Attributes
    ----------
    ALLOWED_GRID_TYPES : list[str]
        Valid grid types: ['age-age', 'diff-age']
    ALLOWED_TRANSFORMS : list[Optional[str]]
        Valid transformations: [None, 'alr', 'clr', 'ilr']
    ALLOWED_TYPES : list[str]
        Valid prior types: ['global', 'partial', 'full']

    Parameters
    ----------
    grid_type : str, default='age-age'
        Type of grid structure:
        - 'age-age': Full age-by-age contact matrix
        - 'diff-age': Age difference representation
    transform : str or None, default=None
        Compositional data transformation to apply:
        - None: No transformation (simplex constraint)
        - 'alr': Additive log-ratio transformation
        - 'clr': Centered log-ratio transformation
        - 'ilr': Isometric log-ratio transformation
    prior_type : str, default='global'
        Structure of the prior:
        - 'global': Single shared prior across the entire matrix
        - 'partial': Separate priors for each row/column
        - 'full': Separate priors for diagonal and off-diagonal elements

    Notes
    -----
    This is an abstract base class. Concrete implementations must define:
    - `set_age_bounds(min_age, max_age)`: Configure age range
    - `_set_grid()`: Initialize grid structure
    - `sample()`: Sample from the prior distribution

    The class automatically computes effective dimensions based on the chosen
    transformation and prior type, accounting for the compositional nature of
    contact data.

    Examples
    --------
    >>> # Concrete subclass implementation
    >>> class MyPrior(Prior2D):
    ...     def set_age_bounds(self, min_age, max_age):
    ...         self.min_age = min_age
    ...         self.max_age = max_age
    ...         self.A = max_age - min_age + 1
    ...         self._set_grid()
    ...
    ...     def _set_grid(self):
    ...         # Initialize grid-specific structures
    ...         pass
    ...
    ...     def sample(self):
    ...         # Implement sampling logic
    ...         pass

    See Also
    --------
    IGMRF2D : Intrinsic Gaussian Markov Random Field prior
    HSGP2D : Hilbert Space Gaussian Process prior
    GMRF2D : Gaussian Markov Random Field prior
    """

    ALLOWED_GRID_TYPES = ["age-age", "diff-age"]
    ALLOWED_TRANSFORMS = [None, "alr", "clr", "ilr"]
    ALLOWED_TYPES = ["global", "partial", "full"]

    def __init__(
        self,
        grid_type: str = "age-age",
        transform: Optional[str] = None,
        prior_type: str = "global",
    ):
        """
        Initialize the Prior2D base class.

        Parameters
        ----------
        grid_type : str, default='age-age'
            Type of grid structure.
        transform : str or None, default=None
            Compositional data transformation.
        prior_type : str, default='global'
            Structure of the prior.

        Raises
        ------
        ValueError
            If any parameter is not in the allowed set.
        """
        self.validate_params(grid_type, transform, prior_type)
        self.grid_type = grid_type
        self.transform = transform
        self.type = prior_type  # Use 'type' for consistency with subclasses
        self.prior_type = prior_type  # Keep for backward compatibility

        # Initialize attributes that may be accessed before set_age_bounds
        self.A = None
        self.event_dim = None
        self.event_dim_eff = None
        self.trans_loc = None

    def validate_params(
        self, grid_type: str, transform: Optional[str], prior_type: str
    ) -> None:
        """
        Validate initialization parameters.

        Parameters
        ----------
        grid_type : str
            Grid type to validate.
        transform : str or None
            Transform type to validate.
        prior_type : str
            Prior type to validate.

        Raises
        ------
        ValueError
            If any parameter is invalid.
        """
        if grid_type not in self.ALLOWED_GRID_TYPES:
            raise ValueError(
                f"grid_type must be one of {self.ALLOWED_GRID_TYPES}, got '{grid_type}'"
            )

        if transform not in self.ALLOWED_TRANSFORMS:
            raise ValueError(
                f"transform must be one of {self.ALLOWED_TRANSFORMS}, got '{transform}'"
            )

        if prior_type not in self.ALLOWED_TYPES:
            raise ValueError(
                f"prior_type must be one of {self.ALLOWED_TYPES}, got '{prior_type}'"
            )

    def set_loc(self, loc: Union[int, float, np.ndarray, jnp.ndarray]) -> None:
        """
        Set the location parameter (prior mean) with shape validation and transformation.

        This method handles different input shapes and applies the appropriate
        transformation (ALR, CLR, or ILR) if specified. The location parameter
        represents the prior mean in the transformed space.

        Parameters
        ----------
        loc : int, float, or array-like
            Location parameter with flexible shape:
            - Scalar: Broadcasted to (event_dim_eff, A, A)
            - (event_dim_eff, A, A): Full specification
            - (event_dim_eff, A): Broadcasted along last dimension
            where A is the number of age groups.

        Raises
        ------
        ValueError
            If A or event_dim_eff are not initialized (call set_age_bounds first).
            If loc shape doesn't match expected dimensions.

        Notes
        -----
        For compositional data with transformations:
        - The input loc is assumed to be in the original (simplex) space
        - It is automatically transformed to the appropriate space (ALR, CLR, or ILR)
        - For 'alr' and 'ilr', the effective dimension is reduced by 1

        Examples
        --------
        >>> prior = MyPrior(transform='alr')
        >>> prior.set_age_bounds(0, 10)  # 11 age groups
        >>> prior.set_event_dim(11)
        >>>
        >>> # Scalar location
        >>> prior.set_loc(0.0)
        >>>
        >>> # Full specification
        >>> loc_full = np.random.randn(10, 11, 11)  # event_dim_eff=10 due to ALR
        >>> prior.set_loc(loc_full)
        """
        # Validate prerequisites
        if self.A is None:
            raise ValueError(
                "Age bounds must be set before setting location parameter. "
                "Call set_age_bounds(min_age, max_age) first."
            )
        if self.event_dim_eff is None:
            raise ValueError(
                "Event dimension must be set before setting location parameter. "
                "Call set_event_dim(event_dim) first."
            )

        # Convert to JAX array
        loc = jnp.asarray(loc)

        # Handle scalar case
        if loc.ndim == 0:
            self.trans_loc = jnp.full((self.event_dim_eff, self.A, self.A), loc)
            return

        # Define valid shapes and their transformation functions
        valid_shapes = {
            (self.event_dim_eff, self.A, self.A): lambda x: x,
            (self.event_dim_eff, self.A): lambda x: jnp.repeat(
                x[:, :, None], self.A, axis=2
            ),
        }

        # Find matching shape and apply transformation
        data = None
        for shape, transform_fn in valid_shapes.items():
            if loc.shape == shape:
                data = transform_fn(loc)
                break

        if data is None:
            raise ValueError(
                f"Invalid loc shape {loc.shape}. Expected one of:\n"
                f"  - Scalar (broadcasted to ({self.event_dim_eff}, {self.A}, {self.A}))\n"
                f"  - ({self.event_dim_eff}, {self.A}, {self.A})\n"
                f"  - ({self.event_dim_eff}, {self.A}) (broadcasted along last dim)\n"
                f"Hint: If shape mismatch is unexpected, check that set_age_bounds "
                f"and set_event_dim were called with correct values."
            )

        # Apply compositional transformation if specified
        if self.transform:
            transform_func = {"alr": alr, "clr": clr, "ilr": ilr}.get(self.transform)
            if transform_func:
                # Apply transformation along the appropriate axis (last axis for A x A matrices)
                # Note: This assumes data represents compositional data in simplex
                self.trans_loc = transform_func(data, axis=-1)
            else:
                # This should never happen due to validate_params, but keep as safeguard
                raise ValueError(f"Unknown transform: {self.transform}")
        else:
            # No transform, use data directly
            self.trans_loc = data

    def set_event_dim(self, event_dim: int) -> None:
        """
        Set the effective event dimension based on transformation and prior type.

        This method computes the effective dimension of the event space, accounting for:
        - Compositional constraints (ALR/ILR reduce dimension by 1)
        - Prior structure (global, partial, or full)
        - Diagonal vs off-diagonal elements (for full priors)

        Parameters
        ----------
        event_dim : int
            The base event dimension (number of parameters before transformation).
            For a full age-by-age matrix, this would be A² where A is the number
            of age groups.

        Raises
        ------
        AssertionError
            If event_dim is not a positive integer.

        Notes
        -----
        Effective dimensions by prior type:

        - **global**: event_dim_eff = 1 (single shared parameter)
        - **partial**: event_dim_eff = event_dim or event_dim - 1 (with ALR/ILR)
        - **full**: Separate diagonal and off-diagonal dimensions
          - Diagonal: sqrt(event_dim) elements
          - Off-diagonal: Remaining elements
          - Both adjusted for transformations

        For full priors, also sets:
        - event_dim_diag: Number of diagonal elements
        - event_dim_non_diag: Number of off-diagonal elements

        Examples
        --------
        >>> # Partial prior with ALR transform on 5x5 matrix
        >>> prior = MyPrior(prior_type='partial', transform='alr')
        >>> prior.set_event_dim(25)  # 5² = 25
        >>> prior.event_dim_eff  # 24 (reduced by 1 due to ALR)
        24

        >>> # Full prior with CLR transform
        >>> prior = MyPrior(prior_type='full', transform='clr')
        >>> prior.set_event_dim(25)
        >>> prior.event_dim_diag  # 5 diagonal elements
        5
        >>> prior.event_dim_non_diag  # 20 off-diagonal elements
        20
        """
        assert (
            isinstance(event_dim, int) and event_dim > 0
        ), f"event_dim must be a positive integer, got {event_dim}"
        self.event_dim = event_dim

        if self.prior_type == "full":
            # For full priors, separate diagonal and off-diagonal
            self.event_dim_diag = int(np.sqrt(self.event_dim))

            # Validate that event_dim is a perfect square
            if self.event_dim_diag**2 != self.event_dim:
                raise ValueError(
                    f"For prior_type='full', event_dim must be a perfect square. "
                    f"Got event_dim={self.event_dim}, sqrt={self.event_dim_diag}"
                )

            if self.transform in ["alr", "ilr"]:
                self.event_dim_eff = self.event_dim - 1
                self.event_dim_diag -= 1
                self.event_dim_non_diag = self.event_dim_eff - self.event_dim_diag
            else:  # clr or None
                self.event_dim_eff = self.event_dim
                self.event_dim_non_diag = self.event_dim - self.event_dim_diag

        elif self.prior_type == "partial":
            # Partial prior: one parameter per row/column
            if self.transform in ["alr", "ilr"]:
                self.event_dim_eff = self.event_dim - 1
            else:
                self.event_dim_eff = self.event_dim

        else:  # global
            # Global prior: single shared parameter
            self.event_dim_eff = 1

    @abstractmethod
    def set_age_bounds(self, min_age: int, max_age: int) -> None:
        """
        Set the age bounds for the contact matrix.

        This method must be implemented by subclasses to configure the age range
        and initialize any age-dependent structures.

        Parameters
        ----------
        min_age : int
            Minimum age in the contact matrix.
        max_age : int
            Maximum age in the contact matrix (inclusive).

        Notes
        -----
        Implementations typically:
        1. Store min_age and max_age
        2. Compute A = max_age - min_age + 1 (number of age groups)
        3. Call _set_grid() to initialize grid structures
        """
        pass

    @abstractmethod
    def _set_grid(self) -> None:
        """
        Initialize grid-specific structures.

        This method must be implemented by subclasses to set up any
        grid-dependent data structures, such as:
        - Precision matrices for GMRFs
        - Basis functions for splines
        - Kernel matrices for GPs

        This is typically called at the end of set_age_bounds().
        """
        pass

    @abstractmethod
    def sample(self):
        """
        Sample from the prior distribution.

        This method must be implemented by subclasses to generate samples
        from the specific prior distribution. The implementation should:

        1. Use NumPyro's sampling primitives (numpyro.sample)
        2. Return samples in the appropriate shape for the prior type
        3. Apply inverse transformations if necessary to return to simplex

        Returns
        -------
        array-like
            Samples from the prior distribution. Shape depends on prior_type:
            - global: (A, A) symmetric matrix
            - partial: (event_dim_eff, A, A)
            - full: Complex structure with diagonal and off-diagonal components

        Notes
        -----
        Subclasses should handle:
        - Proper NumPyro plate contexts for batching
        - Transformation from latent space to observation space
        - Symmetry constraints (if applicable)
        """
        pass
