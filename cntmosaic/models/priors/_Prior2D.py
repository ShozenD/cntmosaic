from abc import ABC, abstractmethod
from typing import Union

import jax.numpy as jnp
import numpy as np
from jax.typing import ArrayLike


class Prior2D(ABC):
    """
    Abstract base class for 2D prior distributions used in contact matrix estimation.

    This class provides a common interface for various prior distributions over 2D grids,
    supporting different grid types (age-age, age difference) and prior structures (global, partial, full).

    The class handles:
    - Grid configuration for age-structured contact matrices
    - Prior location (mean) parameter management with shape validation
    - Event dimension calculation based on transformation and prior type

    Attributes
    ----------
    ALLOWED_GRID_TYPES : list[str]
        Valid grid types: ['age-age', 'diff-age']
    ALLOWED_TYPES : list[str]
        Valid prior types: ['global', 'partial', 'full']

    Parameters
    ----------
    grid_type : str, default='age-age'
        Type of grid structure:
        - 'age-age': Full age-by-age contact matrix
        - 'diff-age': Age difference representation
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

    The class automatically computes effective dimensions based on the chosen prior type.

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
    ALLOWED_TYPES = ["global", "partial", "full"]

    def __init__(
        self,
        grid_type: str = "age-age",
        prior_type: str = "global",
    ):
        """
        Initialize the Prior2D base class.

        Parameters
        ----------
        grid_type : str, default='age-age'
            Type of grid structure.
        prior_type : str, default='global'
            Structure of the prior.

        Raises
        ------
        ValueError
            If any parameter is not in the allowed set.
        """
        self.validate_params(grid_type, prior_type)
        self.grid_type = grid_type
        self.type = prior_type  # Use 'type' for consistency with subclasses
        self.prior_type = prior_type  # Keep for backward compatibility

        # Initialize attributes that may be accessed before set_age_bounds
        self.A = None
        self.event_dim = None
        self.loc = None

    def validate_params(self, grid_type: str, prior_type: str) -> None:
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

        if prior_type not in self.ALLOWED_TYPES:
            raise ValueError(
                f"prior_type must be one of {self.ALLOWED_TYPES}, got '{prior_type}'"
            )

    def set_loc(self, loc: Union[int, float, np.ndarray, jnp.ndarray]) -> None:
        """
        Set the location parameter (prior mean) with shape validation.

        This method handles different input shapes for the location parameter.

        Parameters
        ----------
        loc : int, float, or array-like
            Location parameter with flexible shape:
            - Scalar: Broadcasted to (event_dim, A, A)
            - (event_dim, A, A): Full specification
            - (event_dim, A): Broadcasted along last dimension
            where A is the number of age groups.

        Raises
        ------
        ValueError
            If A or event_dim are not initialized (call set_age_bounds first).
            If loc shape doesn't match expected dimensions.

        Examples
        --------
        >>> prior = MyPrior()
        >>> prior.set_age_bounds(0, 10)  # 11 age groups
        >>> prior.set_event_dim(5)
        >>>
        >>> # Scalar location
        >>> prior.set_loc(0.0)
        >>>
        >>> # Full specification
        >>> loc_full = np.random.randn(5, 11, 11)
        >>> prior.set_loc(loc_full)
        """
        # Validate prerequisites
        if self.A is None:
            raise ValueError(
                "Age bounds must be set before setting location parameter. "
                "Call set_age_bounds(min_age, max_age) first."
            )
        if self.event_dim is None:
            raise ValueError(
                "Event dimension must be set before setting location parameter. "
                "Call set_event_dim(event_dim) first."
            )

        # Convert to JAX array
        loc = jnp.asarray(loc)

        # Handle scalar case
        if loc.ndim == 0:
            self.loc = jnp.full((self.event_dim, self.A, self.A), loc)
            return

        # Define valid shapes and their transformation functions
        valid_shapes = {
            (self.event_dim, self.A, self.A): lambda x: x,
            (self.event_dim, self.A): lambda x: jnp.repeat(
                x[:, :, None], self.A, axis=2
            ),
            (self.event_dim, self.A, 1): lambda x: jnp.repeat(x, self.A, axis=2),
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
                f"  - Scalar (broadcasted to ({self.event_dim}, {self.A}, {self.A}))\n"
                f"  - ({self.event_dim}, {self.A}, {self.A})\n"
                f"  - ({self.event_dim}, {self.A}) (broadcasted along last dim)\n"
                f"  - ({self.event_dim}, {self.A}, 1) (broadcasted along last dim)\n"
                f"Hint: If shape mismatch is unexpected, check that set_age_bounds "
                f"and set_event_dim were called with correct values."
            )

        self.loc = data

    def set_event_dim(self, K: int) -> None:
        """
        Set the event dimension based on prior type.

        This method computes the dimension of the event space based on:
        - Prior structure (global, partial, or full)
        - Diagonal vs off-diagonal elements (for full priors)

        Parameters
        ----------
        K : int
            K is the number of unique categories (e.g., sex, SES, etc.) in the stratification variable.
            The number of matrices that needs to be computed (event_dim) varies based on the prior_type:
            - 'global': 1 matrix
            - 'partial': K matrices, one per category
            - 'full': K * K matrices, one for each pair of categories

        Raises
        ------
        AssertionError
            If event_dim is not a positive integer.

        Notes
        -----
        For full priors, also sets:
        - event_dim_diag: Number of diagonal elements
        - event_dim_non_diag: Number of off-diagonal elements
        - event_dim_non_diag_eff: Half of event_dim_non_diag (due to reciprocity constraint)

        Examples
        --------
        >>> # Partial prior
        >>> prior = MyPrior(prior_type='partial')
        >>> prior.set_event_dim(5)
        >>> prior.event_dim  # 5

        >>> # Full prior
        >>> prior = MyPrior(prior_type='full')
        >>> prior.set_event_dim(5)
        >>> prior.event_dim  # 25
        >>> prior.event_dim_diag  # 5 diagonal elements
        >>> prior.event_dim_non_diag  # 20 off-diagonal elements
        >>> prior.event_dim_non_diag_eff  # 10 (halved due to reciprocity)
        """
        assert isinstance(K, int) and K > 0, f"K must be a positive integer, got {K}"
        if self.prior_type == "global" and K != 1:
            raise ValueError(f"For prior_type='global', K must be 1. Got K={K}.")

        if self.prior_type == "full":
            self.event_dim = int(K * K)

            # For full priors, separate diagonal and off-diagonal
            self.event_dim_diag = int(np.sqrt(self.event_dim))
            self.event_dim_non_diag = self.event_dim - self.event_dim_diag

            # Validate that event_dim is a perfect square
            if self.event_dim_diag**2 != self.event_dim:
                raise ValueError(
                    f"For prior_type='full', event_dim must be a perfect square. "
                    f"Got event_dim={self.event_dim}, sqrt={self.event_dim_diag}"
                )

            self.event_dim_non_diag_eff = self.event_dim_non_diag // 2

            # Pre-compute static indices for assembly (avoid JAX tracing issues)
            self._compute_full_prior_indices()

        elif self.prior_type == "partial":
            self.event_dim = K

        else:  # global
            # Global prior: single shared parameter
            self.event_dim = 1

    def _compute_full_prior_indices(self) -> None:
        """
        Pre-compute static indices for assembling full prior blocks.

        This method computes all the indices needed for `_assemble_full_prior_blocks`
        as static numpy arrays to avoid JAX tracing issues. Called automatically
        by `set_event_dim` when `prior_type='full'`.

        Sets the following attributes:
        - _diag_idx: Diagonal indices in flattened K×K grid
        - _flat_idx: Lower-triangular off-diagonal indices
        - _transpose_flat_idx: Corresponding transpose indices
        """
        sqrt_event_dim = self.event_dim_diag

        # Diagonal indices: [0, K+1, 2(K+1), ..., (K-1)(K+1)]
        self._diag_idx = np.array(
            [(i * sqrt_event_dim + i) for i in range(sqrt_event_dim)]
        )

        # All indices in the K×K grid
        all_indices = np.arange(self.event_dim)

        # Non-diagonal indices (all except diagonal)
        non_diag_mask = np.isin(all_indices, self._diag_idx, invert=True)
        non_diag_all = all_indices[non_diag_mask]

        # Compute transpose indices: if idx = row*K + col, transpose = col*K + row
        rows = non_diag_all // sqrt_event_dim
        cols = non_diag_all % sqrt_event_dim
        transpose_indices = cols * sqrt_event_dim + rows

        # Keep only lower-triangular pairs (idx < transpose_idx)
        lower_mask = non_diag_all < transpose_indices
        self._flat_idx = non_diag_all[lower_mask]
        self._transpose_flat_idx = transpose_indices[lower_mask]

    def _assemble_full_prior_blocks(
        self, f_diag: ArrayLike, f_non_diag: ArrayLike
    ) -> ArrayLike:
        """
        Assemble diagonal and off-diagonal blocks into full event grid.

        This method allocates sampled diagonal and off-diagonal contact matrices
        into the full K×K event dimension grid, enforcing the reciprocity constraint
        that off-diagonal blocks (i,j) and (j,i) are transposes of each other.

        Parameters
        ----------
        f_diag : array, shape (event_dim_diag, A, A)
            Sampled diagonal blocks (within-group contacts). These are already
            spatially symmetric: f_diag[k, i, j] = f_diag[k, j, i].
        f_non_diag : array, shape (event_dim_non_diag_eff, A, A)
            Sampled off-diagonal blocks (between-group contacts). Each block will
            be placed at two positions (i,j) and (j,i) as transposes.
            Note: event_dim_non_diag_eff = event_dim_non_diag // 2 due to reciprocity.

        Returns
        -------
        f : array, shape (event_dim, A, A)
            Assembled contact matrix with diagonal blocks at their designated
            positions and off-diagonal blocks satisfying reciprocity constraints.

        Notes
        -----
        For a K×K event grid (K categories):
        - Diagonal indices: [0, K+1, 2(K+1), ..., (K-1)(K+1)]
        - Off-diagonal reciprocity: f[row*K + col] = f[col*K + row].T

        The method handles the index arithmetic to:
        1. Place diagonal blocks at correct positions in the flattened grid
        2. Identify reciprocal pairs of off-diagonal positions
        3. Assign each sampled off-diagonal matrix to both positions as transposes

        Examples
        --------
        For K=3, event_dim=9, the grid layout is:
        ```
        [0, 1, 2]     [(0,0), (0,1), (0,2)]
        [3, 4, 5]  =  [(1,0), (1,1), (1,2)]
        [6, 7, 8]     [(2,0), (2,1), (2,2)]
        ```
        Diagonal indices: [0, 4, 8]
        Reciprocal pairs: (1,3), (2,6), (5,7)
        """
        # Use pre-computed static indices (set during set_event_dim)
        # Initialize array
        f = jnp.zeros((self.event_dim, self.A, self.A))
        f = f.at[self._diag_idx, :, :].set(f_diag)

        # Allocate each sampled off-diagonal matrix to its position and transpose position
        for i in range(self.event_dim_non_diag_eff):
            idx = self._flat_idx[i]
            idx_t = self._transpose_flat_idx[i]
            f = f.at[idx, :, :].set(f_non_diag[i, :, :])
            f = f.at[idx_t, :, :].set(f_non_diag[i, :, :].T)

        return f

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
            - partial: (event_dim, A, A)
            - full: (event_dim, A, A)
        """
        pass
