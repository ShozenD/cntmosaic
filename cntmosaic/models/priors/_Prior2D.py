import warnings
from abc import ABC, abstractmethod
from typing import Optional, Union

import jax.numpy as jnp
import numpy as np
from jax.typing import ArrayLike

from .._math import (
    alr,
    basis_contrast_matrix,
    clr,
    ilr,
    inverse_alr,
    inverse_clr,
    inverse_ilr,
    subdiag_permutation_matrix,
)


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
            - Scalar: Broadcasted to (event_dim_latent, A, A)
            - (event_dim_latent, A, A): Full specification
            - (event_dim_latent, A): Broadcasted along last dimension
            where A is the number of age groups.

        Raises
        ------
        ValueError
            If A or event_dim_latent are not initialized (call set_age_bounds first).
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
        if self.event_dim_latent is None:
            raise ValueError(
                "Event dimension must be set before setting location parameter. "
                "Call set_event_dim(event_dim) first."
            )

        # Convert to JAX array
        loc = jnp.asarray(loc)

        # Handle scalar case
        if loc.ndim == 0:
            self.trans_loc = jnp.full((self.event_dim_latent, self.A, self.A), loc)
            return

        # Define valid shapes and their transformation functions
        valid_shapes = {
            (self.event_dim, self.A, self.A): lambda x: x,
            (self.event_dim, self.A): lambda x: jnp.repeat(
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
                f"  - Scalar (broadcasted to ({self.event_dim_latent}, {self.A}, {self.A}))\n"
                f"  - ({self.event_dim_latent}, {self.A}, {self.A})\n"
                f"  - ({self.event_dim_latent}, {self.A}) (broadcasted along last dim)\n"
                f"Hint: If shape mismatch is unexpected, check that set_age_bounds "
                f"and set_event_dim were called with correct values."
            )

        # Apply compositional transformation if specified
        if self.transform:
            transform_func = {"alr": alr, "clr": clr, "ilr": ilr}.get(self.transform)
            if transform_func:
                # Apply transformation along the appropriate axis (last axis for A x A matrices)
                # Note: This assumes data represents compositional data in simplex
                self.trans_loc = transform_func(data, axis=0)
            else:
                # This should never happen due to validate_params, but keep as safeguard
                raise ValueError(f"Unknown transform: {self.transform}")
        else:
            # No transform, use data directly
            self.trans_loc = data

    def set_event_dim(self, K: int) -> None:
        """
        Set the effective event dimension based on transformation and prior type.

        This method computes the effective dimension of the event space, accounting for:
        - Compositional constraints (ALR/ILR reduce dimension by 1)
        - Prior structure (global, partial, or full)
        - Diagonal vs off-diagonal elements (for full priors)

        Parameters
        ----------
        K : int
            K is the number of unique categories (e.g., sex, SES, etc.) in the stratification variable.
            The number of matrices that needs to be *computed* (event_dim) varies based on the prior_type:
            - 'global': 1 matrix
            - 'partial': K matrices, one per category
            - 'full': K * K matrices, one for each pair of categories

            The number of matrices that needs to be *sampled* (event_dim_eff) is different from the
            number of matrices that needs to be *computed* (event_dim) because of reciprocity constraints
            and compositional transformations.

            If there were K categories:
            - For `prior_type='partial'`, event_dim = K
            - For `prior_type='full'`, event_dim = K * K

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
          - Diagonal: sqrt(K * K) - 1 elements
          - Off-diagonal: (K * K - sqrt(K * K)) / 2 elements (due to reciprocity)

        For full priors, also sets:
        - event_dim_diag: Number of diagonal elements
        - event_dim_non_diag: Number of off-diagonal elements

        Examples
        --------
        >>> # Partial prior with ALR transform
        >>> prior = MyPrior(prior_type='partial', transform='alr')
        >>> prior.set_event_dim(5)
        >>> prior.event_dim  # 5
        >>> prior.event_dim_eff  # 4 (reduced by 1 due to ALR)

        >>> # Full prior with CLR transform
        >>> prior = MyPrior(prior_type='full', transform='clr')
        >>> prior.set_event_dim(5)
        >>> prior.event_dim  # 25
        >>> prior.event_dim_eff  # 25 (no reduction for CLR)
        >>> prior.event_dim_diag  # 5 diagonal elements
        >>> prior.event_dim_non_diag  # 20 off-diagonal elements

        >>> # Full prior with ILR transform
        >>> prior = MyPrior(prior_type='full', transform='ilr')
        >>> prior.set_event_dim(5)
        >>> prior.event_dim  # 25
        >>> prior.event_dim_diag  # 5 diagonal elements
        >>> prior.event_dim_non_diag  # 20 off-diagonal elements
        >>> prior.event_dim_eff  # 24 (reduced by 1 due to ILR)
        >>> prior.event_dim_diag_eff  # 4 (diagonal reduced by 1 due to ILR)
        >>> prior.event_dim_non_diag_eff  # 10 (off-diagonal halved due to reciprocity)
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

            if self.transform in ["alr", "ilr"]:
                self.event_dim_latent = (
                    self.event_dim - 1
                )  # no. matrices in latent space
                self.event_dim_diag_eff = self.event_dim_diag - 1
            else:  # clr or None
                self.event_dim_latent = self.event_dim
                self.event_dim_diag_eff = self.event_dim_diag

            self.event_dim_non_diag_eff = self.event_dim_non_diag // 2
            self.event_dim_eff = self.event_dim_diag_eff + self.event_dim_non_diag_eff

        elif self.prior_type == "partial":
            self.event_dim = K

            # Partial prior: one parameter per row/column
            if self.transform in ["alr", "ilr"]:
                self.event_dim_latent = self.event_dim - 1
            else:
                self.event_dim_latent = self.event_dim

            self.event_dim_eff = self.event_dim_latent

        else:  # global
            # Global prior: single shared parameter
            self.event_dim = 1
            self.event_dim_latent = 1
            self.event_dim_eff = 1

    def apply_inverse_transform(self, f):
        """
        Apply inverse compositional transformation to latent field.

        Transforms the unconstrained latent field f back to the simplex (probability
        space) using the specified inverse log-ratio transformation. This is the final
        step in converting sampled IGMRF values to contact probabilities.

        Parameters
        ----------
        f : array, shape (event_dim_latent, A, A)
            Unconstrained latent matrix of size A x A sampled from the prior

        Returns
        -------
        transformed : array, shape (event_dim, A, A)
            Transformed field on the simplex.
            Sums to 1 along the first axis (event_dim) for each (A, A) matrix.

        Notes
        -----
        The transformation depends on the transform attribute:
        - None: Returns f unchanged (no transformation)
        - 'alr': Additive log-ratio inverse (adds reference category)
        - 'clr': Centered log-ratio inverse (removes centering constraint)
        - 'ilr': Isometric log-ratio inverse (orthonormal basis)

        All transformations map from $R^{D-1}$ to the simplex $S^{D}$ where d is event_dim.

        Examples
        --------
        >>> import jax.numpy as jnp
        >>> from cntmosaic.models.priors._IGMRF2D import IGMRF2D
        >>>
        >>> prior = IGMRF2D(
        ...     num_nodes=(5, 5),
        ...     order=(1, 1),
        ...     transform='clr',
        ...     prior_type='partial'
        ... )
        >>> prior.set_age_bounds(0, 25)
        >>> prior.set_event_dim(3)
        >>>
        >>> # Simulate latent field
        >>> f = jnp.zeros((2, 5, 5))  # event_dim_latent = 2, A = 5 for CLR
        >>> transformed = prior.apply_inverse_transform(f)
        >>> print(transformed.shape)  # (3, 5, 5)
        >>> print(jnp.allclose(transformed.sum(axis=0), 1.0))  # True (on simplex)

        See Also
        --------
        cntmosaic.models._math.inverse_alr : ALR inverse transformation
        cntmosaic.models._math.inverse_clr : CLR inverse transformation
        cntmosaic.models._math.inverse_ilr : ILR inverse transformation
        """
        # Optional transformations
        if self.transform == "alr":
            return inverse_alr(f, axis=0)
        elif self.transform == "clr":
            return inverse_clr(f, axis=0)
        elif self.transform == "ilr":
            return inverse_ilr(f, axis=0)
        else:
            return f

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
        f_diag : array, shape (event_dim_diag_eff, A, A)
            Sampled diagonal blocks (within-group contacts). These are already
            spatially symmetric: f_diag[k, i, j] = f_diag[k, j, i].
        f_non_diag : array, shape (event_dim_non_diag_eff, A, A)
            Sampled off-diagonal blocks (between-group contacts). Each block will
            be placed at two positions (i,j) and (j,i) as transposes.

        Returns
        -------
        f : array, shape (event_dim_latent, A, A)
            Assembled latent contact matrix with diagonal blocks at their designated
            positions and off-diagonal blocks satisfying reciprocity constraints.

        Notes
        -----
        For a K×K event grid (K categories):
        - Diagonal indices: [0, K+1, 2(K+1), ..., (K-1)(K+1)]
        - For ALR/ILR: Only K-1 diagonal blocks are placed (last omitted)
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
        # Compute diagonal indices in the flattened K×K grid
        sqrt_event_dim = jnp.sqrt(self.event_dim).astype(int)
        diag_idx = jnp.array([(i * sqrt_event_dim + i) for i in range(sqrt_event_dim)])

        # Initialize latent array
        f = jnp.zeros((self.event_dim_latent, self.A, self.A))

        # Allocate diagonal elements (within-group contacts)
        if self.transform in ["alr", "ilr"]:
            # Last element left out for ALR and ILR (dimension reduction)
            f = f.at[diag_idx[:-1], :, :].set(f_diag)
        else:
            # Full allocation for CLR and no transform
            f = f.at[diag_idx, :, :].set(f_diag)

        # Allocate off-diagonal elements (between-group contacts with reciprocity)
        # Get all non-diagonal flat indices in the K×K grid
        all_indices = jnp.arange(self.event_dim)
        non_diag_mask = jnp.isin(all_indices, diag_idx, invert=True)
        non_diag_all = all_indices[non_diag_mask]

        # Compute transpose indices for all non-diagonal positions
        # If flat index is row*K + col, its transpose is col*K + row
        rows = non_diag_all // sqrt_event_dim
        cols = non_diag_all % sqrt_event_dim
        transpose_indices = cols * sqrt_event_dim + rows

        # Keep only indices where idx < transpose_idx (lower half of reciprocal pairs)
        # This ensures we process each pair only once
        lower_mask = non_diag_all < transpose_indices
        flat_idx = non_diag_all[lower_mask]
        transpose_flat_idx = transpose_indices[lower_mask]

        # Allocate each sampled off-diagonal matrix to its position and transpose position
        for i in range(self.event_dim_non_diag_eff):
            idx = flat_idx[i]
            idx_t = transpose_flat_idx[i]
            f = f.at[idx, :, :].set(f_non_diag[i, :, :])
            if self.transform != "ilr":
                # For ALR and CLR, assign transpose directly
                # If ILR, reciprocity handled differently below
                f = f.at[idx_t, :, :].set(f_non_diag[i, :, :].T)

        # For ILR transform, apply rotation matrix Q to enforce reciprocity across age pairs
        if self.transform == "ilr":
            # Compute Q if not already cached
            if not hasattr(self, "Q") or self.Q is None:
                U = basis_contrast_matrix(self.event_dim)
                P = subdiag_permutation_matrix(sqrt_event_dim)
                self.Q = U.T @ P @ U

            # Apply Q to all age pairs: f[:,b,a] = Q @ f[:,a,b]
            # This ensures reciprocity is preserved after inverse ILR transform

            for a in range(self.A):
                for b in range(self.A):
                    f_ab = f[:, a, b]  # Shape: (event_dim_latent,)
                    f_ba = self.Q @ f_ab  # Shape: (event_dim_latent,)
                    f = f.at[:, b, a].set(f_ba)

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
