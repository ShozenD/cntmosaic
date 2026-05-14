from typing import Union

import numpy as np
import numpyro
from numpy.typing import NDArray
from numpyro import distributions as dist
from scipy.interpolate import BSpline

from ..._utils import (
    age_age_grid,
    diff_age_age_grid,
    diff_age_age_index,
    symm_from_tril_ix_row,
    tril_ix_row,
)
from ._Prior2D import Prior2D


def validate_init_params(M: int | list[int], degree: int | list[int]):
    """
    Validate initialization parameters for the Spline2D class.

    Parameters
    ----------
    M : int or list of int
        Number of basis functions (interior knots)
    degree : int or list of int
        Degree of B-spline polynomials

    Raises
    ------
    ValueError
        If M or degree are invalid (non-positive or wrong list length)
    """
    if isinstance(M, int):
        if M <= 0:
            raise ValueError(f"M must be greater than 0, got {M}")
    elif isinstance(M, list):
        if len(M) == 0 or len(M) > 2:
            raise ValueError(
                f"M must be scalar or a list of length 1-2, got length {len(M)}"
            )
        if any(m <= 0 for m in M):
            raise ValueError(f"All elements of M must be greater than 0, got {M}")

    if isinstance(degree, int):
        if degree <= 0:
            raise ValueError(f"degree must be greater than 0, got {degree}")
    elif isinstance(degree, list):
        if len(degree) == 0 or len(degree) > 2:
            raise ValueError(
                f"degree must be scalar or a list of length 1-2, got length {len(degree)}"
            )
        if any(d <= 0 for d in degree):
            raise ValueError(
                f"All elements of degree must be greater than 0, got {degree}"
            )


class Spline2D(Prior2D):
    """
    2D tensor product B-spline prior for contact matrix estimation.

    This prior class implements flexible spatial smoothing for contact matrices using
    tensor product B-splines. It represents the latent field as a linear combination
    of B-spline basis functions.

    Mathematical Background
    -----------------------
    The latent field f(x, y) is represented as:

        f(x, y) = ∑ᵢ ∑ⱼ βᵢⱼ Bᵢ(x) Bⱼ(y)

    where Bᵢ(x) and Bⱼ(y) are B-spline basis functions of specified degree, and
    βᵢⱼ are coefficients sampled from a standard normal prior: βᵢⱼ ~ N(0, 1).

    The tensor product basis Φ is constructed as the Kronecker product:

        Φ = Φ₁ ⊗ Φ₂

    where Φ₁ and Φ₂ are B-spline basis matrices for each dimension.

    B-splines provide:
    - Local support: changing one coefficient affects only a local region
    - Smoothness: controlled by the degree parameter
    - Numerical stability: well-conditioned basis

    Prior Types
    -----------
    - **global**: Uses symmetric structure with lower triangular basis. Samples a single
      set of coefficients to produce symmetric contact matrices. Reduces parameters to
      n(n+1)/2 basis evaluations.

    - **partial**: Separate coefficients for each effective dimension after transformation.
      Allows asymmetric contact patterns while sharing the same basis structure across
      dimensions.

    - **full**: Completely separate bases and coefficients for diagonal and off-diagonal
      elements. Maximum flexibility for heterogeneous contact patterns.

    Parameters
    ----------
    prior_type : {'global', 'partial', 'full'}
        The type of prior to use:
        - 'global': single symmetric matrix for global contact rates.
        - 'partial': multiple independent non-symmetric matrices.
        - 'full': multiple matrices with separate modeling of diagonal and off-diagonal elements.
    M : int or list of int, default=30
        Number of interior knots (equivalently, number of basis functions minus endpoints).
        If int, uses the same number for both dimensions. If list of length 2, specifies
        [M_x, M_y] for each dimension separately.
    degree : int or list of int, default=3
        Degree of B-spline polynomials. Common choices:
        - degree=1: Linear splines (piecewise linear)
        - degree=2: Quadratic splines
        - degree=3: Cubic splines (C² smooth, recommended default)
        - degree=4+: Higher order smoothness
        Higher degrees provide smoother interpolation but require more knots.
    grid_type : {'age-age', 'diff-age'}, default='age-age'
        The type of age grid to use. The 'age-age' grid smoothes over age pairs, while the
        'diff-age' grid smoothes over age differences and contact ages.

    Attributes
    ----------
    M : int or list
        Number of interior knots per dimension
    degree : int or list
        B-spline degree per dimension
    n_knots_inner : int
        Total interior knots: M + degree + 1
    n_knots_outer : int
        Boundary knots: 2 * (degree + 1)
    A : int
        Number of age groups (set by set_age_bounds)
    x, y : ndarray
        Scaled grid points in [0, 1]
    PHI : ndarray
        Tensor product basis matrix
    PHI_diag, PHI_non_diag : ndarray
        Separate basis matrices for 'full' prior type
    symm_tril_idx : ndarray
        Indices for symmetrizing lower triangular elements

    Methods
    -------
    set_age_bounds(min_age, max_age)
        Configure age range and construct basis matrices
    sample()
        Sample from the B-spline prior distribution
    tensor_spline_basis(x, y, n_knots, degree)
        Construct tensor product B-spline basis
    _set_grid()
        Initialize and scale grid points
    _set_basis()
        Construct basis matrices for the specified prior type
    _define_knots(x, n_knots, degree)
        Define knot sequence with boundary extensions

    Examples
    --------
    >>> from cntmosaic.models.numpyro.priors._Spline2D import Spline2D
    >>> import numpyro
    >>> import jax.numpy as jnp
    >>> from jax import random
    >>>
    >>> # Global prior for symmetric contact matrix
    >>> prior_global = Spline2D(
    ...     prior_type='global',
    ...     M=20,
    ...     degree=3
    ... )
    >>> prior_global.set_age_bounds(0, 80)  # 16 age groups (5-year bands)
    >>>
    >>> def model_global():
    ...     f = prior_global.sample()
    ...     return f
    >>>
    >>> with numpyro.handlers.seed(rng_seed=42):
    ...     sample = model_global()
    >>> print(sample.shape)  # (17, 17) - symmetric matrix
    >>> print(jnp.allclose(sample, sample.T))  # True
    >>>
    >>> # Partial prior with ILR transformation
    >>> prior_partial = Spline2D(
    ...     prior_type='partial',
    ...     M=25,
    ...     degree=3,
    ...     grid_type='age-age'
    ... )
    >>> prior_partial.set_age_bounds(0, 60)  # 13 age groups
    >>> prior_partial.set_event_dim(4)  # 4 contact settings
    >>> prior_partial.set_loc(jnp.zeros((3, 13, 13)))  # 3 = 4-1 for ILR
    >>>
    >>> def model_partial():
    ...     f = prior_partial.sample()
    ...     return f
    >>>
    >>> with numpyro.handlers.seed(rng_seed=42):
    ...     sample = model_partial()
    >>> print(sample.shape)  # (4, 13, 13)
    >>>
    >>> # Full prior with cubic splines
    >>> prior_full = Spline2D(
    ...     prior_type='full',
    ...     M=30,
    ...     degree=3,
    ... )
    >>> prior_full.set_age_bounds(0, 80)
    >>> prior_full.set_event_dim(9)  # 3x3 grid of settings
    >>> prior_full.set_loc(jnp.zeros((8, 17, 17)))  # 8 = 9-1 for ILR

    Notes
    -----
    - B-splines provide local support, making them computationally efficient for large grids
    - The knot sequence is constructed with 5% boundary extension to avoid edge effects
    - For 'global' prior, basis is evaluated only on lower triangular elements
    - The number of basis functions is approximately M per dimension
    - Computational complexity: O(M² × A²) for basis construction
    - Memory usage: Φ has shape (A² or A(A+1)/2, M²)

    See Also
    --------
    PSpline2D : Penalized B-splines with GMRF prior on coefficients
    HSGP2D : Hilbert space Gaussian process approximation
    Prior2D : Base class for 2D priors

    References
    ----------
    - de Boor, C. (2001). A Practical Guide to Splines. Springer.
    - Eilers, P. H., & Marx, B. D. (1996). Flexible smoothing with B-splines and penalties.
      Statistical Science, 11(2), 89-121.
    - Wood, S. N. (2017). Generalized Additive Models: An Introduction with R (2nd ed.).
      Chapman and Hall/CRC.
    """

    pytree_aux_fields = (
        "self.PHI",
        "self.PHI_T",
        "self.tril_idx",
        "self.symm_tril_idx",
    )

    def __init__(
        self,
        prior_type: str,
        M: int = 30,
        degree: int = 3,
        grid_type: str = "age-age",
        bound_ext: float = 0.05,
    ):

        validate_init_params(M, degree)
        super().__init__(grid_type, prior_type)
        self.M = M  # Number of basis functions (same for both dimensions)
        self.degree = degree  # Degree of B-splines (same for both dimensions)
        self.n_knots_inner = M + degree + 1
        self.n_knots_outer = 2 * (degree + 1)
        self.bound_ext = bound_ext

    def set_age_bounds(self, min_age: int, max_age: int):
        """
        Set age range for the contact matrix and construct B-spline basis.

        This method establishes the age boundaries, computes the number of age groups,
        initializes the grid structure, and constructs the tensor product B-spline
        basis matrices. Must be called before sampling from the prior.

        Parameters
        ----------
        min_age : int
            Minimum age (inclusive). Typically 0 for population-level contact matrices.
        max_age : int
            Maximum age (inclusive). Defines the upper bound of the oldest age group.

        Notes
        -----
        - Number of age groups: A = max_age - min_age + 1
        - Grid points are scaled to [0, 1] for numerical stability
        - Basis construction complexity: O(M² × A²)
        - For 'global' prior, basis is restricted to lower triangular elements

        The method performs the following steps:
        1. Sets age bounds and computes A
        2. Constructs and scales grid points via _set_grid()
        3. Builds B-spline basis matrices via _set_basis()

        Examples
        --------
        >>> prior = Spline2D(prior_type='global', M=20, degree=3)
        >>> prior.set_age_bounds(0, 75)
        >>> print(prior.A)  # Number of age groups
        16
        >>> print(prior.PHI.shape)  # Basis matrix for lower triangular
        (136, 400)  # 136 = 16*17/2, 400 = 20*20
        """
        self.min_age = min_age
        self.max_age = max_age
        self.A = max_age - min_age + 1

        self._set_grid()
        self._set_basis()

    def _set_grid(self):
        """
        Initialize and scale grid points (internal method).

        Constructs the spatial grid based on grid_type, extracts unique coordinate
        values, and scales them to [0, 1] for numerical stability in basis construction.

        The scaled coordinates enable:
        - Consistent knot placement across different age ranges
        - Numerical stability in B-spline evaluation
        - Uniform treatment of different age group sizes

        Notes
        -----
        For 'age-age' grid: Creates full A x A coordinate pairs
        For 'diff-age' grid: Creates age difference representation

        Raises
        ------
        ValueError
            If grid_type is not 'age-age' or 'diff-age'
        """
        if self.grid_type == "age-age":
            X = age_age_grid(self.A)
        elif self.grid_type == "diff-age":
            X = diff_age_age_grid(self.A)
        else:
            raise ValueError("grid_type must be 'age-age' or 'diff-age'")

        x = np.sort(np.unique(X[:, 0]))
        y = np.sort(np.unique(X[:, 1]))

        # Scale x and y to [0, 1]
        self.x = (x - self.min_age) / (self.max_age - self.min_age)
        self.y = (y - self.min_age) / (self.max_age - self.min_age)

        self.symm_tril_idx = symm_from_tril_ix_row(self.A)

    def _define_knots(self, x: NDArray, n_knots: int, degree: int) -> NDArray:
        """
        Define knot sequence for B-spline basis with boundary extensions.

        Constructs a knot vector with appropriate boundary repetitions and interior
        knot placement. The boundary extension prevents edge effects by extrapolating
        slightly beyond the data range.

        Parameters
        ----------
        x : ndarray
            Coordinate values for knot placement, scaled to [0, 1]
        n_knots : int
            Number of interior knots to place
        degree : int
            Degree of B-spline polynomials

        Returns
        -------
        knots : ndarray
            Complete knot sequence with boundary repetitions. Length is:
            2*(degree+1) + n_knots

        Notes
        -----
        Knot structure:
        - Boundary knots: (degree+1) repetitions at each end for proper boundary behavior
        - Interior knots: Placed at quantiles of x for even spacing
        - Extension: 5% beyond data range to avoid boundary artifacts

        The boundary extension improves extrapolation behavior near the edges of
        the age range, which is important for accurate contact matrix estimation
        at extreme ages.

        Examples
        --------
        >>> x = np.linspace(0, 1, 50)
        >>> knots = prior._define_knots(x, n_knots=10, degree=3)
        >>> print(len(knots))  # 2*(3+1) + 10 = 18
        >>> print(knots[:4])  # First 4 are repeated boundary knots
        """
        boundary_extension = (x.max() - x.min()) * self.bound_ext
        x_quantiles = np.quantile(x, np.linspace(0, 1, n_knots))
        return np.hstack(
            [
                [x.min() - boundary_extension] * (degree + 1),
                x_quantiles,
                [x.max() + boundary_extension] * (degree + 1),
            ]
        )

    def tensor_spline_basis(
        self, x: np.ndarray, y: np.ndarray, n_knots: list[int], degree: list[int]
    ) -> np.ndarray:
        """
        Construct tensor product B-spline basis matrix.

        Builds a 2D basis by computing the Kronecker product of 1D B-spline bases
        for each dimension. This creates a separable basis enabling efficient
        computation and interpretation of anisotropic smoothness.

        Parameters
        ----------
        x : ndarray, shape (n_x,)
            Scaled x-coordinates in [0, 1]
        y : ndarray, shape (n_y,)
            Scaled y-coordinates in [0, 1]
        n_knots : list of int
            Number of interior knots for each dimension (currently only first element used)
        degree : list of int
            B-spline degree for each dimension (currently only first element used)

        Returns
        -------
        PHI : ndarray, shape (n_grid, M²)
            Tensor product basis matrix where:
            - n_grid = n_x * n_y for 'age-age' grid
            - n_grid = reduced for 'diff-age' grid
            - M² is the total number of basis functions

        Notes
        -----
        Construction process:
        1. Define knot sequences for both dimensions (using same knots for isotropy)
        2. Evaluate 1D B-spline bases: Φ₁(x) and Φ₂(y)
        3. Form tensor product: Φ = Φ₁ ⊗ Φ₂ (Kronecker product)
        4. For 'diff-age' grid, subsample rows based on age difference structure

        The first column is excluded ([: , 1:]) to avoid numerical issues with
        boundary basis functions.

        Complexity: O(M² × A²) where M is the number of basis functions per dimension
        and A is the number of age groups.

        Examples
        --------
        >>> x = np.linspace(0, 1, 20)
        >>> y = np.linspace(0, 1, 20)
        >>> PHI = prior.tensor_spline_basis(x, y, n_knots=[30], degree=[3])
        >>> print(PHI.shape)  # (400, ~900) for age-age grid
        """

        x_knots = self._define_knots(x, n_knots, degree)
        y_knots = x_knots  # Use same knots for both dimensions to ensure isotropic basis structure

        PHI1 = BSpline(x_knots, np.eye(len(x_knots) - degree - 1), degree)(x)[
            :, 1:
        ]  # shape: (len(x), M)
        PHI2 = BSpline(y_knots, np.eye(len(y_knots) - degree - 1), degree)(y)[
            :, 1:
        ]  # shape: (len(y), M)

        if self.grid_type == "age-age":
            return np.kron(PHI1, PHI2)
        elif self.grid_type == "diff-age":
            diff_age_idx = diff_age_age_index(self.A)
            return np.kron(PHI1, PHI2)[diff_age_idx]

    def _set_basis(self):
        """
        Construct basis matrices based on prior type (internal method).

        Creates the appropriate B-spline basis matrices for the specified prior_type:
        - 'global': Single basis on lower triangular elements for symmetric matrices
        - 'partial': Full basis shared across event dimensions
        - 'full': Separate bases for diagonal and off-diagonal elements

        The method sets:
        - self.PHI: Main basis matrix for global/partial types
        - self.PHI_diag: Diagonal basis for full type
        - self.PHI_non_diag: Off-diagonal basis for full type
        - self.symm_tril_idx: Symmetrization indices

        Notes
        -----
        Basis dimensions:
        - Global: (A(A+1)/2, M²) - lower triangular only
        - Partial: (A², M²) - full grid
        - Full: PHI_diag is (A(A+1)/2, M²), PHI_non_diag is (A², M²)

        The basis construction uses the same knot sequence for both dimensions
        to maintain isotropic properties while allowing anisotropic smoothing
        through the coefficient structure.
        """
        n_basis_functions = (
            self.n_knots_inner - self.n_knots_outer + 1
        )  # Clear variable name
        self.PHI = self.tensor_spline_basis(
            self.x, self.y, n_basis_functions, self.degree
        )

        if self.prior_type == "global":
            self.symm_tril_idx = symm_from_tril_ix_row(self.A)
            self.PHI = self.PHI[tril_ix_row(self.A)]

        elif self.prior_type == "full":  # Full case
            self.symm_tril_idx = symm_from_tril_ix_row(self.A)
            self.PHI_diag = self.PHI[tril_ix_row(self.A)]
            self.PHI_non_diag = self.PHI

    def sample_global(self):
        beta = numpyro.sample(
            "spline_coefs", dist.Normal(0, 1), sample_shape=(self.PHI.shape[-1],)
        )
        f = (self.PHI @ beta)[self.symm_tril_idx].reshape((self.A, self.A))
        return f

    def sample_partial(self):
        beta = numpyro.sample(
            "spline_coefs",
            dist.Normal(0, 1),
            sample_shape=(self.PHI.shape[-1], self.event_dim),
        )
        f = self.PHI @ beta
        f = f.swapaxes(0, 1)
        f = f.reshape((self.event_dim, self.A, self.A))
        return self.loc + f

    def sample_full(self):
        beta_diag = numpyro.sample(
            "spline_coefs_diag",
            dist.Normal(0, 1),
            sample_shape=(self.PHI_diag.shape[-1], self.event_dim_diag),
        )
        beta_non_diag = numpyro.sample(
            "spline_coefs_non_diag",
            dist.Normal(0, 1),
            sample_shape=(self.PHI_non_diag.shape[-1], self.event_dim_non_diag_eff),
        )

        f_diag = self.PHI_diag @ beta_diag
        f_diag = f_diag[self.symm_tril_idx, :].swapaxes(0, 1)  # Must be symmetric
        f_diag = f_diag.reshape((self.event_dim_diag, self.A, self.A))

        f_non_diag = self.PHI_non_diag @ beta_non_diag
        f_non_diag = f_non_diag.swapaxes(0, 1)
        f_non_diag = f_non_diag.reshape((self.event_dim_non_diag_eff, self.A, self.A))

        # Assemble diagonal and off-diagonal blocks into full event grid
        f = self._assemble_full_prior_blocks(f_diag, f_non_diag)

        return self.loc + f

    def sample(self):
        """
        Sample from the tensor product B-spline prior.

        Generates contact matrix samples by drawing coefficients from standard normal
        distributions and combining them with the pre-computed B-spline basis matrices.
        The sampling behavior depends on the prior_type configuration.

        Returns
        -------
        f : array
            Sampled contact matrix with shape depending on prior_type:
            - 'global': (A, A) - symmetric matrix
            - 'partial': (event_dim, A, A) - potentially asymmetric, event_dim matrices
            - 'full': (event_dim, A, A) - with separate diagonal/off-diagonal structure

            After inverse transformation if transform is specified.

        Notes
        -----
        **Global Prior** (symmetric matrices):
        - Samples: β ~ N(0, I_M²)
        - Computes: f = Φ β
        - Symmetrizes using lower triangular indices
        - Returns: (A, A) symmetric matrix

        **Partial Prior** (shared basis, dimension-specific coefficients):
        - Samples: β ~ N(0, I_{M² × event_dim})
        - Computes: f = Φ β for each dimension
        - Adds location parameter: f += loc
        - Returns: (event_dim, A, A)

        **Full Prior** (separate diagonal/off-diagonal):
        - Samples: β_diag ~ N(0, I_{M² × event_dim_diag})
        - Samples: β_non_diag ~ N(0, I_{M² × event_dim_non_diag_eff})
        - Computes: f_diag = Φ_diag β_diag (symmetrized)
        - Computes: f_non_diag = Φ_non_diag β_non_diag
        - Combines diagonal and off-diagonal elements
        - Adds location parameter: f += loc
        - Returns: (event_dim, A, A)

        Raises
        ------
        ValueError
            If prior_type is not one of {'global', 'partial', 'full'}

        Examples
        --------
        >>> import numpyro
        >>> from jax import random
        >>> import jax.numpy as jnp
        >>>
        >>> # Global prior - symmetric contact matrix
        >>> prior = Spline2D(prior_type='global', M=20, degree=3)
        >>> prior.set_age_bounds(0, 50)
        >>>
        >>> def model():
        ...     with numpyro.handlers.seed(rng_seed=42):
        ...         f = prior.sample()
        ...     return f
        >>>
        >>> sample = model()
        >>> print(sample.shape)  # (11, 11)
        >>> print(jnp.allclose(sample, sample.T))  # True - symmetric
        >>>
        >>> # Partial prior with ILR transformation
        >>> prior_partial = Spline2D(
        ...     prior_type='partial',
        ...     M=25,
        ...     degree=3,
        ...     transform='alr'
        ... )
        >>> prior_partial.set_age_bounds(0, 40)
        >>> prior_partial.set_event_dim(4)  # 4 contact settings
        >>> prior_partial.set_loc(jnp.zeros((3, 9, 9)))  # 3 = 4-1 for ILR
        >>>
        >>> def model_partial():
        ...     with numpyro.handlers.seed(rng_seed=42):
        ...         f = prior_partial.sample()
        ...     return f
        >>>
        >>> sample_partial = model_partial()
        >>> print(sample_partial.shape)  # (4, 9, 9)
        >>> # Each slice sums to 1 along first axis (simplex constraint)
        >>> print(jnp.allclose(sample_partial.sum(axis=0), 1.0))  # True

        See Also
        --------
        tensor_spline_basis : Basis matrix construction
        _set_basis : Basis initialization for different prior types
        inverse_alr, inverse_clr, inverse_ilr : Compositional transformations
        """

        if self.prior_type == "global":
            return self.sample_global()

        elif self.prior_type == "partial":
            return self.sample_partial()

        elif self.prior_type == "full":
            return self.sample_full()

        else:
            raise ValueError(
                f"Invalid prior_type: {self.prior_type}. "
                f"Must be one of ['global', 'partial', 'full']"
            )
