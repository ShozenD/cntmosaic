from typing import List, Union

import jax.numpy as jnp
import numpy as np
import numpyro
from jax import vmap
from jax.typing import ArrayLike
from numpyro import distributions as dist
from numpyro.contrib.hsgp.laplacian import eigenfunctions
from numpyro.contrib.hsgp.spectral_densities import diag_spectral_density_matern

from .._utils import age_age_grid, diff_age_age_grid, symm_from_tril_ix_row, tril_ix_row
from ._Prior2D import Prior2D


def validate_init_params(
    nu: float, C: Union[float, List[float]], M: Union[int, List[int]]
):
    """
    Validate initialization parameters for HSGP2D prior.

    Parameters
    ----------
    nu : float
        The smoothness parameter of the Matern kernel.
    C : float or list of float
        The boundary inflation factors for the HSGP approximation.
    M : int or list[int]
        The number of eigenvalues and eigenfunctions to use in the HSGP approximation.

    Raises
    ------
    ValueError
        If nu is not one of [1/2, 3/2, 5/2, 7/2], or if C or M are not positive.
    TypeError
        If C is not a float or list of two floats, or if M is not an int or list of two ints.
    ValueError
        If M includes negative integers.
    TypeError
        If M is not an integer or a list of integers.
    """
    if nu not in [1 / 2, 3 / 2, 5 / 2, 7 / 2]:
        raise ValueError("nu must be one of [1/2, 3/2, 5/2, 7/2]")

    if isinstance(C, float):
        if C <= 0:
            raise ValueError("C must be positive.")
    elif isinstance(C, list):
        if len(C) != 2 or any(c <= 0 for c in C):
            raise ValueError("C must be a list of two positive floats.")
    else:
        raise TypeError("C must be a float or a list of two floats.")

    if isinstance(M, int):
        if M <= 0:
            raise ValueError("M must be a positive integer.")
    elif isinstance(M, list):
        if len(M) != 2 or any(m <= 0 for m in M):
            raise ValueError("M must be a list of two positive integers.")
    else:
        raise TypeError("M must be an integer or a list of two integers.")


class HSGP2D(Prior2D):
    """
    2D Hilbert space approximate Gaussian process (HSGP) prior for contact matrix estimation.

    This prior class implements flexible spatial smoothing for contact matrices using
    a 2D HSGP characterized by the Matern kernel. It provides a computationally efficient
    alternative to traditional Gaussian process priors by approximating the GP using a finite
    set of eigenfunctions derived from the Laplacian operator.

    Prior Types
    -----------
    - **global**: A single symmetric matrix for modeling global contact rate patterns.
      Only the lower triangular part of the matrix is modeled to ensure symmetry.

    - **partial**: Multiple independent and non-symmetric matrices for modeling different
      patterns in contact rates and intensities across different participant subgroups.
      Each matrix is fully modeled without symmetry constraints.

    - **full**: Multiple matrices for modeling contact rates and intensities across different
      participant and contact subgroups. Diagonal and off-diagonal elements are modeled
      separately to ensure reciprocity constraints.


    Parameters
    ----------
    prior_type : {'global', 'partial', 'full'}
        The type of prior to use:
        - 'global': single symmetric matrix for global contact rates.
        - 'partial': multiple independent non-symmetric matrices.
        - 'full': multiple matrices with separate modeling of diagonal and off-diagonal elements.
    grid_type : {'age-age', 'diff-age'}, default='age-age'
        The type of age grid to use. The 'age-age' grid smoothes over age pairs, while the
        'diff-age' grid smoothes over age differences and contact ages.
    nu : float, default=5/2
        The smoothness parameter of the Matern kernel. Common choices are:
        - 3/2: Sampled functions are once differentiable
        - 5/2: Sampled functions are twice differentiable
        - 7/2: Sampled functions are thrice differentiable (becomes similar to squared exponential kernel)
    C: float or list of float, default=[1.5, 1.5]
        The boundary inflation factors for the HSGP approximation. Must be tuned empirically.
    M: int or list[int], default=[30, 30]
        The number of eigenvalues and eigenfunctions to use in the HSGP approximation.
        A higher number increases accuracy but also computational cost.
    transform : {None, 'alr', 'clr', 'ilr'}, default=None
        Compositional transformation to use when mapping from the unconstrained space to the simplex:
        - None: No transformation
        - 'alr': Additive log-ratio transformation (recommended)
        - 'clr': Centered log-ratio transformation
        - 'ilr': Isometric log-ratio transformation

    Examples
    --------
    >>> import numpyro
    >>> from cntmosaic.models.priors import HSGP2D
    >>> import jax.numpy as jnp
    >>>
    >>> # Global rate prior
    >>> prior = HSGP2D(prior_type="global", grid_type="age-age")
    >>> prior.set_age_bounds(0, 84)
    >>> def model_global():
    ...     f = prior.sample()
    ...     return f
    >>>
    >>> with numpyro.handlers.seed(rng_seed=0):
    ...     result = model_global()
    >>> # Resulting contact matrix shape
    >>> print(result.shape) # (85, 85) - symmetric matrix
    >>> print(jnp.allclose(result, result.T))  # True

    See Also
    --------
    Prior2D : Base class for 2D priors.
    Spline2D : Spline with independent normal priors on coefficients.
    PSpline2D : Penalized B-splines with GMRF prior on coefficients.

    References
    ----------
    - Solin, A., & Särkkä, S. (2020). Hilbert space methods for reduced-rank Gaussian process regression.
      Statistics and Computing, 30(2), 419-446.
    - Riutort-Mayol et al. (2022). Practical Hilbert space approximate Bayesian Gaussian processes for
      probabilistic programming. Statistics and Computing, 32(2), 1-21.
    - Dan et al. (2023). Estimating fine age structure and time trends in human contact patterns from
      from coarse contact data: The Bayesian rate consistency model. PLOS Computational Biology. 19(6), e1011191.
    """

    pytree_aux_fields = ("self.nu", "self.C", "self.M", "self.PHI", "self.symm_tril_ix")

    def __init__(
        self,
        prior_type: str,
        grid_type: str = "age-age",
        nu: float = 5 / 2,
        C: Union[float, List[float]] = [2.0, 2.0],
        M: Union[int, List[int]] = [30, 30],
        transform: Union[str, None] = None,
    ) -> None:
        validate_init_params(nu, C, M)
        super().__init__(grid_type, transform, prior_type)
        self.nu = nu
        self.C = C
        self.M = M

    def set_age_bounds(self, min_age: int, max_age: int) -> None:
        """
        Set age range for the contact matrix, construct the grid, and build eigenfunction matrix.

        This method establishes the age boundaries, computes the number of age groups,
        initializes the grid structure, and constructs the eigenfunction matrices.
        Must be called before sampling from the prior.

        Parameters
        ----------
        min_age : int
            Minimum age (inclusive). Typically 0 for population-level contact matrices.
        max_age : int
            Maximum age (inclusive). Defines the upper bound of the oldest age group.

        The method performs the following steps:
        1. Sets age bounds and computes A
        2. Constructs grid points via _set_grid()
        3. Builds eigenfunction matrices via _set_eigenfunctions()
        """
        self.min_age = min_age
        self.max_age = max_age
        self.A = max_age - min_age + 1

        self._set_grid()
        self._set_eigenfunctions()

    def _set_grid(self) -> None:
        """
        Intialize and scale the grid points for the HSGP approximation (internal method).

        This method constructs the grid points based on the specified grid type
        ('age-age' or 'diff-age') and standardizes them such that they can be used
        as inputs to the eigenfunction calculations.

        Notes
        -----
        For 'age-age' grid: Creates full A x A coordinate pairs
        For 'diff-age' grid: Creates age difference representation

        Raises
        ------
        ValueError
            If grid_type is not one of 'age-age' or 'diff-age'.
        """
        if self.grid_type == "age-age":
            X = age_age_grid(self.A)
        elif self.grid_type == "diff-age":
            X = diff_age_age_grid(self.A)
        else:
            raise ValueError("grid_type must be 'age-age' or 'diff-age'")

        Xn = (X - X.mean(axis=0)) / X.std(axis=0)
        self.L = list(np.abs(Xn).max(axis=0) * self.C)

        if self.prior_type == "global":
            tril_idx = tril_ix_row(self.A)
            self.X_tril = Xn[tril_idx]
            self.symm_tril_ix = symm_from_tril_ix_row(self.A)
        if self.prior_type == "full":
            tril_idx = tril_ix_row(self.A)
            self.X = Xn
            self.X_tril = Xn[tril_idx]
            self.symm_tril_ix = symm_from_tril_ix_row(self.A)
        else:
            self.X = Xn

    def _set_eigenfunctions(self) -> None:
        """Build the eigenfunction matrix for the HSGP approximation (internal method)."""
        if self.prior_type == "global":
            self.PHI = eigenfunctions(
                x=self.X_tril, ell=self.L, m=self.M
            )  # (N_tri, M1*M2)
        elif self.prior_type == "full":
            self.PHI_diag = eigenfunctions(
                x=self.X_tril, ell=self.L, m=self.M
            )  # (N_tri, M1*M2)
            self.PHI_non_diag = eigenfunctions(
                x=self.X, ell=self.L, m=self.M
            )  # (N_tri, M1*M2)
        else:
            self.PHI = eigenfunctions(x=self.X, ell=self.L, m=self.M)  # (N, M1*M2)

    def _compute_sqrt_diag_spd(self, alpha, length) -> ArrayLike:
        """
        Compute the square root of the diagonal spectral density for the Matern kernel,
        given the scale (alpha) and lengthscale parameters (internal method).

        Parameters
        ----------
        alpha : float
            The scale parameter of the Matern kernel.
        length : float
            The lengthscale parameter of the Matern kernel.

        Returns
        -------
        spd : ArrayLike
            The diagonal spectral density evaluated at the eigenvalues.
        """
        return jnp.sqrt(
            diag_spectral_density_matern(
                nu=self.nu, alpha=alpha, length=length, ell=self.L, m=self.M, dim=2
            )
        )

    def sample_global(self) -> ArrayLike:
        # sample GP scale parameter
        sigma = numpyro.sample("gp_scale", dist.HalfNormal(1.0))
        # sample GP lengthscale parameter
        lenscale = numpyro.sample("gp_lenscale", dist.HalfNormal(1.0))

        # compute diagonal spectral density
        sqrt_spd = self._compute_sqrt_diag_spd(sigma, lenscale)

        # Sample eigenfunction coefficients
        beta = numpyro.sample(
            "gp_coefs", dist.Normal(0.0, 1.0), sample_shape=(self.PHI.shape[-1],)
        )

        f = self.PHI @ (sqrt_spd * beta)
        f = f[self.symm_tril_ix]
        return f.reshape((self.A, self.A))

    def sample_partial(self) -> ArrayLike:
        # Sample GP scale parameters for each event
        sigma = numpyro.sample(
            "gp_scale", dist.HalfNormal(1.0), sample_shape=(self.event_dim_latent,)
        )
        lenscale = numpyro.sample(
            "gp_lenscale", dist.HalfNormal(1.0), sample_shape=(self.event_dim_latent,)
        )

        # Vectorized SPD (shape: (event_dim_latent, num_basis))
        sqrt_spd = jnp.squeeze(vmap(self._compute_sqrt_diag_spd)(sigma, lenscale))

        # Sample eigenfunction coefficients
        beta = numpyro.sample(
            "gp_coefs",
            dist.Normal(0.0, 1.0),
            sample_shape=(self.event_dim_latent, self.PHI.shape[-1]),
        )

        f = (sqrt_spd * beta) @ self.PHI.T  # shape: (event_dim_latent, N)
        f = f.reshape((self.event_dim_latent, self.A, self.A))
        f = self.trans_loc + f
        return self.apply_inverse_transform(f)

    def sample_full(self) -> ArrayLike:
        # Sample GP scale parameters for diagonal and non-diagonal elements
        sigma_diag = numpyro.sample(
            "gp_scale_diag",
            dist.HalfNormal(1.0),
            sample_shape=(self.event_dim_diag_eff,),
        )
        lenscale_diag = numpyro.sample(
            "gp_lenscale_diag",
            dist.HalfNormal(1.0),
            sample_shape=(self.event_dim_diag_eff,),
        )

        sigma_non_diag = numpyro.sample(
            "gp_scale_non_diag",
            dist.HalfNormal(1.0),
            sample_shape=(self.event_dim_non_diag_eff,),
        )
        lenscale_non_diag = numpyro.sample(
            "gp_lenscale_non_diag",
            dist.HalfNormal(1.0),
            sample_shape=(self.event_dim_non_diag_eff,),
        )

        # Vectorized SPD
        sqrt_spd_diag = jnp.squeeze(
            vmap(self._compute_sqrt_diag_spd)(sigma_diag, lenscale_diag)
        )
        sqrt_spd_non_diag = jnp.squeeze(
            vmap(self._compute_sqrt_diag_spd)(sigma_non_diag, lenscale_non_diag)
        )

        # Sample eigenfunction coefficients
        beta_diag = numpyro.sample(
            "gp_coefs_diag",
            dist.Normal(0.0, 1.0),
            sample_shape=(self.event_dim_diag_eff, self.PHI_diag.shape[-1]),
        )
        beta_non_diag = numpyro.sample(
            "gp_coefs_non_diag",
            dist.Normal(0.0, 1.0),
            sample_shape=(self.event_dim_non_diag_eff, self.PHI_non_diag.shape[-1]),
        )

        f_diag = (sqrt_spd_diag * beta_diag) @ self.PHI_diag.T
        f_diag = f_diag[:, self.symm_tril_ix]
        f_diag = f_diag.reshape((self.event_dim_diag_eff, self.A, self.A))

        f_non_diag = (sqrt_spd_non_diag * beta_non_diag) @ self.PHI_non_diag.T
        f_non_diag = f_non_diag.reshape((self.event_dim_non_diag_eff, self.A, self.A))

        # Allocate diagonal and off-diagonal elements into full K×K contact matrix grid
        f = self._assemble_full_prior_blocks(f_diag, f_non_diag)
        f = self.trans_loc + f
        return self.apply_inverse_transform(f)

    def sample(self) -> ArrayLike:
        """
        Sample from the HSGP2D prior.

        Generates contact matrix samples by drawing from the specified HSGP prior type
        ('global', 'partial', or 'full') and applying the appropriate transformations.

        Returns
        -------
        f : array
            Sampled contact matrix with shape depending on prior_type:
            - 'global': (A, A) - symmetric matrix
            - 'partial': (event_dim, A, A) - asymmetric, event_dim matrices
            - 'full': (event_dim, A, A) - with separate diagonal/off-diagonal structure

        Examples
        --------
        >>> import numpyro
        >>> from jax import random
        >>> import jax.numpy as jnp
        >>>
        >>> # Global prior - symmetric contact matrix
        >>> prior = HSGP2D(prior_type='global')
        >>> prior.set_age_bounds(0, 50)
        >>>
        >>> with numpyro.handlers.seed(rng_seed=42):
        ...     sample = prior.sample()
        >>>
        >>> print(sample.shape)  # (51, 51)
        >>> print(jnp.allclose(sample, sample.T))  # True - symmetric
        >>>
        >>> # Partial prior with ILR transformation
        >>> prior_partial = HSGP2D(
        ...     prior_type='partial',
        ...     transform='ilr'
        ... )
        >>> prior_partial.set_age_bounds(0, 40)
        >>> prior_partial.set_event_dim(2)  # 2 categories
        >>>
        >>> with numpyro.handlers.seed(rng_seed=42):
        ...     sample_partial = prior_partial.sample()
        >>>
        >>> print(sample_partial.shape)  # (2, 41, 41)
        >>>
        >>> # Full prior with ILR transformation
        >>> prior_full = HSGP2D(
        ...     prior_type='full',
        ...     transform='ilr'
        ... )
        >>> prior_full.set_age_bounds(0, 30)
        >>> prior_full.set_event_dim(3)  # 3 categories
        >>> with numpyro.handlers.seed(rng_seed=42):
        ...     sample_full = prior_full.sample()
        >>>
        >>> print(sample_full.shape)  # (9, 31, 31)
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
