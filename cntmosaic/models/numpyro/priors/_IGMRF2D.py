from typing import Optional

import numpyro
from jax.typing import ArrayLike
from numpyro import distributions as dist

from ..distributions._IGMRF2D import IGMRF2D as IGMRF2D_dist
from ..distributions._SymIGMRF2D import SymIGMRF2D
from ..._utils import symm_from_tril_ix_col
from ._Prior2D import Prior2D


class IGMRF2D(Prior2D):
    """
    2D Intrinsic Gaussian Markov Random Field (IGMRF) prior for contact matrix estimation.

    This prior class implements spatial smoothing for contact matrices using intrinsic
    Gaussian Markov random fields with separable precision structure. It wraps the IGMRF2D
    and SymIGMRF2D distributions from the distributions module, providing different prior
    structures (global, partial, full) suitable for social contact matrix inference.

    The IGMRF prior penalizes roughness in the latent contact pattern through finite
    differences, enabling flexible spatial smoothing while maintaining computational
    efficiency. It supports both symmetric (global) and asymmetric (partial, full) contact
    matrices with optional compositional transformations.

    Mathematical Background
    -----------------------
    The prior imposes a Gaussian structure on the latent field f with precision matrix:

        Q = τ₁ ⊗ L₁ ⊗ I₂ + τ₂ ⊗ I₁ ⊗ L₂

    where L₁ and L₂ are Laplacian matrices of specified order, τ₁ and τ₂ control
    smoothness in each dimension, and ⊗ denotes the Kronecker product. Higher orders
    penalize more derivatives, producing smoother fields.

    Prior Types
    -----------
    - **global**: Uses SymIGMRF2D for symmetric contact matrices. Suitable for modeling
      reciprocal contact patterns where n(i,j) = n(j,i). Reduces parameters to n(n+1)/2.

    - **partial**: Uses IGMRF2D_dist with shared precision across rows/columns. Allows
      asymmetry while sharing smoothness parameters. Typically used with compositional
      transformations (ALR, CLR, ILR).

    - **full**: Separate priors for diagonal and off-diagonal elements. Maximum flexibility
      for heterogeneous contact patterns (e.g., within-group vs. between-group contacts).

    Parameters
    ----------
    prior_type : {'global', 'partial', 'full'}
        Structure of the prior:
        - 'global': Symmetric matrix with single shared prior
        - 'partial': Asymmetric with row/column-specific priors
        - 'full': Separate priors for diagonal and off-diagonal elements
    grid_type : {'age-age', 'diff-age'}, default='age-age'
        Grid structure for contact matrix:
        - 'age-age': Standard age-by-age contact matrix
        - 'diff-age': Age difference representation
    num_nodes : tuple of int
        Number of nodes in each dimension (n₁, n₂). For age-structured contact matrices,
        typically (A, A) where A is the number of age groups.
    order : tuple of int
        Order of the finite difference operator in each dimension (k₁, k₂). Common choices:
        - (1, 1): First-order differences (random walk, less smooth)
        - (2, 2): Second-order differences (smoother, penalizes curvature)
        Higher orders produce progressively smoother fields.
    loc : float or array-like, default=0.0
        Prior location (mean) parameter. Can be scalar or array of shape (event_dim, A, A).
        After transformation is applied in Prior2D.set_loc().

    Attributes
    ----------
    num_nodes : tuple of int
        Dimensionality of the 2D grid
    order : tuple of int
        Finite difference order for each dimension
    A : int
        Number of age groups (set by set_age_bounds)
    sym_idx : array
        Indices for symmetrizing lower triangular elements (global prior only)
    loc : array
        Location parameter (mean) (partial/full priors)
    event_dim_diag : int
        Dimension for diagonal elements (full prior only)
    event_dim_non_diag : int
        Dimension for off-diagonal elements (full prior only)

    Methods
    -------
    set_age_bounds(min_age, max_age)
        Configure age range for the contact matrix
    sample()
        Sample from the appropriate prior distribution based on prior_type
    sample_global()
        Sample symmetric contact matrix using SymIGMRF2D
    sample_partial()
        Sample asymmetric contact matrix with shared precision
    sample_full()
        Sample contact matrix with separate diagonal/off-diagonal priors

    Examples
    --------
    >>> from cntmosaic.models.numpyro.priors._IGMRF2D import IGMRF2D
    >>> import numpyro
    >>> from jax import random
    >>>
    >>> # Global prior for symmetric contact matrix (no transformation)
    >>> prior = IGMRF2D(
    ...     prior_type='global',
    ...     order=(2, 2)
    ... )
    >>> prior.set_age_bounds(0, 30) # 31 ages
    >>>
    >>> # Sample from the prior
    >>> def model():
    ...     f = prior.sample()  # Symmetric 31x31 contact matrix
    ...     return f
    >>> key = random.PRNGKey(0)
    >>> with numpyro.handlers.seed(rng_seed=0):
    ...     samples = model()
    >>>
    >>> # Partial prior with CLR transformation for compositional data
    >>> prior_clr = IGMRF2D(
    ...     prior_type='partial',
    ...     transform='clr'
    ... )
    >>> prior_clr.set_event_dim(16)  # K=16 age groups
    >>> prior_clr.set_age_bounds(0, 30)  # 31 ages
    >>>
    >>> # Full prior with separate smoothness for diagonal/off-diagonal
    >>> prior_full = IGMRF2D(
    ...     prior_type='full',
    ...     transform='alr'
    ... )

    Notes
    -----
    - The IGMRF is intrinsic (improper) due to zero eigenvalues. The distribution
      automatically projects onto the subspace orthogonal to polynomial trends.
    - For 'global' prior, the matrix is automatically symmetrized after sampling.
    - The precision parameters τ (tau) control smoothness and are sampled from
      Gamma(2, 0.1) distributions in partial/full priors.
    - Compositional transformations require appropriate loc shapes:
      * ALR/CLR: (A-1, A, A) or (A, A, A) depending on transformation
      * ILR: (A(A-1)/2, A, A)

    See Also
    --------
    IGMRF2D : Distribution class for 2D IGMRFs
    SymIGMRF2D : Distribution for symmetric 2D IGMRFs
    Prior2D : Base class for 2D priors
    HSGP2D : Alternative prior using Hilbert space Gaussian processes

    References
    ----------
    - Rue, H., & Held, L. (2005). Gaussian Markov Random Fields: Theory and Applications.
      Chapman & Hall/CRC.
    - Prem, K., et al. (2017). Projecting social contact matrices in 152 countries using
      contact surveys and demographic data. PLOS Computational Biology.
    """

    def __init__(
        self,
        prior_type: str,
        grid_type: Optional[str] = "age-age",
        order: tuple = (2, 2),
        loc: ArrayLike = 0.0,
    ):
        super().__init__(grid_type, prior_type)
        self.loc = loc
        self.order = order

    def set_age_bounds(self, min_age: int, max_age: int):
        """
        Set the age range for the contact matrix and configure grid structure.

        This method establishes the age boundaries for the contact matrix, computes
        the number of age groups, and initializes the grid structure. Must be called
        before sampling from the prior.

        Parameters
        ----------
        min_age : int
            Minimum age (inclusive). Typically 0 for population-level contact matrices.
        max_age : int
            Maximum age (inclusive). The actual maximum is min(max_age, max_population_age).

        Notes
        -----
        The number of age groups A is computed as: A = max_age - min_age + 1

        For global priors, this method also initializes sym_idx, which maps lower
        triangular elements to the full symmetric matrix.

        Examples
        --------
        >>> prior = IGMRF2D(num_nodes=(16, 16), order=(2, 2))
        >>> prior.set_age_bounds(0, 80)
        >>> print(prior.A)  # Number of age groups
        16
        """
        self.min_age = min_age
        self.max_age = max_age
        self.A = max_age - min_age + 1

        self._set_grid()

    def _set_grid(self):
        """
        Initialize grid structure for symmetrization (internal method).

        For global priors, computes indices to symmetrize the contact matrix from
        lower triangular elements. This enables efficient storage and sampling of
        symmetric contact matrices.
        """
        self.num_nodes = (self.A, self.A)
        self.symm_tril_ix = symm_from_tril_ix_col(self.A)

    def sample_global(self):
        """
        Sample symmetric contact matrix using SymIGMRF2D distribution.

        This method generates a symmetric contact matrix by sampling from the
        SymIGMRF2D distribution, which enforces the constraint n(i,j) = n(j,i).
        Symmetric matrices are appropriate when reciprocal contact patterns are
        expected (e.g., contacts at home or in social settings).

        The distribution samples n(n+1)/2 unique elements (lower triangular) and
        mirrors them to form a full symmetric matrix.

        Returns
        -------
        f : array, shape (A, A)
            Symmetric contact matrix sampled from the IGMRF prior. No transformation
            is applied in global mode.

        Notes
        -----
        - No precision parameters τ are sampled; SymIGMRF2D has implicit unit precision
        - The matrix is automatically symmetrized during sampling
        - Transformations are not applied in global mode (transform parameter is ignored)

        Examples
        --------
        >>> import numpyro
        >>> from jax import random
        >>>
        >>> prior = IGMRF2D(num_nodes=(10, 10), order=(2, 2), prior_type='global')
        >>> prior.set_age_bounds(0, 50)
        >>>
        >>> def model():
        ...     f = prior.sample_global()
        ...     return f
        >>>
        >>> key = random.PRNGKey(0)
        >>> with numpyro.handlers.seed(rng_seed=0):
        ...     sample = model()
        >>> print(sample.shape)  # (10, 10)
        >>> print(jnp.allclose(sample, sample.T))  # True (symmetric)
        """
        f = numpyro.sample("f", SymIGMRF2D(self.num_nodes[0], self.order[0]))
        return f[self.symm_tril_ix].reshape((self.A, self.A))

    def sample_partial(self):
        """
        Sample asymmetric contact matrix with shared precision across dimensions.

        This method generates an asymmetric contact matrix allowing n(i,j) ≠ n(j,i),
        suitable for modeling non-reciprocal contact patterns (e.g., age-specific
        infection transmission). It samples precision parameters τ for each effective
        dimension and applies compositional transformations.

        The latent field f is sampled from IGMRF2D_dist with conditional precision
        τ in the first dimension, then shifted by loc.

        Returns
        -------
        f : array, shape (event_dim, A, A)
            Contact matrix.

        Notes
        -----
        - Precision τ ~ Gamma(2, 0.1) independently for each event dimension
        - loc is added to f

        Examples
        --------
        >>> import numpyro
        >>> import jax.numpy as jnp
        >>> from jax import random
        >>>
        >>> # Partial prior with CLR transformation
        >>> prior = IGMRF2D(
        ...     num_nodes=(16, 16),
        ...     order=(2, 2),
        ...     transform='clr',
        ...     prior_type='partial'
        ... )
        >>> prior.set_age_bounds(0, 80)
        >>> prior.set_loc(jnp.zeros((16, 16, 16)))  # A x A x A
        >>>
        >>> def model():
        ...     f = prior.sample_partial()
        ...     return f
        >>>
        >>> key = random.PRNGKey(0)
        >>> with numpyro.handlers.seed(rng_seed=0):
        ...     sample = model()
        >>> print(sample.shape)  # (16, 16, 16)

        See Also
        --------
        sample_full : Separate priors for diagonal and off-diagonal elements
        apply_inverse_transform : Apply inverse compositional transformation
        """
        tau = numpyro.sample("tau", dist.Gamma(2, 0.1), sample_shape=(self.event_dim,))
        f = numpyro.sample(
            "f", IGMRF2D_dist(self.num_nodes, self.order, cond_prec1=tau)
        ).reshape((self.event_dim, self.A, self.A))

        return self.loc + f

    def sample_full(self):
        """
        Sample contact matrix with separate priors for diagonal and off-diagonal elements.

        This method provides maximum flexibility by using separate IGMRF priors with
        independent precision parameters for diagonal elements (within-group contacts)
        and off-diagonal elements (between-group contacts). This is useful when
        within-group and between-group contact patterns have different smoothness
        characteristics.

        Returns
        -------
        f : array, shape (event_dim, A, A)
            Contact matrix. Diagonal and off-diagonal
            elements are sampled independently then combined.

        Notes
        -----
        - tau_diag ~ Gamma(2, 0.1) for diagonal elements (shape: event_dim_diag)
        - tau_non_diag ~ Gamma(2, 0.1) for off-diagonal elements (shape: event_dim_non_diag_eff)
        - f_diag and f_non_diag are sampled separately then combined using indexing
        - loc is added after assembly

        Examples
        --------
        >>> import numpyro
        >>> import jax.numpy as jnp
        >>> from jax import random
        >>>
        >>> # Full prior with ILR transformation
        >>> prior = IGMRF2D(
        ...     num_nodes=(10, 10),
        ...     order=(2, 2),
        ...     transform='alr',
        ...     prior_type='full'
        ... )
        >>> prior.set_age_bounds(0, 50)
        >>> # ILR requires A(A-1)/2 dimensions
        >>> prior.set_loc(jnp.zeros((45, 10, 10)))  # 45 = 10*9/2
        >>>
        >>> def model():
        ...     f = prior.sample_full()
        ...     numpyro.deterministic("tau_diag", numpyro.get("tau_diag"))
        ...     numpyro.deterministic("tau_non_diag", numpyro.get("tau_non_diag"))
        ...     return f
        >>>
        >>> key = random.PRNGKey(0)
        >>> with numpyro.handlers.seed(rng_seed=0):
        ...     sample = model()
        >>> print(sample.shape)  # (10, 10, 10)

        See Also
        --------
        sample_partial : Shared precision across all elements
        sample_global : Symmetric contact matrix
        """
        tau_diag = numpyro.sample(
            "tau_diag", dist.Gamma(2, 0.1), sample_shape=(self.event_dim_diag,)
        )
        tau_non_diag = numpyro.sample(
            "tau_non_diag",
            dist.Gamma(2, 0.1),
            sample_shape=(self.event_dim_non_diag_eff,),
        )

        f_diag = numpyro.sample(
            "f_diag",
            SymIGMRF2D(
                self.num_nodes[0],
                self.order[0],
                cond_prec=tau_diag,
            ),
        )[:, self.symm_tril_ix].reshape((self.event_dim_diag, self.A, self.A))
        f_non_diag = numpyro.sample(
            "f_non_diag",
            IGMRF2D_dist(
                self.num_nodes,
                self.order,
                cond_prec1=tau_non_diag,
            ),
        ).reshape((self.event_dim_non_diag_eff, self.A, self.A))

        # Assemble diagonal and off-diagonal blocks into full event grid
        f = self._assemble_full_prior_blocks(f_diag, f_non_diag)

        return self.loc + f

    def sample(self):
        """
        Sample from the IGMRF prior based on the configured prior_type.

        This is the main interface for sampling from the prior. It dispatches to
        the appropriate sampling method based on the prior_type attribute.

        Returns
        -------
        f : array
            Sampled contact matrix. Shape and structure depend on prior_type:
            - 'global': (A, A) symmetric matrix
            - 'partial': (A, A, A) possibly asymmetric with transformation
            - 'full': (A, A, A) with separate diagonal/off-diagonal priors

        Raises
        ------
        ValueError
            If prior_type is not one of {'global', 'partial', 'full'}

        Examples
        --------
        >>> import numpyro
        >>> from jax import random
        >>>
        >>> prior = IGMRF2D(num_nodes=(16, 16), order=(2, 2), prior_type='global')
        >>> prior.set_age_bounds(0, 80)
        >>>
        >>> def model():
        ...     f = prior.sample()  # Automatically dispatches to sample_global()
        ...     return f
        >>>
        >>> key = random.PRNGKey(0)
        >>> with numpyro.handlers.seed(rng_seed=0):
        ...     sample = model()

        See Also
        --------
        sample_global : Symmetric contact matrix
        sample_partial : Asymmetric with shared precision
        sample_full : Separate diagonal/off-diagonal priors
        """
        if self.prior_type == "global":
            return self.sample_global()
        elif self.prior_type == "partial":
            return self.sample_partial()
        elif self.prior_type == "full":
            return self.sample_full()
        else:
            raise ValueError(f"Unknown prior_type: {self.prior_type}")
