import jax
import numpyro
from jax.typing import ArrayLike
from numpyro import distributions as dist

from ..._utils import symm_from_tril_ix_col
from ._Prior2D import Prior2D


class Independent2D(Prior2D):
    """
    2D independent normal prior for contact matrix estimation.

    This prior assigns an independent Normal(0, scale) distribution to every
    cell of the contact matrix with no spatial pooling or smoothing. It serves
    as a non-informative baseline against which structured priors (IGMRF2D,
    Spline2D, HSGP2D) can be compared.

    Prior Types
    -----------
    - **global**: Symmetric contact matrix. Samples n(n+1)/2 independent normals
      from the lower triangle and mirrors them to produce a symmetric (A, A) matrix.

    - **partial**: Asymmetric contact matrix with a shared scale across all strata.
      Returns a tensor of shape (event_dim, A, A).

    - **full**: Separate independent normals for diagonal and off-diagonal strata.
      Assembled via the reciprocity constraint inherited from Prior2D.

    Parameters
    ----------
    prior_type : {'global', 'partial', 'full'}
        Structure of the prior.
    grid_type : {'age-age', 'diff-age'}, default='age-age'
        Grid structure for the contact matrix.
    scale : float, default=1.0
        Standard deviation of the independent Normal prior applied to each cell.
        Must be positive.
    loc : float or array-like, default=0.0
        Prior location (mean). Scalar is broadcast; array must match the shape
        expected by Prior2D.set_loc().

    Attributes
    ----------
    scale : float
        Standard deviation of each cell's Normal prior.
    A : int
        Number of age groups (set by set_age_bounds).
    symm_tril_ix : array
        Indices for mirroring the lower triangle to a full symmetric matrix
        (global prior only).

    Methods
    -------
    set_age_bounds(min_age, max_age)
        Configure age range for the contact matrix.
    sample()
        Sample from the prior based on prior_type.
    sample_global()
        Sample a symmetric (A, A) contact matrix.
    sample_partial()
        Sample an asymmetric (event_dim, A, A) contact matrix.
    sample_full()
        Sample with separate diagonal / off-diagonal priors.

    Examples
    --------
    >>> from cntmosaic.models.numpyro.priors import Independent2D
    >>> import numpyro
    >>>
    >>> prior = Independent2D(prior_type='global', scale=1.0)
    >>> prior.set_age_bounds(0, 15)
    >>>
    >>> with numpyro.handlers.seed(rng_seed=0):
    ...     f = prior.sample()   # shape (16, 16), symmetric

    See Also
    --------
    IGMRF2D : Intrinsic GMRF prior with spatial smoothing.
    HSGP2D : Hilbert-space Gaussian process prior.
    Prior2D : Abstract base class.
    """

    def __init__(
        self,
        prior_type: str,
        grid_type: str = "age-age",
        scale: float = 1.0,
        loc: ArrayLike = 0.0,
    ):
        super().__init__(grid_type, prior_type)
        self.scale = scale
        self.loc = loc

    def set_age_bounds(self, min_age: int, max_age: int) -> None:
        """
        Set the age range for the contact matrix and configure grid structure.

        Parameters
        ----------
        min_age : int
            Minimum age (inclusive).
        max_age : int
            Maximum age (inclusive).
        """
        self.min_age = min_age
        self.max_age = max_age
        self.A = max_age - min_age + 1
        self._set_grid()

    def _set_grid(self) -> None:
        self.symm_tril_ix = symm_from_tril_ix_col(self.A)

    def sample_global(self) -> jax.Array:
        """
        Sample a symmetric (A, A) contact matrix.

        Draws A*(A+1)/2 independent normals for the lower triangle, then
        mirrors them to produce a symmetric matrix.

        Returns
        -------
        f : array, shape (A, A)
            Symmetric contact matrix.
        """
        n_tril = self.A * (self.A + 1) // 2
        f_tril = numpyro.sample("f", dist.Normal(0, self.scale), sample_shape=(n_tril,))
        return f_tril[self.symm_tril_ix].reshape((self.A, self.A))

    def sample_partial(self) -> jax.Array:
        """
        Sample an asymmetric (event_dim, A, A) contact matrix.

        Every cell is drawn independently from Normal(0, scale).

        Returns
        -------
        f : array, shape (event_dim, A, A)
        """
        f = numpyro.sample(
            "f",
            dist.Normal(0, self.scale),
            sample_shape=(self.event_dim, self.A, self.A),
        )
        return self.loc + f

    def sample_full(self) -> jax.Array:
        """
        Sample with separate independent normals for diagonal and off-diagonal strata.

        Returns
        -------
        f : array, shape (event_dim, A, A)
        """
        f_diag = numpyro.sample(
            "f_diag",
            dist.Normal(0, self.scale),
            sample_shape=(self.event_dim_diag, self.A, self.A),
        )
        f_non_diag = numpyro.sample(
            "f_non_diag",
            dist.Normal(0, self.scale),
            sample_shape=(self.event_dim_non_diag_eff, self.A, self.A),
        )
        return self.loc + self._assemble_full_prior_blocks(f_diag, f_non_diag)

    def sample(self) -> jax.Array:
        """
        Sample from the prior based on prior_type.

        Returns
        -------
        f : array
            - 'global': shape (A, A), symmetric
            - 'partial': shape (event_dim, A, A)
            - 'full': shape (event_dim, A, A)

        Raises
        ------
        ValueError
            If prior_type is not one of {'global', 'partial', 'full'}.
        """
        if self.prior_type == "global":
            return self.sample_global()
        elif self.prior_type == "partial":
            return self.sample_partial()
        elif self.prior_type == "full":
            return self.sample_full()
        else:
            raise ValueError(f"Unknown prior_type: {self.prior_type}")
