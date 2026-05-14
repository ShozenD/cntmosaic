from typing import Optional

import jax
import jax.numpy as jnp
from jax import lax, Array
from jax.typing import ArrayLike

from numpyro.distributions import constraints
from numpyro.distributions.distribution import Distribution
from numpyro.util import is_prng_key
from numpyro.distributions.util import validate_sample

from ._IGMRF import laplacian


class IGMRF2D(Distribution):
    """
    2-dimensional Intrinsic Gaussian Markov Random Field (IGMRF) distribution.

    This distribution represents a Gaussian Markov Random Field over a 2D lattice with
    separable precision structure. The precision matrix Q has the form:

        Q = cond_prec1 ⊗ L₁ ⊗ I₂ + cond_prec2 ⊗ I₁ ⊗ L₂

    where L₁ and L₂ are Laplacian matrices constructed from finite difference operators
    of specified orders, I₁ and I₂ are identity matrices, and ⊗ denotes the Kronecker product.

    The distribution is intrinsic (improper) because the Laplacian matrices have zero
    eigenvalues corresponding to polynomial trends of degree less than the order.
    For sampling and likelihood computation, the distribution is projected onto the
    subspace orthogonal to these polynomial trends using eigendecomposition.

    Mathematical Background
    -----------------------
    An IGMRF models spatial correlation through finite differences. For a 2D grid,
    the precision matrix penalizes roughness separately in each dimension:

    - order = (1, 1): penalizes first differences (random walk prior)
    - order = (2, 2): penalizes second differences (smooth prior)

    The Kronecker product structure enables efficient eigendecomposition:
    eigenvalues of Q are λ₁[i] * cond_prec1 + λ₂[j] * cond_prec2 for all (i,j) pairs.

    References
    ----------
    - Rue, H., & Held, L. (2005). Gaussian Markov Random Fields: Theory and Applications.
      Chapman & Hall/CRC.

    Examples
    --------
    >>> from cntmosaic.models.numpyro.distributions import IGMRF2D
    >>> import jax.numpy as jnp
    >>> from jax import random
    >>>
    >>> # Create a 2D IGMRF with first-order differences
    >>> dist = IGMRF2D(num_nodes=(10, 10), order=(1, 1), cond_prec1=1.0, cond_prec2=1.0)
    >>>
    >>> # Sample from the distribution
    >>> key = random.PRNGKey(0)
    >>> samples = dist.sample(key, sample_shape=(5,))  # 5 samples of shape (100,)
    >>>
    >>> # Batched precision parameters
    >>> dist_batch = IGMRF2D(
    ...     num_nodes=(5, 5),
    ...     order=(2, 2),
    ...     cond_prec1=jnp.array([1.0, 2.0, 3.0]),
    ...     cond_prec2=jnp.array([0.5, 1.0, 1.5])
    ... )  # batch_shape=(3,), event_shape=(25,)

    See Also
    --------
    IGMRF : 1-dimensional version
    SymIGMRF2D : Symmetric 2D version with single precision parameter
    """

    support = constraints.real_vector
    reparametrized_params = ["loc", "cond_prec1", "cond_prec2"]
    pytree_data_fields = (
        "loc",
        "cond_prec1",
        "cond_prec2",
        "lam1_sub",
        "lam2_sub",
        "U_sub",
        "L1_kron_I2",
        "I1_kron_L2",
    )
    pytree_aux_fields = ("num_nodes", "order")

    def __init__(
        self,
        num_nodes: tuple[int, int],
        order: tuple[int, int],
        loc: ArrayLike = 0.0,
        cond_prec1: Array = 1.0,
        cond_prec2: Array = 1.0,
        *,
        validate_args: Optional[bool] = None,
    ):
        """
        Initialize a 2D Intrinsic Gaussian Markov Random Field distribution.

        Parameters
        ----------
        num_nodes : tuple[int, int]
            Number of nodes in the grid for each dimension (n₁, n₂).
            Must be greater than the corresponding order values.
        order : tuple[int, int]
            Order of finite difference approximation for each dimension.
            - order[0]: order for dimension 1 (rows)
            - order[1]: order for dimension 2 (columns)
            Must satisfy: 0 < order[i] < num_nodes[i] for i ∈ {0, 1}.
        loc : ArrayLike, optional
            Location parameter (mean) of the distribution. Default is 0.0.
            Shape options:
            - Scalar (): broadcasted to (batch_shape, n₁ * n₂)
            - (n₁ * n₂,): no batch dimension
            - (batch_shape, n₁ * n₂): batched means
            The flattened grid uses row-major (C) order.
        cond_prec1 : ArrayLike, optional
            Conditional precision parameter for the first dimension. Default is 1.0.
            Controls smoothness in the row direction.
            Shape options:
            - Scalar (): no batch dimension
            - (batch_shape,): batched precision parameters
            Must be positive.
        cond_prec2 : ArrayLike, optional
            Conditional precision parameter for the second dimension. Default is 1.0.
            Controls smoothness in the column direction.
            Shape options:
            - Scalar (): no batch dimension
            - (batch_shape,): batched precision parameters
            Must be positive.
        validate_args : bool, optional
            Whether to validate input arguments. Default is None.
            When True, checks parameter constraints (e.g., positive precision).

        Attributes
        ----------
        batch_shape : tuple[int, ...]
            Shape of parameter batches, inferred from loc, cond_prec1, cond_prec2.
        event_shape : tuple[int, ...]
            Shape of a single sample, always (n₁ * n₂,).
        L1 : Array
            Laplacian matrix for dimension 1, shape (n₁, n₁).
        L2 : Array
            Laplacian matrix for dimension 2, shape (n₂, n₂).
        lam1_sub : Array
            Non-zero eigenvalues of L1, shape (n₁ - order[0],).
        lam2_sub : Array
            Non-zero eigenvalues of L2, shape (n₂ - order[1],).
        U_sub : Array
            Eigenvectors for non-zero eigenspace, shape (..., n₁*n₂, (n₁-order[0])*(n₂-order[1])).

        Raises
        ------
        ValueError
            If order[i] >= num_nodes[i] for any i.
            If cond_prec1 or cond_prec2 are non-positive (when validate_args=True).
        """

        self.num_nodes = num_nodes
        self.order = order

        # Create Laplacian matrices
        L1 = laplacian(num_nodes[0], order[0])
        L2 = laplacian(num_nodes[1], order[1])

        # Precompute Kronecker products for log_prob efficiency
        n1, n2 = num_nodes
        self.L1_kron_I2 = jnp.kron(L1, jnp.eye(n2))
        self.I1_kron_L2 = jnp.kron(jnp.eye(n1), L2)

        # Eigendecomposition for sampling
        lam1, U1 = jnp.linalg.eigh(L1)
        lam2, U2 = jnp.linalg.eigh(L2)

        U1_sub, U2_sub = U1[:, order[0] :], U2[:, order[1] :]
        self.lam1_sub, self.lam2_sub = lam1[order[0] :], lam2[order[1] :]
        self.U_sub = jnp.kron(U1_sub, U2_sub)  # shape (n1*n2, (n1-order1)*(n2-order2))

        # ===== Determine batch shape from inputs =====
        if jnp.ndim(loc) == 0:  # Scalar loc: no batch dimension
            loc_batch_shape = ()
        elif jnp.ndim(loc) == 1:  # loc has shape (n1*n2,)
            loc_batch_shape = ()
        else:  # loc has shape (batch_shape, n1*n2)
            loc_batch_shape = jnp.shape(loc)[:-1]

        if jnp.ndim(cond_prec1) == 0:  # Scalar cond_prec1: no batch dimension
            cond_prec1_batch_shape = ()
        else:  # cond_prec1 has shape (batch_shape,)
            cond_prec1_batch_shape = jnp.shape(cond_prec1)

        if jnp.ndim(cond_prec2) == 0:  # Scalar cond_prec2: no batch dimension
            cond_prec2_batch_shape = ()
        else:  # cond_prec2 has shape (batch_shape,)
            cond_prec2_batch_shape = jnp.shape(cond_prec2)

        batch_shape = lax.broadcast_shapes(
            loc_batch_shape, cond_prec1_batch_shape, cond_prec2_batch_shape
        )

        # ===== Broadcast adjustments =====
        if jnp.ndim(loc) == 0:
            self.loc = jnp.broadcast_to(
                loc, batch_shape + (num_nodes[0] * num_nodes[1],)
            )
        else:
            self.loc = jnp.broadcast_to(
                loc, batch_shape + (num_nodes[0] * num_nodes[1],)
            )

        # Broadcast conditional precisions to (batch_shape,)
        self.cond_prec1 = jnp.broadcast_to(cond_prec1, batch_shape)
        self.cond_prec2 = jnp.broadcast_to(cond_prec2, batch_shape)

        # Broadcast U_sub if there's a batch dimension
        if batch_shape:
            self.U_sub = jnp.broadcast_to(
                self.U_sub,
                batch_shape
                + (
                    num_nodes[0] * num_nodes[1],
                    (num_nodes[0] - order[0]) * (num_nodes[1] - order[1]),
                ),
            )
        else:
            self.U_sub = self.U_sub

        event_shape = (num_nodes[0] * num_nodes[1],)

        super(IGMRF2D, self).__init__(
            batch_shape=batch_shape,
            event_shape=event_shape,
            validate_args=validate_args,
        )

    def sample(self, key: jax.dtypes.prng_key, sample_shape: tuple[int, ...] = ()):
        """
        Sample from the IGMRF2D distribution.

        Sampling is performed by:
        1. Generating standard normal samples in the reduced eigenspace
        2. Scaling by the inverse square root of eigenvalues
        3. Transforming back to the original space via eigenvectors
        4. Adding the location parameter

        This avoids explicit matrix inversion and leverages the eigendecomposition.

        Parameters
        ----------
        key : jax.dtypes.prng_key
            PRNG key for random number generation.
        sample_shape : tuple[int, ...], optional
            Shape of the sample batch to generate. Default is ().
            Final sample shape will be sample_shape + batch_shape + event_shape.

        Returns
        -------
        Array
            Samples from the distribution with shape:
            sample_shape + batch_shape + (num_nodes[0] * num_nodes[1],)
        """
        assert is_prng_key(key)
        sub_event_shape = (
            (self.num_nodes[0] - self.order[0]) * (self.num_nodes[1] - self.order[1]),
        )
        eps_shape = sample_shape + self.batch_shape + sub_event_shape
        eps = jax.random.normal(key, shape=eps_shape)[..., jnp.newaxis]

        # Reshape cond_prec for proper broadcasting
        cond_prec1_reshaped = self.cond_prec1[..., jnp.newaxis, jnp.newaxis]
        cond_prec2_reshaped = self.cond_prec2[..., jnp.newaxis, jnp.newaxis]

        # Efficient computation: eigenvalues are lam1[i]*prec1 + lam2[j]*prec2
        lam_sub = (
            cond_prec1_reshaped * self.lam1_sub[:, jnp.newaxis]
            + cond_prec2_reshaped * self.lam2_sub[jnp.newaxis, :]
        )
        lam_sub = lam_sub.reshape(self.batch_shape + (-1,))
        scale = jnp.sqrt(1 / lam_sub)[..., jnp.newaxis]
        result = self.loc + jnp.squeeze(jnp.matmul(self.U_sub, scale * eps), axis=-1)

        return result

    @validate_sample
    def log_prob(self, value: ArrayLike) -> ArrayLike:
        """
        Compute the log probability density of the IGMRF2D distribution.

        The log probability is computed using the reduced-rank formulation that
        excludes zero eigenvalues. For a 2D IGMRF with precision Q = τ₁ L₁ ⊗ I₂ + τ₂ I₁ ⊗ L₂,
        the log density is:

        log p(x) ∝ -1/2 * [log|Q₊| + (x - μ)ᵀ Q (x - μ)]

        where Q₊ denotes the pseudo-inverse (or equivalently, the determinant computed
        only over non-zero eigenvalues).

        Parameters
        ----------
        value : ArrayLike
            Sample value at which to evaluate log probability.
            Shape must be batch_shape + event_shape.

        Returns
        -------
        Array
            Log probability density, shape is batch_shape.
        """
        n1, n2 = self.num_nodes
        diff = value - self.loc

        # Compute log determinant using non-zero eigenvalues only
        cond_prec1_reshaped = self.cond_prec1[..., jnp.newaxis, jnp.newaxis]
        cond_prec2_reshaped = self.cond_prec2[..., jnp.newaxis, jnp.newaxis]

        lam_sub = (
            cond_prec1_reshaped * self.lam1_sub[:, jnp.newaxis]
            + cond_prec2_reshaped * self.lam2_sub[jnp.newaxis, :]
        )
        lam_sub = lam_sub.reshape(self.batch_shape + (-1,))
        log_det = jnp.sum(jnp.log(lam_sub), axis=-1)

        # Compute quadratic form: xᵀ Q x using precomputed Kronecker products
        quad1 = self.cond_prec1 * jnp.sum((diff @ self.L1_kron_I2) * diff, axis=-1)
        quad2 = self.cond_prec2 * jnp.sum((diff @ self.I1_kron_L2) * diff, axis=-1)
        quad = quad1 + quad2

        # Number of non-zero eigenvalues
        reduced_dim = (n1 - self.order[0]) * (n2 - self.order[1])

        return jnp.squeeze(-0.5 * (reduced_dim * jnp.log(2 * jnp.pi) - log_det + quad))

    @staticmethod
    def infer_shapes(
        num_nodes=(),
        order=(),
        loc=(),
        cond_prec1=(),
        cond_prec2=(),
    ):
        """
        Infer batch and event shapes from parameter shapes.

        Note: This implementation cannot determine event_shape because it depends
        on the VALUES of num_nodes (not its shape). We raise NotImplementedError
        to be consistent with the 1D IGMRF implementation.

        Parameters
        ----------
        num_nodes : tuple, optional
            Shape of num_nodes parameter (not the value).
        order : tuple, optional
            Shape of order parameter (not the value).
        loc : tuple, optional
            Shape of loc parameter.
        cond_prec1 : tuple, optional
            Shape of cond_prec1 parameter.
        cond_prec2 : tuple, optional
            Shape of cond_prec2 parameter.

        Raises
        ------
        NotImplementedError
            Always raised because event_shape depends on parameter values.
        """
        raise NotImplementedError(
            "IGMRF2D.infer_shapes() cannot be implemented because event_shape "
            "depends on the value of num_nodes parameter, not its shape."
        )
