from typing import Optional

import numpy as np

import jax
from jax import numpy as jnp
from jax import lax, random, Array
from jax.typing import ArrayLike
from jax.scipy.special import factorial

import numpyro
from numpyro.distributions import constraints
from numpyro.distributions.distribution import Distribution
from numpyro.util import is_prng_key
from numpyro.distributions.util import validate_sample, promote_shapes


def diff_matrix(num_nodes: int, order: int) -> Array:
    """
    Construct a finite difference matrix of given order.

    Parameters
    ----------
    num_nodes: int
        The number of nodes in the grid.
    order: int
        The order of the finite difference.

    Returns
    -------
    D: NDArray
        The finite difference matrix of shape (num_nodes - order, num_nodes).
    """
    D = jnp.zeros((num_nodes - order, num_nodes))
    i_vals = jnp.arange(order + 1)
    coeff = (factorial(order) / (factorial(i_vals) * factorial(order - i_vals))) * (
        -1
    ) ** (order - i_vals)
    for i in range(num_nodes - order):
        D = D.at[i, i : i + order + 1].set(coeff)

    return D


def laplacian(num_nodes: int, order: int) -> Array:
    """
    Construct a Laplacian matrix using finite differences.

    Parameters
    ----------
    num_nodes: int
      The number of nodes in the grid.
    order: int
      The order of the finite difference.

    Returns
    -------
    L: NDArray
      The Laplacian matrix of shape (num_nodes - order, num_nodes - order).
    """
    D = diff_matrix(num_nodes, order)

    return D.T @ D


class IGMRF(Distribution):
    arg_constraints = {
        "num_nodes": constraints.positive_integer,
        "order": constraints.positive_integer,
    }
    support = constraints.real_vector
    pytree_data_fields = ("loc", "lam_sub", "cond_prec", "U", "U_sub", "L")
    pytree_aux_fields = ("num_nodes", "order")

    def __init__(
        self,
        num_nodes: int,
        order: int,
        loc: ArrayLike = 0.0,
        cond_prec: Array = 1.0,
        *,
        validate_args: Optional[bool] = None,
    ):
        """
        1-dimensional Intrinsic Gaussian Markov Random Field (IGMRF) distribution.

        Parameters
        ----------
        num_nodes: int
            The number of nodes in the grid.
        order: int
            The order of the finite difference.
        loc: ArrayLike, optional
            The location parameter (mean) of the distribution. Default is 0.0.
            Expected shape is either () or (batch_shape, num_nodes).
        cond_prec: ArrayLike, optional
            The conditional precision parameter of the distribution. Default is 1.0.
            Expected shape is either () or (batch_shape, num_nodes).
        validate_args: bool, optional
            Whether to validate the arguments. Default is None.
        """

        self.num_nodes = num_nodes
        self.order = order
        self.L = laplacian(num_nodes, order)
        lam, U = jnp.linalg.eigh(self.L)

        # Convert to jax arrays
        loc = jnp.asarray(loc)
        cond_prec = jnp.asarray(cond_prec)

        # Determine batch shape from inputs
        if jnp.ndim(loc) == 0:  # Scalar loc: no batch dimension
            loc_batch_shape = ()
        else:  # loc has shape (batch_shape, num_nodes)
            loc_batch_shape = jnp.shape(loc)[:-1]

        if jnp.ndim(cond_prec) == 0:  # Scalar cond_prec: no batch dimension
            cond_prec_batch_shape = ()
        else:  # cond_prec has shape (batch_shape,)
            cond_prec_batch_shape = jnp.shape(cond_prec)

        batch_shape = lax.broadcast_shapes(loc_batch_shape, cond_prec_batch_shape)

        # Broadcast loc to (batch_shape, num_nodes)
        if jnp.ndim(loc) == 0:
            self.loc = jnp.broadcast_to(loc, batch_shape + (num_nodes,))
        else:
            self.loc = jnp.broadcast_to(loc, batch_shape + (num_nodes,))

        # Broadcast cond_prec to (batch_shape,)
        self.cond_prec = jnp.broadcast_to(cond_prec, batch_shape)

        # Store eigendecomposition components
        self.lam_sub = lam[order:]  # shape (num_nodes - order,)
        U_sub = U[:, order:]  # shape (num_nodes, num_nodes - order)

        # Broadcast U_sub if there's a batch dimension
        if batch_shape:
            self.U_sub = jnp.broadcast_to(
                U_sub, batch_shape + (num_nodes, num_nodes - order)
            )
        else:
            self.U_sub = U_sub

        event_shape = (num_nodes,)

        super(IGMRF, self).__init__(
            batch_shape=batch_shape,
            event_shape=event_shape,
            validate_args=validate_args,
        )

    def sample(
        self, key: jax.dtypes.prng_key, sample_shape: tuple[int, ...] = ()
    ) -> ArrayLike:
        assert is_prng_key(key)
        # Sample from the reduced space corresponding to non-zero eigenvalues
        eps_shape = sample_shape + self.batch_shape + (self.num_nodes - self.order,)
        eps = random.normal(key, shape=eps_shape)[..., jnp.newaxis]

        # Add two trailing axes to cond_prec for broadcasting
        cond_prec_reshaped = self.cond_prec[..., jnp.newaxis, jnp.newaxis]

        scale = 1 / self.lam_sub[..., jnp.newaxis]
        scale = scale / cond_prec_reshaped
        scale = jnp.sqrt(scale)

        result = self.loc + jnp.squeeze(jnp.matmul(self.U_sub, scale * eps), axis=-1)

        return result

    @validate_sample
    def log_prob(self, value: ArrayLike) -> ArrayLike:
        n = self.num_nodes
        diff = value - self.loc
        quad = 0.5 * self.cond_prec * jnp.sum((diff @ self.L) * diff, axis=-1)
        result = jnp.squeeze(
            0.5 * (n * jnp.log(self.cond_prec) + jnp.sum(jnp.log(self.lam_sub[0])))
            - quad
        )

        return result

    @staticmethod
    def infer_shapes(
        num_nodes,
        order,
        loc=(),
        cond_prec=(),
    ):
        # IGMRF's event_shape depends on the VALUE of num_nodes, not its shape.
        # Since infer_shapes only receives parameter shapes (not values),
        # we cannot determine the event_shape from shape information alone.
        raise NotImplementedError(
            "IGMRF.infer_shapes() cannot be implemented because event_shape "
            "depends on the value of num_nodes parameter, not its shape."
        )
