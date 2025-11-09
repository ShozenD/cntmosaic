import math

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array
from jax.typing import ArrayLike


def is_square(x: int) -> bool:
    """Check if a number is a perfect square"""
    return x == math.isqrt(x) ** 2


def closure(x: ArrayLike, axis: int = 0) -> Array:
    """Closure operation for a given axis"""
    return x / jnp.sum(x, axis=axis, keepdims=True)


def basis_contrast_matrix(d: int) -> Array:
    """Basis contrast matrix associated with an orthonormal basis of S^D

    See eq. (18) of J.J. Egozcue, V. Pawlowsky-Glahn, G. Mateu-Figueras, and C. Barcelo-Vidal.
    Isometric logratio transformations for compositional data analysis. Mathematical Geology, 35(3):279–300, 2003.

    and eq. (3) of Ana M. Bianco et al. Robust Nonparametric Regression for Compositional Data: the Simplicial-Real case. 2024.
    """
    U = jnp.zeros((d, d - 1), dtype=jnp.float32)
    for i in range(d - 1):
        j = i + 1
        # Pre-compute the normalization factor for numerical stability
        norm_factor = jnp.sqrt(j / (j + 1.0))

        # Use JAX indexing for in-place updates
        U = U.at[:j, i].set(norm_factor / j)
        U = U.at[j, i].set(-norm_factor)

    return U


def alr(x: ArrayLike, axis: int = 0) -> Array:
    """Additive log ratio transformation

    Takes log-ratios of all components except the last (reference) with respect
    to the last component, reducing the dimension by 1 along the specified axis.

    Parameters
    ----------
    x : ArrayLike
        Compositional data (should sum to 1 along specified axis)
    axis : int, default=0
        Axis along which to apply the transformation

    Returns
    -------
    Array
        ALR-transformed data with dimension reduced by 1 along axis

    Examples
    --------
    >>> x = jnp.array([[0.3], [0.7]])  # 2-part composition
    >>> alr(x, axis=0)
    Array([[-0.8472978]], dtype=float32)  # log(0.3/0.7), shape (1,1)
    """
    # Take all components except the last as numerator
    numerator = jnp.take(x, indices=jnp.arange(x.shape[axis] - 1), axis=axis)
    # Last component as denominator (reference)
    denominator = jnp.take(x, indices=jnp.array([-1]), axis=axis)
    return jnp.log(numerator / denominator)


def clr(x: ArrayLike, axis: int = 0) -> Array:
    """Centered log ratio transformation"""
    y = jnp.log(x + jnp.finfo(x.dtype).eps)
    return y - jnp.mean(y, axis=axis, keepdims=True)


def ilr(x: ArrayLike, axis: int = 0) -> Array:
    """Isometric log ratio transformation"""
    shape = list(x.shape)
    U = basis_contrast_matrix(shape[axis])
    y = clr(x)
    return jnp.apply_along_axis(lambda z: jnp.matmul(U.T, z), axis=axis, arr=y)


def inverse_alr(x: ArrayLike, axis: int = 0) -> Array:
    """Inverse additive log ratio transformation"""
    shape = list(x.shape)
    ones_shape = shape.copy()
    ones_shape[axis] = 1
    ones = jnp.ones(ones_shape)

    x = jnp.concatenate((jnp.exp(x), ones), axis=axis)
    return closure(x, axis=axis)


def log_inverse_alr(x: ArrayLike, axis: int = 0) -> Array:
    """Log inverse additive log ratio transformation"""
    return jnp.log(inverse_alr(x, axis=axis))


def inverse_clr(x: ArrayLike, axis: int = 0) -> Array:
    """Inverse centered log ratio transformation"""
    return jax.nn.softmax(x, axis=axis)


def inverse_ilr(x: ArrayLike, axis: int = 0) -> Array:
    """Inverse isometric log ratio transformation"""
    shape = list(x.shape)
    U = basis_contrast_matrix(shape[axis] + 1)
    Uz = jnp.apply_along_axis(lambda z: jnp.matmul(U, z), axis=axis, arr=x)

    return jax.nn.softmax(Uz, axis=axis)


def log_inverse_ilr(x: ArrayLike, axis: int = 0) -> Array:
    """Log inverse isometric log ratio transformation"""
    return jnp.log(inverse_ilr(x, axis=axis))


def subdiag_permutation_matrix(K: int) -> Array:
    """
    Permutation matrix to swap (i, j) with (j, i) for non-diagonal blocks in K x K block matrix.
    This is essentially a transposition operator for K x K matrices vectorized in column-major order.

    Used to enforce symmetry in 2D matrix-valued priors.
    """
    P = np.zeros((K * K, K * K), dtype=int)

    for k in range(K):
        for l in range(K):
            t = k * K + l
            s = l * K + k

            if t == s:
                P[t, s] = 1
            else:
                P[t, s] = 1

    return jnp.array(P)
