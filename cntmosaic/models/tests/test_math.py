import jax
import jax.numpy as jnp
import numpy as np
import pytest

from cntmosaic.models._math import subdiag_permutation_matrix


@pytest.mark.parametrize("K", [2, 3, 4, 20])
def test_subdiag_permutation_matrix(K):
    P = subdiag_permutation_matrix(K)

    # Test that P is a permutation matrix
    assert jnp.all(jnp.sum(P, axis=0) == 1)
    assert jnp.all(jnp.sum(P, axis=1) == 1)

    # Test that P^2 = I
    I = jnp.eye(K * K)
    P2 = jnp.matmul(P, P)
    assert jnp.allclose(P2, I)

    # Test different cases of K
    x = jnp.arange(K * K).reshape((K, K))
    xt = x.T
    expected = xt.flatten()

    result = jnp.array(
        [jnp.argwhere(P[:, j] == 1) for j in range(P.shape[1])]
    ).flatten()

    assert jnp.array_equal(result, expected)
