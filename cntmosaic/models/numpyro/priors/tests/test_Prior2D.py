"""
Unit tests for Prior2D abstract base class.

Tests cover:
- Parameter validation
- Attribute initialization
- Event dimension calculation
- Location parameter setting
- Error handling
- Backward compatibility
"""

from collections import namedtuple

import jax.numpy as jnp
import numpy as np
import numpyro
import pytest
from numpy.testing import assert_allclose

from .._HSGP2D import HSGP2D
from .._IGMRF2D import IGMRF2D
from .._Prior2D import Prior2D
from .._PSpline2D import PSpline2D
from .._Spline2D import Spline2D


class T(namedtuple("TestCase", ["prior", "params"])):
    """
    A test case namedtuple for storing prior distributions and their parameters.

    This class extends namedtuple to create test cases that pair a prior distribution
    with its corresponding parameters. It overrides __new__ to allow passing multiple
    parameters as separate arguments instead of requiring a tuple.

    Attributes:
        prior: The prior distribution object to be tested
        params (tuple): A tuple containing the parameters for the prior distribution

    Args:
        prior: The prior distribution instance
        *params: Variable number of parameters to be stored as a tuple

    Returns:
        T: A new instance of the test case with the prior and parameters

    Example:
        >>> test_case = T(some_prior, param1, param2, param3)
        >>> test_case.prior  # Access the prior
        >>> test_case.params  # Access the parameters tuple
    """

    def __new__(cls, prior, *params):
        return super(T, cls).__new__(cls, prior, params)


CASES = [
    # Global cases
    T(IGMRF2D, "global"),
    T(Spline2D, "global"),
    T(PSpline2D, "global"),
    T(HSGP2D, "global"),
    # Partial cases
    T(IGMRF2D, "partial"),
    T(Spline2D, "partial"),
    T(PSpline2D, "partial"),
    T(HSGP2D, "partial"),
    # Full cases
    T(IGMRF2D, "full"),
    T(Spline2D, "full"),
    T(PSpline2D, "full"),
    T(HSGP2D, "full"),
]

# ==========================================================================
# Prior2D base class tests
# ==========================================================================


# Concrete implementation for testing
class ConcretePrior2D(Prior2D):
    """Minimal concrete implementation of Prior2D for testing."""

    def set_age_bounds(self, min_age: int, max_age: int):
        self.min_age = min_age
        self.max_age = max_age
        self.A = max_age - min_age + 1
        self._set_grid()

    def _set_grid(self):
        """Minimal grid setup."""
        pass

    def sample(self):
        """Minimal sample implementation."""
        return jnp.zeros((self.A, self.A))


class TestInitialization:
    """Test Prior2D initialization and validation."""

    def test_default_initialization(self):
        """Test initialization with default parameters."""
        prior = ConcretePrior2D()
        assert prior.grid_type == "age-age"
        assert prior.type == "global"
        assert prior.prior_type == "global"
        assert prior.A is None
        assert prior.event_dim is None
        assert prior.loc is None

    def test_custom_initialization(self):
        """Test initialization with custom parameters."""
        prior = ConcretePrior2D(grid_type="diff-age", prior_type="partial")
        assert prior.grid_type == "diff-age"
        assert prior.type == "partial"
        assert prior.prior_type == "partial"

    def test_backward_compatibility_type_attribute(self):
        """Test that both 'type' and 'prior_type' attributes exist."""
        prior = ConcretePrior2D(prior_type="full")
        assert hasattr(prior, "type")
        assert hasattr(prior, "prior_type")
        assert prior.type == prior.prior_type == "full"

    def test_invalid_grid_type(self):
        """Test that invalid grid_type raises ValueError."""
        with pytest.raises(ValueError, match="grid_type must be one of"):
            ConcretePrior2D(grid_type="invalid")

    def test_invalid_prior_type(self):
        """Test that invalid prior_type raises ValueError."""
        with pytest.raises(ValueError, match="prior_type must be one of"):
            ConcretePrior2D(prior_type="invalid")


class TestValidateParams:
    """Test parameter validation method."""

    def test_all_valid_grid_types(self):
        """Test all allowed grid types are accepted."""
        for grid_type in Prior2D.ALLOWED_GRID_TYPES:
            prior = ConcretePrior2D(grid_type=grid_type)
            assert prior.grid_type == grid_type

    def test_all_valid_prior_types(self):
        """Test all allowed prior types are accepted."""
        for prior_type in Prior2D.ALLOWED_TYPES:
            prior = ConcretePrior2D(prior_type=prior_type)
            assert prior.type == prior_type


class TestSetAgeBounds:
    """Test age bounds configuration."""

    def test_basic_age_bounds(self):
        """Test setting age bounds."""
        prior = ConcretePrior2D()
        prior.set_age_bounds(0, 10)
        assert prior.min_age == 0
        assert prior.max_age == 10
        assert prior.A == 11

    def test_different_age_ranges(self):
        """Test various age ranges."""
        test_cases = [
            (0, 4, 5),
            (0, 84, 85),
            (5, 15, 11),
            (20, 60, 41),
        ]
        for min_age, max_age, expected_A in test_cases:
            prior = ConcretePrior2D()
            prior.set_age_bounds(min_age, max_age)
            assert prior.A == expected_A


class TestSetEventDim:
    """Test event dimension calculation."""

    def test_global_prior(self):
        """Test event dimension for global prior. Global priors only accept K=1."""
        prior = ConcretePrior2D(prior_type="global")
        prior.set_event_dim(1)  # K=1 for global priors
        assert prior.event_dim == 1

    def test_partial_prior(self):
        """Test partial prior: event_dim = K."""
        prior = ConcretePrior2D(prior_type="partial")
        prior.set_event_dim(10)  # K=10 categories -> event_dim = 10
        assert prior.event_dim == 10

    def test_full_prior(self):
        """Test full prior: event_dim = K * K."""
        prior = ConcretePrior2D(prior_type="full")
        prior.set_event_dim(5)  # K=5 categories -> event_dim = 5*5 = 25
        assert prior.event_dim == 25
        assert prior.event_dim_diag == 5
        assert prior.event_dim_non_diag == 20
        assert prior.event_dim_non_diag_eff == 10  # Halved due to reciprocity

    def test_full_prior_non_square_event_dim(self):
        """Test that K values leading to non-perfect-square event_dim work correctly.
        Since event_dim = K * K, it's always a perfect square by construction."""
        prior = ConcretePrior2D(prior_type="full")
        # K=7 -> event_dim = 49, which is a perfect square
        prior.set_event_dim(7)
        assert prior.event_dim == 49
        assert prior.event_dim_diag == 7

    def test_invalid_event_dim_type(self):
        """Test that non-integer K raises error."""
        prior = ConcretePrior2D()
        with pytest.raises(AssertionError):
            prior.set_event_dim(10.5)

    def test_negative_event_dim(self):
        """Test that negative K raises error."""
        prior = ConcretePrior2D()
        with pytest.raises(AssertionError):
            prior.set_event_dim(-5)

    def test_zero_event_dim(self):
        """Test that zero K raises error."""
        prior = ConcretePrior2D()
        with pytest.raises(AssertionError):
            prior.set_event_dim(0)


class TestSetLoc:
    """Test location parameter setting."""

    def test_scalar_loc(self):
        """Test setting scalar location parameter."""
        prior = ConcretePrior2D(prior_type="partial")
        prior.set_age_bounds(0, 4)  # A=5
        prior.set_event_dim(25)  # K=25 -> event_dim=25
        prior.set_loc(0.5)
        assert prior.loc.shape == (25, 5, 5)
        assert jnp.allclose(prior.loc, 0.5)

    def test_full_array_loc(self):
        """Test setting full array location parameter."""
        prior = ConcretePrior2D(prior_type="partial")
        prior.set_age_bounds(0, 3)  # A=4
        prior.set_event_dim(10)  # K=10 -> event_dim=10

        loc_array = np.random.randn(10, 4, 4)
        prior.set_loc(loc_array)
        assert prior.loc.shape == (10, 4, 4)
        assert jnp.allclose(prior.loc, loc_array)

    def test_broadcasted_loc(self):
        """Test setting location with broadcasting."""
        prior = ConcretePrior2D(prior_type="partial")
        prior.set_age_bounds(0, 2)  # A=3
        prior.set_event_dim(6)  # K=6 -> event_dim=6

        loc_broadcast = np.ones((6, 3))
        prior.set_loc(loc_broadcast)
        assert prior.loc.shape == (6, 3, 3)
        # Check that each row was broadcasted correctly
        for i in range(6):
            for j in range(3):
                assert jnp.allclose(prior.loc[i, j, :], 1.0)

    def test_loc_with_jax_array(self):
        """Test that JAX arrays are handled correctly."""
        prior = ConcretePrior2D()
        prior.set_age_bounds(0, 2)
        prior.set_event_dim(1)  # K=1 for global prior

        loc_jax = jnp.array(0.3)
        prior.set_loc(loc_jax)
        assert isinstance(prior.loc, jnp.ndarray)

    def test_loc_before_age_bounds_raises_error(self):
        """Test that setting loc before age bounds raises error."""
        prior = ConcretePrior2D()
        with pytest.raises(ValueError, match="Age bounds must be set"):
            prior.set_loc(0.5)

    def test_loc_before_event_dim_raises_error(self):
        """Test that setting loc before event_dim raises error."""
        prior = ConcretePrior2D()
        prior.set_age_bounds(0, 4)
        with pytest.raises(ValueError):  # event_dim doesn't exist yet
            prior.set_loc(0.5)

    def test_invalid_loc_shape_raises_error(self):
        """Test that invalid loc shape raises informative error."""
        prior = ConcretePrior2D(prior_type="partial")
        prior.set_age_bounds(0, 4)  # A=5
        prior.set_event_dim(10)  # K=10 -> event_dim=10, event_dim_latent=10

        with pytest.raises(ValueError, match="Invalid loc shape"):
            prior.set_loc(np.ones((5, 5)))  # Wrong shape

    def test_loc_error_message_includes_expected_shapes(self):
        """Test that error message includes expected shapes."""
        prior = ConcretePrior2D(prior_type="partial")
        prior.set_age_bounds(0, 3)  # A=4
        prior.set_event_dim(10)  # K=10 -> event_dim=10

        with pytest.raises(ValueError) as exc_info:
            prior.set_loc(np.ones((5, 5)))

        error_msg = str(exc_info.value)
        assert "(10, 4, 4)" in error_msg
        assert "(10, 4)" in error_msg
        assert "Scalar" in error_msg


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_single_age_group(self):
        """Test with single age group."""
        prior = ConcretePrior2D()
        prior.set_age_bounds(5, 5)  # Single age
        assert prior.A == 1

    def test_event_dim_one(self):
        """Test with event dimension of 1."""
        prior = ConcretePrior2D(prior_type="global")
        prior.set_event_dim(1)  # K=1 for global prior
        assert prior.event_dim == 1

    def test_large_age_range(self):
        """Test with large age range."""
        prior = ConcretePrior2D()
        prior.set_age_bounds(0, 100)
        assert prior.A == 101

    def test_global_prior_rejects_K_not_1(self):
        """Test that global prior rejects K != 1."""
        prior = ConcretePrior2D(prior_type="global")
        with pytest.raises(ValueError, match="K must be 1"):
            prior.set_event_dim(1000)  # K must be 1 for global priors


class TestAbstractMethods:
    """Test that abstract methods are properly defined."""

    def test_cannot_instantiate_abstract_class(self):
        """Test that Prior2D cannot be instantiated directly."""
        with pytest.raises(TypeError):
            Prior2D()

    def test_concrete_class_must_implement_set_age_bounds(self):
        """Test that set_age_bounds must be implemented."""

        class IncompletePrior(Prior2D):
            def _set_grid(self):
                pass

            def sample(self):
                pass

        # Should not raise error during class definition
        # but will raise TypeError when trying to instantiate
        with pytest.raises(TypeError):
            IncompletePrior()

    def test_concrete_class_must_implement_set_grid(self):
        """Test that _set_grid must be implemented."""

        class IncompletePrior(Prior2D):
            def set_age_bounds(self, min_age, max_age):
                pass

            def sample(self):
                pass

        with pytest.raises(TypeError):
            IncompletePrior()

    def test_concrete_class_must_implement_sample(self):
        """Test that sample must be implemented."""

        class IncompletePrior(Prior2D):
            def set_age_bounds(self, min_age, max_age):
                pass

            def _set_grid(self):
                pass

        with pytest.raises(TypeError):
            IncompletePrior()


# ==========================================================================
# Test subclasses using CASES
# ==========================================================================


@pytest.mark.parametrize("prior_cls, params", CASES)
def test_event_dims(prior_cls, params):
    """Test event dimension calculations for various prior subclasses."""
    prior = prior_cls(*params)
    prior.set_age_bounds(0, 9)  # A=10 for testing

    if prior.prior_type == "global":
        K = 1
    else:
        K = 5  # Arbitrary K for testing

    prior.set_event_dim(K)

    if prior.prior_type == "global":
        assert prior.event_dim == 1

    elif prior.prior_type == "partial":
        assert prior.event_dim == K

    elif prior.prior_type == "full":
        expected_event_dim = K * K
        assert prior.event_dim == expected_event_dim
        diag_eff = K

        non_diag = expected_event_dim - K
        non_diag_eff = non_diag // 2  # Halved due to reciprocity

        assert prior.event_dim_diag == K
        assert prior.event_dim_non_diag == non_diag
        assert prior.event_dim_non_diag_eff == non_diag_eff


@pytest.mark.parametrize("prior_cls, params", CASES)
def test_reciprocity_constraints(prior_cls, params):
    """Test that reciprocity constraints are satisfied for global and full cases."""
    # if prior_cls == vdKassteele:
    #     prior = prior_cls(*params)
    # else:
    #     prior = prior_cls(*params)
    prior = prior_cls(*params)
    prior.set_age_bounds(0, 4)  # A=5 for testing

    if prior.prior_type == "global":
        K = 1
    else:
        K = 3  # Arbitrary K for testing

    prior.set_event_dim(K)
    prior.set_loc(0)

    with numpyro.handlers.seed(rng_seed=0):
        sample = prior.sample()

    rtol_diag = 1e-5
    atol_diag = 1e-8
    rtol_offdiag = 1e-5
    atol_offdiag = 1e-8

    if prior.prior_type == "global":
        assert np.allclose(sample, sample.T, rtol=rtol_diag, atol=atol_diag)

    if prior.prior_type == "full":
        all_ix = np.arange(prior.event_dim)
        diag_ix = np.array([0, 4, 8])
        non_diag_ix = np.setdiff1d(all_ix, diag_ix)

        for i in diag_ix:
            assert_allclose(
                sample[i, :, :], sample[i, :, :].T, rtol=rtol_diag, atol=atol_diag
            )

        for i in non_diag_ix:
            j = (i // K) + (i % K) * K
            assert np.allclose(
                sample[i, :, :], sample[j, :, :].T, rtol=rtol_offdiag, atol=atol_offdiag
            )
