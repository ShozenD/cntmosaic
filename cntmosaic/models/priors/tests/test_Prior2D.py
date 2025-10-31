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

import pytest
import numpy as np
import jax.numpy as jnp
from .._Prior2D import Prior2D


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
        assert prior.transform is None
        assert prior.type == "global"
        assert prior.prior_type == "global"
        assert prior.A is None
        assert prior.event_dim is None
        assert prior.event_dim_eff is None
        assert prior.trans_loc is None

    def test_custom_initialization(self):
        """Test initialization with custom parameters."""
        prior = ConcretePrior2D(
            grid_type="diff-age", transform="alr", prior_type="partial"
        )
        assert prior.grid_type == "diff-age"
        assert prior.transform == "alr"
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

    def test_invalid_transform(self):
        """Test that invalid transform raises ValueError."""
        with pytest.raises(ValueError, match="transform must be one of"):
            ConcretePrior2D(transform="invalid")

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

    def test_all_valid_transforms(self):
        """Test all allowed transforms are accepted."""
        for transform in Prior2D.ALLOWED_TRANSFORMS:
            prior = ConcretePrior2D(transform=transform)
            assert prior.transform == transform

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
        """Test event dimension for global prior."""
        prior = ConcretePrior2D(prior_type="global")
        prior.set_event_dim(25)
        assert prior.event_dim == 25
        assert prior.event_dim_eff == 1

    def test_partial_prior_no_transform(self):
        """Test partial prior without transformation."""
        prior = ConcretePrior2D(prior_type="partial", transform=None)
        prior.set_event_dim(10)
        assert prior.event_dim == 10
        assert prior.event_dim_eff == 10

    def test_partial_prior_with_alr(self):
        """Test partial prior with ALR transformation."""
        prior = ConcretePrior2D(prior_type="partial", transform="alr")
        prior.set_event_dim(10)
        assert prior.event_dim == 10
        assert prior.event_dim_eff == 9

    def test_partial_prior_with_ilr(self):
        """Test partial prior with ILR transformation."""
        prior = ConcretePrior2D(prior_type="partial", transform="ilr")
        prior.set_event_dim(10)
        assert prior.event_dim == 10
        assert prior.event_dim_eff == 9

    def test_partial_prior_with_clr(self):
        """Test partial prior with CLR transformation."""
        prior = ConcretePrior2D(prior_type="partial", transform="clr")
        prior.set_event_dim(10)
        assert prior.event_dim == 10
        assert prior.event_dim_eff == 10

    def test_full_prior_no_transform(self):
        """Test full prior without transformation."""
        prior = ConcretePrior2D(prior_type="full", transform=None)
        prior.set_event_dim(25)  # 5x5
        assert prior.event_dim == 25
        assert prior.event_dim_eff == 25
        assert prior.event_dim_diag == 5
        assert prior.event_dim_non_diag == 20

    def test_full_prior_with_alr(self):
        """Test full prior with ALR transformation."""
        prior = ConcretePrior2D(prior_type="full", transform="alr")
        prior.set_event_dim(25)
        assert prior.event_dim == 25
        assert prior.event_dim_eff == 24
        assert prior.event_dim_diag == 4
        assert prior.event_dim_non_diag == 20

    def test_full_prior_with_ilr(self):
        """Test full prior with ILR transformation."""
        prior = ConcretePrior2D(prior_type="full", transform="ilr")
        prior.set_event_dim(16)  # 4x4
        assert prior.event_dim == 16
        assert prior.event_dim_eff == 15
        assert prior.event_dim_diag == 3
        assert prior.event_dim_non_diag == 12

    def test_full_prior_non_square_event_dim(self):
        """Test that non-square event_dim raises error for full prior."""
        prior = ConcretePrior2D(prior_type="full")
        with pytest.raises(ValueError, match="must be a perfect square"):
            prior.set_event_dim(20)

    def test_invalid_event_dim_type(self):
        """Test that non-integer event_dim raises error."""
        prior = ConcretePrior2D()
        with pytest.raises(AssertionError):
            prior.set_event_dim(10.5)

    def test_negative_event_dim(self):
        """Test that negative event_dim raises error."""
        prior = ConcretePrior2D()
        with pytest.raises(AssertionError):
            prior.set_event_dim(-5)

    def test_zero_event_dim(self):
        """Test that zero event_dim raises error."""
        prior = ConcretePrior2D()
        with pytest.raises(AssertionError):
            prior.set_event_dim(0)


class TestSetLoc:
    """Test location parameter setting."""

    def test_scalar_loc(self):
        """Test setting scalar location parameter."""
        prior = ConcretePrior2D(prior_type="partial", transform="alr")
        prior.set_age_bounds(0, 4)  # A=5
        prior.set_event_dim(25)  # event_dim_eff=24

        prior.set_loc(0.5)
        assert prior.trans_loc.shape == (24, 5, 5)
        assert jnp.allclose(prior.trans_loc, 0.5)

    def test_full_array_loc(self):
        """Test setting full array location parameter."""
        prior = ConcretePrior2D(prior_type="partial", transform=None)
        prior.set_age_bounds(0, 3)  # A=4
        prior.set_event_dim(10)  # event_dim_eff=10

        loc_array = np.random.randn(10, 4, 4)
        prior.set_loc(loc_array)
        assert prior.trans_loc.shape == (10, 4, 4)
        assert jnp.allclose(prior.trans_loc, loc_array)

    def test_broadcasted_loc(self):
        """Test setting location with broadcasting."""
        prior = ConcretePrior2D(prior_type="partial", transform=None)
        prior.set_age_bounds(0, 2)  # A=3
        prior.set_event_dim(6)  # event_dim_eff=6

        loc_broadcast = np.ones((6, 3))
        prior.set_loc(loc_broadcast)
        assert prior.trans_loc.shape == (6, 3, 3)
        # Check that it's broadcasted along last axis (no transform, so values preserved)
        for i in range(6):
            for j in range(3):
                assert jnp.allclose(prior.trans_loc[i, j, :], 1.0)

    def test_loc_with_jax_array(self):
        """Test that JAX arrays are handled correctly."""
        prior = ConcretePrior2D()
        prior.set_age_bounds(0, 2)
        prior.set_event_dim(3)

        loc_jax = jnp.array(0.3)
        prior.set_loc(loc_jax)
        assert isinstance(prior.trans_loc, jnp.ndarray)

    def test_loc_before_age_bounds_raises_error(self):
        """Test that setting loc before age bounds raises error."""
        prior = ConcretePrior2D()
        with pytest.raises(ValueError, match="Age bounds must be set"):
            prior.set_loc(0.5)

    def test_loc_before_event_dim_raises_error(self):
        """Test that setting loc before event_dim raises error."""
        prior = ConcretePrior2D()
        prior.set_age_bounds(0, 4)
        with pytest.raises(ValueError, match="Event dimension must be set"):
            prior.set_loc(0.5)

    def test_invalid_loc_shape_raises_error(self):
        """Test that invalid loc shape raises informative error."""
        prior = ConcretePrior2D(prior_type="partial")
        prior.set_age_bounds(0, 4)  # A=5
        prior.set_event_dim(10)  # event_dim_eff=10

        with pytest.raises(ValueError, match="Invalid loc shape"):
            prior.set_loc(np.ones((5, 5)))  # Wrong shape

    def test_loc_error_message_includes_expected_shapes(self):
        """Test that error message includes expected shapes."""
        prior = ConcretePrior2D(prior_type="partial", transform="alr")
        prior.set_age_bounds(0, 3)  # A=4
        prior.set_event_dim(10)  # event_dim_eff=9

        with pytest.raises(ValueError) as exc_info:
            prior.set_loc(np.ones((5, 5)))

        error_msg = str(exc_info.value)
        assert "(9, 4, 4)" in error_msg
        assert "(9, 4)" in error_msg
        assert "Scalar" in error_msg


class TestTransformations:
    """Test compositional transformations in set_loc."""

    def test_no_transform(self):
        """Test that data is unchanged without transformation."""
        prior = ConcretePrior2D(prior_type="partial", transform=None)
        prior.set_age_bounds(0, 2)
        prior.set_event_dim(3)

        loc_data = np.random.rand(3, 3, 3)
        prior.set_loc(loc_data)
        # Should be unchanged (converted to JAX but same values)
        assert jnp.allclose(prior.trans_loc, loc_data)

    def test_alr_transform_applied(self):
        """Test that ALR transformation is applied."""
        prior = ConcretePrior2D(prior_type="partial", transform="alr")
        prior.set_age_bounds(0, 2)
        prior.set_event_dim(4)  # event_dim_eff=3

        # Create compositional data
        loc_data = jnp.array(
            [[[0.25, 0.25, 0.25], [0.25, 0.25, 0.25], [0.25, 0.25, 0.25]]] * 3
        )

        prior.set_loc(loc_data)
        # After ALR, values should be different (log ratios)
        assert prior.trans_loc.shape == (3, 3, 3)
        assert not jnp.allclose(prior.trans_loc, loc_data)

    def test_clr_transform_applied(self):
        """Test that CLR transformation is applied."""
        prior = ConcretePrior2D(prior_type="partial", transform="clr")
        prior.set_age_bounds(0, 2)
        prior.set_event_dim(3)

        loc_data = jnp.ones((3, 3, 3)) * 0.333
        prior.set_loc(loc_data)
        # CLR centers the log values
        assert prior.trans_loc.shape == (3, 3, 3)

    def test_ilr_transform_applied(self):
        """Test that ILR transformation is applied."""
        prior = ConcretePrior2D(prior_type="partial", transform="ilr")
        prior.set_age_bounds(0, 2)
        prior.set_event_dim(4)  # event_dim_eff=3

        loc_data = jnp.ones((3, 3, 3)) * 0.25
        prior.set_loc(loc_data)
        # ILR reduces dimension by 1 along the transformed axis
        # So if transforming along axis=-1 with size 3, result is size 2
        assert prior.trans_loc.shape == (3, 3, 2)


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
        prior.set_event_dim(1)
        assert prior.event_dim_eff == 1

    def test_large_age_range(self):
        """Test with large age range."""
        prior = ConcretePrior2D()
        prior.set_age_bounds(0, 100)
        assert prior.A == 101

    def test_global_prior_ignores_event_dim_value(self):
        """Test that global prior always has event_dim_eff=1."""
        prior = ConcretePrior2D(prior_type="global")
        prior.set_event_dim(1000)  # Large value
        assert prior.event_dim_eff == 1  # Still 1


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


class TestDocumentation:
    """Test that documentation is present and comprehensive."""

    def test_class_has_docstring(self):
        """Test that Prior2D has a docstring."""
        assert Prior2D.__doc__ is not None
        assert len(Prior2D.__doc__) > 100

    def test_init_has_docstring(self):
        """Test that __init__ has a docstring."""
        assert Prior2D.__init__.__doc__ is not None

    def test_validate_params_has_docstring(self):
        """Test that validate_params has a docstring."""
        assert Prior2D.validate_params.__doc__ is not None

    def test_set_loc_has_docstring(self):
        """Test that set_loc has a docstring."""
        assert Prior2D.set_loc.__doc__ is not None
        assert "Parameters" in Prior2D.set_loc.__doc__
        assert "Raises" in Prior2D.set_loc.__doc__

    def test_set_event_dim_has_docstring(self):
        """Test that set_event_dim has a docstring."""
        assert Prior2D.set_event_dim.__doc__ is not None
        assert "Parameters" in Prior2D.set_event_dim.__doc__

    def test_abstract_methods_have_docstrings(self):
        """Test that abstract methods have docstrings."""
        assert Prior2D.set_age_bounds.__doc__ is not None
        assert Prior2D._set_grid.__doc__ is not None
        assert Prior2D.sample.__doc__ is not None
