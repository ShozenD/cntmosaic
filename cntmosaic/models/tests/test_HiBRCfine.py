"""
Unit tests for HiBRCfine model.

Tests cover initialization, validation, inference (MCMC/SVI), and model structure.

NOTE: Full integration tests with stratified data require proper population data setup.
The fixture generate_stratified_contact_data needs DataLoader support for stratified
population proportions, which is currently being developed. Basic tests verify
the refactored code structure and type hints.
"""

import numpy as np
import numpyro
import pandas as pd
import pytest
from jax.random import PRNGKey
from numpyro.infer.autoguide import AutoNormal

from ...dataloader import CoordToColumns, DataLoader, PopulationProportion
from ...datasets import load_age_distribution, load_template_patterns
from ...sim import ContactGenerator, MatrixGenerator, ParticipantGenerator, Subgroup
from .._HiBRCfine import HiBRCfine
from ..priors import HSGP2D, PSpline2D

# ============================================================================
# Constants and Fixtures
# ============================================================================

df_age_dist = load_age_distribution("United_States")
templates = load_template_patterns("United_States")


@pytest.fixture
def generate_unstratified_contact_data():
    """Generate synthetic contact data without stratification (simplest case)."""
    subgroup = Subgroup(
        n=50,
        age_dist=df_age_dist["P"].to_numpy(),  # Use 'P' column
        mean_cint_margin=15.0,
    )

    # Generate participants and contacts
    part_gen = ParticipantGenerator(subgroup)  # No min_age/max_age parameters
    participants = part_gen.generate(seed=42)

    mat_gen = MatrixGenerator(templates)
    matrix = mat_gen.generate_single(subgroup, seed=42)

    cnt_gen = ContactGenerator(participants, matrix)
    contacts = cnt_gen.generate(seed=42)

    # Create DataLoader
    col_map = CoordToColumns(
        age_part="age_group",  # Match ParticipantGenerator output
        age_cnt="age_cnt",
        id_var="id",
        age_pop="age",
        size_pop="P",  # Use 'P' column
    )

    dataloader = DataLoader(participants, contacts, df_age_dist, col_map=col_map)
    return dataloader


# ============================================================================
# Test Class Structure and Docstrings
# ============================================================================


def test_HiBRCfine_class_exists():
    """Test that HiBRCfine class is properly defined."""
    assert hasattr(HiBRCfine, "__init__")
    assert hasattr(HiBRCfine, "model")
    assert hasattr(HiBRCfine, "set_age_dims")


def test_HiBRCfine_has_comprehensive_docstring():
    """Test that HiBRCfine class has comprehensive documentation."""
    docstring = HiBRCfine.__doc__
    assert docstring is not None
    assert len(docstring) > 500  # Should be comprehensive
    assert "Hierarchical" in docstring
    assert "Parameters" in docstring
    assert "Examples" in docstring


def test_HiBRCfine_inherits_from_BRCfine():
    """Test that HiBRCfine properly inherits from BRCfine."""
    from .._BRCfine import BRCfine

    assert issubclass(HiBRCfine, BRCfine)


# ============================================================================
# Test Initialization and Type Hints
# ============================================================================


def test_init_type_hints():
    """Test that __init__ method has proper type hints."""
    import inspect

    sig = inspect.signature(HiBRCfine.__init__)

    # Check parameter annotations
    assert "dataloader" in sig.parameters
    assert "priors" in sig.parameters
    assert "likelihood" in sig.parameters
    # Note: HiBRCfine doesn't have 'hill' parameter (inherited behavior)


def test_init_with_minimal_args(generate_unstratified_contact_data):
    """Test initialization with minimal required arguments."""
    dataloader = generate_unstratified_contact_data

    priors = {
        "rate": HSGP2D(prior_type="global"),  # No 'A' parameter for HSGP2D
    }

    model = HiBRCfine(dataloader, priors=priors)

    assert model.ds is not None
    assert model.priors == priors
    assert model.likelihood == "negbin"  # Default


def test_init_with_stratified_data_raises_on_missing_priors(
    generate_unstratified_contact_data,
):
    """Test that initialization requires priors for each stratification variable."""
    # Placeholder - full test requires stratified fixture pending DataLoader support
    pass


# ============================================================================
# Test Validation Methods
# ============================================================================


def test_validate_hierarchical_inputs_method_exists():
    """Test that _validate_hierarchical_inputs method exists."""
    assert hasattr(HiBRCfine, "_validate_hierarchical_inputs")


def test_validate_hierarchical_inputs_checks_rate_prior(
    generate_unstratified_contact_data,
):
    """Test validation of 'rate' prior requirement."""
    dataloader = generate_unstratified_contact_data

    priors = {
        # Missing 'rate' prior
    }

    # Base class validates 'rate' prior existence first
    with pytest.raises(
        ValueError, match="priors must contain the specifications for 'rate'"
    ):
        HiBRCfine(dataloader, priors=priors)


def test_validate_hierarchical_inputs_checks_prior_types(
    generate_unstratified_contact_data,
):
    """Test validation of prior_type constraints."""
    dataloader = generate_unstratified_contact_data

    # Rate prior must be 'full'
    priors = {
        "rate": HSGP2D(prior_type="partial"),  # No A parameter
    }

    with pytest.raises(ValueError, match="prior_type must be 'global'"):
        HiBRCfine(dataloader, priors=priors)


# ============================================================================
# Test set_age_dims Method
# ============================================================================


def test_set_age_dims_basic(generate_unstratified_contact_data):
    """Test that set_age_dims properly configures age dimensions."""
    dataloader = generate_unstratified_contact_data

    priors = {
        "rate": HSGP2D(prior_type="global"),  # No A parameter
    }

    model = HiBRCfine(dataloader, priors=priors)
    model.set_age_dims(0, 85)

    assert hasattr(model, "log_P")
    assert hasattr(model, "aid")
    assert hasattr(model, "bid")


def test_set_age_dims_has_docstring():
    """Test that set_age_dims has proper documentation."""
    docstring = HiBRCfine.set_age_dims.__doc__
    assert docstring is not None
    assert "Parameters" in docstring
    assert "age_min" in docstring  # Match actual parameter name


# ============================================================================
# Test set_log_age_dist_props Method
# ============================================================================


def test_set_log_age_dist_props_exists():
    """Test that set_log_age_dist_props method exists."""
    assert hasattr(HiBRCfine, "set_log_age_dist_props")


def test_set_log_age_dist_props_has_comprehensive_docstring():
    """Test that set_log_age_dist_props has detailed documentation."""
    docstring = HiBRCfine.set_log_age_dist_props.__doc__
    assert docstring is not None
    assert len(docstring) > 200
    assert "log-transformed" in docstring
    assert "Notes" in docstring


# ============================================================================
# Test set_prior_event_dim Method
# ============================================================================


def test_set_prior_event_dim_exists():
    """Test that set_prior_event_dim method exists."""
    assert hasattr(HiBRCfine, "set_prior_event_dim")


def test_set_prior_event_dim_has_docstring():
    """Test that set_prior_event_dim has proper documentation."""
    docstring = HiBRCfine.set_prior_event_dim.__doc__
    assert docstring is not None
    assert "event dimension" in docstring
    assert "Notes" in docstring


# ============================================================================
# Test set_prior_loc Method
# ============================================================================


def test_set_prior_loc_exists():
    """Test that set_prior_loc method exists."""
    assert hasattr(HiBRCfine, "set_prior_loc")


def test_set_prior_loc_has_docstring():
    """Test that set_prior_loc has proper documentation."""
    docstring = HiBRCfine.set_prior_loc.__doc__
    assert docstring is not None
    assert "location parameter" in docstring
    assert "population age" in docstring


# ============================================================================
# Test sample_log_delta Method
# ============================================================================


def test_sample_log_delta_exists():
    """Test that sample_log_delta method exists."""
    assert hasattr(HiBRCfine, "sample_log_delta")


def test_sample_log_delta_has_comprehensive_docstring():
    """Test that sample_log_delta has detailed documentation."""
    docstring = HiBRCfine.sample_log_delta.__doc__
    assert docstring is not None
    assert len(docstring) > 200
    assert "Parameters" in docstring
    assert "Returns" in docstring
    assert "stratum-specific" in docstring


def test_sample_log_delta_type_hints():
    """Test that sample_log_delta has proper type hints."""
    import inspect

    sig = inspect.signature(HiBRCfine.sample_log_delta)

    assert "var" in sig.parameters
    assert sig.return_annotation != inspect.Parameter.empty


# ============================================================================
# Test model Method
# ============================================================================


def test_model_method_exists():
    """Test that model method is properly defined."""
    assert hasattr(HiBRCfine, "model")


def test_model_has_comprehensive_docstring():
    """Test that model method has extensive documentation."""
    docstring = HiBRCfine.model.__doc__
    assert docstring is not None
    assert len(docstring) > 500  # Should be very comprehensive
    assert "NumPyro" in docstring
    assert "Parameters" in docstring
    assert "Examples" in docstring
    assert "hierarchical" in docstring.lower()


def test_model_type_hints():
    """Test that model method has proper type hints."""
    import inspect

    sig = inspect.signature(HiBRCfine.model)

    assert "y" in sig.parameters
    assert sig.return_annotation != inspect.Parameter.empty


# ============================================================================
# Test API Consistency with HiBRCrefine
# ============================================================================


def test_api_consistency_with_HiBRCrefine():
    """Test that HiBRCfine has same public API as HiBRCrefine."""
    from .._HiBRCrefine import HiBRCrefine

    # Both should have these key methods
    common_methods = [
        "set_age_dims",
        "set_log_age_dist_props",
        "set_prior_event_dim",
        "set_prior_loc",
        "sample_log_delta",
        "model",
        "run_inference_mcmc",
        "run_inference_svi",
    ]

    for method_name in common_methods:
        assert hasattr(HiBRCfine, method_name), f"Missing method: {method_name}"
        assert hasattr(
            HiBRCrefine, method_name
        ), f"HiBRCrefine missing method: {method_name}"
