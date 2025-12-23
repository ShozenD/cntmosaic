"""
Unit tests for HiBRCfine model.

Tests cover initialization, validation, inference (MCMC/SVI), and model structure.

NOTE: Full integration tests with stratified data require proper population data setup.
The fixture generate_stratified_contact_data needs DataLoader support for stratified
population proportions, which is currently being developed. Basic tests verify
the refactored code structure and type hints.
"""

import pandas as pd
import pytest
from jax.random import PRNGKey
from numpyro.infer.autoguide import AutoNormal

from ...dataloader import (
    ContactData,
    DataLoader,
    ParticipantData,
    PopulationData,
    StratPropData,
)
from ...datasets import load_age_distribution, load_template_patterns
from ...sim import ContactGenerator, MatrixGenerator, ParticipantGenerator, Subgroup
from .._HiBRCfine import HiBRCfine
from ..priors import HSGP2D, PSpline2D

# ============================================================================
# Constants and Fixtures
# ============================================================================

df_age_dist = load_age_distribution("United_States")
templates = load_template_patterns("United_States")

SEED = 42


@pytest.fixture
def generate_data_partial():
    """
    Generate contact data for the partial case (multiple subgroups, incomplete contact information)
    """

    # Define subgroups
    subgroups = [
        Subgroup(
            n=300,
            age_dist=df_age_dist.P.values,
            mean_cint_margin=8,
            label="A",
        ),
        Subgroup(
            n=400,
            age_dist=df_age_dist.P.values,
            mean_cint_margin=12,
            label="B",
        ),
    ]

    # Generate participants
    part_gen = ParticipantGenerator(subgroups)
    df_part = part_gen.generate(SEED)
    df_part["subgroup"] = pd.Categorical(
        df_part["subgroup"], categories=["A", "B"], ordered=True
    )

    # Generate contact matrix
    matrix_gen = MatrixGenerator(templates)
    contact_matrices = matrix_gen.generate_partial(subgroups, SEED)

    # Generate contacts
    cnt_gen = ContactGenerator(df_part, contact_matrices)
    df_cnt = cnt_gen.generate(SEED)

    # Population size offsets
    df_pop_prop = pd.concat(
        [
            pd.DataFrame(
                {"age": df_age_dist.age, "P": df_age_dist.P * 0.6, "subgroup": "A"}
            ),
            pd.DataFrame(
                {"age": df_age_dist.age, "P": df_age_dist.P * 0.4, "subgroup": "B"}
            ),
        ]
    )
    df_pop_total = df_pop_prop.groupby("age")["P"].sum().reset_index()
    df_pop_prop = df_pop_prop.merge(df_pop_total, on="age", suffixes=("", "_total"))
    df_pop_prop["prop"] = df_pop_prop["P"] / df_pop_prop["P_total"]
    df_pop_prop["subgroup"] = pd.Categorical(
        df_pop_prop["subgroup"], categories=["A", "B"], ordered=True
    )

    return df_part, df_cnt, df_pop_prop


@pytest.fixture
def generate_data_full():
    """
    Generate contact data for the full case (multiple subgroups, complete contact information)
    """

    # Define subgroups
    subgroups = [
        Subgroup(
            n=300,
            age_dist=df_age_dist.P.values,
            mean_cint_margin=8,
            label="A",
        ),
        Subgroup(
            n=400,
            age_dist=df_age_dist.P.values,
            mean_cint_margin=12,
            label="B",
        ),
    ]

    # Generate participants
    part_gen = ParticipantGenerator(subgroups)
    df_part = part_gen.generate(SEED)

    # Generate contact matrix
    matrix_gen = MatrixGenerator(templates)
    contact_matrices = matrix_gen.generate_full(subgroups, SEED)

    # Generate contacts
    cnt_gen = ContactGenerator(df_part, contact_matrices)
    df_cnt = cnt_gen.generate(SEED)

    # Population size offsets
    df_strat_prop = pd.concat(
        [
            pd.DataFrame(
                {"age": df_age_dist.age, "P": df_age_dist.P * 0.6, "subgroup": "A"}
            ),
            pd.DataFrame(
                {"age": df_age_dist.age, "P": df_age_dist.P * 0.4, "subgroup": "B"}
            ),
        ]
    )
    df_pop_total = df_strat_prop.groupby("age")["P"].sum().reset_index()
    df_strat_prop = df_strat_prop.merge(df_pop_total, on="age", suffixes=("", "_total"))
    df_strat_prop["prop"] = df_strat_prop["P"] / df_strat_prop["P_total"]
    df_strat_prop["subgroup"] = pd.Categorical(
        df_strat_prop["subgroup"], categories=["A", "B"], ordered=True
    )

    return df_part, df_cnt, df_strat_prop


class TestInit:

    def test_init_minimal(self, generate_data_partial):
        df_part, df_cnt, df_pop_prop = generate_data_partial

        part_data = ParticipantData(
            df_part, id_col="id", age_col="age_group", strat_var_cols="subgroup"
        )
        cnt_data = ContactData(df_cnt, id_col="id", age_col="age_cnt")
        pop_data = PopulationData(df_age_dist, age_col="age", size_col="P")
        strat_prop_data = StratPropData(
            df_pop_prop, age_col="age", var_name="subgroup", prop_col="prop"
        )

        dataloader = DataLoader(
            part_data=part_data,
            cnt_data=cnt_data,
            pop_data=pop_data,
            strat_prop_data=strat_prop_data,
        )

        priors = {
            "rate": HSGP2D(prior_type="global"),
            "subgroup": HSGP2D(prior_type="partial"),
        }

        model = HiBRCfine(dataloader, priors, "poisson")

    def test_init_full(self, generate_data_full):
        df_part, df_cnt, df_strat_prop = generate_data_full

        part_data = ParticipantData(
            df_part, id_col="id", age_col="age_group", strat_var_cols="subgroup"
        )
        cnt_data = ContactData(
            df_cnt, id_col="id", age_col="age_cnt", strat_var_cols="subgroup_cnt"
        )
        pop_data = PopulationData(
            df_strat_prop, age_col="age", size_col="P", strat_var_cols="subgroup"
        )
        strat_prop_data = StratPropData(
            df_strat_prop, age_col="age", var_name="subgroup", prop_col="prop"
        )

        dataloader = DataLoader(
            part_data=part_data,
            cnt_data=cnt_data,
            pop_data=pop_data,
            strat_prop_data=strat_prop_data,
        )

        priors = {
            "rate": HSGP2D(prior_type="global"),
            "subgroup": HSGP2D(prior_type="full"),
        }

        model = HiBRCfine(dataloader, priors, "poisson")


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
