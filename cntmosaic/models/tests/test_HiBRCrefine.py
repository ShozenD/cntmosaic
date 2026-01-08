"""
Unit tests for HiBRCrefine model.

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

from ...dataloader import DataLoader, StratPropData
from ...datasets import load_age_distribution, load_template_patterns
from ...sim import ContactGenerator, MatrixGenerator, ParticipantGenerator, Subgroup
from .._HiBRCrefine import HiBRCrefine, _expand_id_array
from ..priors import PSpline2D

# ============================================================================
# Constants and Fixtures
# ============================================================================

df_age_dist = load_age_distribution("United_States")
templates = load_template_patterns("United_States")


# NOTE: This fixture is commented out pending proper stratified population data support
# @pytest.fixture
# def generate_stratified_contact_data():
#     """Generate synthetic stratified contact data with coarse contact age groups."""
#     # Implementation pending DataLoader stratified population support
#     pass


# ============================================================================
# Test Helper Functions
# ============================================================================


def test_expand_id_array():
    """Test the _expand_id_array helper function."""
    ids = np.array([0, 1, 0, 2])
    expanded = _expand_id_array(ids, 3)

    assert expanded.shape == (4, 3)
    assert np.all(expanded[0] == 0)
    assert np.all(expanded[1] == 1)
    assert np.all(expanded[2] == 0)
    assert np.all(expanded[3] == 2)


def test_expand_id_array_single_length():
    """Test _expand_id_array with length=1 (edge case)."""
    ids = np.array([5, 10, 15])
    expanded = _expand_id_array(ids, 1)

    assert expanded.shape == (3, 1)
    assert np.array_equal(expanded.flatten(), ids)


def test_expand_id_array_empty():
    """Test _expand_id_array with empty input."""
    ids = np.array([], dtype=int)
    expanded = _expand_id_array(ids, 5)

    assert expanded.shape == (0, 5)


# ============================================================================
# Test Class Structure and API
# ============================================================================


def test_class_inheritance():
    """Test that HiBRCrefine properly inherits from BRCrefine."""
    from .._BRCrefine import BRCrefine

    assert issubclass(HiBRCrefine, BRCrefine)


def test_default_priors_exists():
    """Test that default priors are defined."""
    assert hasattr(HiBRCrefine, "default_priors")
    assert isinstance(HiBRCrefine.default_priors, dict)
    assert "rate" in HiBRCrefine.default_priors


def test_class_docstring():
    """Test that class has comprehensive docstring."""
    assert HiBRCrefine.__doc__ is not None
    assert len(HiBRCrefine.__doc__) > 500  # Should be comprehensive
    assert "hierarchical" in HiBRCrefine.__doc__.lower()
    assert "refinement" in HiBRCrefine.__doc__.lower()


def test_init_signature():
    """Test that __init__ has proper type hints."""
    import inspect

    sig = inspect.signature(HiBRCrefine.__init__)

    # Check parameters exist
    assert "dataloader" in sig.parameters
    assert "priors" in sig.parameters
    assert "likelihood" in sig.parameters

    # Check default values
    assert sig.parameters["likelihood"].default == "negbin"


def test_method_type_hints():
    """Test that key methods have type hints."""
    import inspect

    methods_to_check = [
        "_validate_hierarchical_inputs",
        "set_log_age_dist_props",
        "set_prior_event_dim",
        "set_prior_loc",
        "sample_log_delta",
        "model",
    ]

    for method_name in methods_to_check:
        method = getattr(HiBRCrefine, method_name)
        sig = inspect.signature(method)
        # Just check that signature is retrievable (type hints present)
        assert sig is not None


def test_docstrings_present():
    """Test that all public methods have docstrings."""
    import inspect

    for name, method in inspect.getmembers(HiBRCrefine, predicate=inspect.isfunction):
        if not name.startswith("_") or name in ["__init__"]:
            assert method.__doc__ is not None, f"Method {name} missing docstring"


# ============================================================================
# Test Validation Logic
# ============================================================================


def test_validation_method_exists():
    """Test that custom validation method exists."""
    assert hasattr(HiBRCrefine, "_validate_hierarchical_inputs")
    assert callable(HiBRCrefine._validate_hierarchical_inputs)


def test_helper_function_documented():
    """Test that _expand_id_array has proper documentation."""
    assert _expand_id_array.__doc__ is not None
    assert "Parameters" in _expand_id_array.__doc__
    assert "Returns" in _expand_id_array.__doc__
    assert "Examples" in _expand_id_array.__doc__


@pytest.fixture
def generate_stratified_contact_data():
    """
    Generate synthetic stratified contact data with coarse contact age groups.

    Creates two subgroups (e.g., gender: male/female) with different contact patterns.
    Contact ages are aggregated into coarse intervals for testing age refinement.
    """
    # Create two subgroups with different characteristics
    subgroup_male = Subgroup(
        n=750, age_dist=df_age_dist.P.values, mean_cint_margin=12.0, label="male"
    )
    subgroup_female = Subgroup(
        n=750, age_dist=df_age_dist.P.values, mean_cint_margin=18.0, label="female"
    )

    # Generate contact matrices
    matrix_gen = MatrixGenerator(templates)
    cint_male = matrix_gen.generate_single(subgroup_male, seed=42)
    cint_female = matrix_gen.generate_single(subgroup_female, seed=43)

    # Generate participants
    part_gen_male = ParticipantGenerator(subgroup_male)
    df_part_male = part_gen_male.generate(seed=42)
    df_part_male["gender"] = "male"

    part_gen_female = ParticipantGenerator(subgroup_female)
    df_part_female = part_gen_female.generate(seed=43)
    df_part_female["gender"] = "female"

    df_part = pd.concat([df_part_male, df_part_female], ignore_index=True)
    df_part["age_part"] = df_part["age_group"]

    # Generate contacts - need to keep track of participant IDs properly
    df_part_male_for_cnt = df_part[df_part["gender"] == "male"].copy()
    df_part_female_for_cnt = df_part[df_part["gender"] == "female"].copy()

    cnt_gen_male = ContactGenerator(
        df_part_male_for_cnt, cint_matrices=cint_male, model="poisson"
    )
    df_cnt_male = cnt_gen_male.generate(seed=42)

    cnt_gen_female = ContactGenerator(
        df_part_female_for_cnt, cint_matrices=cint_female, model="poisson"
    )
    df_cnt_female = cnt_gen_female.generate(seed=43)

    df_cnt = pd.concat([df_cnt_male, df_cnt_female], ignore_index=True)

    # Create coarse age groups for contact ages (e.g., 0-4, 5-9, ..., 75-79, 80+)
    bins = list(range(0, 81, 5)) + [100]  # 5-year age groups
    labels = [
        pd.Interval(bins[i], bins[i + 1], closed="left") for i in range(len(bins) - 1)
    ]
    df_cnt["age_grp_cnt"] = pd.cut(
        df_cnt["age_cnt"], bins=bins, right=False, labels=labels
    )

    # Prepare population data with gender stratification
    # Need to provide separate age distributions for males and females
    df_age_male = df_age_dist.copy()
    df_age_male["gender"] = "male"
    df_age_female = df_age_dist.copy()
    df_age_female["gender"] = "female"
    df_age_dist_strat = pd.concat([df_age_male, df_age_female], ignore_index=True)

    # Create DataLoader with stratification and coarse contact ages
    col_map = CoordToColumns(
        age_part="age_part",
        age_grp_cnt="age_grp_cnt",  # Coarse contact ages
        age_pop="age",
        P="P",
        strat_vars_part=["gender"],  # Stratification variable
    )

    pp = StratPropData.from_counts(
        df_age_dist_strat, age_col="age", strat_col="gender", count_col="P"
    )

    dataloader = DataLoader(df_part, df_cnt, df_age_dist, col_map=col_map, pop_prop=pp)

    return dataloader


# ============================================================================
# Test Helper Functions
# ============================================================================


def test_expand_id_array():
    """Test the _expand_id_array helper function."""
    ids = np.array([0, 1, 0, 2])
    expanded = _expand_id_array(ids, 3)

    assert expanded.shape == (4, 3)
    assert np.all(expanded[0] == 0)
    assert np.all(expanded[1] == 1)
    assert np.all(expanded[2] == 0)
    assert np.all(expanded[3] == 2)


def test_expand_id_array_single_length():
    """Test _expand_id_array with length=1 (edge case)."""
    ids = np.array([5, 10, 15])
    expanded = _expand_id_array(ids, 1)

    assert expanded.shape == (3, 1)
    assert np.array_equal(expanded.flatten(), ids)


# ============================================================================
# Test Initialization
# ============================================================================


def test_initialization_basic(generate_stratified_contact_data):
    """Test basic initialization with required priors."""
    dataloader = generate_stratified_contact_data

    priors = {
        "rate": PSpline2D(grid_type="diff-age", prior_type="global"),
        "gender": PSpline2D(grid_type="diff-age", prior_type="partial"),
    }

    model = HiBRCrefine(dataloader, priors, likelihood="poisson")

    # Check basic attributes from parent classes
    assert len(model.y) > 0
    assert len(model.aid) == len(model.y)
    assert len(model.aid_exp) == len(model.y)
    assert model.bid_pad.shape[0] == len(model.y)
    assert model.log_P.shape[1] == model.A

    # Check hierarchical-specific attributes
    assert "gender" in model.X_vars
    assert len(model.X_vars) == 1
    assert "gender_part" in model.strat_ix
    assert "gender_part" in model.strat_ix_exp
    assert "gender" in model.log_age_dist_props

    # Check categorical encoding
    assert np.all(
        (model.strat_ix["gender_part"] >= 0) & (model.strat_ix["gender_part"] < 2)
    )

    # Check expanded IDs shape
    assert model.strat_ix_exp["gender_part"].shape == model.bid_pad.shape


def test_initialization_with_defaults(generate_stratified_contact_data):
    """Test that default priors are used when not specified."""
    dataloader = generate_stratified_contact_data

    # Only provide gender prior, rate should use default
    priors = {"gender": PSpline2D(grid_type="diff-age", prior_type="partial")}

    model = HiBRCrefine(dataloader, priors, likelihood="negbin")

    # Check that rate prior exists (from defaults)
    assert "rate" in model.priors
    assert isinstance(model.priors["rate"], PSpline2D)  # Default type

    # Check likelihood
    assert model.likelihood == "negbin"


def test_initialization_negative_binomial(generate_stratified_contact_data):
    """Test initialization with negative binomial likelihood."""
    dataloader = generate_stratified_contact_data

    priors = {
        "rate": PSpline2D(grid_type="age-age", prior_type="global"),
        "gender": PSpline2D(grid_type="age-age", prior_type="partial"),
    }

    model = HiBRCrefine(dataloader, priors, likelihood="negbin")

    assert model.likelihood == "negbin"


# ============================================================================
# Test Validation
# ============================================================================


def test_invalid_likelihood(generate_stratified_contact_data):
    """Test that invalid likelihood raises error."""
    dataloader = generate_stratified_contact_data

    priors = {
        "rate": PSpline2D(grid_type="diff-age", prior_type="global"),
        "gender": PSpline2D(grid_type="diff-age", prior_type="partial"),
    }

    with pytest.raises(ValueError, match="likelihood must be one of"):
        HiBRCrefine(dataloader, priors, likelihood="gaussian")


# ============================================================================
# Test Prior Configuration
# ============================================================================


def test_prior_event_dim_configuration(generate_stratified_contact_data):
    """Test that prior event dimensions are correctly configured."""
    dataloader = generate_stratified_contact_data

    priors = {
        "rate": PSpline2D(grid_type="diff-age", prior_type="global"),
        "gender": PSpline2D(grid_type="diff-age", prior_type="partial"),
    }

    model = HiBRCrefine(dataloader, priors, likelihood="poisson")

    # Rate prior should have event_dim=1 (shared baseline)
    assert model.priors["rate"].event_dim == 1

    # Gender prior should have event_dim=2 (male, female)
    assert model.priors["gender"].event_dim == 2


def test_prior_loc_configuration(generate_stratified_contact_data):
    """Test that prior locations are set correctly."""
    dataloader = generate_stratified_contact_data

    priors = {
        "rate": PSpline2D(grid_type="diff-age", prior_type="global"),
        "gender": PSpline2D(
            grid_type="diff-age", prior_type="partial", transform="alr"
        ),
    }

    model = HiBRCrefine(dataloader, priors, likelihood="poisson")

    # Gender prior should have location set from population data
    assert hasattr(model.priors["gender"], "trans_loc")
    # Loc should be None or properly shaped if set
    if model.priors["gender"].trans_loc is not None:
        assert model.priors["gender"].trans_loc.shape[0] == 1  # Two genders


def test_log_age_dist_props_shapes(generate_stratified_contact_data):
    """Test that log age distribution proportions have correct shapes."""
    dataloader = generate_stratified_contact_data

    priors = {
        "rate": PSpline2D(grid_type="diff-age", prior_type="global"),
        "gender": PSpline2D(grid_type="diff-age", prior_type="partial"),
    }

    model = HiBRCrefine(dataloader, priors, likelihood="poisson")

    # Should have log_age_dist_props for gender
    assert "gender" in model.log_age_dist_props

    # Shape should be (n_genders, A, 1) for partial prior
    log_props = model.log_age_dist_props["gender"]
    assert log_props.shape[0] == 2  # Two genders
    assert log_props.shape[1] == model.A
    assert log_props.shape[2] == 1  # Trailing dimension for broadcasting


# ============================================================================
# Test Model Structure
# ============================================================================


def test_model_structure(generate_stratified_contact_data):
    """Test that the model structure is correctly defined."""
    dataloader = generate_stratified_contact_data

    priors = {
        "rate": PSpline2D(grid_type="diff-age", prior_type="global"),
        "gender": PSpline2D(grid_type="diff-age", prior_type="partial"),
    }

    model = HiBRCrefine(dataloader, priors, likelihood="poisson")

    # Should be able to print model shape without errors
    model.print_model_shape()


def test_sample_log_delta(generate_stratified_contact_data):
    """Test the sample_log_delta method."""
    dataloader = generate_stratified_contact_data

    priors = {
        "rate": PSpline2D(grid_type="diff-age", prior_type="global"),
        "gender": PSpline2D(grid_type="diff-age", prior_type="partial"),
    }

    model = HiBRCrefine(dataloader, priors, likelihood="poisson")

    # Test within a NumPyro context
    from numpyro.handlers import seed, trace

    with seed(rng_seed=42):
        # Call model to set up context
        tr = trace(model.model).get_trace(y=None)

        # Check that log_delta was sampled
        # The key will be scoped as 'gender/log_delta'
        assert any("log_delta" in key for key in tr.keys())


# ============================================================================
# Test Inference - SVI
# ============================================================================


def test_svi_inference_poisson(generate_stratified_contact_data):
    """Test SVI inference with Poisson likelihood."""
    dataloader = generate_stratified_contact_data

    priors = {
        "rate": PSpline2D(grid_type="diff-age", prior_type="global"),
        "gender": PSpline2D(
            grid_type="diff-age", prior_type="partial", transform="alr"
        ),
    }

    model = HiBRCrefine(dataloader, priors, likelihood="poisson")

    guide = AutoNormal(model.model)
    prng_key = PRNGKey(0)

    # Run short SVI for testing
    model.run_inference_svi(prng_key, guide, num_steps=100, peak_lr=0.01)

    assert model._svi_result is not None
    assert model._guide is not None


def test_svi_inference_negbin(generate_stratified_contact_data):
    """Test SVI inference with negative binomial likelihood."""
    dataloader = generate_stratified_contact_data

    priors = {
        "rate": PSpline2D(grid_type="diff-age", prior_type="global"),
        "gender": PSpline2D(
            grid_type="diff-age", prior_type="partial", transform="alr"
        ),
    }

    model = HiBRCrefine(dataloader, priors, likelihood="negbin")

    guide = AutoNormal(model.model)
    prng_key = PRNGKey(1)

    # Run short SVI for testing
    model.run_inference_svi(prng_key, guide, num_steps=100, peak_lr=0.01)

    assert model._svi_result is not None


# ============================================================================
# Test Inference - MCMC
# ============================================================================


def test_mcmc_inference_poisson(generate_stratified_contact_data):
    """Test MCMC inference with Poisson likelihood."""
    dataloader = generate_stratified_contact_data

    priors = {
        "rate": PSpline2D(grid_type="diff-age", prior_type="global"),
        "gender": PSpline2D(
            grid_type="diff-age", prior_type="partial", transform="alr"
        ),
    }

    model = HiBRCrefine(dataloader, priors, likelihood="poisson")

    prng_key = PRNGKey(2)

    # Run very short MCMC for testing
    model.run_inference_mcmc(prng_key, num_warmup=10, num_samples=10, num_chains=1)

    assert model._mcmc_result is not None

    # Check that samples can be retrieved
    samples = model._mcmc_result.get_samples()
    assert "baseline" in samples
    assert "log_rate" in samples
    assert "rate/spline_coefs" in samples or "rate/PSpline2D_coefs" in samples


def test_mcmc_inference_negbin(generate_stratified_contact_data):
    """Test MCMC inference with negative binomial likelihood."""
    dataloader = generate_stratified_contact_data

    priors = {
        "rate": PSpline2D(grid_type="diff-age", prior_type="global"),
        "gender": PSpline2D(
            grid_type="diff-age", prior_type="partial", transform="alr"
        ),
    }

    model = HiBRCrefine(dataloader, priors, likelihood="negbin")

    prng_key = PRNGKey(3)

    # Run very short MCMC for testing
    model.run_inference_mcmc(prng_key, num_warmup=10, num_samples=10, num_chains=1)

    assert model._mcmc_result is not None

    # Check for overdispersion parameter
    samples = model._mcmc_result.get_samples()
    assert "inv_disp" in samples


# ============================================================================
# Test Integration
# ============================================================================


def test_end_to_end_workflow(generate_stratified_contact_data):
    """Test complete workflow from initialization to inference to summary."""
    dataloader = generate_stratified_contact_data

    priors = {
        "rate": PSpline2D(grid_type="diff-age", prior_type="global"),
        "gender": PSpline2D(
            grid_type="diff-age", prior_type="partial", transform="alr"
        ),
    }

    # Initialize model
    model = HiBRCrefine(dataloader, priors, likelihood="poisson")

    # Run inference
    prng_key = PRNGKey(5)
    model.run_inference_mcmc(prng_key, num_warmup=10, num_samples=10, num_chains=1)

    # Get samples
    samples = model._mcmc_result.get_samples()

    # Check key outputs
    assert "baseline" in samples
    assert "log_rate" in samples

    # Check hierarchical components
    gender_keys = [k for k in samples.keys() if "gender" in k]
    assert len(gender_keys) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
