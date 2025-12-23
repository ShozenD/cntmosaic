import numpy as np
import pytest

from ...datasets._base import load_age_distribution, load_template_patterns
from .._MatrixGenerator import MatrixGenerator
from .._ParticipantGenerator import Subgroup

# ===== Template Validation Tests =====


def test_init_with_valid_templates():
    """Test initialization with valid templates."""
    templates = {
        "household": np.random.rand(5, 5),
        "school": np.random.rand(5, 5),
        "work": np.random.rand(5, 5),
        "community": np.random.rand(5, 5),
    }
    generator = MatrixGenerator(templates)
    assert generator is not None
    assert hasattr(generator, "templates")


def test_init_missing_required_template():
    """Test that ValueError is raised when required templates are missing."""
    # Missing 'community' template
    templates = {
        "household": np.random.rand(5, 5),
        "school": np.random.rand(5, 5),
        "work": np.random.rand(5, 5),
    }
    with pytest.raises(ValueError, match="Missing required templates"):
        MatrixGenerator(templates)


def test_init_with_non_dict_templates():
    """Test that TypeError is raised for non-dictionary templates."""
    with pytest.raises(TypeError, match="templates must be a dictionary"):
        MatrixGenerator([np.random.rand(5, 5)])


def test_init_with_mismatched_shapes():
    """Test that ValueError is raised when templates have different shapes."""
    templates = {
        "household": np.random.rand(5, 5),
        "school": np.random.rand(6, 6),  # Different shape
        "work": np.random.rand(5, 5),
        "community": np.random.rand(5, 5),
    }
    with pytest.raises(ValueError, match="All templates must have same shape"):
        MatrixGenerator(templates)


def test_init_with_non_square_matrices():
    """Test that ValueError is raised for non-square template matrices."""
    templates = {
        "household": np.random.rand(5, 6),  # Non-square
        "school": np.random.rand(5, 6),
        "work": np.random.rand(5, 6),
        "community": np.random.rand(5, 6),
    }
    with pytest.raises(ValueError, match="Templates must be square matrices"):
        MatrixGenerator(templates)


def test_template_normalization():
    """Test that templates are normalized correctly."""
    # Create templates with known values
    A = 4
    templates = {
        "household": np.ones((A, A)) * 2,
        "school": np.ones((A, A)) * 4,
        "work": np.ones((A, A)) * 6,
        "community": np.ones((A, A)) * 8,
    }

    generator = MatrixGenerator(templates)

    # Check that each normalized template has average marginal intensity of 1
    for name, template in generator.templates.items():
        mean_intensity = template.sum() / A
        assert np.isclose(
            mean_intensity, 1.0
        ), f"Template {name} not normalized correctly"


# ===== Single Matrix Generation Tests =====


def test_generate_single_basic():
    """Test basic single matrix generation."""
    templates = {
        "household": np.random.rand(5, 5),
        "school": np.random.rand(5, 5),
        "work": np.random.rand(5, 5),
        "community": np.random.rand(5, 5),
    }
    generator = MatrixGenerator(templates)
    age_dist = np.array([100, 200, 300, 400, 500])
    subgroup = Subgroup(n=1000, age_dist=age_dist, mean_cint_margin=10.0)

    M = generator.generate_single(subgroup, seed=42)

    # Check shape
    assert M.shape == (5, 5), "Matrix should be 5x5"

    # Check all values are non-negative
    assert np.all(M >= 0), "All values should be non-negative"


def test_generate_single_reciprocity():
    """Test that reciprocity condition PM = (PM)^T is satisfied."""
    templates = {
        "household": np.random.rand(5, 5),
        "school": np.random.rand(5, 5),
        "work": np.random.rand(5, 5),
        "community": np.random.rand(5, 5),
    }
    generator = MatrixGenerator(templates)
    age_dist = np.array([100, 200, 300, 400, 500])
    subgroup = Subgroup(n=1000, age_dist=age_dist, mean_cint_margin=10.0)

    M = generator.generate_single(subgroup, seed=42)
    P = np.diag(age_dist)

    # Check reciprocity: PM = (PM)^T
    PM = P @ M
    assert np.allclose(PM, PM.T), "Reciprocity condition PM = (PM)^T not satisfied"


def test_generate_single_reproducibility():
    """Test reproducibility with same seed."""
    templates = {
        "household": np.random.rand(5, 5),
        "school": np.random.rand(5, 5),
        "work": np.random.rand(5, 5),
        "community": np.random.rand(5, 5),
    }
    generator = MatrixGenerator(templates)
    age_dist = np.array([100, 200, 300, 400, 500])
    subgroup = Subgroup(n=1000, age_dist=age_dist, mean_cint_margin=10.0)

    M1 = generator.generate_single(subgroup, seed=123)
    M2 = generator.generate_single(subgroup, seed=123)

    assert np.allclose(M1, M2), "Same seed should produce identical matrices"


def test_generate_single_different_seeds():
    """Test that different seeds produce different results."""
    templates = {
        "household": np.random.rand(5, 5),
        "school": np.random.rand(5, 5),
        "work": np.random.rand(5, 5),
        "community": np.random.rand(5, 5),
    }
    generator = MatrixGenerator(templates)
    age_dist = np.array([100, 200, 300, 400, 500])
    subgroup = Subgroup(n=1000, age_dist=age_dist, mean_cint_margin=10.0)

    M1 = generator.generate_single(subgroup, seed=111)
    M2 = generator.generate_single(subgroup, seed=222)

    assert not np.allclose(M1, M2), "Different seeds should produce different matrices"


def test_generate_single_invalid_age_dist():
    """Test that errors are raised for invalid inputs."""
    templates = {
        "household": np.random.rand(5, 5),
        "school": np.random.rand(5, 5),
        "work": np.random.rand(5, 5),
        "community": np.random.rand(5, 5),
    }
    generator = MatrixGenerator(templates)

    # Test 2D array - will raise LinAlgError from np.diag
    with pytest.raises((ValueError, np.linalg.LinAlgError)):
        subgroup = Subgroup(
            n=1000, age_dist=np.random.rand(5, 5), mean_cint_margin=10.0
        )
        generator.generate_single(subgroup)

    # Test negative mean_cint_margin - results will be negative but no explicit check
    # Just verify it runs (may want to add validation in the future)
    subgroup = Subgroup(
        n=1000, age_dist=np.array([100, 200, 300, 400, 500]), mean_cint_margin=-10.0
    )
    M = generator.generate_single(subgroup, seed=42)
    assert M.mean() < 0, "Negative mean_cint_margin should produce negative values"


def test_generate_single_scaling():
    """Test that mean_cint_margin properly scales the matrix."""
    templates = {
        "household": np.ones((4, 4)),
        "school": np.ones((4, 4)),
        "work": np.ones((4, 4)),
        "community": np.ones((4, 4)),
    }
    generator = MatrixGenerator(templates)
    age_dist = np.array([100, 100, 100, 100])
    subgroup1 = Subgroup(n=1000, age_dist=age_dist, mean_cint_margin=1.0)
    subgroup2 = Subgroup(n=1000, age_dist=age_dist, mean_cint_margin=2.0)

    M1 = generator.generate_single(subgroup1, seed=42)
    M2 = generator.generate_single(subgroup2, seed=42)

    # M2 should be approximately twice M1 (before reciprocity adjustments)
    # At least check that M2 has larger values on average
    assert M2.mean() > M1.mean(), "Higher mean_cint_margin should produce larger values"


# ===== Partial Matrix Generation Tests =====


def test_generate_partial_basic():
    """Test basic partial matrix generation with subgroups."""
    templates = {
        "household": np.random.rand(4, 4),
        "school": np.random.rand(4, 4),
        "work": np.random.rand(4, 4),
        "community": np.random.rand(4, 4),
    }
    generator = MatrixGenerator(templates)

    subgroups = [
        Subgroup(
            n=500,
            age_dist=np.array([100, 200, 300, 400]),
            mean_cint_margin=15.0,
            label=0,
        ),
        Subgroup(
            n=500,
            age_dist=np.array([400, 300, 200, 100]),
            mean_cint_margin=20.0,
            label=1,
        ),
    ]

    matrices = generator.generate_partial(subgroups, seed=42)

    # Check that we have matrices for each subgroup
    assert len(matrices) == 2, "Should have 2 matrices for 2 subgroups"
    assert "0->All" in matrices, "Should have matrix for subgroup 0"
    assert "1->All" in matrices, "Should have matrix for subgroup 1"

    # Check shapes
    for label, M in matrices.items():
        assert M.shape == (4, 4), f"Matrix for subgroup {label} should be 4x4"


def test_generate_partial_reciprocity():
    """Test reciprocity condition for each subgroup in partial generation."""
    templates = {
        "household": np.random.rand(4, 4),
        "school": np.random.rand(4, 4),
        "work": np.random.rand(4, 4),
        "community": np.random.rand(4, 4),
    }
    generator = MatrixGenerator(templates)

    subgroups = [
        Subgroup(
            n=500,
            age_dist=np.array([100, 200, 300, 400]),
            mean_cint_margin=15.0,
            label=0,
        ),
        Subgroup(
            n=500,
            age_dist=np.array([400, 300, 200, 100]),
            mean_cint_margin=20.0,
            label=1,
        ),
    ]

    matrices = generator.generate_partial(subgroups, seed=42)

    # Check reciprocity for each subgroup
    for label, M in matrices.items():
        label_int = int(label.split("->")[0])
        P = np.diag(subgroups[label_int].age_dist)
        PM = P @ M
        assert np.allclose(PM, PM.T), f"Reciprocity not satisfied for subgroup {label}"


def test_generate_partial_reproducibility():
    """Test reproducibility with seed in partial generation."""
    templates = {
        "household": np.random.rand(4, 4),
        "school": np.random.rand(4, 4),
        "work": np.random.rand(4, 4),
        "community": np.random.rand(4, 4),
    }
    generator = MatrixGenerator(templates)

    subgroups = [
        Subgroup(
            n=500,
            age_dist=np.array([100, 200, 300, 400]),
            mean_cint_margin=15.0,
            label=0,
        ),
        Subgroup(
            n=500,
            age_dist=np.array([400, 300, 200, 100]),
            mean_cint_margin=20.0,
            label=1,
        ),
    ]

    matrices1 = generator.generate_partial(subgroups, seed=999)
    matrices2 = generator.generate_partial(subgroups, seed=999)

    for label in matrices1.keys():
        assert np.allclose(
            matrices1[label], matrices2[label]
        ), f"Matrices for subgroup {label} should be identical"


def test_generate_partial_single_subgroup():
    """Test partial generation with a single subgroup."""
    templates = {
        "household": np.random.rand(4, 4),
        "school": np.random.rand(4, 4),
        "work": np.random.rand(4, 4),
        "community": np.random.rand(4, 4),
    }
    generator = MatrixGenerator(templates)

    subgroups = [
        Subgroup(
            n=1000,
            age_dist=np.array([100, 200, 300, 400]),
            mean_cint_margin=15.0,
            label=0,
        )
    ]

    matrices = generator.generate_partial(subgroups, seed=42)

    assert len(matrices) == 1, "Should have 1 matrix for 1 subgroup"
    assert "0->All" in matrices, "Should have matrix for subgroup 0"
    assert matrices["0->All"].shape == (4, 4), "Matrix should be 4x4"


# ===== Full Matrix Generation Tests =====


def test_generate_full_basic():
    """Test basic full pairwise matrix generation."""
    templates = {
        "household": np.random.rand(4, 4),
        "school": np.random.rand(4, 4),
        "work": np.random.rand(4, 4),
        "community": np.random.rand(4, 4),
    }
    generator = MatrixGenerator(templates)

    subgroups = [
        Subgroup(
            n=500,
            age_dist=np.array([100, 200, 300, 400]),
            mean_cint_margin=15.0,
            label=0,
        ),
        Subgroup(
            n=500,
            age_dist=np.array([400, 300, 200, 100]),
            mean_cint_margin=20.0,
            label=1,
        ),
    ]

    matrices = generator.generate_full(subgroups, seed=42)

    # Should have 4 matrices: (0,0), (0,1), (1,0), (1,1)
    assert len(matrices) == 4, "Should have 4 matrices for 2 subgroups"
    assert "0->0" in matrices, "Should have matrix for 0->0"
    assert "0->1" in matrices, "Should have matrix for 0->1"
    assert "1->0" in matrices, "Should have matrix for 1->0"
    assert "1->1" in matrices, "Should have matrix for 1->1"

    # Check shapes
    for key, M in matrices.items():
        assert M.shape == (4, 4), f"Matrix for {key} should be 4x4"


def test_generate_full_within_subgroup_symmetry():
    """Test that within-subgroup matrices are symmetric."""
    templates = {
        "household": np.random.rand(4, 4),
        "school": np.random.rand(4, 4),
        "work": np.random.rand(4, 4),
        "community": np.random.rand(4, 4),
    }
    generator = MatrixGenerator(templates)

    subgroups = [
        Subgroup(
            n=500,
            age_dist=np.array([100, 200, 300, 400]),
            mean_cint_margin=15.0,
            label=0,
        ),
        Subgroup(
            n=500,
            age_dist=np.array([400, 300, 200, 100]),
            mean_cint_margin=20.0,
            label=1,
        ),
    ]

    matrices = generator.generate_full(subgroups, seed=42)

    # Check diagonal blocks are symmetric (within-subgroup reciprocity)
    for i in range(len(subgroups)):
        M_ii = matrices[f"{i}->{i}"]
        P_i = np.diag(subgroups[i].age_dist)
        PM_ii = P_i @ M_ii
        assert np.allclose(
            PM_ii, PM_ii.T
        ), f"Within-subgroup matrix ({i},{i}) should satisfy PM = PM^T"


def test_generate_full_between_subgroup_reciprocity():
    """Test reciprocity condition between different subgroups."""
    templates = {
        "household": np.random.rand(4, 4),
        "school": np.random.rand(4, 4),
        "work": np.random.rand(4, 4),
        "community": np.random.rand(4, 4),
    }
    generator = MatrixGenerator(templates)

    subgroups = [
        Subgroup(
            n=500,
            age_dist=np.array([100, 200, 300, 400]),
            mean_cint_margin=15.0,
            label=0,
        ),
        Subgroup(
            n=500,
            age_dist=np.array([400, 300, 200, 100]),
            mean_cint_margin=20.0,
            label=1,
        ),
    ]

    matrices = generator.generate_full(subgroups, seed=42)

    # Check between-subgroup reciprocity: P_k M_kl = (P_l M_lk)^T
    M_01 = matrices["0->1"]
    M_10 = matrices["1->0"]
    P_0 = np.diag(subgroups[0].age_dist)
    P_1 = np.diag(subgroups[1].age_dist)

    assert np.allclose(
        P_0 @ M_01, (P_1 @ M_10).T
    ), "Between-subgroup reciprocity not satisfied"


def test_generate_full_three_subgroups():
    """Test full generation with three subgroups."""
    templates = {
        "household": np.random.rand(3, 3),
        "school": np.random.rand(3, 3),
        "work": np.random.rand(3, 3),
        "community": np.random.rand(3, 3),
    }
    generator = MatrixGenerator(templates)

    subgroups = [
        Subgroup(
            n=300, age_dist=np.array([100, 200, 300]), mean_cint_margin=10.0, label=0
        ),
        Subgroup(
            n=300, age_dist=np.array([200, 300, 100]), mean_cint_margin=15.0, label=1
        ),
        Subgroup(
            n=300, age_dist=np.array([300, 100, 200]), mean_cint_margin=20.0, label=2
        ),
    ]

    matrices = generator.generate_full(subgroups, seed=42)

    # Should have 9 matrices (3x3)
    assert len(matrices) == 9, "Should have 9 matrices for 3 subgroups"

    # Check all combinations exist
    for i in range(3):
        for j in range(3):
            assert f"{i}->{j}" in matrices, f"Should have matrix for {i}->{j}"


def test_generate_full_reproducibility():
    """Test reproducibility in full generation."""
    templates = {
        "household": np.random.rand(4, 4),
        "school": np.random.rand(4, 4),
        "work": np.random.rand(4, 4),
        "community": np.random.rand(4, 4),
    }
    generator = MatrixGenerator(templates)

    subgroups = [
        Subgroup(
            n=500,
            age_dist=np.array([100, 200, 300, 400]),
            mean_cint_margin=15.0,
            label=0,
        ),
        Subgroup(
            n=500,
            age_dist=np.array([400, 300, 200, 100]),
            mean_cint_margin=20.0,
            label=1,
        ),
    ]

    matrices1 = generator.generate_full(subgroups, seed=777)
    matrices2 = generator.generate_full(subgroups, seed=777)

    for key in matrices1.keys():
        assert np.allclose(
            matrices1[key], matrices2[key]
        ), f"Matrices for {key} should be identical"


# ===== Contact Rate Matrix Tests =====


def test_get_contact_rate_matrix():
    """Test conversion from contact intensity to contact rate matrix."""
    templates = {
        "household": np.random.rand(4, 4),
        "school": np.random.rand(4, 4),
        "work": np.random.rand(4, 4),
        "community": np.random.rand(4, 4),
    }
    generator = MatrixGenerator(templates)

    age_dist = np.array([100, 200, 300, 400])
    subgroup = Subgroup(n=1000, age_dist=age_dist, mean_cint_margin=10.0)
    M = generator.generate_single(subgroup, seed=42)

    # Get contact rate matrix
    Gamma = generator.get_contact_rate_matrix(M, age_dist)

    # Check shape
    assert Gamma.shape == (4, 4), "Contact rate matrix should be 4x4"

    # Verify relationship: Γ = M P^{-1}
    P_inv = np.linalg.inv(np.diag(age_dist))
    expected_Gamma = M @ P_inv
    assert np.allclose(
        Gamma, expected_Gamma
    ), "Contact rate matrix calculation incorrect"


def test_get_contact_rate_matrix_reciprocity():
    """Test that contact rate matrix is computed correctly from intensity matrix."""
    templates = {
        "household": np.random.rand(4, 4),
        "school": np.random.rand(4, 4),
        "work": np.random.rand(4, 4),
        "community": np.random.rand(4, 4),
    }
    generator = MatrixGenerator(templates)

    age_dist = np.array([100, 200, 300, 400])
    subgroup = Subgroup(n=1000, age_dist=age_dist, mean_cint_margin=10.0)
    M = generator.generate_single(subgroup, seed=42)
    Gamma = generator.get_contact_rate_matrix(M, age_dist)

    P = np.diag(age_dist)

    # Since M satisfies PM = (PM)^T, and Γ = MP^{-1},
    # we should have M = Γ P (reconstruction property)
    M_reconstructed = Gamma @ P
    assert np.allclose(
        M, M_reconstructed
    ), "Should be able to reconstruct M from Γ and P"


# ===== Integration Tests =====


def test_with_real_data():
    """Test with real age distribution and template patterns."""
    df_age_dist = load_age_distribution("United_States", max_age=80)
    patterns = load_template_patterns("United_States", max_age=80)

    age_dist = df_age_dist["P"].values
    generator = MatrixGenerator(patterns)

    # Test generate_single
    subgroup = Subgroup(n=1000, age_dist=age_dist, mean_cint_margin=15.0)
    M = generator.generate_single(subgroup, seed=0)
    assert M.shape == (81, 81), "Matrix should be 81x81"

    # Check reciprocity
    P = np.diag(age_dist)
    PM = P @ M
    assert np.allclose(PM, PM.T), "Reciprocity should be satisfied"


def test_with_real_data_subgroups():
    """Test subgroup generation with real data."""
    df_age_dist = load_age_distribution("United_States", max_age=80)
    patterns = load_template_patterns("United_States", max_age=80)

    age_dist = df_age_dist["P"].values
    generator = MatrixGenerator(patterns)

    # Create subgroups
    subgroups = [
        Subgroup(n=1000, age_dist=age_dist, mean_cint_margin=15.0, label=0),
        Subgroup(n=1000, age_dist=age_dist, mean_cint_margin=20.0, label=1),
    ]

    # Test partial generation
    matrices_partial = generator.generate_partial(subgroups, seed=0)
    assert len(matrices_partial) == 2, "Should have 2 matrices"

    # Test full generation
    matrices_full = generator.generate_full(subgroups, seed=0)
    assert len(matrices_full) == 4, "Should have 4 matrices"
