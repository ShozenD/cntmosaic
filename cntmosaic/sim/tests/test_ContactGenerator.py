import pytest

from ...datasets._base import load_age_distribution, load_template_patterns
from .._ParticipantGenerator import ParticipantGenerator, Subgroup
from .._MatrixGenerator import MatrixGenerator
from .._ContactGenerator import ContactGenerator


def test_basic_functionality():
    df_age_dist = load_age_distribution("United_States", max_age=80)
    patterns = load_template_patterns("United_States", max_age=80)
    age_dist = df_age_dist["P"].values

    # ===== Single subgroup ======
    subgroup = Subgroup(n=1000, age_dist=age_dist, mean_cint_margin=15.0)
    df_part = ParticipantGenerator(subgroup).generate(seed=0)
    cint_matrix = MatrixGenerator(patterns).generate_single(subgroup, seed=0)

    cg = ContactGenerator(df_part, cint_matrix)
    df_cnt = cg.generate()

    # Check the shape of the generated DataFrame (The first dimension is random)
    assert df_cnt.shape[1] == 3

    # ===== Multiple subgroups ======
    subgroups = [
        Subgroup(n=1000, age_dist=age_dist, mean_cint_margin=15.0, label=0),
        Subgroup(n=2000, age_dist=age_dist, mean_cint_margin=20.0, label=1),
    ]
    df_part = ParticipantGenerator(subgroups).generate(seed=0)
    cmg = MatrixGenerator(patterns)
    cint_matrices = {i: cmg.generate_single(subgroups[i], seed=i) for i in range(2)}
    cg = ContactGenerator(df_part, cint_matrices)
    df_cnt = cg.generate(seed=0)

    # Check the shape of the generated DataFrame (The first dimension is random)
    # Multiple subgroups should still have 3 columns (id, age_group, contacts)
    assert df_cnt.shape[1] == 3
