from __future__ import annotations

import numpy as np
import pytest

from .._Stratification import Stratification

REF = np.array([100, 200, 150, 80, 50])  # 5 age groups


def test_default_labels_and_codes():
    s = Stratification(name="ses", n_strata=3, ref_age_dist=REF)
    assert s.labels == ["ses_0", "ses_1", "ses_2"]
    assert s.codes == [0, 1, 2]
    assert s.A == len(REF)


def test_custom_labels():
    labels = ["Low", "Medium", "High"]
    s = Stratification(name="ses", n_strata=3, ref_age_dist=REF, labels=labels)
    assert s.labels == labels
    assert s.codes == [0, 1, 2]


def test_alpha_defaults_and_custom():
    s_default = Stratification(name="ses", n_strata=2, ref_age_dist=REF, seed=0)
    assert 0.01 <= s_default.alpha <= 0.2

    s_custom = Stratification(name="ses", n_strata=2, ref_age_dist=REF, alpha=0.05)
    assert s_custom.alpha == 0.05


def test_lazy_generation():
    s = Stratification(name="ses", n_strata=2, ref_age_dist=REF, seed=42)
    assert s._P is None, "P should not be computed before first access"
    _ = s.P
    assert s._P is not None, "P should be computed after first access"
    _ = s.Q
    assert s._Q is not None, "Q should be computed after first access"


def test_output_shape_and_validity():
    n_strata = 3
    s = Stratification(name="ses", n_strata=n_strata, ref_age_dist=REF, seed=42)

    assert s.P.shape == (n_strata, len(REF))
    assert np.issubdtype(s.P.dtype, np.integer)
    assert (s.P >= 0).all()

    assert s.Q.shape == (n_strata, len(REF))
    assert (s.Q >= 0).all() and (s.Q <= 1).all()
    assert np.allclose(s.Q.sum(axis=0), 1.0)


def test_dataframe_columns_and_length():
    n_strata = 2
    labels = ["Male", "Female"]
    s = Stratification(name="gender", n_strata=n_strata, ref_age_dist=REF, labels=labels, seed=1)

    assert set(s.df_P.columns) == {"age", "gender", "P"}
    assert len(s.df_P) == n_strata * len(REF)
    assert set(s.df_P["gender"].unique()) == set(labels)

    assert set(s.df_Q.columns) == {"age", "gender", "Q"}
    assert len(s.df_Q) == n_strata * len(REF)
    assert set(s.df_Q["gender"].unique()) == set(labels)


def test_reproducibility():
    s1 = Stratification(name="ses", n_strata=2, ref_age_dist=REF, seed=99)
    s2 = Stratification(name="ses", n_strata=2, ref_age_dist=REF, seed=99)
    assert np.array_equal(s1.P, s2.P), "Same seed must produce identical P"

    s3 = Stratification(name="ses", n_strata=2, ref_age_dist=REF, seed=100)
    assert not np.array_equal(s1.P, s3.P), "Different seeds must produce different P"


def test_generate_resets_cache():
    s = Stratification(name="ses", n_strata=2, ref_age_dist=REF, seed=42)
    original_P = s.P.copy()
    _ = s.Q
    _ = s.df_P
    _ = s.df_Q

    s.generate(seed=999)

    assert s._Q is None, "Q cache must be cleared after generate()"
    assert s._df_P is None, "df_P cache must be cleared after generate()"
    assert s._df_Q is None, "df_Q cache must be cleared after generate()"
    assert not np.array_equal(s.P, original_P), "New seed must produce different P"


def test_lenscale_affects_output():
    s1 = Stratification(name="ses", n_strata=2, ref_age_dist=REF, lenscale=1.0, seed=42)
    s2 = Stratification(name="ses", n_strata=2, ref_age_dist=REF, lenscale=10.0, seed=42)
    assert not np.array_equal(s1.P, s2.P), "Different lenscale must produce different P"
