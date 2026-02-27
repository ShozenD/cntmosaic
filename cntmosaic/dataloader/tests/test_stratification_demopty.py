import numpy as np
import pandas as pd
from numpy.testing import assert_allclose

from cntmosaic._types import StratMode
from cntmosaic.dataloader.containers._StratificationData import StratificationData


def test_single_full_variable_marginal_equals_full():
    # Simple synthetic population proportions for two ages and two strata
    df = pd.DataFrame(
        {
            "age": [0, 0, 1, 1],
            "sex": ["A", "B", "A", "B"],
            "prop": [0.6, 0.4, 0.7, 0.3],
        }
    )

    strat_data = StratificationData(
        data=df, age_col="age", strat_var_cols="sex", prop_col="prop"
    )

    strat_modes = {"sex": StratMode.FULL}

    marginal = strat_data.compute_marginal_demopty(strat_modes)
    demopty = strat_data.compute_demopty(strat_modes)

    assert "sex" in marginal
    marg_mat = marginal["sex"]

    # Shapes should match (n_strata^2, A, A)
    assert marg_mat.shape == demopty.shape

    # Values should be identical for the single FULL stratification variable case
    assert_allclose(marg_mat, demopty, atol=1e-12)
