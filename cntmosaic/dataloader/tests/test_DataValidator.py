import numpy as np
import pandas as pd
import pytest

from .._DataValidator import DataValidator
from ..containers import (
    ContactData,
    ParticipantData,
    PopulationData,
    StratificationData,
)

# ================
# Helper functions
# ================


def check_categorical_coding_consistency(
    cat1: pd.Categorical, cat2: pd.Categorical
) -> bool:
    """
    Check if two categorical variables have consistent coding.

    This checks that:
    1. Categories are in the same order
    2. Each category maps to the same code in both categoricals

    Parameters
    ----------
    cat1, cat2 : pd.Categorical
        Categorical variables to compare

    Returns
    -------
    bool
        True if coding is consistent, False otherwise

    Examples
    --------
    >>> cat1 = pd.Categorical(['M', 'F', 'M'], categories=['M', 'F'])
    >>> cat2 = pd.Categorical(['M', 'F', 'M'], categories=['M', 'F'])
    >>> check_categorical_coding_consistency(cat1, cat2)
    True

    >>> cat3 = pd.Categorical(['M', 'F', 'M'], categories=['F', 'M'])  # Different order!
    >>> check_categorical_coding_consistency(cat1, cat3)
    False
    """
    # Check if categories are identical and in same order
    if not cat1.categories.equals(cat2.categories):
        return False

    # For each category, verify it maps to the same code
    for cat_value in cat1.categories:
        # Get the code for this category in both categoricals
        code1 = cat1.categories.get_loc(cat_value)
        code2 = cat2.categories.get_loc(cat_value)
        if code1 != code2:
            return False

    return True


# ================
# Define fixtures
# ================


@pytest.fixture
def data_no_strat():
    df_part = pd.DataFrame(
        {
            "id": [1, 2, 3],
            "age": [0, 1, 2],
        }
    )
    df_cnt = pd.DataFrame(
        {
            "id": [1, 1, 2, 2, 3, 3],
            "age_cnt": [0, 1, 2, 0, 1, 2],
            "y": [1, 1, 1, 1, 1, 1],
        }
    )
    df_pop = pd.DataFrame(
        {
            "age": [0, 1, 2],
            "P": [1000, 2000, 3000],
        }
    )

    part_data = ParticipantData(df_part, "id", "age")
    cnt_data = ContactData(df_cnt, "id", "age_cnt", cnt_col="y")
    pop_data = PopulationData(df_pop, "age", "P")

    return part_data, cnt_data, pop_data


@pytest.fixture
def data_partial_single():
    df_part = pd.DataFrame(
        {
            "id": [1, 2, 3],
            "age": [0, 1, 2],
            "sex": ["M", "F", "M"],
        }
    )
    df_part["sex"] = pd.Categorical(df_part["sex"], categories=["M", "F"])

    df_cnt = pd.DataFrame(
        {
            "id": [1, 1, 2, 2, 3, 3],
            "age_cnt": [0, 1, 2, 0, 1, 2],
            "y": [1, 1, 1, 1, 1, 1],
        }
    )
    df_pop = pd.DataFrame(
        {
            "age": [0, 1, 2],
            "P": [1000, 2000, 3000],
        }
    )
    df_strat = pd.DataFrame(
        {
            "age": [0, 0, 1, 1, 2, 2],
            "sex": ["M", "F", "M", "F", "M", "F"],
            "prop": [0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
        }
    )
    df_strat["sex"] = pd.Categorical(df_strat["sex"], categories=["M", "F"])

    part_data = ParticipantData(df_part, "id", "age", strat_var_cols="sex")
    cnt_data = ContactData(df_cnt, "id", "age_cnt", cnt_col="y")
    pop_data = PopulationData(df_pop, "age", "P")
    strat_data = StratificationData(df_strat, "age", "sex", "prop")

    return part_data, cnt_data, pop_data, strat_data


@pytest.fixture
def data_partial_single_inconsistent_coding():
    df_part = pd.DataFrame(
        {
            "id": [1, 2, 3],
            "age": [0, 1, 2],
            "sex": ["M", "F", "M"],
        }
    )
    df_part["sex"] = pd.Categorical(df_part["sex"], categories=["M", "F"])

    df_cnt = pd.DataFrame(
        {
            "id": [1, 1, 2, 2, 3, 3],
            "age_cnt": [0, 1, 2, 0, 1, 2],
            "y": [1, 1, 1, 1, 1, 1],
        }
    )
    df_pop = pd.DataFrame(
        {
            "age": [0, 1, 2],
            "P": [1000, 2000, 3000],
        }
    )
    df_strat = pd.DataFrame(
        {
            "age": [0, 0, 1, 1, 2, 2],
            "sex": ["M", "F", "M", "F", "M", "F"],
            "prop": [0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
        }
    )
    df_strat["sex"] = pd.Categorical(df_strat["sex"], categories=["F", "M"])

    part_data = ParticipantData(df_part, "id", "age", strat_var_cols="sex")
    cnt_data = ContactData(df_cnt, "id", "age_cnt", cnt_col="y")
    pop_data = PopulationData(df_pop, "age", "P")
    strat_data = StratificationData(df_strat, "age", "sex", "prop")

    return part_data, cnt_data, pop_data, strat_data


@pytest.fixture
def data_partial_multi():
    df_part = pd.DataFrame(
        {
            "id": [1, 2, 3],
            "age": [0, 1, 2],
            "sex": ["M", "F", "M"],
            "hhsize": ["1", "2", "1"],
        }
    )
    df_part["sex"] = pd.Categorical(df_part["sex"], categories=["M", "F"])
    df_part["hhsize"] = pd.Categorical(df_part["hhsize"], categories=["1", "2"])

    df_cnt = pd.DataFrame(
        {
            "id": [1, 1, 2, 2, 3, 3],
            "age_cnt": [0, 1, 2, 0, 1, 2],
            "y": [1, 1, 1, 1, 1, 1],
        }
    )
    df_pop = pd.DataFrame(
        {
            "age": [0, 1, 2],
            "P": [1000, 2000, 3000],
        }
    )
    df_strat = pd.DataFrame(
        {
            "age": [0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2],
            "sex": ["M", "M", "F", "F", "M", "M", "F", "F", "M", "M", "F", "F"],
            "hhsize": ["1", "2", "1", "2", "1", "2", "1", "2", "1", "2", "1", "2"],
            "prop": np.ones(12) / 4,
        }
    )
    df_strat["sex"] = pd.Categorical(df_strat["sex"], categories=["M", "F"])
    df_strat["hhsize"] = pd.Categorical(df_strat["hhsize"], categories=["1", "2"])

    part_data = ParticipantData(df_part, "id", "age", strat_var_cols=["sex", "hhsize"])
    cnt_data = ContactData(df_cnt, "id", "age_cnt", cnt_col="y")
    pop_data = PopulationData(df_pop, "age", "P")
    strat_data = StratificationData(df_strat, "age", ["sex", "hhsize"], "prop")

    return part_data, cnt_data, pop_data, strat_data


@pytest.fixture
def data_full_single():
    df_part = pd.DataFrame(
        {
            "id": [1, 2, 3],
            "age": [0, 1, 2],
            "sex": ["M", "F", "M"],
        }
    )
    df_part["sex"] = pd.Categorical(df_part["sex"], categories=["M", "F"])

    df_cnt = pd.DataFrame(
        {
            "id": [1, 1, 2, 2, 3, 3],
            "age_cnt": [0, 1, 2, 0, 1, 2],
            "sex_cnt": ["M", "M", "F", "F", "M", "M"],
            "y": [1, 1, 1, 1, 1, 1],
        }
    )
    df_cnt["sex_cnt"] = pd.Categorical(df_cnt["sex_cnt"], categories=["M", "F"])

    df_pop = pd.DataFrame(
        {
            "age": [0, 0, 1, 1, 2, 2],
            "sex": ["M", "F", "M", "F", "M", "F"],
            "P": [1000, 2000, 3000, 1000, 2000, 3000],
        }
    )
    df_pop["sex"] = pd.Categorical(df_pop["sex"], categories=["M", "F"])

    df_strat = pd.DataFrame(
        {
            "age": [0, 0, 1, 1, 2, 2],
            "sex": ["M", "F", "M", "F", "M", "F"],
            "prop": [0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
        }
    )
    df_strat["sex"] = pd.Categorical(df_strat["sex"], categories=["M", "F"])

    part_data = ParticipantData(df_part, "id", "age", strat_var_cols="sex")
    cnt_data = ContactData(
        df_cnt, "id", "age_cnt", strat_var_cols="sex_cnt", cnt_col="y"
    )
    pop_data = PopulationData(df_pop, "age", "P", strat_var_cols="sex")
    strat_data = StratificationData(df_strat, "age", "sex", "prop")

    return part_data, cnt_data, pop_data, strat_data


@pytest.fixture
def data_full_single_inconsistent_coding():
    df_part = pd.DataFrame(
        {
            "id": [1, 2, 3],
            "age": [0, 1, 2],
            "sex": ["M", "F", "M"],
        }
    )
    df_part["sex"] = pd.Categorical(df_part["sex"], categories=["M", "F"])

    df_cnt = pd.DataFrame(
        {
            "id": [1, 1, 2, 2, 3, 3],
            "age_cnt": [0, 1, 2, 0, 1, 2],
            "sex_cnt": ["M", "M", "F", "F", "M", "M"],
            "y": [1, 1, 1, 1, 1, 1],
        }
    )
    df_cnt["sex_cnt"] = pd.Categorical(df_cnt["sex_cnt"], categories=["F", "M"])

    df_pop = pd.DataFrame(
        {
            "age": [0, 0, 1, 1, 2, 2],
            "sex": ["M", "F", "M", "F", "M", "F"],
            "P": [1000, 2000, 3000, 1000, 2000, 3000],
        }
    )
    df_pop["sex"] = pd.Categorical(df_pop["sex"], categories=["F", "M"])

    df_strat = pd.DataFrame(
        {
            "age": [0, 0, 1, 1, 2, 2],
            "sex": ["M", "F", "M", "F", "M", "F"],
            "prop": [0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
        }
    )
    df_strat["sex"] = pd.Categorical(df_strat["sex"], categories=["F", "M"])

    part_data = ParticipantData(df_part, "id", "age", strat_var_cols="sex")
    cnt_data = ContactData(
        df_cnt, "id", "age_cnt", strat_var_cols="sex_cnt", cnt_col="y"
    )
    pop_data = PopulationData(df_pop, "age", "P", strat_var_cols="sex")
    strat_data = StratificationData(df_strat, "age", "sex", "prop")

    return part_data, cnt_data, pop_data, strat_data


class TestValid:
    # The tests below should not raise any errors or exceptions

    def test_no_strat(self, data_no_strat):
        part_data, cnt_data, pop_data = data_no_strat

        validator = DataValidator(
            part_data=part_data, cnt_data=cnt_data, pop_data=pop_data
        )
        validator.validate()

    def test_partial_single(self, data_partial_single):
        part_data, cnt_data, pop_data, strat_data = data_partial_single

        validator = DataValidator(
            part_data=part_data,
            cnt_data=cnt_data,
            pop_data=pop_data,
            strat_data=strat_data,
        )
        validator.validate()

    def test_partial_multi(self, data_partial_multi):
        part_data, cnt_data, pop_data, strat_data = data_partial_multi

        validator = DataValidator(
            part_data=part_data,
            cnt_data=cnt_data,
            pop_data=pop_data,
            strat_data=strat_data,
        )
        validator.validate()

    def test_full_single(self, data_full_single):
        part_data, cnt_data, pop_data, strat_data = data_full_single

        validator = DataValidator(
            part_data=part_data,
            cnt_data=cnt_data,
            pop_data=pop_data,
            strat_data=strat_data,
        )
        validator.validate()


class TestInconsistentCoding:
    # The test below checks that inconsistent categorical coding is corrected

    def test_partial_single(self, data_partial_single_inconsistent_coding):
        part_data, cnt_data, pop_data, strat_data = (
            data_partial_single_inconsistent_coding
        )

        # Before validation: check that codes are inconsistent
        # part_data: M=0, F=1 (categories=['M', 'F'])
        # strat_data: F=0, M=1 (categories=['F', 'M'])
        part_sex_cat = part_data.data["sex_part"].cat
        strat_sex_cat = strat_data.data["sex"].cat

        # Verify initial inconsistency
        assert part_sex_cat.categories.to_list() == ["M", "F"]
        assert strat_sex_cat.categories.to_list() == ["F", "M"]
        assert not check_categorical_coding_consistency(part_sex_cat, strat_sex_cat)

        # Run validation
        validator = DataValidator(
            part_data=part_data,
            cnt_data=cnt_data,
            pop_data=pop_data,
            strat_data=strat_data,
        )
        part_data, _, _, strat_data = validator.validate()

        # After validation: check that categories AND codes are consistent
        part_sex_cat = part_data.data["sex_part"].cat
        strat_sex_cat = strat_data.data["sex"].cat

        # Categories should match
        assert part_sex_cat.categories.to_list() == strat_sex_cat.categories.to_list()

        # Codes should now be consistent using our helper function
        assert check_categorical_coding_consistency(part_sex_cat, strat_sex_cat)

        # Verify codes are now consistent
        assert part_sex_cat.categories.to_list() == ["M", "F"]
        assert strat_sex_cat.categories.to_list() == ["M", "F"]

        # Get actual data values to verify code consistency
        part_m_codes = part_data.data[part_data.data["sex_part"] == "M"][
            "sex_part"
        ].cat.codes
        strat_m_codes = strat_data.data[strat_data.data["sex"] == "M"]["sex"].cat.codes

        # All 'M' values should have the same code in both datasets (code 0)
        assert (part_m_codes == 0).all()
        assert (strat_m_codes == 0).all()

    def test_full_single(self, data_full_single_inconsistent_coding):
        part_data, cnt_data, pop_data, strat_data = data_full_single_inconsistent_coding

        # Before validation: check that codes are inconsistent
        part_sex_cat = part_data.data["sex_part"].cat
        cnt_sex_cat = cnt_data.data["sex_cnt"].cat
        pop_sex_cat = pop_data.data["sex"].cat
        strat_sex_cat = strat_data.data["sex"].cat

        # Verify initial inconsistency
        assert part_sex_cat.categories.to_list() == ["M", "F"]
        assert cnt_sex_cat.categories.to_list() == ["F", "M"]
        assert pop_sex_cat.categories.to_list() == ["F", "M"]
        assert strat_sex_cat.categories.to_list() == ["F", "M"]
        assert not check_categorical_coding_consistency(part_sex_cat, strat_sex_cat)
        assert not check_categorical_coding_consistency(part_sex_cat, cnt_sex_cat)
        assert not check_categorical_coding_consistency(part_sex_cat, pop_sex_cat)

        # Run validation
        validator = DataValidator(
            part_data=part_data,
            cnt_data=cnt_data,
            pop_data=pop_data,
            strat_data=strat_data,
        )
        part_data, cnt_data, pop_data, strat_data = validator.validate()

        # After validation: check that categories AND codes are consistent
        part_sex_cat = part_data.data["sex_part"].cat
        cnt_sex_cat = cnt_data.data["sex_cnt"].cat
        pop_sex_cat = pop_data.data["sex"].cat
        strat_sex_cat = strat_data.data["sex"].cat

        # Categories should match
        assert part_sex_cat.categories.to_list() == cnt_sex_cat.categories.to_list()
        assert part_sex_cat.categories.to_list() == pop_sex_cat.categories.to_list()
        assert part_sex_cat.categories.to_list() == strat_sex_cat.categories.to_list()

        # Codes should now be consistent using our helper function
        assert check_categorical_coding_consistency(part_sex_cat, cnt_sex_cat)
        assert check_categorical_coding_consistency(part_sex_cat, pop_sex_cat)
        assert check_categorical_coding_consistency(part_sex_cat, strat_sex_cat)

        # Verify codes are now consistent
        assert part_sex_cat.categories.to_list() == ["M", "F"]
        assert cnt_sex_cat.categories.to_list() == ["M", "F"]
        assert strat_sex_cat.categories.to_list() == ["M", "F"]

        # Get actual data values to verify code consistency
        part_m_codes = part_data.data[part_data.data["sex_part"] == "M"][
            "sex_part"
        ].cat.codes
        cnt_m_codes = cnt_data.data[cnt_data.data["sex_cnt"] == "M"][
            "sex_cnt"
        ].cat.codes
        pop_m_codes = pop_data.data[pop_data.data["sex"] == "M"]["sex"].cat.codes
        strat_m_codes = strat_data.data[strat_data.data["sex"] == "M"]["sex"].cat.codes

        # All 'M' values should have the same code in both datasets (code 0)
        assert (part_m_codes == 0).all()
        assert (cnt_m_codes == 0).all()
        assert (pop_m_codes == 0).all()
        assert (strat_m_codes == 0).all()
