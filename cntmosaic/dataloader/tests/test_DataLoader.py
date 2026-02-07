import numpy as np
import pandas as pd
import pytest

from ..._types import StratMode
from ...datasets import load_age_distribution, load_template_patterns
from ...sim import (
    ContactGenerator,
    MatrixGenerator,
    ParticipantGenerator,
    PopulationConstructor,
    Stratification,
)
from .._DataLoader import DataLoader
from ..containers import (
    ContactData,
    ParticipantData,
    PopulationData,
    StratificationData,
)
from ..containers._ModelData import ModelData

SEED = 42
templates = load_template_patterns("United_States")
df_age_dist = load_age_distribution("United_States")


@pytest.fixture
def data_single():
    """Generate contact data for a single stratification."""
    strata = Stratification(
        name="sex",
        n_strata=2,
        ref_age_dist=df_age_dist["P"].values,
        labels=["M", "F"],
        seed=SEED,
    )

    pop_const = PopulationConstructor(strata)
    df_pop = pop_const.df_P

    # Generate participants
    part_gen = ParticipantGenerator(popcon=pop_const, n_part=1000)
    df_part = part_gen.generate(SEED)

    # Generate contact matrix
    matrix_gen = MatrixGenerator(templates)
    contact_matrix = matrix_gen.generate_single(pop_const, SEED)

    # Generate contacts
    cnt_gen = ContactGenerator(df_part, contact_matrix, random_effects=True)
    df_cnt = cnt_gen.generate(SEED)

    return df_part, df_cnt, df_pop


@pytest.fixture
def data_partial():
    """
    Generate contact data for the partial case (multiple subgroups, incomplete contact information)
    """

    # Define subgroups
    strats = [
        Stratification(
            name="sex",
            n_strata=2,
            ref_age_dist=df_age_dist["P"].values,
            labels=["M", "F"],
            seed=SEED,
        ),
        Stratification(
            name="hhsize",
            n_strata=5,
            ref_age_dist=df_age_dist["P"].values,
            labels=["1", "2", "3", "4", "5+"],
            seed=SEED,
        ),
    ]
    popcon = PopulationConstructor(strats)
    df_pop = popcon.df_P

    # Generate participants
    part_gen = ParticipantGenerator(popcon, n_part=1000)
    df_part = part_gen.generate(SEED)
    df_part["sex"] = pd.Categorical(df_part["sex"], categories=["M", "F"], ordered=True)
    df_part["hhsize"] = pd.Categorical(
        df_part["hhsize"], categories=["1", "2", "3", "4", "5+"], ordered=True
    )

    # Generate contact matrix
    matrix_gen = MatrixGenerator(templates)
    contact_matrices = matrix_gen.generate_partial(popcon, SEED)

    # Generate contacts
    cnt_gen = ContactGenerator(df_part, contact_matrices)
    df_cnt = cnt_gen.generate(SEED)

    # Population size offsets
    df_strat = popcon.df_Q
    df_strat["sex"] = pd.Categorical(
        df_strat["sex"], categories=["M", "F"], ordered=True
    )
    df_strat["hhsize"] = pd.Categorical(
        df_strat["hhsize"], categories=["1", "2", "3", "4", "5+"], ordered=True
    )

    return df_part, df_cnt, df_pop, df_strat


@pytest.fixture
def data_full():
    """
    Generate contact data for the full case (multiple subgroups, complete contact information)
    """

    # Define subgroups
    strats = [
        Stratification(
            name="sex",
            n_strata=2,
            ref_age_dist=df_age_dist["P"].values,
            labels=["M", "F"],
            seed=SEED,
        ),
        Stratification(
            name="hhsize",
            n_strata=5,
            ref_age_dist=df_age_dist["P"].values,
            labels=["1", "2", "3", "4", "5+"],
            seed=SEED,
        ),
    ]

    popcon = PopulationConstructor(strats)
    df_pop = popcon.df_P
    df_pop["sex"] = pd.Categorical(df_pop["sex"], categories=["M", "F"], ordered=True)
    df_pop["hhsize"] = pd.Categorical(
        df_pop["hhsize"], categories=["1", "2", "3", "4", "5+"], ordered=True
    )

    # Generate participants
    part_gen = ParticipantGenerator(popcon, n_part=1000)
    df_part = part_gen.generate(SEED)
    df_part["sex"] = pd.Categorical(df_part["sex"], categories=["M", "F"], ordered=True)
    df_part["hhsize"] = pd.Categorical(
        df_part["hhsize"], categories=["1", "2", "3", "4", "5+"], ordered=True
    )

    # Generate contact matrix
    matrix_gen = MatrixGenerator(templates)
    contact_matrices = matrix_gen.generate_full(popcon, SEED)

    # Generate contacts
    cnt_gen = ContactGenerator(df_part, contact_matrices)
    df_cnt = cnt_gen.generate(SEED)
    df_cnt["sex_cnt"] = pd.Categorical(
        df_cnt["sex_cnt"], categories=["M", "F"], ordered=True
    )
    df_cnt["hhsize_cnt"] = pd.Categorical(
        df_cnt["hhsize_cnt"], categories=["1", "2", "3", "4", "5+"], ordered=True
    )

    # Population size offsets
    df_strat = popcon.df_Q
    df_strat["sex"] = pd.Categorical(
        df_strat["sex"], categories=["M", "F"], ordered=True
    )
    df_strat["hhsize"] = pd.Categorical(
        df_strat["hhsize"], categories=["1", "2", "3", "4", "5+"], ordered=True
    )

    return df_part, df_cnt, df_pop, df_strat


# =================================
# Tests
# =================================
class TestSingle:

    def test_fine_age(self, data_single):
        df_part, df_cnt, df_pop = data_single

        # Create dataclass objects with standardized column names
        part_data = ParticipantData(df_part=df_part, id_col="id", age_col="age")
        cnt_data = ContactData(df_cnt=df_cnt, id_col="id", age_col="age_cnt")
        pop_data = PopulationData(df_pop=df_pop, age_col="age", size_col="P")
        dataloader = DataLoader(part_data, cnt_data, pop_data)

        model_data = dataloader.load()

        # Test if anticipated types
        assert isinstance(model_data, ModelData)
        assert isinstance(model_data.base_data["y"], np.ndarray)
        assert isinstance(model_data.base_data["aid"], np.ndarray)
        assert isinstance(model_data.base_data["bid"], np.ndarray)
        assert isinstance(model_data.base_data["log_N"], np.ndarray)
        assert isinstance(model_data.base_data["log_P"], np.ndarray)
        assert isinstance(model_data.base_data["log_V"], np.ndarray)

        # Test shapes
        len_y = len(model_data.base_data["y"])
        len_aid = len(model_data.base_data["aid"])
        len_bid = len(model_data.base_data["bid"])
        len_log_N = len(model_data.base_data["log_N"])
        len_log_V = len(model_data.base_data["log_V"])
        assert len_y == len_aid == len_bid == len_log_N == len_log_V

        # No stratication configuration
        assert model_data.strat_data == {}

    def test_repeat_effect(self, data_single):
        df_part, df_cnt, df_pop = data_single
        df_part["repeat"] = np.random.randint(0, 10, size=len(df_part))

        # Create dataclass objects with repeat column
        part_data = ParticipantData(df_part, "id", "age", repeat_col="repeat")
        cnt_data = ContactData(df_cnt=df_cnt, id_col="id", age_col="age_cnt")
        pop_data = PopulationData(df_pop=df_pop, age_col="age", size_col="P")
        dataloader = DataLoader(part_data, cnt_data, pop_data)

        model_data = dataloader.load()

        assert isinstance(model_data.base_data["rid"], np.ndarray)
        assert model_data.base_data["rid"].shape == model_data.base_data["y"].shape

        # No stratication configuration
        assert model_data.strat_data == {}

    def test_age_grp_cnt(self, data_single):
        df_part, df_cnt, df_pop = data_single

        # Create age groups for contacts
        bins = [0, 5, 15, 25, 35, 45, 55, 65, 75, 80]
        df_cnt["age_grp_cnt"] = pd.cut(df_cnt["age_cnt"], bins=bins, right=False)
        df_cnt["age_grp_cnt"] = pd.Categorical(df_cnt["age_grp_cnt"])

        # Create dataclass objects with age groups
        part_data = ParticipantData(df_part, "id", "age")
        cnt_data = ContactData(df_cnt=df_cnt, id_col="id", age_grp_col="age_grp_cnt")
        pop_data = PopulationData(df_pop=df_pop, age_col="age", size_col="P")
        dataloader = DataLoader(part_data, cnt_data, pop_data)

        model_data = dataloader.load()

        assert isinstance(model_data.base_data["y"], np.ndarray)
        assert isinstance(model_data.base_data["log_N"], np.ndarray)
        assert isinstance(model_data.base_data["log_P"], np.ndarray)
        assert isinstance(model_data.base_data["log_V"], np.ndarray)
        assert isinstance(model_data.base_data["aid_exp"], np.ndarray)
        assert isinstance(model_data.base_data["bid_pad"], np.ndarray)

        assert (
            model_data.base_data["aid_exp"].shape[0]
            == model_data.base_data["bid_pad"].shape[0]
        )
        assert (
            model_data.base_data["y"].shape[0]
            == model_data.base_data["aid_exp"].shape[0]
        )
        assert (
            model_data.base_data["log_N"].shape == model_data.base_data["log_V"].shape
        )


class TestPartial:

    def test_fine_age(self, data_partial):
        df_part, df_cnt, df_pop, df_strat = data_partial

        print(df_strat)

        # Create dataclass objects with stratification
        part_data = ParticipantData(
            df_part=df_part,
            id_col="id",
            age_col="age",
            strat_var_cols=["sex", "hhsize"],
        )
        cnt_data = ContactData(df_cnt=df_cnt, id_col="id", age_col="age_cnt")
        pop_data = PopulationData(df_pop=df_pop, age_col="age", size_col="P")
        strat_prop_data = StratificationData(
            data=df_strat,
            age_col="age",
            strat_var_cols=["sex", "hhsize"],
            prop_col="Q",
        )

        dataloader = DataLoader(part_data, cnt_data, pop_data, strat_prop_data)

        model_data = dataloader.load()

        assert isinstance(model_data, ModelData)
        assert isinstance(model_data.base_data["y"], np.ndarray)
        assert isinstance(model_data.base_data["aid"], np.ndarray)
        assert isinstance(model_data.base_data["bid"], np.ndarray)
        assert isinstance(model_data.base_data["log_N"], np.ndarray)
        assert isinstance(model_data.base_data["log_P"], np.ndarray)
        assert isinstance(model_data.base_data["log_V"], np.ndarray)

        # Test shapes
        assert model_data.base_data["aid"].shape == model_data.base_data["bid"].shape
        assert model_data.base_data["y"].shape == model_data.base_data["aid"].shape
        assert (
            model_data.base_data["log_N"].shape == model_data.base_data["log_V"].shape
        )

    def test_repeat_effect(self, data_partial):
        df_part, df_cnt, df_pop, df_strat = data_partial
        df_part["repeat"] = np.random.randint(0, 10, size=len(df_part))

        # Create dataclass objects with stratification and repeat
        part_data = ParticipantData(
            df_part=df_part,
            id_col="id",
            age_col="age",
            strat_var_cols=["sex", "hhsize"],
            repeat_col="repeat",
        )

        cnt_data = ContactData(df_cnt=df_cnt, id_col="id", age_col="age_cnt")

        pop_data = PopulationData(df_pop=df_pop, age_col="age", size_col="P")

        pop_prop = StratificationData(
            data=df_strat,
            age_col="age",
            strat_var_cols=["sex", "hhsize"],
            prop_col="Q",
        )

        dataloader = DataLoader(part_data, cnt_data, pop_data, pop_prop)

        model_data = dataloader.load()

        # Test that rid exists and has correct shape
        assert isinstance(model_data.base_data["rid"], np.ndarray)
        assert model_data.base_data["rid"].shape == model_data.base_data["aid"].shape

        # Should have stratification data
        assert model_data.strat_data is not None
        assert model_data.strat_data["modes"].keys() == {"sex", "hhsize"}
        assert model_data.strat_data["modes"] == {
            "sex": StratMode.PARTIAL,
            "hhsize": StratMode.PARTIAL,
        }
        assert model_data.strat_data["labels"]["sex"] == ["M->All", "F->All"]
        assert model_data.strat_data["labels"]["hhsize"] == [
            "1->All",
            "2->All",
            "3->All",
            "4->All",
            "5+->All",
        ]
        assert model_data.strat_data["ixs"].keys() == {"sex", "hhsize"}
        assert model_data.strat_data["ixs"]["sex"] is not None

    def test_age_grp_cnt(self, data_partial):
        df_part, df_cnt, df_pop, df_strat = data_partial

        # Create age groups for contacts
        bins = [0, 5, 15, 25, 35, 45, 55, 65, 75, 80]
        df_cnt["age_grp_cnt"] = pd.cut(df_cnt["age_cnt"], bins=bins, right=False)

        # Create dataclass objects with age groups and stratification
        part_data = ParticipantData(
            df_part=df_part,
            id_col="id",
            age_col="age",
            strat_var_cols=["sex", "hhsize"],
        )

        cnt_data = ContactData(df_cnt=df_cnt, id_col="id", age_grp_col="age_grp_cnt")
        pop_data = PopulationData(df_pop=df_pop, age_col="age", size_col="P")
        pop_prop = StratificationData(
            data=df_strat,
            age_col="age",
            strat_var_cols=["sex", "hhsize"],
            prop_col="Q",
        )
        dataloader = DataLoader(part_data, cnt_data, pop_data, pop_prop)

        model_data = dataloader.load()

        assert isinstance(model_data, ModelData)
        assert isinstance(model_data.base_data["aid_exp"], np.ndarray)
        assert isinstance(model_data.base_data["bid_pad"], np.ndarray)

        # Should have stratification data
        assert model_data.strat_data is not None
        assert model_data.strat_data["modes"].keys() == {"sex", "hhsize"}
        assert model_data.strat_data["labels"]["sex"] == ["M->All", "F->All"]
        assert model_data.strat_data["modes"] == {
            "sex": StratMode.PARTIAL,
            "hhsize": StratMode.PARTIAL,
        }
        assert model_data.strat_data["ixs"].keys() == {"sex", "hhsize"}
        assert model_data.strat_data["ixs"]["sex"] is not None

    def test_partial_multi(self, data_partial):
        df_part, df_cnt, df_pop, df_strat = data_partial

        # Create dataclass objects with stratification
        part_data = ParticipantData(
            df_part=df_part,
            id_col="id",
            age_col="age",
            strat_var_cols=["sex", "hhsize"],
        )
        cnt_data = ContactData(df_cnt=df_cnt, id_col="id", age_col="age_cnt")
        pop_data = PopulationData(df_pop=df_pop, age_col="age", size_col="P")
        strat_prop_data = StratificationData(
            data=df_strat,
            age_col="age",
            strat_var_cols=["sex", "hhsize"],
            prop_col="Q",
        )

        dataloader = DataLoader(part_data, cnt_data, pop_data, strat_prop_data)
        model_data = dataloader.load()
        assert model_data.strat_data is not None
        assert model_data.strat_data["modes"].keys() == {"sex", "hhsize"}
        assert model_data.strat_data["modes"] == {
            "sex": StratMode.PARTIAL,
            "hhsize": StratMode.PARTIAL,
        }
        assert model_data.strat_data["labels"]["sex"] == ["M->All", "F->All"]
        assert model_data.strat_data["ixs"].keys() == {"sex", "hhsize"}
        assert model_data.strat_data["ixs"]["sex"] is not None
        assert model_data.strat_data["ixs"]["hhsize"] is not None
        assert model_data.strat_data["flat_ix"] is not None

        sex_labels = ["M", "F"]
        hhsize_labels = ["1", "2", "3", "4", "5+"]
        expected_full_labels = [
            f"{s1}_{s2}->All" for s1 in sex_labels for s2 in hhsize_labels
        ]
        assert model_data.strat_data["full_labels"] == expected_full_labels


class TestFull:

    def test_fine_age(self, data_full):
        df_part, df_cnt, df_pop, df_strat = data_full

        # Create dataclass objects with stratification
        part_data = ParticipantData(
            df_part=df_part,
            id_col="id",
            age_col="age",
            strat_var_cols=["sex", "hhsize"],
        )
        cnt_data = ContactData(
            df_cnt=df_cnt,
            id_col="id",
            age_col="age_cnt",
            strat_var_cols=["sex_cnt", "hhsize_cnt"],
        )
        pop_data = PopulationData(
            df_pop=df_pop, age_col="age", size_col="P", strat_var_cols=["sex", "hhsize"]
        )
        strat_prop_data = StratificationData(
            data=df_strat,  # Empty since full info is in contacts
            age_col="age",
            strat_var_cols=["sex", "hhsize"],
            prop_col="Q",
        )

        dataloader = DataLoader(part_data, cnt_data, pop_data, strat_prop_data)

        model_data = dataloader.load()

        assert isinstance(model_data, ModelData)
        assert isinstance(model_data.base_data["y"], np.ndarray)
        assert isinstance(model_data.base_data["aid"], np.ndarray)
        assert isinstance(model_data.base_data["bid"], np.ndarray)
        assert isinstance(model_data.base_data["log_N"], np.ndarray)
        assert isinstance(model_data.base_data["log_P"], np.ndarray)
        assert isinstance(model_data.base_data["log_V"], np.ndarray)

        # Test shapes
        y_shape = model_data.base_data["y"].shape
        aid_shape = model_data.base_data["aid"].shape
        bid_shape = model_data.base_data["bid"].shape
        log_N_shape = model_data.base_data["log_N"].shape
        log_V_shape = model_data.base_data["log_V"].shape
        assert aid_shape == bid_shape
        assert y_shape == aid_shape
        assert log_N_shape == log_V_shape

        # Test strat data
        assert model_data.strat_data is not None
        assert model_data.strat_data["modes"].keys() == {"sex", "hhsize"}
        assert model_data.strat_data["modes"] == {
            "sex": StratMode.FULL,
            "hhsize": StratMode.FULL,
        }
        assert model_data.strat_data["ixs"]["sex"] is not None
        assert model_data.strat_data["ixs"]["hhsize"] is not None


class TestMethods:

    def test_make_strat_data_partial(self, data_partial):

        df_part, df_cnt, df_pop, df_strat = data_partial
        part_data = ParticipantData(
            df_part=df_part,
            id_col="id",
            age_col="age",
            strat_var_cols=["sex", "hhsize"],
        )
        cnt_data = ContactData(df_cnt=df_cnt, id_col="id", age_col="age_cnt")
        pop_data = PopulationData(df_pop=df_pop, age_col="age", size_col="P")
        strat_prop_data = StratificationData(
            data=df_strat,
            age_col="age",
            strat_var_cols=["sex", "hhsize"],
            prop_col="Q",
        )

        dataloader = DataLoader(part_data, cnt_data, pop_data, strat_prop_data)
        dataloader.load()

        # Test strat modes
        assert dataloader.model_data.strat_data["modes"] == {
            "sex": StratMode.PARTIAL,
            "hhsize": StratMode.PARTIAL,
        }

        # Test strat dims
        assert dataloader.model_data.strat_data["dims"] == {
            "sex": 2,
            "hhsize": 5,
        }

        # Test strat labels
        assert dataloader.model_data.strat_data["labels"] == {
            "sex": ["M->All", "F->All"],
            "hhsize": ["1->All", "2->All", "3->All", "4->All", "5+->All"],
        }

        # Test strat ixs
        expected_codes_sex = dataloader.df_full["sex_part"].cat.codes.to_numpy()
        np.testing.assert_array_equal(
            dataloader.model_data.strat_data["ixs"]["sex"], expected_codes_sex
        )

        expected_codes_hhsize = dataloader.df_full["hhsize_part"].cat.codes.to_numpy()
        np.testing.assert_array_equal(
            dataloader.model_data.strat_data["ixs"]["hhsize"], expected_codes_hhsize
        )

        # Test flat_ix
        expected_flat_ixs = (
            dataloader.model_data.strat_data["ixs"]["sex"]
            * dataloader.model_data.strat_data["dims"]["hhsize"]
            + dataloader.model_data.strat_data["ixs"]["hhsize"]
        )
        np.testing.assert_array_equal(
            dataloader.model_data.strat_data["flat_ix"], expected_flat_ixs
        )


@pytest.fixture
def data_multi_strat():
    """Generate data with multiple FULL stratification variables."""
    templates = load_template_patterns("United_States")
    df_age_dist = load_age_distribution("United_States")

    # Create two FULL stratification variables
    strats = [
        Stratification(
            name="sex",
            n_strata=2,
            ref_age_dist=df_age_dist["P"].values,
            labels=["M", "F"],
            seed=SEED,
        ),
        Stratification(
            name="setting",
            n_strata=2,
            ref_age_dist=df_age_dist["P"].values,
            labels=["home", "work"],
            seed=SEED,
        ),
    ]
    popcon = PopulationConstructor(strats)

    # Generate participant data
    part_gen = ParticipantGenerator(popcon, n_part=500)
    df_part = part_gen.generate(SEED)
    df_part["sex"] = pd.Categorical(df_part["sex"], categories=["M", "F"], ordered=True)
    df_part["setting"] = pd.Categorical(
        df_part["setting"], categories=["home", "work"], ordered=True
    )

    # Generate contact matrices
    matrix_gen = MatrixGenerator(templates)
    contact_matrices = matrix_gen.generate_full(popcon, SEED)

    # Generate contact data
    cnt_gen = ContactGenerator(df_part, contact_matrices)
    df_cnt = cnt_gen.generate(SEED)
    df_cnt["sex_cnt"] = pd.Categorical(
        df_cnt["sex_cnt"], categories=["M", "F"], ordered=True
    )
    df_cnt["setting_cnt"] = pd.Categorical(
        df_cnt["setting_cnt"], categories=["home", "work"], ordered=True
    )

    # Population data
    df_pop = popcon.df_P
    df_pop["sex"] = pd.Categorical(df_pop["sex"], categories=["M", "F"], ordered=True)
    df_pop["setting"] = pd.Categorical(
        df_pop["setting"], categories=["home", "work"], ordered=True
    )

    # Stratification proportions
    df_strat = popcon.df_Q
    df_strat["sex"] = pd.Categorical(
        df_strat["sex"], categories=["M", "F"], ordered=True
    )
    df_strat["setting"] = pd.Categorical(
        df_strat["setting"], categories=["home", "work"], ordered=True
    )

    return df_part, df_cnt, df_pop, df_strat


class TestStratificationOrdering:
    """Test that flat_ix ordering matches full_strat_labels ordering."""

    def test_flat_ix_matches_labels_single_variable(self, data_full):
        """Test ordering consistency for single FULL variable."""
        df_part, df_cnt, df_pop, df_strat = data_full

        part_data = ParticipantData(
            df_part=df_part, id_col="id", age_col="age", strat_var_cols=["sex"]
        )
        cnt_data = ContactData(
            df_cnt=df_cnt, id_col="id", age_col="age_cnt", strat_var_cols=["sex_cnt"]
        )
        pop_data = PopulationData(
            df_pop=df_pop, age_col="age", size_col="P", strat_var_cols=["sex"]
        )
        strat_prop_data = StratificationData(
            data=df_strat, age_col="age", strat_var_cols=["sex"], prop_col="Q"
        )

        dataloader = DataLoader(part_data, cnt_data, pop_data, strat_prop_data)
        dataloader.load()

        full_labels = dataloader.model_data.strat_data["full_labels"]

        # Expected ordering for sex (M, F):
        # flat_ix 0: M->M, flat_ix 1: M->F, flat_ix 2: F->M, flat_ix 3: F->F
        expected_labels = ["M->M", "M->F", "F->M", "F->F"]
        assert full_labels == expected_labels

    def test_flat_ix_matches_labels_multiple_variables(self, data_multi_strat):
        """Test ordering consistency for multiple FULL variables."""
        df_part, df_cnt, df_pop, df_strat = data_multi_strat

        part_data = ParticipantData(
            df_part=df_part,
            id_col="id",
            age_col="age",
            strat_var_cols=["sex", "setting"],
        )
        cnt_data = ContactData(
            df_cnt=df_cnt,
            id_col="id",
            age_col="age_cnt",
            strat_var_cols=["sex_cnt", "setting_cnt"],
        )
        pop_data = PopulationData(
            df_pop=df_pop,
            age_col="age",
            size_col="P",
            strat_var_cols=["sex", "setting"],
        )
        strat_prop_data = StratificationData(
            data=df_strat,
            age_col="age",
            strat_var_cols=["sex", "setting"],
            prop_col="Q",
        )

        dataloader = DataLoader(part_data, cnt_data, pop_data, strat_prop_data)
        dataloader.load()

        # Get full labels
        flat_ix = dataloader.model_data.strat_data["flat_ix"]
        full_labels = dataloader.model_data.strat_data["full_labels"]

        # Row-major ordering: sex varies SLOWEST, setting varies FASTEST
        # flat_ix = sex_ix * 4 + setting_ix
        # sex labels: M->M(0), M->F(1), F->M(2), F->F(3)
        # setting labels: home->home(0), home->work(1), work->home(2), work->work(3)
        expected_labels = [
            # sex=0 (M->M), setting varies
            "M_home->M_home",  # sex=0, setting=0 → 0*4+0=0
            "M_home->M_work",  # sex=0, setting=1 → 0*4+1=1
            "M_work->M_home",  # sex=0, setting=2 → 0*4+2=2
            "M_work->M_work",  # sex=0, setting=3 → 0*4+3=3
            # sex=1 (M->F), setting varies
            "M_home->F_home",  # sex=1, setting=0 → 1*4+0=4
            "M_home->F_work",  # sex=1, setting=1 → 1*4+1=5
            "M_work->F_home",  # sex=1, setting=2 → 1*4+2=6
            "M_work->F_work",  # sex=1, setting=3 → 1*4+3=7
            # sex=2 (F->M), setting varies
            "F_home->M_home",  # sex=2, setting=0 → 2*4+0=8
            "F_home->M_work",  # sex=2, setting=1 → 2*4+1=9
            "F_work->M_home",  # sex=2, setting=2 → 2*4+2=10
            "F_work->M_work",  # sex=2, setting=3 → 2*4+3=11
            # sex=3 (F->F), setting varies
            "F_home->F_home",  # sex=3, setting=0 → 3*4+0=12
            "F_home->F_work",  # sex=3, setting=1 → 3*4+1=13
            "F_work->F_home",  # sex=3, setting=2 → 3*4+2=14
            "F_work->F_work",  # sex=3, setting=3 → 3*4+3=15
        ]

        assert len(full_labels) == 16
        assert full_labels == expected_labels

        # Verify correspondence with flat_ix computation
        # Use cached df_full (same snapshot used during load()) for consistency
        df = dataloader.df_full
        sex_part_codes = df["sex_part"].cat.codes.to_numpy()
        sex_cnt_codes = df["sex_cnt"].cat.codes.to_numpy()
        setting_part_codes = df["setting_part"].cat.codes.to_numpy()
        setting_cnt_codes = df["setting_cnt"].cat.codes.to_numpy()

        # Vectorised flat_ix check
        sex_ix = sex_part_codes * 2 + sex_cnt_codes
        setting_ix = setting_part_codes * 2 + setting_cnt_codes
        expected_flat_ix = sex_ix * 4 + setting_ix
        np.testing.assert_array_equal(flat_ix, expected_flat_ix)

        # Vectorised label lookup check
        sex_part_vals = df["sex_part"].to_numpy().astype(str)
        sex_cnt_vals = df["sex_cnt"].to_numpy().astype(str)
        setting_part_vals = df["setting_part"].to_numpy().astype(str)
        setting_cnt_vals = df["setting_cnt"].to_numpy().astype(str)

        for i in range(len(df)):
            actual_label = full_labels[flat_ix[i]]
            expected_label = (
                f"{sex_part_vals[i]}_{setting_part_vals[i]}"
                f"->{sex_cnt_vals[i]}_{setting_cnt_vals[i]}"
            )
            assert actual_label == expected_label, (
                f"Row {i}: Expected '{expected_label}' but got '{actual_label}' "
                f"(flat_ix={flat_ix[i]})"
            )

    def test_flat_ix_matches_labels_partial_single_variable(self, data_partial):
        """Test ordering consistency for a single PARTIAL variable."""
        df_part, df_cnt, df_pop, df_strat = data_partial

        part_data = ParticipantData(
            df_part=df_part, id_col="id", age_col="age", strat_var_cols=["sex"]
        )
        cnt_data = ContactData(df_cnt=df_cnt, id_col="id", age_col="age_cnt")
        pop_data = PopulationData(df_pop=df_pop, age_col="age", size_col="P")
        strat_prop_data = StratificationData(
            data=df_strat, age_col="age", strat_var_cols=["sex"], prop_col="Q"
        )

        dataloader = DataLoader(part_data, cnt_data, pop_data, strat_prop_data)
        dataloader.load()

        flat_ix = dataloader.model_data.strat_data["flat_ix"]
        full_labels = dataloader.model_data.strat_data["full_labels"]

        # PARTIAL: labels are "cat->All", flat_ix = sex_part_code
        expected_labels = ["M->All", "F->All"]
        assert full_labels == expected_labels

        # Verify flat_ix matches participant codes
        df = dataloader.df_full
        sex_part_codes = df["sex_part"].cat.codes.to_numpy()
        np.testing.assert_array_equal(flat_ix, sex_part_codes)

        # Verify label lookup
        sex_part_vals = df["sex_part"].to_numpy().astype(str)
        for i in range(len(df)):
            actual_label = full_labels[flat_ix[i]]
            expected_label = f"{sex_part_vals[i]}->All"
            assert actual_label == expected_label, (
                f"Row {i}: Expected '{expected_label}' but got '{actual_label}' "
                f"(flat_ix={flat_ix[i]})"
            )

    def test_flat_ix_matches_labels_partial_multiple_variables(self, data_partial):
        """Test ordering consistency for multiple PARTIAL variables."""
        df_part, df_cnt, df_pop, df_strat = data_partial

        part_data = ParticipantData(
            df_part=df_part,
            id_col="id",
            age_col="age",
            strat_var_cols=["sex", "hhsize"],
        )
        cnt_data = ContactData(df_cnt=df_cnt, id_col="id", age_col="age_cnt")
        pop_data = PopulationData(df_pop=df_pop, age_col="age", size_col="P")
        strat_prop_data = StratificationData(
            data=df_strat,
            age_col="age",
            strat_var_cols=["sex", "hhsize"],
            prop_col="Q",
        )

        dataloader = DataLoader(part_data, cnt_data, pop_data, strat_prop_data)
        dataloader.load()

        flat_ix = dataloader.model_data.strat_data["flat_ix"]
        full_labels = dataloader.model_data.strat_data["full_labels"]

        # PARTIAL for both: strat_dims = {'sex': 2, 'hhsize': 5}
        # flat_ix = sex_ix * 5 + hhsize_ix  (sex varies slowest, hhsize fastest)
        # Labels use "cat->All" format combined as "sex_hhsize->All"
        expected_labels = [
            # sex=0 (M), hhsize varies
            "M_1->All",  # sex=0, hhsize=0 → 0*5+0=0
            "M_2->All",  # sex=0, hhsize=1 → 0*5+1=1
            "M_3->All",  # sex=0, hhsize=2 → 0*5+2=2
            "M_4->All",  # sex=0, hhsize=3 → 0*5+3=3
            "M_5+->All",  # sex=0, hhsize=4 → 0*5+4=4
            # sex=1 (F), hhsize varies
            "F_1->All",  # sex=1, hhsize=0 → 1*5+0=5
            "F_2->All",  # sex=1, hhsize=1 → 1*5+1=6
            "F_3->All",  # sex=1, hhsize=2 → 1*5+2=7
            "F_4->All",  # sex=1, hhsize=3 → 1*5+3=8
            "F_5+->All",  # sex=1, hhsize=4 → 1*5+4=9
        ]

        assert len(full_labels) == 10
        assert full_labels == expected_labels

        # Verify flat_ix computation against participant codes
        df = dataloader.df_full
        sex_codes = df["sex_part"].cat.codes.to_numpy()
        hhsize_codes = df["hhsize_part"].cat.codes.to_numpy()
        expected_flat_ix = sex_codes * 5 + hhsize_codes
        np.testing.assert_array_equal(flat_ix, expected_flat_ix)

        # Verify every observation's flat_ix maps to the correct label
        sex_vals = df["sex_part"].to_numpy().astype(str)
        hhsize_vals = df["hhsize_part"].to_numpy().astype(str)

        for i in range(len(df)):
            actual_label = full_labels[flat_ix[i]]
            expected_label = f"{sex_vals[i]}_{hhsize_vals[i]}->All"
            assert actual_label == expected_label, (
                f"Row {i}: Expected '{expected_label}' but got '{actual_label}' "
                f"(flat_ix={flat_ix[i]})"
            )
