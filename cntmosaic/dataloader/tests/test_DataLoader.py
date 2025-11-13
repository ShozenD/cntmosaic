import numpy as np
import pandas as pd
import pytest

from ..._types import StratMode
from ...datasets import load_age_distribution, load_template_patterns
from ...sim import ContactGenerator, MatrixGenerator, ParticipantGenerator, Subgroup
from .._dataloader import DataLoader
from ..containers import ContactData, ParticipantData, PopulationData, StratPropData
from ..containers._ModelData import ModelBaseData, ModelData, ModelStratData

SEED = 42
templates = load_template_patterns("United_States")
df_age_dist = load_age_distribution("United_States")


@pytest.fixture
def generate_data_single():
    """Generate contact data for a single subgroup."""
    subgroup = Subgroup(
        n=500,
        age_dist=df_age_dist.P.values,
        mean_cint_margin=10,
        label="single",
    )

    # Generate participants
    part_gen = ParticipantGenerator(subgroup)
    df_part = part_gen.generate(SEED)

    # Generate contact matrix
    matrix_gen = MatrixGenerator(templates)
    contact_matrix = matrix_gen.generate_single(subgroup, SEED)

    # Generate contacts
    cnt_gen = ContactGenerator(df_part, contact_matrix)
    df_cnt = cnt_gen.generate(SEED)

    return df_part, df_cnt


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
    df_part["subgroup"] = df_part["subgroup"].astype("category")

    # Generate contact matrix
    matrix_gen = MatrixGenerator(templates)
    contact_matrices = matrix_gen.generate_full(subgroups, SEED)

    # Generate contacts
    cnt_gen = ContactGenerator(df_part, contact_matrices)
    df_cnt = cnt_gen.generate(SEED)
    df_cnt["subgroup_cnt"] = df_cnt["subgroup_cnt"].astype("category")

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


# =================================
# Tests
# =================================
class TestSingle:

    def test_fine_age(self, generate_data_single):
        df_part, df_cnt = generate_data_single

        # Create dataclass objects with standardized column names
        part_data = ParticipantData(df_part=df_part, id_col="id", age_col="age_group")
        cnt_data = ContactData(df_cnt=df_cnt, id_col="id", age_col="age_cnt")
        pop_data = PopulationData(df_pop=df_age_dist, age_col="age", size_col="P")
        dataloader = DataLoader(part_data, cnt_data, pop_data)

        model_data = dataloader.load()

        # Test if anticipated types
        assert isinstance(model_data, ModelData)
        assert isinstance(model_data.base_data["y"], np.ndarray)
        assert isinstance(model_data.base_data["aid"], np.ndarray)
        assert isinstance(model_data.base_data["bid"], np.ndarray)
        assert isinstance(model_data.base_data["log_N"], np.ndarray)
        assert isinstance(model_data.base_data["log_P"], np.ndarray)
        assert isinstance(model_data.base_data["log_S"], np.ndarray)

        # Test shapes
        len_y = len(model_data.base_data["y"])
        len_aid = len(model_data.base_data["aid"])
        len_bid = len(model_data.base_data["bid"])
        len_log_N = len(model_data.base_data["log_N"])
        len_log_S = len(model_data.base_data["log_S"])
        assert len_y == len_aid == len_bid == len_log_N == len_log_S

        # No stratication configuration
        assert model_data.strat_data == {}

    def test_repeat_effect(self, generate_data_single):
        df_part, df_cnt = generate_data_single
        df_part["repeat"] = np.random.randint(0, 10, size=len(df_part))

        # Create dataclass objects with repeat column
        part_data = ParticipantData(
            df_part=df_part, id_col="id", age_col="age_group", repeat_col="repeat"
        )
        cnt_data = ContactData(df_cnt=df_cnt, id_col="id", age_col="age_cnt")
        pop_data = PopulationData(df_pop=df_age_dist, age_col="age", size_col="P")
        dataloader = DataLoader(part_data, cnt_data, pop_data)

        model_data = dataloader.load()

        assert isinstance(model_data.base_data["rid"], np.ndarray)
        assert model_data.base_data["rid"].shape == model_data.base_data["y"].shape

        # No stratication configuration
        assert model_data.strat_data == {}

    def test_age_grp_cnt(self, generate_data_single):
        df_part, df_cnt = generate_data_single

        # Create age groups for contacts
        bins = [0, 5, 15, 25, 35, 45, 55, 65, 75, 80]
        df_cnt["age_grp_cnt"] = pd.cut(df_cnt["age_cnt"], bins=bins, right=False)
        df_cnt["age_grp_cnt"] = pd.Categorical(df_cnt["age_grp_cnt"])

        # Create dataclass objects with age groups
        part_data = ParticipantData(df_part=df_part, id_col="id", age_col="age_group")
        cnt_data = ContactData(df_cnt=df_cnt, id_col="id", age_grp_col="age_grp_cnt")
        pop_data = PopulationData(df_pop=df_age_dist, age_col="age", size_col="P")
        dataloader = DataLoader(part_data, cnt_data, pop_data)

        model_data = dataloader.load()

        assert isinstance(model_data.base_data["y"], np.ndarray)
        assert isinstance(model_data.base_data["log_N"], np.ndarray)
        assert isinstance(model_data.base_data["log_P"], np.ndarray)
        assert isinstance(model_data.base_data["log_S"], np.ndarray)
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
            model_data.base_data["log_N"].shape == model_data.base_data["log_S"].shape
        )


class TestPartial:

    def test_fine_age(self, generate_data_partial):
        df_part, df_cnt, df_pop_prop = generate_data_partial

        # Create dataclass objects with stratification
        part_data = ParticipantData(
            df_part=df_part,
            id_col="id",
            age_col="age_group",
            strat_var_cols="subgroup",
        )

        cnt_data = ContactData(df_cnt=df_cnt, id_col="id", age_col="age_cnt")
        pop_data = PopulationData(df_pop=df_age_dist, age_col="age", size_col="P")
        strat_prop_data = StratPropData(
            data=df_pop_prop,
            age_col="age",
            strat_col="subgroup",
            prop_col="prop",
        )

        dataloader = DataLoader(part_data, cnt_data, pop_data, strat_prop_data)

        model_data = dataloader.load()

        assert isinstance(model_data, ModelData)
        assert isinstance(model_data.base_data["y"], np.ndarray)
        assert isinstance(model_data.base_data["aid"], np.ndarray)
        assert isinstance(model_data.base_data["bid"], np.ndarray)
        assert isinstance(model_data.base_data["log_N"], np.ndarray)
        assert isinstance(model_data.base_data["log_P"], np.ndarray)
        assert isinstance(model_data.base_data["log_S"], np.ndarray)

        # Test shapes
        assert model_data.base_data["aid"].shape == model_data.base_data["bid"].shape
        assert model_data.base_data["y"].shape == model_data.base_data["aid"].shape
        assert (
            model_data.base_data["log_N"].shape == model_data.base_data["log_S"].shape
        )

    def test_repeat_effect(self, generate_data_partial):
        df_part, df_cnt, df_pop_prop = generate_data_partial
        df_part["repeat"] = np.random.randint(0, 10, size=len(df_part))

        # Create dataclass objects with stratification and repeat
        part_data = ParticipantData(
            df_part=df_part,
            id_col="id",
            age_col="age_group",
            strat_var_cols="subgroup",
            repeat_col="repeat",
        )

        cnt_data = ContactData(df_cnt=df_cnt, id_col="id", age_col="age_cnt")

        pop_data = PopulationData(df_pop=df_age_dist, age_col="age", size_col="P")

        pop_prop = StratPropData(
            data=df_pop_prop,
            age_col="age",
            strat_col="subgroup",
            prop_col="prop",
        )

        dataloader = DataLoader(part_data, cnt_data, pop_data, pop_prop)

        model_data = dataloader.load()

        # Test that rid exists and has correct shape
        assert isinstance(model_data.base_data["rid"], np.ndarray)
        assert model_data.base_data["rid"].shape == model_data.base_data["aid"].shape

        # Should have stratification data
        assert model_data.strat_data is not None
        assert model_data.strat_data["strat_vars"].keys() == {"subgroup"}
        assert model_data.strat_data["strat_vars"]["subgroup"] == ["A", "B"]
        assert model_data.strat_data["strat_modes"] == {"subgroup": "partial"}
        assert model_data.strat_data["strat_ix"].keys() == {"subgroup_part"}
        assert model_data.strat_data["strat_ix"]["subgroup_part"] is not None
        assert model_data.strat_data["strat_vars_full"] == {}

    def test_age_grp_cnt(self, generate_data_partial):
        df_part, df_cnt, df_pop_prop = generate_data_partial

        # Create age groups for contacts
        bins = [0, 5, 15, 25, 35, 45, 55, 65, 75, 80]
        df_cnt["age_grp_cnt"] = pd.cut(df_cnt["age_cnt"], bins=bins, right=False)

        # Create dataclass objects with age groups and stratification
        part_data = ParticipantData(
            df_part=df_part,
            id_col="id",
            age_col="age_group",
            strat_var_cols="subgroup",
        )

        cnt_data = ContactData(df_cnt=df_cnt, id_col="id", age_grp_col="age_grp_cnt")
        pop_data = PopulationData(df_pop=df_age_dist, age_col="age", size_col="P")
        pop_prop = StratPropData(
            data=df_pop_prop,
            age_col="age",
            strat_col="subgroup",
            prop_col="prop",
        )
        dataloader = DataLoader(part_data, cnt_data, pop_data, pop_prop)

        model_data = dataloader.load()

        assert isinstance(model_data, ModelData)
        assert isinstance(model_data.base_data["aid_exp"], np.ndarray)
        assert isinstance(model_data.base_data["bid_pad"], np.ndarray)

        # Should have stratification data
        assert model_data.strat_data is not None
        assert model_data.strat_data["strat_vars"].keys() == {"subgroup"}
        assert model_data.strat_data["strat_vars"]["subgroup"] == ["A", "B"]
        assert model_data.strat_data["strat_modes"] == {"subgroup": "partial"}
        assert model_data.strat_data["strat_ix"].keys() == {"subgroup_part"}
        assert model_data.strat_data["strat_ix"]["subgroup_part"] is not None
        assert model_data.strat_data["strat_vars_full"] == {}


class TestFull:

    def test_fine_age(self, generate_data_full):
        df_part, df_cnt, df_strat_prop = generate_data_full

        # Create dataclass objects with stratification
        part_data = ParticipantData(
            df_part=df_part,
            id_col="id",
            age_col="age_group",
            strat_var_cols="subgroup",
        )

        cnt_data = ContactData(
            df_cnt=df_cnt, id_col="id", age_col="age_cnt", strat_var_cols="subgroup_cnt"
        )
        pop_data = PopulationData(
            df_pop=df_strat_prop, age_col="age", size_col="P", strat_var_cols="subgroup"
        )
        strat_prop_data = StratPropData(
            data=df_strat_prop,  # Empty since full info is in contacts
            age_col="age",
            strat_col="subgroup",
            prop_col="prop",
        )

        dataloader = DataLoader(part_data, cnt_data, pop_data, strat_prop_data)

        model_data = dataloader.load()

        assert isinstance(model_data, ModelData)
        assert isinstance(model_data.base_data["y"], np.ndarray)
        assert isinstance(model_data.base_data["aid"], np.ndarray)
        assert isinstance(model_data.base_data["bid"], np.ndarray)
        assert isinstance(model_data.base_data["log_N"], np.ndarray)
        assert isinstance(model_data.base_data["log_P"], dict)
        assert isinstance(model_data.base_data["log_S"], np.ndarray)

        # Test shapes
        y_shape = model_data.base_data["y"].shape
        aid_shape = model_data.base_data["aid"].shape
        bid_shape = model_data.base_data["bid"].shape
        log_N_shape = model_data.base_data["log_N"].shape
        log_S_shape = model_data.base_data["log_S"].shape
        assert aid_shape == bid_shape
        assert y_shape == aid_shape
        assert log_N_shape == log_S_shape

        # Test strat data
        assert model_data.strat_data is not None
        assert model_data.strat_data["strat_vars"].keys() == {"subgroup"}
        assert model_data.strat_data["strat_vars"]["subgroup"] == ["A", "B"]
        assert model_data.strat_data["strat_modes"] == {"subgroup": "full"}
        assert model_data.strat_data["strat_ix"]["subgroup_part"] is not None
        assert model_data.strat_data["strat_ix"]["subgroup_cnt"] is not None
        assert model_data.strat_data["strat_vars_full"] == {
            "subgroup": {"part": ["A", "B"], "cnt": ["A", "B"]}
        }


# ============================================================================
# StratPropData Tests
# ============================================================================


class TestStratPropData:
    """Test suite for the new StratPropData API."""

    def test_basic_initialization(self):
        """Test basic StratPropData initialization with valid data."""
        df = pd.DataFrame(
            {
                "age": [0, 0, 1, 1, 2, 2],
                "gender": ["M", "F", "M", "F", "M", "F"],
                "proportion": [0.51, 0.49, 0.51, 0.49, 0.50, 0.50],
            }
        )

        pop_prop = StratPropData(
            data=df, age_col="age", strat_col="gender", prop_col="proportion"
        )

        assert pop_prop.age_col == "age"
        assert pop_prop.strat_col == "gender"
        assert pop_prop.prop_col == "proportion"

    def test_from_counts(self):
        """Test StratPropData.from_counts() constructor."""
        df_counts = pd.DataFrame(
            {
                "age": [0, 0, 1, 1, 2, 2],
                "gender": ["M", "F", "M", "F", "M", "F"],
                "population": [510, 490, 505, 495, 500, 500],
            }
        )

        pop_prop = StratPropData.from_counts(
            data=df_counts, age_col="age", strat_col="gender", count_col="population"
        )

        # Check proportions were computed correctly
        assert "proportion" in pop_prop.data.columns

        # Check proportions sum to 1 for each age
        for age in [0, 1, 2]:
            age_props = pop_prop.data[pop_prop.data["age"] == age]["proportion"]
            assert np.isclose(age_props.sum(), 1.0)

    def test_missing_columns_error(self):
        """Test that missing columns raise ValueError."""
        df = pd.DataFrame(
            {
                "age": [0, 1, 2],
                "gender": ["M", "F", "M"],
                # Missing 'proportion' column
            }
        )

        with pytest.raises(ValueError, match="Missing required columns"):
            StratPropData(
                data=df,
                age_col="age",
                strat_col="gender",
                prop_col="proportion",
            )

    def test_invalid_proportion_values(self):
        """Test that proportions outside [0,1] raise ValueError."""
        df = pd.DataFrame(
            {
                "age": [0, 0],
                "gender": ["M", "F"],
                "proportion": [0.6, 1.5],  # 1.5 is invalid
            }
        )

        with pytest.raises(ValueError, match="must be in range"):
            StratPropData(
                data=df,
                age_col="age",
                strat_col="gender",
                prop_col="proportion",
            )

    def test_proportions_not_summing_to_one(self):
        """Test that proportions not summing to 1 raise ValueError."""
        df = pd.DataFrame(
            {
                "age": [0, 0, 1, 1],
                "gender": ["M", "F", "M", "F"],
                "proportion": [0.6, 0.3, 0.5, 0.5],  # First age sums to 0.9
            }
        )

        with pytest.raises(ValueError, match="must sum to 1.0"):
            StratPropData(
                data=df,
                age_col="age",
                strat_col="gender",
                prop_col="proportion",
            )

    def test_dataloader(self, generate_data_partial):
        """Test DataLoader with new StratPropData API."""
        df_part, df_cnt, df_pop_prop = generate_data_partial

        # Create StratPropData using new API
        strat_prop = StratPropData(
            data=df_pop_prop,
            age_col="age",
            strat_col="subgroup",
            prop_col="prop",
        )

        # Create dataclass objects
        part_data = ParticipantData(
            df_part=df_part,
            id_col="id",
            age_col="age_group",
            strat_var_cols="subgroup",
        )

        cnt_data = ContactData(df_cnt=df_cnt, id_col="id", age_col="age_cnt")

        pop_data = PopulationData(df_pop=df_age_dist, age_col="age", size_col="P")

        # Use new API
        dataloader = DataLoader(part_data, cnt_data, pop_data, strat_prop)

        model_data = dataloader.load()

    def test_multi_population_proportions(self, generate_data_partial):
        """Test DataLoader with multiple StratPropData objects."""
        df_part, df_cnt, df_pop_prop = generate_data_partial

        # Add another stratification variable
        df_part["region"] = pd.Categorical(
            np.random.choice(["North", "South"], size=len(df_part)),
            categories=["North", "South"],
            ordered=True,
        )
        df_cnt = df_cnt.merge(df_part[["id", "region"]], on="id", how="left")

        # Create region-stratified population
        df_pop_region = pd.DataFrame(
            {
                "age": np.tile(np.arange(76), 2),
                "region": ["North"] * 76 + ["South"] * 76,
                "proportion": [0.55] * 76 + [0.45] * 76,
            }
        )

        pop_prop_subgroup = StratPropData(
            data=df_pop_prop,
            age_col="age",
            strat_col="subgroup",
            prop_col="prop",
        )

        pop_prop_region = StratPropData(
            data=df_pop_region,
            age_col="age",
            strat_col="region",
            prop_col="proportion",
        )

        # Create dataclass objects
        part_data = ParticipantData(
            df_part=df_part,
            id_col="id",
            age_col="age_group",
            strat_var_cols=["subgroup", "region"],
        )

        cnt_data = ContactData(df_cnt=df_cnt, id_col="id", age_col="age_cnt")

        pop_data = PopulationData(df_pop=df_age_dist, age_col="age", size_col="P")

        dataloader = DataLoader(
            part_data,
            cnt_data,
            pop_data,
            [pop_prop_subgroup, pop_prop_region],
        )

        model_data = dataloader.load()
