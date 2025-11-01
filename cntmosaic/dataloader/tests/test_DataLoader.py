import pytest
import xarray as xr
import numpy as np
import pandas as pd

from .._dataloader import CoordToColumns, DataLoader, PopulationProportion
from ...datasets import load_template_patterns, load_age_distribution
from ...sim import Subgroup, ParticipantGenerator, MatrixGenerator, ContactGenerator

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

    return df_part, df_cnt


# =================================
# Tests
# =================================
class TestSingleSubgroup:

    def test_fine_age(self, generate_data_single):
        df_part, df_cnt = generate_data_single

        colmap = CoordToColumns(
            age_part="age_group",
            age_cnt="age_cnt",
            age_pop="age",
            size_pop="P",
        )

        dataloader = DataLoader(
            part=df_part,
            cnt=df_cnt,
            pop=df_age_dist,
            col_map=colmap,
        )

        ds = dataloader.load()

        assert isinstance(ds, xr.Dataset)
        assert ds["y"] is not None
        assert ds["log_N"] is not None
        assert ds["log_P"] is not None
        assert ds["log_S"] is not None
        assert ds["aid"] is not None
        assert ds["bid"] is not None

        assert ds["aid"].shape == ds["bid"].shape
        assert ds["y"].shape == ds["aid"].shape
        assert ds["log_N"].shape == ds["log_S"].shape

    def test_repeat_effect(self, generate_data_single):
        df_part, df_cnt = generate_data_single
        df_part["repeat"] = np.random.randint(0, 10, size=len(df_part))

        colmap = CoordToColumns(
            age_part="age_group",
            age_cnt="age_cnt",
            age_pop="age",
            size_pop="P",
            repeat_part="repeat",
        )

        dataloader = DataLoader(
            part=df_part,
            cnt=df_cnt,
            pop=df_age_dist,
            col_map=colmap,
        )

        ds = dataloader.load()

        assert ds["rid"] is not None
        assert ds["rid"].shape == ds["aid"].shape

    def test_age_grp_cnt(self, generate_data_single):
        df_part, df_cnt = generate_data_single

        # Create age groups for contacts
        bins = [0, 5, 15, 25, 35, 45, 55, 65, 75, 80]
        df_cnt["age_grp_cnt"] = pd.cut(df_cnt["age_cnt"], bins=bins, right=False)

        colmap = CoordToColumns(
            age_part="age_group",
            age_grp_cnt="age_grp_cnt",
            age_pop="age",
            size_pop="P",
        )

        dataloader = DataLoader(
            part=df_part,
            cnt=df_cnt,
            pop=df_age_dist,
            col_map=colmap,
        )

        ds = dataloader.load()

        assert isinstance(ds, xr.Dataset)
        assert ds["y"] is not None
        assert ds["log_N"] is not None
        assert ds["log_P"] is not None
        assert ds["log_S"] is not None
        assert ds["aid_exp"] is not None
        assert ds["bid_pad"] is not None

        assert ds["aid_exp"].shape[0] == ds["bid_pad"].shape[0]
        assert ds["y"].shape[0] == ds["aid_exp"].shape[0]
        assert ds["log_N"].shape == ds["log_S"].shape


class TestPartial:

    def test_fine_age(self, generate_data_partial):
        df_part, df_cnt, df_pop_prop = generate_data_partial

        colmap = CoordToColumns(
            age_part="age_group",
            age_cnt="age_cnt",
            age_pop="age",
            size_pop="P",
            grp_vars_part="subgroup",
        )

        dataloader = DataLoader(
            part=df_part,
            cnt=df_cnt,
            pop=df_age_dist,
            col_map=colmap,
            pop_prop_quads=[(df_pop_prop, "age", "subgroup", "prop")],
        )

        ds = dataloader.load()

        assert isinstance(ds, xr.Dataset)
        assert ds["aid"].shape == ds["bid"].shape
        assert ds["y"].shape == ds["aid"].shape
        assert ds["log_N"].shape == ds["log_S"].shape
        assert ds["subgroup"] is not None
        assert ds["subgroup"].shape == ds["y"].shape
        assert ds["pop_prop_subgroup"] is not None
        assert len(np.unique(ds["subgroup"])) == 2

    def test_repeat_effect(self, generate_data_partial):
        df_part, df_cnt, df_pop_prop = generate_data_partial
        df_part["repeat"] = np.random.randint(0, 10, size=len(df_part))

        colmap = CoordToColumns(
            age_part="age_group",
            age_cnt="age_cnt",
            age_pop="age",
            size_pop="P",
            grp_vars_part="subgroup",
            repeat_part="repeat",
        )

        dataloader = DataLoader(
            part=df_part,
            cnt=df_cnt,
            pop=df_age_dist,
            col_map=colmap,
            pop_prop_quads=[(df_pop_prop, "age", "subgroup", "prop")],
        )

        ds = dataloader.load()

        assert ds["rid"] is not None
        assert ds["rid"].shape == ds["aid"].shape

    def test_age_grp_cnt(self, generate_data_partial):
        df_part, df_cnt, df_pop_prop = generate_data_partial

        # Create age groups for contacts
        bins = [0, 5, 15, 25, 35, 45, 55, 65, 75, 80]
        df_cnt["age_grp_cnt"] = pd.cut(df_cnt["age_cnt"], bins=bins, right=False)

        colmap = CoordToColumns(
            age_part="age_group",
            age_grp_cnt="age_grp_cnt",
            age_pop="age",
            size_pop="P",
            grp_vars_part="subgroup",
        )

        dataloader = DataLoader(
            part=df_part,
            cnt=df_cnt,
            pop=df_age_dist,
            col_map=colmap,
            pop_prop_quads=[(df_pop_prop, "age", "subgroup", "prop")],
        )

        ds = dataloader.load()

        assert isinstance(ds, xr.Dataset)
        assert ds["aid_exp"].shape[0] == ds["bid_pad"].shape[0]
        assert ds["y"].shape[0] == ds["aid_exp"].shape[0]
        assert ds["log_N"].shape == ds["log_S"].shape
        assert ds["subgroup"] is not None
        assert ds["subgroup"].shape == ds["y"].shape
        assert len(np.unique(ds["subgroup"])) == 2


# ============================================================================
# PopulationProportion Tests
# ============================================================================


class TestPopulationProportion:
    """Test suite for the new PopulationProportion API."""

    def test_basic_initialization(self):
        """Test basic PopulationProportion initialization with valid data."""
        df = pd.DataFrame({
            'age': [0, 0, 1, 1, 2, 2],
            'gender': ['M', 'F', 'M', 'F', 'M', 'F'],
            'proportion': [0.51, 0.49, 0.51, 0.49, 0.50, 0.50]
        })
        
        pop_prop = PopulationProportion(
            data=df,
            age_col='age',
            stratify_by='gender',
            proportion_col='proportion'
        )
        
        assert pop_prop.age_col == 'age'
        assert pop_prop.stratify_by == 'gender'
        assert pop_prop.proportion_col == 'proportion'

    def test_from_counts(self):
        """Test PopulationProportion.from_counts() constructor."""
        df_counts = pd.DataFrame({
            'age': [0, 0, 1, 1, 2, 2],
            'gender': ['M', 'F', 'M', 'F', 'M', 'F'],
            'population': [510, 490, 505, 495, 500, 500]
        })
        
        pop_prop = PopulationProportion.from_counts(
            data=df_counts,
            age_col='age',
            stratify_by='gender',
            count_col='population'
        )
        
        # Check proportions were computed correctly
        assert 'proportion' in pop_prop.data.columns
        
        # Check proportions sum to 1 for each age
        for age in [0, 1, 2]:
            age_props = pop_prop.data[pop_prop.data['age'] == age]['proportion']
            assert np.isclose(age_props.sum(), 1.0)

    def test_missing_columns_error(self):
        """Test that missing columns raise ValueError."""
        df = pd.DataFrame({
            'age': [0, 1, 2],
            'gender': ['M', 'F', 'M']
            # Missing 'proportion' column
        })
        
        with pytest.raises(ValueError, match="Missing required columns"):
            PopulationProportion(
                data=df,
                age_col='age',
                stratify_by='gender',
                proportion_col='proportion'
            )

    def test_invalid_proportion_values(self):
        """Test that proportions outside [0,1] raise ValueError."""
        df = pd.DataFrame({
            'age': [0, 0],
            'gender': ['M', 'F'],
            'proportion': [0.6, 1.5]  # 1.5 is invalid
        })
        
        with pytest.raises(ValueError, match="must be in range"):
            PopulationProportion(
                data=df,
                age_col='age',
                stratify_by='gender',
                proportion_col='proportion'
            )

    def test_proportions_not_summing_to_one(self):
        """Test that proportions not summing to 1 raise ValueError."""
        df = pd.DataFrame({
            'age': [0, 0, 1, 1],
            'gender': ['M', 'F', 'M', 'F'],
            'proportion': [0.6, 0.3, 0.5, 0.5]  # First age sums to 0.9
        })
        
        with pytest.raises(ValueError, match="must sum to 1.0"):
            PopulationProportion(
                data=df,
                age_col='age',
                stratify_by='gender',
                proportion_col='proportion'
            )

    def test_to_tuple_backward_compatibility(self):
        """Test conversion to legacy tuple format."""
        df = pd.DataFrame({
            'age': [0, 0],
            'gender': ['M', 'F'],
            'proportion': [0.5, 0.5]
        })
        
        pop_prop = PopulationProportion(
            data=df,
            age_col='age',
            stratify_by='gender',
            proportion_col='proportion'
        )
        
        tuple_format = pop_prop.to_tuple()
        
        assert len(tuple_format) == 4
        assert isinstance(tuple_format[0], pd.DataFrame)
        assert tuple_format[1] == 'age'
        assert tuple_format[2] == 'gender'
        assert tuple_format[3] == 'proportion'

    def test_dataloader_with_new_api(self, generate_data_partial):
        """Test DataLoader with new PopulationProportion API."""
        df_part, df_cnt, df_pop_prop = generate_data_partial
        
        # Create PopulationProportion using new API
        pop_prop = PopulationProportion(
            data=df_pop_prop,
            age_col='age',
            stratify_by='subgroup',
            proportion_col='prop'
        )
        
        colmap = CoordToColumns(
            age_part="age_group",
            age_cnt="age_cnt",
            age_pop="age",
            size_pop="P",
            grp_vars_part="subgroup",
        )
        
        # Use new API
        dataloader = DataLoader(
            part=df_part,
            cnt=df_cnt,
            pop=df_age_dist,
            col_map=colmap,
            population_proportions=[pop_prop]
        )
        
        ds = dataloader.load()
        
        assert isinstance(ds, xr.Dataset)
        assert ds["pop_prop_subgroup"] is not None
        assert ds["pop_prop_subgroup"].shape[0] == 2  # Two subgroups

    def test_dataloader_from_counts(self, generate_data_partial):
        """Test DataLoader with PopulationProportion.from_counts()."""
        df_part, df_cnt, df_pop_prop = generate_data_partial
        
        # Remove proportion column and use counts instead
        df_counts = df_pop_prop.copy()
        df_counts = df_counts.drop(columns=['prop'])
        
        # Create PopulationProportion from counts
        pop_prop = PopulationProportion.from_counts(
            data=df_counts,
            age_col='age',
            stratify_by='subgroup',
            count_col='P'
        )
        
        colmap = CoordToColumns(
            age_part="age_group",
            age_cnt="age_cnt",
            age_pop="age",
            size_pop="P",
            grp_vars_part="subgroup",
        )
        
        dataloader = DataLoader(
            part=df_part,
            cnt=df_cnt,
            pop=df_age_dist,
            col_map=colmap,
            population_proportions=[pop_prop]
        )
        
        ds = dataloader.load()
        
        assert isinstance(ds, xr.Dataset)
        assert ds["pop_prop_subgroup"] is not None

    def test_multiple_population_proportions(self, generate_data_partial):
        """Test DataLoader with multiple PopulationProportion objects."""
        df_part, df_cnt, df_pop_prop = generate_data_partial
        
        # Add another stratification variable
        df_part['region'] = pd.Categorical(
            np.random.choice(['North', 'South'], size=len(df_part)),
            categories=['North', 'South'],
            ordered=True
        )
        df_cnt = df_cnt.merge(
            df_part[['id', 'region']],
            on='id',
            how='left'
        )
        
        # Create region-stratified population
        df_pop_region = pd.DataFrame({
            'age': np.tile(np.arange(76), 2),
            'region': ['North'] * 76 + ['South'] * 76,
            'proportion': [0.55] * 76 + [0.45] * 76
        })
        
        pop_prop_subgroup = PopulationProportion(
            data=df_pop_prop,
            age_col='age',
            stratify_by='subgroup',
            proportion_col='prop'
        )
        
        pop_prop_region = PopulationProportion(
            data=df_pop_region,
            age_col='age',
            stratify_by='region',
            proportion_col='proportion'
        )
        
        colmap = CoordToColumns(
            age_part="age_group",
            age_cnt="age_cnt",
            age_pop="age",
            size_pop="P",
            grp_vars_part=['subgroup', 'region'],
        )
        
        dataloader = DataLoader(
            part=df_part,
            cnt=df_cnt,
            pop=df_age_dist,
            col_map=colmap,
            population_proportions=[pop_prop_subgroup, pop_prop_region]
        )
        
        ds = dataloader.load()
        
        assert isinstance(ds, xr.Dataset)
        assert ds["pop_prop_subgroup"] is not None
        assert ds["pop_prop_region"] is not None

    def test_legacy_api_deprecation_warning(self, generate_data_partial):
        """Test that legacy API raises deprecation warning."""
        df_part, df_cnt, df_pop_prop = generate_data_partial
        
        colmap = CoordToColumns(
            age_part="age_group",
            age_cnt="age_cnt",
            age_pop="age",
            size_pop="P",
            grp_vars_part="subgroup",
        )
        
        # Use legacy API - should raise DeprecationWarning
        with pytest.warns(DeprecationWarning, match="pop_prop_quads.*deprecated"):
            dataloader = DataLoader(
                part=df_part,
                cnt=df_cnt,
                pop=df_age_dist,
                col_map=colmap,
                pop_prop_quads=[(df_pop_prop, "age", "subgroup", "prop")]
            )
        
        ds = dataloader.load()
        assert isinstance(ds, xr.Dataset)

    def test_cannot_use_both_apis(self, generate_data_partial):
        """Test that using both APIs raises ValueError."""
        df_part, df_cnt, df_pop_prop = generate_data_partial
        
        pop_prop = PopulationProportion(
            data=df_pop_prop,
            age_col='age',
            stratify_by='subgroup',
            proportion_col='prop'
        )
        
        colmap = CoordToColumns(
            age_part="age_group",
            age_cnt="age_cnt",
            age_pop="age",
            size_pop="P",
            grp_vars_part="subgroup",
        )
        
        # Try to use both APIs - should raise ValueError
        with pytest.raises(ValueError, match="Cannot specify both"):
            DataLoader(
                part=df_part,
                cnt=df_cnt,
                pop=df_age_dist,
                col_map=colmap,
                population_proportions=[pop_prop],
                pop_prop_quads=[(df_pop_prop, "age", "subgroup", "prop")]
            )

    def test_invalid_population_proportion_type(self, generate_data_partial):
        """Test that passing non-PopulationProportion objects raises TypeError."""
        df_part, df_cnt, df_pop_prop = generate_data_partial
        
        colmap = CoordToColumns(
            age_part="age_group",
            age_cnt="age_cnt",
            age_pop="age",
            size_pop="P",
            grp_vars_part="subgroup",
        )
        
        # Try to pass a tuple instead of PopulationProportion
        with pytest.raises(TypeError, match="must be a PopulationProportion"):
            DataLoader(
                part=df_part,
                cnt=df_cnt,
                pop=df_age_dist,
                col_map=colmap,
                population_proportions=[(df_pop_prop, "age", "subgroup", "prop")]
            )

