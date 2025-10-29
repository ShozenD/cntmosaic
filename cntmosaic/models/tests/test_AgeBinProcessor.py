import pytest
import numpy as np
import pandas as pd
from ...utils import AgeBins
from .._SocialMix import AgeBinProcessor

# Language: python

# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def simple_age_bins():
    """Create simple age bins for testing."""
    return AgeBins(min=0, max=40, cuts=[10, 20, 30])


@pytest.fixture
def sample_dataframe():
    """Create sample dataframe with ages."""
    return pd.DataFrame({"id": [1, 2, 3, 4, 5, 6], "age": [5, 15, 25, 35, 8, 22]})


@pytest.fixture
def dataframe_with_age_groups():
    """Create dataframe with existing age groups."""
    return pd.DataFrame(
        {
            "id": [1, 2, 3, 4, 5, 6],
            "age_grp": [
                pd.Interval(0, 10, closed="left"),
                pd.Interval(10, 20, closed="left"),
                pd.Interval(20, 30, closed="left"),
                pd.Interval(30, 40, closed="left"),
                pd.Interval(0, 10, closed="left"),
                pd.Interval(20, 30, closed="left"),
            ],
        }
    )


# ============================================================================
# Test assign_age_groups
# ============================================================================


class TestAssignAgeGroups:
    """Test assign_age_groups method."""

    def test_basic_assignment(self, simple_age_bins, sample_dataframe):
        """Test basic age group assignment."""
        processor = AgeBinProcessor(simple_age_bins)
        result = processor.assign_age_groups(sample_dataframe, "age", "age_grp")

        assert "age_grp" in result.columns
        assert len(result) == len(sample_dataframe)
        assert result["age_grp"].dtype.name == "category"

    def test_assignment_correctness(self, simple_age_bins, sample_dataframe):
        """Test that ages are assigned to correct groups."""
        processor = AgeBinProcessor(simple_age_bins)
        result = processor.assign_age_groups(sample_dataframe, "age", "age_grp")

        # Check specific assignments
        assert result.loc[result["age"] == 5, "age_grp"].iloc[0] == pd.Interval(
            0, 10, closed="left"
        )
        assert result.loc[result["age"] == 15, "age_grp"].iloc[0] == pd.Interval(
            10, 20, closed="left"
        )
        assert result.loc[result["age"] == 25, "age_grp"].iloc[0] == pd.Interval(
            20, 30, closed="left"
        )
        assert result.loc[result["age"] == 35, "age_grp"].iloc[0] == pd.Interval(
            30, 41, closed="left"
        )

    def test_number_of_categories(self, simple_age_bins, sample_dataframe):
        """Test that correct number of categories are created."""
        processor = AgeBinProcessor(simple_age_bins)
        result = processor.assign_age_groups(sample_dataframe, "age", "age_grp")

        # Should have 4 age groups: [0,10), [10,20), [20,30), [30,41)
        assert len(result["age_grp"].cat.categories) == 4

    def test_does_not_modify_original(self, simple_age_bins, sample_dataframe):
        """Test that original dataframe is not modified."""
        processor = AgeBinProcessor(simple_age_bins)
        original_columns = sample_dataframe.columns.tolist()

        result = processor.assign_age_groups(sample_dataframe, "age", "age_grp")

        assert sample_dataframe.columns.tolist() == original_columns
        assert "age_grp" not in sample_dataframe.columns

    def test_boundary_values(self, simple_age_bins):
        """Test assignment of boundary values."""
        processor = AgeBinProcessor(simple_age_bins)
        df = pd.DataFrame({"age": [0, 10, 20, 30, 39]})
        result = processor.assign_age_groups(df, "age", "age_grp")

        # 0 should be in [0, 10)
        assert result.loc[result["age"] == 0, "age_grp"].iloc[0] == pd.Interval(
            0, 10, closed="left"
        )
        # 10 should be in [10, 20)
        assert result.loc[result["age"] == 10, "age_grp"].iloc[0] == pd.Interval(
            10, 20, closed="left"
        )
        # 39 should be in [30, 41)
        assert result.loc[result["age"] == 39, "age_grp"].iloc[0] == pd.Interval(
            30, 41, closed="left"
        )


# ============================================================================
# Test merge_zero_groups
# ============================================================================


class TestMergeZeroGroups:
    """Test merge_zero_groups static method."""

    def test_no_zeros_returns_original(self):
        """Test that intervals without zeros are unchanged."""
        intervals = [
            pd.Interval(0, 10, closed="left"),
            pd.Interval(10, 20, closed="left"),
            pd.Interval(20, 30, closed="left"),
        ]
        counts = np.array([5, 10, 8])

        merged = AgeBinProcessor.merge_zero_groups(intervals, counts)

        assert len(merged) == 3
        assert merged == intervals

    def test_single_zero_merged_with_previous(self):
        """Test that single zero is merged with previous non-zero."""
        intervals = [
            pd.Interval(0, 10, closed="left"),
            pd.Interval(10, 20, closed="left"),
            pd.Interval(20, 30, closed="left"),
        ]
        counts = np.array([10, 0, 15])

        merged = AgeBinProcessor.merge_zero_groups(intervals, counts)

        assert len(merged) == 2
        assert merged[0] == pd.Interval(0, 20, closed="left")
        assert merged[1] == pd.Interval(20, 30, closed="left")

    def test_multiple_consecutive_zeros(self):
        """Test merging of multiple consecutive zeros."""
        intervals = [
            pd.Interval(0, 10, closed="left"),
            pd.Interval(10, 20, closed="left"),
            pd.Interval(20, 30, closed="left"),
            pd.Interval(30, 40, closed="left"),
        ]
        counts = np.array([10, 0, 0, 15])

        merged = AgeBinProcessor.merge_zero_groups(intervals, counts)

        assert len(merged) == 2
        assert merged[0] == pd.Interval(0, 30, closed="left")
        assert merged[1] == pd.Interval(30, 40, closed="left")

    def test_leading_zeros(self):
        """Test that leading zeros are merged with first non-zero."""
        intervals = [
            pd.Interval(0, 10, closed="left"),
            pd.Interval(10, 20, closed="left"),
            pd.Interval(20, 30, closed="left"),
        ]
        counts = np.array([0, 0, 15])

        merged = AgeBinProcessor.merge_zero_groups(intervals, counts)

        assert len(merged) == 1
        assert merged[0] == pd.Interval(0, 30, closed="left")

    def test_trailing_zeros(self):
        """Test that trailing zeros are merged with previous."""
        intervals = [
            pd.Interval(0, 10, closed="left"),
            pd.Interval(10, 20, closed="left"),
            pd.Interval(20, 30, closed="left"),
        ]
        counts = np.array([10, 0, 0])

        merged = AgeBinProcessor.merge_zero_groups(intervals, counts)

        assert len(merged) == 1
        assert merged[0] == pd.Interval(0, 30, closed="left")

    def test_alternating_zeros(self):
        """Test merging with alternating zero and non-zero groups."""
        intervals = [
            pd.Interval(0, 10, closed="left"),
            pd.Interval(10, 20, closed="left"),
            pd.Interval(20, 30, closed="left"),
            pd.Interval(30, 40, closed="left"),
        ]
        counts = np.array([10, 0, 15, 0])

        merged = AgeBinProcessor.merge_zero_groups(intervals, counts)

        assert len(merged) == 2
        assert merged[0] == pd.Interval(0, 20, closed="left")
        assert merged[1] == pd.Interval(20, 40, closed="left")

    def test_all_zeros_raises_error(self):
        """Test that all zeros raises ValueError."""
        intervals = [
            pd.Interval(0, 10, closed="left"),
            pd.Interval(10, 20, closed="left"),
        ]
        counts = np.array([0, 0])

        with pytest.raises(ValueError, match="All age groups have zero"):
            AgeBinProcessor.merge_zero_groups(intervals, counts)

    def test_single_non_zero_group(self):
        """Test with only one non-zero group."""
        intervals = [
            pd.Interval(0, 10, closed="left"),
            pd.Interval(10, 20, closed="left"),
            pd.Interval(20, 30, closed="left"),
        ]
        counts = np.array([0, 10, 0])

        merged = AgeBinProcessor.merge_zero_groups(intervals, counts)

        assert len(merged) == 1
        assert merged[0] == pd.Interval(0, 30, closed="left")


# ============================================================================
# Test reassign_age_groups
# ============================================================================


class TestReassignAgeGroups:
    """Test reassign_age_groups static method."""

    def test_basic_reassignment(self, dataframe_with_age_groups):
        """Test basic reassignment of age groups."""
        merged_intervals = [
            pd.Interval(0, 20, closed="left"),
            pd.Interval(20, 40, closed="left"),
        ]

        result = AgeBinProcessor.reassign_age_groups(
            dataframe_with_age_groups, "age_grp", merged_intervals, "age_grp_new"
        )

        assert "age_grp_new" in result.columns
        assert len(result) == len(dataframe_with_age_groups)
        assert result["age_grp_new"].dtype.name == "category"

    def test_reassignment_correctness(self, dataframe_with_age_groups):
        """Test that reassignment maps correctly."""
        merged_intervals = [
            pd.Interval(0, 20, closed="left"),
            pd.Interval(20, 40, closed="left"),
        ]

        result = AgeBinProcessor.reassign_age_groups(
            dataframe_with_age_groups, "age_grp", merged_intervals, "age_grp_new"
        )

        # Original [0,10) and [10,20) should map to [0,20)
        mask_0_10 = dataframe_with_age_groups["age_grp"] == pd.Interval(
            0, 10, closed="left"
        )
        assert all(
            result.loc[mask_0_10, "age_grp_new"] == pd.Interval(0, 20, closed="left")
        )

        mask_10_20 = dataframe_with_age_groups["age_grp"] == pd.Interval(
            10, 20, closed="left"
        )
        assert all(
            result.loc[mask_10_20, "age_grp_new"] == pd.Interval(0, 20, closed="left")
        )

        # Original [20,30) and [30,40) should map to [20,40)
        mask_20_30 = dataframe_with_age_groups["age_grp"] == pd.Interval(
            20, 30, closed="left"
        )
        assert all(
            result.loc[mask_20_30, "age_grp_new"] == pd.Interval(20, 40, closed="left")
        )

        mask_30_40 = dataframe_with_age_groups["age_grp"] == pd.Interval(
            30, 40, closed="left"
        )
        assert all(
            result.loc[mask_30_40, "age_grp_new"] == pd.Interval(20, 40, closed="left")
        )

    def test_overwrite_original_column(self, dataframe_with_age_groups):
        """Test that original column can be overwritten."""
        merged_intervals = [
            pd.Interval(0, 20, closed="left"),
            pd.Interval(20, 40, closed="left"),
        ]

        result = AgeBinProcessor.reassign_age_groups(
            dataframe_with_age_groups, "age_grp", merged_intervals
        )

        # Should have only 2 unique categories now
        assert len(result["age_grp"].cat.categories) == 2
        assert result["age_grp"].cat.categories[0] == pd.Interval(0, 20, closed="left")
        assert result["age_grp"].cat.categories[1] == pd.Interval(20, 40, closed="left")

    def test_does_not_modify_original(self, dataframe_with_age_groups):
        """Test that original dataframe is not modified."""
        merged_intervals = [
            pd.Interval(0, 20, closed="left"),
            pd.Interval(20, 40, closed="left"),
        ]

        original_age_grps = dataframe_with_age_groups["age_grp"].copy()

        result = AgeBinProcessor.reassign_age_groups(
            dataframe_with_age_groups, "age_grp", merged_intervals, "age_grp_new"
        )

        # Original should be unchanged
        pd.testing.assert_series_equal(
            dataframe_with_age_groups["age_grp"], original_age_grps
        )

    def test_no_merge_returns_same_groups(self, dataframe_with_age_groups):
        """Test that no merging returns same categories."""
        merged_intervals = [
            pd.Interval(0, 10, closed="left"),
            pd.Interval(10, 20, closed="left"),
            pd.Interval(20, 30, closed="left"),
            pd.Interval(30, 40, closed="left"),
        ]

        result = AgeBinProcessor.reassign_age_groups(
            dataframe_with_age_groups, "age_grp", merged_intervals, "age_grp_new"
        )

        # Should be identical to original
        assert all(result["age_grp_new"] == dataframe_with_age_groups["age_grp"])

    def test_handles_na_values(self):
        """Test that NA values are handled gracefully."""
        df = pd.DataFrame(
            {
                "id": [1, 2, 3],
                "age_grp": [
                    pd.Interval(0, 10, closed="left"),
                    pd.NA,
                    pd.Interval(10, 20, closed="left"),
                ],
            }
        )

        merged_intervals = [pd.Interval(0, 20, closed="left")]

        result = AgeBinProcessor.reassign_age_groups(
            df, "age_grp", merged_intervals, "age_grp_new"
        )

        assert pd.isna(result.loc[1, "age_grp_new"])
        assert result.loc[0, "age_grp_new"] == pd.Interval(0, 20, closed="left")
        assert result.loc[2, "age_grp_new"] == pd.Interval(0, 20, closed="left")

    def test_invalid_interval_raises_error(self, dataframe_with_age_groups):
        """Test that interval not fitting raises error."""
        # Merged intervals that don't contain all original intervals
        merged_intervals = [pd.Interval(0, 15, closed="left")]

        with pytest.raises(ValueError, match="does not fit within"):
            AgeBinProcessor.reassign_age_groups(
                dataframe_with_age_groups, "age_grp", merged_intervals, "age_grp_new"
            )

    def test_categorical_with_correct_order(self, dataframe_with_age_groups):
        """Test that result has correct categorical ordering."""
        merged_intervals = [
            pd.Interval(0, 20, closed="left"),
            pd.Interval(20, 40, closed="left"),
        ]

        result = AgeBinProcessor.reassign_age_groups(
            dataframe_with_age_groups, "age_grp", merged_intervals, "age_grp_new"
        )

        assert result["age_grp_new"].cat.ordered is True
        assert list(result["age_grp_new"].cat.categories) == merged_intervals


# ============================================================================
# Test create_interval_mapping
# ============================================================================


class TestCreateIntervalMapping:
    """Test create_interval_mapping static method."""

    def test_basic_mapping(self):
        """Test basic interval mapping creation."""
        original = [
            pd.Interval(0, 10, closed="left"),
            pd.Interval(10, 20, closed="left"),
            pd.Interval(20, 30, closed="left"),
        ]
        merged = [pd.Interval(0, 20, closed="left"), pd.Interval(20, 30, closed="left")]

        mapping = AgeBinProcessor.create_interval_mapping(original, merged)

        assert len(mapping) == 3
        assert mapping[pd.Interval(0, 10, closed="left")] == pd.Interval(
            0, 20, closed="left"
        )
        assert mapping[pd.Interval(10, 20, closed="left")] == pd.Interval(
            0, 20, closed="left"
        )
        assert mapping[pd.Interval(20, 30, closed="left")] == pd.Interval(
            20, 30, closed="left"
        )

    def test_no_merge_identity_mapping(self):
        """Test that no merging creates identity mapping."""
        intervals = [
            pd.Interval(0, 10, closed="left"),
            pd.Interval(10, 20, closed="left"),
        ]

        mapping = AgeBinProcessor.create_interval_mapping(intervals, intervals)

        assert mapping[pd.Interval(0, 10, closed="left")] == pd.Interval(
            0, 10, closed="left"
        )
        assert mapping[pd.Interval(10, 20, closed="left")] == pd.Interval(
            10, 20, closed="left"
        )

    def test_all_merge_to_one(self):
        """Test mapping when all intervals merge to one."""
        original = [
            pd.Interval(0, 10, closed="left"),
            pd.Interval(10, 20, closed="left"),
            pd.Interval(20, 30, closed="left"),
        ]
        merged = [pd.Interval(0, 30, closed="left")]

        mapping = AgeBinProcessor.create_interval_mapping(original, merged)

        assert len(mapping) == 3
        assert all(v == pd.Interval(0, 30, closed="left") for v in mapping.values())

    def test_invalid_mapping_raises_error(self):
        """Test that unmappable intervals raise error."""
        original = [
            pd.Interval(0, 10, closed="left"),
            pd.Interval(10, 20, closed="left"),
        ]
        merged = [pd.Interval(0, 5, closed="left")]  # Doesn't cover all originals

        with pytest.raises(ValueError, match="does not fit within"):
            AgeBinProcessor.create_interval_mapping(original, merged)


# ============================================================================
# Integration Tests
# ============================================================================


class TestAgeBinProcessorIntegration:
    """Integration tests combining multiple methods."""

    def test_full_workflow(self, simple_age_bins, sample_dataframe):
        """Test complete workflow: assign -> merge -> reassign."""
        processor = AgeBinProcessor(simple_age_bins)

        # Step 1: Assign initial age groups
        df = processor.assign_age_groups(sample_dataframe, "age", "age_grp")

        # Step 2: Compute sample sizes
        sample_sizes = df.groupby("age_grp", observed=False).size()

        # Step 3: Simulate zero group (manually set one to zero for testing)
        counts = sample_sizes.values.copy()
        counts[1] = 0  # Set second age group to zero

        # Step 4: Merge zero groups
        merged_intervals = AgeBinProcessor.merge_zero_groups(
            sample_sizes.index.tolist(), counts
        )

        # Step 5: Reassign based on merged intervals
        df_reassigned = AgeBinProcessor.reassign_age_groups(
            df, "age_grp", merged_intervals, "age_grp_merged"
        )

        # Verify result
        assert "age_grp_merged" in df_reassigned.columns
        assert len(df_reassigned) == len(sample_dataframe)
        assert len(df_reassigned["age_grp_merged"].cat.categories) == len(
            merged_intervals
        )

    def test_multiple_dataframes_same_mapping(self, simple_age_bins):
        """Test applying same mapping to multiple dataframes."""
        processor = AgeBinProcessor(simple_age_bins)

        # Create two dataframes
        df1 = pd.DataFrame({"id": [1, 2], "age": [5, 15]})
        df2 = pd.DataFrame({"id": [3, 4], "age": [25, 35]})

        # Assign age groups to both
        df1 = processor.assign_age_groups(df1, "age", "age_grp")
        df2 = processor.assign_age_groups(df2, "age", "age_grp")

        # Create merged intervals
        merged_intervals = [
            pd.Interval(0, 20, closed="left"),
            pd.Interval(20, 41, closed="left"),
        ]

        # Reassign both with same mapping
        df1_new = AgeBinProcessor.reassign_age_groups(
            df1, "age_grp", merged_intervals, "age_grp_new"
        )
        df2_new = AgeBinProcessor.reassign_age_groups(
            df2, "age_grp", merged_intervals, "age_grp_new"
        )

        # Both should have same categories
        assert list(df1_new["age_grp_new"].cat.categories) == list(
            df2_new["age_grp_new"].cat.categories
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
