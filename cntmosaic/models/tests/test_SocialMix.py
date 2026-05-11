import numpy as np
import pytest

from ...analysis.summariser import ModelSummariserSocialMix
from ...utils import AgeBins
from .._SocialMix import SocialMix
from .fixtures import (
    full_large_sample,
    full_multi_strat_large_sample,
    full_small_sample,
    partial_large_sample,
    partial_multi_strat_large_sample,
    partial_small_sample,
    single_large_sample,
    single_small_sample,
)

# Language: python


class TestInit:

    def test_single(self, single_large_sample):
        part_data, cnt_data, pop_data = single_large_sample

        age_bins = AgeBins(0, 80, 5)
        sm = SocialMix(part_data, cnt_data, age_bins, pop_data)

        assert sm.K == 1
        assert sm.strat_mode == "single"
        assert sm.apply_reciprocity is True

    def test_single_no_pop(self, single_large_sample):
        part_data, cnt_data, _ = single_large_sample

        age_bins = AgeBins(0, 80, 5)

        with pytest.warns(
            UserWarning,
            match="Reciprocity adjustment requested but no population data provided",
        ):
            sm = SocialMix(part_data, cnt_data, age_bins, apply_reciprocity=True)

        assert sm.K == 1
        assert sm.strat_mode == "single"
        assert sm.apply_reciprocity is False

    def test_partial(self, partial_large_sample):
        part_data, cnt_data, pop_data = partial_large_sample

        # Define age bins
        age_bins = AgeBins(0, 80, 5)

        # Initialize SocialMix
        sm = SocialMix(part_data, cnt_data, age_bins, pop_data, apply_reciprocity=False)

        assert sm.K == 2
        assert sm.strat_mode == "partial"
        assert sm.strat_vars_part == ["sex"]
        assert sm.strat_vars_cnt == []

    def test_partial_multi_strat(self, partial_multi_strat_large_sample):
        part_data, cnt_data, pop_data = partial_multi_strat_large_sample

        # Define age bins
        age_bins = AgeBins(0, 80, 5)

        # Initialize SocialMix
        with pytest.warns(
            UserWarning,
            match="Reciprocity adjustment is not applicable for partial stratification",
        ):
            sm = SocialMix(part_data, cnt_data, age_bins, pop_data)

        assert sm.K == 4
        assert sm.strat_mode == "partial"
        assert sm.strat_vars_part == ["sex", "ses"]
        assert sm.strat_vars_cnt == []
        assert sm.strat_vars_shared == []

    def test_full(self, full_large_sample):
        part_data, cnt_data, pop_data = full_large_sample

        # Define age bins
        age_bins = AgeBins(0, 80, 5)

        # Initialize SocialMix
        sm = SocialMix(part_data, cnt_data, age_bins, pop_data)

        assert sm.K == 4
        assert sm.strat_mode == "full"
        assert sm.strat_vars_part == ["sex"]
        assert sm.strat_vars_cnt == ["sex"]
        assert sm.strat_vars_shared == ["sex"]

    def test_full_multi_strat(self, full_multi_strat_large_sample):
        part_data, cnt_data, pop_data = full_multi_strat_large_sample
        age_bins = AgeBins(0, 80, 5)
        sm = SocialMix(part_data, cnt_data, age_bins, pop_data)

        assert sm.K == 16
        assert sm.strat_mode == "full"
        assert sm.strat_vars_part == ["sex", "ses"]
        assert sm.strat_vars_cnt == ["sex", "ses"]
        assert sm.strat_vars_shared == sorted(["sex", "ses"])


class TestCint:
    """Tests for contact intensity matrix computation"""

    def test_cint_single_no_reciprocity(self, single_large_sample):
        """Contact intensity shape and non-negativity without reciprocity"""
        part_data, cnt_data, _ = single_large_sample
        age_bins = AgeBins(0, 80, 5)

        sm = SocialMix(part_data, cnt_data, age_bins, apply_reciprocity=False)
        cint_dict = sm.cint()

        # Check structure
        assert "All->All" in cint_dict
        assert len(cint_dict) == 1

        M = cint_dict["All->All"]
        assert M.shape == (16, 16)  # 0-80 in 5-year bins = 16 groups
        assert np.all(M >= 0), "Contact intensities must be non-negative"
        assert not np.all(M == 0), "Should have some non-zero contacts"

    def test_cint_reciprocity_single(self, single_large_sample):
        """Reciprocity adjustment satisfies M[c,d]*P[c] ≈ M[d,c]*P[d]"""
        part_data, cnt_data, pop_data = single_large_sample
        age_bins = AgeBins(0, 80, 5)

        sm = SocialMix(part_data, cnt_data, age_bins, pop_data, apply_reciprocity=True)
        M = sm.cint()["All->All"]

        # Check reciprocity constraint: M[c,d]*P[c] ≈ M[d,c]*P[d]
        P = sm.P
        left = M * P[:, np.newaxis]  # Broadcasting P along rows
        right = M.T * P[np.newaxis, :]  # Broadcasting P along columns (transposed)

        np.testing.assert_allclose(
            left, right, rtol=1e-8, err_msg="Reciprocity constraint violated"
        )

    def test_cint_reciprocity_full(self, full_large_sample):
        """Reciprocity works correctly with full stratification"""
        part_data, cnt_data, pop_data = full_large_sample

        age_bins = AgeBins(0, 80, 5)

        sm = SocialMix(part_data, cnt_data, age_bins, pop_data, apply_reciprocity=True)
        cint_dict = sm.cint()

        # Check within-stratum reciprocity (M->M, F->F)
        for stratum in ["M->M", "F->F"]:
            M = cint_dict[stratum]
            k = 1 if stratum == "M->M" else 0
            P = sm.P[k, :]

            left = M * P[:, np.newaxis]
            right = M.T * P[np.newaxis, :]
            np.testing.assert_allclose(
                left, right, rtol=1e-8, err_msg=f"{stratum} reciprocity failed"
            )

        # Check between-stratum reciprocity: M[M->F] * P[M] ≈ M[F->M] * P[F]
        M_MF = cint_dict["M->F"]
        M_FM = cint_dict["F->M"]
        P_F = sm.P[0, :]
        P_M = sm.P[1, :]

        left = M_MF / P_F[np.newaxis, :]
        right = (M_FM / P_M[np.newaxis, :]).T
        np.testing.assert_allclose(
            left, right, rtol=1e-8, err_msg="Between-stratum reciprocity failed"
        )

    def test_cint_reciprocity_full_multi_strat(self, full_multi_strat_large_sample):
        part_data, cnt_data, pop_data = full_multi_strat_large_sample

        # Define age bins
        age_bins = AgeBins(0, 80, 5)

        # Initialize SocialMix
        sm = SocialMix(part_data, cnt_data, age_bins, pop_data, apply_reciprocity=True)

        # Compute contact intensity matrices
        cint_dict = sm.cint()

        # Check within-stratum reciprocity (M->M, F->F)
        for k, stratum in enumerate(
            [
                "F_High->F_High",
                "F_Low->F_Low",
                "M_High->M_High",
                "M_Low->M_Low",
            ]
        ):
            M = cint_dict[stratum]
            P = sm.P[k, :]

            left = M * P[:, np.newaxis]
            right = M.T * P[np.newaxis, :]
            np.testing.assert_allclose(
                left, right, rtol=1e-8, err_msg=f"{stratum} reciprocity failed"
            )

        # Check between-stratum reciprocity for a few combinations
        combinations = [
            ("M_High->F_Low", "F_Low->M_High", 2, 1),
            ("M_Low->F_High", "F_High->M_Low", 3, 0),
        ]

        for stratum_A, stratum_B, idx_A, idx_B in combinations:
            M_A = cint_dict[stratum_A]
            M_B = cint_dict[stratum_B]
            P_A = sm.P[idx_A, :]
            P_B = sm.P[idx_B, :]

            left = M_A / P_B[np.newaxis, :]
            right = (M_B / P_A[np.newaxis, :]).T
            np.testing.assert_allclose(
                left,
                right,
                rtol=1e-8,
                err_msg=f"{stratum_A} and {stratum_B} reciprocity failed",
            )


class TestSmallSample:
    """Tests for small sample behavior"""

    def test_cint_single_small_sample(self, single_small_sample):
        """Contact intensity computation on a very small sample"""

        part_data, cnt_data, pop_data = single_small_sample

        # Define age bins
        age_bins = AgeBins(0, 80, 5)

        # When adaptive_merge is False will raise error due to insufficient data
        with pytest.raises(ValueError):
            sm = SocialMix(
                part_data, cnt_data, age_bins, pop_data, adaptive_merge=False
            )

        # Initialize SocialMix without population data
        with pytest.warns(UserWarning):
            sm = SocialMix(part_data, cnt_data, age_bins, pop_data, adaptive_merge=True)

        # Compute contact intensity (check that this runs without error)
        cint_dict = sm.cint()

        # Check structure and non-negativity
        assert "All->All" in cint_dict
        M = cint_dict["All->All"]
        assert M.shape[0] <= 16, "Age bins should be merged for small sample"
        assert M.shape[1] <= 16, "Age bins should be merged for small sample"
        assert np.all(M >= 0), "Contact intensities must be non-negative"

    def test_cint_partial_small_sample(self, partial_small_sample):
        """Contact intensity computation on a small sample with partial stratification"""

        part_data, cnt_data, pop_data = partial_small_sample

        # Define age bins
        age_bins = AgeBins(0, 80, 5)

        with pytest.warns(UserWarning):
            sm = SocialMix(part_data, cnt_data, age_bins, pop_data, adaptive_merge=True)

        # Compute contact intensity (check that this runs without error)
        cint_dict = sm.cint()

        # Check structure and non-negativity
        for stratum in ["M->All", "F->All"]:
            assert stratum in cint_dict
            M = cint_dict[stratum]
            assert M.shape[0] <= 16, "Age bins should be merged for small sample"
            assert M.shape[1] <= 16, "Age bins should be merged for small sample"
            assert np.all(M >= 0), "Contact intensities must be non-negative"

    def test_cint_full_small_sample(self, full_small_sample):
        """Contact intensity computation on a small sample with full stratification"""

        part_data, cnt_data, pop_data = full_small_sample

        # Define age bins
        age_bins = AgeBins(0, 80, 5)

        with pytest.warns(UserWarning):
            sm = SocialMix(part_data, cnt_data, age_bins, pop_data, adaptive_merge=True)

        # Compute contact intensity (check that this runs without error)
        cint_dict = sm.cint()

        # Check structure and non-negativity
        for stratum in ["M->M", "M->F", "F->M", "F->F"]:
            assert stratum in cint_dict
            M = cint_dict[stratum]
            assert M.shape[0] <= 16, "Age bins should be merged for small sample"
            assert M.shape[1] <= 16, "Age bins should be merged for small sample"
            assert np.all(M >= 0), "Contact intensities must be non-negative"


class TestBootstrap:
    """Tests for bootstrap resampling"""

    N_BOOT = 10

    def test_bootstrap_single(self, single_large_sample):
        part_data, cnt_data, pop_data = single_large_sample
        age_bins = AgeBins(0, 80, 5)

        sm = SocialMix(
            part_data,
            cnt_data,
            age_bins,
            pop_data,
            adaptive_merge=True,
            validate_for_bootstrap=True,
        )

        sm.run_inference_bootstrap(n_boot=self.N_BOOT, random_state=42)

        assert sm._boot.n_boot == self.N_BOOT

        M = sm._boot.mean()
        assert M["All->All"].shape == (16, 16)

    def test_bootstrap_single_small(self, single_small_sample):
        part_data, cnt_data, pop_data = single_small_sample
        age_bins = AgeBins(0, 80, 5)

        with pytest.warns(UserWarning):
            sm = SocialMix(
                part_data,
                cnt_data,
                age_bins,
                pop_data,
                adaptive_merge=True,
                validate_for_bootstrap=True,
            )

        sm.run_inference_bootstrap(n_boot=self.N_BOOT, random_state=42)

        assert sm._boot.n_boot == self.N_BOOT

        M = sm._boot.mean()
        assert M["All->All"].shape[0] <= 16
        assert M["All->All"].shape[1] <= 16

    def test_boostrap_partial(self, partial_large_sample):
        part_data, cnt_data, pop_data = partial_large_sample
        age_bins = AgeBins(0, 80, 5)

        sm = SocialMix(
            part_data,
            cnt_data,
            age_bins,
            pop_data,
            adaptive_merge=True,
            apply_reciprocity=False,
            validate_for_bootstrap=True,
        )

        sm.run_inference_bootstrap(n_boot=self.N_BOOT, random_state=42)

        assert sm._boot.n_boot == self.N_BOOT

        M_dict = sm._boot.mean()
        assert M_dict["M->All"].shape == (16, 16)
        assert M_dict["F->All"].shape == (16, 16)

    def test_bootstrap_partial_multi_strat(self, partial_multi_strat_large_sample):
        part_data, cnt_data, pop_data = partial_multi_strat_large_sample
        age_bins = AgeBins(0, 80, 5)

        with pytest.warns(UserWarning):
            sm = SocialMix(
                part_data,
                cnt_data,
                age_bins,
                pop_data,
                adaptive_merge=True,
                apply_reciprocity=False,
                validate_for_bootstrap=True,
            )

        sm.run_inference_bootstrap(n_boot=self.N_BOOT, random_state=42)

        assert sm._boot.n_boot == self.N_BOOT

        M_dict = sm._boot.mean()
        for stratum in ["M_High->All", "M_Low->All", "F_High->All", "F_Low->All"]:
            assert M_dict[stratum].shape[0] <= 16
            assert M_dict[stratum].shape[1] <= 16

    def test_bootstrap_partial_small(self, partial_small_sample):
        part_data, cnt_data, pop_data = partial_small_sample
        age_bins = AgeBins(0, 80, 5)

        with pytest.warns(UserWarning):
            sm = SocialMix(
                part_data,
                cnt_data,
                age_bins,
                pop_data,
                adaptive_merge=True,
                apply_reciprocity=False,
                validate_for_bootstrap=True,
            )

        sm.run_inference_bootstrap(n_boot=self.N_BOOT, random_state=42)

        assert sm._boot.n_boot == self.N_BOOT

        M_dict = sm._boot.mean()
        for stratum in ["M->All", "F->All"]:
            assert M_dict[stratum].shape[0] <= 16
            assert M_dict[stratum].shape[1] <= 16

    def test_bootstrap_full(self, full_large_sample):
        part_data, cnt_data, pop_data = full_large_sample
        age_bins = AgeBins(0, 80, 5)

        sm = SocialMix(
            part_data,
            cnt_data,
            age_bins,
            pop_data,
            adaptive_merge=True,
            validate_for_bootstrap=True,
        )

        sm.run_inference_bootstrap(n_boot=self.N_BOOT, random_state=42)

        assert sm._boot.n_boot == self.N_BOOT

        M_dict = sm._boot.mean()
        for stratum in ["M->M", "M->F", "F->M", "F->F"]:
            assert M_dict[stratum].shape == (16, 16)

    def test_bootstrap_full_multi_strat(self, full_multi_strat_large_sample):
        part_data, cnt_data, pop_data = full_multi_strat_large_sample
        age_bins = AgeBins(0, 80, 5)

        with pytest.warns(UserWarning):
            sm = SocialMix(
                part_data,
                cnt_data,
                age_bins,
                pop_data,
                adaptive_merge=True,
                validate_for_bootstrap=True,
            )

        sm.run_inference_bootstrap(n_boot=self.N_BOOT, random_state=42)

        assert sm._boot.n_boot == self.N_BOOT

        M_dict = sm._boot.mean()
        for stratum in [
            "M_High->M_High",
            "M_High->M_Low",
            "M_Low->M_High",
            "M_Low->M_Low",
            "F_High->F_High",
            "F_High->F_Low",
            "F_Low->F_High",
            "F_Low->F_Low",
        ]:
            assert M_dict[stratum].shape[0] <= 16
            assert M_dict[stratum].shape[1] <= 16

    def test_bootstrap_full_small(self, full_small_sample):
        part_data, cnt_data, pop_data = full_small_sample
        age_bins = AgeBins(0, 80, 5)

        with pytest.warns(UserWarning):
            sm = SocialMix(
                part_data,
                cnt_data,
                age_bins,
                pop_data,
                adaptive_merge=True,
                validate_for_bootstrap=True,
            )

        sm.run_inference_bootstrap(n_boot=self.N_BOOT, random_state=42)

        assert sm._boot.n_boot == self.N_BOOT

        M_dict = sm._boot.mean()
        for stratum in ["M->M", "M->F", "F->M", "F->F"]:
            assert M_dict[stratum].shape[0] <= 16
            assert M_dict[stratum].shape[1] <= 16


class TestSummarizerIntegration:
    N_BOOT = 500

    def test_single(self, single_large_sample):
        part_data, cnt_data, pop_data = single_large_sample
        age_bins = AgeBins(0, 80, 5)

        sm = SocialMix(
            part_data,
            cnt_data,
            age_bins,
            pop_data,
            adaptive_merge=True,
            validate_for_bootstrap=True,
        )

        sm.run_inference_bootstrap(n_boot=self.N_BOOT, random_state=42)
        summarizer = ModelSummariserSocialMix(sm)

        cint_sum = summarizer.summarise_cint(alpha=0.05)
        assert "All->All" in cint_sum
        assert cint_sum["All->All"][1].shape[0] <= 16
        assert cint_sum["All->All"][1].shape[1] <= 16

        cint_sum_depix = summarizer.summarise_cint(alpha=0.05, return_depixilated=True)
        assert cint_sum_depix["All->All"].shape == (3, 81, 81)

    def test_single_small(self, single_small_sample):
        part_data, cnt_data, pop_data = single_small_sample
        age_bins = AgeBins(0, 80, 5)

        with pytest.warns(UserWarning):
            sm = SocialMix(
                part_data,
                cnt_data,
                age_bins,
                pop_data,
                adaptive_merge=True,
                validate_for_bootstrap=True,
            )

        sm.run_inference_bootstrap(n_boot=self.N_BOOT, random_state=42)
        summarizer = ModelSummariserSocialMix(sm)

        cint_sum = summarizer.summarise_cint(alpha=0.05)
        assert "All->All" in cint_sum
        assert cint_sum["All->All"][1].shape[0] <= 16
        assert cint_sum["All->All"][1].shape[1] <= 16

        cint_sum_depix = summarizer.summarise_cint(alpha=0.05, return_depixilated=True)
        assert cint_sum_depix["All->All"].shape == (3, 81, 81)

    def test_partial(self, partial_large_sample):
        part_data, cnt_data, pop_data = partial_large_sample
        age_bins = AgeBins(0, 80, 5)

        sm = SocialMix(
            part_data,
            cnt_data,
            age_bins,
            pop_data,
            adaptive_merge=True,
            apply_reciprocity=False,
            validate_for_bootstrap=True,
        )

        sm.run_inference_bootstrap(n_boot=self.N_BOOT, random_state=42)
        summarizer = ModelSummariserSocialMix(sm)

        cint_sum = summarizer.summarise_cint(alpha=0.05)
        for stratum in ["M->All", "F->All"]:
            assert stratum in cint_sum
            assert cint_sum[stratum][1].shape[0] <= 16
            assert cint_sum[stratum][1].shape[1] <= 16

        cint_sum_depix = summarizer.summarise_cint(alpha=0.05, return_depixilated=True)
        for stratum in ["M->All", "F->All"]:
            assert cint_sum_depix[stratum].shape == (3, 81, 81)

    def test_partial_small(self, partial_small_sample):
        part_data, cnt_data, pop_data = partial_small_sample
        age_bins = AgeBins(0, 80, 5)

        with pytest.warns(UserWarning):
            sm = SocialMix(
                part_data,
                cnt_data,
                age_bins,
                pop_data,
                adaptive_merge=True,
                apply_reciprocity=False,
                validate_for_bootstrap=True,
            )

        sm.run_inference_bootstrap(n_boot=self.N_BOOT, random_state=42)
        summarizer = ModelSummariserSocialMix(sm)

        cint_sum = summarizer.summarise_cint(alpha=0.05)
        for stratum in ["M->All", "F->All"]:
            assert stratum in cint_sum
            assert cint_sum[stratum][1].shape[0] <= 16
            assert cint_sum[stratum][1].shape[1] <= 16

        cint_sum_depix = summarizer.summarise_cint(alpha=0.05, return_depixilated=True)
        for stratum in ["M->All", "F->All"]:
            assert cint_sum_depix[stratum].shape == (3, 81, 81)

    def test_full(self, full_large_sample):
        part_data, cnt_data, pop_data = full_large_sample
        age_bins = AgeBins(0, 80, 5)

        sm = SocialMix(
            part_data,
            cnt_data,
            age_bins,
            pop_data,
            adaptive_merge=True,
            validate_for_bootstrap=True,
        )

        sm.run_inference_bootstrap(n_boot=self.N_BOOT, random_state=42)
        summarizer = ModelSummariserSocialMix(sm)

        cint_sum = summarizer.summarise_cint(alpha=0.05)
        for stratum in ["M->M", "M->F", "F->M", "F->F"]:
            assert stratum in cint_sum
            assert cint_sum[stratum][1].shape[0] <= 16
            assert cint_sum[stratum][1].shape[1] <= 16

        cint_sum_depix = summarizer.summarise_cint(alpha=0.05, return_depixilated=True)
        for stratum in ["M->M", "M->F", "F->M", "F->F"]:
            assert cint_sum_depix[stratum].shape == (3, 81, 81)

    def test_full_small(self, full_small_sample):
        part_data, cnt_data, pop_data = full_small_sample
        age_bins = AgeBins(0, 80, 5)

        with pytest.warns(UserWarning):
            sm = SocialMix(
                part_data,
                cnt_data,
                age_bins,
                pop_data,
                adaptive_merge=True,
                validate_for_bootstrap=True,
            )

        sm.run_inference_bootstrap(n_boot=self.N_BOOT, random_state=42)
        summarizer = ModelSummariserSocialMix(sm)

        cint_sum = summarizer.summarise_cint(alpha=0.05)
        for stratum in ["M->M", "M->F", "F->M", "F->F"]:
            assert stratum in cint_sum
            assert cint_sum[stratum][1].shape[0] <= 16
            assert cint_sum[stratum][1].shape[1] <= 16

        cint_sum_depix = summarizer.summarise_cint(alpha=0.05, return_depixilated=True)
        for stratum in ["M->M", "M->F", "F->M", "F->F"]:
            assert cint_sum_depix[stratum].shape == (3, 81, 81)
