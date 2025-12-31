"""
Bootstrap Uncertainty Quantification for SocialMix

This module provides bootstrap resampling for contact matrix estimation.
Uses efficient NumPy operations for maximum performance.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from tqdm import tqdm

from ..dataloader import ContactData, ParticipantData, PopulationData
from ..utils import AgeBins


@dataclass
class BootstrapResults:
    """
    Container for bootstrap estimation results.

    Attributes
    ----------
    cint_samples : List[Dict[str, NDArray]]
        List of contact intensity matrices from each bootstrap iteration
    rate_samples : List[Dict[str, NDArray]]
        List of contact rate matrices from each bootstrap iteration
    n_boot : int
        Number of bootstrap iterations performed
    success_rate : float
        Fraction of successful bootstrap iterations
    """

    cint_samples: List[Dict[str, NDArray]]
    rate_samples: List[Dict[str, NDArray]]
    n_boot: int
    success_rate: float

    def __repr__(self) -> str:
        """Fast repr for debugger compatibility."""
        n_samples = len(self.cint_samples)
        strata_keys = list(self.cint_samples[0].keys()) if n_samples > 0 else []
        return (
            f"BootstrapResults(n_boot={self.n_boot}, "
            f"success_rate={self.success_rate:.2%}, "
            f"samples={n_samples}, strata={strata_keys})"
        )

    def mean(self, statistic: str = "cint") -> Dict[str, NDArray]:
        """
        Compute mean across bootstrap samples.

        Parameters
        ----------
        statistic : str, default "cint"
            Which statistic to compute mean for: "cint" or "rate"

        Returns
        -------
        mean_dict : Dict[str, NDArray]
            Mean of the specified statistic across bootstrap samples
        """
        if statistic == "cint":
            samples = self.cint_samples
        elif statistic == "rate":
            samples = self.rate_samples
        else:
            raise ValueError(f"statistic must be 'cint' or 'rate', got {statistic}")

        mean_dict = {}
        keys = list(samples[0].keys())

        for key in keys:
            stack = np.stack([s[key] for s in samples])
            mean_dict[key] = np.mean(stack, axis=0)

        return mean_dict

    def std(self, statistic: str = "cint") -> Dict[str, NDArray]:
        """
        Compute standard deviation across bootstrap samples.

        Parameters
        ----------
        statistic : str, default "cint"
            Which statistic to compute std for: "cint" or "rate"

        Returns
        -------
        std_dict : Dict[str, NDArray]
            Standard deviation of the specified statistic across bootstrap samples
        """
        if statistic == "cint":
            samples = self.cint_samples
        elif statistic == "rate":
            samples = self.rate_samples
        else:
            raise ValueError(f"statistic must be 'cint' or 'rate', got {statistic}")

        std_dict = {}
        keys = list(samples[0].keys())

        for key in keys:
            stack = np.stack([s[key] for s in samples])
            std_dict[key] = np.std(stack, axis=0, ddof=1)

        return std_dict

    def quantiles(self, q: List[float], statistic: str = "cint") -> Dict[str, NDArray]:
        """
        Compute quantiles across bootstrap samples.

        Parameters
        ----------
        q : List[float]
            Quantiles to compute (e.g., [0.025, 0.975] for 95% CI)
        statistic : str, default "cint"
            Which statistic to compute quantiles for: "cint" or "rate"

        Returns
        -------
        quant_dict : Dict[str, NDArray]
            Quantiles of the specified statistic, shape (len(q), C, D) for each key
        """
        if statistic == "cint":
            samples = self.cint_samples
        elif statistic == "rate":
            samples = self.rate_samples
        else:
            raise ValueError(f"statistic must be 'cint' or 'rate', got {statistic}")

        quant_dict = {}
        keys = list(samples[0].keys())

        for key in keys:
            stack = np.stack([s[key] for s in samples])
            quant_dict[key] = np.quantile(stack, q, axis=0)

        return quant_dict


class SocialMixBootstrap:
    """
    Performs efficient bootstrap uncertainty quantification for SocialMix.

    Uses pre-computed arrays and vectorized operations for maximum performance.
    """

    def __init__(
        self,
        part_data: ParticipantData,
        cnt_data: ContactData,
        age_bins: AgeBins,
        pop_data: Optional[PopulationData] = None,
        apply_reciprocity: bool = True,
        n_boot: int = 1000,
        random_state: Optional[int] = None,
    ):
        """
        Initialize bootstrap estimator.

        Parameters
        ----------
        part_data : ParticipantData
            Participant data container
        cnt_data : ContactData
            Contact data container
        age_bins : AgeBins
            Age stratification bins
        pop_data : PopulationData, optional
            Population data for reciprocity adjustment
        apply_reciprocity : bool, default True
            Whether to apply reciprocity adjustment within each bootstrap iteration
        n_boot : int, default 1000
            Number of bootstrap samples
        random_state : int, optional
            Random seed for reproducibility
        """
        self.part_data = part_data
        self.cnt_data = cnt_data
        self.age_bins = age_bins
        self.pop_data = pop_data
        self.apply_reciprocity = apply_reciprocity
        self.n_boot = n_boot
        self.rng = np.random.default_rng(random_state)

        # Stratification info (will be set during preparation)
        self.C: int = 0  # Number of participant age groups
        self.D: int = 0  # Number of contact age groups
        self.K_part: int = 1  # Number of participant strata
        self.K_cnt: int = 1  # Number of contact strata
        self.strat_mode: str = "single"
        self.strat_vars_part: List[str] = []
        self.strat_vars_cnt: List[str] = []
        self.P: Optional[NDArray] = None  # Population sizes

        # Assign age groups
        self._assign_age_groups()

    def _assign_age_groups(self) -> None:
        """Assign age groups to participants and contacts if not already provided."""
        # Construct bin edges
        bin_edges = self.age_bins.left + [self.age_bins.right[-1]]

        # Create interval labels
        intervals = [
            pd.Interval(left=l, right=r, closed="left")
            for l, r in zip(self.age_bins.left, self.age_bins.right)
        ]

        # Assign age groups to participants if not present
        if "age_grp_part" not in self.part_data.data.columns:
            if "age_part" in self.part_data.data.columns:
                ages = self.part_data.data["age_part"]
                age_grps = pd.cut(ages, bins=bin_edges, right=False, labels=intervals)
                self.part_data.data["age_grp_part"] = age_grps
            else:
                raise ValueError(
                    "ParticipantData must have either 'age_part' or 'age_grp_part' column."
                )

        # Assign age groups to contacts if not present
        if "age_grp_cnt" not in self.cnt_data.data.columns:
            if "age_cnt" in self.cnt_data.data.columns:
                ages = self.cnt_data.data["age_cnt"]
                age_grps = pd.cut(ages, bins=bin_edges, right=False, labels=intervals)
                self.cnt_data.data["age_grp_cnt"] = age_grps
            else:
                raise ValueError(
                    "ContactData must have either 'age_cnt' or 'age_grp_cnt' column."
                )

        # Assign age groups to population if provided
        if self.pop_data is not None and "age_grp" not in self.pop_data.data.columns:
            if "age" in self.pop_data.data.columns:
                ages = self.pop_data.data["age"]
                age_grps = pd.cut(ages, bins=bin_edges, right=False, labels=intervals)
                self.pop_data.data["age_grp"] = age_grps
            else:
                raise ValueError(
                    "PopulationData must have either 'age' or 'age_grp' column."
                )

    def run(
        self,
        progress: bool = True,
        min_success_rate: float = 0.5,
    ) -> BootstrapResults:
        """
        Run bootstrap resampling with efficient NumPy operations.

        Parameters
        ----------
        progress : bool, default True
            Show progress bar
        min_success_rate : float, default 0.5
            Minimum fraction of successful iterations required

        Returns
        -------
        BootstrapResults
            Container with bootstrap samples and summary statistics

        Raises
        ------
        ValueError
            If too many bootstrap iterations fail
        """
        # Prepare data structures
        Y_raw, age_codes, part_strat_codes, boot_indices = (
            self._prepare_bootstrap_data()
        )

        # Storage for results
        cint_samples = []
        rate_samples = []

        iterator = tqdm(range(self.n_boot), desc="Bootstrapping", disable=not progress)

        for b in iterator:
            try:
                boot_idx = boot_indices[b]
                cint_boot, rate_boot = self._bootstrap_iteration(
                    Y_raw, age_codes, part_strat_codes, boot_idx
                )
                cint_samples.append(cint_boot)
                rate_samples.append(rate_boot)
            except (ValueError, ZeroDivisionError) as e:
                if progress:
                    iterator.write(f"Bootstrap iteration {b} failed: {str(e)[:100]}")
                continue
            except Exception as e:
                if progress:
                    iterator.write(
                        f"Bootstrap iteration {b} unexpected error: {str(e)[:100]}"
                    )
                continue

        n_successful = len(cint_samples)
        success_rate = n_successful / self.n_boot

        if success_rate < min_success_rate:
            raise ValueError(
                f"Bootstrap failed: only {n_successful}/{self.n_boot} "
                f"iterations succeeded ({success_rate:.1%}). "
                f"This may indicate insufficient sample sizes in some age groups."
            )

        if progress and n_successful < self.n_boot:
            print(
                f"\nCompleted {n_successful}/{self.n_boot} bootstrap iterations "
                f"({success_rate:.1%} success rate)"
            )

        return BootstrapResults(
            cint_samples=cint_samples,
            rate_samples=rate_samples,
            n_boot=n_successful,
            success_rate=success_rate,
        )

    def _prepare_bootstrap_data(
        self,
    ) -> Tuple[NDArray, NDArray, Optional[NDArray], NDArray]:
        """
        Pre-compute arrays for efficient bootstrap resampling.

        Returns
        -------
        Y_raw : NDArray, shape (N, K_cnt, D)
            Contact count tensor per participant
        age_codes : NDArray, shape (N,)
            Participant age group codes
        part_strat_codes : NDArray, shape (N,) or None
            Participant stratum codes (None if no participant stratification)
        boot_indices : NDArray, shape (n_boot, N)
            Pre-generated bootstrap sample indices
        """
        # Extract stratification info
        self.strat_vars_part = self.part_data.get_strat_vars(suffix=False)
        self.strat_vars_cnt = self.cnt_data.get_strat_vars(suffix=False)

        # Determine stratification mode
        if len(self.strat_vars_part) == 0 and len(self.strat_vars_cnt) == 0:
            self.strat_mode = "single"
        elif len(self.strat_vars_cnt) == 0:
            self.strat_mode = "partial"
        elif set(self.strat_vars_part) == set(self.strat_vars_cnt):
            self.strat_mode = "full"
        else:
            self.strat_mode = "mixed"

        # Get dimensions
        self.C = len(self.part_data.data["age_grp_part"].cat.categories)
        self.D = len(self.cnt_data.data["age_grp_cnt"].cat.categories)

        # Calculate K_part and K_cnt
        if self.strat_vars_part:
            self.K_part = 1
            for var in self.strat_vars_part:
                col_name = f"{var}_part"
                self.K_part *= self.part_data.data[col_name].nunique()
        else:
            self.K_part = 1

        if self.strat_vars_cnt:
            self.K_cnt = 1
            for var in self.strat_vars_cnt:
                col_name = f"{var}_cnt"
                self.K_cnt *= self.cnt_data.data[col_name].nunique()
        else:
            self.K_cnt = 1

        # Build contact count tensor
        Y_raw = self._build_contact_tensor()  # (N, K_cnt, D)

        # Extract participant age codes
        age_codes = self.part_data.data["age_grp_part"].cat.codes.values.astype(
            np.int32
        )

        # Extract participant stratum codes (if applicable)
        if self.K_part > 1:
            part_strat_codes = self._get_participant_stratum_codes()
        else:
            part_strat_codes = None

        # Pre-generate all bootstrap indices
        N = len(self.part_data.data)
        boot_indices = self.rng.integers(0, N, size=(self.n_boot, N), dtype=np.int32)

        # Load population data if available
        if self.pop_data is not None:
            self._load_population_data()

        return Y_raw, age_codes, part_strat_codes, boot_indices

    def _build_contact_tensor(self) -> NDArray:
        """
        Build contact count tensor of shape (N, K_cnt, D).

        For each participant i, contact stratum k, and contact age group d,
        Y_raw[i, k, d] = total contacts participant i had with age group d in stratum k.
        """
        N = len(self.part_data.data)
        Y_raw = np.zeros((N, self.K_cnt, self.D), dtype=np.float64)

        # Get all participant IDs
        all_ids = self.part_data.data["id"].values
        id_to_idx = {id_val: idx for idx, id_val in enumerate(all_ids)}

        if self.K_cnt == 1:
            # No contact stratification: aggregate by (id, age_grp_cnt)
            df_agg = (
                self.cnt_data.data.groupby(["id", "age_grp_cnt"], observed=True)["y"]
                .sum()
                .reset_index()
            )

            for _, row in df_agg.iterrows():
                i = id_to_idx.get(row["id"])
                if i is not None:
                    d = row["age_grp_cnt"]
                    d_code = self.cnt_data.data["age_grp_cnt"].cat.categories.get_loc(d)
                    Y_raw[i, 0, d_code] = row["y"]
        else:
            # Contact stratification: aggregate by (id, strat_vars_cnt, age_grp_cnt)
            group_cols = (
                ["id"] + [f"{v}_cnt" for v in self.strat_vars_cnt] + ["age_grp_cnt"]
            )
            df_agg = (
                self.cnt_data.data.groupby(group_cols, observed=True)["y"]
                .sum()
                .reset_index()
            )

            # Get contact stratum mapping
            cnt_strat_mapping = self._get_contact_stratum_mapping()

            for _, row in df_agg.iterrows():
                i = id_to_idx.get(row["id"])
                if i is not None:
                    # Get contact stratum code
                    strat_tuple = tuple(row[f"{v}_cnt"] for v in self.strat_vars_cnt)
                    k = cnt_strat_mapping[strat_tuple]

                    # Get age code
                    d = row["age_grp_cnt"]
                    d_code = self.cnt_data.data["age_grp_cnt"].cat.categories.get_loc(d)

                    Y_raw[i, k, d_code] = row["y"]

        return Y_raw

    def _get_participant_stratum_codes(self) -> NDArray:
        """Get stratum codes for each participant."""
        # Build mapping from stratum tuple to code
        strat_mapping = self._get_participant_stratum_mapping()

        # Extract stratum for each participant
        N = len(self.part_data.data)
        codes = np.zeros(N, dtype=np.int32)

        for i in range(N):
            strat_tuple = tuple(
                self.part_data.data.iloc[i][f"{v}_part"] for v in self.strat_vars_part
            )
            codes[i] = strat_mapping[strat_tuple]

        return codes

    def _get_participant_stratum_mapping(self) -> Dict:
        """Create mapping from participant stratum tuple to integer code."""
        import itertools

        # Get all category combinations
        categories = []
        for var in self.strat_vars_part:
            col_name = f"{var}_part"
            cats = self.part_data.data[col_name].cat.categories.tolist()
            categories.append(cats)

        # Create mapping
        mapping = {}
        for idx, combo in enumerate(itertools.product(*categories)):
            mapping[combo] = idx

        return mapping

    def _get_contact_stratum_mapping(self) -> Dict:
        """Create mapping from contact stratum tuple to integer code."""
        import itertools

        # Get all category combinations
        categories = []
        for var in self.strat_vars_cnt:
            col_name = f"{var}_cnt"
            cats = self.cnt_data.data[col_name].cat.categories.tolist()
            categories.append(cats)

        # Create mapping
        mapping = {}
        for idx, combo in enumerate(itertools.product(*categories)):
            mapping[combo] = idx

        return mapping

    def _load_population_data(self):
        """Load and process population data for reciprocity adjustment."""
        # Assign age groups to population data
        bin_edges = self.age_bins.left + [self.age_bins.right[-1]]
        intervals = [
            pd.Interval(left=l, right=r, closed="left")
            for l, r in zip(self.age_bins.left, self.age_bins.right)
        ]

        if "age_grp" not in self.pop_data.data.columns:
            ages = self.pop_data.data["age"]
            age_grps = pd.cut(ages, bins=bin_edges, right=False, labels=intervals)
            self.pop_data.data["age_grp"] = age_grps

        # Aggregate population by age group and stratum
        if self.K_cnt == 1:
            # No stratification
            pop_sizes = (
                self.pop_data.data.groupby("age_grp", observed=False)["P"]
                .sum()
                .reindex(pd.Index(intervals), fill_value=0)
            )
            self.P = pop_sizes.values.astype(np.float64)
        else:
            # With stratification
            strat_vars_pop = self.pop_data.get_strat_vars(suffix=False)
            group_cols = strat_vars_pop + ["age_grp"]
            pop_sizes = self.pop_data.data.groupby(group_cols, observed=False)[
                "P"
            ].sum()

            # Reshape to (K_cnt, D)
            self.P = np.zeros((self.K_cnt, self.D), dtype=np.float64)
            cnt_strat_mapping = self._get_contact_stratum_mapping()

            for idx, value in pop_sizes.items():
                if self.strat_mode == "single":
                    age_grp = idx
                    d_code = intervals.index(age_grp)
                    self.P[d_code] = value
                else:
                    # idx is a tuple: (*strat_values, age_grp)
                    strat_tuple = idx[:-1]
                    age_grp = idx[-1]
                    k = cnt_strat_mapping[strat_tuple]
                    d_code = intervals.index(age_grp)
                    self.P[k, d_code] = value

    def _bootstrap_iteration(
        self,
        Y_raw: NDArray,
        age_codes: NDArray,
        part_strat_codes: Optional[NDArray],
        boot_idx: NDArray,
    ) -> Tuple[Dict[str, NDArray], Dict[str, NDArray]]:
        """
        Execute single bootstrap iteration.

        Parameters
        ----------
        Y_raw : NDArray, shape (N, K_cnt, D)
            Contact count tensor
        age_codes : NDArray, shape (N,)
            Participant age codes
        part_strat_codes : NDArray, shape (N,) or None
            Participant stratum codes
        boot_idx : NDArray, shape (N,)
            Bootstrap sample indices

        Returns
        -------
        cint_boot : Dict[str, NDArray]
            Contact intensity matrices for this bootstrap sample
        rate_boot : Dict[str, NDArray]
            Contact rate matrices for this bootstrap sample
        """
        # Aggregate bootstrap sample into Y_boot and N_boot
        Y_boot, N_boot = self._aggregate_bootstrap_sample(
            Y_raw, age_codes, part_strat_codes, boot_idx
        )

        # Compute contact intensities
        cint_boot = self._compute_cint_from_YN(Y_boot, N_boot)

        # Apply reciprocity if requested
        if self.apply_reciprocity and self.pop_data is not None:
            cint_boot = self._apply_reciprocity(cint_boot)

        # Compute rates
        rate_boot = self._compute_rate_from_cint(cint_boot)

        return cint_boot, rate_boot

    def _aggregate_bootstrap_sample(
        self,
        Y_raw: NDArray,
        age_codes: NDArray,
        part_strat_codes: Optional[NDArray],
        boot_idx: NDArray,
    ) -> Tuple[Union[NDArray, Dict[str, NDArray]], Union[NDArray, Dict[str, NDArray]]]:
        """
        Aggregate resampled contacts into Y_boot and N_boot.

        Parameters
        ----------
        Y_raw : NDArray, shape (N, K_cnt, D)
            Contact count tensor
        age_codes : NDArray, shape (N,)
            Participant age codes
        part_strat_codes : NDArray, shape (N,) or None
            Participant stratum codes
        boot_idx : NDArray, shape (N,)
            Bootstrap sample indices

        Returns
        -------
        Y_boot : NDArray or Dict[str, NDArray]
            Aggregated contact counts
            - Single: (C, D)
            - Partial: Dict with (C, D) arrays
            - Full: Dict with (C, D) arrays
            - Mixed: Dict with (C, D) arrays
        N_boot : NDArray or Dict[str, NDArray]
            Participant counts
            - Single: (C,)
            - Partial: Dict with (C,) arrays
            - Full: Dict with (C,) arrays
            - Mixed: Dict with (C,) arrays
        """
        # Resample
        Y_sampled = Y_raw[boot_idx]  # (N, K_cnt, D)
        age_sampled = age_codes[boot_idx]  # (N,)

        if self.strat_mode == "single":
            # Aggregate: Y_boot[c, d] = sum over i where age_sampled[i] == c
            Y_boot = np.zeros((self.C, self.D), dtype=np.float64)
            N_boot = np.zeros(self.C, dtype=np.int32)

            for i in range(len(boot_idx)):
                c = age_sampled[i]
                Y_boot[c, :] += Y_sampled[i, 0, :]  # K_cnt=1
                N_boot[c] += 1

            return Y_boot, N_boot

        # Stratified cases
        part_strat_sampled = (
            part_strat_codes[boot_idx] if part_strat_codes is not None else None
        )

        # Build stratum labels
        strat_labels = self._create_stratum_labels()

        Y_boot_dict = {}
        N_boot_dict = {}

        if self.strat_mode == "partial":
            # Only participant stratification: Y[p][c, d]
            for p in range(self.K_part):
                label = f"{strat_labels[p]}->All"
                Y_boot_dict[label] = np.zeros((self.C, self.D), dtype=np.float64)
                N_boot_dict[label] = np.zeros(self.C, dtype=np.int32)

            for i in range(len(boot_idx)):
                c = age_sampled[i]
                p = part_strat_sampled[i]
                label = f"{strat_labels[p]}->All"
                Y_boot_dict[label][c, :] += Y_sampled[i, 0, :]  # K_cnt=1
                N_boot_dict[label][c] += 1

        elif self.strat_mode == "full":
            # Both sides stratified, same categories: Y[p,q][c, d]
            part_labels = self._create_participant_stratum_labels()
            cnt_labels = self._create_contact_stratum_labels()

            for p in range(self.K_part):
                for q in range(self.K_cnt):
                    label = f"{part_labels[p]}->{cnt_labels[q]}"
                    Y_boot_dict[label] = np.zeros((self.C, self.D), dtype=np.float64)
                    N_boot_dict[label] = np.zeros(self.C, dtype=np.int32)

            for i in range(len(boot_idx)):
                c = age_sampled[i]
                p = part_strat_sampled[i]
                for q in range(self.K_cnt):
                    label = f"{part_labels[p]}->{cnt_labels[q]}"
                    Y_boot_dict[label][c, :] += Y_sampled[i, q, :]
                    N_boot_dict[label][c] += 1

        else:  # mixed
            # General case: build labels from actual data
            # For simplicity, we'll use the same logic as full mode
            # but handle K_part != K_cnt cases
            part_labels = self._create_participant_stratum_labels()
            cnt_labels = self._create_contact_stratum_labels()

            for p in range(max(1, self.K_part)):
                for q in range(self.K_cnt):
                    p_label = part_labels[p] if self.K_part > 1 else "All"
                    q_label = cnt_labels[q] if self.K_cnt > 1 else "All"
                    label = f"{p_label}->{q_label}"
                    Y_boot_dict[label] = np.zeros((self.C, self.D), dtype=np.float64)
                    N_boot_dict[label] = np.zeros(self.C, dtype=np.int32)

            for i in range(len(boot_idx)):
                c = age_sampled[i]
                p = part_strat_sampled[i] if part_strat_sampled is not None else 0

                for q in range(self.K_cnt):
                    p_label = part_labels[p] if self.K_part > 1 else "All"
                    q_label = cnt_labels[q] if self.K_cnt > 1 else "All"
                    label = f"{p_label}->{q_label}"
                    Y_boot_dict[label][c, :] += Y_sampled[i, q, :]
                    N_boot_dict[label][c] += 1

        return Y_boot_dict, N_boot_dict

    def _create_stratum_labels(self) -> List[str]:
        """Create human-readable stratum labels (for backward compatibility)."""
        if self.K_part > 1:
            return self._create_participant_stratum_labels()
        elif self.K_cnt > 1:
            return self._create_contact_stratum_labels()
        else:
            return ["All"]

    def _create_participant_stratum_labels(self) -> List[str]:
        """Create human-readable labels for participant strata."""
        import itertools

        if self.K_part == 1:
            return ["All"]

        # Get category combinations for participant strata
        categories = []
        for var in self.strat_vars_part:
            col_name = f"{var}_part"
            cats = self.part_data.data[col_name].cat.categories.tolist()
            categories.append(cats)

        # Build labels
        labels = []
        for combo in itertools.product(*categories):
            if len(combo) == 1:
                labels.append(str(combo[0]))
            else:
                labels.append("_".join(str(c) for c in combo))

        return labels

    def _create_contact_stratum_labels(self) -> List[str]:
        """Create human-readable labels for contact strata."""
        import itertools

        if self.K_cnt == 1:
            return ["All"]

        # Get category combinations for contact strata
        categories = []
        for var in self.strat_vars_cnt:
            col_name = f"{var}_cnt"
            cats = self.cnt_data.data[col_name].cat.categories.tolist()
            categories.append(cats)

        # Build labels
        labels = []
        for combo in itertools.product(*categories):
            if len(combo) == 1:
                labels.append(str(combo[0]))
            else:
                labels.append("_".join(str(c) for c in combo))

        return labels

    def _compute_cint_from_YN(
        self,
        Y_boot: Union[NDArray, Dict[str, NDArray]],
        N_boot: Union[NDArray, Dict[str, NDArray]],
    ) -> Dict[str, NDArray]:
        """
        Compute contact intensities from Y_boot and N_boot.

        Parameters
        ----------
        Y_boot : NDArray or Dict[str, NDArray]
            Aggregated contact counts
        N_boot : NDArray or Dict[str, NDArray]
            Participant counts

        Returns
        -------
        cint_boot : Dict[str, NDArray]
            Contact intensity matrices
        """
        if self.strat_mode == "single":
            # Y_boot: (C, D), N_boot: (C,)
            cint = Y_boot / N_boot[:, np.newaxis]
            cint = np.nan_to_num(cint, nan=0.0, posinf=0.0, neginf=0.0)
            return {"All->All": cint}

        # Stratified cases
        cint_boot = {}
        for label in Y_boot.keys():
            cint = Y_boot[label] / N_boot[label][:, np.newaxis]
            cint = np.nan_to_num(cint, nan=0.0, posinf=0.0, neginf=0.0)
            cint_boot[label] = cint

        return cint_boot

    def _apply_reciprocity(self, cint: Dict[str, NDArray]) -> Dict[str, NDArray]:
        """
        Apply reciprocity adjustment to contact intensities.

        Parameters
        ----------
        cint : Dict[str, NDArray]
            Contact intensity matrices

        Returns
        -------
        cint_adj : Dict[str, NDArray]
            Reciprocity-adjusted contact intensity matrices
        """
        if self.strat_mode == "single":
            # Within-stratum reciprocity: m†[c,d] = (m[c,d]·P[c] + m[d,c]·P[d]) / (2·P[c])
            m = cint["All->All"]
            P = self.P

            numerator = m * P[:, np.newaxis] + m.T * P[np.newaxis, :]
            denominator = 2 * P[:, np.newaxis]
            m_adj = numerator / denominator
            m_adj = np.nan_to_num(m_adj, nan=0.0, posinf=0.0, neginf=0.0)

            return {"All->All": m_adj}

        # Stratified cases
        cint_adj = {}

        # Get stratum label mappings
        part_labels = self._create_participant_stratum_labels()
        cnt_labels = self._create_contact_stratum_labels()

        for label, m in cint.items():
            source, target = label.split("->")
            reverse_label = f"{target}->{source}"

            if source == target and source != "All":
                # Within-stratum (but not "All->All" which is handled above)
                if self.P.ndim == 1:
                    P = self.P
                else:
                    # Get stratum index using contact labels
                    k = cnt_labels.index(source)
                    P = self.P[k, :]

                numerator = m * P[:, np.newaxis] + m.T * P[np.newaxis, :]
                denominator = 2 * P[:, np.newaxis]
                m_adj = numerator / denominator
                m_adj = np.nan_to_num(m_adj, nan=0.0, posinf=0.0, neginf=0.0)
                cint_adj[label] = m_adj

            elif target == "All":
                # Partial stratification (source->All)
                # Within-stratum reciprocity
                if self.P.ndim == 1:
                    P = self.P
                else:
                    # Get stratum index using participant labels
                    k = part_labels.index(source)
                    P = self.P[k, :]

                numerator = m * P[:, np.newaxis] + m.T * P[np.newaxis, :]
                denominator = 2 * P[:, np.newaxis]
                m_adj = numerator / denominator
                m_adj = np.nan_to_num(m_adj, nan=0.0, posinf=0.0, neginf=0.0)
                cint_adj[label] = m_adj

            else:
                # Between-stratum
                if reverse_label in cint:
                    m_reverse = cint[reverse_label]

                    # Get population for each stratum using contact labels
                    k_source = cnt_labels.index(source)
                    k_target = cnt_labels.index(target)

                    if self.P.ndim == 1:
                        P_source = self.P
                        P_target = self.P
                    else:
                        P_source = self.P[k_source, :]
                        P_target = self.P[k_target, :]

                    # m†[c,d] = (m[c,d] + m_reverse[d,c]·P[d]/P[c]) / 2
                    ratio = P_target[np.newaxis, :] / P_source[:, np.newaxis]
                    m_adj = (m + m_reverse.T * ratio) / 2
                    m_adj = np.nan_to_num(m_adj, nan=0.0, posinf=0.0, neginf=0.0)
                    cint_adj[label] = m_adj
                else:
                    # No reverse matrix, keep original
                    cint_adj[label] = m

        return cint_adj

    def _compute_rate_from_cint(self, cint: Dict[str, NDArray]) -> Dict[str, NDArray]:
        """
        Compute contact rates from contact intensities.

        Parameters
        ----------
        cint : Dict[str, NDArray]
            Contact intensity matrices

        Returns
        -------
        rate : Dict[str, NDArray]
            Contact rate matrices (per-capita contact rates)
        """
        if self.pop_data is None:
            # Cannot compute rates without population data
            return {label: np.zeros_like(m) for label, m in cint.items()}

        rate = {}

        if self.strat_mode == "single":
            # rate = cint / P[newaxis, :]
            P = self.P
            rate["All->All"] = cint["All->All"] / P[np.newaxis, :]
            rate["All->All"] = np.nan_to_num(
                rate["All->All"], nan=0.0, posinf=0.0, neginf=0.0
            )
        else:
            # Stratified
            # Get stratum label mappings
            part_labels = self._create_participant_stratum_labels()
            cnt_labels = self._create_contact_stratum_labels()

            for label, m in cint.items():
                if self.P.ndim == 1:
                    P = self.P
                else:
                    # Extract target stratum
                    parts = label.split("->")
                    target = parts[1]

                    # For "source->All" format, use source (participant) stratum for population
                    if target == "All":
                        source = parts[0]
                        k = part_labels.index(source)
                    else:
                        # For "source->target" format, use target (contact) stratum
                        k = cnt_labels.index(target)

                    P = self.P[k, :]

                rate[label] = m / P[np.newaxis, :]
                rate[label] = np.nan_to_num(
                    rate[label], nan=0.0, posinf=0.0, neginf=0.0
                )

        return rate
