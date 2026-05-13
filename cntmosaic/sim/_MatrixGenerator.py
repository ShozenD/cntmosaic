from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from .._types import StratumLabel  # noqa: F401
from ._PopulationConstructor import PopulationConstructor
from ._Stratification import Stratification


@dataclass
class _StratInfo:
    """Lightweight stratification info extracted from DataFrame."""

    name: str
    n_strata: int
    labels: List[str]


class MatrixGenerator:
    """
    Generate synthetic contact intensity matrices using template-based approach.

    This class implements the mathematical framework for
    generating plausible social contact patterns. It combines normalized template
    matrices (household, school, work, community) with stratification-specific
    deviations to create age-structured contact patterns.

    Three generation modes:
    - single: Global baseline contact matrix (unstratified)
    - partial: One matrix per stratum to general population
    - full: All pairwise stratum-to-stratum contact matrices

    The generation process:
    1. Mix normalized templates using Dirichlet weights
    2. Scale by mean contact intensity
    3. Apply stratification-specific deviations (partial/full modes)
    4. Enforce reciprocity constraints
    5. Normalize deviations using population proportions

    Parameters
    ----------
    templates : dict of NDArray
        Template contact patterns with keys: 'household', 'school', 'work', 'community'.
        Each matrix should be A\timesA where A is the number of age groups.
        Templates are assumed to be pre-smoothed.
        Use `cntmosaic.datasets.load_template_patterns()` to load defaults.

    Attributes
    ----------
    templates : dict of NDArray
        Normalized template matrices (avg marginal intensity = 1)
    n_ages : int
        Number of age groups (from template shape)

    Examples
    --------
    >>> from cntmosaic.datasets import load_template_patterns
    >>> from cntmosaic.sim import Stratification, PopulationConstructor, MatrixGenerator
    >>> import numpy as np

    >>> # Load templates
    >>> templates = load_template_patterns('United_States', max_age=80)
    >>> mg = MatrixGenerator(templates)

    >>> # Create stratified population
    >>> ref_age_dist = np.array([1000, 1500, 2000, 1800, 1200])
    >>> gender_strat = Stratification('gender', 2, ref_age_dist,
    ...                               labels=['Male', 'Female'], seed=42)
    >>> pop = PopulationConstructor(gender_strat)

    >>> # Generate baseline matrix
    >>> M_baseline = mg.generate_single(pop, mean_intensity=15.0, seed=123)

    >>> # Generate partial matrices (each stratum to general population)
    >>> M_partial = mg.generate_partial(pop, mean_intensity=15.0, seed=123)
    >>> # M_partial = {0: M^0, 1: M^1}

    >>> # Generate full matrices (all stratum pairs)
    >>> M_full = mg.generate_full(pop, mean_intensity=15.0, seed=123)
    >>> # M_full = {(0,0): M^{0,0}, (0,1): M^{0,1}, (1,0): M^{1,0}, (1,1): M^{1,1}}

    Notes
    -----
    Mathematical Framework:
    - Template mixture: T = υ^h T^h + υ^s T^s + υ^w T^w + υ^c T^c
    - Baseline intensity: M = C \times T
    - Reciprocity: M̃ = ½(M + P M^T P^{-1})
    - Deviation matrix: D^{k,l} = exp(η E^{k,l})
    - Combined deviation: d^{s,t} = ∏_j D^{k_j(s), l_j(t)}
    - Normalized deviation: \delta^{s,t} = d^{s,t} / \sum_{u,v} d^{u,v} S^{u,v}
    - Stratified intensity: m^{s,t} = \gamma \delta^{s,t} P^t

    References
    ----------
    Mistry, D., Litvinova, M., Pastore y Piontti, A., et al. (2021).
    Inferring high-resolution human mixing patterns for disease modeling.
    Nature Communications, 12(1), 323.
    """

    REQUIRED_TEMPLATES = {"household", "school", "work", "community"}

    def __init__(self, templates: Dict[str, NDArray]):
        """
        Initialize generator with contact pattern templates.

        Parameters
        ----------
        templates : dict of NDArray
            Must contain keys: 'household', 'school', 'work', 'community'.
            Each matrix should be A\timesA where A is the number of age groups.
            Use cntmosaic.datasets.load_template_patterns() to load default templates.

        Raises
        ------
        TypeError
            If templates is not a dictionary.
        ValueError
            If required templates are missing or have inconsistent shapes.
        """
        self._validate_templates(templates)
        self.templates = self._normalize_templates(templates)
        self.n_ages = list(self.templates.values())[0].shape[0]

    def _validate_templates(self, templates: Dict[str, NDArray]) -> None:
        """
        Validate template structure.

        Checks:
        1. templates is a dictionary
        2. All required templates present
        3. All templates have same shape
        4. Templates are square matrices
        """
        if not isinstance(templates, dict):
            raise TypeError("templates must be a dictionary")

        missing = self.REQUIRED_TEMPLATES - set(templates.keys())
        if missing:
            raise ValueError(f"Missing required templates: {missing}")

        # Check all templates have same shape
        shapes = [t.shape for t in templates.values()]
        if len(set(shapes)) > 1:
            raise ValueError(f"All templates must have same shape. Got: {shapes}")

        # Check square matrices
        first_shape = shapes[0]
        if first_shape[0] != first_shape[1]:
            raise ValueError(
                f"Templates must be square matrices. Got shape: {first_shape}"
            )

    def _normalize_templates(self, templates: Dict[str, NDArray]) -> Dict[str, NDArray]:
        """
        Normalize templates so average marginal intensity equals 1.

        For each template T, computes: T̄ = T / (A^{-1} \sum_a \sum_b t_{a,b})
        """
        normalized = {}
        for name, T in templates.items():
            A = T.shape[0]
            mean_intensity = T.sum() / A
            normalized[name] = T / mean_intensity if mean_intensity > 0 else T
        return normalized

    def _preprocess_df(
        self,
        df: pd.DataFrame,
        strat_var_cols: Optional[List[str]] = None,
        age_col: str = "age",
        pop_col: str = "P",
    ) -> Tuple[NDArray, NDArray, NDArray, List[_StratInfo]]:
        """
        Preprocess a population DataFrame into arrays needed for matrix generation.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame containing population sizes with columns for age,
            population size, and optionally stratification variables.
        strat_var_cols : list of str, optional
            Names of columns to use as stratification variables.
            If None, all columns except age_col and pop_col are used.
        age_col : str, default "age"
            Name of the column containing age values.
        pop_col : str, default "P"
            Name of the column containing population sizes.

        Returns
        -------
        ref_age_dist : NDArray
            Reference age distribution (population counts by age). Shape: (n_ages,)
        Q : NDArray
            Population proportion matrix Q[s, a] = P(stratum=s | age=a).
            Shape: (n_strata, n_ages)
        P : NDArray
            Population counts matrix P[s, a]. Shape: (n_strata, n_ages)
        strat_infos : list of _StratInfo
            Stratification metadata for each stratification variable.

        Raises
        ------
        ValueError
            If required columns are missing.
        """
        # Validate required columns
        if age_col not in df.columns:
            raise ValueError(f"Age column '{age_col}' not found in DataFrame")
        if pop_col not in df.columns:
            raise ValueError(f"Population column '{pop_col}' not found in DataFrame")

        # Determine stratification variables
        if strat_var_cols is None:
            strat_var_cols = [c for c in df.columns if c not in [age_col, pop_col]]

        for var in strat_var_cols:
            if var not in df.columns:
                raise ValueError(
                    f"Stratification variable '{var}' not found in DataFrame"
                )

        # Get unique ages (sorted)
        ages = np.sort(df[age_col].unique())
        n_ages = len(ages)
        age_to_idx = {age: idx for idx, age in enumerate(ages)}

        # Handle case with no stratification variables
        if len(strat_var_cols) == 0:
            # Unstratified: single stratum
            ref_age_dist = np.zeros(n_ages)
            for _, row in df.iterrows():
                age_idx = age_to_idx[row[age_col]]
                ref_age_dist[age_idx] = row[pop_col]

            Q = np.ones((1, n_ages))
            P = ref_age_dist[np.newaxis, :]
            strat_infos = []
            return ref_age_dist, Q, P, strat_infos

        # Get unique categories for each stratification variable
        strat_categories = {var: sorted(df[var].unique()) for var in strat_var_cols}

        # Create _StratInfo for each variable
        strat_infos = [
            _StratInfo(
                name=var,
                n_strata=len(strat_categories[var]),
                labels=strat_categories[var],
            )
            for var in strat_var_cols
        ]

        # Build stratum tuples (all combinations)
        strat_tuples = list(product(*[strat_categories[var] for var in strat_var_cols]))
        n_strata = len(strat_tuples)

        # Create mapping from stratum tuple to index
        strat_to_idx = {tup: idx for idx, tup in enumerate(strat_tuples)}

        # Build population matrix P[s, a]
        P_matrix = np.zeros((n_strata, n_ages))

        for _, row in df.iterrows():
            age_idx = age_to_idx[row[age_col]]
            strat_tuple = tuple(row[var] for var in strat_var_cols)
            if strat_tuple in strat_to_idx:
                strat_idx = strat_to_idx[strat_tuple]
                P_matrix[strat_idx, age_idx] = row[pop_col]

        # Compute reference age distribution (sum across strata)
        ref_age_dist = P_matrix.sum(axis=0)

        # Compute Q matrix: Q[s, a] = P[s, a] / sum_s(P[s, a])
        with np.errstate(divide="ignore", invalid="ignore"):
            Q = P_matrix / ref_age_dist[np.newaxis, :]
            # Set Q to uniform for ages with zero population
            zero_pop_ages = ref_age_dist == 0
            if zero_pop_ages.any():
                Q[:, zero_pop_ages] = 1.0 / n_strata

        return ref_age_dist, Q, P_matrix, strat_infos

    def _get_stratum_label_from_infos(
        self, stratum_idx: int, strat_infos: List[_StratInfo]
    ) -> str:
        """
        Get stratum label from _StratInfo objects.

        Parameters
        ----------
        stratum_idx : int
            Global stratum index.
        strat_infos : list of _StratInfo
            Stratification metadata.

        Returns
        -------
        str
            Label for the stratum (e.g., "Male" or "Male_Urban").
        """
        if len(strat_infos) == 0:
            return "All"

        if len(strat_infos) == 1:
            return strat_infos[0].labels[stratum_idx]

        # Multiple stratifications: decode index
        indices = self._decode_stratum_index_from_infos(stratum_idx, strat_infos)
        labels = [strat_infos[j].labels[indices[j]] for j in range(len(strat_infos))]
        return "_".join(labels)

    def _decode_stratum_index_from_infos(
        self, stratum_idx: int, strat_infos: List[_StratInfo]
    ) -> List[int]:
        """
        Decode global stratum index into per-stratification category indices.

        Parameters
        ----------
        stratum_idx : int
            Global stratum index.
        strat_infos : list of _StratInfo
            Stratification metadata.

        Returns
        -------
        list of int
            Category indices [k_1, k_2, ..., k_J].
        """
        indices = []
        remaining = stratum_idx

        # Compute strides (in reverse order)
        strides = []
        stride = 1
        for info in reversed(strat_infos):
            strides.append(stride)
            stride *= info.n_strata
        strides = list(reversed(strides))

        for j, info in enumerate(strat_infos):
            idx = remaining // strides[j]
            indices.append(idx)
            remaining = remaining % strides[j]

        return indices

    def _sample_mixture_weights(self, rng: np.random.Generator) -> NDArray:
        """
        Sample mixing weights from Dirichlet(1,1,1,1).

        Returns
        -------
        NDArray
            Array [υ^h, υ^s, υ^w, υ^c] of shape (4,)
        """
        return rng.dirichlet(np.ones(4))

    def _create_mixed_pattern(
        self, weights: NDArray, order: List[str] = None
    ) -> NDArray:
        """
        Create mixed contact pattern from templates.

        T = υ^h T^h + υ^s T^s + υ^w T^w + υ^c T^c

        Parameters
        ----------
        weights : NDArray
            Mixing weights [υ^h, υ^s, υ^w, υ^c]
        order : list of str, optional
            Order of templates. Default: ['household', 'school', 'work', 'community']

        Returns
        -------
        NDArray
            Mixed template pattern (A, A)
        """
        if order is None:
            order = ["household", "school", "work", "community"]

        pattern = sum(weights[i] * self.templates[name] for i, name in enumerate(order))
        return pattern

    def _enforce_reciprocity(self, M: NDArray, P_diag: NDArray) -> NDArray:
        """
        Enforce reciprocity condition: PM = (PM)^T

        Applies: M̃ = ½(M + P^{-1} M^T P)

        Parameters
        ----------
        M : NDArray
            Contact intensity matrix (A, A)
        P_diag : NDArray
            Diagonal population matrix (A, A) or population vector (A,)

        Returns
        -------
        NDArray
            Reciprocal matrix M̃ (A, A)
        """
        # Handle both diagonal matrix and vector input
        if P_diag.ndim == 1:
            P_diag = np.diag(P_diag)

        P_inv = np.linalg.inv(P_diag)
        return 0.5 * (M + P_inv @ M.T @ P_diag)

    def generate_single(
        self,
        popcon: PopulationConstructor,
        mean_intensity: float = 15.0,
        seed: Optional[int] = None,
    ) -> Dict[str, NDArray]:
        """
        Generate global baseline contact intensity matrix (unstratified).

        Creates a single contact matrix representing average contact patterns
        across the entire population, ignoring stratifications.

        Process:
        1. Sample mixture weights from Dirichlet(1,1,1,1)
        2. Create mixed template pattern T
        3. Scale by mean intensity: M = C \times T
        4. Enforce reciprocity: M̃ = ½(M + P^{-1} M^T P)

        Parameters
        ----------
        popcon : PopulationConstructor
            Population structure (stratifications are ignored for baseline)
        mean_intensity : float, default=15.0
            Average marginal contact intensity C
        seed : int, optional
            Random seed for reproducibility

        Returns
        -------
        dict
            Dictionary with single key "All->All" mapping to contact intensity matrix M (A, A)
            Element M[a,b] = expected contact intensity from age a to age b

        Examples
        --------
        >>> from cntmosaic.datasets import load_template_patterns
        >>> from cntmosaic.sim import Stratification, PopulationConstructor, MatrixGenerator
        >>> import numpy as np

        >>> templates = load_template_patterns('United_States', max_age=10)
        >>> mg = MatrixGenerator(templates)

        >>> ref_age_dist = np.array([1000, 1500, 2000])
        >>> gender_strat = Stratification('gender', 2, ref_age_dist, seed=42)
        >>> pop = PopulationConstructor(gender_strat)

        >>> M_dict = mg.generate_single(pop, mean_intensity=15.0, seed=123)
        >>> M_dict.keys()
        dict_keys(['All->All'])
        >>> M_dict['All->All'].shape
        (3, 3)
        """
        rng = np.random.default_rng(seed)

        # Sample mixture weights and create mixed pattern
        weights = self._sample_mixture_weights(rng)
        pattern = self._create_mixed_pattern(weights)

        # Scale by mean intensity
        M = pattern * mean_intensity

        # Enforce reciprocity using global population
        P_global = np.diag(popcon.ref_age_dist)
        M = self._enforce_reciprocity(M, P_global)

        return {"All->All": M}

    def generate_single_from_df(
        self,
        df: pd.DataFrame,
        mean_intensity: float = 15.0,
        seed: Optional[int] = None,
        age_col: str = "age",
        pop_col: str = "P",
    ) -> Dict[str, NDArray]:
        """
        Generate global baseline contact intensity matrix from a DataFrame.

        This is an alternative to `generate_single` that accepts a pandas
        DataFrame directly instead of a PopulationConstructor.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame containing population sizes with columns for age and
            population size.
        mean_intensity : float, default=15.0
            Average marginal contact intensity C.
        seed : int, optional
            Random seed for reproducibility.
        age_col : str, default "age"
            Name of the column containing age values.
        pop_col : str, default "P"
            Name of the column containing population sizes.

        Returns
        -------
        dict
            Dictionary with single key "All->All" mapping to contact intensity
            matrix M (A, A).

        Examples
        --------
        >>> import pandas as pd
        >>> from cntmosaic.sim import MatrixGenerator
        >>> from cntmosaic.datasets import load_template_patterns

        >>> templates = load_template_patterns('United_States', max_age=10)
        >>> mg = MatrixGenerator(templates)

        >>> df_pop = pd.DataFrame({
        ...     'age': [0, 1, 2],
        ...     'P': [1000, 1500, 2000]
        ... })
        >>> M_dict = mg.generate_single_from_df(df_pop, mean_intensity=15.0, seed=123)
        >>> M_dict['All->All'].shape
        (3, 3)
        """
        rng = np.random.default_rng(seed)

        # Preprocess DataFrame (ignore stratification for baseline)
        ref_age_dist, _, _, _ = self._preprocess_df(
            df, strat_var_cols=[], age_col=age_col, pop_col=pop_col
        )

        # Sample mixture weights and create mixed pattern
        weights = self._sample_mixture_weights(rng)
        pattern = self._create_mixed_pattern(weights)

        # Scale by mean intensity
        M = pattern * mean_intensity

        # Enforce reciprocity using global population
        P_global = np.diag(ref_age_dist)
        M = self._enforce_reciprocity(M, P_global)

        return {"All->All": M}

    def _generate_deviation_matrix(
        self,
        strat_idx_pair: Tuple[int, int],
        eta: float,
        rng: np.random.Generator,
        assortativity: float = 0.0,
        intra_group: bool = False,
    ) -> NDArray:
        """
        Generate deviation matrix D^{k,l} for a stratification category pair.

        Process:
        1. Sample template mixture T^{k,l}
        2. Center in log-space: E^{k,l}_{a,b} = log T^{k,l}_{a,b} - mean(log T^{k,l})
        3. Apply scaling: D^{k,l} = exp(η E^{k,l})
        4. Enforce reciprocity on diagonal/off-diagonal pairs

        Parameters
        ----------
        strat_idx_pair : tuple of int
            Category indices (k, l) within a single stratification variable
        eta : float
            Deviation strength parameter (0 = no deviation, higher = stronger)
        rng : np.random.Generator
            Random number generator

        Returns
        -------
        NDArray
            Deviation matrix D^{k,l} (A, A)
        """
        k, ell = strat_idx_pair

        # Sample template mixture for this category pair
        weights = self._sample_mixture_weights(rng)
        T = self._create_mixed_pattern(weights)

        # Center in log-space
        log_T = np.log(T + 1e-10)  # Add small constant to avoid log(0)
        mean_log_T = log_T.mean()
        E = log_T - mean_log_T

        # Apply scaling
        D = np.exp(eta * E + assortativity * intra_group)

        # Enforce reciprocity
        is_diagonal = k == ell
        D = self._enforce_deviation_reciprocity(D, is_diagonal)

        return D

    def _enforce_deviation_reciprocity(self, D: NDArray, is_diagonal: bool) -> NDArray:
        """
        Enforce reciprocity on deviation matrix.

        For diagonal blocks (k=k): D[a,b] ← sqrt(D[a,b] \times D[b,a])
        For off-diagonal blocks (k≠l): handled via pairing D^{k,l} and D^{l,k}

        Parameters
        ----------
        D : NDArray
            Deviation matrix (A, A)
        is_diagonal : bool
            Whether this is a diagonal block (same category)

        Returns
        -------
        NDArray
            Reciprocal deviation matrix (A, A)
        """
        if is_diagonal:
            # D[a,b] ← sqrt(D[a,b] \times D[b,a])
            D = np.sqrt(D * D.T)
        # For off-diagonal, reciprocity is enforced by ensuring D^{l,k}[b,a] = D^{k,l}[a,b]
        # This is handled in the calling code by proper indexing

        return D

    def _combine_deviations(self, deviation_list: List[NDArray]) -> NDArray:
        """
        Combine deviations across multiple stratification variables.

        d^{s,t}_{a,b} = ∏_j D^{k_j(s), l_j(t)}_{a,b}

        Parameters
        ----------
        deviation_list : list of NDArray
            List of deviation matrices [D^{k_1(s), l_1(t)}, D^{k_2(s), l_2(t)}, ...]

        Returns
        -------
        NDArray
            Combined deviation d^{s,t} (A, A)
        """
        combined = np.ones((self.n_ages, self.n_ages))
        for D in deviation_list:
            combined *= D
        return combined

    def _normalize_deviations(
        self,
        d_all: Dict[Tuple[int, int], NDArray],
        Q: NDArray,
    ) -> Dict[Tuple[int, int], NDArray]:
        """
        Normalize deviations using population proportions.

        \delta^{s,t}_{a,b} = d^{s,t}_{a,b} / \sum_{u,v} d^{u,v}_{a,b} S^{u,v}_{a,b}

        where S^{s,t}_{a,b} = (P^s_a / P_a) \times (P^t_b / P_b) = Q[s,a] \times Q[t,b]

        Parameters
        ----------
        d_all : dict
            Combined deviation matrices {(s,t): d^{s,t}}
        Q : NDArray
            Population proportion matrix (n_strata, n_ages)
            Q[s,a] = P(stratum s | age a)

        Returns
        -------
        dict
            Normalized deviation matrices {(s,t): \delta^{s,t}}
        """
        n_strata = Q.shape[0]

        # Convert dict to 4D array for vectorized operations
        # d_array[s, t, a, b] = d^{s,t}_{a,b}
        d_array = np.zeros((n_strata, n_strata, self.n_ages, self.n_ages))
        for s in range(n_strata):
            for t in range(n_strata):
                d_array[s, t, :, :] = d_all[(s, t)]

        # Compute S^{u,v}_{a,b} = Q[u,a] * Q[v,b] for all combinations using broadcasting
        # S[u, v, a, b]
        S = Q[:, None, :, None] * Q[None, :, None, :]

        # Compute denominator: \sum_{u,v} d^{u,v}_{a,b} S^{u,v}_{a,b}
        # Sum over strata dimensions (0, 1), leaving age dimensions (a, b)
        denominator = (d_array * S).sum(axis=(0, 1))  # shape: (n_ages, n_ages)

        # Normalize: \delta^{s,t}_{a,b} = d^{s,t}_{a,b} / denominator_{a,b}
        delta_array = d_array / denominator[None, None, :, :]

        # Convert back to dict format
        delta_all = {}
        for s in range(n_strata):
            for t in range(n_strata):
                delta_all[(s, t)] = delta_array[s, t, :, :]

        return delta_all

    def generate_partial(
        self,
        popcon: PopulationConstructor,
        mean_intensity: float = 15.0,
        seed: Optional[int] = None,
    ) -> Dict[str, NDArray]:
        """
        Generate partial contact matrices: one per stratum to general population.

        Each stratum has a contact pattern representing contacts with the general
        population (averaged across all strata).

        Process:
        1. Generate baseline rate matrix \gamma
        2. Generate deviations for each stratum (diagonal only, no reciprocity)
        3. Normalize deviations: $\delta^s_{a,b} = d^s_{a,b} / \sum_u d^u_{a,b} S^u_a$
        4. Compute intensities: m^s = \gamma \delta^s P

        Parameters
        ----------
        popcon : PopulationConstructor
            Population structure with stratifications
        mean_intensity : float, default=15.0
            Average marginal contact intensity
        seed : int, optional
            Random seed for reproducibility

        Returns
        -------
        dict
            Maps stratum label to contact intensity matrix {"label->All": M^s}
            M^s represents contacts from stratum s to general population

        Examples
        --------
        >>> templates = load_template_patterns('United_States', max_age=10)
        >>> mg = MatrixGenerator(templates)

        >>> ref_age_dist = np.array([1000, 1500, 2000])
        >>> gender_strat = Stratification('gender', 2, ref_age_dist, labels=['Male', 'Female'], seed=42)
        >>> pop = PopulationConstructor(gender_strat)

        >>> M_partial = mg.generate_partial(pop, mean_intensity=15.0, seed=123)
        >>> list(M_partial.keys())
        ['Male->All', 'Female->All']
        >>> M_partial['Male->All'].shape
        (3, 3)

        Notes
        -----
        For partial matrices, we generate stratum-specific contact patterns
        but average over the contact recipient's stratum (general population).
        No reciprocity is enforced on deviation matrices in the partial case.
        """
        rng = np.random.default_rng(seed)

        # Generate baseline matrix and convert to rates (use internal generation)
        weights = self._sample_mixture_weights(rng)
        pattern = self._create_mixed_pattern(weights)
        M_baseline = pattern * mean_intensity
        P_global_diag = np.diag(popcon.ref_age_dist)
        M_baseline = self._enforce_reciprocity(M_baseline, P_global_diag)
        Gamma_baseline = M_baseline @ np.linalg.inv(P_global_diag)

        # Check if we have stratifications
        if isinstance(popcon.strats, Stratification):
            strats = [popcon.strats]
        else:
            strats = popcon.strats

        n_strata = popcon.Q.shape[0]
        Q = popcon.Q

        # Sample eta for each stratification variable
        eta_values = {strat.name: rng.uniform(0, 1) for strat in strats}

        # Generate deviation matrices for each stratification variable (diagonal only)
        deviation_matrices_by_strat = {}
        for strat in strats:
            eta = eta_values[strat.name]
            dev_dict = {}
            for k in range(strat.n_strata):
                # For partial, diagonal deviations only, NO reciprocity enforcement
                weights = self._sample_mixture_weights(rng)
                T = self._create_mixed_pattern(weights)

                # Center in log-space
                log_T = np.log(T + 1e-10)
                mean_log_T = log_T.mean()
                E = log_T - mean_log_T

                # Apply scaling (no reciprocity for partial case)
                D = np.exp(eta * E)
                dev_dict[k] = D
            deviation_matrices_by_strat[strat.name] = dev_dict

        # Combine deviations for each stratum
        d_all = {}
        for s in range(n_strata):
            # Get stratification indices for this stratum
            if len(strats) == 1:
                strat_indices = [s]
            else:
                # Multi-stratification: decode stratum index
                strat_indices = self._decode_stratum_index(s, strats)

            # Combine deviations across stratification variables
            deviation_list = [
                deviation_matrices_by_strat[strat.name][strat_indices[j]]
                for j, strat in enumerate(strats)
            ]
            d_all[s] = self._combine_deviations(deviation_list)

        # Normalize deviations using simplified formula for partial case
        # Convert to array for vectorization: d_array[s, a, b]
        d_array = np.zeros((n_strata, self.n_ages, self.n_ages))
        for s in range(n_strata):
            d_array[s, :, :] = d_all[s]

        # S^s_a = Q[s, a] for partial case
        # Compute denominator: \sum_u d^u_{a,b} S^u_a = \sum_u d^u_{a,b} Q[u,a]
        # Broadcasting: Q[:, :, None] has shape (n_strata, n_ages, 1)
        #               d_array has shape (n_strata, n_ages, n_ages)
        # Weighted sum over strata: (n_ages, n_ages)
        denominator = (d_array * Q[:, :, None]).sum(axis=0)

        # Normalize: \delta^s_{a,b} = d^s_{a,b} / denominator_{a,b}
        delta_array = d_array / denominator[None, :, :]

        # Compute stratified contact intensities: m^s = \gamma \delta^s P
        partial_matrices = {}
        for s in range(n_strata):
            M_s = (Gamma_baseline * delta_array[s, :, :]) @ P_global_diag
            label = self._get_stratum_label(s, strats)
            partial_matrices[f"{label}->All"] = M_s

        return partial_matrices

    def generate_partial_from_df(
        self,
        df: pd.DataFrame,
        strat_var_cols: Optional[List[str]] = None,
        mean_intensity: float = 15.0,
        seed: Optional[int] = None,
        age_col: str = "age",
        pop_col: str = "P",
    ) -> Dict[str, NDArray]:
        """
        Generate partial contact matrices from a DataFrame.

        This is an alternative to `generate_partial` that accepts a pandas
        DataFrame directly instead of a PopulationConstructor.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame containing population sizes with columns for age,
            population size, and stratification variables.
        strat_var_cols : list of str, optional
            Names of columns to use as stratification variables.
            If None, all columns except age_col and pop_col are used.
        mean_intensity : float, default=15.0
            Average marginal contact intensity.
        seed : int, optional
            Random seed for reproducibility.
        age_col : str, default "age"
            Name of the column containing age values.
        pop_col : str, default "P"
            Name of the column containing population sizes.

        Returns
        -------
        dict
            Maps stratum label to contact intensity matrix {"label->All": M^s}.

        Examples
        --------
        >>> import pandas as pd
        >>> from cntmosaic.sim import MatrixGenerator
        >>> from cntmosaic.datasets import load_template_patterns

        >>> templates = load_template_patterns('United_States', max_age=10)
        >>> mg = MatrixGenerator(templates)

        >>> df_pop = pd.DataFrame({
        ...     'age': [0, 0, 1, 1, 2, 2],
        ...     'gender': ['Male', 'Female', 'Male', 'Female', 'Male', 'Female'],
        ...     'P': [500, 500, 750, 750, 1000, 1000]
        ... })
        >>> M_partial = mg.generate_partial_from_df(
        ...     df_pop, strat_var_cols=['gender'], mean_intensity=15.0, seed=123
        ... )
        >>> list(M_partial.keys())
        ['Female->All', 'Male->All']
        """
        rng = np.random.default_rng(seed)

        # Preprocess DataFrame
        ref_age_dist, Q, P_matrix, strat_infos = self._preprocess_df(
            df, strat_var_cols=strat_var_cols, age_col=age_col, pop_col=pop_col
        )

        n_strata = Q.shape[0]
        n_ages = len(ref_age_dist)

        # Generate baseline matrix and convert to rates
        weights = self._sample_mixture_weights(rng)
        pattern = self._create_mixed_pattern(weights)
        M_baseline = pattern * mean_intensity
        P_global_diag = np.diag(ref_age_dist)
        M_baseline = self._enforce_reciprocity(M_baseline, P_global_diag)
        Gamma_baseline = M_baseline @ np.linalg.inv(P_global_diag)

        # Handle unstratified case
        if len(strat_infos) == 0:
            return {"All->All": M_baseline}

        # Sample eta for each stratification variable
        eta_values = {info.name: rng.uniform(0, 1) for info in strat_infos}

        # Generate deviation matrices for each stratification variable (diagonal only)
        deviation_matrices_by_strat = {}
        for info in strat_infos:
            eta = eta_values[info.name]
            dev_dict = {}
            for k in range(info.n_strata):
                # For partial, diagonal deviations only, NO reciprocity enforcement
                weights = self._sample_mixture_weights(rng)
                T = self._create_mixed_pattern(weights)

                # Center in log-space
                log_T = np.log(T + 1e-10)
                mean_log_T = log_T.mean()
                E = log_T - mean_log_T

                # Apply scaling (no reciprocity for partial case)
                D = np.exp(eta * E)
                dev_dict[k] = D
            deviation_matrices_by_strat[info.name] = dev_dict

        # Combine deviations for each stratum
        d_all = {}
        for s in range(n_strata):
            # Get stratification indices for this stratum
            if len(strat_infos) == 1:
                strat_indices = [s]
            else:
                strat_indices = self._decode_stratum_index_from_infos(s, strat_infos)

            # Combine deviations across stratification variables
            deviation_list = [
                deviation_matrices_by_strat[info.name][strat_indices[j]]
                for j, info in enumerate(strat_infos)
            ]
            d_all[s] = self._combine_deviations(deviation_list)

        # Normalize deviations
        d_array = np.zeros((n_strata, n_ages, n_ages))
        for s in range(n_strata):
            d_array[s, :, :] = d_all[s]

        denominator = (d_array * Q[:, :, None]).sum(axis=0)
        delta_array = d_array / denominator[None, :, :]

        # Compute stratified contact intensities
        partial_matrices = {}
        for s in range(n_strata):
            M_s = (Gamma_baseline * delta_array[s, :, :]) @ P_global_diag
            label = self._get_stratum_label_from_infos(s, strat_infos)
            partial_matrices[f"{label}->All"] = M_s

        return partial_matrices

    def generate_full(
        self,
        popcon: PopulationConstructor,
        mean_intensity: float = 15.0,
        assortativity: float = 0.0,
        seed: Optional[int] = None,
    ) -> Dict[str, NDArray]:
        """
        Generate full stratified contact matrices for all stratum pairs.

        Creates contact patterns for every possible pair of strata, capturing
        heterogeneous mixing patterns between different population groups.

        Process:
        1. Generate baseline matrix and rate matrix
        2. Sample eta ~ Uniform(0,1) for each stratification variable
        3. Generate deviation matrices D^{k,l} for each stratification
        4. Combine deviations: d^{s,t} = ∏_j D^{k_j(s), l_j(t)}
        5. Normalize deviations: \delta^{s,t} using population proportions
        6. Compute stratified intensities: m^{s,t} = \gamma \delta^{s,t} P^t

        Parameters
        ----------
        popcon : PopulationConstructor
            Population structure with stratifications
        mean_intensity : float, default=15.0
            Average marginal contact intensity
        assortativity : float, default=0.0
            Controls the strength of preferential within-stratum mixing.
            Higher values increase contact intensity for same-stratum pairs
            (s = t) relative to cross-stratum pairs.
        seed : int, optional
            Random seed for reproducibility

        Returns
        -------
        dict
            Maps "source_label->target_label" to contact intensity matrix
            {"label_s->label_t": M^{s,t}} where M^{s,t}[a,b] is intensity from age a in
            stratum s to age b in stratum t

        Examples
        --------
        >>> templates = load_template_patterns('United_States', max_age=10)
        >>> mg = MatrixGenerator(templates)

        >>> ref_age_dist = np.array([1000, 1500, 2000])
        >>> gender_strat = Stratification('gender', 2, ref_age_dist, labels=['Male', 'Female'], seed=42)
        >>> pop = PopulationConstructor(gender_strat)

        >>> M_full = mg.generate_full(pop, mean_intensity=15.0, seed=123)
        >>> list(M_full.keys())
        ['Male->Male', 'Male->Female', 'Female->Male', 'Female->Female']
        >>> M_full['Male->Female'].shape
        (3, 3)

        Notes
        -----
        The full generation enforces:
        - Reciprocity: \gamma^{s,t} P^t = (\gamma^{t,s} P^s)^T
        - Weighted normalization: \sum_{s,t} \delta^{s,t} Q^s ⊗ Q^t = 1
        """
        rng = np.random.default_rng(seed)

        # Generate baseline matrix and convert to rates (use internal generation)
        weights = self._sample_mixture_weights(rng)
        pattern = self._create_mixed_pattern(weights)
        M_baseline = pattern * mean_intensity
        P_global_diag = np.diag(popcon.ref_age_dist)
        M_baseline = self._enforce_reciprocity(M_baseline, P_global_diag)
        Gamma_baseline = M_baseline @ np.linalg.inv(P_global_diag)

        # Get stratification structure
        if isinstance(popcon.strats, Stratification):
            strats = [popcon.strats]
        else:
            strats = popcon.strats

        n_strata = popcon.Q.shape[0]
        Q = popcon.Q

        # Sample eta for each stratification variable
        eta_values = {strat.name: rng.uniform(0, 1) for strat in strats}

        # Generate all deviation matrices for each stratification variable
        deviation_matrices_by_strat = {}
        for strat in strats:
            eta = eta_values[strat.name]
            dev_dict = {}

            # Generate deviation matrices for all category pairs
            # For off-diagonal pairs, generate once and use transpose for reciprocal
            generated_pairs = set()

            for k in range(strat.n_strata):
                for ell in range(strat.n_strata):
                    if (k, ell) in generated_pairs:
                        continue

                    if k == ell:
                        # Diagonal: enforce symmetry
                        D = self._generate_deviation_matrix(
                            (k, ell), eta, rng, assortativity, True
                        )
                        dev_dict[(k, ell)] = D
                        generated_pairs.add((k, ell))
                    else:
                        # Off-diagonal: generate D^{k,l}, then set D^{l,k} = D^{k,l}.T
                        D_kl = self._generate_deviation_matrix((k, ell), eta, rng)
                        dev_dict[(k, ell)] = D_kl
                        dev_dict[(ell, k)] = D_kl.T
                        generated_pairs.add((k, ell))
                        generated_pairs.add((ell, k))

            deviation_matrices_by_strat[strat.name] = dev_dict

        # Combine deviations for all stratum pairs
        d_all = {}
        for s in range(n_strata):
            for t in range(n_strata):
                # Get stratification indices for each stratum
                if len(strats) == 1:
                    s_indices = [s]
                    t_indices = [t]
                else:
                    s_indices = self._decode_stratum_index(s, strats)
                    t_indices = self._decode_stratum_index(t, strats)

                # Combine deviations across stratification variables
                deviation_list = [
                    deviation_matrices_by_strat[strat.name][
                        (s_indices[j], t_indices[j])
                    ]
                    for j, strat in enumerate(strats)
                ]
                d_all[(s, t)] = self._combine_deviations(deviation_list)

        # Normalize deviations
        delta_all = self._normalize_deviations(d_all, Q)

        # Compute stratified contact intensities: m^{s,t} = \gamma \delta^{s,t} P^t
        M_full = {}
        for s in range(n_strata):
            for t in range(n_strata):
                P_t = np.diag(popcon.P[t, :])
                # Element-wise multiplication, then matrix multiply
                M_st = (Gamma_baseline * delta_all[(s, t)]) @ P_t
                label_s = self._get_stratum_label(s, strats)
                label_t = self._get_stratum_label(t, strats)
                M_full[f"{label_s}->{label_t}"] = M_st

        return M_full

    def generate_full_from_df(
        self,
        df: pd.DataFrame,
        strat_var_cols: Optional[List[str]] = None,
        mean_intensity: float = 15.0,
        assortativity: float = 0.0,
        seed: Optional[int] = None,
        age_col: str = "age",
        pop_col: str = "P",
    ) -> Dict[str, NDArray]:
        """
        Generate full stratified contact matrices from a DataFrame.

        This is an alternative to `generate_full` that accepts a pandas
        DataFrame directly instead of a PopulationConstructor.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame containing population sizes with columns for age,
            population size, and stratification variables.
        strat_var_cols : list of str, optional
            Names of columns to use as stratification variables.
            If None, all columns except age_col and pop_col are used.
        mean_intensity : float, default=15.0
            Average marginal contact intensity.
        assortativity : float, default=0.0
            Controls the strength of preferential within-stratum mixing.
            Higher values increase contact intensity for same-stratum pairs
            (s = t) relative to cross-stratum pairs.
        seed : int, optional
            Random seed for reproducibility.
        age_col : str, default "age"
            Name of the column containing age values.
        pop_col : str, default "P"
            Name of the column containing population sizes.

        Returns
        -------
        dict
            Maps "source_label->target_label" to contact intensity matrix.

        Examples
        --------
        >>> import pandas as pd
        >>> from cntmosaic.sim import MatrixGenerator
        >>> from cntmosaic.datasets import load_template_patterns

        >>> templates = load_template_patterns('United_States', max_age=10)
        >>> mg = MatrixGenerator(templates)

        >>> df_pop = pd.DataFrame({
        ...     'age': [0, 0, 1, 1, 2, 2],
        ...     'gender': ['Male', 'Female', 'Male', 'Female', 'Male', 'Female'],
        ...     'P': [500, 500, 750, 750, 1000, 1000]
        ... })
        >>> M_full = mg.generate_full_from_df(
        ...     df_pop, strat_var_cols=['gender'], mean_intensity=15.0, seed=123
        ... )
        >>> sorted(M_full.keys())
        ['Female->Female', 'Female->Male', 'Male->Female', 'Male->Male']
        """
        rng = np.random.default_rng(seed)

        # Preprocess DataFrame
        ref_age_dist, Q, P_matrix, strat_infos = self._preprocess_df(
            df, strat_var_cols=strat_var_cols, age_col=age_col, pop_col=pop_col
        )

        n_strata = Q.shape[0]
        n_ages = len(ref_age_dist)

        # Generate baseline matrix and convert to rates
        weights = self._sample_mixture_weights(rng)
        pattern = self._create_mixed_pattern(weights)
        M_baseline = pattern * mean_intensity
        P_global_diag = np.diag(ref_age_dist)
        M_baseline = self._enforce_reciprocity(M_baseline, P_global_diag)
        Gamma_baseline = M_baseline @ np.linalg.inv(P_global_diag)

        # Handle unstratified case
        if len(strat_infos) == 0:
            return {"All->All": M_baseline}

        # Sample eta for each stratification variable
        eta_values = {info.name: rng.uniform(0, 1) for info in strat_infos}

        # Generate all deviation matrices for each stratification variable
        deviation_matrices_by_strat = {}
        for info in strat_infos:
            eta = eta_values[info.name]
            dev_dict = {}

            # Generate deviation matrices for all category pairs
            generated_pairs = set()

            for k in range(info.n_strata):
                for ell in range(info.n_strata):
                    if (k, ell) in generated_pairs:
                        continue

                    if k == ell:
                        # Diagonal: enforce symmetry
                        D = self._generate_deviation_matrix(
                            (k, ell), eta, rng, assortativity, True
                        )
                        dev_dict[(k, ell)] = D
                        generated_pairs.add((k, ell))
                    else:
                        # Off-diagonal: generate D^{k,l}, then set D^{l,k} = D^{k,l}.T
                        D_kl = self._generate_deviation_matrix((k, ell), eta, rng)
                        dev_dict[(k, ell)] = D_kl
                        dev_dict[(ell, k)] = D_kl.T
                        generated_pairs.add((k, ell))
                        generated_pairs.add((ell, k))

            deviation_matrices_by_strat[info.name] = dev_dict

        # Combine deviations for all stratum pairs
        d_all = {}
        for s in range(n_strata):
            for t in range(n_strata):
                # Get stratification indices for each stratum
                if len(strat_infos) == 1:
                    s_indices = [s]
                    t_indices = [t]
                else:
                    s_indices = self._decode_stratum_index_from_infos(s, strat_infos)
                    t_indices = self._decode_stratum_index_from_infos(t, strat_infos)

                # Combine deviations across stratification variables
                deviation_list = [
                    deviation_matrices_by_strat[info.name][(s_indices[j], t_indices[j])]
                    for j, info in enumerate(strat_infos)
                ]
                d_all[(s, t)] = self._combine_deviations(deviation_list)

        # Normalize deviations
        delta_all = self._normalize_deviations(d_all, Q)

        # Compute stratified contact intensities: m^{s,t} = \gamma \delta^{s,t} P^t
        M_full = {}
        for s in range(n_strata):
            for t in range(n_strata):
                P_t = np.diag(P_matrix[t, :])
                # Element-wise multiplication, then matrix multiply
                M_st = (Gamma_baseline * delta_all[(s, t)]) @ P_t
                label_s = self._get_stratum_label_from_infos(s, strat_infos)
                label_t = self._get_stratum_label_from_infos(t, strat_infos)
                M_full[f"{label_s}->{label_t}"] = M_st

        return M_full

    def _decode_stratum_index(
        self, stratum_idx: int, strats: List[Stratification]
    ) -> List[int]:
        """
        Decode global stratum index into per-stratification category indices.

        For multiple stratifications with n_1, n_2, ..., n_J categories,
        stratum index s ∈ [0, n_1 \times n_2 \times ... \times n_J) maps to
        (k_1, k_2, ..., k_J) where k_j \in [0, n_j).

        Parameters
        ----------
        stratum_idx : int
            Global stratum index
        strats : list of Stratification
            List of stratification variables

        Returns
        -------
        list of int
            Category indices [k_1, k_2, ..., k_J]
        """
        indices = []
        remaining = stratum_idx

        # Compute strides for each stratification
        strides = []
        stride = 1
        for strat in reversed(strats):
            strides.insert(0, stride)
            stride *= strat.n_strata

        # Decode indices
        for j, strat in enumerate(strats):
            k_j = remaining // strides[j]
            indices.append(k_j)
            remaining = remaining % strides[j]

        return indices

    def _get_stratum_label(self, stratum_idx: int, strats: List[Stratification]) -> str:
        """
        Get string label for a stratum index.

        For single stratification, returns the label directly.
        For multiple stratifications, concatenates labels with underscore.

        Parameters
        ----------
        stratum_idx : int
            Global stratum index
        strats : list of Stratification
            List of stratification variables

        Returns
        -------
        str
            Stratum label (e.g., "M", "M_Low", "F_High")

        Examples
        --------
        Single stratification with labels ["M", "F"]:
        - stratum_idx=0 → "M"
        - stratum_idx=1 → "F"

        Two stratifications with labels ["M", "F"] and ["Low", "Mid", "High"]:
        - stratum_idx=0 → "M_Low"
        - stratum_idx=1 → "M_Mid"
        - stratum_idx=3 → "F_Low"
        """
        if len(strats) == 1:
            return strats[0].labels[stratum_idx]
        else:
            indices = self._decode_stratum_index(stratum_idx, strats)
            labels = [strat.labels[idx] for strat, idx in zip(strats, indices)]
            return "_".join(labels)
