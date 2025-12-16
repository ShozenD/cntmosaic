from typing import Dict, List, Tuple, Union

import numpy as np
from numpy.typing import NDArray

from ._ParticipantGenerator import Subgroup


class MatrixGenerator:
    """
    Generate synthetic contact intensity matrices for epidemiological modeling.

    Three generation modes:
    - single: One matrix for a homogeneous population
    - partial: One matrix per subgroup against general population
    - full: All pairwise subgroup matrices
    """

    REQUIRED_TEMPLATES = {"household", "school", "work", "community"}

    def __init__(self, templates: dict[str, NDArray]):
        """
        Initialize generator with contact pattern templates.

        Parameters
        ----------
        templates : dict of NDArray
                Must contain keys: 'household', 'school', 'work', 'community'.
                Each matrix should be AxA where A is the number of age groups.
                Use cntmosaic.datasets.load_template_patterns() to load default templates.
        """
        self._validate_templates(templates)
        self.templates = self._normalize_templates(templates)

    def _validate_templates(self, templates: Dict[str, NDArray]) -> None:
        """Validate template structure."""
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
        """Normalize templates so average marginal intensity equals 1."""
        normalized = {}
        for name, T in templates.items():
            A = T.shape[0]
            mean_intensity = T.sum() / A
            normalized[name] = T / mean_intensity if mean_intensity > 0 else T
        return normalized

    def _generate_mixed_pattern(self, rng: np.random.Generator) -> NDArray:
        """
        Generate mixed contact pattern from templates.

        Uses Dirichlet(1,1,1,1) to sample mixing weights.
        """
        weights = rng.dirichlet(np.ones(4))

        pattern = (
            weights[0] * self.templates["community"]
            + weights[1] * self.templates["school"]
            + weights[2] * self.templates["work"]
            + weights[3] * self.templates["household"]
        )
        return pattern

    def _enforce_reciprocity(self, M: NDArray, P: NDArray) -> NDArray:
        """
        Enforce reciprocity condition: PM = (PM)ᵀ

        Applies: M† = ½(M + P⁻¹MᵀP)
        """
        P_inv = np.linalg.inv(P)
        return 0.5 * (M + P_inv @ M.T @ P)

    def generate_single(self, subgroup: Subgroup, seed: int = None) -> NDArray:
        """
        Generate contact intensity matrix for a homogeneous population.

        Parameters
        ----------
        subgroup : Subgroup
                        Subgroup object containing population size and age distribution
        seed : int, optional
                        Random seed for reproducibility

        Returns
        -------
        NDArray
                        Contact intensity matrix M (A×A)
                        Element M[i,j] = expected contact intensity from age i to age j
        """
        rng = np.random.default_rng(seed)

        # Generate mixed pattern and scale
        pattern = self._generate_mixed_pattern(rng)
        M = pattern * subgroup.mean_cint_margin

        # Enforce reciprocity
        P = np.diag(subgroup.age_dist)
        M = self._enforce_reciprocity(M, P)

        return M

    def generate_partial(
        self, subgroups: List[Subgroup], seed: int = None
    ) -> Dict[int, NDArray]:
        """
        Generate contact matrices for subgroups (partial case).

        Each subgroup has its own contact pattern with the general population.
        Reciprocity is enforced: Pₖ Mₖ = (Pₖ Mₖ)ᵀ for each subgroup k.

        Parameters
        ----------
        subgroups : list of Subgroup
                        List of Subgroup objects, labeled automatically as 0, 1, 2, ...
        seed : int, optional
                        Random seed for reproducibility

        Returns
        -------
        dict
                        Maps subgroup labels (integers) to contact intensity matrices
                        {label: M_k} where M_k is the contact matrix for subgroup k

        Examples
        --------
        >>> subgroups = [
        ...     Subgroup(young_dist, mean_intensity=20.0),
        ...     Subgroup(old_dist, mean_intensity=10.0)
        ... ]
        >>> matrices = generator.generate_partial(subgroups, seed=42)
        >>> young_contacts = matrices[0]
        """
        rng = np.random.default_rng(seed)

        matrices = {}
        for subgroup in subgroups:
            # Generate mixed pattern and scale
            pattern = self._generate_mixed_pattern(rng)
            M_k = pattern * subgroup.mean_cint_margin

            # Enforce reciprocity for this subgroup
            P_k = np.diag(subgroup.age_dist)
            M_k = self._enforce_reciprocity(M_k, P_k)

            matrices[subgroup.label + "->All"] = M_k

        return matrices

    def generate_full(
        self, subgroups: List[Subgroup], seed: int = None
    ) -> Dict[Tuple, NDArray]:
        """
        Generate all pairwise subgroup contact matrices (full case).

        Enforces reciprocity conditions:
        - Within subgroup: Γₖₖ = Γₖₖᵀ  (symmetric)
        - Between subgroups: Γₖₗ = Γₗₖᵀ  (transpose relation)

        Parameters
        ----------
        subgroups : list of Subgroup
                        List of Subgroup objects, labeled automatically as 0, 1, 2, ...
        seed : int, optional
                        Random seed for reproducibility

        Returns
        -------
        dict
                        Maps (source, target) tuples to contact matrices
                        {(k, l): M_kl} where M_kl represents contacts from k to l

        Examples
        --------
        >>> subgroups = [
        ...     Subgroup(young_dist, 20.0),
        ...     Subgroup(old_dist, 10.0)
        ... ]
        >>> matrices = generator.generate_full(subgroups, seed=42)
        >>> young_to_old = matrices[(0, 1)]
        >>> old_to_young = matrices[(1, 0)]
        """
        rng = np.random.default_rng(seed)

        matrices = {}

        # Step 1: Generate diagonal blocks (within-subgroup contacts)
        for subgroup in subgroups:
            pattern = self._generate_mixed_pattern(rng)
            M_kk = pattern * subgroup.mean_cint_margin

            # Enforce within-subgroup reciprocity
            P_k = np.diag(subgroup.age_dist)
            M_kk = self._enforce_reciprocity(M_kk, P_k)

            matrices[(subgroup.label, subgroup.label)] = M_kk

        # Step 2: Generate off-diagonal blocks (between-subgroup contacts)
        for i, subgroup_k in enumerate(subgroups):
            for subgroup_l in subgroups[i + 1 :]:
                # Generate M_kl
                pattern = self._generate_mixed_pattern(rng)
                # Use average of the two subgroup intensities
                mean_intensity = (
                    subgroup_k.mean_cint_margin + subgroup_l.mean_cint_margin
                ) / 2
                M_kl = pattern * mean_intensity

                # Construct reciprocal matrix M_lk
                # M_lk = P_l⁻¹ M_klᵀ P_k
                P_k = np.diag(subgroup_k.age_dist)
                P_l = np.diag(subgroup_l.age_dist)
                P_l_inv = np.linalg.inv(P_l)

                M_lk = P_l_inv @ M_kl.T @ P_k

                matrices[subgroup_k.label + "->" + subgroup_l.label] = M_kl
                matrices[subgroup_l.label + "->" + subgroup_k.label] = M_lk

        return matrices

    def get_contact_rate_matrix(
        self, contact_intensity: NDArray, age_dist: NDArray
    ) -> NDArray:
        """
        Convert contact intensity matrix to contact rate matrix.

        Γ = M P⁻¹

        Parameters
        ----------
        contact_intensity : NDArray
                        Contact intensity matrix M
        age_dist : NDArray
                        Population age distribution

        Returns
        -------
        NDArray
                        Contact rate matrix Γ where γᵢⱼ = rate of contacts from i to j
        """
        P = np.diag(age_dist)
        P_inv = np.linalg.inv(P)
        return contact_intensity @ P_inv
