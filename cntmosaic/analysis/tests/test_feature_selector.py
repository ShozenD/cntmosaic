"""
Tests for FeatureSelector, ModelConfig, and FeatureSelectionResult.

Covers:
- Validation: non-subset strat vars, FULL-to-PARTIAL demotion, duplicate names
- _build_kwargs_dict: correct keys for unstratified and PARTIAL-subset candidates
- Integration smoke test via run() with minimal SVI steps
- FeatureSelectionResult.best_model property
"""

import numpy as np
import pandas as pd
import pytest
from jax.random import PRNGKey


# ---------------------------------------------------------------------------
# Helpers to build minimal loaders (no .load() call needed for validation tests)
# ---------------------------------------------------------------------------


def _make_loader(part_strat_vars=None, cnt_strat_vars=None, n_ages=5):
    """
    Build a ContactSurveyLoader from minimal synthetic data.

    Parameters are unprefixed names (e.g. 'sex', not 'part_sex').
    """
    from cntmosaic.dataloader import (
        ContactData,
        ContactSurveyLoader,
        ParticipantData,
        PopulationData,
        StratificationData,
    )

    rng = np.random.default_rng(0)
    ages = list(range(n_ages))
    sexes = ["M", "F"]
    n_part = 20

    # Participant DataFrame
    part_records = {
        "id": np.arange(n_part),
        "age": rng.choice(ages, n_part),
    }
    if part_strat_vars and "sex" in part_strat_vars:
        part_records["sex"] = pd.Categorical(
            rng.choice(sexes, n_part), categories=sexes
        )
    if part_strat_vars and "educ" in part_strat_vars:
        educs = ["Low", "Mid", "High"]
        part_records["educ"] = pd.Categorical(
            rng.choice(educs, n_part), categories=educs
        )
    df_part = pd.DataFrame(part_records)

    # Contact DataFrame
    cnt_records = {
        "id": rng.choice(df_part["id"].values, 40),
        "cnt_age": rng.choice(ages, 40),
    }
    if cnt_strat_vars and "sex" in cnt_strat_vars:
        cnt_records["sex"] = pd.Categorical(
            rng.choice(sexes, 40), categories=sexes
        )
    df_cnt = pd.DataFrame(cnt_records)

    # Population DataFrame
    pop_records = {"age": ages, "P": rng.integers(1000, 2000, n_ages)}
    if part_strat_vars and "sex" in part_strat_vars:
        rows = []
        for age in ages:
            for sex in sexes:
                rows.append(
                    {"age": age, "sex": sex, "P": rng.integers(500, 1000)}
                )
        df_pop = pd.DataFrame(rows)
        df_pop["sex"] = pd.Categorical(df_pop["sex"], categories=sexes)
    else:
        df_pop = pd.DataFrame(pop_records)

    # Build containers
    part_sv = list(part_strat_vars) if part_strat_vars else None
    cnt_sv = list(cnt_strat_vars) if cnt_strat_vars else None

    part_data = ParticipantData(df_part, id_col="id", age_col="age", strat_var_cols=part_sv)
    cnt_data = ContactData(df_cnt, id_col="id", age_col="cnt_age", strat_var_cols=cnt_sv)

    if part_strat_vars and "sex" in part_strat_vars:
        pop_data = PopulationData(df_pop, age_col="age", size_col="P", strat_var_cols=["sex"])
        df_strat = df_pop.copy()
        df_strat["Q"] = df_strat.groupby("age")["P"].transform(lambda x: x / x.sum())
        strat_data = StratificationData(
            df_strat, age_col="age", strat_var_cols=["sex"], prop_col="Q"
        )
        return ContactSurveyLoader.from_containers(part_data, cnt_data, pop_data, strat_data)
    else:
        pop_data = PopulationData(
            pd.DataFrame({"age": ages, "P": rng.integers(1000, 2000, n_ages)}),
            age_col="age",
            size_col="P",
        )
        return ContactSurveyLoader.from_containers(part_data, cnt_data, pop_data)


# ---------------------------------------------------------------------------
# Validation tests (no SVI, no .load())
# ---------------------------------------------------------------------------


def test_validate_non_subset_raises():
    """Candidate with a strat var not in reference must raise ValueError."""
    from cntmosaic.analysis import FeatureSelector, ModelConfig
    from cntmosaic.models import AgeMixFF

    ref_loader = _make_loader(part_strat_vars=["sex"])
    cand_loader = _make_loader(part_strat_vars=["educ"])

    selector = FeatureSelector(
        reference_config=ModelConfig("ref", AgeMixFF, ref_loader, {}),
        candidate_configs=[ModelConfig("cand", AgeMixFF, cand_loader, {})],
    )
    with pytest.raises(ValueError, match="educ"):
        selector._validate()


def test_validate_full_to_partial_raises():
    """FULL-to-PARTIAL demotion (drop contact side only) must raise ValueError."""
    from cntmosaic.analysis import FeatureSelector, ModelConfig
    from cntmosaic.models import AgeMixFF

    # Reference: sex is FULL (both part and cnt)
    ref_loader = _make_loader(part_strat_vars=["sex"], cnt_strat_vars=["sex"])
    # Candidate: sex only on participant side → demotion attempt
    cand_loader = _make_loader(part_strat_vars=["sex"], cnt_strat_vars=None)

    selector = FeatureSelector(
        reference_config=ModelConfig("ref", AgeMixFF, ref_loader, {}),
        candidate_configs=[ModelConfig("cand", AgeMixFF, cand_loader, {})],
    )
    with pytest.raises(ValueError, match="FULL-to-PARTIAL"):
        selector._validate()


def test_validate_duplicate_names_raises():
    """Two configs sharing a name must raise ValueError."""
    from cntmosaic.analysis import FeatureSelector, ModelConfig
    from cntmosaic.models import AgeMixFF

    loader = _make_loader()
    selector = FeatureSelector(
        reference_config=ModelConfig("same", AgeMixFF, loader, {}),
        candidate_configs=[ModelConfig("same", AgeMixFF, loader, {})],
    )
    with pytest.raises(ValueError, match="Duplicate"):
        selector._validate()


def test_validate_no_candidates_raises():
    """Empty candidate list must raise ValueError."""
    from cntmosaic.analysis import FeatureSelector, ModelConfig
    from cntmosaic.models import AgeMixFF

    loader = _make_loader()
    selector = FeatureSelector(
        reference_config=ModelConfig("ref", AgeMixFF, loader, {}),
        candidate_configs=[],
    )
    with pytest.raises(ValueError, match="candidate"):
        selector._validate()


# ---------------------------------------------------------------------------
# _build_kwargs_dict tests (requires loaded reference)
# ---------------------------------------------------------------------------


def test_build_kwargs_dict_unstratified():
    """Unstratified candidate should receive only base arrays (no flat_ix/flat_pixs)."""
    from cntmosaic.analysis import FeatureSelector, ModelConfig
    from cntmosaic.models import AgeMixFF

    ref_loader = _make_loader(part_strat_vars=["sex"])
    ref_loader.load()

    cand_loader = _make_loader()

    selector = FeatureSelector(
        reference_config=ModelConfig("ref", AgeMixFF, ref_loader, {}),
        candidate_configs=[ModelConfig("simple", AgeMixFF, cand_loader, {})],
    )

    kwargs = selector._build_kwargs_dict(
        ModelConfig("simple", AgeMixFF, cand_loader, {}),
        ref_loader,
    )
    assert "y" in kwargs
    assert "log_N" in kwargs
    assert "flat_ix" not in kwargs
    assert "flat_pixs" not in kwargs


def test_build_kwargs_dict_partial_subset():
    """
    For a PARTIAL sex-only candidate evaluated on a sex+educ reference,
    flat_ix values must be in [0, n_sex_categories - 1].
    """
    from cntmosaic.analysis import FeatureSelector, ModelConfig
    from cntmosaic.models import AgeMixFF, GenMixFF
    from cntmosaic.models.numpyro.priors import Spline2D

    ref_loader = _make_loader(part_strat_vars=["sex"])
    ref_loader.load()

    cand_loader = _make_loader(part_strat_vars=["sex"])

    selector = FeatureSelector(
        reference_config=ModelConfig("ref", GenMixFF, ref_loader, {}),
        candidate_configs=[ModelConfig("sex", GenMixFF, cand_loader, {})],
    )

    kwargs = selector._build_kwargs_dict(
        ModelConfig("sex", GenMixFF, cand_loader, {}),
        ref_loader,
    )

    assert "flat_ix" in kwargs
    assert "flat_pixs" in kwargs
    # sex has 2 categories (F, M) → indices must be 0 or 1
    assert kwargs["flat_ix"].min() >= 0
    assert kwargs["flat_ix"].max() <= 1


def test_build_kwargs_dict_shape_matches_reference():
    """kwargs_dict arrays must have the same length as the reference observation grid."""
    from cntmosaic.analysis import FeatureSelector, ModelConfig
    from cntmosaic.models import AgeMixFF, GenMixFF

    ref_loader = _make_loader(part_strat_vars=["sex"])
    ref_loader.load()
    ref_n = len(ref_loader.model_data.y)

    cand_loader = _make_loader()  # unstratified candidate

    selector = FeatureSelector(
        reference_config=ModelConfig("ref", GenMixFF, ref_loader, {}),
        candidate_configs=[ModelConfig("simple", AgeMixFF, cand_loader, {})],
    )
    kwargs = selector._build_kwargs_dict(
        ModelConfig("simple", AgeMixFF, cand_loader, {}),
        ref_loader,
    )
    assert len(kwargs["y"]) == ref_n
    assert len(kwargs["log_N"]) == ref_n


# ---------------------------------------------------------------------------
# Integration smoke test
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def synthetic_data():
    """Generate a small synthetic contact survey dataset for integration tests."""
    from cntmosaic.dataloader import (
        ContactData,
        ContactSurveyLoader,
        ParticipantData,
        PopulationData,
        StratificationData,
    )
    from cntmosaic.datasets import load_age_distribution, load_template_patterns
    from cntmosaic.sim import (
        ContactSampler,
        MatrixSampler,
        ParticipantSampler,
        Population,
        Stratification,
    )

    df_age_dist = load_age_distribution("United_States", max_age=20)
    templates = load_template_patterns("United_States", max_age=20)

    strats = [
        Stratification(
            name="sex",
            n_strata=2,
            ref_age_dist=df_age_dist["P"].values,
            labels=["M", "F"],
            seed=0,
        ),
    ]
    pc = Population(strats)
    df_pop = pc.df_P
    df_pop_prop = pc.df_Q

    pg = ParticipantSampler(pc, n_part=150)
    df_part = pg.sample(seed=0)

    mg = MatrixSampler(templates)
    cint_matrices = mg.generate_partial(pc, 5, seed=0)

    cg = ContactSampler(df_part, cint_matrices, "poisson", random_effects=False)
    df_cnt = cg.sample(seed=0)

    return df_part, df_cnt, df_pop, df_pop_prop


def test_run_returns_result(synthetic_data):
    """FeatureSelector.run() returns a FeatureSelectionResult with all models."""
    from numpyro.infer.autoguide import AutoNormal
    from numpyro.infer.initialization import init_to_value

    from cntmosaic.analysis import FeatureSelector, FeatureSelectionResult, ModelConfig
    from cntmosaic.dataloader import (
        ContactData,
        ContactSurveyLoader,
        ParticipantData,
        PopulationData,
        StratificationData,
    )
    from cntmosaic.models import AgeMixFF, GenMixFF
    from cntmosaic.models.numpyro.priors import Spline2D

    df_part, df_cnt, df_pop, df_pop_prop = synthetic_data

    loader_sex = ContactSurveyLoader.from_containers(
        ParticipantData(df_part, id_col="id", age_col="age", strat_var_cols=["sex"]),
        ContactData(df_cnt, id_col="id", age_col="cnt_age"),
        PopulationData(df_pop, age_col="age", size_col="P", strat_var_cols=["sex"]),
        StratificationData(df_pop_prop, age_col="age", strat_var_cols=["sex"], prop_col="Q"),
    )
    loader_simple = ContactSurveyLoader.from_containers(
        ParticipantData(df_part, id_col="id", age_col="age"),
        ContactData(df_cnt, id_col="id", age_col="cnt_age"),
        PopulationData(df_pop, age_col="age", size_col="P"),
    )

    def make_guide(model):
        return AutoNormal(
            model.model,
            init_loc_fn=init_to_value(values={"baseline": -float(model.log_P.mean())}),
        )

    selector = FeatureSelector(
        reference_config=ModelConfig(
            "sex",
            GenMixFF,
            loader_sex,
            priors={"rate": Spline2D("global"), "sex": Spline2D("partial")},
        ),
        candidate_configs=[
            ModelConfig(
                "no_strat",
                AgeMixFF,
                loader_simple,
                priors={"rate": Spline2D("global")},
            ),
        ],
        guide_factory=make_guide,
        num_steps=200,
        num_samples=100,
    )

    result = selector.run(PRNGKey(0))

    assert isinstance(result, FeatureSelectionResult)
    assert set(result.models.keys()) == {"sex", "no_strat"}
    assert set(result.idatas.keys()) == {"sex", "no_strat"}
    assert list(result.comparison.index) == sorted(
        result.comparison.index, key=lambda n: result.comparison.loc[n, "elpd_loo"], reverse=True
    )


def test_best_model_property(synthetic_data):
    """result.best_model returns the ContactModel with the highest ELPD-LOO."""
    from numpyro.infer.autoguide import AutoNormal
    from numpyro.infer.initialization import init_to_value

    from cntmosaic.analysis import FeatureSelector, ModelConfig
    from cntmosaic.dataloader import (
        ContactData,
        ContactSurveyLoader,
        ParticipantData,
        PopulationData,
        StratificationData,
    )
    from cntmosaic.models import AgeMixFF, GenMixFF
    from cntmosaic.models.numpyro.priors import Spline2D

    df_part, df_cnt, df_pop, df_pop_prop = synthetic_data

    loader_sex = ContactSurveyLoader.from_containers(
        ParticipantData(df_part, id_col="id", age_col="age", strat_var_cols=["sex"]),
        ContactData(df_cnt, id_col="id", age_col="cnt_age"),
        PopulationData(df_pop, age_col="age", size_col="P", strat_var_cols=["sex"]),
        StratificationData(df_pop_prop, age_col="age", strat_var_cols=["sex"], prop_col="Q"),
    )
    loader_simple = ContactSurveyLoader.from_containers(
        ParticipantData(df_part, id_col="id", age_col="age"),
        ContactData(df_cnt, id_col="id", age_col="cnt_age"),
        PopulationData(df_pop, age_col="age", size_col="P"),
    )

    def make_guide(model):
        return AutoNormal(
            model.model,
            init_loc_fn=init_to_value(values={"baseline": -float(model.log_P.mean())}),
        )

    selector = FeatureSelector(
        reference_config=ModelConfig(
            "sex",
            GenMixFF,
            loader_sex,
            priors={"rate": Spline2D("global"), "sex": Spline2D("partial")},
        ),
        candidate_configs=[
            ModelConfig(
                "no_strat",
                AgeMixFF,
                loader_simple,
                priors={"rate": Spline2D("global")},
            ),
        ],
        guide_factory=make_guide,
        num_steps=200,
        num_samples=100,
    )

    result = selector.run(PRNGKey(0))
    best_name = result.comparison.index[0]
    assert result.best_model is result.models[best_name]
