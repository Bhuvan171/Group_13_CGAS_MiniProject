import numpy as np
import pandas as pd
import pytest

from sustainome import config as C
from sustainome import normalize, nutrition, pareto
from sustainome import stats as st


# ---------------------------------------------------------------------------
# normalisation arithmetic and denominator guards
# ---------------------------------------------------------------------------
def _foods():
    d = {k: [1.0, 1.0, 1.0, np.nan] for k in C.NUTRIENT_KEYS}
    d["energy_kcal"] = [200.0, 5.0, 100.0, np.nan]
    d["protein_g"] = [20.0, 0.5, 1.0, 10.0]
    df = pd.DataFrame(d)
    df["ciqual_code"] = ["a", "b", "c", "d"]
    for k in C.ENV_KEYS:
        df[f"{k}_per_100g"] = [2.0, 2.0, 2.0, 2.0]
    return df


def test_mass_view_is_identity():
    fu = normalize.to_view(_foods(), "mass")
    assert fu["climate"].tolist() == [2.0] * 4
    assert fu["fu_defined"].all()


def test_energy_view_arithmetic_and_guard():
    fu = normalize.to_view(_foods(), "energy", energy_floor=10)
    # 2.0 per 100 g at 200 kcal/100 g -> 1.0 per 100 kcal
    assert fu.loc[0, "climate"] == pytest.approx(1.0)
    assert fu.loc[0, "protein_g"] == pytest.approx(10.0)
    assert not fu.loc[1, "fu_defined"] and np.isnan(fu.loc[1, "climate"])  # 5 kcal < floor
    assert "energy <" in fu.loc[1, "fu_undefined_reason"]
    assert fu.loc[3, "fu_undefined_reason"] == "energy missing"


def test_protein_view_arithmetic_and_guard():
    fu = normalize.to_view(_foods(), "protein", protein_floor=2)
    # 2.0 per 100 g at 20 g protein -> 1.0 per 10 g protein
    assert fu.loc[0, "climate"] == pytest.approx(1.0)
    assert fu.loc[0, "protein_g"] == pytest.approx(10.0)
    assert not fu.loc[1, "fu_defined"] and not fu.loc[2, "fu_defined"]
    assert fu.loc[3, "fu_defined"]


def test_relative_change_floor_handles_negative_and_tiny_baselines():
    r = normalize.relative_change([1.0, -0.001, 0.0], [0.5, 0.001, 0.2], floor=0.1)
    assert r[0] == pytest.approx(-0.5)
    assert r[1] == pytest.approx(0.002 / 0.1)
    assert np.isfinite(r).all()


# ---------------------------------------------------------------------------
# nutrition indices
# ---------------------------------------------------------------------------
def test_nr9_caps_each_component_at_100():
    fu = pd.DataFrame({k: [C.NUTRIENT_REF[k] * 5] for k in C.NR_KEYS})
    assert nutrition.nr9(fu).iloc[0] == pytest.approx(900.0)


def test_nr9_requires_complete_components_by_default():
    fu = pd.DataFrame({k: [C.NUTRIENT_REF[k] * 0.1] for k in C.NR_KEYS})
    fu.loc[0, "zinc_mg"] = np.nan
    assert np.isnan(nutrition.nr9(fu).iloc[0])
    assert nutrition.nr9(fu, min_available=7).iloc[0] == pytest.approx(90.0)


def test_lim3_is_uncapped_and_separate():
    fu = pd.DataFrame({k: [C.NUTRIENT_REF[k] * 2] for k in C.LIM_KEYS})
    assert nutrition.lim3(fu).iloc[0] == pytest.approx(600.0)


def test_pairwise_distance_ignores_missing_and_enforces_min_shared():
    Za = np.array([[0.0, 0.0, np.nan], [0.0, np.nan, np.nan]])
    Zb = np.array([[3.0, 4.0, 100.0]])
    D, S = nutrition.pairwise_distance(Za, Zb, np.ones(3), min_shared=2)
    assert D[0, 0] == pytest.approx(np.sqrt((9 + 16) / 2))
    assert S[1, 0] == 1 and np.isnan(D[1, 0])


# ---------------------------------------------------------------------------
# Pareto dominance on hand-constructed examples
# ---------------------------------------------------------------------------
def test_dominates_mixed_senses():
    # minimise climate, maximise nutrition
    assert pareto.dominates([1, 10], [2, 5], [1, -1])
    assert not pareto.dominates([1, 5], [2, 10], [1, -1])
    assert not pareto.dominates([1, 5], [1, 5], [1, -1])  # equal points do not dominate


def test_pareto_mask_and_depth():
    X = np.array([[1, 1], [2, 2], [1, 3], [3, 1], [3, 3], [np.nan, 0]])
    mask = pareto.pareto_mask(X, [1, 1])
    assert mask.tolist() == [True, False, False, False, False, False]
    depth = pareto.nondominated_rank(X, [1, 1])
    assert depth[:5].tolist() == [1, 2, 2, 2, 3]
    assert np.isnan(depth[5])
    assert pareto.domination_count(X, [1, 1])[4] == 4


def test_pareto_tradeoff_front():
    X = np.array([[1, 5], [2, 4], [3, 3], [4, 4]])  # minimise both
    assert pareto.pareto_mask(X, [1, 1]).tolist() == [True, True, True, False]


# ---------------------------------------------------------------------------
# statistics
# ---------------------------------------------------------------------------
def test_cliffs_delta_extremes():
    assert st.cliffs_delta([5, 6, 7], [1, 2, 3]) == 1.0
    assert st.cliffs_delta([1, 2, 3], [5, 6, 7]) == -1.0
    assert st.cliffs_delta([1, 2], [1, 2]) == 0.0


def test_seeded_bootstrap_is_repeatable():
    x, y = np.arange(30.0), np.arange(30.0) ** 1.5 + np.sin(np.arange(30))
    a = st.spearman_ci(x, y, n_boot=200, seed=7)
    b = st.spearman_ci(x, y, n_boot=200, seed=7)
    assert a == b
    assert st.bootstrap_ci(st.cliffs_delta, x, y, n_boot=100, seed=3) == st.bootstrap_ci(st.cliffs_delta, x, y, n_boot=100, seed=3)
