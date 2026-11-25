"""Integration tests on the frozen snapshots: D1 reconciliation, D3 graph rules, D4 constraints."""

import numpy as np
import pandas as pd
import pytest

from sustainome import config as C
from sustainome import matrix as MX
from sustainome import network as N
from sustainome import optimize as O


# ---------------------------------------------------------------------------
# D1
# ---------------------------------------------------------------------------
def test_stage_counts_reconcile(matrix_build):
    sc = matrix_build.audit["stage_counts"].set_index("stage")["count"]
    matched = sc["Exact matches (inner join on normalised CIQUAL code)"]
    assert matched + sc["CIQUAL-only codes (no AGRIBALYSE record)"] == sc["CIQUAL unique codes"]
    assert matched + sc["AGRIBALYSE-only codes (not in CIQUAL 2025)"] == sc["AGRIBALYSE unique CIQUAL codes (after duplicate resolution)"]
    assert len(matrix_build.matrix) == matched


def test_join_is_exact_and_unique(m):
    assert m["ciqual_code"].is_unique
    assert m["agb_code"].notna().all()


def test_missing_nutrients_not_zero_filled(m):
    for k in C.NUTRIENT_KEYS:
        st = m[f"{k}__status"]
        assert m.loc[st == "missing", k].isna().all()
        assert m.loc[st == "structural_zero", k].eq(0).all()
        assert (m.loc[st == "structural_zero", f"{k}__raw"].fillna("-").str.strip() == "-").all()


def test_structural_zero_only_for_b12_in_plant_groups(m):
    sz = m[m["vitamin_b12_ug__status"] == "structural_zero"]
    assert not sz.empty
    assert not sz["food_name"].str.contains("milk|cheese|egg|fish|meat|fortified", case=False).any()
    for k in C.NUTRIENT_KEYS:
        if k != "vitamin_b12_ug":
            assert not (m[f"{k}__status"] == "structural_zero").any()


def test_environment_units_harmonised(m):
    assert np.allclose(m["climate_per_100g"] * 10, m["climate_per_kg"])


def test_packaging_fix_rows_kept(m):
    # CIQUAL 26037 (smoked salmon): corrected row has 7.85 kg CO2e/kg (red uncorrected row 7.36)
    assert m.set_index("ciqual_code").loc["26037", "climate_per_kg"] == pytest.approx(7.85)


# ---------------------------------------------------------------------------
# D3
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def net(m):
    return N.run(m)


def test_graph_edges_respect_eligibility(net):
    nodes = net.nodes
    roles = nodes["roles"].fillna("").str.split("|")
    for i, j in zip(net.edges["i"], net.edges["j"]):
        same_group = nodes.at[i, "analysis_group_code"] == nodes.at[j, "analysis_group_code"]
        shared_role = bool(set(r for r in roles[i] if r) & set(r for r in roles[j] if r))
        assert same_group or shared_role


def test_graph_construction_is_leakage_free(m, net):
    """Shuffling every environmental column must not change the edge set."""
    g = np.random.default_rng(0)
    m2 = m.copy()
    env_cols = [c for c in m.columns if c.endswith("_per_kg") or c.endswith("_per_100g")] + ["dqr"]
    for c in env_cols:
        m2[c] = g.permutation(m2[c].to_numpy())
    alt = N.run(m2, with_arcs=False)
    assert set(map(tuple, alt.edges[["i", "j"]].to_numpy())) == set(map(tuple, net.edges[["i", "j"]].to_numpy()))


def test_graph_is_reproducible(m, net):
    again = N.run(m, with_arcs=False)
    assert again.tau == net.tau
    assert again.edges[["i", "j"]].equals(net.edges[["i", "j"]])


def test_edge_components_inspectable(net):
    cols = {"distance", "climate_gain", "d_land_rel", "d_water_rel", "dqr_max", "retention_status", "edge_class"}
    assert cols <= set(net.arcs.columns)


# ---------------------------------------------------------------------------
# D4
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def problem(m):
    return O.build_problem(m, MX.load_declared_tables()["baskets"], "S2")


def test_baseline_is_feasible(problem):
    assert O.check(problem, problem.x0)["satisfied"].all()


def test_optimum_satisfies_every_constraint(problem):
    r = O.solve(problem, "climate", eps_dev=0.10)
    chk = O.check(problem, r["x"])
    assert chk["satisfied"].all(), chk[~chk["satisfied"]]
    assert r["mean_deviation"] <= 0.10 + 1e-6
    assert r["climate"] <= O.summarize(problem, problem.x0)["climate"] + 1e-9


def test_epsilon_budgets_are_monotone(problem):
    tight = O.solve(problem, "climate", eps_dev=0.02)
    loose = O.solve(problem, "climate", eps_dev=0.20)
    assert loose["climate"] <= tight["climate"] + 1e-9


def test_swap_cardinality_respected(problem):
    for k in [0, 1, 2]:
        r = O.solve(problem, "climate", max_new=k)
        assert r["n_new_foods"] <= k


def test_unconstrained_environmental_baseline_shows_violations(problem):
    r = O.solve(problem, "climate", nutrition=False)
    assert r["n_violations"] > 0  # reported explicitly, not hidden


def test_random_baselines_seeded(problem):
    a = O.random_swaps(problem, 50, seed=1)
    b = O.random_swaps(problem, 50, seed=1)
    pd.testing.assert_frame_equal(a, b)
