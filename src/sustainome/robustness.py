"""Deliverable 5 - Robustness, Uncertainty and Counterexample Study.

Every function re-runs part of D1-D4 under one pre-declared alternative assumption
(config.ROBUST) and reports how much the conclusion moves. DQR is used for filtering,
stratification, penalties and a clearly-labelled perturbation stress test - never as a
probability or a standard error.
"""

from __future__ import annotations

import dataclasses
import itertools

import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score

from . import config as C
from . import matrix as MX
from . import network as N
from . import optimize as O
from . import pareto
from . import stats as st
from . import tradeoff as T


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def edge_codes(res: N.NetworkResult, edges: pd.DataFrame | None = None) -> set[frozenset]:
    e = res.edges if edges is None else edges
    codes = res.nodes["ciqual_code"].to_numpy()
    return {frozenset((codes[i], codes[j])) for i, j in zip(e["i"], e["j"])}


def strong_codes(A: pd.DataFrame) -> set[tuple]:
    s = A[A["edge_class"] == "strong"]
    return set(zip(s["src_code"], s["tgt_code"]))


def rearc(ref: N.NetworkResult, pairs: pd.DataFrame, edges: pd.DataFrame, tau: float) -> pd.DataFrame:
    """Reference arcs with distances / graph membership from an alternative graph."""
    dist = {(min(i, j), max(i, j)): d for i, j, d in zip(pairs["i"], pairs["j"], pairs["distance"])}
    A = ref.arcs.copy()
    A["distance"] = [dist.get((min(s, t), max(s, t)), np.nan) for s, t in zip(A["src"], A["tgt"])]
    A = A.dropna(subset=["distance"])
    return N.reclassify(A, tau, edges)


def naive_equivalent_share(A: pd.DataFrame) -> float:
    rc = N.recommender_comparison(A)
    return float(rc["naive_equivalent"].mean()) if len(rc) else np.nan


# ---------------------------------------------------------------------------
# D3 robustness: nutrient weights, feature subsets, k / threshold grid, normalisation
# ---------------------------------------------------------------------------
def graph_weight_sensitivity(ref: N.NetworkResult, n_draws: int, seed: int) -> pd.DataFrame:
    g = np.random.default_rng(seed)
    ref_edges, ref_strong = edge_codes(ref), strong_codes(ref.arcs)
    rows = []
    for d in range(n_draws):
        w = g.dirichlet(np.ones(len(C.NUTRIENT_KEYS)))
        spec = N.GraphSpec(weights=w)
        pairs = N.candidate_pairs(ref.nodes, spec)
        edges, tau = N.knn_graph(ref.nodes, pairs, spec)
        A = rearc(ref, pairs, edges, tau)
        s = strong_codes(A)
        rows.append({"draw": d, "tau": tau, "edges": len(edges),
                     "edge_jaccard": st.jaccard(ref_edges, edge_codes(ref, edges)),
                     "strong": len(s), "strong_retained": len(s & ref_strong) / max(1, len(ref_strong)),
                     "naive_equivalent_share": naive_equivalent_share(A),
                     **{f"w_{k}": wk for k, wk in zip(C.NUTRIENT_KEYS, w)}})
    return pd.DataFrame(rows)


def graph_feature_subsets(m: pd.DataFrame, ref: N.NetworkResult) -> pd.DataFrame:
    ref_edges, ref_strong = edge_codes(ref), strong_codes(ref.arcs)
    subsets = {f"drop {k}": [x for x in C.NUTRIENT_KEYS if x != k] for k in C.NUTRIENT_KEYS}
    subsets["macronutrients + limits only"] = [k for k in C.NUTRIENT_KEYS if k not in C.MICRO_KEYS]
    subsets["micronutrients + protein only"] = ["protein_g"] + C.MICRO_KEYS
    subsets["extended (+ vit D, B6, folate, retinol, P)"] = C.NUTRIENT_KEYS + [k for k in C.EXTENDED_KEYS if k not in ("water_g", "alcohol_g")]
    rows = []
    for name, keys in subsets.items():
        spec = N.GraphSpec(keys=keys, max_missing=3 if len(keys) > 8 else 1)
        pairs = N.candidate_pairs(ref.nodes, spec)
        edges, tau = N.knn_graph(ref.nodes, pairs, spec)
        A = rearc(ref, pairs, edges, tau)
        s = strong_codes(A)
        rows.append({"subset": name, "n_features": len(keys), "edges": len(edges),
                     "edge_jaccard": st.jaccard(ref_edges, edge_codes(ref, edges)),
                     "strong": len(s), "strong_retained": len(s & ref_strong) / max(1, len(ref_strong)),
                     "naive_equivalent_share": naive_equivalent_share(A)})
    return pd.DataFrame(rows)


def graph_grid(ref: N.NetworkResult, ref_comm: dict, seed: int = C.SEED) -> pd.DataFrame:
    ref_edges, ref_strong = edge_codes(ref), strong_codes(ref.arcs)
    rows = []
    for k, q in itertools.product(C.ROBUST["knn_grid"], C.ROBUST["quantile_grid"]):
        spec = N.GraphSpec(k=k, quantile=q)
        edges, tau = N.knn_graph(ref.nodes, ref.pairs, spec)
        G = N.build_graph(ref.nodes, edges, tau)
        comm = N.communities(G, seed=seed)
        met = N.network_metrics(G, comm)
        common = [n for n in G if G.degree(n) > 0 and ref.G.degree(n) > 0]
        ari = adjusted_rand_score([ref_comm[n] for n in common], [comm[n] for n in common]) if common else np.nan
        A = N.reclassify(ref.arcs, tau, edges)
        s = strong_codes(A)
        rows.append({"k": k, "quantile": q, "tau": tau, "edges": len(edges), "coverage": met["non_isolated_share"],
                     "communities": met["communities"], "modularity": met["modularity"],
                     "nmi_vs_group": met["nmi_vs_analysis_group"], "community_ari_vs_reference": ari,
                     "edge_jaccard": st.jaccard(ref_edges, edge_codes(ref, edges)),
                     "strong": len(s), "strong_jaccard": st.jaccard(ref_strong, s),
                     "naive_equivalent_share": naive_equivalent_share(A)})
    return pd.DataFrame(rows)


def energy_view_graph(m: pd.DataFrame, ref: N.NetworkResult) -> dict:
    """Same graph built on per-100 kcal profiles (calorie-for-calorie substitution)."""
    alt = N.run(m, N.GraphSpec(view="energy"))
    ref_edges = edge_codes(ref)
    return {"edges": len(alt.edges), "tau": alt.tau, "edge_jaccard": st.jaccard(ref_edges, edge_codes(alt)),
            "strong": int((alt.arcs["edge_class"] == "strong").sum()),
            "strong_jaccard": st.jaccard(strong_codes(ref.arcs), strong_codes(alt.arcs)),
            "naive_equivalent_share": naive_equivalent_share(alt.arcs), "result": alt}


def missing_value_edges(m: pd.DataFrame, ref: N.NetworkResult) -> pd.DataFrame:
    """Graph edges computed on < 15 shared nutrients that would vanish if the missing
    values were set to their analysis-group median (an edge 'driven by missing values')."""
    e = ref.edges[ref.edges["shared_features"] < len(C.NUTRIENT_KEYS)].copy()
    if e.empty:
        return e
    nodes = ref.nodes.copy()
    filled = nodes.copy()
    for k in C.NUTRIENT_KEYS:
        med = nodes.groupby("analysis_group_code")[k].transform("median")
        filled[k] = nodes[k].fillna(med).fillna(nodes[k].median())
    spec = N.GraphSpec()
    Zf = N.profile_matrix(filled, spec, scaler=N.nutrition.ProfileScaler().fit(nodes))
    w = N.nutrition.equal_weights()
    d_imp = np.array([N.nutrition.pairwise_distance(Zf[[i]], Zf[[j]], w, 1)[0][0, 0] for i, j in zip(e["i"], e["j"])])
    e["distance_imputed"] = d_imp
    e["vanishes_when_imputed"] = d_imp > ref.tau
    e["src_name"] = nodes.loc[e["i"], "food_name"].to_numpy()
    e["tgt_name"] = nodes.loc[e["j"], "food_name"].to_numpy()
    e["missing_features"] = [
        "|".join(k for k in C.NUTRIENT_KEYS if pd.isna(nodes.at[i, k]) or pd.isna(nodes.at[j, k])) for i, j in zip(e["i"], e["j"])]
    return e


# ---------------------------------------------------------------------------
# D2 robustness: NR weights, DQR filtering, indicator subsets, reversals, bootstrap
# ---------------------------------------------------------------------------
def nr_weight_pareto_stability(vl: pd.DataFrame, n_draws: int, seed: int) -> pd.DataFrame:
    """Frequency with which each food is efficient when NR9 components get random weights."""
    g = np.random.default_rng(seed)
    pct = pd.DataFrame({k: (100.0 * vl[k] / C.NUTRIENT_REF[k]).clip(upper=100) for k in C.NR_KEYS})
    counts = np.zeros(len(vl))
    defined = np.zeros(len(vl))
    for _ in range(n_draws):
        w = g.dirichlet(np.ones(len(C.NR_KEYS))) * len(C.NR_KEYS)
        alt = vl.copy()
        alt["nr9"] = (pct * w).sum(axis=1, min_count=len(C.NR_KEYS)).where(pct.notna().all(axis=1))
        lab = T.pareto_labels(alt, two_d=None, depth=False)
        ok = lab["analysed"].to_numpy()
        counts[ok] += lab["efficient"].fillna(0).to_numpy()[ok]
        defined[ok] += 1
    out = vl[["ciqual_code", "food_name", "view", "analysis_group_code", "analysis_group"]].copy()
    out["freq_efficient"] = np.where(defined > 0, counts / np.maximum(defined, 1), np.nan)
    out["stable"] = (out["freq_efficient"] >= 0.9) | (out["freq_efficient"] <= 0.1)
    return out[defined > 0].reset_index(drop=True)


def pareto_under(vl: pd.DataFrame, ref_lab: pd.DataFrame, objectives: dict | None = None,
                 keep: pd.Series | None = None) -> dict:
    """Agreement of efficient sets with the reference under an alternative definition."""
    d = vl if keep is None else vl[keep.to_numpy()]
    lab = T.pareto_labels(d, objectives=objectives or C.PARETO_OBJECTIVES, two_d=None, depth=False)
    a = lab[lab["analysed"]]
    r = ref_lab.loc[a.index]
    both = r["analysed"].to_numpy()
    e_alt = set(a.index[(a["efficient"] == 1).to_numpy() & both])
    e_ref = set(r.index[(r["efficient"] == 1).to_numpy() & both])
    return {"n_analysed": int(len(a)), "jaccard_vs_reference": st.jaccard(e_alt, e_ref),
            "status_agreement": float((a.loc[both, "efficient"].to_numpy() == r.loc[both, "efficient"].to_numpy()).mean()) if both.any() else np.nan}


def indicator_subset_pareto(vl: pd.DataFrame, ref_lab: pd.DataFrame) -> pd.DataFrame:
    sets = {
        "climate + NR9 + LIM3": {"climate": 1, "nr9": -1, "lim3": 1},
        "climate, land, water + NR9 + LIM3 (reference)": C.PARETO_OBJECTIVES,
        "all 7 indicators + NR9 + LIM3": {**{k: 1 for k in C.ENV_CORE}, "nr9": -1, "lim3": 1},
        "EF single score + NR9 + LIM3": {"ef_score": 1, "nr9": -1, "lim3": 1},
    }
    return pd.DataFrame([{"objective_set": k, **pareto_under(vl, ref_lab, objectives=v)} for k, v in sets.items()])


def dqr_filter_pareto(vl: pd.DataFrame, ref_lab: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for c in C.ROBUST["dqr_cutoffs"]:
        keep = vl["dqr"] <= c
        rows.append({"dqr_max": c, "foods_kept_share": float(keep.mean()), **pareto_under(vl, ref_lab, keep=keep)})
    return pd.DataFrame(rows)


def view_reversals(vl: pd.DataFrame, min_size: int = C.MIN_GROUP_SIZE) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Pairwise climate-ordering reversals between functional units, within groups."""
    piv = vl.pivot_table(index=["analysis_group_code", "ciqual_code"], columns="view", values="climate", aggfunc="first")
    names = vl.drop_duplicates("ciqual_code").set_index("ciqual_code")["food_name"]
    rows, examples = [], []
    for grp, g in piv.groupby(level=0):
        g = g.droplevel(0)
        rec = {"analysis_group_code": grp}
        for v1, v2 in [("mass", "energy"), ("mass", "protein"), ("energy", "protein")]:
            h = g[[v1, v2]].dropna()
            if len(h) < min_size:
                continue
            a1, a2 = h[v1].to_numpy(), h[v2].to_numpy()
            i, j = np.triu_indices(len(h), 1)
            s1, s2 = np.sign(a1[i] - a1[j]), np.sign(a2[i] - a2[j])
            valid = (s1 != 0) & (s2 != 0)
            flip = valid & (s1 != s2)
            rec[f"reversal_{v1}_{v2}"] = float(flip.sum() / max(1, valid.sum()))
            rec[f"pairs_{v1}_{v2}"] = int(valid.sum())
            if flip.any():
                mag = np.minimum(np.abs(np.log(a1[i] / a1[j])), np.abs(np.log(a2[i] / a2[j])))
                mag = np.where(flip, mag, -1)
                for t in np.argsort(mag)[::-1][:3]:
                    if mag[t] <= 0:
                        break
                    ci, cj = h.index[i[t]], h.index[j[t]]
                    examples.append({"analysis_group_code": grp, "views": f"{v1} vs {v2}", "food_a": names[ci], "food_b": names[cj],
                                     f"a_over_b_{v1}": a1[i[t]] / a1[j[t]], f"a_over_b_{v2}": a2[i[t]] / a2[j[t]],
                                     "min_log_ratio": mag[t]})
        rows.append(rec)
    ex = pd.DataFrame(examples).sort_values("min_log_ratio", ascending=False) if examples else pd.DataFrame()
    return pd.DataFrame(rows), ex


def bootstrap_pareto_probability(vl: pd.DataFrame, ref_lab: pd.DataFrame, n_boot: int, seed: int) -> pd.DataFrame:
    """P(efficient | food sampled) when the comparison set is resampled with replacement.

    Note: resampling can only remove competitors, so a reference-efficient food is efficient in
    every resample by construction. The informative quantity is the fragility of *dominated*
    labels: a dominated food that becomes efficient whenever its few dominators are not drawn.
    ``robustly_dominated`` = reference-dominated and efficient in <= 10 % of resamples.
    """
    g = np.random.default_rng(seed)
    d = vl.join(ref_lab[["group", "analysed", "efficient"]])
    d = d[d["analysed"]]
    obj = list(C.PARETO_OBJECTIVES)
    sense = list(C.PARETO_OBJECTIVES.values())
    hits = pd.Series(0.0, index=d.index)
    seen = pd.Series(0.0, index=d.index)
    for _, grp in d.groupby(["view", "group"]):
        X = grp[obj].to_numpy(float)
        idx = grp.index.to_numpy()
        for _ in range(n_boot):
            u = np.unique(g.integers(0, len(grp), len(grp)))
            eff = pareto.pareto_mask(X[u], sense)
            seen.loc[idx[u]] += 1
            hits.loc[idx[u]] += eff
    out = d[["ciqual_code", "food_name", "view", "group", "analysis_group", "efficient"]].copy()
    out["p_efficient"] = hits / seen.replace(0, np.nan)
    out["robustly_dominated"] = (out["efficient"] == 0) & (out["p_efficient"] <= 0.1)
    return out.reset_index(drop=True)


def bootstrap_category_ratio(vl: pd.DataFrame, n_boot: int, seed: int, reference: str = "legumes") -> pd.DataFrame:
    """Bootstrap CI for median(category climate) / median(legume climate), per view."""
    rl = T.role_long(vl)
    rl = rl[(rl["role"] == "protein_main") & rl["protein_category"].notna() & rl["fu_defined"]]
    rows = []
    for v, gv in rl.groupby("view"):
        ref = gv.loc[gv["protein_category"] == reference, "climate"].to_numpy()
        for i, (cat, gc) in enumerate(gv.groupby("protein_category")):
            if cat == reference:
                continue
            x = gc["climate"].to_numpy()
            pt, lo, hi = st.bootstrap_ci(lambda a, b: np.median(a) / np.median(b), x, ref, n_boot=n_boot, seed=seed + i)
            rows.append({"view": v, "category": cat, "n": len(x), "ratio": pt, "ci_lo": lo, "ci_hi": hi})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# D1 assumption variants (parser policy, duplicate rule, structural zero)
# ---------------------------------------------------------------------------
def matrix_variants(m_ref: pd.DataFrame, ref_net: N.NetworkResult, ref_vl: pd.DataFrame, ref_lab: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    variants = {
        "parser: '< x' -> 0": dict(parse_policy="zero"),
        "parser: '< x' -> x (upper bound)": dict(parse_policy="loq"),
        "duplicates: mean of variants": dict(dup_rule="mean"),
        "duplicates: max-impact variant": dict(dup_rule="max"),
        "duplicates: min-impact variant": dict(dup_rule="min"),
        "B12 structural-zero rule OFF": dict(structural_zero=False),
    }
    ref_edges = edge_codes(ref_net)
    ref_eff = set(zip(ref_lab.loc[ref_lab["efficient"] == 1, "ciqual_code"], ref_lab.loc[ref_lab["efficient"] == 1, "view"]))
    rows, mats = [], {}
    for name, kw in variants.items():
        mv = MX.build_matrix(**kw).matrix
        mats[name] = mv
        vl = T.view_long(mv)
        lab = T.pareto_labels(vl)
        eff = set(zip(lab.loc[lab["efficient"] == 1, "ciqual_code"], lab.loc[lab["efficient"] == 1, "view"]))
        net = N.run(mv, with_arcs=True)
        changed = mv.set_index("ciqual_code")["climate_per_kg"].sub(m_ref.set_index("ciqual_code")["climate_per_kg"]).abs().gt(1e-12).sum()
        nr_ref = ref_vl[ref_vl["view"] == "energy"].set_index("ciqual_code")["nr9"]
        nr_alt = vl[vl["view"] == "energy"].set_index("ciqual_code")["nr9"]
        common = nr_ref.dropna().index.intersection(nr_alt.dropna().index)
        rows.append({"variant": name, "foods_with_changed_climate": int(changed),
                     "graph_nodes": len(net.nodes), "profile_complete": int(mv["profile_complete"].sum()),
                     "nr9_spearman_vs_ref": st.spearman(nr_ref[common], nr_alt[common]),
                     "efficient_jaccard_vs_ref": st.jaccard(ref_eff, eff),
                     "edge_jaccard_vs_ref": st.jaccard(ref_edges, edge_codes(net)),
                     "strong_jaccard_vs_ref": st.jaccard(strong_codes(ref_net.arcs), strong_codes(net.arcs)),
                     "naive_equivalent_share": naive_equivalent_share(net.arcs)})
    return pd.DataFrame(rows), mats


# ---------------------------------------------------------------------------
# D4 robustness
# ---------------------------------------------------------------------------
def optimizer_tolerance_grid(m: pd.DataFrame, baskets: pd.DataFrame, scenarios: list[str]) -> pd.DataFrame:
    rows = []
    for sc in scenarios:
        for et, mm, pm in itertools.product(C.ROBUST["energy_tol_grid"], C.ROBUST["micro_min_grid"], C.ROBUST["protein_min_grid"]):
            P = O.build_problem(m, baskets, sc, {"energy_tol": et, "micro_min": mm, "protein_min": pm})
            r = O.solve(P, "climate")
            rows.append({"scenario": sc, "energy_tol": et, "micro_min": mm, "protein_min": pm,
                         "feasible": r is not None and r["feasible"],
                         "climate_change": np.nan if r is None else r["climate_change"],
                         "water_change": np.nan if r is None else r["water_change"],
                         "mean_deviation": np.nan if r is None else r["mean_deviation"]})
    return pd.DataFrame(rows)


def optimizer_data_variants(m: pd.DataFrame, mats: dict, baskets: pd.DataFrame, scenarios: list[str]) -> pd.DataFrame:
    rows = []
    cases = [("reference", m, {}), ("DQR <= 3.0 candidates", m, {"max_dqr": 3.0}),
             ("DQR <= 2.5 candidates", m, {"max_dqr": 2.5}),
             ("include egg/quinoa-flagged foods", m, {"exclude_issues": False})]
    cases += [(name, mv, {}) for name, mv in mats.items()]
    for sc in scenarios:
        for name, mm, kw in cases:
            try:
                P = O.build_problem(mm, baskets, sc, **kw)
                r = O.solve(P, "climate")
                r10 = O.solve(P, "climate", eps_dev=0.10)
            except ValueError as exc:  # baseline food unavailable under this variant
                rows.append({"scenario": sc, "variant": name, "note": str(exc)})
                continue
            rows.append({"scenario": sc, "variant": name, "n_candidates": P.n,
                         "climate_change_opt": np.nan if r is None else r["climate_change"],
                         "climate_change_dev10": np.nan if r10 is None else r10["climate_change"],
                         "water_change_opt": np.nan if r is None else r["water_change"], "note": ""})
    return pd.DataFrame(rows)


def impact_perturbation(m: pd.DataFrame, baskets: pd.DataFrame, scenario: str, n: int, a: float, seed: int,
                        eps_dev: float | None = 0.10) -> pd.DataFrame:
    """Stress test, NOT a confidence interval: each food's climate impact is multiplied by
    U(1 - a*DQR, 1 + a*DQR) - a worse-rated food is allowed a wider error band - and the
    optimum is re-solved. Reports how much the headline reduction and the chosen foods move."""
    g = np.random.default_rng(seed)
    P = O.build_problem(m, baskets, scenario)
    ref = O.solve(P, "climate", eps_dev=eps_dev)
    ref_set = set(P.var.loc[ref["x"] > 0.5, "ciqual_code"])
    ci = C.ENV_KEYS.index("climate")
    rows = []
    for t in range(n):
        dqr = P.var["dqr"].to_numpy()
        f = g.uniform(1 - a * dqr, 1 + a * dqr)
        A = P.A_env.copy()
        A[:, ci] = A[:, ci] * f
        Pp = dataclasses.replace(P, A_env=A, params=dict(P.params))
        r = O.solve(Pp, "climate", eps_dev=eps_dev)
        if r is None:
            continue
        chosen = set(P.var.loc[r["x"] > 0.5, "ciqual_code"])
        true_change = float(P.A_env[:, ci] @ r["x"] / (P.A_env[:, ci] @ P.x0) - 1)
        rows.append({"scenario": scenario, "draw": t, "perturbed_world_change": r["climate_change"],
                     "reference_world_change": true_change, "regret_vs_reference_opt": true_change - ref["climate_change"],
                     "food_set_jaccard": st.jaccard(ref_set, chosen)})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# claims register
# ---------------------------------------------------------------------------
def label_claim(tests: dict[str, bool]) -> str:
    vals = list(tests.values())
    if all(vals):
        return "robust"
    if sum(vals) >= max(1, len(vals) - 1):
        return "conditional"
    return "exploratory / withdrawn"
