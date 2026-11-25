"""Deliverable 3 - Nutritional-Equivalence Substitution Network.

Graph construction uses *only* nutrient profiles and declared eligibility (same analysis
group or shared basket role). Environmental indicators and DQR are attached afterwards as
edge attributes, so the graph cannot leak the outcome it is used to evaluate.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import networkx as nx
import numpy as np
import pandas as pd

from . import config as C
from . import normalize, nutrition


@dataclass
class GraphSpec:
    k: int = C.KNN_K
    quantile: float = C.DIST_QUANTILE
    weights: np.ndarray | None = None
    keys: list[str] = field(default_factory=lambda: list(C.NUTRIENT_KEYS))
    max_missing: int = len(C.NUTRIENT_KEYS) - C.MIN_SHARED_FEATURES  # at most 3 features may be unobserved
    view: str = "mass"  # nutrient profile basis: 'mass' (per 100 g) or 'energy' (per 100 kcal)


# ---------------------------------------------------------------------------
# eligibility and distances
# ---------------------------------------------------------------------------
def eligibility_blocks(nodes: pd.DataFrame) -> dict[str, np.ndarray]:
    """Blocks of mutually eligible nodes: each analysis group and each declared role."""
    blocks = {f"group:{g}": idx.to_numpy() for g, idx in nodes.groupby("analysis_group_code").groups.items()}
    roles = nodes["roles"].fillna("").str.split("|")
    for role in sorted({r for rs in roles for r in rs if r}):
        idx = np.where(roles.map(lambda rs: role in rs).to_numpy())[0]
        blocks[f"role:{role}"] = nodes.index.to_numpy()[idx]
    return blocks


def profile_matrix(nodes: pd.DataFrame, spec: GraphSpec, scaler: nutrition.ProfileScaler | None = None) -> np.ndarray:
    X = nodes[spec.keys].copy()
    if spec.view == "energy":
        f, _ = normalize.fu_factor(nodes, "energy")
        X = X.mul(f, axis=0)
        X = X.drop(columns=[c for c in ["energy_kcal"] if c in X])
    sc = scaler or nutrition.ProfileScaler(list(X.columns)).fit(X)
    return sc.transform(X)


def candidate_pairs(nodes: pd.DataFrame, spec: GraphSpec) -> pd.DataFrame:
    """All eligible unordered pairs with their nutrient distance (NaN if < min_shared)."""
    Z = profile_matrix(nodes, spec)
    keys = spec.keys if spec.view == "mass" else [k for k in spec.keys if k != "energy_kcal"]
    w = spec.weights if spec.weights is not None else nutrition.equal_weights(keys)
    if spec.view == "energy" and spec.weights is not None and len(spec.weights) == len(spec.keys):
        w = np.delete(spec.weights, spec.keys.index("energy_kcal"))
    min_shared = max(1, len(keys) - spec.max_missing)
    pos = {ix: p for p, ix in enumerate(nodes.index)}
    seen: dict[tuple[int, int], list] = {}
    for block, idx in eligibility_blocks(nodes).items():
        if len(idx) < 2:
            continue
        p = np.array([pos[i] for i in idx])
        D, S = nutrition.pairwise_distance(Z[p], Z[p], w, min_shared)
        iu, ju = np.triu_indices(len(p), 1)
        for a, b, d, s in zip(p[iu], p[ju], D[iu, ju], S[iu, ju]):
            key = (a, b) if a < b else (b, a)
            if key in seen:
                seen[key][3].add(block)
            else:
                seen[key] = [d, s, block.startswith("group:"), {block}]
    rows = [(a, b, v[0], v[1], v[2], "|".join(sorted(v[3]))) for (a, b), v in seen.items()]
    pairs = pd.DataFrame(rows, columns=["i", "j", "distance", "shared_features", "same_group", "blocks"])
    pairs["same_group"] = pairs["blocks"].str.contains("group:")
    return pairs


def knn_graph(nodes: pd.DataFrame, pairs: pd.DataFrame, spec: GraphSpec) -> tuple[pd.DataFrame, float]:
    """Mutual k-NN edges with distance <= tau, tau = quantile of 1-NN distances."""
    p = pairs.dropna(subset=["distance"])
    both = pd.concat([p[["i", "j", "distance"]], p.rename(columns={"i": "j", "j": "i"})[["i", "j", "distance"]]])
    both = both.sort_values(["i", "distance", "j"])
    nn1 = both.groupby("i")["distance"].min()
    tau = float(np.quantile(nn1, spec.quantile)) if spec.quantile < 1 else float(np.inf)
    both["rank"] = both.groupby("i").cumcount() + 1
    knn = set(map(tuple, both.loc[both["rank"] <= spec.k, ["i", "j"]].to_numpy()))
    mutual = np.array([(a, b) in knn and (b, a) in knn for a, b in zip(p["i"], p["j"])], dtype=bool)
    edges = p[mutual & (p["distance"] <= tau)].copy()
    return edges.reset_index(drop=True), tau


def build_graph(nodes: pd.DataFrame, edges: pd.DataFrame, tau: float) -> nx.Graph:
    G = nx.Graph()
    for pos, (ix, r) in enumerate(nodes.iterrows()):
        G.add_node(pos, ciqual_code=r["ciqual_code"], name=r["food_name"], group=r["analysis_group"],
                   group_code=r["analysis_group_code"], ciqual_group=r["ciqual_group"],
                   climate=r["climate_per_100g"], dqr=r["dqr"])
    scale = tau if np.isfinite(tau) else edges["distance"].max()
    for r in edges.itertuples():
        G.add_edge(int(r.i), int(r.j), distance=float(r.distance), weight=float(max(1e-6, 1 - r.distance / (scale * 1.0001))))
    return G


# ---------------------------------------------------------------------------
# directed substitution arcs and their components
# ---------------------------------------------------------------------------
def retention(src: pd.DataFrame, tgt: pd.DataFrame) -> pd.DataFrame:
    """Nutrient-retention checks for arcs src -> tgt (gram-for-gram, per 100 g).

    * protein: target keeps >= 80 % of source protein (if source >= 2 g/100 g);
    * energy: target/source energy ratio within [0.67, 1.5] (if source >= 10 kcal/100 g);
    * micronutrients: for each micronutrient the source supplies meaningfully (>= 5 % of
      the reference intake per 100 g) the target keeps >= 50 %;
    * limit nutrients: sodium / saturated fat / sugars may not exceed 1.25 x source plus a
      small absolute tolerance.
    A check whose inputs are missing is 'unknown' - never silently passed.
    """
    n = len(src)
    out = pd.DataFrame(index=src.index)
    ps, pt = src["protein_g"].to_numpy(float), tgt["protein_g"].to_numpy(float)
    es, et = src["energy_kcal"].to_numpy(float), tgt["energy_kcal"].to_numpy(float)
    with np.errstate(divide="ignore", invalid="ignore"):
        pr = np.where(ps >= C.PROTEIN_FLOOR_G, pt / ps, np.nan)
        er = np.where(es >= C.ENERGY_FLOOR_KCAL, et / es, np.nan)
    out["protein_ratio"], out["energy_ratio"] = pr, er
    fail_cols: dict[str, np.ndarray] = {
        "protein": np.isfinite(pr) & (pr < C.PROTEIN_RETENTION),
        "energy": np.isfinite(er) & ((er < C.ENERGY_RATIO[0]) | (er > C.ENERGY_RATIO[1])),
    }
    unk_cols: dict[str, np.ndarray] = {}
    min_ret = np.full(n, np.nan)
    for k in C.MICRO_KEYS:
        s, t = src[k].to_numpy(float), tgt[k].to_numpy(float)
        meaningful = s >= C.MICRO_SOURCE_DV * C.NUTRIENT_REF[k]
        with np.errstate(divide="ignore", invalid="ignore"):
            r = np.where(meaningful, t / s, np.nan)
        min_ret = np.fmin(min_ret, r)
        fail_cols[k] = meaningful & np.isfinite(t) & (r < C.MICRO_RETENTION)
        unk_cols[k] = (meaningful & np.isnan(t)) | np.isnan(s)
    for k, (fac, tol) in C.LIMIT_TOL.items():
        s, t = src[k].to_numpy(float), tgt[k].to_numpy(float)
        miss = np.isnan(s) | np.isnan(t)
        fail_cols[k] = ~miss & (t > fac * s + tol)
        unk_cols[k] = miss
    out["min_micro_retention"] = min_ret
    def joined(cols: dict[str, np.ndarray]) -> np.ndarray:
        s = np.full(n, "", dtype=object)
        for name, mask in cols.items():
            s = np.where(mask, s + name + "|", s)
        return np.array([x.rstrip("|") for x in s], dtype=object)

    out["retention_fails"] = joined(fail_cols)
    out["retention_unknown"] = joined(unk_cols)
    any_fail = np.any(np.column_stack(list(fail_cols.values())), axis=1)
    any_unk = np.any(np.column_stack(list(unk_cols.values())), axis=1)
    out["retention_status"] = np.where(any_fail, "fail", np.where(any_unk, "unknown", "pass"))
    return out


def arcs_from_pairs(nodes: pd.DataFrame, pairs: pd.DataFrame, tau: float, edges: pd.DataFrame | None = None,
                    floors: dict | None = None) -> pd.DataFrame:
    """Both directions of every pair, with separate nutrition / environment / quality components."""
    floors = floors or normalize.env_floors(nodes)
    fwd = pairs.rename(columns={"i": "src", "j": "tgt"})
    rev = pairs.rename(columns={"i": "tgt", "j": "src"})
    A = pd.concat([fwd, rev], ignore_index=True)
    edge_set = set() if edges is None else set(zip(edges["i"], edges["j"])) | set(zip(edges["j"], edges["i"]))
    A["in_graph"] = [(s, t) in edge_set for s, t in zip(A["src"], A["tgt"])]
    S = nodes.iloc[A["src"].to_numpy()].reset_index(drop=True)
    T = nodes.iloc[A["tgt"].to_numpy()].reset_index(drop=True)
    A["src_code"], A["tgt_code"] = S["ciqual_code"].values, T["ciqual_code"].values
    A["src_name"], A["tgt_name"] = S["food_name"].values, T["food_name"].values
    A["src_group"], A["tgt_group"] = S["analysis_group"].values, T["analysis_group"].values
    for k in C.ENV_CORE:
        A[f"{k}_src"] = S[f"{k}_per_100g"].values
        A[f"{k}_tgt"] = T[f"{k}_per_100g"].values
        A[f"d_{k}_rel"] = normalize.relative_change(S[f"{k}_per_100g"], T[f"{k}_per_100g"], floors[k])
    A["climate_gain"] = -A["d_climate_rel"]
    A["n_indicators_improved"] = (A[[f"d_{k}_rel" for k in C.ENV_CORE]] < 0).sum(axis=1)
    A["worst_other_change"] = A[[f"d_{k}_rel" for k in ["land", "water"]]].max(axis=1)
    A["dqr_src"], A["dqr_tgt"] = S["dqr"].values, T["dqr"].values
    A["dqr_max"] = np.maximum(A["dqr_src"], A["dqr_tgt"])
    A["issue"] = (S["issue_direct"].fillna("") + "|" + T["issue_direct"].fillna("")).str.strip("|").values
    ret = retention(S, T)
    A = pd.concat([A, ret.reset_index(drop=True)], axis=1)
    A["nutrition_similarity"] = 1 - np.minimum(A["distance"] / tau, 1.0) if np.isfinite(tau) else 1 - A["distance"] / A["distance"].max()
    A["equivalent"] = (A["distance"] <= tau) & (A["retention_status"] == "pass")
    A["attractive"] = A["climate_gain"] >= C.STRONG_CLIMATE_GAIN
    A["burden_shift"] = A["worst_other_change"] > C.BURDEN_SHIFT_TOL
    A["reliable"] = (A["dqr_max"] <= C.DQR_RELIABLE) & (A["issue"] == "")
    A["edge_class"] = classify(A, tau)
    A["display_score"] = A["climate_gain"].clip(lower=0) * A["nutrition_similarity"] * (1 - (A["dqr_max"] - 1) / 8)
    return A


def classify(A: pd.DataFrame, tau: float) -> pd.Series:
    """Mutually exclusive edge classes.

    * strong - attractive (>= 20 % climate gain), nutritionally equivalent, a mutual-kNN
      graph edge, no burden shift to land/water, reliable DQR and no publisher-flagged issue;
    * equivalent, caveated - attractive and equivalent but failing one of the last conditions;
    * misleading - attractive but demonstrably NOT equivalent (distance > tau or a failed
      retention check);
    * unverifiable - attractive, within tau, but a retention check cannot be evaluated;
    * not attractive - climate gain below 20 %.
    """
    cls = np.full(len(A), "not attractive", dtype=object)
    att = A["attractive"].to_numpy()
    eq = A["equivalent"].to_numpy()
    nonequiv = ~eq & ((A["distance"].to_numpy() > tau) | (A["retention_status"].to_numpy() == "fail"))
    cls[att & nonequiv] = "misleading"
    cls[att & ~eq & ~nonequiv] = "unverifiable (missing nutrients)"
    strong = att & eq & A["in_graph"].to_numpy() & ~A["burden_shift"].to_numpy() & A["reliable"].to_numpy()
    cls[att & eq] = "equivalent, caveated"
    cls[strong] = "strong"
    return pd.Series(cls, index=A.index)


def misleading_reasons(A: pd.DataFrame, tau: float) -> pd.Series:
    r = []
    for d, f in zip(A["distance"], A["retention_fails"]):
        parts = []
        if d > tau:
            parts.append("profile distance > tau")
        if f:
            parts.append("fails: " + f.replace("|", ", "))
        r.append("; ".join(parts))
    return pd.Series(r, index=A.index)


def reclassify(A: pd.DataFrame, tau: float, edges: pd.DataFrame) -> pd.DataFrame:
    """Re-derive tau-dependent columns without recomputing retention (used by D5 grids)."""
    A = A.copy()
    es = set(zip(edges["i"], edges["j"])) | set(zip(edges["j"], edges["i"]))
    A["in_graph"] = [(s, t) in es for s, t in zip(A["src"], A["tgt"])]
    A["nutrition_similarity"] = 1 - np.minimum(A["distance"] / tau, 1.0) if np.isfinite(tau) else 1 - A["distance"] / A["distance"].max()
    A["equivalent"] = (A["distance"] <= tau) & (A["retention_status"] == "pass")
    A["edge_class"] = classify(A, tau)
    return A


def threshold_curve(nodes: pd.DataFrame, pairs: pd.DataFrame, A: pd.DataFrame, k: int = C.KNN_K,
                    quantiles=(0.4, 0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0)) -> pd.DataFrame:
    """Coverage vs edge validity as tau varies (validity = graph arcs passing retention)."""
    rows = []
    for q in quantiles:
        e, tau = knn_graph(nodes, pairs, GraphSpec(k=k, quantile=q))
        es = set(zip(e["i"], e["j"])) | set(zip(e["j"], e["i"]))
        ing = np.array([(s, t) in es for s, t in zip(A["src"], A["tgt"])])
        sub = A[ing]
        evaluable = sub["retention_status"] != "unknown"
        fail = float((sub.loc[evaluable, "retention_status"] == "fail").mean()) if evaluable.any() else np.nan
        deg = pd.concat([e["i"], e["j"]]).value_counts()
        cov = len(deg) / len(nodes)
        rows.append({"quantile": q, "tau": tau, "edges": len(e), "coverage": cov, "isolated_share": 1 - cov,
                     "retention_fail_share": fail, "retention_unknown_share": float((~evaluable).mean()),
                     "cross_group_edges": int((~e["same_group"]).sum()), "score": cov * (1 - fail)})
    return pd.DataFrame(rows)


def select_quantile(curve: pd.DataFrame, tol: float = 0.05) -> float:
    """Most conservative (smallest tau) quantile whose coverage x validity score is within
    ``tol`` (relative) of the maximum - validity is preferred over coverage on a plateau."""
    best = curve["score"].max()
    ok = curve[curve["score"] >= (1 - tol) * best]
    return float(ok["quantile"].min())


# ---------------------------------------------------------------------------
# recommenders: naive (environment only) vs equivalence-constrained
# ---------------------------------------------------------------------------
def recommender_comparison(A: pd.DataFrame, scope: str = "role") -> pd.DataFrame:
    """For each source food: the greenest eligible target vs the greenest *equivalent* target."""
    if scope == "role":
        A = A[A["blocks"].str.contains("role:")]
    A = A[A["climate_gain"] > 0]
    order = A.sort_values(["src_code", "climate_gain", "tgt_code"], ascending=[True, False, True])
    naive = order.drop_duplicates("src_code")
    eq = order[order["equivalent"]].drop_duplicates("src_code").set_index("src_code")
    out = pd.DataFrame({
        "src_code": naive["src_code"].to_numpy(), "src_name": naive["src_name"].to_numpy(),
        "src_group": naive["src_group"].to_numpy(), "naive_tgt": naive["tgt_name"].to_numpy(),
        "naive_gain": naive["climate_gain"].to_numpy(), "naive_class": naive["edge_class"].to_numpy(),
        "naive_equivalent": naive["equivalent"].to_numpy(bool), "naive_retention": naive["retention_status"].to_numpy(),
        "naive_fails": naive["retention_fails"].to_numpy(), "naive_distance": naive["distance"].to_numpy(),
    })
    out["equiv_tgt"] = out["src_code"].map(eq["tgt_name"])
    out["equiv_gain"] = out["src_code"].map(eq["climate_gain"]).fillna(0.0)
    out["price_of_equivalence"] = out["naive_gain"] - out["equiv_gain"]
    return out


# ---------------------------------------------------------------------------
# network metrics and communities
# ---------------------------------------------------------------------------
def communities(G: nx.Graph, seed: int = C.SEED, resolution: float = 1.0) -> dict[int, int]:
    comms = nx.community.louvain_communities(G, weight="weight", seed=seed, resolution=resolution)
    return {n: ci for ci, c in enumerate(sorted(comms, key=len, reverse=True)) for n in c}


def network_metrics(G: nx.Graph, comm: dict[int, int]) -> dict:
    from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

    non_iso = [n for n in G if G.degree(n) > 0]
    H = G.subgraph(non_iso)
    comps = sorted((len(c) for c in nx.connected_components(H)), reverse=True)
    parts = {}
    for n, c in comm.items():
        parts.setdefault(c, set()).add(n)
    mod = nx.community.modularity(G, [parts[c] for c in parts], weight="weight")
    nodes_ni = [n for n in non_iso]
    labels_c = [comm[n] for n in nodes_ni]
    labels_g = [G.nodes[n]["group_code"] for n in nodes_ni]
    labels_cg = [G.nodes[n]["ciqual_group"] for n in nodes_ni]
    degs = np.array([d for _, d in G.degree()])
    return {
        "nodes": G.number_of_nodes(), "edges": G.number_of_edges(), "isolated_nodes": int((degs == 0).sum()),
        "non_isolated_share": len(non_iso) / G.number_of_nodes(),
        "mean_degree": float(degs.mean()), "max_degree": int(degs.max()),
        "density": nx.density(G), "components": len(comps), "largest_component": comps[0] if comps else 0,
        "avg_clustering": nx.average_clustering(H, weight=None) if len(H) else np.nan,
        "communities": len(set(labels_c)), "modularity": mod,
        "nmi_vs_analysis_group": normalized_mutual_info_score(labels_g, labels_c),
        "ari_vs_analysis_group": adjusted_rand_score(labels_g, labels_c),
        "nmi_vs_ciqual_group": normalized_mutual_info_score(labels_cg, labels_c),
        "group_assortativity": nx.attribute_assortativity_coefficient(H, "group_code") if len(H) else np.nan,
    }


def community_table(G: nx.Graph, comm: dict[int, int]) -> pd.DataFrame:
    rows = []
    for n, c in comm.items():
        if G.degree(n) == 0:
            continue
        d = G.nodes[n]
        rows.append({"community": c, "ciqual_code": d["ciqual_code"], "food_name": d["name"],
                     "analysis_group": d["group"], "ciqual_group": d["ciqual_group"], "climate_per_100g": d["climate"],
                     "dqr": d["dqr"], "degree": G.degree(n)})
    t = pd.DataFrame(rows)
    summ = t.groupby("community").agg(
        size=("ciqual_code", "size"), n_groups=("analysis_group", "nunique"),
        dominant_group=("analysis_group", lambda s: s.value_counts().index[0]),
        dominant_share=("analysis_group", lambda s: s.value_counts(normalize=True).iloc[0]),
        median_climate=("climate_per_100g", "median"),
        climate_ratio_max_min=("climate_per_100g", lambda s: s.max() / s.min() if s.min() > 0 else np.nan),
        climate_cv=("climate_per_100g", lambda s: s.std() / s.mean()),
        example_foods=("food_name", lambda s: "; ".join(s.head(4))),
    ).reset_index()
    return t, summ


def community_stability(G: nx.Graph, n_seeds: int, seed: int = C.SEED) -> tuple[float, np.ndarray]:
    from sklearn.metrics import adjusted_rand_score

    nodes = [n for n in G if G.degree(n) > 0]
    labs = []
    for s in range(n_seeds):
        c = communities(G, seed=seed + s)
        labs.append([c[n] for n in nodes])
    M = np.ones((n_seeds, n_seeds))
    for a in range(n_seeds):
        for b in range(a + 1, n_seeds):
            M[a, b] = M[b, a] = adjusted_rand_score(labs[a], labs[b])
    return float(M[np.triu_indices(n_seeds, 1)].mean()), M


# ---------------------------------------------------------------------------
# one-call pipeline (used by D3 and repeatedly by D5)
# ---------------------------------------------------------------------------
@dataclass
class NetworkResult:
    nodes: pd.DataFrame
    pairs: pd.DataFrame
    edges: pd.DataFrame
    tau: float
    G: nx.Graph
    arcs: pd.DataFrame | None = None


def run(m: pd.DataFrame, spec: GraphSpec | None = None, with_arcs: bool = True) -> NetworkResult:
    spec = spec or GraphSpec()
    nodes = m[m["graph_eligible"]].reset_index(drop=True)
    pairs = candidate_pairs(nodes, spec)
    edges, tau = knn_graph(nodes, pairs, spec)
    G = build_graph(nodes, edges, tau)
    arcs = None
    if with_arcs:
        arcs = arcs_from_pairs(nodes, pairs.dropna(subset=["distance"]), tau, edges, normalize.env_floors(m))
        arcs.attrs["tau"] = tau
        arcs["misleading_reason"] = np.where(arcs["edge_class"] == "misleading", misleading_reasons(arcs, tau), "")
        arcs["view"] = spec.view
    return NetworkResult(nodes, pairs, edges, tau, G, arcs)
