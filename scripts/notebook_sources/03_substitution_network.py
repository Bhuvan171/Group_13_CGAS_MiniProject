# %% [markdown]
# # 03 · Deliverable 3 — Nutritional-Equivalence Substitution Network
#
# **Question answered:** *which pairwise replacements are nutritionally plausible, what structure
# emerges across them, and which environmentally attractive swaps are nutritionally misleading?*
#
# **Edge rule (documented, reproducible, leakage-free)**
# 1. *Nodes*: matched foods with ≥ 12 of the 15 frozen nutrients observed (missing values are not imputed).
# 2. *Eligibility*: same analysis group **or** a shared declared basket role (`config/roles.csv`).
# 3. *Nutrient distance*: weighted RMS difference of asinh-compressed, median/IQR-scaled profiles per
#    100 g, over the features observed in both foods (equal weights; varied in D5).
# 4. *Graph*: mutual k-nearest eligible neighbours (k = 5) whose distance ≤ τ, where τ is a quantile
#    of each food's nearest-neighbour distance chosen by an explicit calibration rule (§2).
# 5. *Only after* the graph exists are environmental indicators and DQR attached as edge attributes —
#    a unit test confirms that shuffling every environmental column leaves the edge set unchanged.
#
# **Substitution classes** for a directed swap *source → target* (gram for gram):
# *attractive* = ≥ 20 % climate reduction; *equivalent* = distance ≤ τ **and** all retention checks pass
# (protein ≥ 80 %; energy ratio 0.67–1.5; ≥ 50 % of every micronutrient the source supplies at ≥ 5 % of
# its reference intake; sodium/saturated fat/sugars ≤ 1.25× + tolerance).
# **strong** = attractive + equivalent + graph edge + no burden shift (land/water ≤ +20 %) + both DQR ≤ 3
# and no publisher-flagged error · **misleading** = attractive but demonstrably not equivalent ·
# **unverifiable** = attractive, close, but a retention check needs a missing value.

# %%
import sys
from pathlib import Path

ROOT = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt

from sustainome import config as C, matrix as MX, network as N
from sustainome.plotting import setup, save, PALETTE, SEM, short, group_colors

setup()
pd.set_option("display.width", 220, "display.max_colwidth", 55, "display.max_columns", 30, "display.max_rows", 60)
m = MX.load_matrix()

# %% [markdown]
# ## 1. Eligible pairs and nutrient distances

# %%
nodes = m[m["graph_eligible"]].reset_index(drop=True)
spec0 = N.GraphSpec()
pairs = N.candidate_pairs(nodes, spec0)
print(f"{len(nodes)} nodes; {len(pairs):,} eligible unordered pairs "
      f"({int(pairs['same_group'].sum()):,} within group, {int((~pairs['same_group']).sum()):,} only via a shared role); "
      f"{int(pairs['distance'].isna().sum())} pairs share < 12 observed nutrients (no distance)")

# %% [markdown]
# ## 2. Calibrating the equivalence threshold τ
#
# τ trades **coverage** (share of foods with at least one equivalent neighbour) against **validity**
# (share of graph edges that also pass the *independent* directional retention checks). Rule, fixed in
# advance: take the *most conservative* τ whose coverage × validity score is within 5 % of the maximum.

# %%
A_all = N.arcs_from_pairs(nodes, pairs.dropna(subset=["distance"]), np.inf, None, None)  # tau-free components
curve = N.threshold_curve(nodes, pairs, A_all)
q_sel = N.select_quantile(curve)
print("selected quantile:", q_sel, "| declared in config:", C.DIST_QUANTILE)
assert q_sel == C.DIST_QUANTILE
curve.round(3)

# %%
fig, ax = plt.subplots(figsize=(8.5, 4.4))
ax.plot(curve["quantile"], curve["coverage"], "o-", color=PALETTE[0], label="coverage (non-isolated share)")
ax.plot(curve["quantile"], 1 - curve["retention_fail_share"], "s-", color=PALETTE[2], label="validity (edges passing retention)")
ax.plot(curve["quantile"], curve["score"], "D-", color="k", label="coverage x validity")
best = curve["score"].max()
ax.axhspan(0.95 * best, best, color="#DDDDDD", alpha=0.6, label="within 5% of best")
ax.axvline(q_sel, color=SEM["misleading"], ls="--", lw=1)
ax.text(q_sel + 0.01, 0.12, f"selected q = {q_sel:.2f}\ntau = {curve.set_index('quantile').loc[q_sel, 'tau']:.3f}", color=SEM["misleading"], fontsize=8)
ax.set_xlabel("tau = quantile of each food's nearest-neighbour distance"); ax.set_ylabel("share"); ax.set_ylim(0, 1)
ax.legend(fontsize=7.5, loc="lower right")
save(fig, 3, "Threshold calibration", "most conservative tau on the coverage x validity plateau; varied again in D5")

# %% [markdown]
# ## 3. Build the graph and its directed substitution arcs

# %%
res = N.run(m)
G, A, tau = res.G, res.arcs, res.tau
comm = N.communities(G)
met = N.network_metrics(G, comm)
pd.Series(met).to_frame("value").to_csv(C.TABLES / "D3_network_metrics.csv")
pd.Series(met).to_frame("value")

# %%
cls = A["edge_class"].value_counts().rename("arcs").to_frame()
cls["in_graph"] = A[A["in_graph"]]["edge_class"].value_counts()
cls

# %%
# Published node / edge tables (edge components kept separate and inspectable)
deg = dict(G.degree())
node_tab = res.nodes[["ciqual_code", "food_name", "analysis_group", "ciqual_group", "roles", "dqr",
                      "climate_per_100g", "land_per_100g", "water_per_100g", "n_profile_available"]].copy()
node_tab["degree"] = [deg[i] for i in range(len(node_tab))]
node_tab["community"] = [comm.get(i, -1) if deg[i] > 0 else -1 for i in range(len(node_tab))]
node_tab.to_csv(C.TABLES / "D3_nodes.csv", index=False)
edge_tab = res.edges.assign(src_code=res.nodes.loc[res.edges["i"], "ciqual_code"].to_numpy(),
                            tgt_code=res.nodes.loc[res.edges["j"], "ciqual_code"].to_numpy(),
                            src_name=res.nodes.loc[res.edges["i"], "food_name"].to_numpy(),
                            tgt_name=res.nodes.loc[res.edges["j"], "food_name"].to_numpy())
edge_tab.to_csv(C.TABLES / "D3_edges.csv", index=False)
keep_cols = ["src_code", "src_name", "src_group", "tgt_code", "tgt_name", "tgt_group", "blocks", "distance",
             "nutrition_similarity", "shared_features", "in_graph", "climate_gain"] + [f"d_{k}_rel" for k in C.ENV_CORE] + \
            ["n_indicators_improved", "worst_other_change", "dqr_src", "dqr_tgt", "dqr_max", "issue", "protein_ratio",
             "energy_ratio", "min_micro_retention", "retention_status", "retention_fails", "retention_unknown",
             "equivalent", "attractive", "burden_shift", "reliable", "edge_class", "misleading_reason", "display_score", "view"]
A_pub = A[A["in_graph"] | A["attractive"]][keep_cols]
A_pub.to_csv(C.TABLES / "D3_substitution_arcs.csv", index=False)
print(f"tau = {tau:.3f}; published {len(edge_tab)} edges and {len(A_pub):,} arcs (graph arcs + all attractive candidates)")

# %% [markdown]
# ## 4. Network structure: degrees, components, communities

# %%
comp_sizes = sorted((len(c) for c in nx.connected_components(G.subgraph([n for n in G if G.degree(n) > 0]))), reverse=True)
fig, axes = plt.subplots(1, 3, figsize=(14, 3.9))
degs = np.array([d for _, d in G.degree()])
axes[0].bar(*np.unique(degs, return_counts=True), color=PALETTE[0])
axes[0].set_xlabel("degree"); axes[0].set_ylabel("foods"); axes[0].set_title(f"Degree distribution ({int((degs == 0).sum())} isolated)")
axes[1].hist(comp_sizes, bins=np.logspace(0, np.log10(max(comp_sizes)) + 0.05, 20), color=PALETTE[1])
axes[1].set_xscale("log"); axes[1].set_xlabel("component size (foods)"); axes[1].set_ylabel("components")
axes[1].set_title(f"{len(comp_sizes)} components; largest = {comp_sizes[0]}")
stab, M_ari = N.community_stability(G, C.ROBUST["louvain_seeds"])
im = axes[2].imshow(M_ari, vmin=0.8, vmax=1, cmap="viridis"); axes[2].grid(False)
fig.colorbar(im, ax=axes[2], fraction=0.045, label="ARI")
axes[2].set_title(f"Louvain stability across 20 seeds (mean ARI {stab:.3f})"); axes[2].set_xlabel("seed"); axes[2].set_ylabel("seed")
fig.tight_layout()
save(fig, 3, "Degree components and community stability", f"mutual {C.KNN_K}-NN graph, tau = {tau:.3f}; modularity {met['modularity']:.3f}")

# %%
ctab, csum = N.community_table(G, comm)
csum = csum.sort_values("size", ascending=False)
csum.to_csv(C.TABLES / "D3_communities.csv", index=False)
mixed = csum[csum["n_groups"] > 1].sort_values("size", ascending=False)
print(f"{len(csum)} communities; NMI vs analysis groups = {met['nmi_vs_analysis_group']:.3f}; "
      f"{len(mixed)} communities span >1 analysis group")
print("\nCommunities with the widest climate spread (substitution opportunity zones):")
csum[csum["size"] >= 5].sort_values("climate_ratio_max_min", ascending=False).head(10)[
    ["community", "size", "n_groups", "dominant_group", "median_climate", "climate_ratio_max_min", "example_foods"]]

# %%
H = G.subgraph([n for n in G if G.degree(n) > 0]).copy()
# component-packed layout: lay out each component separately, then pack largest-first in rows
pos, x, y, row_h, row_w = {}, 0.0, 0.0, 0.0, 75.0
for comp in sorted(nx.connected_components(H), key=len, reverse=True):
    sub = H.subgraph(comp)
    p = nx.kamada_kawai_layout(sub) if len(comp) > 2 else {n: (i, 0.0) for i, n in enumerate(comp)}
    arr = np.array([p[n] for n in comp], dtype=float)
    arr -= arr.min(axis=0)
    span = arr.max(axis=0)
    size = 1.6 * np.sqrt(len(comp))
    arr = arr / max(span.max(), 1e-9) * size
    w, h = arr[:, 0].max() + 1.0, arr[:, 1].max() + 1.0
    if x + w > row_w:
        x, y, row_h = 0.0, y - row_h - 0.8, 0.0
    for n, (a, b) in zip(comp, arr):
        pos[n] = (x + a, y - b)
    x += w + 0.8
    row_h = max(row_h, h)
cg = {n: H.nodes[n]["ciqual_group"].replace("\n", " ") for n in H}
gcol = group_colors(cg.values())
ys = np.array([p[1] for p in pos.values()])
fig, ax = plt.subplots(figsize=(16, float(np.clip(13.5 * (ys.max() - ys.min() + 2) / row_w, 5, 14))))
same = [(u, v) for u, v in H.edges() if H.nodes[u]["group_code"] == H.nodes[v]["group_code"]]
cross = [(u, v) for u, v in H.edges() if H.nodes[u]["group_code"] != H.nodes[v]["group_code"]]
nx.draw_networkx_edges(H, pos, edgelist=same, width=0.4, alpha=0.35, edge_color="#999999", ax=ax)
nx.draw_networkx_edges(H, pos, edgelist=cross, width=0.9, alpha=0.8, edge_color="k", ax=ax)
clim = np.array([H.nodes[n]["climate"] for n in H])
sizes = 6 + 30 * (np.log10(clim) - np.log10(clim).min()) / np.ptp(np.log10(clim))
nx.draw_networkx_nodes(H, pos, node_size=sizes, node_color=[gcol[cg[n]] for n in H], linewidths=0, ax=ax)
for g, c in gcol.items():
    ax.scatter([], [], color=c, label=short(g, 30), s=25)
ax.legend(loc="upper center", bbox_to_anchor=(0.5, 0.0), ncol=6, fontsize=7.5, title="CIQUAL group", title_fontsize=8, frameon=False)
ax.set_aspect("equal")
ax.margins(0.01)
ax.set_ylim(min(p[1] for p in pos.values()) - 1, max(p[1] for p in pos.values()) + 1)
ax.axis("off")
save(fig, 3, "Substitution network overview",
     f"{H.number_of_nodes()} connected foods, {H.number_of_edges()} edges, components packed largest first; black = cross-group edges via a shared role; node size ~ log climate per 100 g")

# %% [markdown]
# ## 5. Edge trade-off map: environmental gain vs nutritional distance

# %%
cand = A[(A["climate_gain"] > 0)].copy()
sample = cand.sample(min(len(cand), 25000), random_state=C.SEED)
colors = {"strong": SEM["strong"], "equivalent, caveated": SEM["caveated"], "misleading": SEM["misleading"],
          "unverifiable (missing nutrients)": SEM["unverifiable"], "not attractive": SEM["not attractive"]}
fig, (ax, ax2) = plt.subplots(1, 2, figsize=(14, 5.2), gridspec_kw={"width_ratios": [1.5, 1]})
for c in ["not attractive", "misleading", "unverifiable (missing nutrients)", "equivalent, caveated", "strong"]:
    d = sample[sample["edge_class"] == c]
    ax.scatter(d["distance"], 100 * d["climate_gain"], s=4 if c in ("not attractive", "misleading") else 12,
               alpha=0.35 if c in ("not attractive", "misleading") else 0.8, color=colors[c], lw=0, label=f"{c} ({(cand['edge_class'] == c).sum():,})")
ax.axvline(tau, color="k", ls="--", lw=1); ax.text(tau, 101, f" tau = {tau:.2f}", fontsize=8)
ax.axhline(100 * C.STRONG_CLIMATE_GAIN, color="k", ls=":", lw=0.8)
ax.set_xlabel("nutrient-profile distance (RMS of robust-z differences)"); ax.set_ylabel("climate reduction (%)")
ax.set_xlim(0, cand["distance"].quantile(0.995)); ax.legend(fontsize=7, markerscale=2, loc="lower right")
ax.set_title("All eligible swaps with any climate reduction (sample of 25,000)")
fails = A.loc[A["edge_class"] == "misleading", "retention_fails"].str.split("|").explode()
fails = fails[fails != ""].value_counts()
dist_only = int(((A["edge_class"] == "misleading") & (A["retention_fails"] == "")).sum())
fails = pd.concat([fails, pd.Series({"(distance > tau only)": dist_only})]).sort_values()
ax2.barh(fails.index, fails.to_numpy(), color=SEM["misleading"], alpha=0.8)
ax2.set_xlabel("misleading swaps failing this check"); ax2.set_title("Why attractive swaps fail equivalence")
fig.tight_layout()
save(fig, 3, "Edge tradeoff map and failure reasons", "each point is a directed gram-for-gram swap inside a group or declared role")

# %% [markdown]
# ## 6. Naive "greenest swap" vs equivalence-constrained recommendation
#
# For every food in a declared role, a naive environment-only recommender proposes the eligible food with
# the largest climate reduction. We check whether that proposal survives the equivalence rule, and how much
# climate gain remains if we insist on an equivalent swap (the *price of equivalence*).

# %%
rc = N.recommender_comparison(A)
rc.to_csv(C.TABLES / "D3_recommender_comparison.csv", index=False)
src_role = m.set_index("ciqual_code")["roles"].str.split("|").str[0]
rc["role"] = rc["src_code"].map(src_role)
by_role = rc.groupby("role").agg(sources=("src_code", "size"), naive_equivalent=("naive_equivalent", "mean"),
                                 median_naive_gain=("naive_gain", "median"), median_equiv_gain=("equiv_gain", "median"),
                                 has_equivalent=("equiv_tgt", lambda s: s.notna().mean()))
print(f"naive greenest swap is nutritionally equivalent for {100 * rc['naive_equivalent'].mean():.1f}% of {len(rc)} source foods")
print(f"median naive gain {100 * rc['naive_gain'].median():.0f}% vs median equivalent gain {100 * rc['equiv_gain'].median():.0f}%; "
      f"{100 * rc['equiv_tgt'].notna().mean():.0f}% of sources have at least one equivalent improving swap")
by_role.round(3)

# %%
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.6))
bins = np.linspace(0, 100, 26)
ax1.hist(100 * rc["naive_gain"], bins=bins, alpha=0.6, color=SEM["misleading"], label="naive greenest swap")
ax1.hist(100 * rc["equiv_gain"], bins=bins, alpha=0.6, color=SEM["strong"], label="best equivalent swap")
ax1.set_xlabel("climate reduction of the recommended swap (%)"); ax1.set_ylabel("source foods"); ax1.legend()
ax1.set_title("Price of nutritional equivalence")
br = by_role.sort_values("naive_equivalent")
ax2.barh(br.index, 100 * (1 - br["naive_equivalent"]), color=SEM["misleading"], alpha=0.8, label="naive swap NOT equivalent")
ax2.barh(br.index, 100 * br["has_equivalent"], color=SEM["strong"], alpha=0.35, label="sources with >= 1 equivalent improving swap")
ax2.set_xlabel("% of source foods in role"); ax2.legend(fontsize=7, loc="lower right"); ax2.set_title("By declared role")
fig.tight_layout()
save(fig, 3, "Naive vs equivalence constrained recommendations", "each source food in a declared role; climate per 100 g, gram-for-gram")

# %% [markdown]
# ## 7. Strong alternatives vs environmentally attractive but misleading swaps

# %%
strong = A[A["edge_class"] == "strong"].sort_values("display_score", ascending=False)
mis = A[(A["edge_class"] == "misleading") & A["blocks"].str.contains("role:")].sort_values("climate_gain", ascending=False)
top_s = strong.drop_duplicates("src_code").head(12)
top_m = mis.drop_duplicates("src_group").head(12)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6.2))
for ax, d, col, ttl in [(ax1, top_s, SEM["strong"], "Strong: equivalent, reliable, no burden shift"),
                        (ax2, top_m, SEM["misleading"], "Misleading: large gain, fails equivalence")]:
    lbl = [f"{short(a, 30)} → {short(b, 30)}" for a, b in zip(d["src_name"], d["tgt_name"])]
    ax.barh(range(len(d)), 100 * d["climate_gain"], color=col, alpha=0.8)
    ax.set_yticks(range(len(d))); ax.set_yticklabels(lbl, fontsize=7.5); ax.invert_yaxis()
    ax.set_xlabel("climate reduction (%)"); ax.set_title(ttl); ax.set_xlim(0, 105)
    for y, (_, r) in enumerate(d.iterrows()):
        note = f"d={r['distance']:.2f}, DQR {r['dqr_max']:.1f}" if col == SEM["strong"] else r["retention_fails"].replace("|", ", ")[:40]
        ax.text(2, y, note, va="center", fontsize=6.5, color="white" if col == SEM["misleading"] else "black")
fig.tight_layout()
save(fig, 3, "Top strong and misleading substitutions", "one example per source food (strong) or per source group (misleading); gram-for-gram per 100 g")
pd.concat([top_s.assign(kind="strong"), top_m.assign(kind="misleading")])[
    ["kind", "src_name", "tgt_name", "distance", "climate_gain", "d_land_rel", "d_water_rel", "dqr_max", "retention_fails"]].round(3)

# %% [markdown]
# ## 8. Manual audit of representative edges
#
# Thirty arcs were inspected by hand — the 10 highest-scoring strong swaps, the 10 largest misleading
# role-level swaps (one per source group) and 10 random graph arcs — and a verdict was recorded in
# `config/manual_edge_audit.csv` (culinary plausibility and whether the class is warranted).

# %%
rand = A[A["in_graph"]].sample(10, random_state=C.SEED)
audit = pd.concat([strong.head(10).assign(sample="top strong"), mis.drop_duplicates("src_group").head(10).assign(sample="top misleading"),
                   rand.assign(sample="random graph arc")])
audit = audit[["sample", "src_code", "src_name", "tgt_code", "tgt_name", "edge_class", "distance", "climate_gain",
               "d_land_rel", "d_water_rel", "dqr_max", "retention_status", "retention_fails"]]
vpath = C.CONFIG_DIR / "manual_edge_audit.csv"
if vpath.exists():
    v = pd.read_csv(vpath, dtype={"src_code": str, "tgt_code": str})
    audit = audit.merge(v[["src_code", "tgt_code", "verdict", "comment"]], on=["src_code", "tgt_code"], how="left")
audit.to_csv(C.TABLES / "D3_manual_edge_audit.csv", index=False)
audit.round(3)

# %% [markdown]
# ## Key findings (D3)

# %%
print(f"1. Graph: {met['nodes']} foods, {met['edges']} edges (tau = {tau:.3f}), {met['communities']} communities, "
      f"modularity {met['modularity']:.2f}, NMI with CIQUAL analysis groups {met['nmi_vs_analysis_group']:.2f}.")
print(f"2. Of {int(A['attractive'].sum()):,} environmentally attractive swaps (>= 20% climate gain): "
      f"{int((A['edge_class']=='misleading').sum()):,} misleading, {int((A['edge_class']=='strong').sum())} strong.")
print(f"3. Naive greenest swap is nutritionally equivalent for only {rc['naive_equivalent'].mean():.1%} of {len(rc)} foods; "
      f"median gain falls from {rc['naive_gain'].median():.0%} (naive) to {rc['equiv_gain'].median():.0%} (equivalent).")
print(f"4. Most frequent failure reasons: {', '.join(fails.sort_values(ascending=False).index[:4])}.")

# %% [markdown]
# **Interpretation.** Nutritional equivalence is the binding constraint on pairwise substitution: the greenest
# eligible swap almost always sacrifices protein density, B12, zinc or energy gram for gram. Strong, reliable
# swaps exist but are concentrated among close relatives (oils, dairy, fish, breads), where gains are modest.
# **Limitations:** swaps are gram-for-gram (portion compensation is a basket-level question, D4); the 15-nutrient
# profile misses fatty-acid quality (see the linseed-oil audit note); edges resting on missing values are tested in D5.

# %% [markdown]
# ## 9. Acceptance criteria

# %%
acc = pd.DataFrame([
    ("graph construction reproducible and leakage-free (tests/test_pipeline.py)", True),
    ("edge components remain inspectable (D3_substitution_arcs.csv)", {"distance", "climate_gain", "dqr_max", "retention_fails"} <= set(A_pub.columns)),
    ("threshold choice justified (calibration rule, selected q = declared q)", q_sel == C.DIST_QUANTILE),
    ("sensitivity to alternate thresholds delegated to D5", True),
    ("representative top edges manually audited", vpath.exists() and audit["verdict"].notna().sum() >= 25),
], columns=["D3 acceptance criterion", "met"])
acc.to_csv(C.TABLES / "D3_acceptance.csv", index=False)
acc
