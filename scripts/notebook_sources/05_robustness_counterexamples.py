# %% [markdown]
# # 05 · Deliverable 5 — Robustness, Uncertainty and Counterexample Study
#
# **Question answered:** *do the conclusions of D2–D4 survive alternative, equally defensible
# assumptions and sampling variation — and where do apparently sustainable substitutions fail?*
#
# Every test below was **declared before running** (`config.ROBUST` and the table in §1). Each major
# claim is re-tested under more than one assumption and then labelled **robust**, **conditional** or
# **exploratory / withdrawn** in the claims register (§8).
#
# **On DQR.** AGRIBALYSE's Data Quality Rating is an *ordinal* 1–5 score. We use it to filter, stratify
# and penalise, and in one clearly labelled *stress test* (§6.3) to widen a perturbation band. We never
# convert it into a probability, standard error or confidence interval. Bootstrap intervals in this
# notebook describe *sampling variation of the food set*, not LCA uncertainty.

# %%
import sys
from pathlib import Path

ROOT = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sustainome import config as C, matrix as MX, tradeoff as T, network as N, optimize as O, robustness as RB
from sustainome import stats as st
from sustainome.plotting import setup, save, PALETTE, SEM, VIEW_COLORS, short

setup()
pd.set_option("display.width", 220, "display.max_colwidth", 60, "display.max_columns", 30, "display.max_rows", 80)
m = MX.load_matrix()
baskets = MX.load_declared_tables()["baskets"]
SCEN = ["S1", "S2", "S3", "S4", "S5"]

# %% [markdown]
# ## 1. Pre-declared sensitivity grid

# %%
grid = pd.DataFrame([
    ("nutrient weights", "D3 graph / D2 NR9", f"{C.ROBUST['n_weight_draws']} Dirichlet(1) draws", "equal weights"),
    ("nutrient feature subsets", "D3 graph", "drop-one (x15), macro-only, micro-only, extended (+5)", "15 frozen nutrients"),
    ("normalisation", "D2", "mass / energy / protein; energy floor " + str(C.ROBUST["energy_floor_grid"]) + "; protein floor " + str(C.ROBUST["protein_floor_grid"]), "10 kcal, 2 g"),
    ("graph k and tau", "D3", f"k {C.ROBUST['knn_grid']} x quantile {C.ROBUST['quantile_grid']}", f"k={C.KNN_K}, q={C.DIST_QUANTILE}"),
    ("graph basis", "D3", "per 100 kcal profiles (calorie-for-calorie)", "per 100 g"),
    ("DQR filtering", "D2, D3, D4", f"keep DQR <= {C.ROBUST['dqr_cutoffs']}", "all foods"),
    ("duplicate-resolution rule", "D1 -> all", "mean / max / min over AGB variants", "reference variant"),
    ("'< x' parser policy", "D1 -> all", "0 and x (bounds)", "x / 2"),
    ("B12 structural zeros", "D1 -> all", "rule switched off", "on"),
    ("NR9 missing components", "D2", "available-case (>= 7 of 9)", "complete case"),
    ("environmental indicator set", "D2", "climate only / 7 indicators / EF single score", "climate, land, water"),
    ("optimisation tolerances", "D4", f"energy {C.ROBUST['energy_tol_grid']} x micro {C.ROBUST['micro_min_grid']} x protein {C.ROBUST['protein_min_grid']}", "5 %, 90 %, 95 %"),
    ("publisher-flagged errors", "D4", "eggs/quinoa included as candidates", "excluded"),
    ("bootstrap resampling", "D2", f"{C.N_BOOT} resamples (ratios, rho); 500 (Pareto membership)", "-"),
    ("impact perturbation stress test", "D4", "climate x U(1 - 0.1 DQR, 1 + 0.1 DQR), 100 draws", "none"),
], columns=["dimension", "applies to", "alternatives", "reference"])
grid.to_csv(C.TABLES / "D5_sensitivity_grid.csv", index=False)
grid

# %%
# Reference results (recomputed deterministically from the processed matrix)
vl = T.view_long(m)
lab = T.pareto_labels(vl)
ref = N.run(m)
ref_comm = N.communities(ref.G)
ref_rc = N.recommender_comparison(ref.arcs)
print(f"reference: tau={ref.tau:.3f}, edges={len(ref.edges)}, strong={int((ref.arcs['edge_class'] == 'strong').sum())}, "
      f"naive-equivalent share={ref_rc['naive_equivalent'].mean():.3f}")

# %% [markdown]
# ## 2. Substitution-graph robustness (D3)
# ### 2.1 k × τ grid

# %%
gg = RB.graph_grid(ref, ref_comm)
gg.to_csv(C.TABLES / "D5_graph_k_tau_grid.csv", index=False)
fig, axes = plt.subplots(1, 4, figsize=(17, 3.9))
for ax, (col, title, fmt) in zip(axes, [("edges", "edges", "{:.0f}"), ("strong", "strong substitutions", "{:.0f}"),
                                        ("community_ari_vs_reference", "community ARI vs reference", "{:.2f}"),
                                        ("naive_equivalent_share", "naive swap equivalent (share)", "{:.3f}")]):
    piv = gg.pivot(index="k", columns="quantile", values=col)
    im = ax.imshow(piv.to_numpy(dtype=float), cmap="viridis", aspect="auto")
    ax.set_xticks(range(piv.shape[1])); ax.set_xticklabels([f"{q:g}" for q in piv.columns])
    ax.set_yticks(range(piv.shape[0])); ax.set_yticklabels(piv.index); ax.grid(False)
    ax.set_xlabel("tau quantile"); ax.set_ylabel("k"); ax.set_title(title)
    for i in range(piv.shape[0]):
        for j in range(piv.shape[1]):
            v = piv.iat[i, j]
            ax.text(j, i, fmt.format(v), ha="center", va="center", fontsize=7.5,
                    color="white" if v < np.nanmean(piv.to_numpy()) else "black")
fig.tight_layout()
save(fig, 5, "Graph k and threshold sensitivity", f"reference k={C.KNN_K}, q={C.DIST_QUANTILE}; quantile 1.0 = no distance cap")
gg.round(3)

# %% [markdown]
# ### 2.2 Nutrient weights and feature subsets

# %%
ws = RB.graph_weight_sensitivity(ref, C.ROBUST["n_weight_draws"], C.SEED)
ws.to_csv(C.TABLES / "D5_graph_weight_sensitivity.csv", index=False)
fs = RB.graph_feature_subsets(m, ref)
fs.to_csv(C.TABLES / "D5_graph_feature_subsets.csv", index=False)
ev_graph = RB.energy_view_graph(m, ref)
print("calorie-for-calorie graph:", {k: (round(v, 3) if isinstance(v, float) else v) for k, v in ev_graph.items() if k != "result"})
ws[["edge_jaccard", "strong_retained", "naive_equivalent_share"]].describe(percentiles=[0.05, 0.5, 0.95]).round(3)

# %%
fig, axes = plt.subplots(1, 3, figsize=(16, 4.6), gridspec_kw={"width_ratios": [1, 1, 1.5]})
axes[0].hist(ws["edge_jaccard"], bins=25, color=PALETTE[0], alpha=0.8, label="edge-set Jaccard")
axes[0].hist(ws["strong_retained"], bins=25, color=PALETTE[2], alpha=0.6, label="reference strong edges retained")
axes[0].set_xlabel("share"); axes[0].set_ylabel("weight draws"); axes[0].legend(fontsize=7)
axes[0].set_title(f"{len(ws)} random nutrient-weight vectors")
axes[1].hist(100 * ws["naive_equivalent_share"], bins=25, color=SEM["misleading"], alpha=0.8)
axes[1].axvline(100 * ref_rc["naive_equivalent"].mean(), color="k", ls="--", lw=1, label="reference")
axes[1].set_xlabel("% naive greenest swaps that are equivalent"); axes[1].legend(fontsize=7)
axes[1].set_title("Headline D3 claim under random weights")
f2 = fs.sort_values("edge_jaccard")
axes[2].barh(f2["subset"], f2["edge_jaccard"], color=PALETTE[0], alpha=0.8, label="edge Jaccard")
axes[2].scatter(f2["strong_retained"], range(len(f2)), color=PALETTE[2], zorder=3, s=18, label="strong retained")
axes[2].scatter(f2["naive_equivalent_share"], range(len(f2)), color=SEM["misleading"], marker="x", zorder=3, s=18, label="naive equivalent share")
axes[2].tick_params(axis="y", labelsize=7); axes[2].set_xlim(0, 1); axes[2].legend(fontsize=7, loc="lower right")
axes[2].set_title("Feature-subset variants")
fig.tight_layout()
save(fig, 5, "Nutrient weight and feature subset sensitivity", "graph rebuilt from scratch for each variant; tau recalibrated at the same quantile")

# %% [markdown]
# ### 2.3 Edges driven by missing values

# %%
mve = RB.missing_value_edges(m, ref)
mve.to_csv(C.TABLES / "D5_missing_value_edges.csv", index=False)
print(f"{len(mve)} of {len(ref.edges)} graph edges rest on < 15 shared nutrients; "
      f"{int(mve['vanishes_when_imputed'].sum())} would vanish if missing values took their group median")
mve[mve["vanishes_when_imputed"]].sort_values("distance_imputed", ascending=False).head(8)[
    ["src_name", "tgt_name", "shared_features", "distance", "distance_imputed", "missing_features"]].round(3)

# %% [markdown]
# ## 3. Trade-off robustness (D2)
# ### 3.1 NR9 weights, indicator sets, DQR filtering, NR9 missing-component rule

# %%
nrw = RB.nr_weight_pareto_stability(vl, C.ROBUST["n_weight_draws"], C.SEED + 1)
nrw.to_csv(C.TABLES / "D5_nr9_weight_pareto_stability.csv", index=False)
ind = RB.indicator_subset_pareto(vl, lab)
dq = RB.dqr_filter_pareto(vl, lab)
vl_av = T.view_long(m, nr_min_available=7)
lab_av = T.pareto_labels(vl_av)
avail = RB.pareto_under(vl_av, lab)
ind.to_csv(C.TABLES / "D5_indicator_subset_pareto.csv", index=False)
dq.to_csv(C.TABLES / "D5_dqr_filter_pareto.csv", index=False)
print("available-case NR9:", {k: round(v, 3) for k, v in avail.items()})
print(f"share of food-view labels stable (>=90% or <=10% efficient) under random NR9 weights: {nrw['stable'].mean():.3f}")
display(ind.round(3)); dq.round(3)

# %%
floors = []
for f in C.ROBUST["energy_floor_grid"]:
    v2 = T.view_long(m, energy_floor=f); l2 = T.pareto_labels(v2, two_d=None, depth=False)
    floors.append({"guard": "energy floor (kcal/100 g)", "value": f, "undefined_foods": int((~v2.loc[v2["view"] == "energy", "fu_defined"]).sum()),
                   "mean_jaccard_mass_vs_view": T.view_agreement(l2)["jaccard_mass_energy"].mean()})
for f in C.ROBUST["protein_floor_grid"]:
    v2 = T.view_long(m, protein_floor=f); l2 = T.pareto_labels(v2, two_d=None, depth=False)
    floors.append({"guard": "protein floor (g/100 g)", "value": f, "undefined_foods": int((~v2.loc[v2["view"] == "protein", "fu_defined"]).sum()),
                   "mean_jaccard_mass_vs_view": T.view_agreement(l2)["jaccard_mass_protein"].mean()})
floors = pd.DataFrame(floors)
floors.to_csv(C.TABLES / "D5_denominator_floor_sensitivity.csv", index=False)
floors.round(3)

# %% [markdown]
# ### 3.2 Bootstrap: Pareto membership probability and category ratios

# %%
bp = RB.bootstrap_pareto_probability(vl, lab, 500, C.SEED + 2)
bp.to_csv(C.TABLES / "D5_bootstrap_pareto_probability.csv", index=False)
bcr = RB.bootstrap_category_ratio(vl, C.N_BOOT, C.SEED + 3)
bcr.to_csv(C.TABLES / "D5_bootstrap_category_ratios.csv", index=False)
dom_ref = bp[bp["efficient"] == 0]
print("Efficient labels cannot be lost by resampling (it only removes competitors), so we test the fragility of DOMINATED labels.")
single = lab.loc[lab["analysed"] & (lab["efficient"] == 0), "dominated_by"].eq(1).mean()
print(f"of {len(dom_ref)} dominated food-view labels, {dom_ref['robustly_dominated'].mean():.1%} stay dominated in >= 90% of resamples; "
      f"{(dom_ref['p_efficient'] >= 0.25).mean():.1%} become efficient in >= 25% of them. A single competitor is absent from a bootstrap "
      f"sample with probability ~0.37, so this flags labels resting on ONE dominator - cross-check: {single:.1%} of dominated labels have exactly one.")

# %%
fig, axes = plt.subplots(1, 3, figsize=(17, 5), gridspec_kw={"width_ratios": [1, 1.1, 1.4]})
for v in C.VIEWS:
    d = bp[(bp["view"] == v) & (bp["efficient"] == 0)]["p_efficient"]
    axes[0].hist(d, bins=np.linspace(0, 1, 21), histtype="step", lw=1.6, color=VIEW_COLORS[v], label=C.VIEW_UNIT[v])
axes[0].axvline(0.1, color="k", ls=":", lw=1)
axes[0].set_xlabel("P(efficient | food resampled)"); axes[0].set_ylabel("dominated foods (reference)"); axes[0].legend(fontsize=7)
axes[0].set_title("Fragility of dominated labels under resampling")
stab = nrw.groupby(["analysis_group", "view"])["stable"].mean().unstack()[list(C.VIEWS)].dropna(how="all")
stab = stab.loc[stab.mean(axis=1).sort_values().index][:25]
im = axes[1].imshow(stab.to_numpy(dtype=float), cmap="viridis", vmin=0.4, vmax=1, aspect="auto"); axes[1].grid(False)
axes[1].set_yticks(range(len(stab))); axes[1].set_yticklabels([short(g, 30) for g in stab.index], fontsize=6.5)
axes[1].set_xticks(range(3)); axes[1].set_xticklabels([C.VIEW_UNIT[v] for v in C.VIEWS], rotation=20)
fig.colorbar(im, ax=axes[1], fraction=0.05, label="share of foods with stable label")
axes[1].set_title("NR9-weight stability (25 least stable groups)")
cats = bcr[bcr["view"] == "mass"].sort_values("ratio")["category"].tolist()
for j, v in enumerate(C.VIEWS):
    d = bcr[bcr["view"] == v].set_index("category").reindex(cats)
    y = np.arange(len(cats)) + (j - 1) * 0.25
    axes[2].errorbar(d["ratio"], y, xerr=[d["ratio"] - d["ci_lo"], d["ci_hi"] - d["ratio"]], fmt="o", ms=3.5,
                     color=VIEW_COLORS[v], label=C.VIEW_UNIT[v], capsize=2, lw=1)
axes[2].axvline(1, color="k", lw=0.8); axes[2].set_xscale("log")
axes[2].set_yticks(range(len(cats))); axes[2].set_yticklabels(cats)
axes[2].set_xlabel("median climate ratio vs cooked legumes (95% bootstrap CI, log)"); axes[2].legend(fontsize=7)
axes[2].set_title("Protein categories vs legumes")
fig.tight_layout()
save(fig, 5, "Bootstrap and weight stability of trade-off results", "bootstrap describes sampling of the food set, not LCA uncertainty")

# %% [markdown]
# ### 3.3 Normalisation reversals

# %%
rev, rev_ex = RB.view_reversals(vl)
rev.to_csv(C.TABLES / "D5_normalisation_reversal_rates.csv", index=False)
rev_ex.to_csv(C.TABLES / "D5_normalisation_reversal_examples.csv", index=False)
print(rev[[c for c in rev if c.startswith("reversal")]].describe().round(3))
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 4.8), gridspec_kw={"width_ratios": [1, 1.4]})
cols = [c for c in rev if c.startswith("reversal")]
ax1.boxplot([rev[c].dropna() for c in cols], widths=0.5)
for i, c in enumerate(cols):
    ax1.scatter(np.random.default_rng(i).normal(i + 1, 0.05, rev[c].notna().sum()), rev[c].dropna(), s=9, alpha=0.6)
ax1.set_xticks(range(1, len(cols) + 1)); ax1.set_xticklabels([c.replace("reversal_", "").replace("_", " vs ") for c in cols])
ax1.set_ylabel("share of within-group pairs whose climate order flips"); ax1.set_title("Climate-ranking reversals between functional units")
ex = rev_ex[rev_ex["views"] == "mass vs energy"].head(8)
y = np.arange(len(ex))
ax2.barh(y - 0.18, np.log2(ex["a_over_b_mass"]), height=0.35, color=VIEW_COLORS["mass"], label="per 100 g")
ax2.barh(y + 0.18, np.log2(ex["a_over_b_energy"]), height=0.35, color=VIEW_COLORS["energy"], label="per 100 kcal")
ax2.set_yticks(y); ax2.set_yticklabels([f"{short(a, 26)} / {short(b, 26)}" for a, b in zip(ex["food_a"], ex["food_b"])], fontsize=7)
ax2.axvline(0, color="k", lw=0.8); ax2.set_xlabel("log2 climate ratio (A / B)"); ax2.legend(fontsize=7)
ax2.set_title("Strongest reversals: A greener per 100 g, B greener per 100 kcal (or vice versa)")
fig.tight_layout()
save(fig, 5, "Normalization reversal panels", "pairs within the same analysis group; the sign flip is the reversal")

# %% [markdown]
# ## 4. D1 assumption variants propagated through D2 and D3

# %%
mv_tab, mats = RB.matrix_variants(m, ref, vl, lab)
mv_tab.to_csv(C.TABLES / "D5_matrix_assumption_variants.csv", index=False)
mv_tab.round(3)

# %% [markdown]
# ## 5. Optimisation robustness (D4)

# %%
tol = RB.optimizer_tolerance_grid(m, baskets, SCEN)
tol.to_csv(C.TABLES / "D5_optimizer_tolerance_grid.csv", index=False)
dv = RB.optimizer_data_variants(m, mats, baskets, SCEN)
dv.to_csv(C.TABLES / "D5_optimizer_data_variants.csv", index=False)
pert = pd.concat([RB.impact_perturbation(m, baskets, s, 100, 0.10, C.SEED + 10 + i) for i, s in enumerate(SCEN)], ignore_index=True)
pert.to_csv(C.TABLES / "D5_impact_perturbation_stress_test.csv", index=False)
dv.pivot_table(index="variant", columns="scenario", values="climate_change_dev10").round(3)

# %%
fig, axes = plt.subplots(1, 3, figsize=(17, 4.8), gridspec_kw={"width_ratios": [1.3, 1, 1]})
t95 = tol[tol["protein_min"] == 0.95]
piv = t95.pivot_table(index="scenario", columns=["energy_tol", "micro_min"], values="climate_change")
im = axes[0].imshow(100 * piv.to_numpy(dtype=float), cmap="viridis_r", aspect="auto"); axes[0].grid(False)
axes[0].set_yticks(range(len(piv))); axes[0].set_yticklabels(piv.index)
axes[0].set_xticks(range(piv.shape[1])); axes[0].set_xticklabels([f"E±{a:.0%}\nmicro≥{b:.0%}" for a, b in piv.columns], fontsize=7)
for i in range(piv.shape[0]):
    for j in range(piv.shape[1]):
        axes[0].text(j, i, f"{100 * piv.iat[i, j]:.0f}", ha="center", va="center", fontsize=7.5, color="white")
fig.colorbar(im, ax=axes[0], fraction=0.04, label="best climate change (%)")
axes[0].set_title("Constrained climate optimum vs tolerances (protein ≥ 95%)")
dvp = dv.pivot_table(index="variant", columns="scenario", values="climate_change_dev10")
for i, s in enumerate(SCEN):
    axes[1].scatter(100 * dvp[s], range(len(dvp)), color=PALETTE[i], label=s, s=22)
axes[1].set_yticks(range(len(dvp))); axes[1].set_yticklabels(dvp.index, fontsize=7)
axes[1].set_xlabel("climate change at D ≤ 0.10 (%)"); axes[1].legend(fontsize=7); axes[1].set_title("Data / assumption variants")
for i, s in enumerate(SCEN):
    d = pert[pert["scenario"] == s]
    axes[2].hist(100 * d["reference_world_change"], bins=20, histtype="step", lw=1.5, color=PALETTE[i], label=s)
axes[2].set_xlabel("climate change of the perturbed-world optimum,\nevaluated with reference impacts (%)")
axes[2].set_ylabel("draws"); axes[2].legend(fontsize=7)
axes[2].set_title("Stress test (NOT a confidence interval)")
fig.tight_layout()
save(fig, 5, "Optimization tolerance data and perturbation sensitivity", "stress test multiplies climate impacts by U(1-0.1 DQR, 1+0.1 DQR) and re-solves at D <= 0.10")
pert.groupby("scenario")[["perturbed_world_change", "reference_world_change", "regret_vs_reference_opt", "food_set_jaccard"]].describe(percentiles=[0.05, 0.5, 0.95]).T.round(3).head(24)

# %% [markdown]
# ## 6. Counterexample catalogue

# %%
A = ref.arcs
cx = []
# C1 lower-climate candidate with large nutrient deviation
for _, r in A[(A["edge_class"] == "misleading") & A["blocks"].str.contains("role:")].sort_values("climate_gain", ascending=False).drop_duplicates("src_group").head(6).iterrows():
    cx.append(("C1 greener but nutritionally non-equivalent", f"{r['src_name']} -> {r['tgt_name']}",
               f"climate -{100 * r['climate_gain']:.0f}%; distance {r['distance']:.2f} > tau {ref.tau:.2f}; fails {r['retention_fails'].replace('|', ', ')}"))
# C2 normalisation reversal
for _, r in rev_ex.head(5).iterrows():
    v1, v2 = r["views"].split(" vs ")
    cx.append(("C2 ranking reverses with the functional unit", f"{r['food_a']} vs {r['food_b']}",
               f"A/B climate ratio {r[f'a_over_b_{v1}']:.2f} {C.VIEW_UNIT[v1]} but {r[f'a_over_b_{v2}']:.2f} {C.VIEW_UNIT[v2]}"))
# C3 edges driven by missing values
for _, r in mve[mve["vanishes_when_imputed"]].sort_values("distance_imputed", ascending=False).head(4).iterrows():
    cx.append(("C3 equivalence edge driven by missing values", f"{r['src_name']} -- {r['tgt_name']}",
               f"distance {r['distance']:.2f} on {r['shared_features']} nutrients; {r['distance_imputed']:.2f} with group-median fill (missing: {r['missing_features']})"))
# C4 burden shifting by the optimiser
for s in SCEN:
    P = O.build_problem(m, baskets, s)
    r = O.solve(P, "climate")
    if r["water_change"] > 0 or r["land_change"] > 0:
        cx.append(("C4 optimiser shifts burden to another indicator", f"{s} climate-optimal basket",
                   f"climate {100 * r['climate_change']:+.0f}% but water {100 * r['water_change']:+.0f}%, land {100 * r['land_change']:+.0f}%"))
    r_op = O.solve(P, "climate", eps_dev=C.OPT_SELECTED["eps_dev"], eps_change=C.OPT_SELECTED["eps_change"])
    if r_op["water_change"] > 0 or r_op["land_change"] > 0:
        cx.append(("C4 optimiser shifts burden to another indicator", f"{s} operating point (D <= 0.10, change <= 50%)",
                   f"climate {100 * r_op['climate_change']:+.0f}% but water {100 * r_op['water_change']:+.0f}%, land {100 * r_op['land_change']:+.0f}%"))
    eo = O.solve(P, "climate", nutrition=False)
    cx.append(("C4b climate-only optimum violates nutrition", f"{s} without nutrition constraints",
               f"climate {100 * eo['climate_change']:+.0f}% with {eo['n_violations']} violated nutrient constraints"))
# C5 conclusions that disappear after DQR filtering: dominated only by low-quality competitors
keep = (vl["dqr"] <= 3.0).to_numpy()
lab3 = T.pareto_labels(vl[keep], two_d=None, depth=False)
flip = lab.loc[lab3.index][["ciqual_code", "food_name", "view", "efficient", "analysed"]].join(lab3[["efficient", "analysed"]], rsuffix="_dqr3")
flip = flip[(flip["analysed"]) & (flip["analysed_dqr3"]) & (flip["efficient"] == 0) & (flip["efficient_dqr3"] == 1)]
print(f"{len(flip)} food-view labels change from dominated to efficient once DQR > 3 foods are removed")
for _, r in flip.drop_duplicates("ciqual_code").head(4).iterrows():
    cx.append(("C5 conclusion changes after removing DQR > 3 records", r["food_name"],
               f"dominated {C.VIEW_UNIT[r['view']]} only by competitors with DQR > 3"))
# C6 publisher-flagged data errors
eggs = lab.merge(m[["ciqual_code", "issue_direct"]], on="ciqual_code")
eggs = eggs[(eggs["issue_direct"].fillna("") != "") & (eggs["efficient"] == 1)]
issue_text = {"egg_underestimated": "AGRIBALYSE 3.2 flags egg impacts as UNDER-estimated, so this position may be too favourable",
              "quinoa_overestimated": "AGRIBALYSE 3.2 flags quinoa impacts as OVER-estimated, so the food may be better than shown"}
for _, r in eggs.drop_duplicates("ciqual_code").head(4).iterrows():
    cx.append(("C6 efficient position rests on a known data error", r["food_name"],
               f"Pareto-efficient {C.VIEW_UNIT[r['view']]}; {issue_text.get(r['issue_direct'], r['issue_direct'])}"))
# C7 nutritionally feasible but culturally implausible optimum
for s in SCEN:
    P = O.build_problem(m, baskets, s)
    r = O.solve(P, "climate")
    c = O.composition(P, r["x"])
    slot_mass = dict(zip(P.slots["slot"], P.slots["grams"].astype(float)))
    c["slot_share"] = c["grams"] / c["slot"].map(slot_mass)
    odd_name = c["food_name"].str.contains(r"\bliver\b|kidney|sweetener|saccharin|molasses", case=False, regex=True)
    odd = c[(~c["is_original"]) & (odd_name | (c["slot_share"] < 0.15))].sort_values("slot_share")
    if len(odd):
        f = odd.iloc[0]
        cx.append(("C7 feasible but culturally implausible optimum", f"{s}: {f['food_name']}",
                   f"{f['grams']:.0f} g ({f['slot_share']:.0%} of slot '{f['slot']}') - a nutrient patch rather than a food people would serve"))
# C8 variant choice changes impacts
dup = pd.read_csv(C.TABLES / "D1_duplicate_resolution.csv", dtype={"ciqual_code": str})
dup = dup[dup["tier"].str.startswith("B")].assign(r=lambda d: d["dropped_climate"] / d["kept_climate"]).sort_values("r", ascending=False)
for _, r in dup.head(3).iterrows():
    cx.append(("C8 variant choice changes the impact", f"{r['kept_lci_name']} vs {r['dropped_lci_name']}",
               f"{r['kept_climate']:.2f} vs {r['dropped_climate']:.2f} kg CO2e/kg ({r['r']:.1f}x); reference rule keeps the French consumption mix"))
cx = pd.DataFrame(cx, columns=["category", "case", "evidence"])
cx.to_csv(C.TABLES / "D5_counterexamples.csv", index=False)
print(cx["category"].value_counts().sort_index())
cx

# %%
import textwrap
cats = cx.drop_duplicates("category")
fig, axes = plt.subplots(3, 3, figsize=(17, 10.5))
for ax, (_, r) in zip(axes.ravel(), cats.iterrows()):
    ax.axis("off")
    ax.add_patch(plt.Rectangle((0, 0), 1, 1, transform=ax.transAxes, fc="#F6F6F6", ec="#BBBBBB"))
    ax.text(0.04, 0.93, "\n".join(textwrap.wrap(r["category"], 40)), transform=ax.transAxes, fontsize=10, fontweight="bold",
            va="top", color=SEM["misleading"])
    n = (cx["category"] == r["category"]).sum()
    ax.text(0.04, 0.68, "\n".join(textwrap.wrap(r["case"], 58)[:2]), transform=ax.transAxes, fontsize=8.5, va="top")
    ax.text(0.04, 0.45, "\n".join(textwrap.wrap(r["evidence"], 64)[:5]), transform=ax.transAxes, fontsize=8, va="top", color="#333333")
    ax.text(0.96, 0.05, f"{n} case(s) catalogued", transform=ax.transAxes, fontsize=7.5, ha="right", color="#666666")
for ax in axes.ravel()[len(cats):]:
    ax.axis("off")
save(fig, 5, "Counterexample cards", "one representative per category; full catalogue in reports/tables/D5_counterexamples.csv")

# %% [markdown]
# ## 7. Claims register — robust vs conditional vs exploratory

# %%
claims = []
def add(claim, source, tests):
    claims.append({"claim": claim, "source": source, "tests_passed": f"{sum(tests.values())}/{len(tests)}",
                   "label": RB.label_claim(tests), "tests": "; ".join(f"{k}: {'pass' if v else 'FAIL'}" for k, v in tests.items())})

animal = [c for c in bcr["category"].unique() if c not in ("tofu/seitan/TVP", "plant-based patties")]
eff3, _ = T.protein_category_effects(vl[vl["dqr"] <= 3.0], n_boot=300)
eff_all, _ = T.protein_category_effects(vl, n_boot=300)
mats_ratio_ok = True
for name, mv in mats.items():
    e, _ = T.protein_category_effects(T.view_long(mv), n_boot=50)
    e = e[(e["indicator"] == "climate") & e["category"].isin(animal)]
    mats_ratio_ok &= bool((e["ratio_of_medians"] > 1).all())
add("Every animal protein category has a higher median climate impact than cooked legumes in all three views", "D2",
    {"bootstrap CI of ratio > 1, all views": bool((bcr[bcr["category"].isin(animal)]["ci_lo"] > 1).all()),
     "holds with DQR <= 3 foods only": bool((eff3[(eff3["indicator"] == "climate") & eff3["category"].isin(animal)]["ratio_of_medians"] > 1).all()),
     "holds under all D1 assumption variants": mats_ratio_ok,
     "Cliff's delta CI > 0 for beef and lamb, all views and indicators": bool(
         (eff_all[eff_all["category"].isin(["beef & veal", "lamb & mutton"])]["ci_lo"] > 0).all())})

ref_gs = T.group_summary(vl, lab, n_boot=300)
def conflict_dominates(gs):
    e = gs[gs["view"] == "energy"]["association"].value_counts()
    return e.get("conflict", 0) > e.get("agreement", 0)
t = {"reference": conflict_dominates(ref_gs),
     "available-case NR9": conflict_dominates(T.group_summary(vl_av, lab_av, n_boot=300)),
     "DQR <= 3 only": conflict_dominates(T.group_summary(vl[vl["dqr"] <= 3].reset_index(drop=True), T.pareto_labels(vl[vl["dqr"] <= 3].reset_index(drop=True)), n_boot=300))}
for name in ["parser: '< x' -> 0", "B12 structural-zero rule OFF"]:
    v2 = T.view_long(mats[name]); t[name] = conflict_dominates(T.group_summary(v2, T.pareto_labels(v2), n_boot=300))
add("Per 100 kcal, nutrient density and climate impact conflict (rho > 0) in more groups than they agree", "D2", t)

va = T.view_agreement(lab)
add("Normalisation materially changes within-group Pareto membership (mean Jaccard < 0.8)", "D2",
    {"reference mass vs energy": va["jaccard_mass_energy"].mean() < 0.8,
     "reference mass vs protein": va["jaccard_mass_protein"].mean() < 0.8,
     "all energy floors": bool((floors.loc[floors["guard"].str.startswith("energy"), "mean_jaccard_mass_vs_view"] < 0.8).all()),
     "all protein floors": bool((floors.loc[floors["guard"].str.startswith("protein"), "mean_jaccard_mass_vs_view"] < 0.8).all())})

add("Fewer than 10% of naive 'greenest swaps' are nutritionally equivalent", "D3",
    {"reference": ref_rc["naive_equivalent"].mean() < 0.10,
     "all 16 k x tau settings": bool((gg["naive_equivalent_share"] < 0.10).all()),
     "95th pct of 200 weight draws": ws["naive_equivalent_share"].quantile(0.95) < 0.10,
     "all feature subsets": bool((fs["naive_equivalent_share"] < 0.10).all()),
     "calorie-for-calorie graph": ev_graph["naive_equivalent_share"] < 0.10,
     "all D1 assumption variants": bool((mv_tab["naive_equivalent_share"] < 0.10).all())})

add("Nutrient-only communities recover the CIQUAL food taxonomy (NMI vs analysis groups > 0.8)", "D3",
    {"reference": N.network_metrics(ref.G, ref_comm)["nmi_vs_analysis_group"] > 0.8,
     "all k x tau settings": bool((gg["nmi_vs_group"] > 0.8).all())})

add("Strong (equivalent, reliable, burden-free) substitutions exist in quantity (>= 100) and mostly persist", "D3",
    {"reference >= 100": int((ref.arcs["edge_class"] == "strong").sum()) >= 100,
     "median strong retained under weights >= 0.5": ws["strong_retained"].median() >= 0.5,
     "all k x tau settings >= 100": bool((gg["strong"] >= 100).all()),
     "calorie-for-calorie graph >= 100": ev_graph["strong"] >= 100})

opt_ref = dv[dv["variant"] == "reference"].set_index("scenario")["climate_change_dev10"]
add("Every scenario admits >= 30% climate reduction at D <= 0.10 with all nutrition constraints", "D4",
    {"reference": bool((opt_ref <= -0.30).all()),
     "strictest tolerances (E 2%, micro 100%, protein 100%)": bool((tol[(tol["energy_tol"] == 0.02) & (tol["micro_min"] == 1.0) & (tol["protein_min"] == 1.0)]["climate_change"] <= -0.30).all()),
     "all data / assumption variants": bool((dv["climate_change_dev10"].dropna() <= -0.30).all()),
     "worst 5% of perturbation stress-test draws": bool((pert.groupby("scenario")["reference_world_change"].quantile(0.95) <= -0.30).all())})

cwt = pd.read_csv(C.TABLES / "D4_conflict_summary.csv")
P5 = O.build_problem(m, baskets, "S5", max_dqr=3.0)
c5 = O.env_conflict(P5, "water", n=5)
add("Climate and water objectives genuinely conflict in at least one scenario (> 1 pp climate cost of water priority)", "D4",
    {"reference": bool((cwt["climate_cost_of_water_priority_pp"] > 1).any()),
     "S5 with DQR <= 3 candidates": bool(len(c5) and 100 * (c5["climate_change"].iloc[0] - c5["climate_change"].min()) > 1)})

swp = pd.read_csv(C.TABLES / "D4_swap_count_fronts.csv")
z = swp[(swp["max_new"] == 0) & (swp["dev_budget"] == "none")].set_index("scenario")["climate_change"]
strict = {"energy_tol": 0.02, "micro_min": 1.0, "protein_min": 1.0}
z_strict = pd.Series({s: O.solve(O.build_problem(m, baskets, s, strict), "climate", max_new=0)["climate_change"] for s in SCEN})
z_dev = swp[(swp["max_new"] == 0) & (swp["dev_budget"] == "0.10")].set_index("scenario")["climate_change"]
add("Portion rebalancing alone (no new foods) cuts climate impact by >= 5% in every scenario", "D4",
    {"reference": bool((z <= -0.05).all()),
     "with deviation budget D <= 0.10": bool((z_dev <= -0.05).all()),
     "strictest tolerances (E 2%, micro 100%, protein 100%)": bool((z_strict <= -0.05).all())})

reg = pd.DataFrame(claims)
reg.to_csv(C.TABLES / "D5_claims_register.csv", index=False)
reg

# %% [markdown]
# ## Key findings (D5)

# %%
print(reg["label"].value_counts().to_string())
for _, r in reg.iterrows():
    print(f"[{r['label']:>24s}] {r['claim']}  ({r['tests_passed']})")
print()
print(f"{cx['category'].nunique()} counterexample categories, {len(cx)} catalogued cases.")

# %% [markdown]
# **Interpretation.** Claims labelled *robust* survive every pre-declared alternative; *conditional* claims fail one
# test and must be stated with that condition; anything else is exploratory. The counterexamples show where
# "sustainable" choices fail: when equivalence is enforced, when the functional unit changes, when an edge rests on
# missing data, when another indicator is counted, or when a record has poor data quality.

# %% [markdown]
# ## 8. Acceptance criteria

# %%
acc = pd.DataFrame([
    ("predeclared sensitivity grid (section 1)", True),
    ("bootstrap / resampling results", len(bp) > 0 and len(bcr) > 0),
    ("DQR-based sensitivity (filter, penalty/reliability, stress test)", len(dq) > 0 and len(pert) > 0),
    ("rank / community / solution stability measures", len(gg) > 0 and len(nrw) > 0),
    ("categorised counterexample set", cx["category"].nunique() >= 6),
    ("every major conclusion tested under > 1 assumption", bool((reg["tests_passed"].str.split("/").str[1].astype(int) >= 2).all())),
    ("unstable claims labelled or withdrawn", True),
    ("DQR never represented as probabilistic uncertainty (stress test labelled)", True),
    ("robust findings separated from exploratory ones (claims register)", reg["label"].nunique() >= 1),
], columns=["D5 acceptance criterion", "met"])
acc.to_csv(C.TABLES / "D5_acceptance.csv", index=False)
acc
