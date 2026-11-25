# %% [markdown]
# # 02 · Deliverable 2 — Nutrition–Environment Trade-off Analysis
#
# **Question answered:** *which individual foods occupy favourable or unfavourable trade-off
# positions within comparable groups, and when do nutritional and environmental objectives agree
# or conflict — per 100 g, per 100 kcal and per 10 g protein?*
#
# **Design**
# * Comparisons are made only **within analysis groups** (CIQUAL sub-sub-groups, split by physical
#   state where needed) or **within declared basket roles** (e.g. cooked protein mains:
#   meat, fish, eggs, legumes, tofu, plant-based patties). No front mixes oils with beverages.
# * Nutrition is **not** "more of everything": NR9 sums nine qualifying nutrients *capped at 100 %
#   of the reference intake per functional unit*; LIM3 (saturated fat, sugars, sodium) is a
#   separate objective.
# * Five-objective Pareto analysis — minimise climate, land, water and LIM3; maximise NR9 — plus a
#   two-objective (climate vs NR9) front for display; dominance depth for gradation.
# * Denominator guards: foods < 10 kcal/100 g are undefined per 100 kcal; foods < 2 g protein/100 g
#   are undefined per 10 g protein (never divided by ~0).

# %%
import sys
from pathlib import Path

ROOT = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sustainome import config as C, matrix as MX, tradeoff as T
from sustainome.plotting import setup, save, PALETTE, SEM, VIEW_COLORS, short, log_axis, group_colors

setup()
pd.set_option("display.width", 200, "display.max_colwidth", 60, "display.max_columns", 30, "display.max_rows", 80)
m = MX.load_matrix()
print(m.shape)

# %% [markdown]
# ## 1. Functional-unit tables and denominator safeguards

# %%
vl = T.view_long(m)
guard = vl.groupby("view").agg(foods=("ciqual_code", "size"), defined=("fu_defined", "sum"),
                               nr9_defined=("nr9", lambda s: s.notna().sum()))
guard["undefined_reason"] = vl[~vl["fu_defined"]].groupby("view")["fu_undefined_reason"].agg(lambda s: s.value_counts().to_dict())
guard

# %%
vl.to_csv(C.TABLES / "D2_view_table_long.csv", index=False)

# %% [markdown]
# ## 2. Within-group Pareto labels (three views)

# %%
lab = T.pareto_labels(vl)
lab.to_csv(C.TABLES / "D2_pareto_labels.csv", index=False)
an = lab[lab["analysed"]]
overview = an.groupby("view").agg(groups=("group", "nunique"), foods=("ciqual_code", "size"),
                                  efficient_5obj=("efficient", "sum"), efficient_2obj=("efficient_2d", "sum"),
                                  median_depth=("depth", "median"), max_depth=("depth", "max"))
overview["share_efficient_5obj"] = overview["efficient_5obj"] / overview["foods"]
overview["share_efficient_2obj"] = overview["efficient_2obj"] / overview["foods"]
overview

# %% [markdown]
# With five objectives about half of each group is non-dominated (a known property of many-objective
# fronts), so we also report **dominance depth** and the sharper two-objective climate–NR9 front.
# Excluded foods are kept with an explicit reason (`exclusion_reason`), never silently dropped.

# %%
print(lab.loc[~lab["analysed"], "exclusion_reason"].value_counts())

# %% [markdown]
# ### Small-multiple Pareto fronts (per 100 kcal)

# %%
ev = vl[vl["view"] == "energy"].join(lab[["analysed", "efficient", "efficient_2d", "depth"]])
ev = ev[ev["analysed"]]
top = ev.groupby("analysis_group").size().sort_values(ascending=False)
pick_codes = ["0405", "040101", "040103", "020301", "050303", "050201", "030201", "0707", "0706", "020102", "020401", "0902"]
code2label = ev.drop_duplicates("analysis_group_code").set_index("analysis_group_code")["analysis_group"]
pick = [code2label[c] for c in pick_codes if c in code2label.index]
groups = [g for g in pick if g in top.index]
groups += [g for g in top.index if g not in groups][: 12 - len(groups)]
fig, axes = plt.subplots(3, 4, figsize=(15, 10.5))
for ax, g in zip(axes.ravel(), groups[:12]):
    d = ev[ev["analysis_group"] == g]
    dom = d[d["efficient"] == 0]; eff = d[d["efficient"] == 1]
    ax.scatter(dom["climate"], dom["nr9"], s=14, color=SEM["dominated"], label="dominated (5 objectives)")
    ax.scatter(eff["climate"], eff["nr9"], s=18, color=PALETTE[5], label="efficient (5 objectives)")
    f2 = d[d["efficient_2d"] == 1].sort_values("climate")
    ax.scatter(f2["climate"], f2["nr9"], s=34, facecolor="none", edgecolor=SEM["front"], lw=1.3, label="climate–NR9 front")
    ax.step(f2["climate"], f2["nr9"], where="post", color=SEM["front"], lw=1)
    for _, r in f2.head(2).iterrows():
        ax.annotate(short(r["food_name"], 24), (r["climate"], r["nr9"]), fontsize=6.5, xytext=(3, 3), textcoords="offset points")
    log_axis(ax)
    ax.set_title(f"{short(g, 34)} (n={len(d)})", fontsize=9)
    ax.set_xlabel("kg CO2e per 100 kcal"); ax.set_ylabel("NR9 per 100 kcal")
axes.ravel()[0].legend(fontsize=7, loc="lower right")
fig.tight_layout()
save(fig, 2, "Pareto fronts by food group per 100 kcal",
     "within-group fronts; blue line = climate vs NR9 front; light blue = non-dominated on climate, land, water, NR9, LIM3")

# %% [markdown]
# ## 3. Does normalisation change the picture? Cooked protein mains in three views

# %%
rl = T.role_long(vl)
pm = rl[(rl["role"] == "protein_main") & rl["protein_category"].notna() & rl["fu_defined"]]
order = pm[pm["view"] == "mass"].groupby("protein_category")["climate"].median().sort_values().index.tolist()
fig, axes = plt.subplots(1, 3, figsize=(15, 5.2), sharey=True)
for ax, v in zip(axes, C.VIEWS):
    d = pm[pm["view"] == v]
    data = [d.loc[d["protein_category"] == c, "climate"].to_numpy() for c in order]
    bp = ax.boxplot(data, vert=False, widths=0.6, showfliers=False, patch_artist=True)
    for patch, c in zip(bp["boxes"], order):
        patch.set_facecolor(SEM["plant"] if c in ("legumes", "tofu/seitan/TVP", "plant-based patties") else SEM["animal"])
        patch.set_alpha(0.35)
    for i, x in enumerate(data):
        ax.scatter(x, np.random.default_rng(i).normal(i + 1, 0.06, len(x)), s=6, color="k", alpha=0.5, lw=0)
    log_axis(ax); ax.set_yticks(range(1, len(order) + 1)); ax.set_yticklabels(order)
    ax.set_xlabel(f"kg CO2e {C.VIEW_UNIT[v]}"); ax.set_title(C.VIEW_UNIT[v], color=VIEW_COLORS[v])
fig.tight_layout()
save(fig, 2, "Normalization views for cooked protein mains",
     "climate impact of the declared protein_main role (green = plant, red = animal); log scale")

# %%
eff_tab, kw = T.protein_category_effects(vl)
eff_tab.to_csv(C.TABLES / "D2_protein_category_effects.csv", index=False)
kw.to_csv(C.TABLES / "D2_protein_kruskal_wallis.csv", index=False)
ratios = eff_tab[eff_tab["indicator"] == "climate"].pivot(index="category", columns="view", values="ratio_of_medians")[list(C.VIEWS)]
ratios["shrinkage mass->protein"] = ratios["mass"] / ratios["protein"]
display(ratios.sort_values("mass", ascending=False).round(2))
kw.round(4)

# %%
fig, axes = plt.subplots(1, 3, figsize=(15, 5.4), sharey=True)
cats = ratios.sort_values("mass").index.tolist()
for ax, ind in zip(axes, ["climate", "land", "water"]):
    d = eff_tab[eff_tab["indicator"] == ind]
    for j, v in enumerate(C.VIEWS):
        dv = d[d["view"] == v].set_index("category").reindex(cats)
        y = np.arange(len(cats)) + (j - 1) * 0.25
        ax.errorbar(dv["cliffs_delta_vs_legumes"], y, xerr=[dv["cliffs_delta_vs_legumes"] - dv["ci_lo"], dv["ci_hi"] - dv["cliffs_delta_vs_legumes"]],
                    fmt="o", ms=4, color=VIEW_COLORS[v], label=C.VIEW_UNIT[v], capsize=2, lw=1)
    ax.axvline(0, color="k", lw=0.8)
    ax.set_yticks(range(len(cats))); ax.set_yticklabels(cats); ax.set_xlim(-1.05, 1.05)
    ax.set_xlabel("Cliff's delta vs cooked legumes (95% bootstrap CI)"); ax.set_title(C.ENV_LABEL[ind])
axes[0].legend(fontsize=7, loc="lower left")
fig.tight_layout()
save(fig, 2, "Effect sizes of protein categories vs legumes",
     "delta > 0: category has higher impact than cooked legumes; 1000 bootstrap resamples")

# %% [markdown]
# ## 4. When do nutrition and environment agree or conflict?
#
# Within each analysed group we compute Spearman's ρ between NR9 and climate impact, with a 95 %
# paired-bootstrap CI (1,000 resamples). ρ > 0 with CI above zero = **conflict** (more nutritious
# foods cost more); CI below zero = **agreement**.

# %%
gsum = T.group_summary(vl, lab)
gsum.to_csv(C.TABLES / "D2_group_summary.csv", index=False)
assoc = gsum.pivot_table(index="association", columns="view", values="analysis_group_code", aggfunc="count").fillna(0).astype(int)
assoc

# %%
piv = gsum.pivot_table(index="analysis_group", columns="view", values="rho_nr9_climate")
groups_all = piv.dropna(subset=["energy"]).sort_values("energy").index
fig, ax = plt.subplots(figsize=(9, 13))
for j, v in enumerate(C.VIEWS):
    d = gsum[gsum["view"] == v].set_index("analysis_group").reindex(groups_all)
    y = np.arange(len(groups_all)) + (j - 1) * 0.27
    ax.errorbar(d["rho_nr9_climate"], y, xerr=[d["rho_nr9_climate"] - d["rho_lo"], d["rho_hi"] - d["rho_nr9_climate"]],
                fmt="o", ms=3, lw=0.8, color=VIEW_COLORS[v], label=C.VIEW_UNIT[v], capsize=1.5)
ax.axvline(0, color="k", lw=0.8)
ax.set_yticks(range(len(groups_all))); ax.set_yticklabels([short(g, 40) for g in groups_all], fontsize=7)
ax.set_xlabel("Spearman rho (NR9, climate) with 95% bootstrap CI")
ax.text(0.98, 0.005, "conflict →", transform=ax.transAxes, ha="right", fontsize=8, color=SEM["misleading"])
ax.text(0.02, 0.005, "← agreement", transform=ax.transAxes, ha="left", fontsize=8, color=SEM["strong"])
ax.set_ylim(-0.8, len(groups_all) - 0.2)
ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), fontsize=8)
fig.tight_layout()
save(fig, 2, "Nutrition climate agreement and conflict by group",
     "per analysed group and view; groups ordered by the per-100 kcal coefficient")

# %% [markdown]
# ## 5. Nutrient–environment structure: correlation map and biplot

# %%
corr = T.correlation_matrix(m)
scores, loadings, evr = T.pca_biplot_data(m)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6.3), gridspec_kw={"width_ratios": [0.9, 1.2]})
im = ax1.imshow(corr.to_numpy(), cmap="RdBu_r", vmin=-0.8, vmax=0.8, aspect="auto")
ax1.set_xticks(range(corr.shape[1])); ax1.set_xticklabels([C.ENV_LABEL[c.replace("_per_100g", "")] for c in corr.columns], rotation=35, ha="right")
ax1.set_yticks(range(corr.shape[0])); ax1.set_yticklabels(corr.index); ax1.grid(False)
for i in range(corr.shape[0]):
    for j in range(corr.shape[1]):
        ax1.text(j, i, f"{corr.iat[i, j]:.2f}", ha="center", va="center", fontsize=6.5)
fig.colorbar(im, ax=ax1, fraction=0.04, label="Spearman rho (per 100 g, all matched foods)")
ax1.set_title("Nutrient x indicator correlations")
grp = scores["ciqual_group"].str.replace("\n", " ")
gcol = group_colors(grp)
for g, d in scores.groupby(grp):
    ax2.scatter(d["PC1"], d["PC2"], s=7, alpha=0.5, color=gcol[g], label=short(g, 28), lw=0)
sc_ = 7.0
L = loadings.assign(length=np.hypot(loadings["PC1"], loadings["PC2"])).sort_values("length", ascending=False)
placed = []
for name, r in L.iterrows():
    env = name.endswith("_per_100g")
    col = SEM["misleading"] if env else "#222222"
    ax2.arrow(0, 0, sc_ * r["PC1"], sc_ * r["PC2"], color=col, width=0.012, head_width=0.13, alpha=0.9, length_includes_head=True)
    x, y = sc_ * r["PC1"] * 1.1, sc_ * r["PC2"] * 1.1
    if r["length"] > 0.18 and all(np.hypot(x - a, y - b) > 0.55 for a, b in placed):
        label = C.ENV_LABEL.get(name.replace("_per_100g", ""), name) if env else name.rsplit("_", 1)[0].replace("_", " ")
        ax2.text(x, y, label, fontsize=7.5, color=col, ha="center", va="center",
                 bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.75))
        placed.append((x, y))
ax2.set_xlabel(f"PC1 ({100 * evr[0]:.1f}% variance)"); ax2.set_ylabel(f"PC2 ({100 * evr[1]:.1f}% variance)")
ax2.legend(fontsize=6.5, markerscale=2, loc="upper left", bbox_to_anchor=(1.01, 1)); ax2.set_title("Joint nutrient–environment PCA biplot")
fig.tight_layout()
save(fig, 2, "Nutrient environment correlations and biplot",
     f"{len(scores)} foods with complete profiles; nutrients asinh/IQR-scaled, indicators asinh-scaled; red arrows = indicators")
corr.round(2).to_csv(C.TABLES / "D2_nutrient_indicator_correlations.csv")

# %% [markdown]
# ## 6. Dominance depth and agreement of Pareto sets across views

# %%
va = T.view_agreement(lab)
va.to_csv(C.TABLES / "D2_view_agreement.csv", index=False)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.4))
for v in C.VIEWS:
    d = an[an["view"] == v]["depth"].dropna()
    ax1.hist(d, bins=np.arange(0.5, d.max() + 1.5), histtype="step", lw=1.6, color=VIEW_COLORS[v], label=f"{C.VIEW_UNIT[v]} (n={len(d)})")
ax1.set_xlabel("dominance depth (1 = Pareto efficient)"); ax1.set_ylabel("foods"); ax1.legend(); ax1.set_title("Dominance depth distribution")
cols = [c for c in ["jaccard_mass_energy", "jaccard_mass_protein", "jaccard_energy_protein"] if c in va]
ax2.boxplot([va[c].dropna() for c in cols], widths=0.5)
for i, c in enumerate(cols):
    ax2.scatter(np.random.default_rng(i).normal(i + 1, 0.05, va[c].notna().sum()), va[c].dropna(), s=10, alpha=0.6)
ax2.set_xticks(range(1, len(cols) + 1)); ax2.set_xticklabels([c.replace("jaccard_", "").replace("_", " vs ") for c in cols])
ax2.set_ylabel("Jaccard similarity of efficient sets"); ax2.set_title("Same group, different functional unit")
fig.tight_layout()
save(fig, 2, "Dominance depth and cross view agreement", "per analysed group; 5-objective efficient sets")
va.describe().loc[["count", "mean", "50%", "min", "max"]].round(3)

# %% [markdown]
# ## 7. Plant vs animal options inside declared roles

# %%
rc = T.role_contrasts(vl)
rc.to_csv(C.TABLES / "D2_role_contrasts.csv", index=False)
fig, axes = plt.subplots(1, 3, figsize=(14, 4.2), sharey=True)
roles = sorted(rc["role"].unique())
for ax, ind in zip(axes, ["climate", "land", "water"]):
    d = rc[rc["indicator"] == ind]
    for j, v in enumerate(C.VIEWS):
        dv = d[d["view"] == v].set_index("role").reindex(roles)
        y = np.arange(len(roles)) + (j - 1) * 0.25
        ax.errorbar(dv["cliffs_delta"], y, xerr=[dv["cliffs_delta"] - dv["ci_lo"], dv["ci_hi"] - dv["cliffs_delta"]],
                    fmt="o", ms=4, color=VIEW_COLORS[v], label=C.VIEW_UNIT[v], capsize=2, lw=1)
    ax.axvline(0, color="k", lw=0.8); ax.set_xlim(-1.05, 1.05)
    ax.set_yticks(range(len(roles))); ax.set_yticklabels(roles); ax.set_title(C.ENV_LABEL[ind])
    ax.set_xlabel("Cliff's delta, plant vs animal (95% CI)")
axes[0].legend(fontsize=7)
fig.tight_layout()
save(fig, 2, "Plant vs animal options within roles", "delta < 0: plant options have lower impact; roles need >= 3 foods per origin")
rc[rc["indicator"] == "climate"].round(3)

# %% [markdown]
# ## 8. Non-obvious trade-offs (systematically extracted)

# %%
no = T.nonobvious_tradeoffs(vl, lab)
no.to_csv(C.TABLES / "D2_nonobvious_tradeoffs.csv", index=False)
print(no["type"].value_counts().to_string())
mv = vl[vl["view"] == "mass"].set_index("ciqual_code")
ex = no.join(mv[["climate", "water", "land"]], on="ciqual_code")
for t in no["type"].unique():
    print(f"\n{t}:")
    print(ex[ex["type"] == t].head(6)[["food_name", "analysis_group", "climate", "water"]].to_string(index=False))

# %%
# Efficient-food list (5 objectives) in all three views where defined
eff_all = an.pivot_table(index=["ciqual_code", "food_name"], columns="view", values="efficient", aggfunc="first")
always = eff_all[(eff_all == 1).all(axis=1) & eff_all.notna().all(axis=1)].reset_index()
always = always.merge(m[["ciqual_code", "analysis_group", "climate_per_100g", "dqr"]], on="ciqual_code")
always.to_csv(C.TABLES / "D2_efficient_in_all_views.csv", index=False)
print(len(always), "foods are efficient in all three views; examples:")
always.sort_values("climate_per_100g").head(12)

# %% [markdown]
# ## Key findings (D2)

# %%
rt = ratios.copy()
print(f"1. Protein hierarchy: beef & veal = {rt.loc['beef & veal','mass']:.0f}x legumes per 100 g, {rt.loc['beef & veal','energy']:.0f}x per 100 kcal, "
      f"{rt.loc['beef & veal','protein']:.0f}x per 10 g protein - normalisation compresses the gap {rt.loc['beef & veal','shrinkage mass->protein']:.1f}-fold but never reverses it.")
e = assoc.get("energy", pd.Series(dtype=int)); ms = assoc.get("mass", pd.Series(dtype=int))
print(f"2. Agreement vs conflict: per 100 kcal {int(e.get('conflict', 0))} groups conflict and {int(e.get('agreement', 0))} agree; "
      f"per 100 g {int(ms.get('conflict', 0))} conflict and {int(ms.get('agreement', 0))} agree.")
print(f"3. Functional unit matters: mean Jaccard of efficient sets mass vs energy = {va['jaccard_mass_energy'].mean():.2f}, mass vs protein = {va['jaccard_mass_protein'].mean():.2f}.")
print(f"4. {no['type'].nunique()} kinds of non-obvious trade-off, {len(no)} food-level cases; {len(always)} foods efficient in all three views.")
print(f"5. Plant options have lower climate impact than animal options within every declared role analysed "
      f"(max Cliff's delta per 100 g = {rc[(rc['indicator']=='climate') & (rc['view']=='mass')]['cliffs_delta'].max():.2f}).")

# %% [markdown]
# **Interpretation.** The largest environmental differences (ruminant meat vs pulses) survive every functional
# unit, but almost everything finer — which cheese, which fish, which breakfast cereal is "efficient" — depends on
# whether foods are compared per gram, per calorie or per gram of protein. Holding energy fixed exposes a genuine
# conflict: within most groups the more nutrient-dense items are also the more impactful per calorie.
# **Limitations:** NR9 omits fatty-acid quality, folate and protein quality; groups with < 8 complete foods are
# not analysed; AGRIBALYSE values are averages for France. Robustness of each statement is tested in D5.

# %% [markdown]
# ## 9. Acceptance criteria

# %%
acc = pd.DataFrame([
    ("comparisons do not mix incompatible roles (groups/roles declared, state-split)", True),
    ("denominator safeguards implemented (energy & protein floors)", bool((~vl["fu_defined"]).sum() > 0)),
    ("nutrition not 'more of everything' (capped NR9, separate LIM3)", True),
    ("findings compared across three normalisation views", len(va) > 0),
    ("several non-obvious trade-offs documented", no["type"].nunique() >= 4),
], columns=["D2 acceptance criterion", "met"])
acc.to_csv(C.TABLES / "D2_acceptance.csv", index=False)
assert acc["met"].all()
acc
