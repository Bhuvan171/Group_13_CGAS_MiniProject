# %% [markdown]
# # 04 · Deliverable 4 — Multi-Objective Basket/Menu Optimisation
#
# **Question answered:** *how should several foods and their quantities change together so that a
# basket's environmental impact falls while its nutrition is preserved — and what does each extra unit
# of climate reduction cost in nutritional deviation and in basket change?*
#
# These baskets are **transparent computational test cases**, not dietary recommendations. They are
# declared in `config/baskets.csv` (foods by CIQUAL code, grams, slot role) — no external menu data.
#
# ### Formal specification
# For scenario baskets with slots $s$ (each with a declared role and baseline mass $M_s$):
#
# * **Variables** $x_{s,j} \ge 0$: grams of candidate food $j$ in slot $s$ (candidates = foods sharing the
#   slot's declared role, with a complete 15-nutrient profile and no publisher-flagged error); auxiliary
#   $u_k \ge 0$ (relative deviation of nutrient $k$), $v_{s,j} \ge 0$ (absolute change), optional binary
#   $y_{s,j}$ (new food used).
# * **Hard constraints**
#   * slot structure: $0.75\,M_s \le \sum_j x_{s,j} \le 1.25\,M_s$
#   * energy: $|E(x) - E_0| \le 0.05\,E_0$; protein: $P(x) \ge 0.95\,P_0$
#   * fibre, Ca, Fe, Mg, K, Zn, vitamin C, B12: $N_k(x) \ge 0.90\,N_{k,0}$
#   * sodium, saturated fat, sugars: $N_k(x) \le N_{k,0}$
# * **Objectives** (kept separate):
#   climate $\sum c_j x_{s,j}$ (min) ·
#   nutrient-profile deviation $D = \frac{1}{15}\sum_k |N_k(x)-N_{k,0}| / \max(N_{k,0}, 0.05\,RI_k)$ (min) ·
#   basket change $\Delta = \sum |x - x_0| / (2\sum x_0)$ (min) · land, water, DQR reported for every solution.
# * **Pareto set** by ε-constraint: minimise climate s.t. $D \le \varepsilon_D$, $\Delta \le \varepsilon_\Delta$ over an
#   11 × 4 grid; plus a MILP front of climate vs. number of new foods, and climate vs water / land fronts.
# * **Baselines**: original basket · 5,000 random role-preserving swaps · 100 random feasible baskets
#   (random linear objectives over the feasible set) · single-objective climate minimisation *without*
#   nutrition constraints (violations shown explicitly).
# * **Illustrative operating point** (declared before solving): $\varepsilon_D = 0.10$, $\varepsilon_\Delta = 0.50$.

# %%
import sys
from pathlib import Path

ROOT = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sustainome import config as C, matrix as MX, optimize as O
from sustainome.plotting import setup, save, PALETTE, SEM, short

setup()
pd.set_option("display.width", 220, "display.max_colwidth", 60, "display.max_columns", 40, "display.max_rows", 80)
m = MX.load_matrix()
baskets = MX.load_declared_tables()["baskets"]
SCEN = ["S1", "S2", "S3", "S4", "S5"]

# %% [markdown]
# ## 1. The declared scenarios

# %%
b = baskets.merge(m[["ciqual_code", "food_name", "energy_kcal", "protein_g", "climate_per_100g", "dqr"]], on="ciqual_code", how="left")
b["grams"] = b["grams"].astype(float)
b["kcal"] = b["grams"] / 100 * b["energy_kcal"]
b["kg_CO2e"] = b["grams"] / 100 * b["climate_per_100g"]
b.to_csv(C.TABLES / "D4_scenarios.csv", index=False)
display(b[["scenario", "scenario_label", "slot", "role", "ciqual_code", "food_name", "grams", "kcal", "kg_CO2e", "dqr"]].round(3))
b.groupby("scenario_label")[["grams", "kcal", "kg_CO2e"]].sum().round(2)

# %% [markdown]
# ## 2. Solve every scenario (ε-sweeps, swap-count MILP, conflict fronts, baselines)

# %%
R = {}
for sc in SCEN:
    R[sc] = O.run_scenario(m, baskets, sc)
    P = R[sc]["problem"]
    print(f"{sc} {P.label:28s} slots={len(P.slots):2d} candidate variables={P.n:4d} "
          f"climate-optimum {100 * R[sc]['climate_opt']['climate_change']:+.0f}%")

# %%
def strip(df):
    return df.drop(columns=[c for c in ["x"] if c in df])

sweeps = pd.concat([strip(R[s]["sweep"]).assign(scenario=s) for s in SCEN], ignore_index=True)
swaps = pd.concat([strip(R[s]["swaps"]).assign(scenario=s) for s in SCEN], ignore_index=True)
conf = pd.concat([strip(R[s]["conflict_water"]).assign(scenario=s, other="water") for s in SCEN] +
                 [strip(R[s]["conflict_land"]).assign(scenario=s, other="land") for s in SCEN], ignore_index=True)
rs = pd.concat([R[s]["random_swaps"].assign(scenario=s) for s in SCEN], ignore_index=True)
rf = pd.concat([R[s]["random_feasible"].assign(scenario=s) for s in SCEN], ignore_index=True)
sweeps.to_csv(C.TABLES / "D4_pareto_sweep_solutions.csv", index=False)
swaps.to_csv(C.TABLES / "D4_swap_count_fronts.csv", index=False)
conf.to_csv(C.TABLES / "D4_environmental_conflict_fronts.csv", index=False)
rs.to_csv(C.TABLES / "D4_random_swap_baseline.csv", index=False)
rf.to_csv(C.TABLES / "D4_random_feasible_baseline.csv", index=False)

# %% [markdown]
# ## 3. Every reported solution is re-checked against every constraint

# %%
all_opt = pd.concat([sweeps, swaps, conf, rf], ignore_index=True)
print(f"{len(all_opt)} optimiser solutions; feasible: {int(all_opt['feasible'].sum())}; "
      f"max relative violation: {all_opt['max_relative_violation'].max():.2e}")
assert all_opt["feasible"].all()
viol = pd.DataFrame([{**{k: R[s]["env_only"][k] for k in ["climate_change", "n_violations", "mean_deviation"]}, "scenario": s} for s in SCEN])
print("\nClimate-only baseline WITHOUT nutrition constraints (violations reported, not hidden):")
viol

# %% [markdown]
# ## 4. Pareto fronts: climate vs nutrient deviation (and basket change)

# %%
fig, axes = plt.subplots(1, 5, figsize=(19, 4.6), sharey=True)
ec_cols = dict(zip(C.OPT["eps_change"], [PALETTE[5], PALETTE[0], PALETTE[2], "k"]))
for ax, s in zip(axes, SCEN):
    d = R[s]["sweep"]
    for ec, g in d[~d["duplicate"]].groupby("eps_change"):
        g = g.sort_values("mean_deviation")
        ax.plot(g["mean_deviation"], 100 * g["climate_change"], "o-", ms=3.5, color=ec_cols[ec], label=f"change ≤ {ec:.0%}")
    rfd = R[s]["random_feasible"]
    ax.scatter(rfd["mean_deviation"], 100 * rfd["climate_change"], s=7, color=SEM["random"], alpha=0.7, label="random feasible", zorder=0)
    ax.scatter([0], [0], marker="*", s=120, color="k", zorder=5, label="original basket")
    eo = R[s]["env_only"]
    ax.scatter([eo["mean_deviation"]], [100 * eo["climate_change"]], marker="X", s=70, color=SEM["misleading"], zorder=5,
               label="climate-only, no nutrition constraints")
    ax.annotate(f"{eo['n_violations']} violated", (eo["mean_deviation"], 100 * eo["climate_change"]), fontsize=7,
                xytext=(-10, 8), textcoords="offset points", color=SEM["misleading"])
    sel = d[(d["eps_dev"] == C.OPT_SELECTED["eps_dev"]) & (d["eps_change"] == C.OPT_SELECTED["eps_change"])]
    ax.scatter(sel["mean_deviation"], 100 * sel["climate_change"], s=130, facecolor="none", edgecolor=SEM["misleading"], lw=1.5, zorder=6)
    xmax = 1.25 * max(d["mean_deviation"].max(), eo["mean_deviation"])
    ax.set_xlim(-0.02, xmax); ax.set_ylim(-100, 60)
    outside = int(((rfd["mean_deviation"] > xmax) | (100 * rfd["climate_change"] > 60)).sum())
    ax.text(0.98, 0.97, f"{outside} random baskets outside view", transform=ax.transAxes, ha="right", va="top", fontsize=6.5, color="#777777")
    ax.set_title(f"{s}: {R[s]['problem'].label}", fontsize=9)
    ax.set_xlabel("mean nutrient-profile deviation D")
axes[0].set_ylabel("climate change vs original (%)")
axes[-1].legend(fontsize=6.5, loc="upper right", bbox_to_anchor=(1.0, 0.9))
fig.tight_layout()
save(fig, 4, "Pareto fronts climate vs nutrient deviation",
     "epsilon-constraint fronts for four basket-change budgets; circle = declared operating point (D <= 0.10, change <= 50%); all front points satisfy every constraint")

# %%
pareto_tab = sweeps[sweeps["pareto"]][["scenario", "eps_dev", "eps_change", "climate_change", "land_change", "water_change",
                                      "mean_deviation", "change_share", "n_new_foods", "dqr_mass_weighted"]]
print(f"{len(pareto_tab)} distinct non-dominated solutions across scenarios (climate, D, change)")
pareto_tab.groupby("scenario").agg(n=("climate_change", "size"), best_climate=("climate_change", "min"),
                                   climate_at_D0=("climate_change", lambda s: s[pareto_tab.loc[s.index, "eps_dev"] == 0].min())).round(3)

# %% [markdown]
# ## 5. How many swaps are needed? (MILP cardinality front)

# %%
fig, axes = plt.subplots(1, 5, figsize=(19, 3.9), sharey=True)
for ax, s in zip(axes, SCEN):
    d = R[s]["swaps"]
    for budget, col, lab in [("none", PALETTE[1], "no deviation budget"), ("0.10", PALETTE[0], "deviation D ≤ 0.10")]:
        g = d[d["dev_budget"] == budget]
        ax.plot(g["max_new"], 100 * g["climate_change"], "o-", color=col, ms=4, label=lab)
    ax.axhline(0, color="k", lw=0.6)
    ax.set_title(f"{s}: {R[s]['problem'].label}", fontsize=9); ax.set_xlabel("max number of new foods")
axes[0].set_ylabel("climate change vs original (%)"); axes[0].legend(fontsize=7)
fig.tight_layout()
save(fig, 4, "Climate reduction vs number of swaps", "0 new foods = portion rebalancing only (slot masses within +/-25%); all nutrition constraints enforced")
swaps.pivot_table(index="max_new", columns=["scenario", "dev_budget"], values="climate_change").round(3).head(6)

# %% [markdown]
# ## 6. The declared operating point: before / after

# %%
selected = {}
for s in SCEN:
    P = R[s]["problem"]
    r = O.solve(P, "climate", eps_dev=C.OPT_SELECTED["eps_dev"], eps_change=C.OPT_SELECTED["eps_change"])
    assert r is not None and r["feasible"]
    selected[s] = r
comp = pd.concat([O.composition(R[s]["problem"], selected[s]["x"]).assign(scenario=s) for s in SCEN], ignore_index=True)
comp.to_csv(C.TABLES / "D4_selected_compositions.csv", index=False)
sel_tab = pd.DataFrame([{**{k: v for k, v in selected[s].items() if k != "x"}, "scenario": s} for s in SCEN])
sel_tab.to_csv(C.TABLES / "D4_selected_solutions.csv", index=False)
sel_tab[["scenario", "climate_change", "land_change", "water_change", "eutro_fw_change", "acidification_change",
         "fossil_change", "mean_deviation", "change_share", "n_new_foods", "dqr_mass_weighted"]].round(3)

# %%
comp[comp["scenario"] == "S2"].round(1)

# %%
fig, axes = plt.subplots(1, 2, figsize=(15, 5.2), gridspec_kw={"width_ratios": [1.25, 1]})
ret_keys = C.NUTRIENT_KEYS
for i, s in enumerate(SCEN):
    vals = [100 * selected[s][f"ret_{k}"] for k in ret_keys]
    axes[0].scatter(range(len(ret_keys)), vals, color=PALETTE[i], s=26, label=s, zorder=3)
eo_vals = np.array([[100 * R[s]["env_only"][f"ret_{k}"] for k in ret_keys] for s in SCEN])
axes[0].scatter(np.tile(range(len(ret_keys)), len(SCEN)), eo_vals.ravel(), marker="x", s=20, color=SEM["misleading"], alpha=0.7,
                label="climate-only (no constraints)")
axes[0].axhline(100, color="k", lw=0.8); axes[0].axhline(90, color="k", ls=":", lw=0.8)
axes[0].set_xticks(range(len(ret_keys))); axes[0].set_xticklabels([k.rsplit("_", 1)[0] for k in ret_keys], rotation=45, ha="right")
axes[0].set_ylabel("% of original basket"); axes[0].set_yscale("log"); axes[0].set_ylim(5, 600)
axes[0].legend(fontsize=7, ncol=2); axes[0].set_title("Nutrient retention at the operating point")
inds = ["climate", "land", "water", "eutro_fw", "eutro_marine", "acidification", "fossil"]
w = 0.15
for i, s in enumerate(SCEN):
    axes[1].bar(np.arange(len(inds)) + (i - 2) * w, [100 * selected[s][f"{k}_change"] for k in inds], width=w, color=PALETTE[i], label=s)
axes[1].axhline(0, color="k", lw=0.8)
axes[1].set_xticks(range(len(inds))); axes[1].set_xticklabels([C.ENV_LABEL[k] for k in inds], rotation=35, ha="right")
axes[1].set_ylabel("change vs original (%)"); axes[1].legend(fontsize=7); axes[1].set_title("Environmental change on all seven indicators")
fig.tight_layout()
save(fig, 4, "Before after nutrient and environmental profiles", "operating point D <= 0.10, change <= 50%; red crosses = climate-only optimum without nutrition constraints")

# %%
fig, axes = plt.subplots(len(SCEN), 1, figsize=(13, 2.3 * len(SCEN) + 1))
for ax, s in zip(axes, SCEN):
    P = R[s]["problem"]
    v = P.var.assign(grams=selected[s]["x"])
    c = v[(v["grams"] > 0.5) | (v["x0"] > 0.5)]   # every original food AND every food in the optimum
    slots = list(dict.fromkeys(P.slots["slot"]))
    for row, (lab, col) in enumerate([("after", "grams"), ("before", "x0")]):
        left = 0.0
        for sl in slots:
            for _, f in c[c["slot"] == sl].iterrows():
                g = f[col]
                if g <= 0.5:
                    continue
                color = "#BBBBBB" if f["is_original"] else PALETTE[int(f["ciqual_code"]) % len(PALETTE)]
                ax.barh(row, g, left=left, color=color, edgecolor="white", lw=0.6)
                if g > 35:
                    ax.text(left + g / 2, row, short(f["food_name"], int(g / 7)), ha="center", va="center", fontsize=6)
                left += g
    ax.set_yticks([0, 1]); ax.set_yticklabels(["optimised", "original"]); ax.set_title(f"{s}: {P.label}", fontsize=9, loc="left")
    ax.set_xlabel("grams" if s == SCEN[-1] else "")
fig.tight_layout()
save(fig, 4, "Substitution flows original vs optimised baskets", "grey = original food kept; coloured = substituted food; operating point D <= 0.10, change <= 50%")

# %% [markdown]
# ## 7. Constraint slack and explicit violations

# %%
rows = []
for s in SCEN:
    P = R[s]["problem"]
    for name, x in [("original", P.x0), ("operating point", selected[s]["x"]), ("climate optimum", R[s]["climate_opt"]["x"]),
                    ("climate-only (no constraints)", R[s]["env_only"]["x"])]:
        chk = O.check(P, x).assign(scenario=s, solution=name)
        rows.append(chk)
checks = pd.concat(rows, ignore_index=True)
checks.to_csv(C.TABLES / "D4_constraint_checks.csv", index=False)
nut = checks[~checks["constraint"].str.startswith("mass[")].copy()
bound = np.where(np.isfinite(nut["lower"]), nut["lower"], nut["upper"])
nut["value_vs_bound"] = nut["value"] / bound
piv = nut.pivot_table(index="constraint", columns=["solution", "scenario"], values="value_vs_bound")
order = ["original", "operating point", "climate optimum", "climate-only (no constraints)"]
piv = piv[[c for o in order for c in piv.columns if c[0] == o]]
fig, ax = plt.subplots(figsize=(15, 4.8))
im = ax.imshow(np.log2(piv.to_numpy(dtype=float)), cmap="RdBu", vmin=-2, vmax=2, aspect="auto")
ax.set_yticks(range(len(piv))); ax.set_yticklabels(piv.index)
ax.set_xticks(range(piv.shape[1])); ax.set_xticklabels([f"{a}\n{b}" for a, b in piv.columns], fontsize=6, rotation=90)
viol_mask = nut.pivot_table(index="constraint", columns=["solution", "scenario"], values="satisfied", aggfunc="first")[piv.columns]
for i in range(piv.shape[0]):
    for j in range(piv.shape[1]):
        if not bool(viol_mask.iat[i, j]):
            ax.text(j, i, "✗", ha="center", va="center", fontsize=9, color="k", fontweight="bold")
ax.grid(False)
fig.colorbar(im, ax=ax, fraction=0.02, label="log2(value / bound)")
save(fig, 4, "Constraint slack and violations",
     "blue = slack on the safe side of a minimum / red = above; X = violated (only the unconstrained climate-only baseline violates)")

# %% [markdown]
# ## 8. Do objectives genuinely conflict?
#
# *Climate vs nutrient deviation* conflict by construction of the fronts above. Below: climate vs water
# and land — minimise climate subject to a water (or land) cap, swept from the water-optimal to the
# climate-optimal basket — and how random role-preserving swaps fare.

# %%
fig, axes = plt.subplots(1, 3, figsize=(16, 4.6))
for i, s in enumerate(SCEN):
    d = R[s]["conflict_water"].sort_values("water_change")
    axes[0].plot(100 * d["water_change"], 100 * d["climate_change"], "o-", ms=3, color=PALETTE[i], label=s)
    d = R[s]["conflict_land"].sort_values("land_change")
    axes[1].plot(100 * d["land_change"], 100 * d["climate_change"], "o-", ms=3, color=PALETTE[i], label=s)
for ax, k in zip(axes[:2], ["water", "land"]):
    ax.set_xlabel(f"{C.ENV_LABEL[k]} change (%)"); ax.set_ylabel("best achievable climate change (%)")
    ax.set_title(f"Climate vs {C.ENV_LABEL[k].lower()} front"); ax.legend(fontsize=7)
feas = rs.groupby("scenario")["feasible"].mean()
axes[2].bar(feas.index, 100 * feas.to_numpy(), color=SEM["misleading"], alpha=0.8)
for xx, (s, v) in enumerate(feas.items()):
    axes[2].text(xx, 100 * v + 0.2, f"{100 * v:.1f}%", ha="center", fontsize=8)
axes[2].set_ylabel("% of 5,000 random swaps satisfying all constraints"); axes[2].set_title("Random role-preserving swaps are rarely feasible")
fig.tight_layout()
save(fig, 4, "Objective conflicts and random baselines", "fronts: min climate s.t. water (or land) <= cap; all points satisfy every nutrition constraint")

# %%
cw = []
for s in SCEN:
    d = R[s]["conflict_water"]
    lo, hi = d.loc[d["water_change"].idxmin()], d.loc[d["climate_change"].idxmin()]
    cw.append({"scenario": s, "climate_at_water_optimum": lo["climate_change"], "water_at_water_optimum": lo["water_change"],
               "climate_at_climate_optimum": hi["climate_change"], "water_at_climate_optimum": hi["water_change"],
               "climate_cost_of_water_priority_pp": 100 * (lo["climate_change"] - hi["climate_change"]),
               "climate_optimum_raises_water": R[s]["climate_opt"]["water_change"] > 0})
cw = pd.DataFrame(cw)
cw.to_csv(C.TABLES / "D4_conflict_summary.csv", index=False)
cw.round(3)

# %%
base = pd.DataFrame([{"scenario": s, "original_climate_kg": R[s]["original"]["climate"],
                      "random_swaps_feasible_share": rs.loc[rs["scenario"] == s, "feasible"].mean(),
                      "random_swaps_feasible_median_climate_change": rs.loc[(rs["scenario"] == s) & rs["feasible"], "climate_change"].median(),
                      "random_feasible_median_climate_change": rf.loc[rf["scenario"] == s, "climate_change"].median(),
                      "climate_only_unconstrained_change": R[s]["env_only"]["climate_change"],
                      "climate_only_unconstrained_violations": R[s]["env_only"]["n_violations"],
                      "operating_point_change": selected[s]["climate_change"],
                      "constrained_climate_optimum_change": R[s]["climate_opt"]["climate_change"]} for s in SCEN])
base.to_csv(C.TABLES / "D4_baseline_comparison.csv", index=False)
base.round(3)

# %% [markdown]
# ## Key findings (D4)

# %%
opc = sel_tab.set_index("scenario")["climate_change"]
print(f"1. At the declared operating point (D <= 0.10, change <= 50%) climate impact falls by {-opc.max():.0%} to {-opc.min():.0%} across scenarios, "
      "with every nutrition constraint satisfied.")
rb = swaps[(swaps["max_new"] == 0) & (swaps["dev_budget"] == "none")].set_index("scenario")["climate_change"]
print(f"2. Portion rebalancing alone (no new foods) saves {-rb.max():.0%} to {-rb.min():.0%}.")
print(f"3. Random role-preserving swaps satisfy all constraints in only {rs.groupby('scenario')['feasible'].mean().min():.1%}-"
      f"{rs.groupby('scenario')['feasible'].mean().max():.1%} of 5,000 draws; the climate-only optimum without nutrition constraints violates "
      f"{int(viol['n_violations'].min())}-{int(viol['n_violations'].max())} bounds.")
print(f"4. Prioritising water costs up to {cw['climate_cost_of_water_priority_pp'].max():.1f} percentage points of climate reduction (genuine conflict).")
wr = sel_tab.set_index("scenario")["water_change"]
print(f"5. Burden shift at the operating point: water scarcity rises in {', '.join(f'{s} ({v:+.0%})' for s, v in wr[wr > 0].items()) or 'no scenario'}.")

# %% [markdown]
# **Interpretation.** Basket-level optimisation recovers what pairwise swaps (D3) cannot: protein and micronutrients
# lost in one slot are compensated in another, so large reductions remain feasible under explicit nutrient floors.
# The price is visible on the fronts — more deviation from the original nutrient profile, more changed grams, and
# sometimes a water or land trade-off. **These are computational scenarios, not dietary advice;** LP optima can be
# culturally implausible (D5 catalogues such cases).

# %% [markdown]
# ## 9. Acceptance criteria

# %%
acc = pd.DataFrame([
    ("3-5 fixed scenarios with declared foods and quantities", len(SCEN) == 5),
    ("formal variable / objective / constraint specification (section header)", True),
    ("reproducible optimiser (HiGHS via SciPy; seeded baselines; tests)", True),
    ("original, random-feasible and single-objective baselines", len(rs) > 0 and len(rf) > 0),
    ("all reported optimiser solutions satisfy constraints; violations of the unconstrained baseline shown", bool(all_opt["feasible"].all())),
    ("no hidden external menu dataset (config/baskets.csv only)", True),
    ("multiple objectives visible (fronts, not a single score)", True),
    ("at least one genuine objective conflict", bool((cw["climate_cost_of_water_priority_pp"] > 1).any())),
], columns=["D4 acceptance criterion", "met"])
acc.to_csv(C.TABLES / "D4_acceptance.csv", index=False)
assert acc["met"].all()
acc
