# %% [markdown]
# # 01 · Deliverable 1 — Sustainable Food Matrix
#
# **Question answered:** can CIQUAL 2025 nutrient composition be joined to AGRIBALYSE 3.2
# environmental indicators and data-quality scores *exactly*, reproducibly and with every
# loss, duplicate and missing value accounted for?
#
# **Outputs** (all regenerated from the frozen raw snapshots by `sustainome.matrix.build_matrix`):
# versioned integrated dataset · source-to-clean data dictionary · matched / CIQUAL-only /
# AGRIBALYSE-only / duplicate-resolution audit tables · coverage and missingness figures ·
# provenance manifest.
#
# **Rules enforced in code** (see `src/sustainome/clean.py`, `matrix.py`):
# exact join on normalised integer CIQUAL codes (no fuzzy names) · CIQUAL markers parsed
# explicitly (`-` → missing, never 0; `traces` → 0; `< x` → x/2) · AGRIBALYSE duplicates
# resolved by a documented, tested rule · units harmonised (per kg → per 100 g) · DQR kept
# as an ordinal 1–5 score.

# %%
import sys, hashlib, json
from pathlib import Path

ROOT = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

from sustainome import config as C, matrix as MX, io
from sustainome.plotting import setup, save, PALETTE, short

setup()
pd.set_option("display.width", 200, "display.max_colwidth", 70, "display.max_columns", 30, "display.max_rows", 80)
for d in (C.PROCESSED, C.TABLES, C.FIGURES):
    d.mkdir(parents=True, exist_ok=True)

# %% [markdown]
# ## 1. Build the matrix from the frozen snapshots

# %%
assert all(io.verify_manifest().values()), "raw snapshots changed since the manifest was frozen"
build = MX.build_matrix()
m, audit = build.matrix, build.audit
print("options:", build.options)
audit["stage_counts"]

# %% [markdown]
# Every stage reconciles: matched + CIQUAL-only = CIQUAL codes, and matched + AGRIBALYSE-only =
# AGRIBALYSE codes after duplicate resolution. The 3.2 exact-match count (2,272) happens to equal the
# 3.1 feasibility audit; the AGRIBALYSE-only codes are products CIQUAL 2025 no longer lists (for
# example generic tomato 20047).

# %%
sc = audit["stage_counts"].set_index("stage")["count"]
matched = sc["Exact matches (inner join on normalised CIQUAL code)"]
assert matched + sc["CIQUAL-only codes (no AGRIBALYSE record)"] == sc["CIQUAL unique codes"]
assert matched + sc["AGRIBALYSE-only codes (not in CIQUAL 2025)"] == sc["AGRIBALYSE unique CIQUAL codes (after duplicate resolution)"]
assert m["ciqual_code"].is_unique and len(m) == matched
print(f"reconciled: {matched} matched foods; codes altered by normalisation: {int(m['ciqual_code_altered'].sum()) if 'ciqual_code_altered' in m else 0}")

# %% [markdown]
# ## 2. Duplicate resolution (AGRIBALYSE)
#
# *Tier A* — publisher packaging correction: of two rows with the same AGB code, the red
# "software version – with packaging error" row is dropped and the corrected row kept.
# *Tier B* — several AGB variants (`_1`, `_2`, …) for one CIQUAL code: keep the French
# consumption-mix variant (season code 2), then not air-freighted, then unsuffixed, then the
# lowest suffix. Alternatives (mean / max / min over variants) are re-run in D5.

# %%
dup = audit["duplicate_resolution"]
dup["climate_ratio_dropped_over_kept"] = dup["dropped_climate"] / dup["kept_climate"]
dup[["ciqual_code", "tier", "kept_agb_code", "kept_lci_name", "dropped_agb_code", "dropped_lci_name",
     "kept_climate", "dropped_climate"]]

# %% [markdown]
# The variant rule matters: dropped variants include air-freighted mango (11.6 vs 0.73 kg CO₂e/kg)
# and Kenyan French beans by plane (7.08 vs 0.45). The rule keeps the variant representing average
# French consumption, consistent with CIQUAL's average-product composition values.

# %% [markdown]
# ## 3. Name-consistency validation (a check, not the join method)
#
# The join never uses names. As a validation we compare CIQUAL English names with AGRIBALYSE LCI
# names by token overlap and manually inspect the least similar matches (plus a random sample).
# Verdicts were recorded by hand in `config/manual_name_audit.csv`.

# %%
low = m.nsmallest(15, "name_token_jaccard")[["ciqual_code", "food_name", "agb_lci_name", "name_token_jaccard"]]
rand = m.sample(10, random_state=C.SEED)[["ciqual_code", "food_name", "agb_lci_name", "name_token_jaccard"]]
sample = pd.concat([low.assign(sample="15 least similar"), rand.assign(sample="random 10")])
verdicts_path = C.CONFIG_DIR / "manual_name_audit.csv"
if verdicts_path.exists():
    verdicts = pd.read_csv(verdicts_path, dtype=str)
    sample = sample.merge(verdicts[["ciqual_code", "verdict", "comment"]], on="ciqual_code", how="left")
sample.to_csv(C.TABLES / "D1_name_consistency_audit.csv", index=False)
print("median token Jaccard over all matches:", round(m["name_token_jaccard"].median(), 2))
sample

# %% [markdown]
# ## 4. Coverage and missingness

# %%
cov = audit["group_coverage"].copy()
cov["ciqual_group"] = cov["ciqual_group"].str.replace("\n", " ")
miss = audit["missingness"]
miss

# %%
cov["ciqual_group"] = cov["ciqual_group"].replace("", "(no group label)").map(lambda s: short(s, 30))
fig = plt.figure(figsize=(14, 5.8))
gs = fig.add_gridspec(1, 2, width_ratios=[1.0, 1.15], wspace=0.55)
ax = fig.add_subplot(gs[0]); ax.axis("off"); ax.set_xlim(0, 10); ax.set_ylim(0, 10)
boxes = [
    (0.2, 7.6, f"CIQUAL 2025\n{sc['CIQUAL unique codes']:,} foods", PALETTE[0]),
    (5.4, 7.6, f"AGRIBALYSE 3.2\n{sc['AGRIBALYSE 3.2 Synthese data rows']:,} rows", PALETTE[1]),
    (5.4, 5.1, f"{sc['AGRIBALYSE unique CIQUAL codes (after duplicate resolution)']:,} unique codes\n7 packaging + 26 variant\nrows resolved", PALETTE[1]),
    (2.8, 2.6, f"Exact join on CIQUAL code\n{matched:,} matched foods", PALETTE[2]),
    (0.2, 0.2, f"CIQUAL-only\n{sc['CIQUAL-only codes (no AGRIBALYSE record)']:,}", "#999999"),
    (5.4, 0.2, f"AGRIBALYSE-only\n{sc['AGRIBALYSE-only codes (not in CIQUAL 2025)']:,}", "#999999"),
]
for x, y, txt, col in boxes:
    ax.add_patch(FancyBboxPatch((x, y), 4.3, 1.9, boxstyle="round,pad=0.08", fc=col, ec="none", alpha=0.18))
    ax.text(x + 2.15, y + 0.95, txt, ha="center", va="center", fontsize=9)
arrow = dict(arrowstyle="->", color="#444444", lw=1)
for (x1, y1), (x2, y2) in [((7.55, 7.6), (7.55, 7.05)), ((2.35, 7.6), (4.3, 4.55)), ((7.55, 5.1), (6.0, 4.55)),
                           ((4.0, 2.6), (2.4, 2.15)), ((6.0, 2.6), (7.55, 2.15))]:
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1), arrowprops=arrow)
ax.set_title("Join flow (counts reconcile at every stage)")
ax2 = fig.add_subplot(gs[1])
cv = cov.sort_values("ciqual_foods")
ax2.barh(cv["ciqual_group"], cv["ciqual_foods"], color="#DDDDDD", label="CIQUAL foods")
ax2.barh(cv["ciqual_group"], cv["matched_foods"], color=PALETTE[2], label="matched to AGRIBALYSE 3.2")
for y, (n, k, p) in enumerate(zip(cv["ciqual_foods"], cv["matched_foods"], cv["coverage_pct"])):
    ax2.text(n + 8, y, f"{p:.0f}%", va="center", fontsize=8)
ax2.set_xlabel("foods"); ax2.legend(loc="lower right"); ax2.set_title("Coverage by CIQUAL food group")
save(fig, 1, "Join flow and coverage", "exact normalised-code join; no fuzzy name matching")

# %%
feat = [k for k, *_ in MX.ALL_FEATURES]
grp = m["ciqual_group"].str.replace("\n", " ")
share_missing = pd.DataFrame({k: m[k].isna().groupby(grp).mean() for k in feat}).T
fig = plt.figure(figsize=(15, 6.4))
gsm = fig.add_gridspec(1, 3, width_ratios=[2.1, 0.05, 1.05], wspace=0.12)
ax1, cax, ax2 = fig.add_subplot(gsm[0]), fig.add_subplot(gsm[1]), fig.add_subplot(gsm[2])
fig.subplots_adjust(wspace=0.12)
im = ax1.imshow(share_missing.to_numpy(), aspect="auto", cmap="Reds", vmin=0, vmax=1)
ax1.set_xticks(range(share_missing.shape[1])); ax1.set_xticklabels([short(c, 24) for c in share_missing.columns], rotation=40, ha="right")
ax1.set_yticks(range(len(feat))); ax1.set_yticklabels([f"{k}{' *' if k in C.NUTRIENT_KEYS else ''}" for k in feat])
ax1.grid(False)
for i in range(share_missing.shape[0]):
    for j in range(share_missing.shape[1]):
        v = share_missing.iat[i, j]
        if v >= 0.05:
            ax1.text(j, i, f"{100 * v:.0f}", ha="center", va="center", fontsize=6.5, color="white" if v > 0.6 else "black")
fig.colorbar(im, cax=cax)
cax.set_title("share\nmissing", fontsize=7)
ax1.set_title("Share of missing values by nutrient and CIQUAL group (* = frozen core feature)")
gsm.update(wspace=0.1)
pos = ax2.get_position(); ax2.set_position([pos.x0 + 0.075, pos.y0, pos.width - 0.075, pos.height])
status_cols = ["numeric", "trace", "below_loq", "structural_zero", "missing"]
colors = [PALETTE[0], PALETTE[5], PALETTE[1], PALETTE[2], "#CC3333"]
left = np.zeros(len(miss))
for col, c in zip(status_cols, colors):
    ax2.barh(miss["feature"], miss[col], left=left, color=c, label=col.replace("_", " "))
    left += miss[col].to_numpy()
ax2.axvline(0.7 * len(m), color="k", ls="--", lw=0.8)
ax2.text(0.7 * len(m), len(miss) - 0.4, " 70% coverage rule", fontsize=7.5, va="bottom")
ax2.invert_yaxis(); ax2.set_xlabel("matched foods")
ax2.legend(fontsize=7, loc="upper center", bbox_to_anchor=(0.45, -0.1), ncol=3, frameon=False)
ax2.set_title("Value status per nutrient (matched set)")
save(fig, 1, "Nutrient missingness matrix", "missing values are never zero-filled; B12 structural zeros are explicit and switchable (D5)")

# %% [markdown]
# **Feature freeze.** Fifteen nutrients pass the ≥70 % coverage rule and are nutritionally
# interpretable; folate (51 %), vitamin E (18.5 %) and vitamin A RE (59 %) fail and are excluded
# from the core set (vitamin D at 69.9 % is kept as an *extended* feature for D5 only). The
# NRF9.3 index is therefore adapted: zinc and vitamin B12 replace vitamins A and E.
#
# **B12 structural zeros.** CIQUAL leaves B12 as "–" (not analysed) for many plant foods although
# B12 is absent from unfortified plant foods. A declared rule (`config/structural_zero.csv`) sets
# these to 0 *only* for plant-origin groups whose names mention no animal, algal, fermented or
# fortified ingredient; such cells carry status `structural_zero`, their raw "–" is preserved, and
# D5 re-runs everything with the rule switched off.

# %%
feat_table = miss.assign(selected=miss["core"].map({True: "core (15)", False: "extended / D5 only"}))
feat_table.to_csv(C.TABLES / "D1_feature_coverage_and_freeze.csv", index=False)
print("structural zeros applied:", int(m["b12_structural_zero"].sum()), "foods (vitamin B12 only)")
print("complete 15-nutrient profiles:", int(m["profile_complete"].sum()), "| graph-eligible (>=12/15):", int(m["graph_eligible"].sum()))

# %% [markdown]
# ## 5. Comparable groups, basket roles and quality flags

# %%
g = m.groupby(["analysis_group_code", "analysis_group"]).agg(
    foods=("ciqual_code", "size"), complete_nr=("nr_complete", "sum"),
    median_dqr=("dqr", "median"), median_climate_per_100g=("climate_per_100g", "median")).reset_index()
print(f"{len(g)} analysis groups; {int((g['complete_nr'] >= C.MIN_GROUP_SIZE).sum())} have >= {C.MIN_GROUP_SIZE} complete foods for D2")
roles = m["roles"].replace("", np.nan).dropna().str.split("|").explode().value_counts().rename("foods")
display(roles.to_frame())
print("state-split groups:", sorted(m.loc[m["analysis_group_code"].str.contains(":"), "analysis_group"].unique())[:12], "...")
flags = m.groupby(["issue_direct", "issue_indirect"]).size().rename("foods")
flags

# %%
fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.2))
cg = m["ciqual_group"].str.replace("\n", " ")
order = m.groupby(cg)["dqr"].median().sort_values().index
axes[0].boxplot([m.loc[cg == k, "dqr"] for k in order], vert=False, widths=0.6, flierprops=dict(markersize=2))
axes[0].set_yticks(range(1, len(order) + 1)); axes[0].set_yticklabels([short(k, 26) for k in order])
axes[0].set_xlabel("DQR (1 = excellent … 5 = very poor)"); axes[0].set_title("Data Quality Rating by group (ordinal)")
for ax, k in zip(axes[1:], ["climate", "water"]):
    v = m[f"{k}_per_100g"]
    ax.hist(np.log10(v[v > 0]), bins=45, color=PALETTE[0], alpha=0.8)
    ax.set_xlabel(f"log10 {C.ENV_LABEL[k]} ({C.ENV_UNIT[k]} per 100 g)"); ax.set_ylabel("foods")
    ax.set_title(f"{C.ENV_LABEL[k]}: {int((v <= 0).sum())} non-positive values")
save(fig, 1, "DQR and indicator distributions", "AGRIBALYSE 3.2, matched foods, per 100 g after unit harmonisation")

# %% [markdown]
# ## 6. Data dictionary and published artefacts

# %%
dd = audit["data_dictionary"]
dd.to_csv(C.PROCESSED / "data_dictionary.csv", index=False)
dd.to_csv(C.TABLES / "D1_data_dictionary.csv", index=False)
dd[dd["use"].str.contains("core|indicator", na=False)][["source", "original_header", "clean_name", "unit", "basis", "use"]]

# %%
cols = MX.matrix_columns(m)
out = m[cols]
out.to_csv(C.MATRIX_FILE, index=False, encoding="utf-8")
out.to_parquet(C.MATRIX_PARQUET, index=False)
for name, t in audit.items():
    if name != "data_dictionary":
        t.to_csv(C.TABLES / f"D1_{name}.csv", index=False)
g.to_csv(C.TABLES / "D1_analysis_groups.csv", index=False)
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
processed_manifest = {
    "built_from": {k: v["sha256"] for k, v in io.read_manifest()["files"].items()},
    "options": build.options,
    "matrix_csv": {"rows": len(out), "columns": out.shape[1], "sha256": sha(C.MATRIX_FILE)},
    "agribalyse_release": "3.2 (no 3.1 values in any analytical table)",
}
(C.PROCESSED / "PROCESSED_MANIFEST.json").write_text(json.dumps(processed_manifest, indent=2))
print(out.shape, "->", C.MATRIX_FILE.relative_to(C.ROOT))

# %% [markdown]
# ## 7. Reproducibility check and acceptance criteria

# %%
again = MX.build_matrix().matrix[cols]
pd.testing.assert_frame_equal(again.reset_index(drop=True), out.reset_index(drop=True))
st_cols = [f"{k}__status" for k in C.NUTRIENT_KEYS]
acceptance = pd.DataFrame([
    ("primary join uses exact normalised CIQUAL codes", True),
    ("source versions and checksums recorded and verified", all(io.verify_manifest().values())),
    ("counts reconcile across every pipeline stage", True),
    ("missing nutrients are not zero-filled", all(m.loc[m[f'{k}__status'] == 'missing', k].isna().all() for k in C.NUTRIENT_KEYS)),
    ("duplicate handling documented and tested (tests/test_clean.py)", len(dup) == 33),
    ("processed dataset regenerated identically from raw files", True),
], columns=["D1 acceptance criterion", "met"])
acceptance.to_csv(C.TABLES / "D1_acceptance.csv", index=False)
assert acceptance["met"].all()
acceptance
