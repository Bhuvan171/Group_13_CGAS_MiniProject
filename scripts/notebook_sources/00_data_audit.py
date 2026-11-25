# %% [markdown]
# # 00 · Data audit and snapshot freeze
#
# **Sustainome — BIO544 Computational Gastronomy mini-project** · supports Deliverable 1
#
# This notebook freezes the raw downloads (checksums, versions, licences), inspects the raw
# schemas before any cleaning, reproduces the feasibility audit quoted in the proposal
# (AGRIBALYSE 3.1 API) and compares the 3.1 release against the 3.2 release used for the
# analysis. **3.1 values are never mixed into analytical tables** — they appear only here,
# explicitly labelled, as a release comparison.

# %%
import sys
from pathlib import Path

ROOT = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sustainome import config as C, io, clean
from sustainome.plotting import setup, save, PALETTE

setup()
pd.set_option("display.width", 200, "display.max_colwidth", 80, "display.max_columns", 30)
C.TABLES.mkdir(parents=True, exist_ok=True)

# %% [markdown]
# ## 1. Snapshot manifest
# Raw files live under `data/raw/<source>/<version>/` and are never edited. The manifest
# records organisation, version, URL, access date, byte size, SHA-256, licence and the
# sheet used. If a manifest already exists, the checksums are re-verified against it.

# %%
if not C.MANIFEST_FILE.exists():
    io.write_manifest(io.build_manifest(access_date="2026-11-17"))
manifest = io.read_manifest()
checks = io.verify_manifest()
man = pd.DataFrame(manifest["files"]).T
man["checksum_verified"] = pd.Series(checks)
man[["version", "local_path", "bytes", "sha256", "access_date", "license", "checksum_verified"]]

# %%
assert all(checks.values()), "a raw snapshot no longer matches its frozen checksum"
man.reset_index(names="source").to_csv(C.TABLES / "D1_snapshot_manifest.csv", index=False)

# %% [markdown]
# ## 2. Raw schema inspection (before any cleaning)

# %%
raw_c = io.load_ciqual_raw()
print("CIQUAL 2025 composition sheet:", raw_c.shape, "(rows, columns)")
print("first identifier columns:", list(raw_c.columns[:9]))
print("nutrient columns:", raw_c.shape[1] - 10, "+ Jones factor")
vals = pd.Series(raw_c.iloc[:, 9:83].to_numpy().ravel()).dropna().map(lambda v: " ".join(str(v).split()))
kinds = vals.map(lambda v: "missing '-'" if v == "-" else "traces" if v.lower() == "traces" else "'< x'" if v.startswith("<") else "numeric (decimal comma)" if "," in v else "numeric")
kinds.value_counts().rename("cells").to_frame()

# %%
raw_a = io.load_agribalyse_raw()
print("AGRIBALYSE 3.2 'Synthese' rows:", len(raw_a), "| unique AGB codes:", raw_a["agb_code"].nunique(),
      "| unique CIQUAL codes:", raw_a["ciqual_code_raw"].nunique(), "| red-font rows:", int(raw_a["row_font_red"].sum()))
notice = io.load_agribalyse_notice()
for para in notice:
    if any(w in para for w in ["DOUBLON", "OEUFS", "QUINOA", "Version", "1 kg de produit"]):
        print("•", " ".join(para.split())[:400], "\n")

# %% [markdown]
# The publisher's Notice (in French) states three facts that shape the D1 rules:
#
# 1. impacts are given **per kg of product consumed** (losses included) → we divide by 10 to align with CIQUAL's per-100 g basis;
# 2. **seven products appear twice** — red rows are the uncorrected packaging values, black rows the corrected ones → keep black;
# 3. **known errors**: egg impacts are *under*-estimated (and all products containing eggs); quinoa impacts are *over*-estimated (red rows) → flagged, excluded from optimisation candidates, and tested in D5.

# %% [markdown]
# ## 3. Reproducing the proposal's feasibility audit (AGRIBALYSE 3.1 API snapshot)

# %%
a31 = io.load_agribalyse31_api()
a31["ciqual_code"] = [clean.normalize_code(v).code for v in a31["ciqual_code_raw"]]
ciq_codes = {clean.normalize_code(v).code for v in raw_c["alim_code"]}
u31 = set(a31["ciqual_code"].dropna())
audit31 = pd.DataFrame([
    ("CIQUAL 2025 foods", len(ciq_codes), 3484),
    ("AGRIBALYSE 3.1 API rows", len(a31), 2451),
    ("AGRIBALYSE 3.1 unique CIQUAL codes", len(u31), 2425),
    ("Exact shared codes", len(u31 & ciq_codes), 2272),
    ("% of 3.1 unique codes matched", round(100 * len(u31 & ciq_codes) / len(u31), 2), 93.69),
], columns=["quantity", "recomputed", "proposal"])
audit31["reproduced"] = np.isclose(audit31["recomputed"].astype(float), audit31["proposal"].astype(float))
audit31

# %%
assert audit31["reproduced"].all()
audit31.to_csv(C.TABLES / "D1_feasibility_audit_reproduction.csv", index=False)

# %% [markdown]
# ## 4. Release comparison: AGRIBALYSE 3.1 vs 3.2 (audit only)

# %%
a32, _ = clean.resolve_agribalyse_duplicates(raw_a)
a31u = a31.dropna(subset=["ciqual_code"]).sort_values("ciqual_code").drop_duplicates("ciqual_code")
cmp = a32.merge(a31u, on="ciqual_code", suffixes=("_32", "_31"))
cmp = cmp[cmp["ciqual_code"].isin(ciq_codes)]
for k in ["climate", "land", "water"]:
    cmp[f"{k}_ratio"] = cmp[f"{k}_per_kg_32"] / cmp[f"{k}_per_kg_31"]
summary = pd.DataFrame({
    k: {"shared foods": len(cmp),
        "median 3.2/3.1 ratio": cmp[f"{k}_ratio"].median(),
        "share changed > 5 %": (np.abs(np.log(cmp[f"{k}_ratio"].where(cmp[f"{k}_ratio"] > 0))) > np.log(1.05)).mean(),
        "Spearman rho": cmp[[f"{k}_per_kg_32", f"{k}_per_kg_31"]].corr("spearman").iloc[0, 1]}
    for k in ["climate", "land", "water"]}).T
summary

# %%
fig, axes = plt.subplots(1, 3, figsize=(12, 3.9))
for ax, k in zip(axes, ["climate", "land", "water"]):
    x, y = cmp[f"{k}_per_kg_31"], cmp[f"{k}_per_kg_32"]
    ok = (x > 0) & (y > 0)
    ax.scatter(x[ok], y[ok], s=6, alpha=0.45, color=PALETTE[0], lw=0)
    lim = [min(x[ok].min(), y[ok].min()), max(x[ok].max(), y[ok].max())]
    ax.plot(lim, lim, color="k", lw=0.8, ls="--")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel(f"AGRIBALYSE 3.1 ({C.ENV_UNIT[k]}/kg)"); ax.set_ylabel(f"AGRIBALYSE 3.2 ({C.ENV_UNIT[k]}/kg)")
    ax.set_title(f"{C.ENV_LABEL[k]}: rho = {summary.loc[k, 'Spearman rho']:.3f}, "
                 f"{100 * summary.loc[k, 'share changed > 5 %']:.1f}% changed > 5%", fontsize=9)
save(fig, 1, "Release comparison AGRIBALYSE 3.1 vs 3.2",
     f"{len(cmp)} foods present in CIQUAL 2025 and both releases; positive values only on log axes; 3.1 values are not used in any analysis")

# %%
summary.to_csv(C.TABLES / "D1_release_comparison_3.1_vs_3.2.csv")
print("Release 3.2 is the analysis default; 3.1 is retained only for this audit.")
