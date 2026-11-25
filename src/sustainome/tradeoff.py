"""Deliverable 2 - Nutrition-Environment Trade-off Analysis (food level)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import config as C
from . import normalize, nutrition, pareto
from . import stats as st

META = ["ciqual_code", "food_name", "analysis_group_code", "analysis_group", "comp_group_code",
        "ciqual_group", "roles", "dqr", "issue_direct", "prep_state"]

# Category labels inside declared roles (used for role-level contrasts)
PROTEIN_CATEGORY = {
    "040101": "beef & veal", "040102": "pork", "040103": "poultry", "040105": "lamb & mutton",
    "040107": "other meats", "040108": "offal", "0405": "fish", "0407": "seafood",
    "0409": "fish products", "041001": "eggs", "020301": "legumes", "1009": "tofu/seitan/TVP",
    "010309": "plant-based patties",
}
PLANT_GROUPS = {"020301", "1009", "010309", "060205", "050205", "050306", "040309", "0902", "1010"}


def origin(m: pd.DataFrame) -> pd.Series:
    """Plant vs animal origin for role-level contrasts (declared group list + name rule for fats)."""
    plant = m["comp_group_code"].isin(PLANT_GROUPS)
    other_fat = m["comp_group_code"].eq("0905") & ~m["food_name"].str.contains(r"lard|duck|goose|tallow|butter", case=False)
    return pd.Series(np.where(plant | other_fat, "plant", "animal"), index=m.index)


def protein_category(m: pd.DataFrame) -> pd.Series:
    return m["comp_group_code"].map(PROTEIN_CATEGORY)


# ---------------------------------------------------------------------------
# view tables
# ---------------------------------------------------------------------------
def view_long(m: pd.DataFrame, energy_floor: float = C.ENERGY_FLOOR_KCAL, protein_floor: float = C.PROTEIN_FLOOR_G,
              nr_min_available: int | None = None) -> pd.DataFrame:
    parts = []
    for v in C.VIEWS:
        fu = normalize.to_view(m, v, energy_floor=energy_floor, protein_floor=protein_floor)
        fu = nutrition.add_scores(fu, min_available=nr_min_available)
        fu = fu.drop(columns=["ciqual_code"]).join(m[META])
        parts.append(fu)
    out = pd.concat(parts, ignore_index=True)
    out["origin"] = pd.concat([origin(m)] * len(C.VIEWS), ignore_index=True)
    out["protein_category"] = pd.concat([protein_category(m)] * len(C.VIEWS), ignore_index=True)
    return out


# ---------------------------------------------------------------------------
# Pareto labelling
# ---------------------------------------------------------------------------
def pareto_labels(vl: pd.DataFrame, by: str = "analysis_group_code", objectives: dict = C.PARETO_OBJECTIVES,
                  min_size: int = C.MIN_GROUP_SIZE, two_d: dict | None = C.PARETO_2D, depth: bool = True) -> pd.DataFrame:
    """Within-group Pareto labels for every view.

    A group is analysed in a view only if it has >= ``min_size`` foods with every objective
    defined; excluded foods/groups are kept with ``analysed=False`` and a reason.
    """
    n = len(vl)
    obj = vl[list(objectives)].to_numpy(float)
    sense = list(objectives.values())
    complete = np.isfinite(obj).all(axis=1) & vl["fu_defined"].to_numpy(bool)
    obj2 = vl[list(two_d)].to_numpy(float) if two_d else None
    sense2 = list(two_d.values()) if two_d else None
    ngc = np.zeros(n, dtype=int)
    analysed = np.zeros(n, dtype=bool)
    eff, dep, domby, eff2, dep2 = (np.full(n, np.nan) for _ in range(5))
    group = vl[by].astype(str).to_numpy()
    for _, idx in vl.groupby(["view", by], sort=False).indices.items():
        ci = idx[complete[idx]]
        ngc[idx] = len(ci)
        if len(ci) < min_size:
            continue
        analysed[ci] = True
        X = obj[ci]
        eff[ci] = pareto.pareto_mask(X, sense)
        if depth:
            dep[ci] = pareto.nondominated_rank(X, sense)
            domby[ci] = pareto.domination_count(X, sense)
        if two_d:
            X2 = obj2[ci]
            eff2[ci] = pareto.pareto_mask(X2, sense2)
            if depth:
                dep2[ci] = pareto.nondominated_rank(X2, sense2)
    fu = vl["fu_defined"].to_numpy(bool)
    reason = np.where(~fu, "undefined in view", np.where(~complete, "missing objective",
                      np.where(ngc < min_size, f"group n<{min_size}", "")))
    out = vl[["ciqual_code", "food_name", "view"]].copy()
    out["group"] = group
    out["n_group_complete"] = ngc
    out["analysed"] = analysed
    out["exclusion_reason"] = reason
    out["efficient"], out["depth"], out["dominated_by"] = eff, dep, domby
    out["efficient_2d"], out["depth_2d"] = eff2, dep2
    return out


def role_long(vl: pd.DataFrame) -> pd.DataFrame:
    """Explode the view table so each (food, role) pair is one row."""
    r = vl[vl["roles"].fillna("") != ""].copy()
    r["role"] = r["roles"].str.split("|")
    return r.explode("role").reset_index(drop=True)


# ---------------------------------------------------------------------------
# group statistics
# ---------------------------------------------------------------------------
def classify_association(lo: float, hi: float) -> str:
    if np.isnan(lo) or np.isnan(hi):
        return "undetermined"
    if lo > 0:
        return "conflict"   # more nutritious <-> more impact
    if hi < 0:
        return "agreement"  # more nutritious <-> less impact
    return "no clear association"


def group_summary(vl: pd.DataFrame, labels: pd.DataFrame, n_boot: int = C.N_BOOT, seed: int = C.SEED) -> pd.DataFrame:
    d = vl.join(labels[["group", "analysed", "efficient", "depth", "efficient_2d"]])
    d = d[d["analysed"]]
    rows = []
    for i, ((view, grp), g) in enumerate(d.groupby(["view", "group"], sort=True)):
        rho, lo, hi = st.spearman_ci(g["nr9"], g["climate"], n_boot=n_boot, seed=seed + i)
        rows.append({
            "view": view, "analysis_group_code": grp, "analysis_group": g["analysis_group"].iloc[0],
            "n": len(g), "n_efficient_5d": int(g["efficient"].sum()), "share_efficient_5d": g["efficient"].mean(),
            "n_efficient_2d": int(g["efficient_2d"].sum()), "max_depth": g["depth"].max(),
            "median_climate": g["climate"].median(), "iqr_climate": g["climate"].quantile(.75) - g["climate"].quantile(.25),
            "climate_range_ratio": g["climate"].max() / g["climate"].min() if g["climate"].min() > 0 else np.nan,
            "median_nr9": g["nr9"].median(),
            "rho_nr9_climate": rho, "rho_lo": lo, "rho_hi": hi,
            "association": classify_association(lo, hi),
            "rho_nr9_land": st.spearman(g["nr9"], g["land"]), "rho_nr9_water": st.spearman(g["nr9"], g["water"]),
            "rho_lim3_climate": st.spearman(g["lim3"], g["climate"]),
            "rho_climate_water": st.spearman(g["climate"], g["water"]),
        })
    return pd.DataFrame(rows)


def view_agreement(labels: pd.DataFrame) -> pd.DataFrame:
    """Per group: Jaccard similarity of efficient sets between views and status flips."""
    a = labels[labels["analysed"]]
    piv = a.pivot_table(index=["group", "ciqual_code"], columns="view", values="efficient", aggfunc="first")
    rows = []
    for grp, g in piv.groupby(level=0):
        rec = {"analysis_group_code": grp}
        for v1, v2 in [("mass", "energy"), ("mass", "protein"), ("energy", "protein")]:
            if v1 in g and v2 in g:
                both = g[[v1, v2]].dropna()
                if len(both) >= C.MIN_GROUP_SIZE:
                    s1 = set(both.index[both[v1] == 1])
                    s2 = set(both.index[both[v2] == 1])
                    rec[f"jaccard_{v1}_{v2}"] = st.jaccard(s1, s2)
                    rec[f"flip_rate_{v1}_{v2}"] = float((both[v1] != both[v2]).mean())
                    rec[f"n_{v1}_{v2}"] = len(both)
        rows.append(rec)
    return pd.DataFrame(rows)


def role_contrasts(vl: pd.DataFrame, n_boot: int = C.N_BOOT, seed: int = C.SEED) -> pd.DataFrame:
    """Plant vs animal Cliff's delta on climate per view inside each declared role
    (negative delta = plant options have lower impact)."""
    rl = role_long(vl)
    rows = []
    for i, ((role, view), g) in enumerate(rl.groupby(["role", "view"])):
        g = g[g["fu_defined"] & g["climate"].notna()]
        p = g.loc[g["origin"] == "plant", "climate"].to_numpy()
        a = g.loc[g["origin"] == "animal", "climate"].to_numpy()
        if len(p) < 3 or len(a) < 3:
            continue
        for ind in ["climate", "land", "water"]:
            pv = g.loc[g["origin"] == "plant", ind].to_numpy()
            av = g.loc[g["origin"] == "animal", ind].to_numpy()
            d, lo, hi = st.bootstrap_ci(st.cliffs_delta, pv, av, n_boot=n_boot, seed=seed + i)
            rows.append({"role": role, "view": view, "indicator": ind, "n_plant": len(pv), "n_animal": len(av),
                         "median_plant": np.median(pv), "median_animal": np.median(av),
                         "cliffs_delta": d, "ci_lo": lo, "ci_hi": hi})
    return pd.DataFrame(rows)


def protein_category_effects(vl: pd.DataFrame, reference: str = "legumes", n_boot: int = C.N_BOOT,
                             seed: int = C.SEED) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Cliff's delta of each protein category vs legumes, per view and indicator, plus
    Kruskal-Wallis across categories."""
    rl = role_long(vl)
    rl = rl[(rl["role"] == "protein_main") & rl["protein_category"].notna() & rl["fu_defined"]]
    eff, kw = [], []
    combos = [(v, ind) for v in C.VIEWS for ind in ["climate", "land", "water"]]
    for j, (view, ind) in enumerate(combos):
        g = rl[rl["view"] == view]
        ref = g.loc[g["protein_category"] == reference, ind].to_numpy()
        groups = [x[ind].to_numpy() for _, x in g.groupby("protein_category")]
        H, p, e2 = st.kruskal_epsilon2(groups)
        kw.append({"view": view, "indicator": ind, "H": H, "p_value": p, "epsilon2": e2, "n": len(g)})
        for k, (cat, x) in enumerate(g.groupby("protein_category")):
            if cat == reference:
                continue
            d, lo, hi = st.bootstrap_ci(st.cliffs_delta, x[ind].to_numpy(), ref, n_boot=n_boot, seed=seed + 100 * j + k)
            eff.append({"view": view, "indicator": ind, "category": cat, "n": len(x), "median": x[ind].median(),
                        "median_ref": np.median(ref), "ratio_of_medians": x[ind].median() / np.median(ref),
                        "cliffs_delta_vs_legumes": d, "ci_lo": lo, "ci_hi": hi})
    return pd.DataFrame(eff), pd.DataFrame(kw)


# ---------------------------------------------------------------------------
# non-obvious trade-offs
# ---------------------------------------------------------------------------
def nonobvious_tradeoffs(vl: pd.DataFrame, labels: pd.DataFrame) -> pd.DataFrame:
    """Systematically extract counter-intuitive positions (not hand-picked anecdotes)."""
    d = vl.join(labels[["group", "analysed", "efficient", "efficient_2d", "depth"]])
    d = d[d["analysed"]]
    out = []
    piv = d.pivot_table(index="ciqual_code", columns="view", values="efficient", aggfunc="first")
    names = d.drop_duplicates("ciqual_code").set_index("ciqual_code")[["food_name", "analysis_group"]]

    # 1. normalisation reversal of Pareto status
    for v in ["energy", "protein"]:
        if v not in piv:
            continue
        flip = piv[(piv["mass"] == 1) & (piv[v] == 0)].index
        for c in flip:
            out.append({"type": f"efficient per 100 g, dominated {C.VIEW_UNIT[v]}", "ciqual_code": c})
        flip = piv[(piv["mass"] == 0) & (piv[v] == 1)].index
        for c in flip:
            out.append({"type": f"dominated per 100 g, efficient {C.VIEW_UNIT[v]}", "ciqual_code": c})

    # 2. burden shifting: bottom-quartile climate but top-quartile water in its group (mass view)
    mv = d[d["view"] == "mass"]
    for grp, g in mv.groupby("group"):
        if len(g) < C.MIN_GROUP_SIZE:
            continue
        lowc = g["climate"] <= g["climate"].quantile(.25)
        hiw = g["water"] >= g["water"].quantile(.75)
        hil = g["land"] >= g["land"].quantile(.75)
        for c in g.loc[lowc & hiw, "ciqual_code"]:
            out.append({"type": "low climate but high water within group (burden shift)", "ciqual_code": c})
        for c in g.loc[lowc & hil, "ciqual_code"]:
            out.append({"type": "low climate but high land use within group (burden shift)", "ciqual_code": c})

    # 3. efficient only because of a secondary objective (dominated in 2-D climate/NR9)
    for c in d.loc[(d["view"] == "energy") & (d["efficient"] == 1) & (d["efficient_2d"] == 0), "ciqual_code"]:
        out.append({"type": "dominated on climate+NR9 but efficient once land/water/LIM3 count (per 100 kcal)", "ciqual_code": c})

    # 4. nutrient-dense but environmentally costly: top-quartile NR9 and top-quartile climate (energy view)
    ev = d[d["view"] == "energy"]
    for grp, g in ev.groupby("group"):
        hin = g["nr9"] >= g["nr9"].quantile(.75)
        hic = g["climate"] >= g["climate"].quantile(.75)
        for c in g.loc[hin & hic & (g["efficient"] == 1), "ciqual_code"]:
            out.append({"type": "efficient despite top-quartile climate (high nutrient density)", "ciqual_code": c})

    res = pd.DataFrame(out)
    if res.empty:
        return res
    res = res.join(names, on="ciqual_code")
    return res.drop_duplicates().reset_index(drop=True)


def correlation_matrix(m: pd.DataFrame) -> pd.DataFrame:
    """Spearman correlations between nutrients (per 100 g) and indicators (per 100 g)."""
    cols = C.NUTRIENT_KEYS + [f"{k}_per_100g" for k in C.ENV_CORE]
    corr = m[cols].corr(method="spearman")
    return corr.loc[C.NUTRIENT_KEYS, [f"{k}_per_100g" for k in C.ENV_CORE]]


def pca_biplot_data(m: pd.DataFrame, n_components: int = 2):
    """Joint PCA of robust-scaled nutrients and log environmental indicators (complete cases)."""
    from sklearn.decomposition import PCA

    env_cols = [f"{k}_per_100g" for k in ["climate", "land", "water", "eutro_fw", "acidification"]]
    d = m[m["profile_complete"]].copy()
    sc = nutrition.ProfileScaler().fit(d)
    Zn = sc.transform(d)
    E = np.column_stack([np.arcsinh(d[c].to_numpy() / np.nanmedian(np.abs(d[c]))) for c in env_cols])
    E = (E - np.nanmedian(E, axis=0)) / (np.nanpercentile(E, 75, axis=0) - np.nanpercentile(E, 25, axis=0))
    X = np.column_stack([Zn, E])
    ok = np.isfinite(X).all(axis=1)
    pca = PCA(n_components=n_components, random_state=C.SEED).fit(X[ok])
    scores = pd.DataFrame(pca.transform(X[ok]), columns=[f"PC{i+1}" for i in range(n_components)], index=d.index[ok])
    scores = scores.join(d[["ciqual_code", "food_name", "ciqual_group", "climate_per_100g"]])
    loadings = pd.DataFrame(pca.components_.T, index=C.NUTRIENT_KEYS + env_cols,
                            columns=[f"PC{i+1}" for i in range(n_components)])
    return scores, loadings, pca.explained_variance_ratio_
