"""Deliverable 1 - the Sustainable Food Matrix.

Builds the CIQUAL 2025 x AGRIBALYSE 3.2 integrated table by exact CIQUAL code, with a
source-to-clean data dictionary, stage-by-stage row counts and audit tables. All options
that D5 varies (parser policy, duplicate rule, B12 structural-zero rule) are arguments.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from . import clean
from . import config as C
from . import io
from .io import norm_text

ALL_FEATURES = C.NUTRIENTS + C.EXTENDED_NUTRIENTS


@dataclass
class MatrixBuild:
    matrix: pd.DataFrame
    audit: dict[str, pd.DataFrame] = field(default_factory=dict)
    options: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _txt(s: pd.Series) -> pd.Series:
    return s.fillna("").map(lambda v: " ".join(str(v).split()))


def _tokens(s: str) -> set[str]:
    stop = {"and", "or", "with", "the", "of", "in", "a", "to", "from", "raw", "cooked", "prepacked"}
    return {t for t in re.findall(r"[a-z]+", norm_text(s)) if len(t) > 2 and t not in stop}


def name_jaccard(a: str, b: str) -> float:
    ta, tb = _tokens(a), _tokens(b)
    if not ta or not tb:
        return float("nan")
    return len(ta & tb) / len(ta | tb)


def load_declared_tables() -> dict[str, pd.DataFrame]:
    rd = lambda f: pd.read_csv(C.CONFIG_DIR / f, dtype=str, keep_default_na=False)
    return {
        "group_merges": rd("group_merges.csv"),
        "roles": rd("roles.csv"),
        "baskets": rd("baskets.csv"),
        "structural_zero": rd("structural_zero.csv"),
    }


# ---------------------------------------------------------------------------
# CIQUAL parsing + dictionary
# ---------------------------------------------------------------------------
def parse_ciqual(raw: pd.DataFrame, policy: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Parse every nutrient column; return (parsed table, data dictionary)."""
    cols = list(raw.columns)
    id_cols = cols[:9]
    nutrient_cols = [c for c in cols[9:] if norm_text(c) != "jones factor"]
    selected = {norm_text(h): (k, u, fam, "core feature" if (k, h, u, fam, r) in C.NUTRIENTS else "extended (D5 only)")
                for k, h, u, fam, r in ALL_FEATURES}

    out = pd.DataFrame(index=raw.index)
    codes = [clean.normalize_code(v) for v in raw["alim_code"]]
    out["ciqual_code"] = [c.code for c in codes]
    out["ciqual_code_raw"] = [c.raw for c in codes]
    out["ciqual_code_altered"] = [c.altered for c in codes]
    out["food_name"] = _txt(raw["alim_nom_eng"])
    out["food_name_sci"] = _txt(raw["alim_nom_sci"])
    out["ciqual_group_code"] = _txt(raw["alim_grp_code"])
    out["ciqual_group"] = _txt(raw["alim_grp_nom_eng"])
    out["ciqual_subgroup_code"] = _txt(raw["alim_ssgrp_code"])
    out["ciqual_subgroup"] = _txt(raw["alim_ssgrp_nom_eng"])
    out["ciqual_ssgroup_code"] = _txt(raw["alim_ssssgrp_code"])
    out["ciqual_ssgroup"] = _txt(raw["alim_ssssgrp_nom_eng"])

    dict_rows = []
    for c in id_cols:
        dict_rows.append({"source": "CIQUAL 2025", "original_header": " ".join(str(c).split()),
                          "clean_name": {"alim_code": "ciqual_code", "alim_nom_eng": "food_name",
                                         "alim_nom_sci": "food_name_sci"}.get(c, c.replace("alim_", "ciqual_")),
                          "unit": "", "basis": "", "type": "identifier/label", "use": "identity / grouping"})
    parsed: dict[str, object] = {}
    for c in nutrient_cols:
        label, unit, basis = clean.split_header(c)
        key_info = selected.get(norm_text(c))
        clean_name = key_info[0] if key_info else f"{clean.snake(label)}_{clean.snake(unit) or 'na'}"
        vals, stat = clean.parse_series(raw[c], policy)
        parsed[clean_name] = vals.values
        parsed[f"{clean_name}__status"] = stat.values
        parsed[f"{clean_name}__raw"] = raw[c].values
        note = ""
        if "100mg" in basis:
            note = "header prints '100mg'; CIQUAL values are per 100 g of edible portion"
            basis = "100g"
        dict_rows.append({
            "source": "CIQUAL 2025", "original_header": " ".join(str(c).split()), "clean_name": clean_name,
            "unit": unit, "basis": basis or "100g", "type": "nutrient (parsed float + __status + __raw)",
            "use": key_info[3] if key_info else "parsed, not analysed", "note": note,
        })
    out = pd.concat([out, pd.DataFrame(parsed, index=raw.index)], axis=1)
    return out, pd.DataFrame(dict_rows)


def apply_structural_zero(df: pd.DataFrame, sz: pd.DataFrame) -> pd.Series:
    """Apply the declared structural-zero rules; return a boolean mask per row."""
    applied = pd.Series(False, index=df.index)
    for _, rule in sz.iterrows():
        key = rule["nutrient"]
        groups = set(rule["group_code"].split("|"))
        in_group = df["ciqual_subgroup_code"].isin(groups) | df["ciqual_ssgroup_code"].isin(groups)
        excluded = df["food_name"].str.contains(rule["exclude_name_regex"], case=False, regex=True)
        mask = in_group & ~excluded & (df[f"{key}__status"] == clean.STATUS_MISSING)
        df.loc[mask, key] = 0.0
        df.loc[mask, f"{key}__status"] = "structural_zero"
        applied |= mask
    return applied


# ---------------------------------------------------------------------------
# groups and roles
# ---------------------------------------------------------------------------
def assign_groups(m: pd.DataFrame, merges: pd.DataFrame, min_state_split: int = 3) -> pd.DataFrame:
    m = m.copy()
    ss = m["ciqual_ssgroup_code"]
    use_ss = ss.ne("000000") & ss.ne("")
    m["comp_group_code"] = np.where(use_ss, ss, m["ciqual_subgroup_code"])
    m["comp_group"] = np.where(use_ss, m["ciqual_ssgroup"], m["ciqual_subgroup"])
    m["prep_state"] = m["food_name"].map(clean.preparation_state)

    # declared merges of very small, semantically equivalent groups
    m["group_merged_from"] = ""
    for _, r in merges.iterrows():
        mask = m["comp_group_code"].eq(r["from_code"])
        m.loc[mask, "group_merged_from"] = r["from_code"]
        m.loc[mask, "comp_group_code"] = r["to_code"]
        both = m["comp_group_code"].eq(r["to_code"])
        m.loc[both, "comp_group"] = r["merged_label"]

    # Split groups that mix physical states. Raw, dry and ready-to-eat forms differ mainly
    # in water content, which distorts per-100 g comparisons. Groups whose CIQUAL label
    # already encodes a state, and declared state-compatible groups, are never split.
    m["state_class"] = m["prep_state"].map({"raw": "raw", "dry": "dry"}).fillna("ready")
    m["analysis_group_code"] = m["comp_group_code"]
    m["analysis_group"] = m["comp_group"]
    for code, g in m.groupby("comp_group_code"):
        if code in STATE_SPLIT_EXEMPT or re.search(r"raw|cooked|dried|dehydrated", g["comp_group"].iloc[0]):
            continue
        counts = g["state_class"].value_counts()
        big = counts[counts >= min_state_split]
        if len(big) < 2:
            continue
        dominant = big.index[0]
        for idx, row in g.iterrows():
            st = row["state_class"] if row["state_class"] in big.index else dominant
            m.loc[idx, "analysis_group_code"] = f"{code}:{st}"
            m.loc[idx, "analysis_group"] = f"{row['comp_group']} [{st}]"

    # CIQUAL reuses some sub-sub-group labels under different sub-groups (e.g. 'beef and veal'
    # under both 'cooked meat' and 'raw meat'): qualify such labels with the parent sub-group.
    n_codes = m.groupby("analysis_group")["analysis_group_code"].transform("nunique")
    dup = n_codes > 1
    m.loc[dup, "analysis_group"] = m.loc[dup, "analysis_group"] + " (" + m.loc[dup, "ciqual_subgroup"] + ")"
    for col in ("comp_group",):
        n_codes = m.groupby(col)["comp_group_code"].transform("nunique")
        d = n_codes > 1
        m.loc[d, col] = m.loc[d, col] + " (" + m.loc[d, "ciqual_subgroup"] + ")"
    assert m.groupby("analysis_group")["analysis_group_code"].nunique().max() == 1
    return m


# Groups declared state-compatible (composition per 100 g barely changes with cooking).
STATE_SPLIT_EXEMPT = {"041001"}  # plain eggs: raw and cooked merged on purpose (group_merges.csv)


def assign_roles(m: pd.DataFrame, roles: pd.DataFrame) -> pd.Series:
    out = [[] for _ in range(len(m))]
    sub = m["ciqual_subgroup_code"].to_numpy()
    sss = m["ciqual_ssgroup_code"].to_numpy()
    state = m["prep_state"].to_numpy()
    prot = m["protein_g"].to_numpy()
    for _, r in roles.iterrows():
        g = r["group_code"]
        in_group = (sub == g) | (sss == g)
        if r["state_filter"]:
            in_group &= np.isin(state, r["state_filter"].split("|"))
        if r["protein_min_g"]:
            in_group &= np.nan_to_num(prot, nan=-1) >= float(r["protein_min_g"])
        for i in np.where(in_group)[0]:
            if r["role"] not in out[i]:
                out[i].append(r["role"])
    return pd.Series(["|".join(x) for x in out], index=m.index)


# ---------------------------------------------------------------------------
# main build
# ---------------------------------------------------------------------------
def build_matrix(parse_policy: str = C.PARSE_POLICY, dup_rule: str = "reference",
                 structural_zero: bool = True) -> MatrixBuild:
    tables = load_declared_tables()
    stages = []

    raw_c = io.load_ciqual_raw()
    stages.append(("CIQUAL 2025 workbook rows", len(raw_c), "food composition sheet"))
    ciq, dictionary = parse_ciqual(raw_c, parse_policy)
    bad_codes = ciq["ciqual_code"].isna().sum()
    dup_c = ciq["ciqual_code"].duplicated().sum()
    stages.append(("CIQUAL rows with valid integer code", int(ciq["ciqual_code"].notna().sum()), f"{bad_codes} invalid"))
    stages.append(("CIQUAL unique codes", int(ciq["ciqual_code"].nunique()), f"{dup_c} duplicated codes"))
    if dup_c:
        raise ValueError("CIQUAL codes are expected to be unique")

    raw_a = io.load_agribalyse_raw()
    stages.append(("AGRIBALYSE 3.2 Synthese data rows", len(raw_a), "rows with an AGB or CIQUAL code"))
    stages.append(("AGRIBALYSE unique AGB codes", int(raw_a["agb_code"].nunique()), "7 codes repeated for packaging fix"))
    agb, dup_audit = clean.resolve_agribalyse_duplicates(raw_a, rule=dup_rule)
    n_a = (dup_audit["tier"] == "A_packaging_fix").sum()
    n_b = dup_audit["tier"].str.startswith("B_variant").sum()
    stages.append(("AGRIBALYSE unique CIQUAL codes (after duplicate resolution)", len(agb),
                   f"{n_a} packaging-error rows dropped; {n_b} variant rows resolved ({dup_rule})"))
    if agb["ciqual_code"].duplicated().any():
        raise AssertionError("duplicate resolution failed")

    # exact join
    cset, aset = set(ciq["ciqual_code"]), set(agb["ciqual_code"])
    matched_codes = cset & aset
    m = ciq[ciq["ciqual_code"].isin(matched_codes)].merge(
        agb.drop(columns=["ciqual_code_raw"]).rename(columns={"agb_code": "agb_code"}),
        on="ciqual_code", how="inner", validate="one_to_one")
    stages.append(("Exact matches (inner join on normalised CIQUAL code)", len(m),
                   f"{100 * len(m) / len(aset):.2f}% of AGRIBALYSE codes; {100 * len(m) / len(cset):.2f}% of CIQUAL codes"))
    ciqual_only = ciq[~ciq["ciqual_code"].isin(aset)][["ciqual_code", "food_name", "ciqual_group", "ciqual_subgroup"]]
    agb_only = agb[~agb["ciqual_code"].isin(cset)][["ciqual_code", "agb_code", "agb_lci_name", "agb_group_fr", "agb_subgroup_fr"]]
    stages.append(("CIQUAL-only codes (no AGRIBALYSE record)", len(ciqual_only), ""))
    stages.append(("AGRIBALYSE-only codes (not in CIQUAL 2025)", len(agb_only), ""))
    assert len(m) + len(ciqual_only) == len(cset)
    assert len(m) + len(agb_only) == len(aset)

    # structural zeros
    m["b12_structural_zero"] = False
    if structural_zero:
        m["b12_structural_zero"] = apply_structural_zero(m, tables["structural_zero"]).values

    # environmental unit harmonisation: AGRIBALYSE is per kg consumed, CIQUAL per 100 g
    for k in C.ENV_KEYS:
        m[f"{k}_per_100g"] = m[f"{k}_per_kg"] / 10.0

    # groups, roles, flags
    m = assign_groups(m, tables["group_merges"])
    m["roles"] = assign_roles(m, tables["roles"])
    flags = clean.known_issue_flags(m)
    m["issue_direct"], m["issue_indirect"] = flags["issue_direct"], flags["issue_indirect"]
    m["name_token_jaccard"] = [name_jaccard(a, b) for a, b in zip(m["food_name"], m["agb_lci_name"])]

    # completeness indicators (no imputation)
    prof = m[C.NUTRIENT_KEYS]
    m["n_profile_available"] = prof.notna().sum(axis=1)
    m["profile_complete"] = m["n_profile_available"].eq(len(C.NUTRIENT_KEYS))
    m["graph_eligible"] = m["n_profile_available"].ge(C.MIN_SHARED_FEATURES) & m["energy_kcal"].notna()
    m["nr_complete"] = m[C.NR_KEYS + C.LIM_KEYS].notna().all(axis=1) & m["energy_kcal"].notna()
    stages.append(("Matched foods with complete 15-nutrient profile", int(m["profile_complete"].sum()),
                   f"B12 structural zeros applied to {int(m['b12_structural_zero'].sum())} foods" if structural_zero else "structural-zero rule OFF"))
    stages.append(("Matched foods eligible for the substitution graph (>= 12/15 nutrients)", int(m["graph_eligible"].sum()), ""))
    stages.append(("Matched foods with complete NR9 + LIM3 components", int(m["nr_complete"].sum()), ""))

    m = m.sort_values(["comp_group_code", "ciqual_code"]).reset_index(drop=True)

    # missingness table (selected + extended features, matched set)
    miss_rows = []
    for k, h, u, fam, r in ALL_FEATURES:
        st = m[f"{k}__status"]
        miss_rows.append({"feature": k, "unit": f"{u}/100 g", "family": fam,
                          "core": k in C.NUTRIENT_KEYS,
                          **{s: int((st == s).sum()) for s in ["numeric", "trace", "below_loq", "missing", "structural_zero"]},
                          "coverage_pct": round(100 * m[k].notna().mean(), 1)})
    missingness = pd.DataFrame(miss_rows)

    # group coverage: share of each CIQUAL group matched
    cov = (ciq.assign(matched=ciq["ciqual_code"].isin(aset))
           .groupby(["ciqual_group_code", "ciqual_group"])["matched"].agg(["size", "sum"]).reset_index()
           .rename(columns={"size": "ciqual_foods", "sum": "matched_foods"}))
    cov["coverage_pct"] = (100 * cov["matched_foods"] / cov["ciqual_foods"]).round(1)

    # environmental dictionary rows
    for k, fr, unit, label in C.ENV:
        dictionary = pd.concat([dictionary, pd.DataFrame([{
            "source": "AGRIBALYSE 3.2", "original_header": fr, "clean_name": f"{k}_per_kg | {k}_per_100g",
            "unit": unit, "basis": "1 kg consumed (per_kg); /10 for per_100g", "type": "environmental indicator (EF 3.1)",
            "use": "core indicator" if k in C.ENV_CORE else "sensitivity only (aggregated score)", "note": label}])])
    for col, desc in [("dqr", "Data Quality Rating, ordinal 1 (excellent) to 5 (very poor); not a probability"),
                      ("season_code", "0 off-season, 1 in-season, 2 French consumption mix"),
                      ("plane_code", "1 = air-freighted"), ("row_font_red", "publisher red-font row (known error)")]:
        dictionary = pd.concat([dictionary, pd.DataFrame([{"source": "AGRIBALYSE 3.2", "original_header": col,
                                                           "clean_name": col, "type": "metadata", "use": "quality / provenance",
                                                           "note": desc}])])

    audit = {
        "stage_counts": pd.DataFrame(stages, columns=["stage", "count", "note"]),
        "duplicate_resolution": dup_audit,
        "ciqual_only": ciqual_only.reset_index(drop=True),
        "agribalyse_only": agb_only.reset_index(drop=True),
        "missingness": missingness,
        "group_coverage": cov,
        "data_dictionary": dictionary.reset_index(drop=True),
    }
    return MatrixBuild(m, audit, {"parse_policy": parse_policy, "dup_rule": dup_rule, "structural_zero": structural_zero})


def matrix_columns(m: pd.DataFrame) -> list[str]:
    """Columns written to the published processed matrix (drops non-feature parsed nutrients)."""
    keep_feat = [k for k, *_ in ALL_FEATURES]
    base = ["ciqual_code", "ciqual_code_raw", "food_name", "food_name_sci", "ciqual_group_code", "ciqual_group",
            "ciqual_subgroup_code", "ciqual_subgroup", "ciqual_ssgroup_code", "ciqual_ssgroup",
            "comp_group_code", "comp_group", "group_merged_from", "analysis_group_code", "analysis_group",
            "prep_state", "state_class", "roles", "agb_code", "agb_lci_name", "agb_name_fr", "agb_group_fr", "agb_subgroup_fr",
            "n_agb_variants", "dup_tier", "season_code", "plane_code", "delivery", "packaging_approach", "preparation",
            "dqr", "row_font_red", "issue_direct", "issue_indirect", "name_token_jaccard"]
    feats = []
    for k in keep_feat:
        feats += [k, f"{k}__status", f"{k}__raw"]
    env = [f"{k}_per_kg" for k in C.ENV_KEYS] + [f"{k}_per_100g" for k in C.ENV_KEYS]
    tail = ["b12_structural_zero", "n_profile_available", "profile_complete", "graph_eligible", "nr_complete"]
    return [c for c in base + feats + env + tail if c in m.columns]


def load_matrix() -> pd.DataFrame:
    """Load the published processed matrix with correct dtypes."""
    if C.MATRIX_PARQUET.exists():
        return pd.read_parquet(C.MATRIX_PARQUET)
    return pd.read_csv(C.MATRIX_FILE, dtype={"ciqual_code": str, "comp_group_code": str,
                                             "analysis_group_code": str, "ciqual_subgroup_code": str,
                                             "ciqual_ssgroup_code": str, "ciqual_group_code": str},
                       keep_default_na=True)
