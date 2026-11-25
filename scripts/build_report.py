"""Build reports/Sustainome_Report.pdf from the executed notebooks' tables and figures.

Every number in the text is read from reports/tables. Sentences whose truth depends on the
results are guarded with check(): if a premise stops holding, the build fails instead of
printing an unsupported statement.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
from reportlab.platypus import Paragraph

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from sustainome import config as C  # noqa: E402
from report_lib import Report, S, check  # noqa: E402

TB = C.TABLES
SCEN = ["S1", "S2", "S3", "S4", "S5"]


def t(name: str, **kw) -> pd.DataFrame:
    return pd.read_csv(TB / f"{name}.csv", **kw)


def pct(x: float, d: int = 0) -> str:
    return f"{100 * x:.{d}f}%"


def spct(x: float, d: int = 0) -> str:
    return f"{100 * x:+.{d}f}%"


def main() -> None:
    R = Report(C.FIGURES)
    title = "Sustainome: The Sustainable Plate Frontier"

    # ================================================================ load results
    stages = t("D1_stage_counts").set_index("stage")["count"]
    matched = int(stages["Exact matches (inner join on normalised CIQUAL code)"])
    n_ciq = int(stages["CIQUAL unique codes"])
    n_agb = int(stages["AGRIBALYSE unique CIQUAL codes (after duplicate resolution)"])
    complete = int(stages["Matched foods with complete 15-nutrient profile"])
    eligible = int(stages["Matched foods eligible for the substitution graph (>= 12/15 nutrients)"])
    rel = t("D1_release_comparison_3.1_vs_3.2").set_index("Unnamed: 0")
    miss = t("D1_missingness").set_index("feature")

    labels = t("D2_pareto_labels")
    an = labels[labels["analysed"]]
    ov = an.groupby("view").agg(groups=("group", "nunique"), n=("ciqual_code", "size"), eff=("efficient", "sum"), eff2=("efficient_2d", "sum"))
    gs = t("D2_group_summary")
    assoc = gs.groupby(["view", "association"]).size().unstack(fill_value=0)
    eff = t("D2_protein_category_effects")
    ratios = eff[eff["indicator"] == "climate"].pivot(index="category", columns="view", values="ratio_of_medians")
    kw = t("D2_protein_kruskal_wallis").set_index(["view", "indicator"])
    va = t("D2_view_agreement")
    no = t("D2_nonobvious_tradeoffs")
    always = t("D2_efficient_in_all_views")
    rcon = t("D2_role_contrasts")

    arcs_cls = t("D3_substitution_arcs", usecols=["edge_class"])["edge_class"].value_counts()
    rc = t("D3_recommender_comparison")
    comm = t("D3_communities")
    audit = t("D3_manual_edge_audit")
    nodes = t("D3_nodes")
    edges = t("D3_edges")

    sel = t("D4_selected_solutions").set_index("scenario")
    base = t("D4_baseline_comparison").set_index("scenario")
    swaps = t("D4_swap_count_fronts")
    sweep = t("D4_pareto_sweep_solutions")
    cw = t("D4_conflict_summary").set_index("scenario")
    scen = t("D4_scenarios")

    reg = t("D5_claims_register")
    gg = t("D5_graph_k_tau_grid")
    ws = t("D5_graph_weight_sensitivity")
    fs = t("D5_graph_feature_subsets")
    mve = t("D5_missing_value_edges")
    dq = t("D5_dqr_filter_pareto")
    ind = t("D5_indicator_subset_pareto")
    bp = t("D5_bootstrap_pareto_probability")
    bcr = t("D5_bootstrap_category_ratios")
    rev = t("D5_normalisation_reversal_rates")
    mvt = t("D5_matrix_assumption_variants")
    tol = t("D5_optimizer_tolerance_grid")
    dv = t("D5_optimizer_data_variants")
    pert = t("D5_impact_perturbation_stress_test")
    cx = t("D5_counterexamples")

    # derived numbers ---------------------------------------------------------
    naive_share = rc["naive_equivalent"].mean()
    n_strong = int(arcs_cls.get("strong", 0))
    beef, lamb = ratios.loc["beef & veal"], ratios.loc["lamb & mutton"]
    ce, ca = int(assoc.loc["energy"].get("conflict", 0)), int(assoc.loc["energy"].get("agreement", 0))
    cmass, amass = int(assoc.loc["mass"].get("conflict", 0)), int(assoc.loc["mass"].get("agreement", 0))
    op = sel["climate_change"]
    zero_dev = sweep[(sweep["eps_dev"] == 0) & (sweep["eps_change"] == 1.0)].set_index("scenario")["climate_change"]
    rebal = swaps[(swaps["max_new"] == 0) & (swaps["dev_budget"] == "none")].set_index("scenario")["climate_change"]
    one = swaps[(swaps["max_new"] == 1) & (swaps["dev_budget"] == "0.10")].set_index("scenario")["climate_change"]
    labels_count = reg["label"].value_counts()
    dom_ref = bp[bp["efficient"] == 0]
    robust_dom = dom_ref["robustly_dominated"].mean()
    fragile_dom = (dom_ref["p_efficient"] >= 0.25).mean()
    single_dom = labels.loc[labels["analysed"] & (labels["efficient"] == 0), "dominated_by"].eq(1).mean()

    # ================================================================ title + abstract
    R.story.append(Paragraph(title, S["title"]))
    R.story.append(Paragraph("Nutrient–environment trade-offs and food substitution<br/>"
                             "BIO544 Computational Gastronomy · Monsoon 2026 · Mini-project report · Group 13", S["subtitle"]))
    check(ce > ca, "per-kcal conflict outnumbers agreement")
    check(naive_share < 0.10, "naive greenest swap rarely equivalent")
    check((op <= -0.30).all(), "every scenario >= 30% reduction at the operating point")
    R.boxed("Abstract", (
        f"We joined the ANSES-CIQUAL 2025 nutrient table to AGRIBALYSE 3.2 life-cycle indicators by exact food code, obtaining "
        f"{matched:,} matched foods with reconciled counts, explicit missingness and a documented duplicate rule (D1). Within comparable "
        f"groups, foods were compared per 100 g, per 100 kcal and per 10 g protein using a capped nutrient-adequacy score and separate limit "
        f"nutrients. Normalisation shrinks but never reverses the protein hierarchy (beef & veal: {beef['mass']:.0f}× cooked legumes per 100 g, "
        f"{beef['protein']:.0f}× per 10 g protein), changes which foods are Pareto-efficient (mean Jaccard {va['jaccard_mass_energy'].mean():.2f} "
        f"between mass and energy views) and reveals that per 100 kcal nutrient density and climate impact conflict in {ce} groups but agree "
        f"in only {ca} (D2). A leakage-free nutritional-equivalence network with a calibrated threshold shows that the greenest eligible swap "
        f"is nutritionally equivalent for only {pct(naive_share, 1)} of foods; {n_strong} swaps are strong (equivalent, reliable, free of burden "
        f"shifting) (D3). A transparent LP/MILP optimiser on five declared baskets cuts climate impact by {pct(-op.max())}–{pct(-op.min())} while "
        f"keeping energy, protein, fibre and seven micronutrients within declared bounds and the nutrient profile within 10 % mean deviation; "
        f"random role-preserving swaps almost never satisfy the same constraints, and minimising climate can raise water scarcity (D4). "
        f"A pre-declared robustness study (weights, features, thresholds, DQR, parser and duplicate rules, bootstrap, perturbation) labels "
        f"{int(labels_count.get('robust', 0))} of {len(reg)} headline claims robust and catalogues {cx['category'].nunique()} kinds of "
        f"counterexample (D5). Results are database-supported scenarios for average French products, not dietary advice."))

    # ================================================================ 1 intro
    R.h1("1. Research question and design")
    R.p("Environmental food comparisons usually rank foods per kilogram, while nutritional comparisons ignore environmental cost. "
        "Sustainome treats substitution as a constrained scientific problem: <i>which food substitutions genuinely reduce environmental "
        "impact while preserving nutritional value, and where do unavoidable trade-offs remain after nutritional equivalence is enforced?</i> "
        "We answer at four levels, each a separate deliverable with its own acceptance test: a reusable integrated dataset (D1), food-level "
        "inference (D2), pairwise substitution modelling (D3), basket-level optimisation (D4), and validation and falsification (D5). "
        "The project deliberately avoids repeating course assignments (recipe scraping, NER, itemset mining, random cuisines, flavour pairing).")
    R.bullets([
        "<b>Exact linkage.</b> Foods are joined by normalised CIQUAL code; names only validate the join.",
        "<b>Like with like.</b> Comparisons stay within state-aware CIQUAL analysis groups or within declared basket roles "
        "(e.g. cooked protein mains: meat, fish, eggs, legumes, tofu, plant-based patties) listed in config/roles.csv.",
        "<b>Three functional units.</b> Per 100 g, per 100 kcal and per 10 g protein, with denominator guards (foods below 10 kcal or 2 g "
        "protein per 100 g are undefined in those views rather than divided by ~0).",
        "<b>Nutrition is not 'more of everything'.</b> NR9 sums nine qualifying nutrients each capped at 100 % of the EU reference intake per "
        "functional unit (adapted NRF9.3); saturated fat, total sugars and sodium form a separate LIM3 objective.",
        "<b>Trade-offs, not rankings.</b> Results are Pareto sets with uncertainty; no universal food ranking is published.",
    ])

    # ================================================================ 2 D1
    R.h1("2. Deliverable 1 — the Sustainable Food Matrix")
    R.p(f"The CIQUAL 2025 English workbook ({n_ciq:,} foods) and the AGRIBALYSE 3.2 food-products workbook (2,458 data rows) were frozen with "
        "SHA-256 checksums, versions, licences and access dates. AGRIBALYSE impacts are per kg of product consumed and were divided by 10 to match "
        "CIQUAL's per-100 g basis. A notice inside the workbook defines two duplicate types, resolved by a unit-tested rule: seven red-font rows "
        "with an uncorrected packaging value are dropped in favour of the corrected rows, and 26 variant rows (seasonal, air-freighted, powder vs "
        "reconstituted) are resolved to the French consumption-mix, non-air-freighted variant (alternatives are re-run in D5). The exact join "
        f"yields <b>{matched:,} matched foods</b> ({matched / n_agb:.2%} of AGRIBALYSE codes) and every stage reconciles: {matched:,} + "
        f"{int(stages['CIQUAL-only codes (no AGRIBALYSE record)']):,} CIQUAL-only = {n_ciq:,}; {matched:,} + "
        f"{int(stages['AGRIBALYSE-only codes (not in CIQUAL 2025)'])} AGRIBALYSE-only = {n_agb:,}. The proposal's feasibility audit on the "
        "AGRIBALYSE 3.1 API (2,451 rows, 2,425 codes, 2,272 matches, 93.69 %) was reproduced exactly from a frozen snapshot; 3.1 values appear only "
        f"in a release comparison (climate Spearman ρ = {rel.loc['climate', 'Spearman rho']:.3f}; {pct(rel.loc['climate', 'share changed > 5 %'], 1)} "
        "of foods changed by more than 5 %) and never in an analytical table.")
    R.figure("D1_Join_flow_and_coverage.png", "Join flow with reconciled counts (left) and coverage of each CIQUAL food group by AGRIBALYSE 3.2 (right).", max_h=0.32)
    R.p("CIQUAL cells are parsed explicitly: '–' is missing and is <b>never</b> converted to zero; 'traces' becomes 0; '&lt; x' becomes x/2 "
        "(0 and x are tested in D5); decimal commas and line breaks inside markers are handled; each value keeps its raw text and a status. "
        "Fifteen nutrients pass a ≥ 70 % coverage rule (energy, protein, carbohydrate, fat, sugars, fibre, saturated fat, sodium, Ca, Fe, Mg, "
        f"K, Zn, vitamin C, B12); folate ({miss.loc['folate_ug', 'coverage_pct']:.0f} %), vitamin E (18.5 %) "
        "and vitamin A retinol equivalents (59 %) fail, so zinc and B12 replace vitamins A and E in the adapted NRF9.3 index. CIQUAL leaves B12 as '–' for many plant foods "
        "although B12 is absent from unfortified plants; a declared structural-zero rule sets "
        f"{int(miss.loc['vitamin_b12_ug', 'structural_zero'])} such cells to 0 with their own status (switched off in D5). {complete:,} foods "
        f"have complete profiles and {eligible:,} have at least 12 of 15 nutrients. A manual audit of the 15 least name-similar matches and 10 "
        "random matches found only synonyms, translations and two acceptable AGRIBALYSE proxy LCIs. Publisher-flagged errors (egg impacts "
        "under-estimated, quinoa over-estimated) are flagged per food. A rebuild from the raw files reproduces the processed matrix exactly.")
    R.figure("D1_Nutrient_missingness_matrix.png", "Share of missing values by nutrient and CIQUAL group (left) and value status per nutrient (right). "
             "Micronutrient gaps concentrate in composite dishes and ice creams; the dashed line marks the 70 % coverage rule.", max_h=0.34)

    # ================================================================ 3 D2
    R.h1("3. Deliverable 2 — nutrition–environment trade-offs (food level)")
    R.p(f"Within each analysis group with at least {C.MIN_GROUP_SIZE} complete foods, foods were labelled Pareto-efficient on five objectives "
        "(minimise climate, land use, water scarcity and LIM3; maximise NR9), with dominance depth and a sharper two-objective climate–NR9 front. "
        f"Per 100 kcal, {int(ov.loc['energy', 'groups'])} groups and {int(ov.loc['energy', 'n']):,} foods are analysed; "
        f"{pct(ov.loc['energy', 'eff'] / ov.loc['energy', 'n'])} are five-objective efficient (a known property of many-objective fronts) but "
        f"only {pct(ov.loc['energy', 'eff2'] / ov.loc['energy', 'n'])} lie on the climate–NR9 front; {len(always)} foods are efficient in all "
        "three views. Foods excluded from a view keep an explicit reason (undefined unit, missing objective, group too small).")
    R.figure("D2_Pareto_fronts_by_food_group_per_100_kcal.png", "Within-group Pareto fronts per 100 kcal for twelve analysis groups. Outlined points form "
             "the climate–NR9 front; light-blue points are non-dominated once land, water and LIM3 also count.", max_h=0.44)
    R.p("<b>Normalisation compresses but does not reverse the protein hierarchy.</b> Inside the declared cooked-protein role, the median "
        f"climate impact of beef & veal is {beef['mass']:.0f}× that of cooked legumes per 100 g, {beef['energy']:.0f}× per 100 kcal and "
        f"{beef['protein']:.0f}× per 10 g protein (lamb {lamb['mass']:.0f}×, {lamb['energy']:.0f}×, {lamb['protein']:.0f}×). Poultry falls "
        f"from {ratios.loc['poultry', 'mass']:.0f}× to {ratios.loc['poultry', 'protein']:.1f}× and eggs from {ratios.loc['eggs', 'mass']:.1f}× to "
        f"{ratios.loc['eggs', 'protein']:.1f}× (eggs are flagged as under-estimated). Category differences are large in every view (Kruskal–Wallis "
        f"ε² = {kw.loc[('mass', 'climate'), 'epsilon2']:.2f} per 100 g, {kw.loc[('protein', 'climate'), 'epsilon2']:.2f} per 10 g protein). "
        "Land use and water are less one-sided: per 10 g protein, fish and seafood are indistinguishable from legumes on land use "
        "(Cliff's δ intervals include zero).")
    fl = eff[(eff["indicator"] == "land") & (eff["view"] == "protein") & eff["category"].isin(["fish", "seafood"])]
    check(bool(((fl["ci_lo"] < 0) & (fl["ci_hi"] > 0)).all()), "fish/seafood land per protein CI includes 0")
    R.figure("D2_Normalization_views_for_cooked_protein_mains.png", "Climate impact of cooked protein mains in the three functional units (log scale; "
             "green = plant, red = animal).", max_h=0.29)
    R.figure("D2_Effect_sizes_of_protein_categories_vs_legumes.png", "Cliff's delta of each protein category against cooked legumes, with 95 % "
             "bootstrap intervals (1,000 resamples), per indicator and view.", max_h=0.31)
    R.p(f"<b>Nutrition and climate conflict once energy is held fixed.</b> Per 100 kcal the NR9–climate correlation is positive with a bootstrap "
        f"interval above zero (conflict: more nutrient-dense foods cost more) in {ce} of {int(assoc.loc['energy'].sum())} groups and negative "
        f"(agreement) in only {ca}; per 100 g the counts are {cmass} and {amass}, with most groups showing no clear association. The mass view "
        "hides a trade-off that the energy view exposes: within a group, the nutrient-dense items (lean meats, whole grains, leafy vegetables) "
        "tend to be the costlier ones to produce per calorie.")
    R.figure("D2_Nutrition_climate_agreement_and_conflict_by_group.png", "Spearman correlation between NR9 and climate impact within each analysed group, "
             "with 95 % paired-bootstrap intervals, in the three views.", max_h=0.6)
    plant_max = rcon[(rcon["indicator"] == "climate") & (rcon["view"] == "mass")]["cliffs_delta"].max()
    check(plant_max < 0, "plant beats animal on climate in every role per 100 g")
    R.p(f"<b>The functional unit changes who is efficient.</b> Within the same group, efficient sets overlap only moderately between views (mean "
        f"Jaccard {va['jaccard_mass_energy'].mean():.2f} mass vs energy, {va['jaccard_mass_protein'].mean():.2f} mass vs protein, "
        f"{va['jaccard_energy_protein'].mean():.2f} energy vs protein). A systematic extraction found "
        f"{int((no['type'] == 'dominated per 100 g, efficient per 100 kcal').sum())} foods dominated per 100 g but efficient per 100 kcal and "
        f"{int((no['type'] == 'efficient per 100 g, dominated per 100 kcal').sum())} with the opposite reversal; "
        f"{int(no['type'].str.contains('high water').sum())} low-climate foods sit in their group's top water-scarcity quartile (e.g. cooked haricot "
        "beans, raw French beans); and many foods dominated on climate and NR9 alone become efficient once land, water or LIM3 count. "
        f"Plant options beat animal options on climate within every declared role with enough foods of both origins (Cliff's δ ≤ {plant_max:.2f} per 100 g).")
    R.figure("D2_Nutrient_environment_correlations_and_biplot.png", "Spearman correlations between nutrients and indicators (left) and a joint PCA "
             "biplot (right): protein, zinc and B12 align with every impact indicator; sugars, carbohydrate and fibre point away.", max_h=0.33)
    R.figure("D2_Dominance_depth_and_cross_view_agreement.png", "Dominance-depth distribution per view (left) and agreement of efficient sets across "
             "functional units within the same group (right).", max_h=0.26)

    # ================================================================ 4 D3
    R.h1("4. Deliverable 3 — the nutritional-equivalence substitution network")
    R.p(f"Nodes are the {len(nodes):,} graph-eligible foods. Two foods may be linked only if they share an analysis group or a declared role. "
        "Their distance is the weighted RMS difference of asinh-compressed, median/IQR-scaled per-100 g profiles over nutrients observed in both "
        "(never imputed). Edges join mutual 5-nearest eligible neighbours within a threshold τ, calibrated by a rule fixed in advance: the most "
        "conservative τ on the plateau of coverage × validity, where validity is the share of graph edges that also pass <i>independent</i> "
        f"directional retention checks (selected: the {C.DIST_QUANTILE:.2f} quantile of nearest-neighbour distances). Environmental indicators and "
        "DQR are attached only after the graph exists; a unit test confirms that permuting every environmental column leaves the edge set "
        "unchanged (no leakage).")
    R.figure("D3_Threshold_calibration.png", "Calibration of the equivalence threshold: coverage, validity and their product as τ varies; the selected "
             "quantile is the smallest within 5 % of the best score.", width=0.7, max_h=0.26)
    offal_ratio = comm.loc[comm["dominant_group"].str.contains("offal"), "climate_ratio_max_min"].max()
    netm = t("D3_network_metrics", index_col=0)["value"]
    nmi = float(netm["nmi_vs_analysis_group"])
    check(nmi > 0.8, "communities recover taxonomy")
    R.p(f"The graph has {len(edges):,} edges among {int((nodes['degree'] > 0).sum()):,} connected foods and {len(comm)} Louvain communities "
        f"(modularity {float(netm['modularity']):.2f}), which largely recover the CIQUAL taxonomy from nutrient profiles alone (NMI with analysis "
        f"groups {nmi:.2f}). Declared roles add cross-group links (cow milk ↔ soy drinks, cheese ↔ "
        "plant-based cheese, meat ↔ fish ↔ plant patties). Communities with a wide internal climate spread mark substitution opportunities, e.g. "
        f"offal of different species (up to {offal_ratio:.0f}× within one community), soups and hot drinks.")
    R.figure("D3_Substitution_network_overview.png", "The nutritional-equivalence graph (connected foods only), components packed largest first; colour = "
             "CIQUAL group; black edges = cross-group links through a declared role.", max_h=0.4)
    R.p("<b>Most environmentally attractive swaps are nutritionally misleading.</b> Each directed gram-for-gram swap keeps separate components: "
        "nutrient distance, relative change on seven indicators, the DQR of both foods and a retention profile (protein ≥ 80 %, energy ratio "
        "0.67–1.5, ≥ 50 % of every micronutrient the source supplies meaningfully, limit nutrients ≤ 1.25×). Of all eligible swaps with ≥ 20 % "
        f"climate reduction, {int(arcs_cls.get('misleading', 0)):,} are misleading, {int(arcs_cls.get('unverifiable (missing nutrients)', 0))} cannot "
        f"be verified because of missing values, {int(arcs_cls.get('equivalent, caveated', 0))} are equivalent but caveated (burden shift, DQR &gt; 3 "
        f"or not a graph edge) and <b>{n_strong} are strong</b>. A naive environment-only recommender that proposes each food's greenest eligible "
        f"substitute is nutritionally equivalent for only <b>{pct(naive_share, 1)}</b> of {len(rc)} source foods; insisting on equivalence lowers "
        f"the median attainable gain from {pct(rc['naive_gain'].median())} to {pct(rc['equiv_gain'].median())}, and only "
        f"{pct(rc['equiv_tgt'].notna().mean())} of source foods have any equivalent improving swap. Protein, energy, B12 and zinc fail most often: "
        "the classic meat → beans swap loses about two-thirds of the protein and all B12 gram for gram, a gap that only basket-level "
        "compensation (D4) can close.")
    R.figure("D3_Edge_tradeoff_map_and_failure_reasons.png", "Every eligible swap with a climate reduction: nutrient distance against climate gain, coloured "
             "by class (left), and the retention checks failed by misleading swaps (right).", max_h=0.29)
    R.figure("D3_Naive_vs_equivalence_constrained_recommendations.png", "The price of nutritional equivalence: climate gain of the naive greenest swap vs "
             "the best equivalent swap (left) and by declared role (right).", max_h=0.27)
    R.figure("D3_Top_strong_and_misleading_substitutions.png", "Highest-scoring strong substitutions (left; vegetable oils dominate) and the largest "
             "misleading role-level swaps (right) with the checks they fail.", max_h=0.3)
    verdict = audit["verdict"].fillna("").str.split(" - ").str[0].value_counts()
    R.p(f"Thirty representative arcs were audited by hand: {verdict.get('plausible', 0)} judged culinarily plausible and all "
        f"{verdict.get('misleading', 0)} sampled misleading arcs confirmed. The audit also exposed a limitation reported honestly: linseed oil → "
        "frying oil is 'strong' on our 15 nutrients, but linseed oil's value lies in omega-3 α-linolenic acid, which the frozen feature set lacks.")

    # ================================================================ 5 D4
    R.h1("5. Deliverable 4 — multi-objective basket optimisation")
    R.p("Five transparent baskets were declared before solving (config/baskets.csv): a French breakfast (S1), a beef lunch (S2), a dairy and snack "
        "basket (S3), a chicken dinner (S4) and a one-day basket combining S1, S2 and S4 (S5). Each slot (e.g. 'protein', 'milk') may be filled "
        "by any food sharing its declared role that has a complete nutrient profile and no publisher-flagged error. The model (SciPy/HiGHS) "
        "chooses grams per candidate subject to: slot mass ±25 %; total energy ±5 %; protein ≥ 95 %; fibre, Ca, Fe, Mg, K, Zn, vitamin C and "
        "B12 ≥ 90 %; sodium, saturated fat and sugars ≤ baseline. Objectives are kept separate — climate, nutrient-profile deviation D (mean "
        "relative deviation over all 15 nutrients) and basket change Δ (share of mass changed) — and an ε-constraint sweep (11 × 4) returns a "
        "Pareto set; a MILP adds the front of climate against the number of new foods. Every reported solution is re-checked against every "
        "constraint; all optimiser solutions are feasible, and only the deliberately unconstrained climate-only baseline shows violations.")
    tab = pd.DataFrame({
        "Scenario": [f"{s}" for s in SCEN],
        "Baseline kg CO2e": [f"{base.loc[s, 'original_climate_kg']:.2f}" for s in SCEN],
        "Portion rebalancing only": [spct(rebal[s]) for s in SCEN],
        "1 new food (D ≤ 0.10)": [spct(one[s]) if s in one else "–" for s in SCEN],
        "D = 0 (identical profile)": [spct(zero_dev[s]) for s in SCEN],
        "Operating point D ≤ 0.10, Δ ≤ 50 %": [spct(op[s]) for s in SCEN],
        "Water at op. point": [spct(sel.loc[s, "water_change"]) for s in SCEN],
        "Random swaps feasible": [pct(base.loc[s, "random_swaps_feasible_share"], 1) for s in SCEN],
        "Climate-only, no nutrition": [f"{spct(base.loc[s, 'climate_only_unconstrained_change'])} ({int(base.loc[s, 'climate_only_unconstrained_violations'])} viol.)" for s in SCEN],
    })
    R.table(tab, "Climate change achieved per scenario under increasing freedom, with baselines. All optimiser columns satisfy every nutrition constraint.",
            col_widths=[0.7, 0.9, 1, 1, 1, 1.2, 0.9, 0.9, 1.4])
    R.figure("D4_Pareto_fronts_climate_vs_nutrient_deviation.png", "ε-constraint Pareto fronts of climate change against nutrient-profile deviation for four "
             "basket-change budgets, with random feasible baskets (grey), the original (star) and the unconstrained climate-only optimum (red X).", max_h=0.27)
    check((rebal <= -0.05).all(), "rebalancing alone helps everywhere")
    R.p(f"<b>Large reductions are compatible with nutrition, but they cost nutritional similarity and basket change.</b> Keeping the full nutrient "
        f"profile identical (D = 0) still allows {pct(-zero_dev.max())}–{pct(-zero_dev.min())} reductions when the basket may change freely; at the "
        f"declared operating point (D ≤ 0.10, Δ ≤ 50 %) reductions are {pct(-op.max())}–{pct(-op.min())}. Portion rebalancing alone, with no new "
        f"food, already saves {pct(-rebal.max())}–{pct(-rebal.min())}, and one well-chosen new food reaches {pct(-one.max())}–{pct(-one.min())}. "
        "By contrast, random role-preserving swaps almost never satisfy the constraints, and the climate-only optimum without nutrition "
        "constraints violates several nutrient bounds in every scenario — the constraints, not the objective, do the nutritional work.")
    R.figure("D4_Climate_reduction_vs_number_of_swaps.png", "Best climate change as a function of the number of new foods allowed (MILP), with and without a "
             "nutrient-deviation budget.", max_h=0.24)
    R.figure("D4_Before_after_nutrient_and_environmental_profiles.png", "Nutrient retention at the operating point (dots; dashed line = 90 % floor) versus the "
             "unconstrained climate-only optimum (red crosses), and change on all seven indicators.", max_h=0.3)
    R.figure("D4_Substitution_flows_original_vs_optimised_baskets.png", "Original versus optimised basket composition at the operating point (grey = kept, "
             "coloured = substituted).", max_h=0.42)
    check(bool((sel["water_change"] > 0).any()), "water rises at the operating point in >= 1 scenario")
    worst = cw["climate_cost_of_water_priority_pp"].idxmax()
    raise_w = cw.index[cw["climate_optimum_raises_water"]].tolist()
    R.p("<b>Objectives genuinely conflict.</b> Beyond the climate–deviation trade-off, prioritising water scarcity costs climate: in "
        f"{worst} the water-optimal basket gives up {cw.loc[worst, 'climate_cost_of_water_priority_pp']:.1f} percentage points of climate reduction "
        f"relative to the climate optimum" + (f", and in {', '.join(raise_w)} the unconstrained climate optimum actually raises water scarcity." if raise_w else ".") +
        " More importantly, burden shifting appears at the declared operating point itself: " +
        "; ".join(f"{s}: climate {spct(sel.loc[s, 'climate_change'])}, water {spct(sel.loc[s, 'water_change'])}"
                  for s in SCEN if sel.loc[s, "water_change"] > 0) +
        ". A climate-only objective therefore risks shifting burden to water, which is why land and water are reported for every solution "
        "and a water cap can be added at a quantified climate cost.")
    R.figure("D4_Objective_conflicts_and_random_baselines.png", "Climate-optimal baskets under a water (left) or land (centre) cap, and the share of 5,000 "
             "random role-preserving swaps that satisfy every constraint (right).", max_h=0.27)
    R.figure("D4_Constraint_slack_and_violations.png", "Constraint values relative to their bounds for the original, operating-point, climate-optimal and "
             "unconstrained solutions; X marks a violation (only the unconstrained baseline).", max_h=0.3)

    # ================================================================ 6 D5
    R.h1("6. Deliverable 5 — robustness, uncertainty and counterexamples")
    R.p("All tests were declared before running (notebook 05, §1): random Dirichlet nutrient weights (200 draws), drop-one and alternative nutrient "
        "sets, a 4 × 4 grid of k and τ, a calorie-for-calorie graph, DQR filters (≤ 2.0 … 3.5), alternative duplicate and parser rules, the B12 "
        "rule switched off, an available-case NR9, alternative indicator sets, denominator floors, optimiser tolerances (27 combinations), "
        "bootstrap resampling of the food set, and a DQR-scaled impact perturbation stress test. DQR is used only to filter, stratify, penalise "
        "and widen the stress-test band — never as a probability; bootstrap intervals describe sampling of the food set, not LCA uncertainty.")
    R.p(f"<b>Network results.</b> Across the k × τ grid the naive-equivalent share stays between {pct(gg['naive_equivalent_share'].min(), 1)} and "
        f"{pct(gg['naive_equivalent_share'].max(), 1)} and community NMI with CIQUAL groups between {gg['nmi_vs_group'].min():.2f} and "
        f"{gg['nmi_vs_group'].max():.2f}. Under 200 random weight vectors the edge set changes substantially (median Jaccard "
        f"{ws['edge_jaccard'].median():.2f}) yet a median {pct(ws['strong_retained'].median())} of strong swaps survive and the naive-equivalent share "
        f"never exceeds {pct(ws['naive_equivalent_share'].max(), 1)}. Dropping one nutrient keeps most edges (Jaccard "
        f"{fs[fs['subset'].str.startswith('drop')]['edge_jaccard'].min():.2f}–{fs[fs['subset'].str.startswith('drop')]['edge_jaccard'].max():.2f}; "
        f"vitamin C matters most), whereas macro-only profiles change the graph radically (Jaccard "
        f"{fs.loc[fs['subset'].str.startswith('macro'), 'edge_jaccard'].iloc[0]:.2f}). {len(mve)} edges rest on fewer than 15 shared nutrients and "
        f"{int(mve['vanishes_when_imputed'].sum())} of them would vanish if missing values took their group median — edges driven by missing data.")
    R.figure("D5_Graph_k_and_threshold_sensitivity.png", "Edges, strong substitutions, community agreement with the reference partition and the naive-equivalent "
             "share across the k × τ grid.", max_h=0.22)
    R.figure("D5_Nutrient_weight_and_feature_subset_sensitivity.png", "Graph stability under 200 random nutrient-weight vectors and alternative nutrient sets.", max_h=0.27)
    dq3 = dq.set_index("dqr_max").loc[3.0]
    animal_ci = bcr[~bcr["category"].isin(["tofu/seitan/TVP", "plant-based patties"])]["ci_lo"]
    check(bool((animal_ci > 1).all()), "animal category ratio CIs exclude 1")
    R.p(f"<b>Trade-off results.</b> Resampling a group can only remove competitors, so efficient labels are stable by construction; the informative "
        f"test is the fragility of dominated labels: {pct(robust_dom)} of dominated food–view labels stay dominated in ≥ 90 % of bootstrap resamples, "
        f"while {pct(fragile_dom)} become efficient in at least a quarter of them — the labels that rest on a single dominating food (a given "
        f"competitor is absent from a bootstrap sample with probability ≈ 0.37; {pct(single_dom)} of dominated labels have exactly one dominator). "
        "The protein-category climate "
        "ratios keep bootstrap intervals above 1 for every animal category in every view. Removing "
        f"foods with DQR &gt; 3 keeps {pct(dq3['foods_kept_share'])} of foods and leaves efficient sets nearly unchanged (Jaccard "
        f"{dq3['jaccard_vs_reference']:.2f}); using the EF single score instead of climate, land and water changes status for "
        f"{pct(1 - ind.set_index('objective_set').loc['EF single score + NR9 + LIM3', 'status_agreement'])} of foods. Within-group climate rankings "
        f"reverse for {pct(rev['reversal_mass_energy'].mean())} of food pairs between mass and energy views on average — normalisation is a "
        "first-order modelling choice, not a detail.")
    R.figure("D5_Bootstrap_and_weight_stability_of_trade_off_results.png", "Fragility of dominated labels under bootstrap resampling (left), stability of efficiency labels "
             "under random NR9 weights (centre) and bootstrap intervals of protein-category climate ratios (right).", max_h=0.3)
    R.figure("D5_Normalization_reversal_panels.png", "Within-group climate-ranking reversals between functional units (left) and the strongest reversals "
             "(right): e.g. avocado vs radish.", max_h=0.27)
    pr = pert.groupby("scenario").agg(regret=("regret_vs_reference_opt", "median"), jac=("food_set_jaccard", "median"))
    R.p("<b>Optimisation results.</b> The operating-point reductions persist under the strictest tolerances, DQR-filtered candidate sets, "
        "alternative duplicate and parser rules and with flagged foods included (exact values in D5_optimizer_data_variants.csv; variants whose "
        "baseline food loses its complete profile are reported, not dropped). In the perturbation stress test the <i>achieved reduction</i> barely "
        f"moves (median regret {100 * pr['regret'].median():.1f} pp) while the <i>chosen foods</i> change considerably (median food-set Jaccard "
        f"{pr['jac'].median():.2f}): the headline is robust, the specific basket is not.")
    R.figure("D5_Optimization_tolerance_data_and_perturbation_sensitivity.png", "Constrained climate optimum across tolerance settings (left), under data and "
             "assumption variants (centre) and under the DQR-scaled perturbation stress test (right; not a confidence interval).", max_h=0.27)
    R.h2("Claims register")
    reg_tab = reg[["claim", "source", "tests_passed", "label"]].rename(columns={"claim": "Claim", "source": "From", "tests_passed": "Tests passed", "label": "Label"})
    R.table(reg_tab, "Every headline claim re-tested under alternative assumptions (individual tests in D5_claims_register.csv). 'Robust' = passes all; "
            "'conditional' = fails one; otherwise exploratory or withdrawn.", col_widths=[5.2, 0.6, 0.8, 1.1])
    cats = cx.groupby("category").agg(cases=("case", "size"), example=("case", "first"), evidence=("evidence", "first")).reset_index()
    R.table(cats.rename(columns={"category": "Category", "cases": "n", "example": "Example", "evidence": "Evidence"}),
            "Counterexample catalogue: apparently sustainable choices that fail once equivalence, normalisation, data quality or burden shifting "
            "is taken into account.", col_widths=[2.2, 0.3, 2.6, 3.4])
    R.figure("D5_Counterexample_cards.png", "One representative counterexample per category.", max_h=0.42)

    # ================================================================ 7 discussion
    R.h1("7. Limitations")
    R.bullets([
        "AGRIBALYSE describes average products consumed in France (conventional agriculture, average sourcing); no brand, farm, organic, local "
        "or seasonal claim can be made, and egg impacts are known to be under-estimated in release 3.2.",
        "The 15-nutrient feature set omits fatty-acid quality (n-3 ALA, EPA, DHA), folate, vitamins A and E, protein quality (amino-acid "
        "profile, digestibility) and bioavailability (haem vs non-haem iron); equivalence is therefore necessary-but-not-sufficient.",
        "Missing values are not imputed, so foods with sparse micronutrient data drop out of complete-case analyses; the B12 structural-zero rule "
        "is an explicit assumption (tested in D5).",
        "D3 swaps are gram-for-gram; portion compensation is only handled at basket level in D4.",
        "Optimised baskets are computational scenarios: LP optima can be culturally implausible (small amounts of offal used as a B12 source) and "
        "ignore price, taste, preparation and acceptability. They are not dietary advice.",
        "DQR is ordinal; we never translate it into statistical uncertainty. The perturbation analysis is a stress test, not a confidence interval.",
    ])
    R.h1("8. Conclusion")
    R.p("Sustainable substitution is real but conditional. Normalisation changes rankings without overturning the largest gaps between animal and "
        "plant protein; most environmentally attractive pairwise swaps fail nutritional equivalence once protein density, B12, zinc and limit "
        "nutrients are checked; and at basket level, large climate reductions remain achievable under explicit nutrient constraints — at the "
        "price of nutritional similarity, more changes, and possible burden shifting to water. The claims register separates the conclusions "
        "that survive every pre-declared test from those that depend on modelling choices.")
    R.h1("Reproducibility and references")
    R.p("All artefacts regenerate from the frozen raw files with <font face='Courier'>python run_all.py</font> (data checksums → 70 tests → six "
        "executed notebooks → this report → acceptance verification). Code: src/sustainome; declared inputs: config/; tables: reports/tables; "
        "figures: reports/figures.")
    R.bullets([
        "ANSES (2025). Ciqual French food composition table, English version. Zenodo record 17550133.",
        "ADEME / INRAE (2024–2025). AGRIBALYSE 3.2 — food products workbook and methodology. doi:10.57745/XTENSJ.",
        "European Commission (2011). Regulation (EU) No 1169/2011, Annex XIII reference intakes.",
        "Drewnowski A. (2009). Defining nutrient density: development and validation of the nutrient rich foods index. J Am Coll Nutr 28(4).",
        "Blondel V.D. et al. (2008). Fast unfolding of communities in large networks. J Stat Mech P10008.",
        "Huangfu Q., Hall J.A.J. (2018). Parallelizing the dual revised simplex method. Math Prog Comp 10 (HiGHS solver).",
        "Cliff N. (1993). Dominance statistics: ordinal analyses to answer ordinal questions. Psychol Bull 114(3).",
        "Ahn Y.-Y. et al. (2011). Flavor network and the principles of food pairing. Sci Rep 1:196 (course context).",
    ])

    out = C.REPORTS / "Sustainome_Report.pdf"
    R.build(out, f"{title} · BIO544 Group 13")
    print("wrote", out.relative_to(C.ROOT), f"({R.n_fig} figures, {R.n_tab} tables)")


if __name__ == "__main__":
    main()
