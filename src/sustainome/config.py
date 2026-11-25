"""Frozen configuration: paths, source versions, feature sets, thresholds and seeds.

Every threshold or weight used anywhere in the analysis is declared here so that it is
logged in one place and can be varied systematically in Deliverable 5.
"""

from __future__ import annotations

from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
RAW = DATA / "raw"
INTERIM = DATA / "interim"
PROCESSED = DATA / "processed"
CONFIG_DIR = ROOT / "config"
REPORTS = ROOT / "reports"
FIGURES = REPORTS / "figures"
TABLES = REPORTS / "tables"

CIQUAL_FILE = RAW / "ciqual" / "2025" / "Table_Ciqual_2025_ENG_2025_11_03.xlsx"
AGB_FILE = RAW / "agribalyse" / "3.2" / "AGRIBALYSE3.2_Tableur_produits_alimentaires_PublieAOUT25.xlsx"
AGB31_API_FILE = RAW / "agribalyse" / "3.1_api" / "agribalyse-31-synthese_lines.json"
MANIFEST_FILE = RAW / "SNAPSHOT_MANIFEST.json"

MATRIX_FILE = PROCESSED / "sustainable_food_matrix.csv"
MATRIX_PARQUET = PROCESSED / "sustainable_food_matrix.parquet"

SEED = 20261117  # global seed; every randomized routine derives from it

# ---------------------------------------------------------------------------
# Source registry (provenance; checksums are computed and frozen in the manifest)
# ---------------------------------------------------------------------------
SOURCES = {
    "ciqual_2025": {
        "organization": "ANSES (French Agency for Food, Environmental and Occupational Health & Safety)",
        "title": "ANSES-CIQUAL French food composition table 2025 (English workbook)",
        "version": "2025 (file dated 2025-11-03)",
        "url": "https://zenodo.org/api/records/17550133/files/Table%20Ciqual%202025_ENG_2025_11_03.xlsx/content",
        "record": "https://zenodo.org/records/17550133",
        "license": "Etalab Open Licence 2.0 - attribution to ANSES-CIQUAL required",
        "sheet": "food composition",
        "path": CIQUAL_FILE,
    },
    "agribalyse_3.2": {
        "organization": "ADEME / INRAE (GIS REVALIM)",
        "title": "AGRIBALYSE 3.2 - food products workbook (Tableur produits alimentaires)",
        "version": "3.2 (September 2024 release; workbook last modified 2025-08-21)",
        "url": "https://entrepot.recherche.data.gouv.fr/api/access/datafile/641911",
        "record": "https://doi.org/10.57745/XTENSJ",
        "license": "Licence Ouverte Etalab - attribution to ADEME/AGRIBALYSE required",
        "sheet": "Synthese",
        "path": AGB_FILE,
    },
    "agribalyse_3.1_api": {
        "organization": "ADEME (data.ademe.fr)",
        "title": "AGRIBALYSE 3.1 synthesis API (feasibility-audit source only)",
        "version": "3.1",
        "url": "https://data.ademe.fr/data-fair/api/v1/datasets/agribalyse-31-synthese/lines?size=3000",
        "record": "https://data.ademe.fr/datasets/agribalyse-31-synthese",
        "license": "Licence Ouverte Etalab",
        "sheet": "JSON 'results'",
        "path": AGB31_API_FILE,
    },
}

# ---------------------------------------------------------------------------
# Nutrient feature set (frozen after the D1 coverage audit)
# ---------------------------------------------------------------------------
# Rule: keep nutrients that are nutritionally interpretable AND have >= 70 % non-missing
# coverage in the matched set. Folate (51 %), vitamin E (18 %) and vitamin A RE (59 %) fail.
# `header` is the whitespace-normalised CIQUAL 2025 English header.
# `ref` is the EU reference intake (Reg. 1169/2011 Annex XIII) used for %DV / %MRV.
NUTRIENTS = [
    # key,               header,                                                   unit,      family,  ref
    ("energy_kcal",      "Energy, Regulation EU No 1169 2011 (kcal 100g)",         "kcal",    "energy", 2000.0),
    ("protein_g",        "Protein (g 100g)",                                        "g",       "macro",  50.0),
    ("carbohydrate_g",   "Carbohydrate (g 100g)",                                   "g",       "macro",  260.0),
    ("fat_g",            "Fat (g 100g)",                                            "g",       "macro",  70.0),
    ("sugars_g",         "Sugars (g 100g)",                                         "g",       "limit",  90.0),
    ("fibre_g",          "Fibres (g 100g)",                                         "g",       "macro",  25.0),
    ("sat_fat_g",        "FA saturated (g 100g)",                                   "g",       "limit",  20.0),
    ("sodium_mg",        "Sodium (mg 100g)",                                        "mg",      "limit",  2400.0),
    ("calcium_mg",       "Calcium (mg 100g)",                                       "mg",      "micro",  800.0),
    ("iron_mg",          "Iron (mg 100g)",                                          "mg",      "micro",  14.0),
    ("magnesium_mg",     "Magnesium (mg 100g)",                                     "mg",      "micro",  375.0),
    ("potassium_mg",     "Potassium (mg 100g)",                                     "mg",      "micro",  2000.0),
    ("zinc_mg",          "Zinc (mg 100g)",                                          "mg",      "micro",  10.0),
    ("vitamin_c_mg",     "Vitamin C (mg 100g)",                                     "mg",      "micro",  80.0),
    ("vitamin_b12_ug",   "Vitamin B12 (µg 100g)",                                   "µg",      "micro",  2.5),
]
NUTRIENT_KEYS = [n[0] for n in NUTRIENTS]
NUTRIENT_REF = {n[0]: n[4] for n in NUTRIENTS}
NUTRIENT_UNIT = {n[0]: n[2] for n in NUTRIENTS}
NUTRIENT_FAMILY = {n[0]: n[3] for n in NUTRIENTS}
MICRO_KEYS = [k for k in NUTRIENT_KEYS if NUTRIENT_FAMILY[k] == "micro"]
LIMIT_KEYS = ["sat_fat_g", "sugars_g", "sodium_mg"]

# Extended nutrients: parsed and kept in the matrix, used only in D5 feature-subset tests.
EXTENDED_NUTRIENTS = [
    ("vitamin_d_ug",     "Vitamin D (µg 100g)",                                     "µg",      "micro",  5.0),
    ("vitamin_b6_mg",    "Vitamin B6 (mg 100g)",                                    "mg",      "micro",  1.4),
    ("folate_ug",        "Vitamin B9 or total folates (µg 100g)",                   "µg",      "micro",  200.0),
    ("retinol_ug",       "Retinol (µg 100g)",                                       "µg",      "micro",  800.0),
    ("phosphorus_mg",    "Phosphorus (mg 100g)",                                    "mg",      "micro",  700.0),
    ("water_g",          "Water (g 100g)",                                          "g",       "other",  None),
    ("alcohol_g",        "Alcohol (g 100g)",                                        "g",       "other",  None),
]
EXTENDED_KEYS = [n[0] for n in EXTENDED_NUTRIENTS]

# Nutrient-rich-food index adapted from NRF9.3 (Drewnowski 2009). Vitamins A and E are
# replaced by zinc and vitamin B12 because their CIQUAL coverage is too low (see D1).
# Qualifying nutrients are capped at 100 % of the reference intake per functional unit,
# so "more" stops counting once adequacy is reached.
NR_KEYS = ["protein_g", "fibre_g", "calcium_mg", "iron_mg", "magnesium_mg",
           "potassium_mg", "zinc_mg", "vitamin_c_mg", "vitamin_b12_ug"]
LIM_KEYS = ["sat_fat_g", "sugars_g", "sodium_mg"]  # sugars = total sugars (added sugars not in CIQUAL)

# Value parser policy for CIQUAL markers
PARSE_POLICY = "half_loq"  # '< x' -> x/2 ; 'traces' -> 0 ; '-' -> missing. Alternatives in D5.
PARSE_POLICIES = ("half_loq", "zero", "loq")

# ---------------------------------------------------------------------------
# Environmental indicators (AGRIBALYSE 3.2, EF 3.1, per kg of product consumed)
# ---------------------------------------------------------------------------
# key, French indicator header (accent-insensitive match), unit, short label
ENV = [
    ("climate",        "Changement climatique",                    "kg CO2 eq",      "Climate change"),
    ("land",           "Utilisation du sol",                       "Pt",             "Land use"),
    ("water",          "Epuisement des ressources eau",            "m3 depriv.",     "Water scarcity"),
    ("eutro_fw",       "Eutrophisation eaux douces",               "kg P eq",        "Freshwater eutrophication"),
    ("eutro_marine",   "Eutrophisation marine",                    "kg N eq",        "Marine eutrophication"),
    ("acidification",  "Acidification terrestre et eaux douces",   "mol H+ eq",      "Acidification"),
    ("fossil",         "Epuisement des ressources energetiques",   "MJ",             "Fossil resource use"),
    ("ef_score",       "Score unique EF 3.1",                      "mPt",            "EF single score"),
]
ENV_KEYS = [e[0] for e in ENV]
ENV_UNIT = {e[0]: e[2] for e in ENV}
ENV_LABEL = {e[0]: e[3] for e in ENV}
ENV_CORE = ["climate", "land", "water", "eutro_fw", "eutro_marine", "acidification", "fossil"]
ENV_HEADLINE = ["climate", "land", "water"]  # primary environmental objectives

# ---------------------------------------------------------------------------
# Normalization views (functional units) and denominator safeguards
# ---------------------------------------------------------------------------
VIEWS = ("mass", "energy", "protein")
VIEW_UNIT = {"mass": "per 100 g", "energy": "per 100 kcal", "protein": "per 10 g protein"}
ENERGY_FLOOR_KCAL = 10.0   # foods below this (kcal/100 g) are undefined in the energy view
PROTEIN_FLOOR_G = 2.0      # foods below this (g/100 g) are undefined in the protein view

# ---------------------------------------------------------------------------
# D2 Pareto analysis
# ---------------------------------------------------------------------------
MIN_GROUP_SIZE = 8         # minimum complete foods for within-group Pareto/statistics
PARETO_OBJECTIVES = {      # +1 minimise, -1 maximise
    "climate": 1, "land": 1, "water": 1, "nr9": -1, "lim3": 1,
}
PARETO_2D = {"climate": 1, "nr9": -1}
N_BOOT = 1000

# ---------------------------------------------------------------------------
# D3 substitution network
# ---------------------------------------------------------------------------
MIN_SHARED_FEATURES = 12   # of 15 profile nutrients needed to compute a distance
KNN_K = 5                  # mutual k-nearest neighbours
DIST_QUANTILE = 0.70       # tau = this quantile of 1-NN distances; selected by network.select_quantile (D3 notebook)
STRONG_CLIMATE_GAIN = 0.20 # >= 20 % climate reduction to be an "attractive" substitution
BURDEN_SHIFT_TOL = 0.20    # other indicator may not worsen by more than 20 %
DQR_RELIABLE = 3.0         # both foods' DQR <= 3 for a "reliable" edge
PROTEIN_RETENTION = 0.80
MICRO_RETENTION = 0.50     # for micronutrients where the source supplies >= 5 % DV/100 g
MICRO_SOURCE_DV = 0.05
ENERGY_RATIO = (0.67, 1.5)
LIMIT_TOL = {"sodium_mg": (1.25, 100.0), "sat_fat_g": (1.25, 1.0), "sugars_g": (1.25, 2.5)}

# ---------------------------------------------------------------------------
# D4 optimisation
# ---------------------------------------------------------------------------
OPT = {
    "energy_tol": 0.05,        # total energy within +/- 5 % of baseline
    "protein_min": 0.95,       # protein >= 95 % of baseline
    "micro_min": 0.90,         # each micronutrient + fibre >= 90 % of baseline
    "limit_max": 1.00,         # sodium, sat fat, sugars <= baseline
    "role_mass_tol": 0.25,     # each slot's total mass within +/- 25 %
    "dev_floor_dv": 0.05,      # relative-deviation denominator floor (5 % of reference intake)
    "eps_dev": [0.0, 0.02, 0.05, 0.08, 0.12, 0.16, 0.20, 0.25, 0.30, 0.40, 0.60],
    "eps_change": [0.10, 0.25, 0.50, 1.00],
    "n_random": 5000,
}
# Illustrative operating point reported in before/after figures - declared before solving.
OPT_SELECTED = {"eps_dev": 0.10, "eps_change": 0.50}
OPT_MICRO = ["fibre_g", "calcium_mg", "iron_mg", "magnesium_mg", "potassium_mg",
             "zinc_mg", "vitamin_c_mg", "vitamin_b12_ug"]

# ---------------------------------------------------------------------------
# D5 robustness grid (pre-declared)
# ---------------------------------------------------------------------------
ROBUST = {
    "n_weight_draws": 200,
    "knn_grid": [3, 5, 8, 12],
    "quantile_grid": [0.50, 0.70, 0.85, 1.00],
    "dqr_cutoffs": [2.0, 2.5, 3.0, 3.5],
    "energy_floor_grid": [5.0, 10.0, 20.0, 40.0],
    "protein_floor_grid": [1.0, 2.0, 5.0],
    "energy_tol_grid": [0.02, 0.05, 0.10],
    "micro_min_grid": [0.80, 0.90, 1.00],
    "protein_min_grid": [0.90, 0.95, 1.00],
    "louvain_seeds": 20,
}

# Plot style
DPI = 200
