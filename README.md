# Sustainome

**The Sustainable Plate Frontier: Nutrient–Environment Trade-offs and Food Substitution**
BIO544 Computational Gastronomy · Monsoon 2026 · Mini-project · Group 13

> *Which food substitutions genuinely reduce environmental impact while preserving nutritional value, and where do unavoidable trade-offs remain after nutritional equivalence is enforced?*

Sustainome joins the **ANSES-CIQUAL 2025** nutrient-composition table to **AGRIBALYSE 3.2** life-cycle indicators by exact CIQUAL food code and studies substitution at four levels — food (D2), pair (D3), basket (D4) and validation (D5) — on top of a reproducible integrated dataset (D1).

## Submission contents (Group 13)

| Item | File |
|---|---|
| Presentation (PPT) | [`Sustainome_Slides.pptx`](Sustainome_Slides.pptx) · PDF: [`Sustainome_Slides.pdf`](Sustainome_Slides.pdf) · LaTeX source: [`Sustainome_Slides.tex`](Sustainome_Slides.tex) (compile with XeLaTeX, twice) |
| Project report (4 pages) | [`Group_13_Report.pdf`](Group_13_Report.pdf) |
| Full technical report (17 pages) | [`reports/Sustainome_Report.pdf`](reports/Sustainome_Report.pdf) |
| Code (commented, tested) | [`src/sustainome/`](src/sustainome) modules, [`notebooks/`](notebooks) 00–05, [`scripts/`](scripts), [`tests/`](tests) |
| README | this file |

Quick start: `pip install -r requirements.txt`, then `python run_all.py` (rebuilds everything) or `python verify_submission.py` (acceptance checks). Course reference PDFs (`references/`) are not redistributed in this repository.


## Headline results

| | Finding |
|---|---|
| **D1** | Exact join of CIQUAL 2025 × AGRIBALYSE 3.2: **2,272 matched foods** (93.69 % of AGRIBALYSE codes); every stage reconciles; 7 packaging-error and 26 variant rows resolved by a tested rule; proposal audit on AGRIBALYSE 3.1 reproduced exactly. |
| **D2** | Normalisation compresses but never reverses the protein hierarchy: beef & veal = **63× / 35× / 17×** cooked legumes per 100 g / 100 kcal / 10 g protein. Per 100 kcal, nutrient density and climate **conflict in 19 groups and agree in 1**. Efficient sets overlap only moderately across functional units (mean Jaccard 0.61). |
| **D3** | In a leakage-free equivalence network (calibrated τ), the naive "greenest swap" is nutritionally equivalent for only **4.5 %** of foods; 37,280 attractive swaps are misleading and **156 are strong** (equivalent, reliable, burden-free). Protein, energy, B12 and zinc fail most often. |
| **D4** | Across five declared baskets, climate impact falls **38–82 %** at the declared operating point (nutrient deviation ≤ 0.10, ≤ 50 % of grams changed) with every nutrition constraint met; portion rebalancing alone saves 9–17 %; random role-preserving swaps are feasible only 0.3–8 % of the time; water scarcity **rises** at the operating point in three baskets (burden shifting). |
| **D5** | Pre-declared robustness grid (weights, features, k × τ, DQR, parser/duplicate/B12 rules, bootstrap, tolerances, perturbation): **6 of 9 headline claims robust, 3 conditional**; 9 categories of counterexample (38 cases) catalogued. |

*Database-supported scenarios for average products consumed in France — not dietary advice.*

## Deliverables → where to find them

| # | Deliverable | Notebook | Key outputs |
|---|---|---|---|
| D1 | Sustainable Food Matrix | [`00_data_audit`](notebooks/00_data_audit.ipynb), [`01_sustainable_food_matrix`](notebooks/01_sustainable_food_matrix.ipynb) | `data/processed/sustainable_food_matrix.csv` (+ `.parquet`), `data_dictionary.csv`, `PROCESSED_MANIFEST.json`, `data/raw/SNAPSHOT_MANIFEST.json`, `reports/tables/D1_*` |
| D2 | Nutrition–Environment Trade-off Analysis | [`02_tradeoff_analysis`](notebooks/02_tradeoff_analysis.ipynb) | Pareto labels in three views, group statistics with bootstrap CIs, effect sizes, non-obvious trade-offs (`reports/tables/D2_*`) |
| D3 | Nutritional-Equivalence Substitution Network | [`03_substitution_network`](notebooks/03_substitution_network.ipynb) | node/edge/arc tables, calibrated edge rule, communities, recommender comparison, manual edge audit (`reports/tables/D3_*`) |
| D4 | Multi-Objective Basket/Menu Optimization | [`04_basket_optimization`](notebooks/04_basket_optimization.ipynb) | 5 declared scenarios, ε-constraint Pareto sets, swap-count MILP fronts, baselines, constraint checks (`reports/tables/D4_*`) |
| D5 | Robustness, Uncertainty and Counterexample Study | [`05_robustness_counterexamples`](notebooks/05_robustness_counterexamples.ipynb) | pre-declared sensitivity grid, bootstrap, DQR analyses, counterexample catalogue, **claims register** (`reports/tables/D5_*`) |

Every figure is a titled PNG in [`reports/figures/`](reports/figures) named `D<deliverable>_<Title>.png`; every deliverable ends with an acceptance table (`reports/tables/D<n>_acceptance.csv`) checked by `verify_submission.py`.

## Presentation

`presentation/export/Sustainome_Presentation.pdf` and `.pptx` — the 33-slide deck (one divider per deliverable, D1–D5). One PNG per slide sits beside them. The PPTX holds each slide as a full-slide picture; the editable slide source (HTML) is in `presentation/deck/`, the slide images in `presentation/assets/`, and `python presentation/export_deck.py` re-renders the PDF/PPTX.

## Getting started on a new machine

1. Unzip anywhere and open a terminal in the project folder.
2. Install Python 3.11+ (tested with 3.13), then `pip install -r requirements.txt`.
3. Check everything is intact: `python verify_submission.py` (should end with `"ALL_CHECKS_PASSED": true`) and `python -m pytest -q`.
4. Rebuild from scratch if you change anything: `python run_all.py` (raw data are included under `data/raw/`, so no download is needed).

To change the analysis, edit the declared inputs in `config/` or the code in `src/sustainome/`; notebook text and code live in `scripts/notebook_sources/*.py` (the `.ipynb` files are generated from them by `python scripts/build_notebooks.py`).

## Reproduce

```bash
pip install -r requirements.txt        # Python 3.13 (tested on Windows 11)
python run_all.py                       # data check -> tests -> notebooks 00-05 -> report -> verification
```

`run_all.py` (≈ 45 min on a laptop; most of it is the D4 MILP fronts and the D5 robustness grid) verifies the raw snapshots against their SHA-256 checksums (downloading them if absent), runs the test suite (70 tests), executes the six notebooks in order, rebuilds the PDF report and writes `reports/verification.json`. Individual pieces:

```bash
python scripts/download_data.py               # fetch / verify raw snapshots
python -m pytest -q                           # unit + integration tests
python scripts/build_notebooks.py 02 03       # (re)execute selected notebooks
python scripts/build_report.py                # rebuild the PDF report
python verify_submission.py                   # acceptance checks
```

All randomised procedures are seeded (`config.SEED`); every threshold and weight lives in [`src/sustainome/config.py`](src/sustainome/config.py) and the declared inputs in [`config/`](config).

## Repository layout

```text
config/                  declared inputs: roles.csv, baskets.csv, group_merges.csv,
                         structural_zero.csv, manual_name_audit.csv, manual_edge_audit.csv
data/raw/                immutable snapshots + SNAPSHOT_MANIFEST.json (checksums, licences)
data/processed/          integrated matrix, data dictionary, processed manifest (generated)
notebooks/               00-05, executed (orchestration + narrative only)
src/sustainome/          tested modules: io, clean, matrix, normalize, nutrition, pareto,
                         stats, tradeoff, network, optimize, robustness, plotting, config
tests/                   70 pytest tests (parser, codes, duplicates, Pareto, graph rules,
                         leakage, optimiser constraints, reproducibility)
scripts/                 notebook sources + builder, report builder, data downloader
reports/                 Sustainome_Report.pdf, figures/, tables/, verification.json
docs/, references/       locked project context, proposal, data-source documentation
```

## Data sources and licences

| Source | Version | Use | Licence |
|---|---|---|---|
| ANSES-CIQUAL French food composition table (English) — [Zenodo 17550133](https://zenodo.org/records/17550133) | 2025 (2025-11-03) | nutrients, food identity and hierarchy | Etalab Open Licence 2.0 |
| AGRIBALYSE food-products workbook — [doi:10.57745/XTENSJ](https://doi.org/10.57745/XTENSJ) | 3.2 (workbook of 2025-08-21) | environmental indicators (EF 3.1), DQR | Licence Ouverte Etalab |
| AGRIBALYSE 3.1 synthesis API — data.ademe.fr | 3.1 | reproduction of the proposal's feasibility audit only | Licence Ouverte Etalab |

Raw files are included unmodified so the analysis can be reproduced offline; attribution: *ANSES-CIQUAL 2025; ADEME/INRAE AGRIBALYSE 3.2*.

## Scientific guardrails (enforced in code)

- Exact normalised-code join; names used only for validation.
- Missing nutrients never zero-filled; B12 structural zeros are explicit, flagged and switchable.
- Comparisons only within analysis groups or declared roles; three functional units with denominator guards.
- Capped NR9 + separate LIM3 (no "more of every nutrient is better").
- DQR is an ordinal 1–5 score: used for filtering, stratification, penalties and a labelled stress test — never as a probability.
- No brand, farm, organic/conventional, local/imported or seasonal claims; baskets are computational test cases, **not dietary advice**.
- No universal food ranking: results are Pareto sets with uncertainty and a claims register separating robust from exploratory findings.

Project context and the locked proposal: [`CLAUDE.md`](CLAUDE.md), [`docs/`](docs) (methodology, deliverable acceptance criteria, non-overlap with Assignments 1–2).
