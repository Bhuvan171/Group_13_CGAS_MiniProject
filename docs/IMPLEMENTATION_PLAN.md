# Implementation plan

## Recommended repository layout

```text
Cgas_Mini_Project/
  CLAUDE.md
  README.md
  data/
    raw/
      ciqual/2025/
      agribalyse/3.2/
    interim/
    processed/
  docs/
  notebooks/
    00_data_audit.ipynb
    01_sustainable_food_matrix.ipynb
    02_tradeoff_analysis.ipynb
    03_substitution_network.ipynb
    04_basket_optimization.ipynb
    05_robustness_counterexamples.ipynb
  references/
  reports/
    figures/
    tables/
  src/sustainome/
    config.py
    io.py
    clean.py
    normalize.py
    pareto.py
    network.py
    optimize.py
    robustness.py
    plotting.py
  tests/
  environment.yml or pyproject.toml
```

Notebooks should orchestrate and explain. Reusable logic belongs under `src/sustainome/`.

## Suggested Python stack

- data: `pandas`, `numpy`, `openpyxl`, optionally `pyarrow`;
- statistics: `scipy`, `statsmodels`, `scikit-learn`;
- graph: `networkx`, optionally `python-louvain` or `leidenalg`;
- optimization: `scipy.optimize`, `pulp`, `cvxpy`, or `pymoo` depending on formulation;
- plots: `matplotlib`, `seaborn`, `plotly` only where interactivity adds value;
- validation: `pytest`.

Freeze versions after the first successful environment build.

## Week 1 - data and definitions

### Goals

- freeze CIQUAL 2025 and AGRIBALYSE 3.2 snapshots;
- record source manifest and checksums;
- inspect workbook schemas;
- implement value parsing and code normalization;
- create exact joined matrix and audit tables;
- define comparable food groups;
- freeze candidate nutrients and environmental indicators;
- draft 3-5 reference baskets.

### Exit criterion

D1 should be substantially complete. Final join counts, missingness, duplicates, and selected features are known.

## Week 2 - food-level and network analyses

### Goals

- implement mass/calorie/protein transformations;
- define nutrient-profile distance;
- compute group-wise Pareto fronts;
- create D2 figures and findings;
- construct the substitution graph;
- select graph threshold/k and run community analysis;
- manually audit representative edges;
- formulate the basket optimization model and baselines.

### Exit criterion

D2 and D3 are complete enough to write. The optimizer solves at least one scenario end to end.

## Week 3 - optimization and falsification

### Goals

- solve all fixed scenarios and generate Pareto sets;
- compare baselines;
- run sensitivity grid and bootstraps;
- perform DQR filtering/penalty analysis;
- collect counterexamples and normalization reversals;
- finalize all figures/tables;
- write concise research report and presentation narrative;
- run clean-environment reproducibility check.

### Exit criterion

All five deliverables satisfy `docs/DELIVERABLES.md`. A fresh run from raw snapshots reproduces the principal tables and plots.

## Minimum tests

- CIQUAL code normalization preserves valid identities;
- nutrient parser handles numeric, missing, trace, and inequality cases;
- per-calorie/per-protein normalization handles zero/near-zero denominators;
- Pareto dominance is correct on hand-constructed examples;
- network eligibility prevents cross-group invalid edges;
- optimizer solutions satisfy declared constraints;
- seeded robustness runs are repeatable.

## Scope cuts if time becomes tight

Cut optional indicators, interactive views, or extra basket scenarios before cutting robustness. Do not replace D5 with a dashboard. Preserve the exact-key data artifact, the three normalization views, the substitution graph, at least three optimization scenarios, and the counterexample analysis.

