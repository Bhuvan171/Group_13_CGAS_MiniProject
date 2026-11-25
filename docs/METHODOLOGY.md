# Methodology and figure plan

## Analytical principle

The project does not ask which food is globally "best." It asks whether a candidate replacement improves environmental outcomes while remaining nutritionally similar enough for a declared comparison context.

## Stage 1 - Sustainable Food Matrix

### Processing

1. Freeze CIQUAL 2025 and AGRIBALYSE 3.2 snapshots.
2. Standardize column names while retaining a source-to-clean data dictionary.
3. Parse nutrient values and units explicitly.
4. Normalize and validate CIQUAL codes.
5. Resolve or preserve AGRIBALYSE duplicate codes using a documented rule.
6. Perform the exact join and publish coverage/missingness tables.
7. Select the final nutrient and environmental feature set based on meaning and coverage.

### Required evidence

- join-flow diagram;
- matched/unmatched counts;
- food-group coverage bars;
- nutrient/indicator missingness matrix;
- duplicate and parsing audit.

## Stage 2 - Nutrition-Environment Trade-off Analysis

### Three normalization views

1. **Mass view:** values per 100 g as provided or converted.
2. **Energy view:** nutrients and impacts per fixed energy unit, with low-energy denominator safeguards.
3. **Protein view:** nutrients and impacts per fixed protein unit, restricted or flagged when protein is near zero.

No single view is the ground truth. Conclusions must be compared across views.

### Nutritional comparison

Avoid a naive rule that more of every nutrient is better. Use a combination of:

- nutrient-profile distance after robust scaling;
- adequacy or retention relative to a baseline/reference food;
- separate treatment of beneficial/essential nutrients and limit nutrients;
- sensitivity to alternative nutrient weights.

### Pareto analysis

Run Pareto analysis within comparable food groups. Candidate objectives may include minimizing climate, land, water, and nutrient deviation. Do not create an unqualified Pareto front across unrelated foods such as beverages, spices, oils, and complete dishes.

### Planned figures

- small-multiple Pareto fronts by food group;
- mass/calorie/protein comparison panels;
- nutrient-environment biplots or embeddings;
- dominance-depth distributions;
- group-wise effect-size plots with intervals.

## Stage 3 - Nutritional-Equivalence Substitution Network

### Graph definition

- nodes: matched foods;
- eligibility: same comparable group or explicitly compatible basket role;
- similarity: weighted nutrient-profile distance;
- edge direction: from original food to a nutritionally similar candidate;
- edge attributes: nutrient deviation, environmental change by indicator, DQR, and normalization view.

Prefer a mutual k-nearest-neighbour or thresholded graph to avoid a nearly complete network. The threshold must be selected transparently and varied in Deliverable 5.

### Candidate edge score

Keep component scores separate even if a composite display score is added:

- nutrition similarity;
- climate/land/water improvement;
- DQR/reliability penalty;
- food-group compatibility;
- optional substitution simplicity.

### Analyses and figures

- degree and centrality distributions;
- connected components and community stability;
- network plot by food group;
- environmental-gain versus nutrient-distance scatter;
- top reliable substitutions and top misleading substitutions;
- manual audit of representative edges.

## Stage 4 - Multi-Objective Basket/Menu Optimization

### Scenario design

Use 3-5 fixed and transparent computational scenarios built from matched foods with declared quantities. They are examples for method evaluation, not claims of representative diets.

Possible scenario types:

- breakfast-style basket;
- protein-centred meal;
- dairy/snack basket;
- mixed meal;
- one-day illustrative basket if time permits.

### Decision variables and constraints

Variables may be continuous food amounts plus optional binary inclusion variables. Constraints should preserve:

- total energy within a declared tolerance;
- protein within a declared tolerance or minimum;
- selected micronutrients above a baseline-retention threshold;
- sodium and saturated fat not exceeding declared limits/baseline;
- food-group or meal-role structure;
- bounded change in total mass and number of substitutions.

### Objectives

Produce a Pareto set over objectives such as:

- climate impact;
- land/water impact;
- nutrient-profile deviation;
- number/magnitude of substitutions;
- DQR penalty.

Use an epsilon-constraint sweep, repeated weighted optimization, or a multi-objective solver. Compare against the original basket, random feasible substitutions, and a single-objective environmental baseline.

### Planned figures

- optimizer Pareto fronts;
- before/after environmental bars;
- before/after nutrient-retention profiles;
- substitution flow plots;
- constraint-slack or violation plots.

## Stage 5 - Robustness, uncertainty, and counterexamples

### Sensitivity dimensions

- nutrient weights;
- nutrient feature subsets;
- mass/calorie/protein normalization;
- graph k or similarity threshold;
- DQR exclusion/penalty rules;
- duplicate-resolution rules;
- bootstrap resampling;
- alternative environmental-indicator subsets;
- optimization tolerances.

### DQR handling

DQR ranges from 1 (better) to 5 (worse). Use it for filtering, stratification, penalties, or sensitivity analysis. Do not mechanically convert DQR into a standard deviation or confidence interval without a justified model.

### Counterexamples

Actively search for:

- lower-climate candidates with large nutrient deviation;
- substitutions that reverse under calorie or protein normalization;
- visually compelling network edges driven by missing values;
- optimization solutions dominated after a different indicator is included;
- conclusions that disappear after low-quality DQR records are removed.

### Planned figures

- bootstrap confidence intervals;
- rank/stability heatmaps;
- threshold sensitivity curves;
- normalization reversal panels;
- documented counterexample cards.

