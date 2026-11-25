# Project context and decision record

## Final decision

The selected mini-project is:

**Sustainome: The Sustainable Plate Frontier - Nutrient-Environment Trade-offs and Food Substitution**

The decision followed a feasibility-first review of multiple computational-gastronomy ideas. Sustainome was chosen because it combines a clear scientific question, exact-key database integration, multi-objective analysis, network science, optimization, uncertainty analysis, and strong plot potential without depending on scraping, gated data, a fragile API, or large-model training.

## Core research question

Which food substitutions genuinely reduce environmental impact while preserving nutritional value, and where do unavoidable trade-offs remain after nutritional equivalence is enforced?

## Why this project is defensible

1. **Reliable linkage:** CIQUAL food codes are present in both the nutrition and environmental sources, so the primary integration does not require fuzzy name matching.
2. **Substantive computation:** the work moves beyond descriptive plots into Pareto dominance, nutrient-profile distance, graph modelling, constrained optimization, bootstrapping, and stability analysis.
3. **Natural visual evidence:** Pareto fronts, biplots, networks, optimization curves, and uncertainty figures are direct outputs of the analyses.
4. **Distinct deliverables:** dataset construction, food-level inference, graph analysis, basket-level optimization, and validation have separate outputs and acceptance tests.
5. **Feasible scope:** the input data are only a few thousand food records; methodological depth, not data volume, provides the difficulty.

## Feasibility evidence

The initial audit downloaded and opened the public CIQUAL 2025 English workbook and retrieved the public AGRIBALYSE 3.1 summary API. It found:

- 3,484 CIQUAL foods;
- 2,451 AGRIBALYSE API rows;
- 2,425 unique AGRIBALYSE CIQUAL codes;
- 2,272 exact shared codes;
- 93.69% coverage of the API's unique codes.

These counts are audit evidence, not an immutable project requirement. The implementation should use frozen source versions and publish its own final counts.

## Important version decision

AGRIBALYSE 3.2 is now the current official release. The repository includes its food-products workbook and English methodology. Therefore:

- default implementation source: AGRIBALYSE 3.2 food-products workbook;
- comparison/audit source: the 3.1 summary API used in the original feasibility test;
- never mix 3.1 and 3.2 values in one analytical table;
- document which release is used in every output;
- if 3.2 changes the exact-match count, update internal audit tables but do not change the locked deliverables.

## Scope boundary

The project uses approximately 12-20 nutrient variables and 5-8 environmental indicators after a missingness and interpretability audit. The final variable set should be frozen before the main analysis.

Candidate nutrient families:

- energy and macronutrients: energy, protein, carbohydrate, fat, fibre;
- limit nutrients: sodium/salt and saturated fat;
- selected micronutrients: calcium, iron, magnesium, potassium, zinc, vitamin C, folate, vitamin B12, and vitamin D where coverage permits.

Candidate environmental indicators:

- climate change;
- land use;
- water-resource depletion;
- freshwater and marine eutrophication;
- terrestrial/freshwater acidification;
- energy/fossil-resource depletion;
- the Environmental Footprint single score only as a sensitivity view, not the sole environmental outcome.

## What the project does not claim

- It does not provide medical or individualized dietary advice.
- It does not rank specific brands, farms, production systems, or geographic sourcing choices.
- It does not establish causal effects of eating a particular food.
- It does not claim that a single normalization denominator is universally correct.
- It does not claim that DQR directly yields statistical confidence intervals.
- It does not treat a computationally optimized basket as a culturally complete or behaviorally realistic diet.

## Deliverable separation

- Deliverable 2 asks: *Which individual foods occupy favourable or unfavourable trade-off positions?*
- Deliverable 3 asks: *Which pairwise replacements are nutritionally plausible, and what structure emerges across those replacements?*
- Deliverable 4 asks: *How should multiple foods and quantities change together under basket-level constraints?*
- Deliverable 5 asks: *Do those conclusions survive alternative assumptions and sampling variation?*

This separation must be maintained in implementation and reporting.

## Communication style

The final project should read like a compact research study rather than a product pitch. A web application is optional and cannot replace any of the five analyses. Notebooks, reusable modules, publication-quality figures, a short report, and reproducible data artifacts are sufficient.

