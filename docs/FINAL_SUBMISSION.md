# Locked form submission

This is the final copy-paste text approved for the proposal form. Do not alter the scientific meaning or replace deliverables without explicit user approval.

## Project Title

**Sustainome: The Sustainable Plate Frontier - Nutrient-Environment Trade-offs and Food Substitution**

## Description

Sustainome investigates which food substitutions genuinely reduce environmental impact while preserving nutritional value. We will integrate the ANSES-CIQUAL 2025 nutrient-composition database with AGRIBALYSE environmental indicators through their shared CIQUAL food codes, providing over 2,200 directly matched foods without fuzzy name matching. Foods will be compared under mass-, calorie-, and protein-normalized perspectives using selected nutrients and environmental indicators. We will identify Pareto-efficient alternatives, construct a nutrition-aware substitution network, and perform constrained multi-objective optimization on fixed reference food baskets or menus. Robustness analyses will incorporate AGRIBALYSE data-quality scores, alternative nutritional assumptions and bootstrap uncertainty. Pareto fronts, nutrient-environment biplots, substitution networks, optimization curves and uncertainty plots will be produced as evidence throughout the analyses.

## Deliverable 1

**Sustainable Food Matrix:** Build and validate a reproducible dataset joining CIQUAL nutrient composition and food categories with AGRIBALYSE environmental indicators and data-quality scores using exact CIQUAL food-code matches. Include unit harmonization, a data dictionary, provenance, coverage and missingness checks.

## Deliverable 2

**Nutrition-Environment Trade-off Analysis:** Identify Pareto-efficient and dominated foods within comparable food groups and quantify when nutritional and environmental objectives agree or conflict under mass-, calorie-, and protein-normalized comparisons. Report the findings through Pareto fronts, nutrient-environment biplots and group-wise statistical comparisons.

## Deliverable 3

**Nutritional-Equivalence Substitution Network:** Construct a graph connecting foods with sufficiently similar nutritional profiles. Score each candidate substitution using nutrient deviation, environmental improvement and data quality; analyze network communities and distinguish strong alternatives from environmentally attractive but nutritionally misleading substitutions.

## Deliverable 4

**Multi-Objective Basket/Menu Optimization:** Develop a constrained optimizer that redesigns a small set of fixed, transparent reference baskets or menus constructed from matched foods with declared quantities. Reduce environmental impact while preserving energy, protein and selected micronutrient requirements, returning a Pareto set of alternatives rather than one arbitrary optimum.

## Deliverable 5

**Robustness, Uncertainty and Counterexample Study:** Test whether conclusions remain stable under alternative nutrient weights, mass/calorie/protein normalization, graph-similarity thresholds, AGRIBALYSE data-quality weighting and bootstrap resampling. Document counterexamples where apparently sustainable substitutions fail after nutritional equivalence is enforced.

## Supporting file

Upload `docs/Sustainome_Project_Proposal_Note.pdf`. It contains the verified data path, methodology diagram, acceptance criteria, planned figures, schedule, limitations, and official source links.

