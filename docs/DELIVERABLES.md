# Deliverables and acceptance criteria

## D1 - Sustainable Food Matrix

### Required outputs

- versioned integrated dataset;
- source-to-clean data dictionary;
- reproducible build module/notebook;
- matched/unmatched/duplicate audit tables;
- coverage and missingness figures;
- provenance and snapshot manifest.

### Acceptance criteria

- primary join uses exact normalized CIQUAL codes;
- source versions and checksums are recorded;
- counts reconcile across every pipeline stage;
- missing nutrients are not silently zero-filled;
- duplicate handling is documented and tested;
- the processed dataset can be regenerated from frozen raw files.

## D2 - Nutrition-Environment Trade-off Analysis

### Required outputs

- comparable food-group definition;
- mass-, calorie-, and protein-normalized analytical tables;
- Pareto-efficient/dominated labels;
- group-wise statistics and effect sizes;
- Pareto fronts and nutrient-environment figures;
- concise scientific findings and limitations.

### Acceptance criteria

- comparisons do not mix incompatible food roles without labelling;
- denominator safeguards are implemented;
- nutritional value is not reduced to “more of every nutrient is better”;
- findings are compared across normalization views;
- at least several non-obvious trade-offs are documented.

## D3 - Nutritional-Equivalence Substitution Network

### Required outputs

- graph node/edge tables;
- documented nutrient-distance and edge-eligibility rule;
- network metrics and community analysis;
- reliable and misleading substitution examples;
- network and edge trade-off figures.

### Acceptance criteria

- graph construction is reproducible and leakage-free;
- edge components remain inspectable;
- threshold choice is justified;
- results include sensitivity to alternate graph thresholds in D5;
- representative top edges are manually audited.

## D4 - Multi-Objective Basket/Menu Optimization

### Required outputs

- 3-5 fixed scenarios with declared foods and quantities;
- formal variable, objective, and constraint specification;
- reproducible optimizer;
- original, random-feasible, and single-objective baselines;
- Pareto solution sets and diagnostic figures.

### Acceptance criteria

- all reported solutions satisfy constraints or display violations explicitly;
- no hidden external menu dataset is required;
- multiple objectives remain visible rather than being hidden in one score;
- results include at least one case where objectives genuinely conflict;
- outputs are interpreted as computational scenarios, not dietary advice.

## D5 - Robustness, Uncertainty and Counterexample Study

### Required outputs

- predeclared sensitivity grid;
- bootstrap or resampling results where appropriate;
- DQR-based sensitivity;
- rank/community/solution stability measures;
- categorized counterexample set;
- stability and uncertainty figures.

### Acceptance criteria

- every major conclusion is tested under more than one plausible assumption;
- unstable claims are labelled or withdrawn;
- DQR is not misrepresented as probabilistic uncertainty;
- negative or counterintuitive cases are included rather than discarded;
- the final report separates robust findings from exploratory findings.

## Distinctness rule

D1 is a reusable data artifact. D2 is food-level inference. D3 is pairwise/network modelling. D4 is joint basket-level decision optimization. D5 is validation and falsification. A single notebook or figure cannot be counted toward multiple deliverables unless it produces separate, explicitly evaluated outputs.

