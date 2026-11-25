# Data sources and reproducibility manifest

## 1. ANSES-CIQUAL 2025

**Purpose:** nutrient composition, food identity, and food hierarchy.

**Official record:** <https://zenodo.org/records/17550133>

**English XLSX direct download:**

`https://zenodo.org/api/records/17550133/files/Table%20Ciqual%202025_ENG_2025_11_03.xlsx/content`

**Methodology PDF:** `references/data_sources/CIQUAL_2025_Documentation_EN.pdf`

**Audited workbook facts:**

- sheet: `food composition`;
- 3,484 data rows;
- 84 columns in the composition sheet;
- exact food key: `alim_code`;
- English food label: `alim_nom_eng`;
- group fields include `alim_grp_code`, `alim_ssgrp_code`, and corresponding English labels.

### Parsing cautions

- Nutrient headers contain line breaks and units; normalize column names while preserving an original-to-clean mapping.
- Values can include missing markers, trace values, inequalities, or non-numeric symbols. Define an explicit parser and preserve the raw value beside the numeric value when necessary.
- Do not replace missing values with zero.
- Confidence/quality information described by CIQUAL should not be conflated with AGRIBALYSE DQR.

## 2. AGRIBALYSE 3.2 - implementation default

**Purpose:** environmental impact indicators, CIQUAL linkage, and DQR.

**Official dataset DOI:** <https://doi.org/10.57745/XTENSJ>

**Dataverse record:** <https://entrepot.recherche.data.gouv.fr/dataset.xhtml?persistentId=doi:10.57745/XTENSJ>

**Food-products workbook:** Dataverse datafile ID `641911`

Direct API download:

`https://entrepot.recherche.data.gouv.fr/api/access/datafile/641911`

**Current English methodology:** `references/data_sources/AGRIBALYSE_3.2_Food_Methodology_EN.pdf`

**Current English change report:** `references/data_sources/AGRIBALYSE_3.2_Change_Report_EN.pdf`

**Current release slides:** `references/data_sources/AGRIBALYSE_3.2_Webinar_Slides_EN.pdf`

### Version rule

Inspect the 3.2 workbook before coding against its columns. Record workbook filename, release date, sheet names, row count, unique CIQUAL-code count, duplicates, and checksum. Resolve duplicates with a documented rule rather than silently dropping them.

## 3. AGRIBALYSE 3.1 API - feasibility-audit source only

**Endpoint used in the proposal audit:**

`https://data.ademe.fr/data-fair/api/v1/datasets/agribalyse-31-synthese/lines?size=3000`

**Audit results:**

- 2,451 returned rows;
- 2,425 unique CIQUAL codes;
- 2,272 exact matches to CIQUAL 2025;
- 93.69% of unique API CIQUAL codes matched.

Example fields observed in the API included `Code_CIQUAL`, `LCI_Name`, `DQR`, `Changement_climatique`, `Utilisation_du_sol`, `Eutrophisation_marine`, `Eutrophisation_eaux_douces`, `Acidification_terrestre_et_eaux_douces`, `Epuisement_des_ressources_eau`, and `Score_unique_EF` (French accents may appear in raw headers).

Use this endpoint only to reproduce the original feasibility audit or to compare releases. Do not mix its impact values with the 3.2 workbook.

## 4. Join specification

1. Preserve original codes as strings.
2. Trim whitespace and normalize numeric-like codes without losing leading-zero information.
3. Validate uniqueness separately in both sources.
4. Join on exact normalized CIQUAL code.
5. Produce matched, CIQUAL-only, AGRIBALYSE-only, and duplicate-resolution tables.
6. Manually inspect a sample of matched food names as a validation check, not as the join method.
7. Record row counts before and after every filtering step.

## 5. Snapshot manifest requirements

For every raw file record:

- source organization and dataset title;
- version/release;
- download URL;
- access date;
- filename and byte size;
- SHA-256 checksum;
- license/attribution requirement;
- sheet/table used;
- any local transformation.

Raw files belong under `data/raw/<source>/<version>/` and must never be edited in place.

