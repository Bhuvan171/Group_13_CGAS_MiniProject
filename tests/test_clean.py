import math

import pandas as pd
import pytest

from sustainome import clean


@pytest.mark.parametrize("raw, code, altered", [
    ("1234", "1234", False),
    (" 1234 ", "1234", True),
    ("01234", "1234", True),
    ("1234.0", "1234", True),
    (1234, "1234", False),   # integer cell: its text form is already canonical
    (1234.0, "1234", True),  # float cell renders as '1234.0'
])
def test_normalize_code_valid(raw, code, altered):
    r = clean.normalize_code(raw)
    assert r.code == code
    assert r.altered is altered


@pytest.mark.parametrize("raw", [None, float("nan"), "", "abc", "1234_2", "12.5", "-3"])
def test_normalize_code_rejects_non_integer_codes(raw):
    assert clean.normalize_code(raw).code is None


def test_normalize_code_preserves_identity_of_distinct_codes():
    codes = ["1", "10", "100", "1000"]
    assert len({clean.normalize_code(c).code for c in codes}) == 4


@pytest.mark.parametrize("raw, value, status", [
    ("12,5", 12.5, "numeric"),
    ("0", 0.0, "numeric"),
    ("3700", 3700.0, "numeric"),
    ("traces", 0.0, "trace"),
    ("Traces", 0.0, "trace"),
    ("< 0,5", 0.25, "below_loq"),
    ("<\n0,03", 0.015, "below_loq"),
    ("<0,2", 0.1, "below_loq"),
])
def test_parse_value_cases(raw, value, status):
    v, s = clean.parse_value(raw)
    assert s == status
    assert v == pytest.approx(value)


@pytest.mark.parametrize("raw", ["-", "", None, float("nan"), " - "])
def test_missing_is_never_zero(raw):
    v, s = clean.parse_value(raw)
    assert s == "missing"
    assert math.isnan(v)


def test_parse_policies_for_inequality():
    assert clean.parse_value("< 2", "half_loq")[0] == 1.0
    assert clean.parse_value("< 2", "zero")[0] == 0.0
    assert clean.parse_value("< 2", "loq")[0] == 2.0
    with pytest.raises(ValueError):
        clean.parse_value("< 2", "made-up")


def test_parse_value_rejects_garbage():
    with pytest.raises(ValueError):
        clean.parse_value("about 3")


def test_split_header():
    assert clean.split_header("Protein\n(g\n100g)") == ("Protein", "g", "100g")
    label, unit, basis = clean.split_header("Vitamin A\nactivity,\nretinol\nequivalent\n(µg\n100mg)")
    assert unit == "µg" and basis == "100mg"


@pytest.mark.parametrize("name, state", [
    ("Lentil, green, boiled/cooked in water", "prepared"),
    ("Orange juice, from concentrate", "as_sold"),
    ("Couscous (pre-cooked durum wheat semolina), raw", "raw"),
    ("French fries or chips, frozen, pre-fried, raw, to cook in the oven", "raw"),
    ("Milk powder, skimmed", "dry"),
    ("Baby instant cereal (powder to be reconstituted), from 4/6 months", "dry"),
    ("Camembert cheese, from cow's milk, from raw milk or thermised milk", "as_sold"),
])
def test_preparation_state(name, state):
    assert clean.preparation_state(name) == state


def _agb_rows():
    base = {f"{k}_per_kg": 1.0 for k in ["climate", "land", "water", "eutro_fw", "eutro_marine",
                                         "acidification", "fossil", "ef_score"]}
    rows = [
        # tier A: packaging error twin (same AGB code)
        dict(agb_code="100", ciqual_code_raw=100, agb_lci_name="X (software version - with packaging error)",
             row_font_red=True, season_code=2, plane_code=0, dqr=2.0, **{**base, "climate_per_kg": 1.0}),
        dict(agb_code="100", ciqual_code_raw=100, agb_lci_name="X (packaging fixed)",
             row_font_red=False, season_code=2, plane_code=0, dqr=2.0, **{**base, "climate_per_kg": 1.5}),
        # tier B: seasonal and air-freighted variants
        dict(agb_code="200_2", ciqual_code_raw=200, agb_lci_name="Y in-season", row_font_red=False,
             season_code=1, plane_code=0, dqr=2.0, **{**base, "climate_per_kg": 0.5}),
        dict(agb_code="200_1", ciqual_code_raw=200, agb_lci_name="Y", row_font_red=False,
             season_code=2, plane_code=0, dqr=2.0, **{**base, "climate_per_kg": 0.6}),
        dict(agb_code="200_3", ciqual_code_raw=200, agb_lci_name="Y by plane", row_font_red=False,
             season_code=2, plane_code=1, dqr=2.0, **{**base, "climate_per_kg": 9.0}),
        # red row WITHOUT a twin (known-error product) must be kept
        dict(agb_code="300", ciqual_code_raw=300, agb_lci_name="Quinoa, raw", row_font_red=True,
             season_code=2, plane_code=0, dqr=2.0, **base),
    ]
    return pd.DataFrame(rows)


def test_duplicate_resolution_reference_rule():
    res, audit = clean.resolve_agribalyse_duplicates(_agb_rows())
    assert res["ciqual_code"].is_unique
    r = res.set_index("ciqual_code")
    assert r.loc["100", "climate_per_kg"] == 1.5          # corrected packaging row
    assert r.loc["200", "agb_code"] == "200_1"            # French consumption mix, not air-freighted
    assert "300" in r.index                               # red row without twin kept
    assert (audit["tier"] == "A_packaging_fix").sum() == 1
    assert audit["tier"].str.startswith("B_variant").sum() == 2


@pytest.mark.parametrize("rule, expected", [("mean", (0.5 + 0.6 + 9.0) / 3), ("max", 9.0), ("min", 0.5)])
def test_duplicate_resolution_alternatives(rule, expected):
    res, _ = clean.resolve_agribalyse_duplicates(_agb_rows(), rule=rule)
    assert res.set_index("ciqual_code").loc["200", "climate_per_kg"] == pytest.approx(expected)
