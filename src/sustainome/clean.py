"""Explicit parsing, code normalisation and duplicate resolution (Deliverable 1)."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass

import numpy as np
import pandas as pd

from . import config as C
from .io import norm_text

# ---------------------------------------------------------------------------
# CIQUAL food-code normalisation
# ---------------------------------------------------------------------------
_CODE_RE = re.compile(r"^\d+$")


@dataclass(frozen=True)
class NormalizedCode:
    raw: str
    code: str | None
    altered: bool
    reason: str


def normalize_code(value: object) -> NormalizedCode:
    """Canonical CIQUAL code: a base-10 integer string without padding.

    CIQUAL codes are integers, so leading zeros carry no identity. The raw text is kept
    and ``altered`` records whether normalisation changed it (e.g. ' 01234 ', '1234.0').
    Anything that is not an integer code (empty, alphabetic, '1234_2') is rejected rather
    than guessed.
    """
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return NormalizedCode("", None, False, "missing")
    raw = str(value)
    s = raw.strip()
    if isinstance(value, float) and value.is_integer():
        s = str(int(value))
    elif re.fullmatch(r"\d+\.0+", s):
        s = s.split(".")[0]
    if not _CODE_RE.match(s):
        return NormalizedCode(raw, None, False, "non-integer code")
    code = str(int(s))
    return NormalizedCode(raw, code, code != raw, "ok")


# ---------------------------------------------------------------------------
# CIQUAL value parser
# ---------------------------------------------------------------------------
STATUS_NUMERIC = "numeric"
STATUS_TRACE = "trace"
STATUS_BELOW_LOQ = "below_loq"
STATUS_MISSING = "missing"

_LT_RE = re.compile(r"^<\s*(\d+(?:[.,]\d+)?)$")
_NUM_RE = re.compile(r"^-?\d+(?:[.,]\d+)?(?:[eE][-+]?\d+)?$")


def parse_value(raw: object, policy: str = C.PARSE_POLICY) -> tuple[float, str]:
    """Parse one CIQUAL cell into ``(value, status)``.

    * ``'-'`` / empty  -> (nan, 'missing')  -- never zero
    * ``'traces'``     -> (0.0, 'trace')
    * ``'< x'``        -> policy: 'half_loq' x/2 (default), 'zero' 0, 'loq' x
    * ``'12,5'``       -> (12.5, 'numeric')   (decimal comma)
    """
    if policy not in C.PARSE_POLICIES:
        raise ValueError(f"unknown parse policy {policy!r}")
    if raw is None or (isinstance(raw, float) and math.isnan(raw)):
        return float("nan"), STATUS_MISSING
    if isinstance(raw, (int, float)):
        return float(raw), STATUS_NUMERIC
    s = " ".join(str(raw).split())  # removes embedded newlines such as '<\n0,03'
    if s in {"", "-"}:
        return float("nan"), STATUS_MISSING
    if s.casefold() == "traces":
        return 0.0, STATUS_TRACE
    m = _LT_RE.match(s)
    if m:
        loq = float(m.group(1).replace(",", "."))
        value = {"half_loq": loq / 2.0, "zero": 0.0, "loq": loq}[policy]
        return value, STATUS_BELOW_LOQ
    if _NUM_RE.match(s):
        return float(s.replace(",", ".")), STATUS_NUMERIC
    raise ValueError(f"unparseable CIQUAL value: {raw!r}")


def parse_series(s: pd.Series, policy: str = C.PARSE_POLICY) -> tuple[pd.Series, pd.Series]:
    out = [parse_value(v, policy) for v in s.tolist()]
    vals = pd.Series([o[0] for o in out], index=s.index, dtype=float)
    stat = pd.Series([o[1] for o in out], index=s.index, dtype="string")
    return vals, stat


# ---------------------------------------------------------------------------
# Column names and data dictionary
# ---------------------------------------------------------------------------
_HDR_RE = re.compile(r"^(?P<label>.*?)\s*\((?P<unit>[^()]*?)\s*100\s*(?P<basis>m?g)\)\s*$")


def split_header(header: str) -> tuple[str, str, str]:
    """'Protein\\n(g\\n100g)' -> ('Protein', 'g', '100g')."""
    h = " ".join(str(header).split())
    m = _HDR_RE.match(h)
    if not m:
        return h, "", ""
    return m.group("label").strip(), m.group("unit").strip(), "100" + m.group("basis")


def snake(label: str) -> str:
    s = norm_text(label)
    s = re.sub(r"[^0-9a-z]+", "_", s).strip("_")
    return s


def find_column(columns: list[str], header: str) -> str:
    target = norm_text(header)
    for c in columns:
        if norm_text(c) == target:
            return c
    raise KeyError(f"CIQUAL header not found: {header!r}")


# ---------------------------------------------------------------------------
# Preparation state (labelling only; used to avoid mixing raw/dry/cooked forms)
# ---------------------------------------------------------------------------
_PREPARED = r"cooked|boiled|steamed|baked|roasted|fried|grilled|braised|stewed|poached|microwaved|toasted|rehydrated|reconstituted|ready[- ]to[- ](?:drink|eat|feed)|drained|scrambled|sauteed|puree|mashed"
_DRY = r"\bdried\b|dehydrated|\bpowder\b|freeze-dried|\binstant\b"
_RAW = r"\braw\b"
# phrases describing what will happen later, not the state in which the food is described
_FUTURE = r"pre-?cooked|pre-?fried|to be [a-z-]+|to (?:cook|deep-fry|fry|reconstitute|rehydrate)\b[a-z ]*|for cooking|to be microwave|raw milk|raw vegetables"


def preparation_state(name: str) -> str:
    """Coarse physical state from the English name: prepared / dry / raw / as_sold."""
    n = re.sub(_FUTURE, " ", norm_text(name))
    if re.search(_RAW, n):
        return "raw"
    if re.search(_PREPARED, n):
        return "prepared"
    if re.search(_DRY, n):
        return "dry"
    if re.search(_RAW, n):
        return "raw"
    return "as_sold"


# ---------------------------------------------------------------------------
# AGRIBALYSE duplicate resolution
# ---------------------------------------------------------------------------
def _suffix(agb_code: str) -> int:
    m = re.search(r"_(\d+)$", str(agb_code))
    return int(m.group(1)) if m else 0


def resolve_agribalyse_duplicates(agb: pd.DataFrame, rule: str = "reference") -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return one row per CIQUAL code plus an audit table of every decision.

    Tier A (publisher packaging correction): when two rows share an AGB code and one is in
    red font with 'software version - with packaging error' in its name, keep the corrected
    (black) row, as instructed in the workbook Notice.

    Tier B (variants of one CIQUAL code, AGB codes '<code>_1', '_2', ...), ``rule``:
      * 'reference' (primary): prefer the French consumption mix (season code 2), then not
        air-freighted (plane code 0), then the unsuffixed AGB code, then the lowest suffix.
      * 'mean' / 'max' / 'min': aggregate impacts over variants (D5 sensitivity only).
    """
    a = agb.copy()
    a["ciqual_code"] = [normalize_code(v).code for v in a["ciqual_code_raw"]]
    a["dup_tier"] = ""
    audit = []

    # Tier A
    software = a["agb_lci_name"].astype(str).str.contains("software version", case=False)
    twins = a.groupby("agb_code")["agb_code"].transform("size") > 1
    drop_a = a["row_font_red"] & software & twins
    for _, r in a[drop_a].iterrows():
        keep = a[(a["agb_code"] == r["agb_code"]) & ~drop_a].iloc[0]
        audit.append({
            "ciqual_code": r["ciqual_code"], "tier": "A_packaging_fix", "kept_agb_code": keep["agb_code"],
            "kept_lci_name": keep["agb_lci_name"], "dropped_agb_code": r["agb_code"],
            "dropped_lci_name": r["agb_lci_name"], "kept_climate": keep["climate_per_kg"],
            "dropped_climate": r["climate_per_kg"], "reason": "publisher-corrected packaging row kept (Notice)",
        })
    a = a[~drop_a].copy()

    # Tier B
    impact_cols = [f"{k}_per_kg" for k in C.ENV_KEYS] + ["dqr"]
    out_rows = []
    for code, g in a.groupby("ciqual_code", sort=False):
        if len(g) == 1:
            out_rows.append(g.iloc[0].to_dict())
            continue
        g = g.assign(
            _season=(g["season_code"] != 2).astype(int),
            _plane=(g["plane_code"] == 1).astype(int),
            _suffix=[_suffix(c) for c in g["agb_code"]],
        ).sort_values(["_season", "_plane", "_suffix"])
        ref = g.iloc[0].to_dict()
        chosen = dict(ref)
        if rule in {"mean", "max", "min"}:
            agg = getattr(g[impact_cols], rule)()
            for col in impact_cols:
                chosen[col] = float(agg[col])
        elif rule != "reference":
            raise ValueError(rule)
        chosen["dup_tier"] = "B_variant"
        out_rows.append({k: v for k, v in chosen.items() if not k.startswith("_")})
        for _, r in g.iloc[1:].iterrows():
            audit.append({
                "ciqual_code": code, "tier": f"B_variant[{rule}]", "kept_agb_code": ref["agb_code"],
                "kept_lci_name": ref["agb_lci_name"], "dropped_agb_code": r["agb_code"],
                "dropped_lci_name": r["agb_lci_name"], "kept_climate": chosen["climate_per_kg"],
                "dropped_climate": r["climate_per_kg"],
                "reason": "season code 2 > plane code 0 > unsuffixed > lowest suffix"
                if rule == "reference" else f"impacts = {rule} over {len(g)} variants",
            })
    resolved = pd.DataFrame(out_rows)
    resolved["n_agb_variants"] = resolved["ciqual_code"].map(a.groupby("ciqual_code").size())
    return resolved.reset_index(drop=True), pd.DataFrame(audit)


def known_issue_flags(df: pd.DataFrame, name_col: str = "food_name", subgroup_col: str = "ciqual_subgroup_code") -> pd.DataFrame:
    """Flags from the AGRIBALYSE 3.2 Notice: eggs under-estimated, quinoa over-estimated.

    ``issue_direct``: the food itself is the affected product (egg sub-group; quinoa rows
    printed in red). ``issue_indirect``: the English name mentions an egg or quinoa
    ingredient - indicative only, because recipes are not available in the summary sheet.
    """
    name = df[name_col].fillna("").map(norm_text)
    egg_direct = df[subgroup_col].astype(str).eq("0410")
    quinoa_direct = name.str.contains("quinoa") & df.get("row_font_red", pd.Series(False, index=df.index)).fillna(False).astype(bool)
    egg_ind = name.str.contains(r"\beggs?\b|omelet|custard|flan|mayonnaise|meringue|brioche|quiche", regex=True) & ~egg_direct
    quinoa_ind = name.str.contains("quinoa") & ~quinoa_direct
    direct = np.where(egg_direct, "egg_underestimated", np.where(quinoa_direct, "quinoa_overestimated", ""))
    indirect = np.where(egg_ind, "may_contain_egg", np.where(quinoa_ind, "may_contain_quinoa", ""))
    return pd.DataFrame({"issue_direct": direct, "issue_indirect": indirect}, index=df.index)
