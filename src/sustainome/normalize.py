"""Functional-unit views: per 100 g, per 100 kcal and per 10 g protein.

No view is ground truth. Each conversion is guarded: foods whose energy or protein
density is below a declared floor are *undefined* in that view (NaN + a reason), never
divided by a near-zero denominator.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import config as C


def fu_factor(m: pd.DataFrame, view: str, energy_floor: float = C.ENERGY_FLOOR_KCAL,
              protein_floor: float = C.PROTEIN_FLOOR_G) -> tuple[pd.Series, pd.Series]:
    """Multiplier converting per-100 g quantities to the view's functional unit.

    Returns ``(factor, reason)``; factor is NaN where the view is undefined.
    """
    if view == "mass":
        return pd.Series(1.0, index=m.index), pd.Series("", index=m.index)
    if view == "energy":
        e = m["energy_kcal"]
        ok = e.notna() & (e >= energy_floor)
        reason = np.where(e.isna(), "energy missing", np.where(~ok, f"energy < {energy_floor:g} kcal/100 g", ""))
        return (100.0 / e).where(ok), pd.Series(reason, index=m.index)
    if view == "protein":
        p = m["protein_g"]
        ok = p.notna() & (p >= protein_floor)
        reason = np.where(p.isna(), "protein missing", np.where(~ok, f"protein < {protein_floor:g} g/100 g", ""))
        return (10.0 / p).where(ok), pd.Series(reason, index=m.index)
    raise ValueError(view)


def to_view(m: pd.DataFrame, view: str, **floors) -> pd.DataFrame:
    """Nutrients and environmental indicators expressed per functional unit."""
    f, reason = fu_factor(m, view, **floors)
    out = pd.DataFrame(index=m.index)
    out["ciqual_code"] = m["ciqual_code"]
    out["view"] = view
    out["fu"] = C.VIEW_UNIT[view]
    out["fu_defined"] = f.notna()
    out["fu_undefined_reason"] = reason
    out["grams_per_fu"] = 100.0 * f
    for k in C.NUTRIENT_KEYS:
        out[k] = m[k] * f
    for k in C.ENV_KEYS:
        out[k] = m[f"{k}_per_100g"] * f
    return out


def relative_change(base: np.ndarray | pd.Series, new: np.ndarray | pd.Series, floor: float) -> np.ndarray:
    """(new - base) / max(|base|, floor): a relative change that cannot explode when the
    baseline is tiny or negative (land use and water scarcity can be negative in EF 3.1)."""
    base = np.asarray(base, dtype=float)
    new = np.asarray(new, dtype=float)
    return (new - base) / np.maximum(np.abs(base), floor)


def env_floors(m: pd.DataFrame) -> dict[str, float]:
    """Per-indicator denominator floor = 5th percentile of positive per-100 g values."""
    out = {}
    for k in C.ENV_KEYS:
        v = m[f"{k}_per_100g"]
        out[k] = float(np.nanpercentile(v[v > 0], 5))
    return out
