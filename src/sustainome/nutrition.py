"""Nutritional value without 'more of every nutrient is better'.

* NR9 - sum over nine qualifying nutrients of %reference intake per functional unit,
  each capped at 100 % (adequacy, not maximisation).
* LIM3 - sum of %maximum reference value for saturated fat, total sugars and sodium,
  uncapped and reported *separately* from NR9.
* Nutrient-profile geometry - asinh-compressed, robustly scaled profiles and a weighted,
  missing-aware distance used to define nutritional equivalence (D3) and deviation (D4).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import config as C


def nr9(fu: pd.DataFrame, keys: list[str] = C.NR_KEYS, cap: float = 100.0, min_available: int | None = None) -> pd.Series:
    """Capped qualifying-nutrient score per functional unit.

    ``min_available=None`` (primary) requires every component (complete case). An integer
    enables the D5 available-case variant: mean of available capped %RI x len(keys).
    """
    pct = pd.DataFrame({k: (100.0 * fu[k] / C.NUTRIENT_REF[k]).clip(upper=cap) for k in keys})
    if min_available is None:
        return pct.sum(axis=1, min_count=len(keys)).where(pct.notna().all(axis=1))
    n = pct.notna().sum(axis=1)
    return (pct.mean(axis=1) * len(keys)).where(n >= min_available)


def lim3(fu: pd.DataFrame, keys: list[str] = C.LIM_KEYS) -> pd.Series:
    pct = pd.DataFrame({k: 100.0 * fu[k] / C.NUTRIENT_REF[k] for k in keys})
    return pct.sum(axis=1).where(pct.notna().all(axis=1))


def add_scores(fu: pd.DataFrame, **kw) -> pd.DataFrame:
    fu = fu.copy()
    fu["nr9"] = nr9(fu, **kw)
    fu["lim3"] = lim3(fu)
    fu["nrf9_3"] = fu["nr9"] - fu["lim3"]
    return fu


# ---------------------------------------------------------------------------
# nutrient-profile geometry
# ---------------------------------------------------------------------------
class ProfileScaler:
    """asinh(x / s_k) followed by robust (median / IQR) standardisation.

    Fitted on nutrient columns only - environmental data never enter the similarity
    geometry, so the substitution graph cannot leak the outcome it is used to evaluate.
    """

    def __init__(self, keys: list[str] = C.NUTRIENT_KEYS):
        self.keys = list(keys)

    def fit(self, X: pd.DataFrame) -> "ProfileScaler":
        self.s_ = {}
        self.med_ = {}
        self.iqr_ = {}
        for k in self.keys:
            v = X[k].to_numpy(dtype=float)
            pos = v[np.isfinite(v) & (v > 0)]
            self.s_[k] = float(np.median(pos)) if pos.size else 1.0
            z = np.arcsinh(v / self.s_[k])
            self.med_[k] = float(np.nanmedian(z))
            q75, q25 = np.nanpercentile(z, [75, 25])
            self.iqr_[k] = float(q75 - q25) if q75 > q25 else float(np.nanstd(z) or 1.0)
        return self

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        cols = [(np.arcsinh(X[k].to_numpy(dtype=float) / self.s_[k]) - self.med_[k]) / self.iqr_[k] for k in self.keys]
        return np.column_stack(cols)


def pairwise_distance(Za: np.ndarray, Zb: np.ndarray, w: np.ndarray, min_shared: int) -> tuple[np.ndarray, np.ndarray]:
    """Weighted RMS distance over features observed in *both* foods.

    Returns ``(D, shared)`` of shape (len(Za), len(Zb)); D is NaN where fewer than
    ``min_shared`` features are jointly observed. Missing values are never zero-filled.
    """
    w = np.asarray(w, dtype=float)
    ma = np.isfinite(Za)
    mb = np.isfinite(Zb)
    A = np.where(ma, Za, 0.0)
    B = np.where(mb, Zb, 0.0)
    both = ma[:, None, :] & mb[None, :, :]
    diff2 = (A[:, None, :] - B[None, :, :]) ** 2
    num = (diff2 * both * w).sum(axis=2)
    den = (both * w).sum(axis=2)
    shared = both.sum(axis=2)
    with np.errstate(invalid="ignore", divide="ignore"):
        D = np.sqrt(num / den)
    D[shared < min_shared] = np.nan
    return D, shared


def equal_weights(keys: list[str] = C.NUTRIENT_KEYS) -> np.ndarray:
    return np.ones(len(keys)) / len(keys)
