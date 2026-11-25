"""Pareto dominance and non-dominated sorting.

``sense`` is +1 for objectives to minimise and -1 for objectives to maximise. Rows with
any missing objective are excluded (returned as False / NaN) and must be reported by the
caller - they are never treated as zeros.
"""

from __future__ import annotations

import numpy as np


def _as_min(X: np.ndarray, sense) -> np.ndarray:
    return np.asarray(X, dtype=float) * np.asarray(sense, dtype=float)[None, :]


def dominates(a, b, sense) -> bool:
    """True if a Pareto-dominates b (no worse everywhere, strictly better somewhere)."""
    a = np.asarray(a, dtype=float) * np.asarray(sense, dtype=float)
    b = np.asarray(b, dtype=float) * np.asarray(sense, dtype=float)
    return bool(np.all(a <= b) and np.any(a < b))


def dominance_matrix(X: np.ndarray, sense) -> np.ndarray:
    """D[i, j] = True if row i dominates row j."""
    Y = _as_min(X, sense)
    le = (Y[:, None, :] <= Y[None, :, :]).all(axis=2)
    lt = (Y[:, None, :] < Y[None, :, :]).any(axis=2)
    return le & lt


def pareto_mask(X: np.ndarray, sense) -> np.ndarray:
    """Boolean mask of non-dominated rows (NaN rows -> False)."""
    X = np.asarray(X, dtype=float)
    ok = np.isfinite(X).all(axis=1)
    out = np.zeros(len(X), dtype=bool)
    if ok.sum() == 0:
        return out
    D = dominance_matrix(X[ok], sense)
    out[np.where(ok)[0]] = ~D.any(axis=0)
    return out


def nondominated_rank(X: np.ndarray, sense) -> np.ndarray:
    """Dominance depth: 1 = Pareto front, 2 = front after removing front 1, ... (NaN rows -> NaN)."""
    X = np.asarray(X, dtype=float)
    ok = np.where(np.isfinite(X).all(axis=1))[0]
    rank = np.full(len(X), np.nan)
    if ok.size == 0:
        return rank
    D = dominance_matrix(X[ok], sense)
    remaining = np.ones(ok.size, dtype=bool)
    level = 1
    while remaining.any():
        sub = D[np.ix_(remaining, remaining)]
        front_local = ~sub.any(axis=0)
        idx = np.where(remaining)[0][front_local]
        rank[ok[idx]] = level
        remaining[idx] = False
        level += 1
    return rank


def domination_count(X: np.ndarray, sense) -> np.ndarray:
    """How many rows dominate each row (0 for efficient rows; NaN rows -> NaN)."""
    X = np.asarray(X, dtype=float)
    ok = np.isfinite(X).all(axis=1)
    out = np.full(len(X), np.nan)
    if ok.any():
        out[ok] = dominance_matrix(X[ok], sense).sum(axis=0)
    return out
