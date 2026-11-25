"""Small, dependency-light statistics: effect sizes and seeded bootstrap intervals."""

from __future__ import annotations

import numpy as np
from scipy import stats as ss


def rng(seed: int) -> np.random.Generator:
    return np.random.default_rng(seed)


def cliffs_delta(x, y) -> float:
    """P(X > Y) - P(X < Y); in [-1, 1]. Robust, scale-free effect size."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    x, y = x[np.isfinite(x)], y[np.isfinite(y)]
    if x.size == 0 or y.size == 0:
        return float("nan")
    gt = (x[:, None] > y[None, :]).sum()
    lt = (x[:, None] < y[None, :]).sum()
    return float((gt - lt) / (x.size * y.size))


def bootstrap_ci(stat, *samples, n_boot: int = 1000, seed: int = 0, alpha: float = 0.05):
    """Percentile bootstrap; each sample is resampled independently with replacement."""
    g = rng(seed)
    arrs = [np.asarray(s, dtype=float) for s in samples]
    point = stat(*arrs)
    boots = np.empty(n_boot)
    for b in range(n_boot):
        boots[b] = stat(*[a[g.integers(0, a.size, a.size)] for a in arrs])
    lo, hi = np.nanpercentile(boots, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(point), float(lo), float(hi)


def spearman(x, y) -> float:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 4 or np.ptp(x[ok]) == 0 or np.ptp(y[ok]) == 0:
        return float("nan")
    return float(ss.spearmanr(x[ok], y[ok]).statistic)


def _rowwise_pearson(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a = a - a.mean(axis=1, keepdims=True)
    b = b - b.mean(axis=1, keepdims=True)
    den = np.sqrt((a * a).sum(axis=1) * (b * b).sum(axis=1))
    with np.errstate(invalid="ignore", divide="ignore"):
        return (a * b).sum(axis=1) / den


def spearman_ci(x, y, n_boot: int = 1000, seed: int = 0):
    """Paired percentile-bootstrap CI for Spearman's rho (vectorised; ties -> average ranks)."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    point = spearman(x, y)
    if x.size < 4:
        return point, float("nan"), float("nan")
    idx = rng(seed).integers(0, x.size, (n_boot, x.size))
    rx = ss.rankdata(x[idx], axis=1)
    ry = ss.rankdata(y[idx], axis=1)
    boots = _rowwise_pearson(rx, ry)
    lo, hi = np.nanpercentile(boots, [2.5, 97.5])
    return point, float(lo), float(hi)


def kruskal_epsilon2(groups: list) -> tuple[float, float, float]:
    """Kruskal-Wallis H, p-value and epsilon-squared effect size."""
    gs = [np.asarray(g, dtype=float)[np.isfinite(g)] for g in groups]
    gs = [g for g in gs if g.size > 0]
    n = sum(g.size for g in gs)
    H, p = ss.kruskal(*gs)
    return float(H), float(p), float(H / (n - 1)) if n > 1 else float("nan")


def jaccard(a: set, b: set) -> float:
    if not a and not b:
        return 1.0
    return len(a & b) / len(a | b)
