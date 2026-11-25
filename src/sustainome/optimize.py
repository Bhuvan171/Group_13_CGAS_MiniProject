"""Deliverable 4 - Multi-Objective Basket/Menu Optimisation.

Formulation (per scenario, all quantities in grams):

    variables   x[s,j] >= 0  amount of candidate food j placed in basket slot s
                u[k]   >= 0  relative deviation of nutrient k from the baseline basket
                v[s,j] >= 0  absolute change of x[s,j] from the baseline
                y[s,j] in {0,1} (optional) - candidate j (not the original) is used in slot s
    constraints slot mass within +/- 25 %; total energy within +/- 5 %;
                protein >= 95 %; fibre and each micronutrient >= 90 %;
                sodium, saturated fat, sugars <= baseline;
                optional: mean deviation <= eps_D, change share <= eps_C,
                          number of new foods <= K, water <= eps_W
    objectives  climate (min), nutrient-profile deviation D (min),
                basket change C (min); land, water, DQR reported for every solution.

A Pareto set is generated with an epsilon-constraint sweep - objectives are never folded
into one weighted score. Every reported solution is re-checked against every constraint.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import csr_matrix

from . import config as C
from . import pareto

TOL = 1e-6
# Deterministic stopping rule for MILPs: stop within 0.5 % of the proven optimum (a time
# limit alone would make results depend on machine speed). LPs are solved to optimality.
SOLVER_OPTIONS = {"disp": False, "time_limit": 300, "mip_rel_gap": 0.005}


@dataclass
class Problem:
    scenario: str
    label: str
    slots: pd.DataFrame          # one row per basket slot
    var: pd.DataFrame            # one row per (slot, candidate food)
    A_nut: np.ndarray            # (n_var, K) nutrient per gram
    A_env: np.ndarray            # (n_var, E) impact per gram
    x0: np.ndarray
    N0: np.ndarray
    dev_den: np.ndarray
    params: dict = field(default_factory=dict)

    @property
    def n(self) -> int:
        return len(self.var)


# ---------------------------------------------------------------------------
# problem construction
# ---------------------------------------------------------------------------
def candidates(m: pd.DataFrame, role: str, exclude_issues: bool = True, max_dqr: float | None = None) -> pd.DataFrame:
    c = m[m["roles"].fillna("").str.split("|").map(lambda rs: role in rs)]
    c = c[c["profile_complete"]]
    if exclude_issues:
        c = c[c["issue_direct"].fillna("") == ""]
    if max_dqr is not None:
        c = c[c["dqr"] <= max_dqr]
    return c


def build_problem(m: pd.DataFrame, baskets: pd.DataFrame, scenario: str, params: dict | None = None,
                  exclude_issues: bool = True, max_dqr: float | None = None) -> Problem:
    p = dict(C.OPT)
    p.update(params or {})
    if scenario == "S5":  # one-day illustrative basket = S1 + S2 + S4
        b = baskets[baskets["scenario"].isin(["S1", "S2", "S4"])].copy()
        b["slot"] = b["scenario"] + "." + b["slot"]
        label = "One-day basket (S1+S2+S4)"
    else:
        b = baskets[baskets["scenario"] == scenario].copy()
        label = b["scenario_label"].iloc[0]
    b["grams"] = b["grams"].astype(float)
    mi = m.set_index("ciqual_code")
    rows = []
    for _, s in b.iterrows():
        cand = candidates(m, s["role"], exclude_issues, max_dqr)
        if s["ciqual_code"] not in set(cand["ciqual_code"]):
            orig = mi.loc[[s["ciqual_code"]]].reset_index()
            if not bool(orig["profile_complete"].iloc[0]):
                raise ValueError(f"baseline food {s['ciqual_code']} lacks a complete nutrient profile")
            cand = pd.concat([cand, orig], ignore_index=True)
        for _, f in cand.iterrows():
            rows.append({"slot": s["slot"], "role": s["role"], "ciqual_code": f["ciqual_code"],
                         "food_name": f["food_name"], "is_original": f["ciqual_code"] == s["ciqual_code"],
                         "x0": s["grams"] if f["ciqual_code"] == s["ciqual_code"] else 0.0,
                         "slot_mass": s["grams"], "dqr": f["dqr"], "origin_group": f["comp_group"]})
    var = pd.DataFrame(rows)
    F = mi.loc[var["ciqual_code"]]
    A_nut = F[C.NUTRIENT_KEYS].to_numpy(float) / 100.0
    A_env = F[[f"{k}_per_kg" for k in C.ENV_KEYS]].to_numpy(float) / 1000.0
    x0 = var["x0"].to_numpy(float)
    N0 = A_nut.T @ x0
    ref = np.array([C.NUTRIENT_REF[k] for k in C.NUTRIENT_KEYS])
    dev_den = np.maximum(N0, p["dev_floor_dv"] * ref)
    slots = b.reset_index(drop=True)
    return Problem(scenario, label, slots, var, A_nut, A_env, x0, N0, dev_den, p)


# ---------------------------------------------------------------------------
# constraint rows (shared by the solver and the checker)
# ---------------------------------------------------------------------------
def _k(key: str) -> int:
    return C.NUTRIENT_KEYS.index(key)


def nutrition_rows(P: Problem, p: dict) -> list[tuple[str, np.ndarray, float, float]]:
    """(name, coefficients on x, lower, upper) for every hard nutrition constraint."""
    rows = []
    e = _k("energy_kcal")
    rows.append(("energy", P.A_nut[:, e], (1 - p["energy_tol"]) * P.N0[e], (1 + p["energy_tol"]) * P.N0[e]))
    pk = _k("protein_g")
    rows.append(("protein", P.A_nut[:, pk], p["protein_min"] * P.N0[pk], np.inf))
    for key in C.OPT_MICRO:
        k = _k(key)
        rows.append((key, P.A_nut[:, k], p["micro_min"] * P.N0[k], np.inf))
    for key in C.LIMIT_KEYS:
        k = _k(key)
        rows.append((key, P.A_nut[:, k], -np.inf, p["limit_max"] * P.N0[k]))
    return rows


def structure_rows(P: Problem, p: dict) -> list[tuple[str, np.ndarray, float, float]]:
    rows = []
    for s, g in P.var.groupby("slot", sort=False):
        a = np.zeros(P.n)
        a[g.index] = 1.0
        M = g["slot_mass"].iloc[0]
        rows.append((f"mass[{s}]", a, (1 - p["role_mass_tol"]) * M, (1 + p["role_mass_tol"]) * M))
    return rows


def check(P: Problem, x: np.ndarray, p: dict | None = None, structure: bool = True) -> pd.DataFrame:
    """Evaluate every declared constraint for a solution; violations are explicit."""
    p = p or P.params
    key = (structure,) + tuple(sorted((k, v) for k, v in p.items() if not k.startswith("_") and not isinstance(v, (list, dict))))
    cache = P.params.setdefault("_rows_cache", {})
    if key not in cache:
        cache[key] = (structure_rows(P, p) if structure else []) + nutrition_rows(P, p)
    rows = cache[key]
    out = []
    for name, a, lo, hi in rows:
        val = float(a @ x)
        scale = max(1.0, abs(lo) if np.isfinite(lo) else 0, abs(hi) if np.isfinite(hi) else 0)
        viol = max(0.0, (lo - val) if np.isfinite(lo) else 0.0, (val - hi) if np.isfinite(hi) else 0.0)
        out.append({"constraint": name, "value": val, "lower": lo, "upper": hi,
                    "violation": viol, "relative_violation": viol / scale, "satisfied": viol <= 1e-5 * scale})
    return pd.DataFrame(out)


# ---------------------------------------------------------------------------
# solver
# ---------------------------------------------------------------------------
def solve(P: Problem, objective: str = "climate", eps_dev: float | None = None, eps_change: float | None = None,
          max_new: int | None = None, env_caps: dict | None = None, nutrition: bool = True,
          cost: np.ndarray | None = None, params: dict | None = None) -> dict | None:
    """Minimise one objective subject to constraints and epsilon budgets. Returns a result dict."""
    p = dict(P.params)
    p.update(params or {})
    n, K = P.n, len(C.NUTRIENT_KEYS)
    use_y = max_new is not None
    new_idx = np.where(~P.var["is_original"].to_numpy())[0] if use_y else np.array([], int)
    ny = len(new_idx)
    N = n + K + n + ny  # x | u | v | y
    sx, su, sv, sy = 0, n, n + K, n + K + n

    A_rows, lo, hi = [], [], []

    def add(coef_x=None, coef_u=None, coef_v=None, coef_y=None, l=-np.inf, h=np.inf):
        r = np.zeros(N)
        if coef_x is not None:
            r[sx:sx + n] = coef_x
        if coef_u is not None:
            r[su:su + K] = coef_u
        if coef_v is not None:
            r[sv:sv + n] = coef_v
        if coef_y is not None:
            r[sy:sy + ny] = coef_y
        A_rows.append(r)
        lo.append(l)
        hi.append(h)

    for _, a, l, h in structure_rows(P, p) + (nutrition_rows(P, p) if nutrition else []):
        add(coef_x=a, l=l, h=h)
    # deviation: |A_k x - N0_k| / den_k <= u_k
    for k in range(K):
        a = P.A_nut[:, k] / P.dev_den[k]
        e = np.zeros(K)
        e[k] = -1.0
        add(coef_x=a, coef_u=e, h=P.N0[k] / P.dev_den[k])
        add(coef_x=-a, coef_u=e, h=-P.N0[k] / P.dev_den[k])
    if eps_dev is not None:
        add(coef_u=np.ones(K) / K, h=eps_dev)
    # change: |x - x0| <= v
    I = np.eye(n)
    for j in range(n):
        add(coef_x=I[j], coef_v=-I[j], h=P.x0[j])
        add(coef_x=-I[j], coef_v=-I[j], h=-P.x0[j])
    total0 = P.x0.sum()
    if eps_change is not None:
        add(coef_v=np.ones(n) / (2 * total0), h=eps_change)
    if use_y:
        ub_x = (1 + p["role_mass_tol"]) * P.var["slot_mass"].to_numpy()
        for t, j in enumerate(new_idx):
            ry = np.zeros(ny)
            ry[t] = -ub_x[j]
            add(coef_x=I[j], coef_y=ry, h=0.0)
        add(coef_y=np.ones(ny), h=max_new)
    for key, cap in (env_caps or {}).items():
        add(coef_x=P.A_env[:, C.ENV_KEYS.index(key)], h=cap)

    c = np.zeros(N)
    if cost is not None:
        c[sx:sx + n] = cost
    elif objective in C.ENV_KEYS:
        c[sx:sx + n] = P.A_env[:, C.ENV_KEYS.index(objective)]
        scale = abs(P.A_env[:, C.ENV_KEYS.index(objective)] @ P.x0) or 1.0
        c = c / scale
        c[su:su + K] += 1e-4 / K           # tie-breakers: avoid weakly dominated points
        c[sv:sv + n] += 1e-4 / (2 * total0)
    elif objective == "deviation":
        c[su:su + K] = 1.0 / K
    elif objective == "change":
        c[sv:sv + n] = 1.0 / (2 * total0)
    else:
        raise ValueError(objective)

    ub = np.full(N, np.inf)
    ub[sx:sx + n] = (1 + p["role_mass_tol"]) * P.var["slot_mass"].to_numpy()
    lb = np.zeros(N)
    integrality = np.zeros(N)
    if use_y:
        ub[sy:sy + ny] = 1.0
        integrality[sy:sy + ny] = 1
    res = milp(c, constraints=LinearConstraint(csr_matrix(np.array(A_rows)), lo, hi), bounds=Bounds(lb, ub),
               integrality=integrality, options=SOLVER_OPTIONS)
    # status 0 = optimal (MILPs: within the declared relative gap); 1 = time limit with an incumbent
    if res.x is None or res.status not in (0, 1):
        return None
    x = np.clip(res.x[sx:sx + n], 0, None)
    return summarize(P, x, p, tag={"objective": objective, "eps_dev": eps_dev, "eps_change": eps_change,
                                   "max_new": max_new, "env_caps": str(env_caps or {}), "nutrition_constraints": nutrition,
                                   "solver_status": int(res.status), "mip_gap": float(getattr(res, "mip_gap", 0.0) or 0.0)})


def summarize(P: Problem, x: np.ndarray, p: dict | None = None, tag: dict | None = None) -> dict:
    p = p or P.params
    K = len(C.NUTRIENT_KEYS)
    Nx = P.A_nut.T @ x
    dev = np.abs(Nx - P.N0) / P.dev_den
    env = P.A_env.T @ x
    env0 = P.A_env.T @ P.x0
    used = x > 0.5
    new = used & ~P.var["is_original"].to_numpy()
    chk = check(P, x, p)
    rec = {**(tag or {}),
           "mean_deviation": float(dev.mean()), "max_deviation": float(dev.max()),
           "change_share": float(np.abs(x - P.x0).sum() / (2 * P.x0.sum())),
           "n_new_foods": int(new.sum()), "n_foods": int(used.sum()), "total_mass": float(x.sum()),
           "dqr_mass_weighted": float((P.var["dqr"].to_numpy() * x).sum() / x.sum()),
           "feasible": bool(chk["satisfied"].all()), "n_violations": int((~chk["satisfied"]).sum()),
           "max_relative_violation": float(chk["relative_violation"].max())}
    for i, k in enumerate(C.ENV_KEYS):
        rec[k] = float(env[i])
        rec[f"{k}_change"] = float(env[i] / env0[i] - 1) if env0[i] != 0 else np.nan
    for i, k in enumerate(C.NUTRIENT_KEYS):
        rec[f"ret_{k}"] = float(Nx[i] / P.N0[i]) if P.N0[i] > 0 else np.nan
    rec["x"] = x
    return rec


# ---------------------------------------------------------------------------
# experiments
# ---------------------------------------------------------------------------
def composition(P: Problem, x: np.ndarray, min_g: float = 0.5) -> pd.DataFrame:
    v = P.var.assign(grams=x)
    return v[v["grams"] > min_g][["slot", "role", "ciqual_code", "food_name", "is_original", "x0", "grams", "dqr"]]


def pareto_sweep(P: Problem) -> pd.DataFrame:
    sols = []
    for ec in P.params["eps_change"]:
        for ed in P.params["eps_dev"]:
            r = solve(P, "climate", eps_dev=ed, eps_change=ec)
            if r is not None:
                sols.append(r)
    df = pd.DataFrame(sols)
    if df.empty:
        return df
    X = np.round(df[["climate", "mean_deviation", "change_share"]].to_numpy(), 6)
    df["duplicate"] = pd.DataFrame(X).duplicated().to_numpy()
    df["pareto"] = pareto.pareto_mask(X, [1, 1, 1]) & ~df["duplicate"]
    return df


def swap_front(P: Problem, max_k: int | None = None, eps_dev: float | None = None) -> pd.DataFrame:
    """Climate optimum as a function of the number of new foods allowed (MILP)."""
    max_k = max_k if max_k is not None else min(len(P.slots), 10)
    out = []
    for k in range(0, max_k + 1):
        r = solve(P, "climate", max_new=k, eps_dev=eps_dev)
        if r is not None:
            out.append(r)
    return pd.DataFrame(out)


def env_conflict(P: Problem, other: str = "water", n: int = 9) -> pd.DataFrame:
    """Climate vs another indicator: min climate subject to other <= cap, swept between the
    other-optimal and the climate-optimal basket."""
    r_c = solve(P, "climate")
    r_o = solve(P, other)
    if r_c is None or r_o is None:
        return pd.DataFrame()
    caps = np.linspace(r_o[other], max(r_c[other], r_o[other]), n)
    out = []
    for cap in caps:
        r = solve(P, "climate", env_caps={other: cap * (1 + 1e-9)})
        if r is not None:
            r["cap_" + other] = cap
            out.append(r)
    return pd.DataFrame(out)


def random_swaps(P: Problem, n: int, seed: int, p_swap: float = 0.5) -> pd.DataFrame:
    """Baseline: random role-preserving 1:1 swaps (same grams) - how often are they feasible?"""
    g = np.random.default_rng(seed)
    slot_idx = {s: grp.index.to_numpy() for s, grp in P.var.groupby("slot", sort=False)}
    orig = {s: grp.index[grp["is_original"]][0] for s, grp in P.var.groupby("slot", sort=False)}
    out = []
    for _ in range(n):
        x = np.zeros(P.n)
        for s, idx in slot_idx.items():
            j = g.choice(idx) if g.random() < p_swap else orig[s]
            x[j] = P.var.loc[orig[s], "slot_mass"]
        r = summarize(P, x, tag={"objective": "random swap"})
        r.pop("x")
        out.append(r)
    return pd.DataFrame(out)


def random_feasible(P: Problem, n: int, seed: int) -> pd.DataFrame:
    """Baseline: random feasible baskets - random linear objectives over the feasible set."""
    g = np.random.default_rng(seed)
    out = []
    for _ in range(n):
        r = solve(P, cost=g.normal(size=P.n))
        if r is not None:
            r["objective"] = "random feasible"
            r.pop("x")
            out.append(r)
    return pd.DataFrame(out)


def run_scenario(m: pd.DataFrame, baskets: pd.DataFrame, scenario: str, seed: int = C.SEED,
                 n_random: int | None = None, params: dict | None = None, exclude_issues: bool = True,
                 max_dqr: float | None = None) -> dict:
    P = build_problem(m, baskets, scenario, params, exclude_issues, max_dqr)
    base = summarize(P, P.x0, tag={"objective": "original"})
    env_only = solve(P, "climate", nutrition=False)
    if env_only is not None:
        env_only["objective"] = "climate only (no nutrition constraints)"
    clim = solve(P, "climate")
    sweep = pareto_sweep(P)
    swaps = pd.concat([swap_front(P, eps_dev=None).assign(dev_budget="none"),
                       swap_front(P, eps_dev=0.10).assign(dev_budget="0.10")], ignore_index=True)
    conflict_w = env_conflict(P, "water")
    conflict_l = env_conflict(P, "land")
    nr = n_random if n_random is not None else P.params["n_random"]
    rs = random_swaps(P, nr, seed)
    rf = random_feasible(P, 100, seed + 1)
    return {"problem": P, "original": base, "env_only": env_only, "climate_opt": clim, "sweep": sweep,
            "swaps": swaps, "conflict_water": conflict_w, "conflict_land": conflict_l,
            "random_swaps": rs, "random_feasible": rf}
