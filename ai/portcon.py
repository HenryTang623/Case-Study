"""
portcon.py — Portfolio Construction methods
===========================================
Seven methods spanning the four families of the agentic-SAA paper:

  Heuristic         : equal_weight, inverse_vol
  Risk-structured   : min_variance, risk_parity (ERC)
  Return-optimized  : max_sharpe, black_litterman
  Non-traditional   : cvar_min (Rockafellar-Uryasev LP, 95%)

All methods share the signature

    method(mu, cov, returns, cons) -> pd.Series(weights)

even when they ignore some inputs, so cio.py can loop uniformly:
  * mu      : expected excess returns (annualized)        [Series]
  * cov     : annualized covariance                        [DataFrame]
  * returns : historical EXCESS return scenarios (monthly) [DataFrame] (CVaR)
  * cons    : Constraints dataclass (long-only, caps)

Constraints are applied centrally so every method respects the same IPS.
"""

from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np
import pandas as pd
import cvxpy as cp
from scipy.optimize import minimize

from cma import EQUITY, FIXED_INCOME, RISK_ASSETS, MONTHS


@dataclass
class Constraints:
    """Lightweight IPS: long-only, fully invested, category + per-asset caps."""
    long_only: bool = True
    asset_cap: float = 0.35
    equity_cap: float = 0.65
    fi_cap: float = 0.80
    min_sharpe_floor: float | None = None      # reserved for extensions
    assets: list = field(default_factory=lambda: list(RISK_ASSETS))

    def cvxpy_constraints(self, w):
        c = [cp.sum(w) == 1]
        if self.long_only:
            c += [w >= 0]
        c += [w <= self.asset_cap]
        eq_idx = [self.assets.index(a) for a in EQUITY]
        fi_idx = [self.assets.index(a) for a in FIXED_INCOME]
        c += [cp.sum(w[eq_idx]) <= self.equity_cap]
        c += [cp.sum(w[fi_idx]) <= self.fi_cap]
        return c

    def scipy_bounds_cons(self, n):
        bounds = [(0, self.asset_cap) if self.long_only else (-self.asset_cap, self.asset_cap)
                  for _ in range(n)]
        eq_idx = [self.assets.index(a) for a in EQUITY]
        fi_idx = [self.assets.index(a) for a in FIXED_INCOME]
        cons = [
            {"type": "eq", "fun": lambda w: w.sum() - 1},
            {"type": "ineq", "fun": lambda w: self.equity_cap - w[eq_idx].sum()},
            {"type": "ineq", "fun": lambda w: self.fi_cap - w[fi_idx].sum()},
        ]
        return bounds, cons


def _series(w, assets):
    w = np.asarray(w).flatten()
    w[np.abs(w) < 1e-6] = 0.0
    return pd.Series(w / w.sum(), index=assets)


# ----------------------------------------------------------------------------
# Heuristic
# ----------------------------------------------------------------------------
def equal_weight(mu, cov, returns, cons):
    n = len(cons.assets)
    return _series(np.ones(n) / n, cons.assets)


def inverse_vol(mu, cov, returns, cons):
    vol = np.sqrt(np.diag(cov.values))
    w = (1 / vol)
    return _series(w, cons.assets)   # note: caps enforced via projection below


# ----------------------------------------------------------------------------
# Risk-structured
# ----------------------------------------------------------------------------
def min_variance(mu, cov, returns, cons):
    n = len(cons.assets)
    w = cp.Variable(n)
    prob = cp.Problem(cp.Minimize(cp.quad_form(w, cov.values)), cons.cvxpy_constraints(w))
    prob.solve()
    return _series(w.value, cons.assets)


def risk_parity(mu, cov, returns, cons):
    """Equal risk contribution via SLSQP, then cap-projected."""
    n = len(cons.assets)
    S = cov.values

    def obj(w):
        port_var = w @ S @ w
        mrc = S @ w
        rc = w * mrc
        target = port_var / n
        return np.sum((rc - target) ** 2)

    bounds, scons = cons.scipy_bounds_cons(n)
    res = minimize(obj, np.ones(n) / n, method="SLSQP", bounds=bounds, constraints=scons,
                   options={"maxiter": 1000, "ftol": 1e-12})
    return _series(res.x, cons.assets)


# ----------------------------------------------------------------------------
# Return-optimized
# ----------------------------------------------------------------------------
def max_sharpe(mu, cov, returns, cons):
    """Maximize mu'w / sqrt(w'Sw) via the convex long-only reformulation."""
    n = len(cons.assets)
    w = cp.Variable(n)
    # maximize return for unit-ish risk; sweep is overkill for 6 assets:
    # solve max mu'w  s.t. w'Sw <= t  over a small grid, keep best Sharpe.
    best_w, best_sr = None, -np.inf
    for t in np.linspace(0.0005, 0.06, 40):     # variance budget (annual)
        prob = cp.Problem(cp.Maximize(mu.values @ w),
                          cons.cvxpy_constraints(w) + [cp.quad_form(w, cov.values) <= t])
        prob.solve()
        if w.value is None:
            continue
        wv = np.asarray(w.value).flatten()
        sr = (mu.values @ wv) / np.sqrt(wv @ cov.values @ wv)
        if sr > best_sr:
            best_sr, best_w = sr, wv
    return _series(best_w, cons.assets)


def black_litterman(mu, cov, returns, cons, tau=0.05):
    """
    BL with the equilibrium prior already in `mu` is redundant; here we treat
    the blended mu as the BL posterior input and simply run a constrained
    mean-variance (max utility) optimization with moderate risk aversion.
    """
    n = len(cons.assets)
    delta = 3.0
    w = cp.Variable(n)
    util = mu.values @ w - 0.5 * delta * cp.quad_form(w, cov.values)
    prob = cp.Problem(cp.Maximize(util), cons.cvxpy_constraints(w))
    prob.solve()
    return _series(w.value, cons.assets)


# ----------------------------------------------------------------------------
# Non-traditional: CVaR minimization (Rockafellar-Uryasev, 95%)
# ----------------------------------------------------------------------------
def cvar_min(mu, cov, returns, cons, alpha=0.95):
    """
    Minimize CVaR_alpha of portfolio LOSSES using empirical scenarios.
        min  var + 1/((1-a)T) * sum z_t
        s.t. z_t >= -w'r_t - var,  z_t >= 0,  + IPS constraints
    `returns` are monthly excess return scenarios (rows = months).
    """
    R = returns[cons.assets].dropna().values
    T, n = R.shape
    w = cp.Variable(n)
    var = cp.Variable()
    z = cp.Variable(T)
    losses = -(R @ w)
    objective = var + (1.0 / ((1 - alpha) * T)) * cp.sum(z)
    constraints = cons.cvxpy_constraints(w) + [z >= 0, z >= losses - var]
    cp.Problem(cp.Minimize(objective), constraints).solve()
    return _series(w.value, cons.assets)


METHODS = {
    "Equal Weight": equal_weight,
    "Inverse Vol": inverse_vol,
    "Min Variance": min_variance,
    "Risk Parity": risk_parity,
    "Max Sharpe": max_sharpe,
    "Black-Litterman": black_litterman,
    "CVaR-Min": cvar_min,
}

METHOD_FAMILY = {
    "Equal Weight": "Heuristic", "Inverse Vol": "Heuristic",
    "Min Variance": "Risk-structured", "Risk Parity": "Risk-structured",
    "Max Sharpe": "Return-optimized", "Black-Litterman": "Return-optimized",
    "CVaR-Min": "Non-traditional",
}


def _project_caps(w: pd.Series, cons: Constraints, iters=50) -> pd.Series:
    """Iterative projection for closed-form heuristics (inverse-vol) onto caps."""
    w = w.clip(lower=0).copy()
    for _ in range(iters):
        w = w / w.sum()
        over = w > cons.asset_cap
        if not over.any() and w[EQUITY].sum() <= cons.equity_cap + 1e-9 \
           and w[FIXED_INCOME].sum() <= cons.fi_cap + 1e-9:
            break
        w[over] = cons.asset_cap
        free = ~over
        deficit = 1 - w[over].sum() - w[free].sum()
        if free.any():
            w[free] += deficit * w[free] / w[free].sum()
    return w / w.sum()


def construct_all(mu, cov, returns, cons: Constraints) -> pd.DataFrame:
    """Run every method; return a (method x asset) weight matrix."""
    out = {}
    for name, fn in METHODS.items():
        w = fn(mu, cov, returns, cons)
        if name in ("Inverse Vol",):           # heuristic needs cap projection
            w = _project_caps(w, cons)
        out[name] = w.reindex(cons.assets)
    return pd.DataFrame(out).T
