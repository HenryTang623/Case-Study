import numpy as np
import pandas as pd
import cvxpy as cp
from utils.cma import EQUITY, FIXED_INCOME, RISK_ASSETS

METHOD_FAMILY = {
    "Equal Weight": "Heuristic", 
    "Inverse Vol": "Heuristic",
    "Min Variance": "Risk-structured", 
    "Risk Parity": "Risk-structured",
    "Max Sharpe": "Return-optimized",
    "CVaR-Min": "Non-traditional",
}

class Constraints:
    """IPS: long-only, fully invested, category + per-asset caps, sharpe >= 0."""
    def __init__(self, asset_cap=0.35, equity_cap=0.65, fi_cap=0.80, sharpe_floor=0):
        self.asset_cap = asset_cap
        self.equity_cap = equity_cap
        self.fi_cap = fi_cap
        self.sharpe_floor = sharpe_floor
        self.assets = list(RISK_ASSETS)

    def get_constraints(self, w):
        c = [cp.sum(w) == 1]
        c += [w >= 0]
        c += [w <= self.asset_cap]
        eq_idx = [self.assets.index(a) for a in EQUITY]
        fi_idx = [self.assets.index(a) for a in FIXED_INCOME]
        c += [cp.sum(w[eq_idx]) <= self.equity_cap]
        c += [cp.sum(w[fi_idx]) <= self.fi_cap]
        return c

def normalize_series(w, assets):
    w = np.asarray(w).flatten()
    w[np.abs(w) < 1e-6] = 0.0
    return pd.Series(w / w.sum(), index=assets)

def exert_caps(w_raw, cons):
    """exert caps on heuristic methods"""
    n = len(w_raw)
    w = cp.Variable(n)
    obj = cp.sum_squares(w - w_raw)
    prob = cp.Problem(cp.Minimize(obj), cons.get_constraints(w))
    prob.solve()
    return normalize_series(w.value, cons.assets)

# ----------------------------------------------------------------------------
# Closed-form heuristics
# ----------------------------------------------------------------------------
def equal_weight(cons):
    n = len(cons.assets)
    w_raw = normalize_series(np.ones(n), cons.assets)
    return exert_caps(w_raw, cons)

def inverse_vol(cov, cons):
    vol = np.sqrt(np.diag(cov.values))
    w = 1 / vol
    w_raw = normalize_series(w, cons.assets) 
    return exert_caps(w_raw, cons)

# ----------------------------------------------------------------------------
# Risk-structured
# ----------------------------------------------------------------------------
def min_variance(cov, cons):
    n = len(cons.assets)
    w = cp.Variable(n)
    prob = cp.Problem(cp.Minimize(cp.quad_form(w, cov.values)), cons.get_constraints(w))
    prob.solve()
    return normalize_series(w.value, cons.assets)

def risk_parity(cov, cons):
    """
    Minimize the convex Spinu's objective function 

    w.T @ cov @ w - sum(ln(w_i))

    whose F.O.C. enforces equal risk contribution.
    """
    n = len(cons.assets)
    S = cov.values
    w = cp.Variable(n)
    obj = cp.quad_form(w, S) - cp.sum(cp.log(w))
    prob = cp.Problem(cp.Minimize(obj), cons.get_constraints(w))
    prob.solve()
    return normalize_series(w.value, cons.assets)

# ----------------------------------------------------------------------------
# Return-optimized
# ----------------------------------------------------------------------------
def max_sharpe(mu, cov, cons):
    n = len(cons.assets)
    S = cov.values
    w = cp.Variable(n)
    obj = mu.values @ w
    best_w, best_sr = None, -np.inf

    # sweep through ann risk budget to calculate efficient frontier and find tangency portfolio
    for t in np.linspace(0.0005, 0.06, 40):
        constr = cons.get_constraints(w) + [cp.quad_form(w, S) <= t]
        prob = cp.Problem(cp.Maximize(obj), constr)
        prob.solve()
        wv = w.value
        if wv is None:
            continue
        sr = (mu.values @ wv) / np.sqrt(wv @ S @ wv)
        if sr > best_sr:
            best_sr, best_w = sr, wv

    return normalize_series(best_w, cons.assets)

# ----------------------------------------------------------------------------
# Non-traditional: CVaR minimization (Rockafellar-Uryasev, 95%)
# ----------------------------------------------------------------------------
def cvar_min(returns, cons, alpha=0.95):
    """
    Minimize CVaR_alpha of portfolio LOSSES using empirical scenarios.
        min  var + 1/((1-a)T) * sum z_t
        s.t. z_t >= -w'r_t - var,  z_t >= 0,  + IPS constraints
    `returns` are monthly excess return scenarios (rows = months).
    """
    R = returns.values
    T, n = R.shape
    w = cp.Variable(n)
    var = cp.Variable()
    z = cp.Variable(T)
    losses = -(R @ w)
    obj = var + (1.0 / ((1 - alpha) * T)) * cp.sum(z)
    constr = cons.get_constraints(w) + [z >= 0, z >= losses - var]
    prob = cp.Problem(cp.Minimize(obj), constr)
    prob.solve()
    return normalize_series(w.value, cons.assets)

def construct_all(mu, cov, returns, cons):
    out = {
        "Equal Weight":    equal_weight(cons),
        "Inverse Vol":     inverse_vol(cov, cons),
        "Min Variance":    min_variance(cov, cons),
        "Risk Parity":     risk_parity(cov, cons),
        "Max Sharpe":      max_sharpe(mu, cov, cons),
        "CVaR-Min":        cvar_min(returns, cons),
    }
    return pd.DataFrame(out).T
