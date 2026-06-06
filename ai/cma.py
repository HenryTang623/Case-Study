"""
cma.py — Capital Market Assumptions
====================================
Loads the data, converts to excess-over-cash returns, backfills the short
High-Yield history, and produces the two CMA inputs every portfolio
construction method needs:

    - mu  : expected EXCESS returns (annualized), as a rules-based blend of
            historical / shrunk / equilibrium estimates, clipped to the
            method range (the deterministic analog of the paper's CMA-judge)
    - cov : Ledoit-Wolf shrunk covariance (annualized)

Design choices (see write-up):
  * Work in excess-over-cash space; Cash is the IPS risk-free anchor.
  * HY (starts 2007-05) is backfilled by a 3-factor regression on
    IG + US equity + Treasuries over the overlap, with bootstrapped
    residuals added back so the reconstructed series is not artificially
    smooth.  Seeded for reproducibility.
  * Expected returns: equilibrium (reverse-optimized) base, tilted toward a
    James-Stein shrunk historical mean, clipped within the per-asset method
    range.  Historical means are treated as one noisy input, not truth.
  * Covariance: Ledoit-Wolf shrinkage stabilizes the near-collinear equity
    block for inversion-based optimizers.
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.covariance import LedoitWolf

EQUITY = ["US equity", "Developed ex-US Equity", "Emerging Markets Equity"]
FIXED_INCOME = ["US Treasuries", "US IG Credit", "US HY Credit"]
RISK_ASSETS = EQUITY + FIXED_INCOME          # 6 optimizable assets
CASH = "Cash"
MONTHS = 12
SEED = 42


# ----------------------------------------------------------------------------
# Loading & excess returns
# ----------------------------------------------------------------------------
def load_returns(path: str) -> pd.DataFrame:
    """Load monthly simple returns, parse the date index. Cash kept as a column."""
    df = pd.read_excel(path, sheet_name="Monthly Return")
    df = df.rename(columns={df.columns[0]: "date"}).set_index("date")
    df.index = pd.to_datetime(df.index)
    return df[RISK_ASSETS + [CASH]].sort_index()


def to_excess(df: pd.DataFrame) -> pd.DataFrame:
    """Excess-over-cash returns for the 6 risk assets (Cash dropped)."""
    return df[RISK_ASSETS].sub(df[CASH], axis=0)


# ----------------------------------------------------------------------------
# HY backfill: 3-factor regression + bootstrapped residuals
# ----------------------------------------------------------------------------
def backfill_hy(excess: pd.DataFrame, seed: int = SEED) -> tuple[pd.DataFrame, dict]:
    """
    Reconstruct pre-2007 High-Yield excess returns.

    HY_ex ~ a + b1*IG_ex + b2*USeq_ex + b3*UST_ex + e  (fit on overlap)
    backfill = fitted + residual bootstrapped from in-sample residuals.

    Returns the completed excess frame and a diagnostics dict (for the report).
    Note: regression used for PREDICTION, not inference — collinearity among
    IG/UST inflates individual coefficient SEs but does not bias the fitted
    series used to backfill.
    """
    factors = ["US IG Credit", "US equity", "US Treasuries"]
    hy = "US HY Credit"
    out = excess.copy()

    mask = out[hy].notna()
    X = out.loc[mask, factors].values
    y = out.loc[mask, hy].values
    Xc = np.column_stack([np.ones(len(X)), X])

    beta, *_ = np.linalg.lstsq(Xc, y, rcond=None)
    resid = y - Xc @ beta
    r2 = 1 - resid.var() / y.var()

    miss = out[hy].isna()
    if miss.any():
        Xm = np.column_stack([np.ones(miss.sum()), out.loc[miss, factors].values])
        rng = np.random.default_rng(seed)
        eps = rng.choice(resid, size=miss.sum(), replace=True)
        out.loc[miss, hy] = Xm @ beta + eps

    diag = {
        "alpha": beta[0],
        "beta_IG": beta[1], "beta_USeq": beta[2], "beta_UST": beta[3],
        "r2": r2, "resid_vol_ann": resid.std() * np.sqrt(MONTHS),
        "n_overlap": int(mask.sum()), "n_backfilled": int(miss.sum()),
        "hy_mean_overlap_ann": y.mean() * MONTHS,
        "hy_mean_backfill_ann": (Xm @ beta + eps).mean() * MONTHS if miss.any() else np.nan,
    }
    return out, diag


# ----------------------------------------------------------------------------
# Expected returns: 3 methods + clipped blend
# ----------------------------------------------------------------------------
def _historical_mean(excess: pd.DataFrame) -> pd.Series:
    return excess.mean() * MONTHS


def _shrunk_mean(excess: pd.DataFrame) -> pd.Series:
    """James-Stein shrinkage of annualized means toward the grand mean."""
    mu = excess.mean() * MONTHS
    n, k = len(excess), excess.shape[1]
    grand = mu.mean()
    # per-asset sampling variance of the annualized mean
    var_mu = (excess.var() * MONTHS) / n
    ss = ((mu - grand) ** 2).sum()
    shrink = ((k - 2) * var_mu.mean() / ss) if ss > 0 else 1.0
    shrink = float(np.clip(shrink, 0.0, 1.0))
    return grand + (1 - shrink) * (mu - grand)


def _equilibrium(cov: pd.DataFrame, bench_w: pd.Series,
                 bench_sharpe: float = 0.35) -> pd.Series:
    """
    Reverse-optimized (Black-Litterman prior) excess returns:
        Pi = delta * cov @ w_bench
    delta calibrated so the benchmark hits a target excess Sharpe.
    """
    w = bench_w.reindex(cov.index).values
    bench_vol = float(np.sqrt(w @ cov.values @ w))
    delta = bench_sharpe / bench_vol            # implied risk aversion
    pi = delta * cov.values @ w
    return pd.Series(pi, index=cov.index)


def expected_returns(excess: pd.DataFrame, cov: pd.DataFrame,
                     bench_w: pd.Series, w_eq: float = 0.7) -> pd.DataFrame:
    """
    Blend = w_eq * equilibrium + (1-w_eq) * shrunk, clipped to the per-asset
    [min, max] of the candidate methods (paper's CMA-judge hard constraint).
    Returns a frame with all candidates + the final blend (the Exhibit-8 analog).
    """
    hist = _historical_mean(excess)
    shrunk = _shrunk_mean(excess)
    equil = _equilibrium(cov, bench_w)

    raw = w_eq * equil + (1 - w_eq) * shrunk
    lo = pd.concat([hist, shrunk, equil], axis=1).min(axis=1)
    hi = pd.concat([hist, shrunk, equil], axis=1).max(axis=1)
    blend = raw.clip(lower=lo, upper=hi)

    return pd.DataFrame({
        "Historical": hist, "Shrunk": shrunk,
        "Equilibrium": equil, "Blend": blend,
    })


# ----------------------------------------------------------------------------
# Covariance
# ----------------------------------------------------------------------------
def covariance(excess: pd.DataFrame, common_start: str | None = None) -> pd.DataFrame:
    """
    Ledoit-Wolf shrunk covariance (annualized). If common_start is given,
    estimate on that window so all assets share one sample (internal
    consistency for the matrix).
    """
    data = excess.loc[common_start:] if common_start else excess
    data = data.dropna()
    lw = LedoitWolf().fit(data.values)
    cov = pd.DataFrame(lw.covariance_ * MONTHS, index=data.columns, columns=data.columns)
    return cov


def benchmark_weights() -> pd.Series:
    """60/40: 60% split equally across 3 equity sleeves, 40% across 3 FI sleeves."""
    w = {a: 0.60 / len(EQUITY) for a in EQUITY}
    w.update({a: 0.40 / len(FIXED_INCOME) for a in FIXED_INCOME})
    return pd.Series(w)[RISK_ASSETS]


def build_cmas(path: str, seed: int = SEED):
    """End-to-end CMA build. Returns (mu_blend, cov, excess, diagnostics, mu_table)."""
    raw = load_returns(path)
    excess = to_excess(raw)
    excess, hy_diag = backfill_hy(excess, seed=seed)
    cov = covariance(excess)                      # full sample after backfill
    bench = benchmark_weights()
    mu_table = expected_returns(excess, cov, bench)
    return mu_table["Blend"], cov, excess, hy_diag, mu_table, raw
