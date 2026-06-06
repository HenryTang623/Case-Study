import pandas as pd
import numpy as np
import statsmodels.api as sm
from sklearn.covariance import LedoitWolf

EQUITY = ["US equity", "Developed ex-US Equity", "Emerging Markets Equity"]
FIXED_INCOME = ["US Treasuries", "US IG Credit", "US HY Credit"]
RISK_ASSETS = EQUITY + FIXED_INCOME
CASH = "Cash"
SEED = 623


# ----------------------------------------------------------------------------
# Loading & preprocessing
# ----------------------------------------------------------------------------
def load_returns(path):
    return pd.read_excel(path, sheet_name="Monthly Return", index_col=0, parse_dates=True)


def to_excess(df):
    """Excess-over-cash returns for the 6 risk assets."""
    return df[RISK_ASSETS].sub(df[CASH], axis=0)

def backfill_hy(excess_df, seed=SEED):
    """
    HY_t ~ 1 + IG_t + USeq_t (fit on post-2007)
    backfill = fitted + residual bootstrapped from in-sample residuals.
    """
    # aggregate df
    factors = ["US IG Credit", "US equity"]
    hy = "US HY Credit"
    reg_df = excess_df.copy()

    # masks
    intact = reg_df[hy].notna()
    miss = ~intact

    # regression matrices
    X = sm.add_constant(reg_df.loc[intact, factors])
    y = reg_df.loc[intact, hy]

    # fit
    model = sm.OLS(y, X).fit()
    beta = model.params
    resid = model.resid.values
    r2 = model.rsquared

    # predict
    Xm = sm.add_constant(reg_df.loc[miss, factors])
    fitted = model.predict(Xm)

    # bootstrap residuals
    rng = np.random.default_rng(seed)
    resid_b = rng.choice(resid, size=miss.sum(), replace=True)

    # backfill HY
    reg_df.loc[miss, hy] = fitted + resid_b

    diag = {
        "alpha": beta["const"],
        "beta_IG": beta["US IG Credit"],
        "beta_USeq": beta["US equity"],
        "r2": r2,
    }

    return reg_df, diag

# ----------------------------------------------------------------------------
# Expected returns: 3 methods + clipped blend
# ----------------------------------------------------------------------------
def _historical_mean(excess_df):
    return excess_df.mean() * 12


def _shrunk_mean(excess_df):
    """James-Stein shrinkage of annualized means toward the grand mean."""
    mu = excess_df.mean() * 12
    n, k = len(excess_df), excess_df.shape[1]
    grand = mu.mean()
    # per-asset sampling variance of the annualized mean
    var_mu = (excess_df.var() * 12) / n
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