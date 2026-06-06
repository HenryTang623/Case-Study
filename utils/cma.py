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
# Loading & Preprocessing
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
# Expected Returns
# ----------------------------------------------------------------------------
def _historical_mean(excess_df):
    return excess_df.mean() * 12

# TODO: add other methods to estimate expected returns

def expected_returns(excess_df):
    hist = _historical_mean(excess_df)
    # TODO: other methods

    return pd.DataFrame({
        "Historical": hist
    })

# ----------------------------------------------------------------------------
# Covariance
# ----------------------------------------------------------------------------
def covariance(excess_df):
    """
    Ledoit-Wolf shrunk covariance (annualized).
    """
    lw = LedoitWolf().fit(excess_df.values)
    lw_df = pd.DataFrame(lw.covariance_ * 12, index=excess_df.columns, columns=excess_df.columns)
    return lw_df
