"""
cio.py — Evaluation, scoring, and ensemble (the CIO layer)
==========================================================
Scores every candidate portfolio, ranks them on a transparent composite,
and combines them into a final recommendation via an inverse-tracking-error
ensemble (the paper's March-2026 choice), with simple-average and best-single
shown as comparisons.

Metrics per portfolio (all in excess-over-cash space unless noted):
  ann_return, ann_vol, sharpe, max_drawdown, effective_N (Meucci),
  tracking_error vs 60/40, concentration (HHI).
"""

from __future__ import annotations
import numpy as np
import pandas as pd

from ai.cma import RISK_ASSETS, MONTHS, benchmark_weights


# ----------------------------------------------------------------------------
# Portfolio metrics
# ----------------------------------------------------------------------------
def port_series(weights: pd.Series, excess: pd.DataFrame) -> pd.Series:
    return excess[weights.index].dropna() @ weights


def max_drawdown(monthly: pd.Series) -> float:
    cum = (1 + monthly).cumprod()
    peak = cum.cummax()
    return float((cum / peak - 1).min())


def effective_n(weights: pd.Series, cov: pd.DataFrame) -> float:
    """Meucci effective number of bets via PCA of the risk-weighted portfolio."""
    w = weights.reindex(cov.index).values
    S = cov.values
    evals, evecs = np.linalg.eigh(S)
    # marginal contributions in principal directions
    p = (evecs.T @ w) ** 2 * evals
    p = p / p.sum()
    p = p[p > 0]
    return float(np.exp(-np.sum(p * np.log(p))))   # entropy-based ENB


def tracking_error(weights: pd.Series, cov: pd.DataFrame, bench: pd.Series) -> float:
    d = (weights.reindex(cov.index) - bench.reindex(cov.index)).values
    return float(np.sqrt(d @ cov.values @ d))


def hhi(weights: pd.Series) -> float:
    return float((weights ** 2).sum())


def score_portfolio(weights: pd.Series, excess: pd.DataFrame,
                    cov: pd.DataFrame, bench: pd.Series) -> dict:
    ps = port_series(weights, excess)
    ann_ret = ps.mean() * MONTHS
    ann_vol = ps.std() * np.sqrt(MONTHS)
    return {
        "ann_return": ann_ret,
        "ann_vol": ann_vol,
        "sharpe": ann_ret / ann_vol if ann_vol > 0 else np.nan,
        "max_drawdown": max_drawdown(ps),
        "effective_N": effective_n(weights, cov),
        "tracking_error": tracking_error(weights, cov, bench),
        "concentration_HHI": hhi(weights),
    }


def score_all(weights_df: pd.DataFrame, excess, cov, bench) -> pd.DataFrame:
    rows = {name: score_portfolio(weights_df.loc[name], excess, cov, bench)
            for name in weights_df.index}
    return pd.DataFrame(rows).T


# ----------------------------------------------------------------------------
# Composite ranking
# ----------------------------------------------------------------------------
def composite_score(scores: pd.DataFrame,
                    w_sharpe=0.35, w_dd=0.30, w_enb=0.20, w_te=0.15) -> pd.Series:
    """
    Higher is better. Normalize each metric to [0,1]:
      + sharpe, + effective_N, + (less negative) drawdown, - tracking error.
    Weights documented; downside (sharpe+drawdown) gets the majority, matching
    an institutional downside-aware objective.
    """
    def mm(s, higher_better=True):
        rng = s.max() - s.min()
        if rng == 0:
            return pd.Series(0.5, index=s.index)
        z = (s - s.min()) / rng
        return z if higher_better else 1 - z

    comp = (w_sharpe * mm(scores["sharpe"])
            + w_dd * mm(scores["max_drawdown"], higher_better=True)   # -0.2 > -0.4
            + w_enb * mm(scores["effective_N"])
            + w_te * mm(scores["tracking_error"], higher_better=False))
    return comp.sort_values(ascending=False)


# ----------------------------------------------------------------------------
# Ensembles
# ----------------------------------------------------------------------------
def ensemble_inverse_te(weights_df, cov, bench) -> pd.Series:
    """Weight each method by 1/tracking-error to the 60/40 centroid."""
    te = {n: tracking_error(weights_df.loc[n], cov, bench) for n in weights_df.index}
    inv = pd.Series({n: 1 / max(v, 1e-6) for n, v in te.items()})
    mw = inv / inv.sum()
    return (weights_df.T @ mw).reindex(weights_df.columns)


def ensemble_simple_avg(weights_df) -> pd.Series:
    return weights_df.mean(axis=0)


def ensemble_trimmed(weights_df) -> pd.Series:
    """Per-asset trimmed mean (drop min & max method), renormalized."""
    trimmed = weights_df.apply(lambda col: col.sort_values()[1:-1].mean(), axis=0)
    return trimmed / trimmed.sum()


# ----------------------------------------------------------------------------
# Backtest helpers
# ----------------------------------------------------------------------------
def backtest(weights: pd.Series, excess: pd.DataFrame, cash: pd.Series) -> dict:
    """Static-weight backtest. Total return = excess + cash (held weights)."""
    ps_ex = port_series(weights, excess)
    total = ps_ex + cash.reindex(ps_ex.index)
    return {
        "ann_excess_return": ps_ex.mean() * MONTHS,
        "ann_total_return": total.mean() * MONTHS,
        "ann_vol": ps_ex.std() * np.sqrt(MONTHS),
        "sharpe": (ps_ex.mean() * MONTHS) / (ps_ex.std() * np.sqrt(MONTHS)),
        "max_drawdown": max_drawdown(total),
    }


def subperiod_drawdowns(weights: pd.Series, excess, cash, periods: dict) -> pd.DataFrame:
    rows = {}
    total = port_series(weights, excess) + cash
    for label, (s, e) in periods.items():
        seg = total.loc[s:e]
        rows[label] = {"max_drawdown": max_drawdown(seg),
                       "cum_return": (1 + seg).prod() - 1}
    return pd.DataFrame(rows).T
