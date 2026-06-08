import pandas as pd
import numpy as np
from utils import cma, portcon, cio

pd.set_option("display.width", 200)
pd.set_option("display.float_format", lambda x: f"{x:,.2f}")

DATA = "spd_case_study_data_2026.xlsx"

def pct(df):
    return (df * 100).round(2)

def main():
    # CMA layer
    raw = cma.load_returns(DATA)
    excess = cma.to_excess(raw)
    excess, _ = cma.backfill_hy(excess)
    mu, cov = cma.build_cmas(excess)
    bench = cma.benchmark_weights()

    print("Expected Excess Returns (%)")
    print(pct(mu))

    print("\nCorrelation")
    d = cov.values.diagonal() ** 0.5
    scaler = np.outer(d,d)
    print(cov / scaler)

    # Portfolio Construction layer
    cons = portcon.Constraints()
    weights = portcon.construct_all(mu, cov, excess, cons)

    print("\nCandidate portfolio weights (%)")
    print(pct(weights))

    # CIO layer
    total_returns = excess.add(raw[cma.CASH], axis=0)
    portfolios = {name: cio.Portfolio(weights.loc[name], excess, total_returns) for name in weights.index}

    # Metric-weighting scenarios
    weight_cases = {
        "Risk-Adj. Return Focused": dict(w_sharpe=1.00, w_dd=0.00, w_cvar=0.00, w_conc=0.00, w_te=0.00, w_turnover=0.00),
        "Down. Protection Focused":  dict(w_sharpe=0.00, w_dd=0.50, w_cvar=0.50, w_conc=0.00, w_te=0.00, w_turnover=0.00),
        "Diversification Focused": dict(w_sharpe=0.00, w_dd=0.00, w_cvar=0.00, w_conc=1.00, w_te=0.00, w_turnover=0.00),
        "Governance Focused": dict(w_sharpe=0.00, w_dd=0.00, w_cvar=0.00, w_conc=0.00, w_te=0.50, w_turnover=0.50),
        "Balanced": dict(w_sharpe=0.25, w_dd=0.125, w_cvar=0.125, w_conc=0.25, w_te=0.125, w_turnover=0.125),
    }

    # compute raw metrics for every portfolio (composite scores added per-case below)
    _ = {name: portfolio.calculate_metrics(cov, bench) for name, portfolio in portfolios.items()}

    # composite score under each weighting case
    scores_out = {}

    for case_name, wts in weight_cases.items():
        scores_out[case_name] = cio.calculate_composite_scores(portfolios, **wts)

    scores_out = pd.DataFrame(scores_out).sort_values("Balanced", ascending=False)

    print("\nScorecard (%)")
    print(pct(scores_out))

    # Per-case top-3 ensembles
    ens_rows = {}
    for case_name, wts in weight_cases.items():
        cio.calculate_composite_scores(portfolios, **wts)
        ens_rows[case_name] = cio.ensemble_top3(portfolios)

    ens_df = pd.DataFrame(ens_rows).T
    ens_df.loc["60/40 Benchmark"] = bench

    print("\nEnsemble-portfolio allocations by case (%)")
    print(pct(ens_df))

    # Backtests of ensembles vs benchmark
    print("\nFull-sample backtest vs 60/40")
    bt_rows = {}
    for name in ens_df.index:
        p = cio.Portfolio(ens_df.loc[name], excess, total_returns)
        m = p.calculate_metrics(cov, bench)
        bt_rows[name] = {
            "ann_return": m["ann_return"] * 100,
            "ann_vol": m["ann_vol"] * 100,
            "sharpe": m["sharpe"],
            "max_dd": m["max_dd"] * 100,
        }
    print(pd.DataFrame(bt_rows).T.round(3))

if __name__ == "__main__":
    main()
