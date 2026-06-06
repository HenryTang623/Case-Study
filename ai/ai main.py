"""
main.py — run the full agentic-inspired SAA pipeline end to end
================================================================
    cma.py    -> CMAs (expected excess returns + covariance, HY backfilled)
    portcon.py-> 7 candidate portfolios under one IPS
    cio.py    -> score, rank, ensemble, backtest

Usage:  python main.py [path_to_xlsx]
"""

import numpy as np
import pandas as pd
import cma, portcon as pc, cio


pd.set_option("display.width", 140)
pd.set_option("display.float_format", lambda x: f"{x:,.4f}")

DATA = "spd_case_study_data_2026.xlsx"


def pct(df):
    return (df * 100).round(2)


def main():
    # ---- CMAs ----------------------------------------------------------
    mu, cov, excess, hy_diag, mu_table, raw = cma.build_cmas(DATA)
    cash = raw[cma.CASH]
    bench = cma.benchmark_weights()

    print("=" * 78)
    print("EXHIBIT A — HY backfill regression (3-factor: IG + US equity + UST)")
    print("=" * 78)
    for k, v in hy_diag.items():
        print(f"  {k:24s}: {v:,.4f}" if isinstance(v, float) else f"  {k:24s}: {v}")

    print("\n" + "=" * 78)
    print("EXHIBIT B — Expected excess returns: candidate methods & blend (%, ann.)")
    print("=" * 78)
    print(pct(mu_table))

    print("\n" + "=" * 78)
    print("EXHIBIT C — Ledoit-Wolf covariance -> correlation")
    print("=" * 78)
    d = np.sqrt(np.diag(cov.values))
    corr = pd.DataFrame(cov.values / np.outer(d, d), index=cov.index, columns=cov.index)
    print(corr.round(2))

    # ---- Portfolio construction ---------------------------------------
    cons = pc.Constraints()
    weights = pc.construct_all(mu, cov, excess, cons)

    print("\n" + "=" * 78)
    print("EXHIBIT D — Candidate portfolio weights (%)")
    print("=" * 78)
    print(pct(weights))

    # ---- Scoring & ranking --------------------------------------------
    scores = cio.score_all(weights, excess, cov, bench)
    comp = cio.composite_score(scores)
    scores_out = scores.copy()
    for c in ["ann_return", "ann_vol", "max_drawdown", "tracking_error"]:
        scores_out[c] = scores_out[c] * 100
    scores_out["composite"] = comp
    scores_out = scores_out.sort_values("composite", ascending=False)

    print("\n" + "=" * 78)
    print("EXHIBIT E — Scorecard (returns/vol/DD/TE in %, sorted by composite)")
    print("=" * 78)
    print(scores_out.round(3))

    # ---- Ensembles -----------------------------------------------------
    ens_te = cio.ensemble_inverse_te(weights, cov, bench)
    ens_avg = cio.ensemble_simple_avg(weights)
    ens_trim = cio.ensemble_trimmed(weights)
    best_single = comp.index[0]

    ens_df = pd.DataFrame({
        "Inverse-TE (REC)": ens_te,
        "Simple Avg": ens_avg,
        "Trimmed Mean": ens_trim,
        f"Best Single ({best_single})": weights.loc[best_single],
        "60/40 Benchmark": bench,
    }).T

    print("\n" + "=" * 78)
    print("EXHIBIT F — Ensemble & comparison weights (%)")
    print("=" * 78)
    print(pct(ens_df))

    # ---- Backtests -----------------------------------------------------
    print("\n" + "=" * 78)
    print("EXHIBIT G — Full-sample backtest vs 60/40 (returns/DD in %)")
    print("=" * 78)
    bt_rows = {}
    for name in ens_df.index:
        bt = cio.backtest(ens_df.loc[name], excess, cash)
        bt_rows[name] = {k: (v * 100 if k != "sharpe" else v) for k, v in bt.items()}
    print(pd.DataFrame(bt_rows).T.round(3))

    # ---- Sub-period stress --------------------------------------------
    periods = {
        "GFC 2007-09 to 2009-03": ("2007-09", "2009-03"),
        "2022 rate shock":        ("2022-01", "2022-12"),
        "Full sample":            (excess.index.min(), excess.index.max()),
    }
    print("\n" + "=" * 78)
    print("EXHIBIT H — Recommended (Inverse-TE) vs 60/40: sub-period drawdowns (%)")
    print("=" * 78)
    rec_sp = cio.subperiod_drawdowns(ens_te, excess, cash, periods)
    bch_sp = cio.subperiod_drawdowns(bench, excess, cash, periods)
    sp = pd.concat({"Recommended": rec_sp, "60/40": bch_sp}, axis=1) * 100
    print(sp.round(2))

    print("\n" + "=" * 78)
    print("FINAL RECOMMENDED ALLOCATION (Inverse-TE ensemble, %)")
    print("=" * 78)
    final = (ens_te * 100).round(1).sort_values(ascending=False)
    print(final.to_string())
    print(f"\n  Equity total:       {final[cma.EQUITY].sum():.1f}%")
    print(f"  Fixed income total: {final[cma.FIXED_INCOME].sum():.1f}%")


if __name__ == "__main__":
    main()
