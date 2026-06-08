# Strategic Portfolio Design Case Study

A 3-layer systematic and evidence-based SAA portfolio construction framework that mimics real-world SAA portfolio construction workflow, including Capital Market Assumptions, Portfolio Constructions and CIO Judgement & Decision. 

## Layout

```
main.py                          # orchestrates the full pipeline
utils/
  cma.py                         # data loading, excess returns, CMA (mu, covariance)
  portcon.py                     # IPS constraints and portfolio construction methods
  cio.py                         # backtesting, performance metrics, scoring, ensembling
spd_case_study_data_2026.xlsx    # monthly return data ("Monthly Return" sheet)
readings/                        # background reading material (PDFs)
```

## Pipeline overview

1. **CMA layer** (`utils/cma.py`)
   - Loads monthly total returns and converts risk-asset returns to excess-of-cash returns.
   - Backfills missing US HY Credit history via an OLS regression on US IG Credit and US
     equity (with bootstrapped residuals) since HY history starts later than other assets.
   - Builds expected excess returns (historical annualized mean) and an annualized
     Ledoit-Wolf shrunk covariance matrix.
   - Defines the 60/40 reference portfolio as benchmark (60% split equally across equity sleeves, 40% split
     equally across fixed income sleeves).

2. **Portfolio construction layer** (`utils/portcon.py`)
   - `Constraints` encodes the IPS: long-only, fully invested, per-asset cap (35%),
     equity cap (65%), fixed income cap (80%).
   - Constructs six candidate portfolios spanning four method families:
     - Heuristic: Equal Weight, Inverse Vol
     - Risk-structured: Min Variance, Risk Parity
     - Return-optimized: Max Sharpe (tangency portfolio via efficient frontier sweep)
     - Non-traditional: CVaR-Min (Rockafellar-Uryasev 95% CVaR minimization)

3. **CIO layer** (`utils/cio.py`)
   - `Portfolio` simulates a quarterly-rebalanced backtest with weight drift between
     rebalances, and computes performance metrics: annualized return/vol, Sharpe, max
     drawdown, turnover, ex-ante tracking error vs. benchmark, concentration (HHI), CVaR.
   - Composite scores are computed via weighted-average percentile ranking across
     portfolios, under five lens-specific weighting scenarios (Risk-Adjusted Return,
     Downside Protection, Diversification, Governance, Balanced).
   - For each scenario, the top-3 scoring portfolios are ensembled (simple average) and
     backtested against the 60/40 benchmark.

## Running

```
python main.py
```

This prints, in order: expected excess returns and correlation matrix, candidate
portfolio allocations, the scorecard across priority scenarios, the resulting ensemble
allocations per scenario, and a full-sample backtest of each ensemble vs. the 60/40
benchmark.

## Dependencies

`pandas`, `numpy`, `statsmodels`, `scikit-learn`, `cvxpy`, `openpyxl`