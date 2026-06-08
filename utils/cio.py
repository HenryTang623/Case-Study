import pandas as pd
import numpy as np

class Portfolio:
    """Portfolio backtest and performance metrics."""
    def __init__(self, target_weights, returns_df, total_returns_df):
        self.target_weights = target_weights
        self.returns_df = returns_df
        self.total_returns_df = total_returns_df
        self.weights_df = None
        self.portfolio_series = None

    def get_rebalance_dates(self):
        dates = self.returns_df.index
        rebal_dates = dates[dates.is_quarter_start]

        # Ensure first date is included as a rebalance date
        if dates[0] not in rebal_dates:
            rebal_dates = dates[[0]].append(rebal_dates)

        return rebal_dates

    def construct_weights_df(self):
        dates = self.returns_df.index
        rebal_dates = self.get_rebalance_dates()
        assets = self.target_weights.index

        # initialize wt df
        weights_df = pd.DataFrame(index=dates, columns=assets, dtype=float)

        # plug in target wt
        weights_df.loc[rebal_dates] = self.target_weights.values
        
        # fill in drift wt
        for i in range(len(rebal_dates) - 1):
            start_date = rebal_dates[i]
            end_date = rebal_dates[i + 1]

            # get the period between rebalances (excluding start, including end is next rebal)
            period_mask = (dates > start_date) & (dates < end_date)
            period_dates = dates[period_mask]
            if len(period_dates) == 0:
                continue

            # starting weights (target weights at rebalance)
            w_start = self.target_weights.values

            # calculate drift weights
            for date in period_dates:
                # Get total returns from start_date to current date (drift tracks
                # actual holding growth, not return relative to cash)
                period_returns = self.total_returns_df.loc[start_date:date, assets]

                # calculate cumulative growth for each asset
                cum_growth = (1 + period_returns).prod(axis=0).values

                # drift weights = initial weights * cumulative growth
                drift_weights = w_start * cum_growth

                # normalize to sum to 1
                drift_weights = drift_weights / drift_weights.sum()

                weights_df.loc[date] = drift_weights

        # handle the last period (after last rebalance date)
        last_rebal = rebal_dates[-1]
        after_last_rebal = dates[dates > last_rebal]

        w_start = self.target_weights.values

        for date in after_last_rebal:
            period_returns = self.total_returns_df.loc[last_rebal:date, assets]
            cum_growth = (1 + period_returns).prod(axis=0).values
            drift_weights = w_start * cum_growth
            drift_weights = drift_weights / drift_weights.sum()
            weights_df.loc[date] = drift_weights

        self.weights_df = weights_df
        return weights_df

    def construct_portfolio_series(self):
        """portfolio backtest return series."""
        if self.weights_df is None:
            self.construct_weights_df()
        portfolio_returns = (self.weights_df * self.returns_df[self.target_weights.index]).sum(axis=1)
        self.portfolio_series = portfolio_returns
        return portfolio_returns

    def calculate_metrics(self, cov_matrix, benchmark_wts):
        """
        calculate all portfolio performance metrics:

        ann_return, ann_vol, sharpe, max_drawdown, turnover, tracking_error, concentration_HHI

        and store them on the instance for downstream composite scoring.
        """
        if self.portfolio_series is None:
            self.construct_portfolio_series()

        ps = self.portfolio_series.dropna()

        ann_return = ps.mean() * 12
        ann_vol = ps.std() * np.sqrt(12)
        sharpe = ann_return / ann_vol if ann_vol > 0 else np.nan

        # max dd.
        cum_returns = (1 + ps).cumprod()
        running_max = cum_returns.cummax()
        drawdown = (cum_returns / running_max - 1)
        max_dd = drawdown.min()

        # store metrics for later composite scoring
        self.ann_return = ann_return
        self.ann_vol = ann_vol
        self.sharpe = sharpe
        self.max_drawdown = max_dd
        self.turnover = self.calculate_turnover()
        self.tracking_error_val = self.tracking_error(cov_matrix, benchmark_wts)
        self.concentration_HHI = self.concentration_hhi()
        self.cvar = self.calculate_cvar()

        self.metrics = {
            'ann_return': self.ann_return,
            'ann_vol': self.ann_vol,
            'sharpe': self.sharpe,
            'max_dd': self.max_drawdown,
            'turnover': self.turnover,
            'track_err': self.tracking_error_val,
            'HHI': self.concentration_HHI,
            'cvar': self.cvar,
        }

        return self.metrics

    def calculate_turnover(self):
        periods_per_year = 4
        rebal_dates = self.get_rebalance_dates()
        turnovers = []

        for i in range(1, len(rebal_dates)):
            curr_date = rebal_dates[i]
            dates = self.weights_df.index
            idx = dates.get_loc(curr_date)
            if idx > 0:
                pre_rebal_weights = self.weights_df.iloc[idx - 1]
                post_rebal_weights = self.weights_df.loc[curr_date]
                turnover = (post_rebal_weights - pre_rebal_weights).abs().sum() / 2
                turnovers.append(turnover)

        return np.mean(turnovers) * periods_per_year

    def calculate_cvar(self, alpha=0.95):
        """empirical CVaR_alpha of portfolio losses (positive number = worse)."""
        ps = self.portfolio_series.dropna()
        losses = -ps
        var = losses.quantile(alpha)
        tail = losses[losses >= var]
        return tail.mean()

    def tracking_error(self, cov_matrix, benchmark_wts):
        """ex-ante TE"""
        active_weights = (self.target_weights.reindex(cov_matrix.index) - benchmark_wts.reindex(cov_matrix.index)).values
        return np.sqrt(active_weights @ cov_matrix.values @ active_weights)
    
    def concentration_hhi(self):
        return (self.target_weights ** 2).sum()
    
# ----------------------------------------------------------------------------
# Scoring and Ensemble functions
# ----------------------------------------------------------------------------
def calculate_composite_scores(portfolios_dict, w_sharpe=0.35, w_dd=0.30, w_cvar=0.0, w_te=0.15, w_conc=0.0, w_turnover=0.0):
    """
    calculate composite scores for all portfolios via weighted-avg percentile scoring across all portfolios.
    """
    # extract metrics from all portfolios
    metrics = {}
    for name, portfolio in portfolios_dict.items():
        metrics[name] = {
            'sharpe': portfolio.sharpe,
            'max_dd': portfolio.max_drawdown, # negative here
            'cvar': portfolio.cvar, # positive here, higher = worse
            'track_err': portfolio.tracking_error_val,
            'HHI': portfolio.concentration_HHI,
            'turnover': portfolio.turnover,
        }

    metrics_df = pd.DataFrame(metrics).T

    # percentile scoring: each metric mapped to its rank percentile in [0, 1]
    def percentile_score(s, higher_better=True):
        z = s.rank(pct=True)
        return z if higher_better else 1 - z

    # Calculate composite score
    comp = (
        w_sharpe * percentile_score(metrics_df['sharpe'])
        + w_dd * percentile_score(metrics_df['max_dd'])
        + w_cvar * percentile_score(metrics_df['cvar'], higher_better=False)
        + w_te * percentile_score(metrics_df['track_err'], higher_better=False)
        + w_conc * percentile_score(metrics_df['HHI'], higher_better=False)
        + w_turnover * percentile_score(metrics_df['turnover'], higher_better=False)
    )

    # Store composite score back to each portfolio
    for name, portfolio in portfolios_dict.items():
        portfolio.composite_score = comp[name]

    return comp.sort_values(ascending=False)


def score_all_portfolios(portfolios_dict, cov_matrix, benchmark_wts, w_sharpe=0.35, w_dd=0.30, w_cvar=0.0, w_te=0.15, w_conc=0.0, w_turnover=0.0):
    """compute metrics for every portfolio and rank them by composite score."""
    metrics = {}
    for name, portfolio in portfolios_dict.items():
        metrics[name] = portfolio.calculate_metrics(cov_matrix, benchmark_wts)

    composite = calculate_composite_scores(portfolios_dict, w_sharpe, w_dd, w_cvar, w_te, w_conc, w_turnover)

    scores_df = pd.DataFrame(metrics).T
    scores_df['composite_score'] = composite

    return scores_df


def ensemble_top3(portfolios_dict):
    """ensemble the top 3 portfolios based on composite score via simple average."""
    # Get top 3 portfolios by composite score
    composite_scores = {name: portfolio.composite_score
                       for name, portfolio in portfolios_dict.items()}
    composite_series = pd.Series(composite_scores).sort_values(ascending=False)
    top3_names = composite_series.head(3).index.tolist()

    # Extract weights from top 3 portfolios
    weights_list = []
    for name in top3_names:
        weights_list.append(portfolios_dict[name].target_weights)

    weights_df = pd.DataFrame(weights_list, index=top3_names)

    return ensemble_simple_avg(weights_df)


def ensemble_simple_avg(weights_df):
    """simple average ensemble."""
    return weights_df.mean(axis=0)

