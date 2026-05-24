"""
performance_assessment.py
=========================
Core module for portfolio construction and performance analytics.
Nordvik National Pension Reserve Fund — Case Study 2026.

Functions
---------
renormalize_prices
fix_mix_portfolio_construction
glide_path_weights
changing_weights_portfolio_construction
compute_performance_stats
compute_drawdown
regression_analysis
rolling_performance
compute_ff6_regression
compute_correlation_matrix
scenario_analysis
"""

import warnings
import numpy as np
import pandas as pd
import statsmodels.api as sm
from typing import Dict, Optional, Tuple

warnings.filterwarnings("ignore")

# ─── Constants ────────────────────────────────────────────────────────────────
TRADING_DAYS = 252
RF_ANNUAL_FALLBACK = 0.045   # used if no rf_series is supplied


# ─── Internal helpers ─────────────────────────────────────────────────────────

def _rebal_dates(index: pd.DatetimeIndex, freq: str) -> pd.DatetimeIndex:
    """Return end-of-period business dates for the given frequency string."""
    freq_map = {
        "D": "B", "W": "W-FRI", "M": "ME", "Q": "QE",
        "A": "YE", "Y": "YE", "SA": "2QE",
    }
    f = freq_map.get(freq.upper(), freq)
    dummy = pd.Series(0, index=index)
    return dummy.resample(f).last().index


def _daily_returns(prices: pd.DataFrame) -> pd.DataFrame:
    return prices.pct_change().fillna(0.0)


def _ann_ret(r: pd.Series) -> float:
    n = len(r)
    return float((1.0 + r).prod() ** (TRADING_DAYS / n) - 1.0)


def _ann_vol(r: pd.Series) -> float:
    return float(r.std() * np.sqrt(TRADING_DAYS))


# ─── Price utilities ──────────────────────────────────────────────────────────

def renormalize_prices(prices: pd.DataFrame, base: float = 100.0) -> pd.DataFrame:
    """Rebase every price series to `base` at its first observation."""
    return prices.div(prices.iloc[0]) * base


# ─── Portfolio construction ───────────────────────────────────────────────────

def fix_mix_portfolio_construction(
    weights: pd.Series,
    prices: pd.DataFrame,
    rebal_frequency: str = "Q",
) -> Tuple[pd.Series, pd.DataFrame]:
    """
    Build a fixed-weight portfolio with periodic rebalancing.

    At the close of each rebalancing date the portfolio is reset to target
    weights.  Between rebalancing dates weights drift with market prices.

    Parameters
    ----------
    weights          : target weights (index = asset names ⊆ prices.columns)
    prices           : daily price levels, no NaN on relevant assets
    rebal_frequency  : 'D' | 'W' | 'M' | 'Q' | 'A'

    Returns
    -------
    portfolio       : daily NAV rebased to 100 at the first date
    weight_history  : end-of-day drifted (actual) weights
    """
    assets = weights.index.tolist()
    px = prices[assets].copy()
    w_tgt = (weights / weights.sum()).values.astype(float)

    rets = _daily_returns(px).values
    rebal_set = set(_rebal_dates(px.index, rebal_frequency))

    n, k = rets.shape
    nav = np.empty(n)
    wh = np.empty((n, k))

    nav[0] = 100.0
    w = w_tgt.copy()
    wh[0] = w

    for i in range(1, n):
        gross = w * (1.0 + rets[i])          # position value vector
        nav[i] = nav[i - 1] * gross.sum()    # w sums to 1 → gross.sum() = 1 + port_ret
        w_drifted = gross / gross.sum()
        wh[i] = w_drifted
        # Rebalance at close of rebalancing date → tomorrow holds target
        w = w_tgt.copy() if px.index[i] in rebal_set else w_drifted

    return (
        pd.Series(nav, index=px.index, name="Portfolio"),
        pd.DataFrame(wh, index=px.index, columns=assets),
    )


def glide_path_weights(
    starting_weights: pd.Series,
    terminal_weights: pd.Series,
    prices: pd.DataFrame,
    rebal_frequency: str = "A",
) -> pd.DataFrame:
    """
    Generate a linear glide-path weight schedule.

    Weights are linearly interpolated from `starting_weights` at the first
    rebalancing date to `terminal_weights` at the last.

    Returns
    -------
    schedule : DataFrame  (index = rebalancing dates, columns = assets)
    """
    dates = _rebal_dates(prices.index, rebal_frequency)
    n = len(dates)
    idx = starting_weights.index
    schedule = pd.DataFrame(index=dates, columns=idx, dtype=float)

    for i, d in enumerate(dates):
        alpha = i / (n - 1) if n > 1 else 1.0
        row = starting_weights * (1.0 - alpha) + terminal_weights * alpha
        schedule.loc[d] = row.values

    return schedule


def changing_weights_portfolio_construction(
    weight_schedule: pd.DataFrame,
    prices: pd.DataFrame,
    rebal_frequency: str = "Q",
) -> Tuple[pd.Series, pd.DataFrame]:
    """
    Build a portfolio whose target weights change over time.

    The portfolio rebalances to the *current* target weights at each
    `rebal_frequency` date.  Target weights are updated whenever a date
    in `weight_schedule.index` is reached.

    Parameters
    ----------
    weight_schedule  : rows = schedule dates (annual), columns = assets
    prices           : daily price levels
    rebal_frequency  : intra-period rebalancing frequency (default 'Q')

    Returns
    -------
    portfolio       : daily NAV rebased to 100
    weight_history  : end-of-day drifted weights
    """
    assets = weight_schedule.columns.tolist()
    px = prices[assets].copy()
    rets = _daily_returns(px).values

    # Normalise each row of the schedule
    ws = weight_schedule.div(weight_schedule.sum(axis=1), axis=0)
    sched_dates = sorted(ws.index)
    sched_set = set(sched_dates)
    q_rebal_set = set(_rebal_dates(px.index, rebal_frequency))

    def _target(date: pd.Timestamp) -> np.ndarray:
        applicable = [d for d in sched_dates if d <= date]
        if not applicable:
            return ws.iloc[0].values.astype(float)
        return ws.loc[applicable[-1]].values.astype(float)

    n, k = rets.shape
    nav = np.empty(n)
    wh = np.empty((n, k))

    nav[0] = 100.0
    w = _target(px.index[0])
    wh[0] = w

    for i in range(1, n):
        date = px.index[i]
        gross = w * (1.0 + rets[i])
        nav[i] = nav[i - 1] * gross.sum()
        w_drifted = gross / gross.sum()
        wh[i] = w_drifted

        tgt = _target(date)
        w = tgt if (date in q_rebal_set or date in sched_set) else w_drifted

    return (
        pd.Series(nav, index=px.index, name="Portfolio"),
        pd.DataFrame(wh, index=px.index, columns=assets),
    )


# ─── Performance analytics ────────────────────────────────────────────────────

def compute_performance_stats(
    prices: pd.DataFrame,
    rf_series: Optional[pd.Series] = None,
    benchmark_col: Optional[str] = None,
) -> pd.DataFrame:
    """
    Compute annualised performance metrics for each column in `prices`.

    Metrics
    -------
    Ann. Return, Ann. Volatility, Sharpe, Max Drawdown, Calmar,
    Sortino, Skewness, Excess Kurtosis,
    + Tracking Error, Information Ratio, Jensen's Alpha (if benchmark supplied)
    """
    rets = prices.pct_change().dropna()
    if rf_series is not None:
        rf = rf_series.reindex(rets.index).ffill()
    else:
        rf = pd.Series(RF_ANNUAL_FALLBACK / TRADING_DAYS, index=rets.index)

    results = {}
    for col in rets.columns:
        r = rets[col]
        ann_r = _ann_ret(r)
        ann_v = _ann_vol(r)
        excess = r - rf
        sharpe = excess.mean() * TRADING_DAYS / ann_v if ann_v else np.nan

        dd, _ = compute_drawdown(prices[[col]])
        max_dd = float(dd[col].min())
        calmar = ann_r / abs(max_dd) if max_dd != 0 else np.nan

        neg = r[r < rf]
        dsv = neg.std() * np.sqrt(TRADING_DAYS) if len(neg) > 1 else np.nan
        rf_ann = float(rf.mean() * TRADING_DAYS) if len(rf) else RF_ANNUAL_FALLBACK
        sortino = (ann_r - rf_ann) / dsv if dsv and dsv > 0 else np.nan

        # CVaR (Expected Shortfall) at 95%: expected loss given we are in the
        # worst 5% of daily outcomes, annualised via the square-root-of-time rule.
        sorted_r = np.sort(r.values)
        tail_n   = max(1, int(len(sorted_r) * 0.05))
        cvar_ann = -sorted_r[:tail_n].mean() * np.sqrt(TRADING_DAYS) * 100

        m = {
            "Ann. Return (%)":    round(ann_r * 100, 2),
            "Ann. Volatility (%)":round(ann_v * 100, 2),
            "Sharpe Ratio":       round(sharpe, 3) if np.isfinite(sharpe) else np.nan,
            "Max Drawdown (%)":   round(max_dd * 100, 2),
            "CVaR 95% (ann. %)":  round(cvar_ann, 2),
            "Calmar Ratio":       round(calmar, 3) if np.isfinite(calmar) else np.nan,
            "Sortino Ratio":      round(sortino, 3) if np.isfinite(sortino) else np.nan,
            "Skewness":           round(float(r.skew()), 3),
            "Excess Kurtosis":    round(float(r.kurt()), 3),
        }

        if benchmark_col and benchmark_col in rets.columns and col != benchmark_col:
            bm = rets[benchmark_col]
            active = r - bm
            te = active.std() * np.sqrt(TRADING_DAYS) * 100
            ir = (active.mean() / active.std() * np.sqrt(TRADING_DAYS)
                  if active.std() > 0 else np.nan)
            X = sm.add_constant(bm.values)
            ols = sm.OLS(r.values, X).fit()
            alpha_ann = ((1.0 + float(ols.params[0])) ** TRADING_DAYS - 1.0) * 100
            m.update({
                "Tracking Error (%)": round(te, 2),
                "Information Ratio": round(ir, 3) if np.isfinite(ir) else np.nan,
                "Jensen's Alpha (% p.a.)": round(alpha_ann, 3),
            })

        results[col] = m

    return pd.DataFrame(results).T


def compute_drawdown(
    prices: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Return (drawdown_series, running_max_drawdown).
    Both on a 0-to-negative scale (0 = at peak, -0.20 = 20% below peak).
    """
    dd = prices / prices.cummax() - 1.0
    max_dd = dd.cummin()
    return dd, max_dd


def regression_analysis(
    portfolio_prices: pd.DataFrame,
    benchmark_prices: Optional[pd.Series] = None,
) -> pd.DataFrame:
    """Simple market-model OLS: R_p = alpha + beta * R_bm + eps."""
    rets = portfolio_prices.pct_change().dropna()
    if benchmark_prices is None:
        return pd.DataFrame()
    bm = benchmark_prices.pct_change().dropna().reindex(rets.index).dropna()
    rets = rets.reindex(bm.index)
    out = {}
    for col in rets.columns:
        r = rets[col]
        X = sm.add_constant(bm.values)
        model = sm.OLS(r.values, X).fit()
        out[col] = {
            "Alpha (ann. %)": round(
                ((1 + float(model.params[0])) ** TRADING_DAYS - 1) * 100, 3),
            "Beta": round(float(model.params[1]), 4),
            "R²": round(model.rsquared, 4),
            "t(Alpha)": round(float(model.tvalues[0]), 3),
            "t(Beta)": round(float(model.tvalues[1]), 3),
        }
    return pd.DataFrame(out).T


def rolling_performance(
    prices: pd.DataFrame,
    rolling_window_years: int = 3,
    rf_series: Optional[pd.Series] = None,
) -> Dict[str, pd.DataFrame]:
    """
    Rolling annualised return, volatility, and Sharpe ratio.

    Returns a dict with keys 'Ann. Return', 'Ann. Volatility', 'Sharpe'.
    """
    window = rolling_window_years * TRADING_DAYS
    rets = prices.pct_change().dropna()

    roll_ret = rets.rolling(window).apply(
        lambda x: (1.0 + x).prod() ** (TRADING_DAYS / len(x)) - 1.0,
        raw=True,
    )
    roll_vol = rets.rolling(window).std() * np.sqrt(TRADING_DAYS)
    if rf_series is not None:
        rf = rf_series.reindex(rets.index).ffill().bfill()
        roll_rf = rf.rolling(window).mean() * TRADING_DAYS
    else:
        roll_rf = pd.Series(RF_ANNUAL_FALLBACK, index=rets.index)
    roll_sharpe = roll_ret.sub(roll_rf, axis=0) / roll_vol

    return {
        "Ann. Return": roll_ret * 100,
        "Ann. Volatility": roll_vol * 100,
        "Sharpe": roll_sharpe,
    }


def compute_ff6_regression(
    portfolio_monthly_ret: pd.Series,
    ff6_monthly: pd.DataFrame,
) -> Dict:
    """
    Fama-French 6-factor OLS regression (HC3 robust SEs).

    Parameters
    ----------
    portfolio_monthly_ret : excess return series (portfolio return - RF),
                            monthly frequency, decimal units.
    ff6_monthly           : DataFrame with columns
                            [Mkt-RF, SMB, HML, RMW, CMA, WML], decimal units.

    Returns
    -------
    dict: params, tvalues, pvalues, rsquared, rsquared_adj, nobs, summary_text
    """
    aligned_ff = ff6_monthly.reindex(portfolio_monthly_ret.index).dropna()
    y = portfolio_monthly_ret.reindex(aligned_ff.index).dropna()
    X_df = aligned_ff.reindex(y.index)

    X = sm.add_constant(X_df)
    model = sm.OLS(y, X).fit(cov_type="HC3")

    return {
        "params": model.params,
        "tvalues": model.tvalues,
        "pvalues": model.pvalues,
        "rsquared": round(model.rsquared, 4),
        "rsquared_adj": round(model.rsquared_adj, 4),
        "nobs": int(model.nobs),
        "summary_text": str(model.summary()),
    }


def compute_correlation_matrix(
    prices: pd.DataFrame,
    freq: str = "W",
) -> pd.DataFrame:
    """Compute pairwise return correlations at the given frequency."""
    freq_map = {"W": "W-FRI", "M": "ME", "Q": "QE", "D": "B"}
    f = freq_map.get(freq.upper(), "W-FRI")
    px_resampled = prices.resample(f).last().dropna(how="all")
    rets = px_resampled.pct_change().dropna(how="all")
    return rets.corr()


def scenario_analysis(
    prices: pd.DataFrame,
    scenarios: Dict[str, Tuple[str, str]],
) -> pd.DataFrame:
    """
    Evaluate portfolio performance over named stress periods.

    Parameters
    ----------
    prices    : NAV series (rebased to 100)
    scenarios : {name: (start_date, end_date)}

    Returns
    -------
    DataFrame with total return and maximum intra-period drawdown per scenario.
    """
    rows = []
    for name, (s, e) in scenarios.items():
        sub = prices.loc[s:e]
        if sub.empty or len(sub) < 2:
            continue
        total_ret = (sub.iloc[-1] / sub.iloc[0] - 1.0) * 100
        dd, _ = compute_drawdown(sub)
        max_dd = dd.min() * 100
        row = {}
        for col in prices.columns:
            row[(col, "Total Return (%)")] = round(float(total_ret[col]), 2)
            row[(col, "Max DD (%)")] = round(float(max_dd[col]), 2)
        rows.append(pd.Series(row, name=name))
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    df.columns = pd.MultiIndex.from_tuples(df.columns)
    return df
