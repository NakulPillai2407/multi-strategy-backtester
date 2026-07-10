"""Performance metrics.

Standard formulas, not strategy-specific, and implemented in full. All
rates assume daily periods and 252 trading days/year.
"""

import numpy as np
import pandas as pd

TRADING_DAYS = 252


def compute_metrics(pnl: pd.Series) -> dict:
    """Summary statistics for a per-period return stream.

    Parameters
    ----------
    pnl : pd.Series
        Simple per-period (daily) strategy returns, net or gross.

    Returns
    -------
    dict
        ``total_return`` — cumulative simple return over the sample
        ``cagr`` — annualized compound growth rate
        ``ann_vol`` — annualized volatility of returns
        ``sharpe`` — annualized Sharpe ratio (0% risk-free)
        ``max_drawdown`` — worst peak-to-trough equity decline (negative)
        ``hit_rate`` — share of active periods (nonzero return) that were
        positive
        ``n_periods`` — number of periods in the sample
    """
    pnl = pnl.dropna()
    n = len(pnl)
    if n == 0:
        return {
            "total_return": 0.0, "cagr": 0.0, "ann_vol": 0.0,
            "sharpe": 0.0, "max_drawdown": 0.0, "hit_rate": 0.0,
            "n_periods": 0,
        }

    equity = (1.0 + pnl).cumprod()
    total_return = float(equity.iloc[-1] - 1.0)

    years = n / TRADING_DAYS
    cagr = float(equity.iloc[-1] ** (1.0 / years) - 1.0) if years > 0 else 0.0

    vol = float(pnl.std(ddof=1)) if n > 1 else 0.0
    ann_vol = vol * np.sqrt(TRADING_DAYS)
    sharpe = float(pnl.mean() / vol * np.sqrt(TRADING_DAYS)) if vol > 0 else 0.0

    drawdown = equity / equity.cummax() - 1.0
    max_drawdown = float(drawdown.min())

    active = pnl[pnl != 0.0]
    hit_rate = float((active > 0).mean()) if len(active) else 0.0

    return {
        "total_return": total_return,
        "cagr": cagr,
        "ann_vol": ann_vol,
        "sharpe": sharpe,
        "max_drawdown": max_drawdown,
        "hit_rate": hit_rate,
        "n_periods": n,
    }


def drawdown_series(pnl: pd.Series) -> pd.Series:
    """Running drawdown (equity vs. its high-water mark), for charting."""
    equity = (1.0 + pnl.fillna(0.0)).cumprod()
    return equity / equity.cummax() - 1.0
