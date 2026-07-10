"""Transaction cost and slippage model.

Current implementation: a flat basis-point deduction per unit of turnover.

Planned enhancement: a market-impact term proportional to trade size
relative to average daily volume (e.g. square-root impact), plus separate
treatment of spread cost vs. commission, so large rebalances are penalized
realistically.
"""

import pandas as pd


def apply_costs(
    returns: pd.Series,
    trades: pd.Series,
    commission_bps: float,
    slippage_bps: float,
) -> pd.Series:
    """Deduct trading costs from a gross return stream.

    Parameters
    ----------
    returns : pd.Series
        Gross per-period strategy returns.
    trades : pd.Series
        Absolute turnover per period (|Δposition|, in units of exposure),
        aligned to ``returns``.
    commission_bps, slippage_bps : float
        Flat one-way costs in basis points, charged on each unit of
        turnover.

    Returns
    -------
    pd.Series
        Net per-period returns after costs.
    """
    per_unit_cost = (commission_bps + slippage_bps) / 1e4
    costs = trades.reindex(returns.index).fillna(0.0).abs() * per_unit_cost
    return returns - costs
