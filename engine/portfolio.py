"""Position sizing and P&L accounting.

This is engine plumbing, not strategy-specific, and is implemented in full.
Signals declare *target* positions; execution is lagged one bar so a
signal computed on day T earns day T+1's return, which is the engine's
structural guard against same-bar lookahead.
"""

import pandas as pd


def compute_pnl(signals: pd.DataFrame, prices: pd.DataFrame) -> pd.DataFrame:
    """Turn target positions and prices into a full P&L ledger.

    Parameters
    ----------
    signals : pd.DataFrame
        Must contain a ``position`` column of target exposures
        (+1 long, -1 short, 0 flat; fractional allowed), DatetimeIndex.
    prices : pd.DataFrame
        Close prices from ``load_price_data``. Single-asset accounting for
        now: the first column is traded. (Multi-asset weight vectors are a
        natural extension behind this same signature.)

    Returns
    -------
    pd.DataFrame
        Indexed like ``prices`` with columns:

        * ``price`` — traded asset's close
        * ``position`` — target position declared by the strategy
        * ``held_position`` — position actually held over the period
          (target lagged one bar)
        * ``asset_return`` — asset's simple return
        * ``gross_return`` — ``held_position * asset_return``
        * ``trade_size`` — absolute change in position (turnover)
        * ``equity_gross`` — cumulative gross equity curve, start = 1.0
    """
    px = prices.iloc[:, 0]

    ledger = pd.DataFrame(index=px.index)
    ledger["price"] = px
    ledger["position"] = (
        signals["position"].reindex(px.index).ffill().fillna(0.0)
    )
    ledger["held_position"] = ledger["position"].shift(1).fillna(0.0)
    ledger["asset_return"] = px.pct_change().fillna(0.0)
    ledger["gross_return"] = ledger["held_position"] * ledger["asset_return"]
    ledger["trade_size"] = ledger["position"].diff().abs().fillna(
        ledger["position"].abs()
    )
    ledger["equity_gross"] = (1.0 + ledger["gross_return"]).cumprod()
    return ledger


def extract_trades(ledger: pd.DataFrame) -> pd.DataFrame:
    """Build a human-readable trade log from the P&L ledger.

    Returns one row per position change: date, direction entered
    (LONG / SHORT / FLAT), signed size of the change, and execution price
    (that bar's close).
    """
    changes = ledger[ledger["trade_size"] > 1e-12]
    if changes.empty:
        return pd.DataFrame(columns=["Date", "Direction", "Size", "Price"])

    def direction(target: float) -> str:
        if target > 0:
            return "LONG"
        if target < 0:
            return "SHORT"
        return "FLAT"

    return pd.DataFrame(
        {
            "Date": changes.index.date,
            "Direction": [direction(p) for p in changes["position"]],
            "Size": (changes["position"] - changes["held_position"]).round(4),
            "Price": changes["price"].round(2),
        }
    ).reset_index(drop=True)
