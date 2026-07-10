"""Market data access layer.

Current implementation: a plain yfinance pull of adjusted close prices.

Planned enhancement: guard against lookahead bias, so no data point is
visible to a strategy before its true availability timestamp. Concretely
that means:

* point-in-time fundamentals (use as-first-reported values, not restated),
* survivorship-bias-free universes (include delisted tickers),
* corporate-action adjustments applied only with information available at
  the time,
* an explicit "as-of" API so walk-forward windows can request the data
  exactly as it looked on a given date.

The signature below is the stable contract; future work changes the body,
not the interface.
"""

import pandas as pd
import yfinance as yf


def load_price_data(tickers, start, end) -> pd.DataFrame:
    """Fetch daily close prices for one or more tickers.

    Parameters
    ----------
    tickers : str | list[str]
        Ticker symbol(s) understood by Yahoo Finance.
    start, end : date-like
        Inclusive start / exclusive end of the pull.

    Returns
    -------
    pd.DataFrame
        DatetimeIndex, one column of (auto-adjusted) close prices per
        ticker. Empty DataFrame if nothing could be fetched.
    """
    if isinstance(tickers, str):
        tickers = [t.strip().upper() for t in tickers.split(",") if t.strip()]

    raw = yf.download(
        tickers, start=start, end=end, auto_adjust=True, progress=False
    )
    if raw is None or raw.empty:
        return pd.DataFrame()

    # yfinance returns a column MultiIndex (field, ticker) for multi-ticker
    # pulls and sometimes for single tickers too; normalize to close-per-ticker.
    if isinstance(raw.columns, pd.MultiIndex):
        prices = raw["Close"]
    else:
        prices = raw[["Close"]].rename(columns={"Close": tickers[0]})

    prices = prices.dropna(how="all").ffill().dropna()
    prices.index = pd.to_datetime(prices.index).tz_localize(None)
    return prices
