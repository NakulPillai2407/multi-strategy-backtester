"""Market regime detection.

Current implementation: labels every date "Normal".

Planned enhancement: classify each date into regimes using rolling
realized-volatility terciles (e.g. "Low Vol" / "Normal" / "High Vol") or a
Markov regime-switching model, so the Regime Breakdown tab can attribute
performance to market conditions.
"""

import pandas as pd

# Stable label vocabulary the future classifier will draw from; the UI
# uses this for consistent regime colors.
REGIME_LABELS = ["Low Vol", "Normal", "High Vol"]


def classify_regime(data: pd.DataFrame) -> pd.Series:
    """Assign a regime label to every date in ``data``.

    Parameters
    ----------
    data : pd.DataFrame
        Price panel with a DatetimeIndex.

    Returns
    -------
    pd.Series
        Regime label per date, named ``"regime"``. The current
        implementation returns the constant ``"Normal"``.
    """
    return pd.Series("Normal", index=data.index, name="regime")
