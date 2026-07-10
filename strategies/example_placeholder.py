"""Moving-average crossover strategy.

The first strategy in the registry, used to validate the full pipeline
(signal generation, P&L, metrics, trade log) end to end. Stat Arb,
Sentiment Signal, and Portfolio Optimiser strategies are planned as
additional modules under this package.
"""

import numpy as np
import pandas as pd

from engine.base_strategy import BaseStrategy


class ExamplePlaceholderStrategy(BaseStrategy):
    name = "Example Placeholder (SMA Crossover)"
    description = (
        "A moving-average crossover strategy, the first entry in the "
        "strategy registry, used to validate the pipeline end to end. "
        "Additional strategies (Stat Arb, Sentiment Signal, Portfolio "
        "Optimiser) are planned as separate modules.\n\n"
        "**Logic:** go long (+1) when the fast simple moving average is above the "
        "slow one, short (-1) when it is below. Flat during the initial warm-up "
        "window before the slow average has enough history."
    )
    param_schema = {
        "fast_window": {
            "type": "slider", "min": 5, "max": 50, "default": 10, "step": 1,
            "label": "Fast SMA window (days)",
        },
        "slow_window": {
            "type": "slider", "min": 20, "max": 200, "default": 50, "step": 1,
            "label": "Slow SMA window (days)",
        },
    }

    def generate_signals(self, data: pd.DataFrame, params: dict) -> pd.DataFrame:
        px = data.iloc[:, 0]
        fast = px.rolling(int(params["fast_window"])).mean()
        slow = px.rolling(int(params["slow_window"])).mean()

        position = pd.Series(
            np.where(fast > slow, 1.0, -1.0), index=px.index, name="position"
        )
        position[slow.isna()] = 0.0  # flat until the slow SMA has history

        signals = position.to_frame()
        signals["fast_sma"] = fast
        signals["slow_sma"] = slow
        return signals
