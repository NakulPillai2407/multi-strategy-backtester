"""Strategy registry.

Adding a strategy = one new file in this package + one line here.
Nothing else in the app changes: the sidebar, tabs, and engine are all
driven off ``STRATEGY_REGISTRY`` and the ``BaseStrategy`` interface.
"""

from .example_placeholder import ExamplePlaceholderStrategy
from .sentiment_signal import SentimentSignalStrategy

STRATEGY_REGISTRY = {
    "Example Placeholder (SMA Crossover)": ExamplePlaceholderStrategy(),
    "Sentiment Signal": SentimentSignalStrategy(),
    # Future: "Stat Arb / Pairs Trading": StatArbStrategy(),
    # Future: "Portfolio Optimiser": PortfolioOptimiserStrategy(),
}
