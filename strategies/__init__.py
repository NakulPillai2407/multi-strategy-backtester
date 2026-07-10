"""Strategy registry.

Adding a strategy = one new file in this package + one line here.
Nothing else in the app changes: the sidebar, tabs, and engine are all
driven off ``STRATEGY_REGISTRY`` and the ``BaseStrategy`` interface.
"""

from .example_placeholder import ExamplePlaceholderStrategy

STRATEGY_REGISTRY = {
    "Example Placeholder (SMA Crossover)": ExamplePlaceholderStrategy(),
    # Future: "Stat Arb / Pairs Trading": StatArbStrategy(),
    # Future: "Sentiment Signal": SentimentSignalStrategy(),
    # Future: "Portfolio Optimiser": PortfolioOptimiserStrategy(),
}
