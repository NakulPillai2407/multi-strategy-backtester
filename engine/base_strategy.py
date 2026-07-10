"""Abstract strategy interface.

Every strategy in ``strategies/`` subclasses :class:`BaseStrategy` and is
registered in ``strategies/__init__.py``'s ``STRATEGY_REGISTRY``. The app
shell (sidebar, tabs) is driven entirely by this interface: adding a new
strategy must never require touching ``app.py`` or any engine module.
"""

from abc import ABC, abstractmethod

import pandas as pd


class BaseStrategy(ABC):
    """Contract between the engine/app shell and any concrete strategy.

    Class attributes
    ----------------
    name : str
        Display name, also the key used in ``STRATEGY_REGISTRY``.
    description : str
        Markdown rendered verbatim in the "Overview & Methodology" tab.
    param_schema : dict
        Declarative widget spec the sidebar renders without knowing the
        parameters' meaning. Supported entry shapes::

            {"type": "slider", "min": 10, "max": 252, "default": 60,
             "step": 1, "label": "Lookback (days)", "help": "..."}
            {"type": "number", "min": 0.0, "max": 1.0, "default": 0.5,
             "step": 0.05}
            {"type": "select", "options": ["a", "b"], "default": "a"}
            {"type": "checkbox", "default": True}

        ``label`` and ``help`` are optional everywhere; the sidebar
        title-cases the dict key when ``label`` is absent.
    """

    name: str
    description: str
    param_schema: dict

    @abstractmethod
    def generate_signals(self, data: pd.DataFrame, params: dict) -> pd.DataFrame:
        """Return target positions indexed by date.

        Parameters
        ----------
        data : pd.DataFrame
            Price panel from ``engine.data_handler.load_price_data``:
            one column of close prices per ticker, DatetimeIndex.
        params : dict
            Resolved parameter values keyed exactly like ``param_schema``.

        Returns
        -------
        pd.DataFrame
            Must contain a ``position`` column of target exposures
            (+1 long, -1 short, 0 flat; fractional sizes allowed) on the
            same DatetimeIndex as ``data``. The engine handles execution
            lag, P&L accounting, and costs; strategies only decide the
            desired position.
        """
        ...

    def custom_diagnostics(self, data: pd.DataFrame, params: dict, results: dict) -> None:
        """Optional hook for strategy-specific charts in the Diagnostics tab.

        A strategy may override this and call ``st.*`` / ``st.plotly_chart``
        directly to render extra visuals (e.g. spread z-scores for stat arb,
        sentiment time series for the sentiment strategy). ``results`` is the
        engine's output bundle (pnl frame, metrics dict, trade log).
        Default: no-op.
        """
        return None
