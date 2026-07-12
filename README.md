# Multi-Strategy Backtesting Platform

A Streamlit backtesting engine built around a pluggable strategy interface. Every strategy is a single file under `strategies/` implementing `BaseStrategy`, and the app shell (sidebar, six analysis tabs) and engine never change when a new strategy is added: it's one import line in the strategy registry.

Built as a portfolio project for Quantitative Analyst / Financial Technology roles.

## Key Features

- Pluggable strategy interface: new strategies plug into the same pipeline (data handler, signal generation, portfolio accounting, cost model, metrics) without touching the app shell or engine
- Execution lagged one bar, so a signal computed on day T earns day T+1's return, avoiding same-bar lookahead
- Full P&L ledger with gross and net equity curves, trade log, and standard performance metrics (Sharpe, CAGR, max drawdown, hit rate)
- Walk-forward split into in-sample / out-of-sample windows, with rolling or expanding modes selectable in the sidebar
- Regime overlay attributing net returns to market conditions
- Animated equity curve and count-up metric cards in a dark, fintech-themed dashboard
- Sentiment Signal strategy: FinBERT news scoring vs. a VADER baseline, event-study abnormal returns, and OLS diagnostics rendered through the engine's diagnostics hook

## Methodology

Every strategy is evaluated under four categories of realism:

1. Walk-forward optimization: parameters are meant to be fit in-sample and judged out-of-sample on rolling or expanding windows (`engine/walk_forward.py`).
2. Transaction costs and slippage: commission and slippage are deducted from every trade, with market impact planned as a future addition (`engine/cost_model.py`).
3. Point-in-time data: no data should be visible to a strategy before its true availability timestamp (`engine/data_handler.py`).
4. Regime awareness: performance is attributed across market regimes (`engine/regime.py`).

Portfolio accounting and performance metrics are fully implemented. The cost model, walk-forward split, and regime classifier currently use simpler baseline implementations while the more realistic versions are built out; each tab in the app states plainly what's active now versus planned next.

## Roadmap

Implemented now:

- Streamlit shell, sidebar, and six-tab layout, all driven by the strategy registry
- Portfolio accounting and performance metrics (Sharpe, CAGR, drawdown, hit rate)
- Flat basis-point cost model
- Single fixed 70/30 walk-forward split
- Two strategies: a moving-average crossover used to validate the pipeline end to end, and a Sentiment Signal strategy (FinBERT news scoring, event-study abnormal returns, OLS diagnostics), ported from a standalone research app

Planned next:

- Stat Arb / Pairs Trading and Portfolio Optimiser strategies, each as its own module under `strategies/`
- A market-impact cost model (square-root impact proportional to trade size vs. average daily volume)
- A rolling/expanding walk-forward window generator with per-window parameter re-fitting
- A volatility-tercile or Markov regime classifier in place of the constant label
- Point-in-time data safeguards (as-first-reported fundamentals, survivorship-bias-free universes)

## Repo Structure

```
├── app.py                        # Streamlit shell: sidebar, tabs, theming
├── .streamlit/config.toml        # Dark fintech theme
├── engine/
│   ├── base_strategy.py          # BaseStrategy interface
│   ├── data_handler.py           # Price data access (yfinance)
│   ├── cost_model.py             # Commission/slippage cost model
│   ├── walk_forward.py           # Train/test window generation
│   ├── regime.py                 # Market regime classification
│   ├── portfolio.py              # Position sizing and P&L accounting
│   └── metrics.py                # Sharpe, drawdown, hit rate
└── strategies/
    ├── __init__.py                # STRATEGY_REGISTRY
    ├── example_placeholder.py     # SMA crossover strategy
    ├── sentiment_signal.py        # Sentiment Signal strategy + diagnostics UI
    └── _sentiment_utils.py        # News fetch, FinBERT/VADER scoring, event returns, OLS
```

## Installation & Running Locally

```bash
git clone https://github.com/NakulPillai2407/multi-strategy-backtester.git
cd multi-strategy-backtester
python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

Then open the local URL Streamlit prints (typically http://localhost:8501).

## Tech Stack

Python, Streamlit, Plotly, Pandas / NumPy, yfinance, transformers + torch (FinBERT), NLTK (VADER baseline), statsmodels (OLS diagnostics)

## Limitations

Execution assumes fills at the next bar's close, with no intraday fills, partial fills, or liquidity constraints modeled. The cost model is currently a flat basis-point deduction per unit of turnover, with no market impact, borrow costs, or financing. Data comes from plain yfinance daily closes with no point-in-time safeguards yet, so restated fundamentals, survivorship bias, and delisted tickers are not handled. The walk-forward split is currently a single fixed 70/30 division rather than a rolling re-fit, so results should be read as in-sample. Regime attribution is not yet informative, since every date is currently labeled the same. Sharpe ratios assume a 0% risk-free rate, and no leverage or margin is modeled.

Backtested performance is hypothetical and not indicative of future results.

## Author

**Nakul Pillai**
BSc Economics & Data Science, University of Southampton · Incoming MSc Financial Technology, Imperial College London

- LinkedIn: [linkedin.com/in/nakul-pillai](https://www.linkedin.com/in/nakul-pillai)
- GitHub: [@NakulPillai2407](https://github.com/NakulPillai2407)
