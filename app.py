"""Multi-Strategy Backtesting Platform: Streamlit app shell.

The shell is strategy-agnostic: the sidebar and all six tabs are driven
entirely by ``STRATEGY_REGISTRY`` and the ``BaseStrategy`` interface.
New strategies (Stat Arb, Sentiment Signal, Portfolio Optimiser) plug in
as new files under ``strategies/`` without touching this file.
"""

import datetime as dt

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from engine.base_strategy import BaseStrategy
from engine.cost_model import apply_costs
from engine.data_handler import load_price_data
from engine.metrics import compute_metrics, drawdown_series
from engine.portfolio import compute_pnl, extract_trades
from engine.regime import classify_regime
from engine.walk_forward import split_windows
from strategies import STRATEGY_REGISTRY

# ---------------------------------------------------------------------------
# Theme
# ---------------------------------------------------------------------------

PALETTE = {
    "bg": "#0B0F19",         # deep charcoal-navy
    "panel": "#111827",      # cards / sidebar
    "panel_edge": "#1E2638", # borders & muted gridlines
    "text": "#E6EAF2",
    "text_dim": "#8B93A7",
    "long": "#00D09C",       # buy / long / positive
    "short": "#FF5C7A",      # sell / short / negative
    "accent": "#4F8CFF",     # neutral accent (net curve, links)
    "gold": "#F5B759",       # highlights
}

FONT_STACK = (
    "'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, "
    "'Helvetica Neue', sans-serif"
)

CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

html, body, [class*="css"], .stApp, p, div, span, li, label {
    font-family: __FONT__;
}
.stApp { background: __BG__; }

/* Headline block */
.msb-header {
    padding: 0.2rem 0 0.6rem 0;
    border-bottom: 1px solid __EDGE__;
    margin-bottom: 0.4rem;
}
.msb-header h1 {
    font-size: 1.55rem; font-weight: 800; letter-spacing: -0.02em;
    color: __TEXT__; margin: 0;
}
.msb-header .sub { color: __DIM__; font-size: 0.9rem; margin-top: 0.15rem; }
.msb-chip {
    display: inline-block; font-size: 0.72rem; font-weight: 600;
    padding: 0.15rem 0.6rem; border-radius: 999px; margin-left: 0.5rem;
    background: rgba(0, 208, 156, 0.12); color: __LONG__;
    border: 1px solid rgba(0, 208, 156, 0.35); vertical-align: middle;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background: __PANEL__;
    border-right: 1px solid __EDGE__;
}
section[data-testid="stSidebar"] .stMarkdown h3 {
    font-size: 0.78rem; text-transform: uppercase; letter-spacing: 0.08em;
    color: __DIM__; margin-bottom: 0.2rem;
}

/* Tabs */
.stTabs [data-baseweb="tab-list"] {
    gap: 0.25rem; border-bottom: 1px solid __EDGE__;
}
.stTabs [data-baseweb="tab"] {
    font-weight: 600; font-size: 0.88rem; color: __DIM__;
    padding: 0.6rem 1rem; border-radius: 8px 8px 0 0;
}
.stTabs [aria-selected="true"] {
    color: __TEXT__ !important;
    background: rgba(79, 140, 255, 0.08);
}

/* Implementation-status note */
.msb-note {
    display: inline-block; font-size: 0.75rem; font-weight: 600;
    padding: 0.25rem 0.7rem; border-radius: 999px;
    background: rgba(79, 140, 255, 0.10); color: __ACCENT__;
    border: 1px solid rgba(79, 140, 255, 0.30);
    margin-bottom: 0.6rem;
}

/* Card-style containers for section blocks */
.msb-card {
    background: __PANEL__;
    border: 1px solid __EDGE__;
    border-radius: 12px;
    padding: 1rem 1.2rem;
    box-shadow: 0 2px 12px rgba(0, 0, 0, 0.35);
    margin-bottom: 0.8rem;
}
.msb-card h4 {
    margin: 0 0 0.35rem 0; font-size: 0.95rem; font-weight: 700; color: __TEXT__;
}
.msb-card p { color: __DIM__; font-size: 0.86rem; margin: 0; }

/* Dataframes */
[data-testid="stDataFrame"] {
    border: 1px solid __EDGE__; border-radius: 10px; overflow: hidden;
}

hr { border-color: __EDGE__; }
</style>
"""


def inject_css() -> None:
    css = (
        CUSTOM_CSS
        .replace("__FONT__", FONT_STACK)
        .replace("__BG__", PALETTE["bg"])
        .replace("__PANEL__", PALETTE["panel"])
        .replace("__EDGE__", PALETTE["panel_edge"])
        .replace("__TEXT__", PALETTE["text"])
        .replace("__DIM__", PALETTE["text_dim"])
        .replace("__LONG__", PALETTE["long"])
        .replace("__GOLD__", PALETTE["gold"])
        .replace("__ACCENT__", PALETTE["accent"])
    )
    st.markdown(css, unsafe_allow_html=True)


def themed_layout(**overrides) -> dict:
    """Base Plotly layout matching the app theme."""
    layout = dict(
        template=None,
        paper_bgcolor=PALETTE["bg"],
        plot_bgcolor=PALETTE["bg"],
        font=dict(family=FONT_STACK, color=PALETTE["text"], size=12),
        hoverlabel=dict(
            bgcolor=PALETTE["panel"],
            bordercolor=PALETTE["panel_edge"],
            font=dict(family=FONT_STACK, color=PALETTE["text"], size=12),
        ),
        xaxis=dict(gridcolor=PALETTE["panel_edge"], zeroline=False,
                   linecolor=PALETTE["panel_edge"]),
        yaxis=dict(gridcolor=PALETTE["panel_edge"], zeroline=False,
                   linecolor=PALETTE["panel_edge"]),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
        margin=dict(l=40, r=20, t=40, b=40),
    )
    layout.update(overrides)
    return layout


# ---------------------------------------------------------------------------
# Animated components (Plotly frames + JS count-up metric cards)
# ---------------------------------------------------------------------------

_ANIMATED_CHART_HTML = """
<div id="chart" style="width:100%;height:__PLOT_H__px;"></div>
<script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
<script>
  const fig = __FIG_JSON__;
  const gd = document.getElementById('chart');
  Plotly.newPlot(gd, fig.data, fig.layout,
                 {responsive: true, displaylogo: false,
                  modeBarButtonsToRemove: ['lasso2d', 'select2d']})
    .then(() => {
      if (fig.frames && fig.frames.length) {
        Plotly.addFrames(gd, fig.frames).then(() => {
          // Progressive draw-in on load
          Plotly.animate(gd, fig.frames.map(f => f.name), {
            frame: {duration: __FRAME_MS__, redraw: false},
            transition: {duration: 0},
            mode: 'immediate'
          });
        });
      }
    });
</script>
"""


def render_animated_chart(fig: go.Figure, height: int = 480,
                          frame_ms: int = 25) -> None:
    """Render a Plotly figure with frames that auto-plays on load.

    ``st.plotly_chart`` can't trigger the frames API on load, so the
    figure is embedded via a small HTML iframe that calls
    ``Plotly.animate`` once mounted. The figure's own play/pause buttons
    and scrub slider remain available for replay.
    """
    html = (
        _ANIMATED_CHART_HTML
        .replace("__FIG_JSON__", fig.to_json())
        .replace("__PLOT_H__", str(height - 20))
        .replace("__FRAME_MS__", str(frame_ms))
    )
    st.iframe(html, height=height)


_METRIC_CARDS_HTML = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@500;600;800&display=swap');
.row { display: flex; gap: 14px; font-family: __FONT__; }
.card {
  flex: 1; background: __PANEL__; border: 1px solid __EDGE__;
  border-radius: 12px; padding: 16px 18px;
  box-shadow: 0 2px 12px rgba(0,0,0,0.35);
  transition: transform .2s ease, border-color .2s ease;
}
.card:hover { transform: translateY(-2px); border-color: __ACCENT__; }
.label { font-size: 11px; font-weight: 600; letter-spacing: .08em;
         text-transform: uppercase; color: __DIM__; margin-bottom: 6px; }
.value { font-size: 26px; font-weight: 800; letter-spacing: -0.02em; }
.sub   { font-size: 11px; color: __DIM__; margin-top: 4px; }
</style>
<div class="row">__CARDS__</div>
<script>
  const ease = t => 1 - Math.pow(1 - t, 3);
  document.querySelectorAll('.value').forEach(el => {
    const target = parseFloat(el.dataset.target);
    const dec = parseInt(el.dataset.decimals);
    const suffix = el.dataset.suffix || '';
    const t0 = performance.now(), dur = 1100;
    function tick(now) {
      const p = Math.min((now - t0) / dur, 1);
      el.textContent = (target * ease(p)).toFixed(dec) + suffix;
      if (p < 1) requestAnimationFrame(tick);
    }
    requestAnimationFrame(tick);
  });
</script>
"""


def render_metric_cards(cards: list) -> None:
    """Card-style metrics whose values count up on load.

    ``cards``: list of dicts with keys ``label``, ``value`` (float),
    ``decimals``, ``suffix``, ``color`` (hex), optional ``sub`` caption.
    """
    blocks = []
    for c in cards:
        blocks.append(
            '<div class="card">'
            f'<div class="label">{c["label"]}</div>'
            f'<div class="value" style="color:{c["color"]}"'
            f' data-target="{c["value"]}" data-decimals="{c["decimals"]}"'
            f' data-suffix="{c.get("suffix", "")}">'
            f'{c["value"]:.{c["decimals"]}f}{c.get("suffix", "")}</div>'
            f'<div class="sub">{c.get("sub", "")}</div>'
            "</div>"
        )
    html = (
        _METRIC_CARDS_HTML
        .replace("__FONT__", FONT_STACK)
        .replace("__PANEL__", PALETTE["panel"])
        .replace("__EDGE__", PALETTE["panel_edge"])
        .replace("__ACCENT__", PALETTE["accent"])
        .replace("__DIM__", PALETTE["text_dim"])
        .replace("__CARDS__", "".join(blocks))
    )
    st.iframe(html, height=120)


def build_animated_equity_fig(equity: pd.DataFrame, n_frames: int = 60) -> go.Figure:
    """Equity curve(s) with progressive draw-in frames and a scrub slider.

    ``equity``: DataFrame of equity curves (one line per column), the
    first column drawn in the long/gross color, the second in the accent.
    """
    colors = [PALETTE["long"], PALETTE["accent"], PALETTE["gold"]]
    x = equity.index
    n = len(equity)
    cuts = np.unique(np.linspace(max(2, n // n_frames), n, min(n_frames, n)).astype(int))

    def traces(k: int):
        return [
            go.Scatter(
                x=x[:k], y=equity[col].iloc[:k], name=col, mode="lines",
                line=dict(color=colors[i % len(colors)], width=2.2),
                hovertemplate="%{x|%d %b %Y}<br>" + col + ": %{y:.3f}<extra></extra>",
            )
            for i, col in enumerate(equity.columns)
        ]

    frames = [go.Frame(name=str(i), data=traces(k)) for i, k in enumerate(cuts)]

    ypad = (equity.values.max() - equity.values.min()) * 0.08 + 1e-9
    fig = go.Figure(data=traces(int(cuts[0])), frames=frames)
    fig.update_layout(
        **themed_layout(
            hovermode="x unified",
            height=420,
            yaxis=dict(
                title="Equity (growth of $1)",
                gridcolor=PALETTE["panel_edge"], zeroline=False,
                range=[equity.values.min() - ypad, equity.values.max() + ypad],
            ),
            xaxis=dict(
                gridcolor=PALETTE["panel_edge"], zeroline=False,
                range=[x[0], x[-1]],
            ),
        ),
        updatemenus=[dict(
            type="buttons", direction="left", x=0, y=-0.18,
            xanchor="left", yanchor="top", pad=dict(t=0, r=6),
            showactive=False,
            bgcolor=PALETTE["panel"], bordercolor=PALETTE["panel_edge"],
            font=dict(color=PALETTE["text"]),
            buttons=[
                dict(label="▶ Play", method="animate",
                     args=[None, dict(frame=dict(duration=25, redraw=False),
                                      transition=dict(duration=0),
                                      fromcurrent=True, mode="immediate")]),
                dict(label="⏸ Pause", method="animate",
                     args=[[None], dict(frame=dict(duration=0, redraw=False),
                                        transition=dict(duration=0),
                                        mode="immediate")]),
            ],
        )],
        sliders=[dict(
            x=0.24, y=-0.16, xanchor="left", yanchor="top", len=0.76,
            currentvalue=dict(visible=False), pad=dict(t=6),
            bordercolor=PALETTE["panel_edge"], bgcolor=PALETTE["panel_edge"],
            activebgcolor=PALETTE["accent"], ticklen=0,
            font=dict(color=PALETTE["text_dim"], size=9),
            steps=[dict(
                # sparse labels: a solid block of 60 dates is unreadable
                label=(pd.Timestamp(x[k - 1]).strftime("%b %y")
                       if i % max(1, len(cuts) // 8) == 0 else ""),
                method="animate",
                args=[[str(i)], dict(frame=dict(duration=0, redraw=False),
                                     transition=dict(duration=0),
                                     mode="immediate")],
            ) for i, k in enumerate(cuts)],
        )],
    )
    return fig


def implementation_note(current: str, planned: str) -> None:
    """Small status pill plus caption describing current vs. planned behavior."""
    st.markdown(
        f'<span class="msb-note">Current implementation: {current}</span>',
        unsafe_allow_html=True,
    )
    st.caption(planned)


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

def render_param_widgets(schema: dict) -> dict:
    """Render sidebar widgets purely from a strategy's ``param_schema``.

    The sidebar knows nothing about what the parameters mean, only how
    to turn each schema entry into a widget. Supported types: ``slider``,
    ``number``, ``select``, ``checkbox``.
    """
    params = {}
    for key, spec in schema.items():
        label = spec.get("label", key.replace("_", " ").title())
        kind = spec["type"]
        help_ = spec.get("help")
        if kind == "slider":
            params[key] = st.slider(
                label, spec["min"], spec["max"], spec["default"],
                spec.get("step", 1), help=help_,
            )
        elif kind == "number":
            params[key] = st.number_input(
                label, spec.get("min"), spec.get("max"), spec["default"],
                spec.get("step", 1.0), help=help_,
            )
        elif kind == "select":
            options = spec["options"]
            params[key] = st.selectbox(
                label, options, index=options.index(spec["default"]), help=help_,
            )
        elif kind == "checkbox":
            params[key] = st.checkbox(label, value=spec["default"], help=help_)
        else:
            st.warning(f"Unknown param type {kind!r} for {key!r}, skipped.")
    return params


def render_sidebar() -> dict:
    """Global, engine-owned controls. Never changes when strategies are added."""
    with st.sidebar:
        st.markdown(
            f'<div style="font-weight:800;font-size:1.05rem;letter-spacing:-0.01em;">'
            f'📈 Multi-Strategy<br>Backtester</div>'
            f'<div style="color:{PALETTE["text_dim"]};font-size:0.78rem;'
            f'margin-bottom:0.8rem;">One engine · pluggable strategies</div>',
            unsafe_allow_html=True,
        )

        st.markdown("### Strategy")
        strategy_name = st.selectbox(
            "Strategy", list(STRATEGY_REGISTRY.keys()), label_visibility="collapsed",
        )

        st.markdown("### Data")
        ticker = st.text_input("Ticker", "SPY", help="Yahoo Finance symbol")
        today = dt.date.today()
        col1, col2 = st.columns(2)
        start = col1.date_input("Start", today - dt.timedelta(days=3 * 365))
        end = col2.date_input("End", today)

        st.markdown("### Transaction Costs")
        commission_bps = st.number_input(
            "Commission (bps)", 0.0, 100.0, 1.0, 0.5,
            help="Flat per-trade commission (engine/cost_model.py)",
        )
        slippage_bps = st.number_input(
            "Slippage (bps)", 0.0, 100.0, 2.0, 0.5,
            help="Flat slippage assumption. A market-impact model is "
                 "planned for engine/cost_model.py.",
        )

        st.markdown("### Walk-Forward")
        in_sample_len = st.number_input(
            "In-sample window (days)", 60, 2000, 504, 21,
            help="engine/walk_forward.py (currently a single fixed split)",
        )
        out_sample_len = st.number_input(
            "Out-of-sample window (days)", 21, 1000, 126, 21,
        )
        wf_mode = st.radio(
            "Window mode", ["rolling", "expanding"], horizontal=True,
        )

        st.markdown("### Regime Detection")
        regime_on = st.toggle(
            "Enable regime overlay", value=True,
            help="engine/regime.py (currently a constant 'Normal' label)",
        )

        st.divider()
        st.markdown("### Strategy Parameters")
        strategy = STRATEGY_REGISTRY[strategy_name]
        params = render_param_widgets(strategy.param_schema)

        st.divider()
        st.caption("Cost, walk-forward, and regime modules use baseline "
                   "implementations for now. See the Limitations tab.")

    return dict(
        strategy_name=strategy_name, strategy=strategy, params=params,
        ticker=ticker, start=start, end=end,
        commission_bps=commission_bps, slippage_bps=slippage_bps,
        in_sample_len=int(in_sample_len), out_sample_len=int(out_sample_len),
        wf_mode=wf_mode, regime_on=regime_on,
    )


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

@st.cache_data(ttl=3600, show_spinner=False)
def cached_prices(ticker: str, start: dt.date, end: dt.date) -> pd.DataFrame:
    return load_price_data(ticker, start, end)


def run_pipeline(cfg: dict) -> dict:
    prices = cached_prices(cfg["ticker"], cfg["start"], cfg["end"])
    if prices.empty:
        st.error(
            f"No price data returned for **{cfg['ticker']}** between "
            f"{cfg['start']} and {cfg['end']}. Check the ticker symbol and "
            "your network connection."
        )
        st.stop()

    strategy: BaseStrategy = cfg["strategy"]
    signals = strategy.generate_signals(prices, cfg["params"])
    ledger = compute_pnl(signals, prices)

    net_returns = apply_costs(
        ledger["gross_return"], ledger["trade_size"],
        cfg["commission_bps"], cfg["slippage_bps"],
    )
    ledger["net_return"] = net_returns
    ledger["equity_net"] = (1.0 + net_returns).cumprod()

    return dict(
        prices=prices,
        signals=signals,
        ledger=ledger,
        trades=extract_trades(ledger),
        metrics_gross=compute_metrics(ledger["gross_return"]),
        metrics_net=compute_metrics(net_returns),
        regimes=classify_regime(prices) if cfg["regime_on"] else None,
        windows=split_windows(
            prices, cfg["in_sample_len"], cfg["out_sample_len"], cfg["wf_mode"]
        ),
    )


# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------

def tab_overview(cfg: dict) -> None:
    strategy = cfg["strategy"]
    st.markdown(f"#### {strategy.name}")
    st.markdown(strategy.description)
    st.markdown(
        """
<div class="msb-card">
  <h4>Platform methodology</h4>
  <p>One engine, pluggable strategies. Every strategy plugs into the same
  pipeline (data handler, signals, portfolio accounting, cost model,
  metrics) and is evaluated under four realism pillars: walk-forward
  optimization, transaction costs &amp; slippage, point-in-time data, and
  regime-awareness. Execution is lagged one bar so a signal computed today
  earns tomorrow's return.</p>
</div>
""",
        unsafe_allow_html=True,
    )


def tab_backtest(cfg: dict, results: dict) -> None:
    m = results["metrics_net"]
    render_metric_cards([
        dict(label="Sharpe Ratio (net)", value=m["sharpe"], decimals=2, suffix="",
             color=PALETTE["long"] if m["sharpe"] >= 0 else PALETTE["short"],
             sub="annualized, 0% risk-free"),
        dict(label="Total Return (net)", value=m["total_return"] * 100,
             decimals=1, suffix="%",
             color=PALETTE["long"] if m["total_return"] >= 0 else PALETTE["short"],
             sub=f"CAGR {m['cagr'] * 100:.1f}%"),
        dict(label="Max Drawdown", value=m["max_drawdown"] * 100,
             decimals=1, suffix="%", color=PALETTE["short"],
             sub="peak-to-trough, net equity"),
        dict(label="Hit Rate", value=m["hit_rate"] * 100, decimals=1, suffix="%",
             color=PALETTE["accent"], sub="positive share of active days"),
    ])

    equity = pd.DataFrame({
        "Gross": results["ledger"]["equity_gross"],
        "Net of costs": results["ledger"]["equity_net"],
    })
    render_animated_chart(build_animated_equity_fig(equity), height=540)
    st.caption(
        "Net curve applies the flat basis-point cost model in "
        "`engine/cost_model.py`. A market-impact model is planned for a "
        "future update."
    )

    dd = drawdown_series(results["ledger"]["net_return"])
    fig_dd = go.Figure(go.Scatter(
        x=dd.index, y=dd * 100, mode="lines", fill="tozeroy",
        line=dict(color=PALETTE["short"], width=1.4),
        fillcolor="rgba(255, 92, 122, 0.15)",
        hovertemplate="%{x|%d %b %Y}<br>Drawdown: %{y:.2f}%<extra></extra>",
        name="Drawdown",
    ))
    fig_dd.update_layout(**themed_layout(
        height=220, showlegend=False, title=dict(text="Drawdown (net)", font=dict(size=13)),
        yaxis=dict(title="%", gridcolor=PALETTE["panel_edge"], zeroline=False),
    ))
    st.plotly_chart(fig_dd, width="stretch", config={"displaylogo": False})


def tab_walk_forward(cfg: dict, results: dict) -> None:
    implementation_note(
        "a single fixed 70/30 train/test split.",
        "A rolling/expanding window generator with per-window parameter "
        "re-fitting is planned next. These panels are built to display it "
        "once that lands.",
    )
    windows = results["windows"]
    if not windows:
        st.info("Not enough data to form a train/test split.")
        return

    w = windows[0]
    equity_net = results["ledger"]["equity_net"]
    col_is, col_oos = st.columns(2)
    for col, (label, (seg_start, seg_end), color) in zip(
        (col_is, col_oos),
        [("In-Sample", w["train"], PALETTE["accent"]),
         ("Out-of-Sample", w["test"], PALETTE["long"])],
    ):
        seg = equity_net.loc[seg_start:seg_end]
        seg = seg / seg.iloc[0]
        with col:
            st.markdown(
                f'<div class="msb-card"><h4>{label}</h4>'
                f'<p>{seg_start.date()} → {seg_end.date()} · {len(seg)} days</p></div>',
                unsafe_allow_html=True,
            )
            fig = go.Figure(go.Scatter(
                x=seg.index, y=seg, mode="lines",
                line=dict(color=color, width=2),
                hovertemplate="%{x|%d %b %Y}<br>Equity: %{y:.3f}<extra></extra>",
            ))
            fig.update_layout(**themed_layout(height=300, showlegend=False))
            st.plotly_chart(fig, width="stretch", config={"displaylogo": False})
    st.caption(
        f"Sidebar settings for window length and mode are accepted but not "
        f"yet applied by the current implementation: in-sample "
        f"{cfg['in_sample_len']}d / out-of-sample {cfg['out_sample_len']}d, "
        f"{cfg['wf_mode']} windows."
    )


def tab_regime(cfg: dict, results: dict) -> None:
    implementation_note(
        "every date labeled 'Normal'.",
        "Rolling realized-volatility terciles (or a Markov regime-switching "
        "model) are planned next. This chart is built to display the "
        "breakdown once that lands.",
    )
    if results["regimes"] is None:
        st.info("Regime detection is toggled off in the sidebar.")
        return

    by_regime = (
        results["ledger"]["net_return"]
        .groupby(results["regimes"])
        .agg(ann_return=lambda r: r.mean() * 252 * 100, days="count")
        .reindex(["Low Vol", "Normal", "High Vol"])
        .dropna(how="all")
    )
    fig = go.Figure(go.Bar(
        x=by_regime.index, y=by_regime["ann_return"],
        marker_color=[PALETTE["long"] if v >= 0 else PALETTE["short"]
                      for v in by_regime["ann_return"]],
        customdata=by_regime["days"],
        hovertemplate="%{x}<br>Annualized net return: %{y:.1f}%"
                      "<br>Days in regime: %{customdata}<extra></extra>",
        width=0.45,
    ))
    fig.update_layout(**themed_layout(
        height=360, showlegend=False,
        title=dict(text="Annualized net return by regime", font=dict(size=13)),
        yaxis=dict(title="%", gridcolor=PALETTE["panel_edge"], zeroline=True,
                   zerolinecolor=PALETTE["panel_edge"]),
    ))
    st.plotly_chart(fig, width="stretch", config={"displaylogo": False})


def tab_trades(cfg: dict, results: dict) -> None:
    trades = results["trades"]
    st.markdown(f"#### Trade Log: {len(trades)} position changes")
    if trades.empty:
        st.info("No trades generated over the selected period.")
    else:
        st.dataframe(
            trades.style.map(
                lambda d: (
                    f"color: {PALETTE['long']}" if d == "LONG"
                    else f"color: {PALETTE['short']}" if d == "SHORT"
                    else f"color: {PALETTE['text_dim']}"
                ),
                subset=["Direction"],
            ),
            width="stretch", height=380, hide_index=True,
        )

    st.markdown("#### Strategy Diagnostics")
    strategy: BaseStrategy = cfg["strategy"]
    overrides_hook = (
        type(strategy).custom_diagnostics is not BaseStrategy.custom_diagnostics
    )
    if overrides_hook:
        strategy.custom_diagnostics(results["prices"], cfg["params"], results)
    else:
        st.caption(
            "This strategy defines no custom diagnostics. Strategies may override "
            "`custom_diagnostics()` to render extra charts here."
        )


def tab_limitations() -> None:
    st.markdown(
        """
<div class="msb-card">
  <h4>Methodology & Current Assumptions</h4>
  <p>Cost modeling, walk-forward optimization, and regime detection currently
  use the baseline implementations described below. The moving-average
  crossover strategy is the first entry in the strategy registry, included to
  validate the pipeline end to end; additional strategies are planned next
  (see the README roadmap).</p>
</div>
""",
        unsafe_allow_html=True,
    )
    st.markdown(
        """
**Assumptions & limitations (all strategies)**

- **Execution.** Signals are executed at the next bar's close. No intraday fills,
  partial fills, or liquidity constraints are modeled.
- **Costs.** Currently a flat bps deduction per unit of turnover
  (`engine/cost_model.py`). No market impact, borrow costs, or financing yet.
- **Data.** Plain yfinance daily closes (`engine/data_handler.py`). No
  point-in-time safeguards yet: restated fundamentals, survivorship bias, and
  delisted tickers are not handled.
- **Walk-forward.** Currently a single fixed 70/30 split; parameters are not
  re-fit out-of-sample, so results should be read as in-sample.
- **Regimes.** Every day is currently labeled "Normal"; regime attribution is
  not yet informative.
- **Risk-free rate.** Sharpe ratios assume 0% risk-free.
- **No leverage/margin modeling.** Positions are unit exposures of capital.

Backtested performance is hypothetical and not indicative of future results.
"""
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    st.set_page_config(
        page_title="Multi-Strategy Backtester",
        page_icon="📈",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    inject_css()

    cfg = render_sidebar()

    st.markdown(
        f"""
<div class="msb-header">
  <h1>Multi-Strategy Backtesting Platform</h1>
  <div class="sub">{cfg['strategy_name']} · {cfg['ticker']} ·
      {cfg['start']} → {cfg['end']}</div>
</div>
""",
        unsafe_allow_html=True,
    )

    with st.spinner("Running backtest…"):
        results = run_pipeline(cfg)

    tabs = st.tabs([
        "📖 Overview & Methodology",
        "📊 Backtest Results",
        "🔁 Walk-Forward Analysis",
        "🌡️ Regime Breakdown",
        "🧾 Trade Log & Diagnostics",
        "⚠️ Limitations & Assumptions",
    ])
    with tabs[0]:
        tab_overview(cfg)
    with tabs[1]:
        tab_backtest(cfg, results)
    with tabs[2]:
        tab_walk_forward(cfg, results)
    with tabs[3]:
        tab_regime(cfg, results)
    with tabs[4]:
        tab_trades(cfg, results)
    with tabs[5]:
        tab_limitations()


if __name__ == "__main__":
    main()
