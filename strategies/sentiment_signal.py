"""Sentiment Signal strategy — FinBERT news sentiment as a trading signal.

Ported from the standalone "Sentiment Signal" Streamlit project into the
engine's ``BaseStrategy`` interface. The scoring methodology (FinBERT
primary, VADER comparison baseline), event-day-aligned abnormal returns,
winsorization, and OLS inference all carry over from the original; what
changed is the packaging — per-ticker article trades are aggregated into
one portfolio-level exposure series the engine can turn into an equity
curve, and the Article Explorer / regression diagnostics render through
the ``custom_diagnostics()`` hook instead of their own tabs.
"""

import pandas as pd
import streamlit as st

from engine.base_strategy import BaseStrategy
from strategies import _sentiment_utils as su

# The custom-universe path from the original project (comma-separated
# tickers, Yahoo Finance convention: bare symbols for US listings,
# ".L" suffix for LSE). The sidebar's schema-driven widgets have no
# free-text type, so the custom list lives here as an editable constant,
# selectable via the "Custom" universe option. Capped at
# ``su.MAX_TICKERS`` like every other universe.
CUSTOM_TICKERS = "AAPL, MSFT, AZN.L, HSBA.L"

SENTIMENT_DISCLAIMER = (
    "This is a simplified, illustrative backtest: yfinance news only covers "
    "recent weeks, so the sample is typically well under a few hundred "
    "articles; the engine's cost model is currently a flat-bps stub; and "
    "annualized figures extrapolate a short, noisy sample. It is NOT a "
    "validated trading strategy and should not inform real trading decisions."
)


class SentimentSignalStrategy(BaseStrategy):
    name = "Sentiment Signal"
    description = (
        "Scores live financial news headlines with **FinBERT** "
        "(`ProsusAI/finbert`, a finance-tuned transformer) and trades the "
        "aggregate sentiment of a chosen stock universe. Rebuilt from a "
        "standalone research app that began life as a university coursework "
        "script (VADER on a static 13-stock FTSE 100 sample, R² ≈ 0.02) and "
        "was upgraded with a finance-domain model, event-study abnormal "
        "returns, and honest OLS inference.\n\n"
        "**Signal construction:** every article whose FinBERT score "
        "(P(positive) − P(negative), bounded in [−1, 1]) clears the long "
        "threshold casts a +1 vote for its ticker; below the short threshold, "
        "−1. Votes stay open for the holding window, same-day duplicates per "
        "ticker are averaged, and each day's open votes combine — "
        "equal-weighted or sentiment-magnitude-weighted — into one net "
        "exposure in [−1, +1].\n\n"
        "**How it plugs into this engine:** the engine trades a single "
        "instrument (the ticker in the sidebar), so that aggregate exposure "
        "is applied to it as a market-timing signal — universe-wide news "
        "sentiment breadth timing the selected instrument, rather than a "
        "per-stock long/short book. Days with no qualifying news are flat "
        "(in cash). Because Yahoo Finance only serves recent news (the "
        "lookback window at most), the position is zero outside that "
        "coverage — set the backtest end date to today to see the live "
        "signal.\n\n"
        "⚠️ *Results are illustrative only: small article sample, stub cost "
        "model, single fixed train/test split. See the Limitations tab and "
        "the Diagnostics tab's regression statistics before reading anything "
        "into the equity curve.*"
    )
    param_schema = {
        "universe": {
            "type": "select",
            "options": ["FTSE 100 sample", "S&P 500 sample",
                        "Custom (CUSTOM_TICKERS in strategy file)"],
            "default": "FTSE 100 sample",
            "label": "Ticker universe",
            "help": f"Preset samples of ~12 liquid index constituents (capped at "
                    f"{su.MAX_TICKERS}). The custom option reads the "
                    f"CUSTOM_TICKERS constant in strategies/sentiment_signal.py.",
        },
        "articles_per_ticker": {
            "type": "slider", "min": 3, "max": 25, "default": 8, "step": 1,
            "label": "Articles per ticker",
        },
        "lookback_days": {
            "type": "select", "options": [30, 60, 90], "default": 60,
            "label": "News lookback window (days)",
        },
        "long_threshold": {
            "type": "slider", "min": 0.0, "max": 1.0, "default": 0.3, "step": 0.05,
            "label": "Long threshold (sentiment ≥)",
        },
        "short_threshold": {
            "type": "slider", "min": -1.0, "max": 0.0, "default": -0.3, "step": 0.05,
            "label": "Short threshold (sentiment ≤)",
        },
        "holding_days": {
            "type": "slider", "min": 1, "max": 5, "default": 3, "step": 1,
            "label": "Holding window (trading days)",
        },
        "sizing": {
            "type": "select",
            "options": ["Equal-weight", "Sentiment-magnitude-weighted"],
            "default": "Equal-weight",
            "label": "Position sizing",
            "help": "Equal-weight: every qualifying vote counts ±1. "
                    "Magnitude-weighted: votes scale with |FinBERT score|.",
        },
    }

    # -- helpers ------------------------------------------------------------

    @staticmethod
    def _resolve_universe(choice: str) -> list:
        if choice.startswith("FTSE"):
            return list(su.FTSE100_SAMPLE)[:su.MAX_TICKERS]
        if choice.startswith("S&P"):
            return list(su.SP500_SAMPLE)[:su.MAX_TICKERS]
        return su.parse_custom_tickers(CUSTOM_TICKERS)[:su.MAX_TICKERS]

    def _scored_articles(self, params: dict) -> pd.DataFrame:
        """Run (or hit the cache of) the news → FinBERT/VADER → abnormal
        returns pipeline for the current params."""
        universe = self._resolve_universe(params["universe"])
        if not universe:
            st.error("The custom ticker list is empty — edit CUSTOM_TICKERS in "
                     "strategies/sentiment_signal.py or pick a preset universe.")
            st.stop()
        try:
            return su.run_sentiment_pipeline(
                tuple(universe),
                int(params["articles_per_ticker"]),
                int(params["lookback_days"]),
            )
        except su.MissingDependencyError as e:
            st.error(str(e))
            st.stop()

    # -- BaseStrategy interface ----------------------------------------------

    def generate_signals(self, data: pd.DataFrame, params: dict) -> pd.DataFrame:
        articles = self._scored_articles(params)

        signals = pd.DataFrame(index=data.index)
        if articles.empty:
            st.warning(
                "No news articles found for this universe/lookback window — "
                "the strategy stays flat. Try a larger universe or lookback."
            )
            signals["position"] = 0.0
            return signals

        position = su.build_position_series(
            articles, data.index,
            long_threshold=float(params["long_threshold"]),
            short_threshold=float(params["short_threshold"]),
            holding_days=int(params["holding_days"]),
            magnitude_weighted=(params["sizing"] == "Sentiment-magnitude-weighted"),
        )
        if (position == 0.0).all():
            st.warning(
                "The sentiment signal is flat over the entire backtest range. "
                "Either no article cleared the long/short thresholds (try "
                "narrowing them), or the news window doesn't overlap the "
                "backtest dates — yfinance only serves recent news, so the "
                "backtest end date must be at (or near) today."
            )
        signals["position"] = position
        return signals

    def custom_diagnostics(self, data: pd.DataFrame, params: dict, results: dict) -> None:
        """The original project's research surface: Article Explorer,
        sentiment-vs-abnormal-returns OLS, and FinBERT-vs-VADER
        comparison — rendered inside the shell's Diagnostics tab."""
        articles = self._scored_articles(params)
        if articles.empty:
            st.info("No scored articles available — nothing to diagnose.")
            return

        st.error(f"⚠️ **{SENTIMENT_DISCLAIMER}**")
        n_skipped = articles.attrs.get("n_skipped", 0)
        st.caption(
            f"{len(articles)} scored articles across "
            f"{articles['ticker'].nunique()} tickers, event days "
            f"{articles['event_date'].min().date()} → "
            f"{articles['event_date'].max().date()}"
            + (f" ({n_skipped} articles dropped for missing/insufficient "
               f"price history)." if n_skipped else ".")
        )

        self._render_article_explorer(articles)
        st.divider()
        self._render_regression(articles)
        st.divider()
        self._render_model_comparison(articles)

    # -- diagnostics sections -------------------------------------------------

    @staticmethod
    def _render_article_explorer(articles: pd.DataFrame) -> None:
        st.markdown("##### Article Explorer")
        f1, f2, f3 = st.columns([1.4, 1, 1.2])
        with f1:
            ticker_filter = st.multiselect(
                "Ticker", sorted(articles["ticker"].unique()), key="ss_exp_ticker")
        with f2:
            label_filter = st.selectbox(
                "FinBERT label", ["All", "positive", "negative", "neutral"],
                key="ss_exp_label")
        with f3:
            disagreements_only = st.checkbox(
                "Disagreements only", key="ss_exp_disagree",
                help="Articles where FinBERT and VADER assign opposite "
                     "positive/negative directions — expected on ambiguous "
                     "financial text, and informative about where a general "
                     "lexicon breaks down.")

        filtered = articles
        if ticker_filter:
            filtered = filtered[filtered["ticker"].isin(ticker_filter)]
        if label_filter != "All":
            filtered = filtered[filtered["finbert_label"] == label_filter]
        if disagreements_only:
            fin_dir = filtered["finbert_label"].map(
                {"positive": 1, "negative": -1}).fillna(0)
            vad_dir = filtered["vader_label"].map(
                {"positive": 1, "negative": -1}).fillna(0)
            filtered = filtered[(fin_dir != 0) & (vad_dir != 0) & (fin_dir != vad_dir)]

        display_cols = {
            "ticker": "Ticker", "headline": "Headline",
            "publish_date": "Published", "finbert_label": "FinBERT Label",
            "finbert_score": "FinBERT Score", "finbert_confidence": "Confidence",
            "vader_compound": "VADER Compound",
            "abnormal_return_next_day": "Abn. Return (next-day)",
        }
        available = {k: v for k, v in display_cols.items() if k in filtered.columns}
        table = filtered[list(available)].rename(columns=available)
        if "Published" in table.columns:
            table["Published"] = (
                pd.to_datetime(table["Published"], utc=True)
                .dt.tz_localize(None).dt.strftime("%Y-%m-%d %H:%M")
            )
        st.dataframe(
            table, width="stretch", height=320, hide_index=True,
            column_config={
                "Confidence": st.column_config.ProgressColumn(
                    min_value=0, max_value=1, format="%.2f"),
                "FinBERT Score": st.column_config.NumberColumn(format="%.3f"),
                "VADER Compound": st.column_config.NumberColumn(format="%.3f"),
                "Abn. Return (next-day)": st.column_config.NumberColumn(format="percent"),
            },
        )
        st.caption(f"Showing {len(filtered)} of {len(articles)} articles.")

        most_pos = articles.nlargest(2, "finbert_score")
        most_neg = articles.nsmallest(2, "finbert_score")
        col_pos, col_neg = st.columns(2)
        for col, subset, title in ((col_pos, most_pos, "Most positive (FinBERT)"),
                                   (col_neg, most_neg, "Most negative (FinBERT)")):
            with col:
                st.markdown(f"**{title}**")
                for _, row in subset.iterrows():
                    with st.expander(f"{row['ticker']}: {str(row['headline'])[:70]}"):
                        st.write(row.get("summary") or "*(no summary available)*")
                        st.write(
                            f"FinBERT: **{row['finbert_label']}** "
                            f"(score={row['finbert_score']:.3f}, "
                            f"confidence={row['finbert_confidence']:.2f}) | "
                            f"VADER compound: {row['vader_compound']:.3f}"
                        )

    @staticmethod
    def _render_regression(articles: pd.DataFrame) -> None:
        st.markdown("##### Sentiment vs. abnormal returns (OLS)")
        st.caption(
            "Abnormal return = stock return − benchmark return over the same "
            "window (FTSE 100 for .L names, S&P 500 otherwise) — stripping out "
            "market-wide moves gives firm-specific news a fair chance to show a "
            "relationship."
        )
        c1, c2 = st.columns([1.2, 1])
        with c1:
            window = st.selectbox(
                "Return window", su.WINDOWS,
                format_func=lambda w: su.WINDOW_LABELS[w], index=1,
                key="ss_reg_window")
        with c2:
            winsorize_on = st.toggle(
                "Winsorize returns (1st/99th pct)", value=True, key="ss_reg_wz",
                help="Caps (not deletes) extreme returns so a single outlier "
                     "can't dominate the squared-error fit.")

        abnormal_col = f"abnormal_return_{window}"
        working = articles.dropna(subset=[abnormal_col, "finbert_score"]).copy()
        if working.empty:
            st.info("No articles with usable returns for this window.")
            return
        y = (su.winsorize_series(working[abnormal_col])
             if winsorize_on else working[abnormal_col])

        reg = su.run_ols(working["finbert_score"], y)
        st.plotly_chart(
            su.sentiment_return_scatter(
                working["finbert_score"], y, working["ticker"],
                working["headline"], reg,
                x_label="FinBERT sentiment score (P(positive) − P(negative))",
                y_label=f"Abnormal return ({su.WINDOW_LABELS[window]})",
            ),
            width="stretch", config={"displaylogo": False},
        )
        if reg.error:
            st.warning(reg.error)
            return

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Slope", f"{reg.slope:.4f}")
        m2.metric("R²", f"{reg.r_squared:.4f}")
        m3.metric("t-stat (slope)", f"{reg.slope_tstat:.3f}")
        m4.metric("N (articles)", f"{reg.n_obs}")
        m5, m6, m7, m8 = st.columns(4)
        m5.metric("p-value (slope)", f"{reg.slope_pvalue:.4f}")
        m6.metric("95% CI (slope)", f"[{reg.slope_ci_low:.4f}, {reg.slope_ci_high:.4f}]")
        m7.metric("F-stat p-value", f"{reg.f_pvalue:.4f}")
        m8.metric("Intercept", f"{reg.intercept:.4f}")

        if reg.is_significant:
            st.success(su.significance_statement(reg))
        else:
            st.warning(su.significance_statement(reg))

    @staticmethod
    def _render_model_comparison(articles: pd.DataFrame) -> None:
        st.markdown("##### FinBERT vs. VADER")
        st.caption(
            "The original project's core comparison: does the finance-tuned "
            "transformer actually fit abnormal returns better than a "
            "general-purpose lexicon model?"
        )
        abnormal_col = "abnormal_return_next_day"
        diag = articles.dropna(subset=[abnormal_col, "finbert_score", "vader_compound"])
        reg_fin = su.run_ols(diag["finbert_score"], diag[abnormal_col])
        reg_vad = su.run_ols(diag["vader_compound"], diag[abnormal_col])

        col_a, col_b = st.columns(2)
        for col, reg, label in ((col_a, reg_fin, "FinBERT"), (col_b, reg_vad, "VADER")):
            with col:
                st.markdown(f"**{label}-based regression (next-day abnormal return)**")
                if reg.error:
                    st.warning(reg.error)
                else:
                    st.write(
                        f"Slope: `{reg.slope:.4f}` | R²: `{reg.r_squared:.4f}` | "
                        f"|t|: `{abs(reg.slope_tstat):.3f}` | "
                        f"p: `{reg.slope_pvalue:.4f}` | N: `{reg.n_obs}`"
                    )

        crosstab = pd.crosstab(articles["finbert_label"], articles["vader_label"])
        st.plotly_chart(su.agreement_heatmap(crosstab), width="stretch",
                        config={"displaylogo": False})
        agree = (articles["finbert_label"] == articles["vader_label"]).mean()
        st.caption(
            f"Exact label agreement: **{agree:.1%}**. Disagreement is expected — "
            "VADER is a general-purpose lexicon with no notion of "
            "financial-domain meaning, while FinBERT was fine-tuned on "
            "analyst-labelled financial text."
        )
