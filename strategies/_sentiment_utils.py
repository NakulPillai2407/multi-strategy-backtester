"""Helper pipeline for the Sentiment Signal strategy.

Ported from the standalone "Sentiment Signal" project (FinBERT news
scoring, event-day-aligned abnormal returns, winsorization, statsmodels
OLS diagnostics). The methodology carries over intact; what changed is
the packaging — everything here is driven by strategy params instead of
its own Streamlit tabs, and the heavy stages (news fetch, FinBERT
scoring, price pulls) are cached so the engine can call
``generate_signals`` on every widget change without re-hitting the
network or the model.

Trimmed relative to the original on purpose: the market-model
(beta-adjusted) abnormal-return variant is not ported — abnormal return
here is always flat benchmark subtraction (raw − benchmark over the same
window). The beta-adjusted variant can be re-ported later without
touching the strategy interface.

Heavy dependencies (transformers/torch for FinBERT, nltk for VADER,
statsmodels for OLS) are imported lazily inside the functions that need
them, so merely registering the strategy never slows down or breaks app
startup for users who haven't installed them.
"""

from dataclasses import dataclass
from datetime import timedelta

import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf


class MissingDependencyError(RuntimeError):
    """Raised when an optional heavy dependency isn't installed."""


# ---------------------------------------------------------------------------
# Universe presets (static samples of the original project's bundled
# FTSE 100 / S&P 500 constituent lists — large, liquid, well-covered names
# so yfinance news coverage is decent)
# ---------------------------------------------------------------------------

MAX_TICKERS = 15  # cap ported from the original's yfinance rate-limit guard

FTSE100_SAMPLE = (
    "AZN.L", "HSBA.L", "SHEL.L", "ULVR.L", "BP.L", "GSK.L",
    "RIO.L", "LLOY.L", "BARC.L", "TSCO.L", "VOD.L", "BA.L",
)
SP500_SAMPLE = (
    "AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META",
    "JPM", "JNJ", "XOM", "PG", "KO", "UNH",
)

# Benchmark inference, ported: LSE listings (".L") net out the FTSE 100,
# everything else the S&P 500. A heuristic, not a classification.
def infer_benchmark_for_ticker(ticker: str) -> str:
    return "^FTSE" if ticker.upper().strip().endswith(".L") else "^GSPC"


def parse_custom_tickers(raw_text: str) -> list:
    """Split a comma/newline-separated string into upper-cased, de-duped
    tickers (order-preserving). Ported verbatim from the original."""
    if not raw_text:
        return []
    tokens = [t.strip().upper() for t in raw_text.replace("\n", ",").split(",")]
    out, seen = [], set()
    for t in tokens:
        if t and t not in seen:
            seen.add(t)
            out.append(t)
    return out


# ---------------------------------------------------------------------------
# News retrieval (defensive parsing across yfinance's changing news schema)
# ---------------------------------------------------------------------------

NEWS_COLUMNS = ["ticker", "headline", "summary", "publish_date", "link"]


def _extract_field(article: dict, *candidates, default=""):
    """First present field among possible key names — handles both the
    flat and nested ('content') yfinance news schemas."""
    content = article.get("content", article)
    for key in candidates:
        if key in content and content[key] not in (None, ""):
            val = content[key]
            if isinstance(val, dict):  # e.g. {'canonicalUrl': {'url': ...}}
                val = val.get("url") or val.get("value") or ""
            return val
    return default


def _extract_publish_date(article: dict):
    """Normalize the publish timestamp (epoch int in older yfinance,
    ISO string in newer) to a tz-aware pandas Timestamp."""
    content = article.get("content", article)
    if "providerPublishTime" in content:
        try:
            return pd.Timestamp(content["providerPublishTime"], unit="s", tz="UTC")
        except Exception:
            return None
    for key in ("pubDate", "displayTime"):
        if key in content and content[key]:
            try:
                ts = pd.Timestamp(content[key])
                return ts.tz_localize("UTC") if ts.tzinfo is None else ts
            except Exception:
                continue
    return None


def _fetch_news_for_ticker(ticker: str, max_articles: int, lookback_days: int) -> pd.DataFrame:
    cutoff = pd.Timestamp.now(tz="UTC") - timedelta(days=lookback_days)
    try:
        raw = yf.Ticker(ticker).news or []
    except Exception:
        return pd.DataFrame(columns=NEWS_COLUMNS)

    rows = []
    for article in raw:
        pub_date = _extract_publish_date(article)
        if pub_date is None or pub_date < cutoff:
            continue
        headline = _extract_field(article, "title")
        if not headline:  # no headline = nothing to score
            continue
        rows.append({
            "ticker": ticker,
            "headline": headline,
            "summary": _extract_field(article, "summary", "description"),
            "publish_date": pub_date,
            "link": _extract_field(article, "canonicalUrl", "link", "clickThroughUrl"),
        })
    if not rows:
        return pd.DataFrame(columns=NEWS_COLUMNS)
    return pd.DataFrame(rows[:max_articles], columns=NEWS_COLUMNS)


# ---------------------------------------------------------------------------
# Sentiment scoring — FinBERT (primary) and VADER (comparison baseline)
# ---------------------------------------------------------------------------

FINBERT_MODEL = "ProsusAI/finbert"
# Logit order fixed by the model's config (id2label) — must match exactly.
FINBERT_LABELS = ["positive", "negative", "neutral"]


@st.cache_resource(show_spinner=False)
def _load_finbert():
    """Load and cache FinBERT (~400MB first download). Resource cache:
    the return value is a live model object, not serializable data."""
    try:
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
    except ImportError as e:
        raise MissingDependencyError(
            "The Sentiment Signal strategy needs `transformers` and `torch` "
            "for FinBERT scoring — run `pip install transformers torch` "
            "(they're listed in requirements.txt)."
        ) from e
    tokenizer = AutoTokenizer.from_pretrained(FINBERT_MODEL)
    model = AutoModelForSequenceClassification.from_pretrained(FINBERT_MODEL)
    model.eval()
    return tokenizer, model


def _prepare_text(headline, summary) -> str:
    """Headline + summary concatenated (summary alone loses headline-only
    signal; headline alone loses the summary's richness). NaN-safe."""
    headline = "" if pd.isna(headline) else str(headline).strip()
    summary = "" if pd.isna(summary) else str(summary).strip()
    if summary:
        return f"{headline}. {summary}" if headline else summary
    return headline


def score_articles_finbert(df: pd.DataFrame, batch_size: int = 8) -> pd.DataFrame:
    """Add finbert_{positive,negative,neutral,label,confidence,score}.

    ``finbert_score`` = P(positive) − P(negative) in [−1, 1] — the
    continuous signal used for both trading and regression. It beats a
    bare P(positive) because a mostly-neutral article and a genuine
    50/50 positive-vs-negative split are very different signals.
    """
    if df.empty:
        return df.copy()
    try:
        import torch
    except ImportError as e:
        raise MissingDependencyError(
            "The Sentiment Signal strategy needs `torch` — run "
            "`pip install transformers torch`."
        ) from e

    tokenizer, model = _load_finbert()
    texts = [_prepare_text(h, s) for h, s in zip(df["headline"], df["summary"])]

    all_probs = []
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            inputs = tokenizer(texts[i:i + batch_size], padding=True,
                               truncation=True, max_length=512, return_tensors="pt")
            probs = torch.softmax(model(**inputs).logits, dim=-1).numpy()
            all_probs.append(probs)
    probs = np.vstack(all_probs)

    out = df.copy()
    for idx, label in enumerate(FINBERT_LABELS):
        out[f"finbert_{label}"] = probs[:, idx]
    prob_cols = [f"finbert_{l}" for l in FINBERT_LABELS]
    out["finbert_label"] = out[prob_cols].idxmax(axis=1).str.replace("finbert_", "")
    out["finbert_confidence"] = out[prob_cols].max(axis=1)
    out["finbert_score"] = out["finbert_positive"] - out["finbert_negative"]
    return out


def score_articles_vader(df: pd.DataFrame) -> pd.DataFrame:
    """Add vader_compound / vader_label — retained purely so the
    diagnostics can compare a general-purpose lexicon model against the
    finance-tuned transformer, as in the original project."""
    if df.empty:
        return df.copy()
    try:
        import nltk
        from nltk.sentiment.vader import SentimentIntensityAnalyzer
    except ImportError as e:
        raise MissingDependencyError(
            "The Sentiment Signal strategy needs `nltk` for the VADER "
            "comparison baseline — run `pip install nltk`."
        ) from e
    try:
        nltk.data.find("sentiment/vader_lexicon.zip")
    except LookupError:
        nltk.download("vader_lexicon", quiet=True)
    analyzer = SentimentIntensityAnalyzer()

    def _compound(headline, summary):
        text = _prepare_text(headline, summary)
        return analyzer.polarity_scores(text)["compound"] if text else 0.0

    out = df.copy()
    out["vader_compound"] = [_compound(h, s) for h, s in zip(out["headline"], out["summary"])]
    out["vader_label"] = out["vader_compound"].apply(
        lambda c: "positive" if c >= 0.05 else ("negative" if c <= -0.05 else "neutral")
    )
    return out


# ---------------------------------------------------------------------------
# Event-day alignment + abnormal returns (trimmed port of analysis/returns.py)
# ---------------------------------------------------------------------------

WINDOWS = ["same_day", "next_day", "3day"]
WINDOW_LABELS = {
    "same_day": "Same-day (close-to-close)",
    "next_day": "Next-day",
    "3day": "3-day cumulative",
}
PAD_DAYS = 10


def _find_event_position(index: pd.DatetimeIndex, pub_date) -> int | None:
    """Position of the first trading day >= the article's publish date
    (standard event-study convention: a weekend article anchors to
    Monday). None if publish date is beyond available prices."""
    ts = pd.Timestamp(pub_date)
    pub_day = (ts.tz_localize(None) if ts.tzinfo else ts).normalize()
    pos = index.searchsorted(pub_day, side="left")
    return int(pos) if pos < len(index) else None


def _window_return(closes: np.ndarray, event_pos: int, window: str) -> float:
    n = len(closes)
    if window == "same_day":
        return closes[event_pos] / closes[event_pos - 1] - 1.0 if event_pos >= 1 else np.nan
    if window == "next_day":
        return closes[event_pos + 1] / closes[event_pos] - 1.0 if event_pos + 1 < n else np.nan
    if window == "3day":
        return closes[event_pos + 3] / closes[event_pos] - 1.0 if event_pos + 3 < n else np.nan
    raise ValueError(f"Unknown window {window!r}")


def _fetch_close_history(symbol: str, start: str, end: str) -> pd.Series:
    try:
        hist = yf.Ticker(symbol).history(start=start, end=end, interval="1d", auto_adjust=True)
    except Exception:
        return pd.Series(dtype="float64")
    if hist.empty:
        return pd.Series(dtype="float64")
    closes = hist["Close"]
    closes.index = pd.to_datetime(closes.index).tz_localize(None)
    return closes


def compute_event_returns(articles: pd.DataFrame, lookback_days: int) -> pd.DataFrame:
    """Attach event_date plus raw/benchmark/abnormal returns (all three
    windows, flat benchmark subtraction) to every article.

    Articles whose windows fall outside available price history are
    dropped — mirroring the original pipeline's transparent data-loss
    accounting, the drop count is reported via the returned frame's
    ``attrs["n_skipped"]``.
    """
    if articles.empty:
        return articles.copy()

    end = pd.Timestamp.now().normalize() + timedelta(days=1 + PAD_DAYS)
    start = end - timedelta(days=lookback_days + 2 * PAD_DAYS + 5)
    start_s, end_s = start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d")

    tickers = list(articles["ticker"].unique())
    benchmarks = {infer_benchmark_for_ticker(t) for t in tickers}
    closes = {s: _fetch_close_history(s, start_s, end_s)
              for s in list(dict.fromkeys(tickers + sorted(benchmarks)))}

    rows, skipped = [], 0
    for _, article in articles.iterrows():
        stock = closes.get(article["ticker"], pd.Series(dtype="float64"))
        bench = closes.get(infer_benchmark_for_ticker(article["ticker"]),
                           pd.Series(dtype="float64"))
        if stock.empty or bench.empty:
            skipped += 1
            continue
        event_pos = _find_event_position(stock.index, article["publish_date"])
        if event_pos is None:
            skipped += 1
            continue
        event_date = stock.index[event_pos]
        bench_pos = _find_event_position(bench.index, event_date)
        if bench_pos is None:
            skipped += 1
            continue

        row = {**article.to_dict(), "event_date": event_date}
        any_ok = False
        for window in WINDOWS:
            raw_r = _window_return(stock.values, event_pos, window)
            bench_r = _window_return(bench.values, bench_pos, window)
            row[f"raw_return_{window}"] = raw_r
            if np.isnan(raw_r) or np.isnan(bench_r):
                row[f"abnormal_return_{window}"] = np.nan
            else:
                row[f"abnormal_return_{window}"] = raw_r - bench_r
                any_ok = True
        if not any_ok:
            skipped += 1
            continue
        rows.append(row)

    out = pd.DataFrame(rows)
    out.attrs["n_skipped"] = skipped
    return out


def winsorize_series(series: pd.Series, lower_pct: float = 0.01,
                     upper_pct: float = 0.99) -> pd.Series:
    """Cap extremes at the given percentiles rather than deleting rows —
    keeps every observation (sample size matters at ~100 articles) while
    stopping a single 40% M&A move from dominating a squared-error fit."""
    if series.empty or series.dropna().empty:
        return series
    return series.clip(lower=series.quantile(lower_pct),
                       upper=series.quantile(upper_pct))


# ---------------------------------------------------------------------------
# Cached end-to-end pipeline
# ---------------------------------------------------------------------------

@st.cache_data(ttl=3600, show_spinner=False)
def run_sentiment_pipeline(universe: tuple, articles_per_ticker: int,
                           lookback_days: int) -> pd.DataFrame:
    """News → FinBERT + VADER scores → event-aligned abnormal returns.

    Cached (1h TTL) on the exact universe/window arguments so the engine
    can re-invoke ``generate_signals`` on every slider change for free —
    only threshold/sizing logic reruns, never the fetch or the model.
    """
    frames = [_fetch_news_for_ticker(t, articles_per_ticker, lookback_days)
              for t in universe]
    frames = [f for f in frames if not f.empty]
    if not frames:
        return pd.DataFrame(columns=NEWS_COLUMNS)
    articles = pd.concat(frames, ignore_index=True)
    articles = score_articles_finbert(articles)
    articles = score_articles_vader(articles)
    return compute_event_returns(articles, lookback_days)


# ---------------------------------------------------------------------------
# Signal aggregation: article votes → one portfolio-level exposure series
# ---------------------------------------------------------------------------

def build_position_series(articles: pd.DataFrame, index: pd.DatetimeIndex,
                          long_threshold: float, short_threshold: float,
                          holding_days: int, magnitude_weighted: bool) -> pd.Series:
    """Aggregate per-article sentiment votes into a single net-exposure
    series in [−1, +1] on the traded instrument's calendar.

    Mechanics (ported from the original backtest, then aggregated one
    level further so the engine gets one series instead of per-ticker
    trades):

    * an article with finbert_score ≥ ``long_threshold`` casts a +1 vote
      for its ticker, ≤ ``short_threshold`` a −1 vote; in-between casts
      nothing. Magnitude weighting scales the vote by |score|.
    * multiple qualifying articles for one ticker on one event day
      collapse to their mean vote (one noisy headline shouldn't be
      double-counted because a second one published the same day).
    * each vote stays open for ``holding_days`` trading days from its
      event day; overlapping votes for the same ticker average.
    * the portfolio exposure each day is the equal-weighted mean of that
      day's active ticker votes — the original's "average across every
      ticker with an open signal" — and 0 (flat, in cash) on days with
      no open votes.
    """
    flat = pd.Series(0.0, index=index, name="position")
    if articles.empty or len(index) == 0:
        return flat

    df = articles.dropna(subset=["event_date", "finbert_score", "ticker"]).copy()
    df["signal"] = 0.0
    df.loc[df["finbert_score"] >= long_threshold, "signal"] = 1.0
    df.loc[df["finbert_score"] <= short_threshold, "signal"] = -1.0
    df = df[df["signal"] != 0.0]
    if df.empty:
        return flat
    df["vote"] = (df["signal"] * df["finbert_score"].abs()
                  if magnitude_weighted else df["signal"])

    per_ticker_day = (
        df.groupby(["ticker", "event_date"])["vote"].mean().reset_index()
    )

    n = len(index)
    sums: dict = {}
    counts: dict = {}
    for _, row in per_ticker_day.iterrows():
        # Event dates live on each ticker's own trading calendar; map to
        # the traded instrument's calendar via first-date-on-or-after.
        pos0 = int(index.searchsorted(pd.Timestamp(row["event_date"]), side="left"))
        if pos0 >= n:
            continue
        t = row["ticker"]
        if t not in sums:
            sums[t] = np.zeros(n)
            counts[t] = np.zeros(n)
        end = min(pos0 + int(holding_days), n)
        sums[t][pos0:end] += row["vote"]
        counts[t][pos0:end] += 1.0
    if not sums:
        return flat

    ticker_votes = np.vstack([
        np.divide(sums[t], counts[t], out=np.zeros(n), where=counts[t] > 0)
        for t in sums
    ])
    active = np.vstack([counts[t] > 0 for t in sums])
    n_active = active.sum(axis=0)
    exposure = np.divide(ticker_votes.sum(axis=0), n_active,
                         out=np.zeros(n), where=n_active > 0)
    return pd.Series(exposure, index=index, name="position")


# ---------------------------------------------------------------------------
# OLS diagnostics (trimmed port of analysis/regression.py)
# ---------------------------------------------------------------------------

@dataclass
class RegressionResult:
    n_obs: int
    intercept: float
    slope: float
    slope_se: float
    slope_tstat: float
    slope_pvalue: float
    slope_ci_low: float
    slope_ci_high: float
    r_squared: float
    f_pvalue: float
    is_significant: bool
    alpha: float = 0.05
    error: str | None = None


def _empty_regression(error: str) -> RegressionResult:
    return RegressionResult(
        n_obs=0, intercept=np.nan, slope=np.nan, slope_se=np.nan,
        slope_tstat=np.nan, slope_pvalue=np.nan, slope_ci_low=np.nan,
        slope_ci_high=np.nan, r_squared=np.nan, f_pvalue=np.nan,
        is_significant=False, error=error,
    )


def run_ols(x: pd.Series, y: pd.Series, alpha: float = 0.05) -> RegressionResult:
    """OLS with full inference (statsmodels, not sklearn): standard
    errors, t-stats, p-values, and CIs — needed to answer "is this
    distinguishable from noise?", which a slope and R² alone cannot."""
    try:
        import statsmodels.api as sm
    except ImportError:
        return _empty_regression(
            "statsmodels is not installed — run `pip install statsmodels` "
            "to enable the OLS diagnostics."
        )

    paired = pd.concat([x.rename("x"), y.rename("y")], axis=1).dropna()
    if len(paired) < 3:
        return _empty_regression(
            f"Not enough paired observations for a regression "
            f"(have {len(paired)}, need ≥ 3)."
        )

    model = sm.OLS(paired["y"], sm.add_constant(paired["x"])).fit()
    slope_pvalue = float(model.pvalues.get("x", np.nan))
    ci = model.conf_int(alpha=alpha)
    slope_ci = ci.loc["x"] if "x" in ci.index else (np.nan, np.nan)

    return RegressionResult(
        n_obs=int(model.nobs),
        intercept=float(model.params.get("const", np.nan)),
        slope=float(model.params.get("x", np.nan)),
        slope_se=float(model.bse.get("x", np.nan)),
        slope_tstat=float(model.tvalues.get("x", np.nan)),
        slope_pvalue=slope_pvalue,
        slope_ci_low=float(slope_ci[0]),
        slope_ci_high=float(slope_ci[1]),
        r_squared=float(model.rsquared),
        f_pvalue=float(model.f_pvalue) if model.f_pvalue is not None else np.nan,
        is_significant=bool(slope_pvalue < alpha) if not np.isnan(slope_pvalue) else False,
        alpha=alpha,
    )


def significance_statement(result: RegressionResult) -> str:
    """Honest plain-language summary — worded so it can't be quoted out
    of context as 'sentiment predicts returns' when the data doesn't
    support that."""
    if result.error:
        return result.error
    if result.is_significant:
        direction = "positive" if result.slope > 0 else "negative"
        return (
            f"Statistically significant at the {result.alpha:.0%} level "
            f"(p = {result.slope_pvalue:.4f}). The estimated relationship is "
            f"{direction}, but statistical significance alone does not establish "
            f"economic significance, causality, or out-of-sample predictive value."
        )
    return (
        f"NOT statistically significant at the {result.alpha:.0%} level "
        f"(p = {result.slope_pvalue:.4f}). We cannot reject the possibility that "
        f"the true sentiment–return relationship is zero; any apparent trend in "
        f"the scatter should be treated as noise at this sample size."
    )


# ---------------------------------------------------------------------------
# Diagnostics chart builders (dark-themed for this app; ported from the
# original's plotly_white charts)
# ---------------------------------------------------------------------------

# Mirrors app.py's PALETTE — duplicated because strategies cannot import
# from app.py (app imports strategies; it would be circular).
_DARK = {
    "bg": "rgba(0,0,0,0)", "edge": "#1E2638", "text": "#E6EAF2",
    "dim": "#8B93A7", "long": "#00D09C", "short": "#FF5C7A",
    "accent": "#4F8CFF", "gold": "#F5B759", "panel": "#111827",
}


def _dark_layout(**overrides) -> dict:
    layout = dict(
        paper_bgcolor=_DARK["bg"], plot_bgcolor=_DARK["bg"],
        font=dict(color=_DARK["text"], size=12),
        hoverlabel=dict(bgcolor=_DARK["panel"], bordercolor=_DARK["edge"],
                        font=dict(color=_DARK["text"])),
        xaxis=dict(gridcolor=_DARK["edge"], zeroline=False),
        yaxis=dict(gridcolor=_DARK["edge"], zeroline=False),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
        margin=dict(t=40, l=10, r=10, b=10),
    )
    layout.update(overrides)
    return layout


def sentiment_return_scatter(x: pd.Series, y: pd.Series, hover_ticker: pd.Series,
                             hover_headline: pd.Series, regression: RegressionResult,
                             x_label: str, y_label: str):
    """Scatter with the OLS fit drawn from the *same* fitted intercept/
    slope as the printed statistics, so chart and numbers can't drift."""
    import plotly.graph_objects as go

    headlines = hover_headline.astype(str).str.slice(0, 90)
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=x, y=y, mode="markers",
        marker=dict(size=8, color=_DARK["accent"], opacity=0.65,
                    line=dict(width=0.5, color=_DARK["edge"])),
        text=[f"<b>{t}</b><br>{h}" for t, h in zip(hover_ticker, headlines)],
        hovertemplate="%{text}<br>Sentiment: %{x:.3f}<br>Return: %{y:.2%}<extra></extra>",
        name="Articles",
    ))
    if not regression.error and len(x.dropna()) > 0:
        x_range = np.linspace(float(x.min()), float(x.max()), 50)
        fig.add_trace(go.Scatter(
            x=x_range, y=regression.intercept + regression.slope * x_range,
            mode="lines", line=dict(color=_DARK["gold"], width=3),
            name=f"OLS fit (slope={regression.slope:.4f})", hoverinfo="skip",
        ))
    fig.update_layout(**_dark_layout(
        xaxis_title=x_label, yaxis_title=y_label,
        yaxis=dict(tickformat=".1%", gridcolor=_DARK["edge"], zeroline=False),
        hovermode="closest", height=420,
    ))
    return fig


def agreement_heatmap(crosstab: pd.DataFrame):
    """FinBERT-vs-VADER label crosstab — a confusion-matrix stand-in
    with no ground truth, just two models that may or may not agree."""
    import plotly.express as px

    fig = px.imshow(
        crosstab.values, x=list(crosstab.columns), y=list(crosstab.index),
        text_auto=True, color_continuous_scale="Teal", aspect="auto",
    )
    fig.update_layout(**_dark_layout(
        xaxis_title="VADER label", yaxis_title="FinBERT label",
        coloraxis_showscale=False, height=320,
    ))
    return fig
