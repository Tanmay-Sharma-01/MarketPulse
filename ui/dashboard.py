"""Streamlit dashboard layout and visualizations."""

from __future__ import annotations

import sys
from html import escape
from typing import Any

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from ui.components import header_block, news_card, sentiment_gauge, status_badge
from utils.helpers import format_currency, format_percent


def load_custom_css() -> None:
    """Inject custom stylesheet."""
    from utils.config import STYLE_CSS

    if STYLE_CSS.exists():
        st.markdown(f"<style>{STYLE_CSS.read_text(encoding='utf-8')}</style>", unsafe_allow_html=True)


def render_overview_tab(
    info: dict[str, Any],
    history: pd.DataFrame,
    tech_df: pd.DataFrame,
    news: list[dict[str, Any]],
    risk: dict[str, Any],
    key_prefix: str = "",
) -> None:
    """Overview tab: header, chart, key stats.

    Always uses `history` for the price chart so the Overview tab shows a
    clean daily close line rather than the OHLC candlestick from tech_df.
    The Technicals tab is the right place for the full candlestick + MAs view.
    """
    header_block(info)
    st.markdown("---")
    render_price_chart(
        history,
        show_ma=False,
        title="Price & Moving Averages",
        key_prefix=key_prefix,
    )
    st.markdown("---")

    cols = st.columns(4)
    with cols[0]:
        st.metric("52W High", format_currency(info.get("fifty_two_week_high")))
    with cols[1]:
        st.metric("52W Low", format_currency(info.get("fifty_two_week_low")))
    with cols[2]:
        rg = info.get("revenue_growth")
        st.metric("Revenue Growth", format_percent(rg) if rg is not None else "N/A")
    with cols[3]:
        status_badge("Risk Level", risk.get("risk_level", "moderate"))

    st.subheader("Recent Headlines")
    if news:
        for article in news[:5]:
            news_card(article)
    else:
        st.info("No recent news available for this ticker.")


def render_comparison_tab(comparison: dict[str, Any]) -> None:
    """Render a simple side-by-side comparison for two tickers."""
    tickers = comparison.get("tickers", [])
    if len(tickers) < 2:
        st.info("Provide two tickers to compare (e.g. 'AAPL vs MSFT').")
        return

    a = comparison.get("a", {}) or {}
    b = comparison.get("b", {}) or {}

    st.markdown(f"## Comparison: {tickers[0]} vs {tickers[1]}")
    cols = st.columns(2)

    with cols[0]:
        st.markdown(f"### {a.get('ticker', tickers[0])}")
        st.caption(a.get("name", ""))
        st.metric(
            "Price",
            format_currency(a.get("current_price")),
            delta=format_percent(a.get("daily_change_pct")) if a.get("daily_change_pct") is not None else None,
        )
        st.metric("Market Cap", str(a.get("market_cap", "N/A")))
        st.metric("P/E", str(a.get("pe_ratio", "N/A")))
        st.metric(
            "Revenue Growth",
            format_percent(a.get("revenue_growth")) if a.get("revenue_growth") is not None else "N/A",
        )

    with cols[1]:
        st.markdown(f"### {b.get('ticker', tickers[1])}")
        st.caption(b.get("name", ""))
        st.metric(
            "Price",
            format_currency(b.get("current_price")),
            delta=format_percent(b.get("daily_change_pct")) if b.get("daily_change_pct") is not None else None,
        )
        st.metric("Market Cap", str(b.get("market_cap", "N/A")))
        st.metric("P/E", str(b.get("pe_ratio", "N/A")))
        st.metric(
            "Revenue Growth",
            format_percent(b.get("revenue_growth")) if b.get("revenue_growth") is not None else "N/A",
        )

    st.markdown("---")
    st.caption('Tip: ask follow-ups like "Which is less risky?" or "Compare the technicals".')


def render_technicals_tab(
    technicals: dict[str, Any],
    tech_df: pd.DataFrame,
    key_prefix: str = "",
) -> None:
    """Technicals tab with indicators and charts."""
    latest = technicals.get("latest", {})
    cols = st.columns(5)
    labels = [
        ("RSI (14)", f"{latest.get('rsi', 0):.1f}" if latest.get("rsi") else "N/A"),
        ("MA 20", format_currency(latest.get("ma20"))),
        ("MA 50", format_currency(latest.get("ma50"))),
        ("MACD", f"{latest.get('macd', 0):.4f}" if latest.get("macd") else "N/A"),
        (
            "Volatility",
            format_percent(latest.get("volatility_annualized")) if latest.get("volatility_annualized") else "N/A",
        ),
    ]
    if len(cols) != len(labels):
        raise ValueError(f"Column/label count mismatch: {len(cols)} cols vs {len(labels)} labels")
    for col, (label, val) in zip(cols, labels):
        with col:
            st.metric(label, val)

    interp = technicals.get("interpretation", "")
    if interp:
        st.info(interp)

    insufficient = latest.get("rsi") is None and latest.get("ma20") is None
    if insufficient:
        st.warning(
            "Most indicators show N/A because the selected timeframe / interval doesn't provide enough data points. "
            "Try a longer period (e.g. 6mo or 1y) with the 1d interval to get RSI, MA20, MA50, and volatility."
        )

    render_price_chart(tech_df, show_ma=True, title="Technical Chart", key_prefix=key_prefix)
    st.markdown("<div class='chart-gap'></div>", unsafe_allow_html=True)
    render_macd_chart(tech_df, key_prefix=key_prefix)
    st.markdown("<div class='chart-gap'></div>", unsafe_allow_html=True)
    render_volatility_chart(tech_df, key_prefix=key_prefix)


def render_sentiment_tab(
    sentiment: dict[str, Any],
    news: list[dict[str, Any]],
    key_prefix: str = "",
) -> None:
    """Sentiment tab with gauge and charts."""
    sentiment_gauge(sentiment)
    st.write(sentiment.get("interpretation", ""))
    render_sentiment_charts(sentiment, key_prefix=key_prefix)

    st.subheader("Headline-Level Sentiment")
    items = sentiment.get("items", [])
    if items:
        df = pd.DataFrame(items)
        st.dataframe(df[["headline", "label", "confidence"]], use_container_width=True, hide_index=True)
    else:
        for article in news[:8]:
            news_card(article)


def render_memo_tab(memo: str) -> None:
    """Investment memo tab."""
    st.markdown(memo)


def _chart_dates(df: pd.DataFrame) -> pd.Series:
    """Normalize date column for Plotly."""
    if "Date" in df.columns:
        dates = pd.to_datetime(df["Date"], errors="coerce")
        if getattr(dates.dt, "tz", None) is not None:
            dates = dates.dt.tz_localize(None)
        return dates
    if isinstance(df.index, pd.DatetimeIndex):
        idx = df.index
        if idx.tz is not None:
            idx = idx.tz_localize(None)
        return pd.Series(idx, name="Date")
    return pd.Series(df.index, name="index")


def render_price_chart(
    df: pd.DataFrame,
    show_ma: bool = True,
    title: str = "Stock Price",
    key_prefix: str = "",
) -> None:
    """Plotly line/candlestick chart with optional MAs."""
    if df.empty:
        st.info("No price data available to chart.")
        return

    st.markdown(f"<div class='chart-title'>{escape(title)}</div>", unsafe_allow_html=True)
    fig = go.Figure()
    dates = _chart_dates(df)

    if all(c in df.columns for c in ("Open", "High", "Low", "Close")):
        fig.add_trace(
            go.Candlestick(
                x=dates,
                open=df["Open"],
                high=df["High"],
                low=df["Low"],
                close=df["Close"],
                name="OHLC",
            )
        )
    elif "Close" in df.columns:
        fig.add_trace(
            go.Scatter(
                x=dates,
                y=df["Close"],
                mode="lines",
                name="Close",
                line=dict(color="#38bdf8", width=2),
            )
        )
    else:
        st.warning("Price data is missing expected columns (Open/High/Low/Close).")
        return

    if show_ma and "MA20" in df.columns:
        fig.add_trace(
            go.Scatter(
                x=dates,
                y=df["MA20"],
                mode="lines",
                name="MA20",
                line=dict(color="#f59e0b", width=1.5),
            )
        )
    if show_ma and "MA50" in df.columns:
        fig.add_trace(
            go.Scatter(
                x=dates,
                y=df["MA50"],
                mode="lines",
                name="MA50",
                line=dict(color="#a78bfa", width=1.5),
            )
        )

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis_rangeslider_visible=False,
        height=420,
        margin=dict(l=40, r=40, t=20, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
    )
    st.plotly_chart(fig, use_container_width=True, key=f"{key_prefix}price_chart_{title}")


def render_macd_chart(df: pd.DataFrame, key_prefix: str = "") -> None:
    """MACD subplot."""
    if not all(c in df.columns for c in ("MACD", "MACD_Signal", "MACD_Hist")):
        return
    st.markdown("<div class='chart-title'>MACD Momentum</div>", unsafe_allow_html=True)
    dates = _chart_dates(df)
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.6, 0.4], vertical_spacing=0.08)
    fig.add_trace(go.Scatter(x=dates, y=df["MACD"], name="MACD", line=dict(color="#38bdf8")), row=1, col=1)
    fig.add_trace(go.Scatter(x=dates, y=df["MACD_Signal"], name="Signal", line=dict(color="#f59e0b")), row=1, col=1)
    colors = ["#10b981" if v >= 0 else "#ef4444" for v in df["MACD_Hist"].fillna(0)]
    fig.add_trace(go.Bar(x=dates, y=df["MACD_Hist"], name="Histogram", marker_color=colors), row=2, col=1)
    fig.update_layout(
        template="plotly_dark",
        height=340,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=40, r=40, t=20, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        showlegend=True,
    )
    st.plotly_chart(fig, use_container_width=True, key=f"{key_prefix}macd_chart")


def render_volatility_chart(df: pd.DataFrame, key_prefix: str = "") -> None:
    """Rolling volatility chart."""
    chart_df = df.copy()
    if "Volatility_20d" not in chart_df.columns:
        if "Close" not in chart_df.columns:
            return
        chart_df["Volatility_20d"] = (
            chart_df["Close"].pct_change().rolling(window=20, min_periods=5).std() * (252 ** 0.5)
        )

    chart_df = chart_df[pd.notna(chart_df["Volatility_20d"])].reset_index(drop=True)
    if chart_df.empty:
        st.info("Rolling volatility needs at least a few price changes before it can be charted.")
        return

    dates = _chart_dates(chart_df)
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=dates,
            y=chart_df["Volatility_20d"],
            mode="lines",
            fill="tozeroy",
            name="20d Ann. Volatility",
            line=dict(color="#f472b6"),
        )
    )
    fig.update_layout(
        title="Rolling Volatility (20-day)",
        template="plotly_dark",
        height=280,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
    )
    st.plotly_chart(fig, use_container_width=True, key=f"{key_prefix}volatility_chart")


def render_sentiment_charts(sentiment: dict[str, Any], key_prefix: str = "") -> None:
    """Pie and bar charts for sentiment distribution."""
    counts = sentiment.get("counts", {})
    if not any(counts.values()):
        return

    col1, col2 = st.columns(2)
    labels = list(counts.keys())
    values = [counts[k] for k in labels]
    colors_map = {"positive": "#10b981", "negative": "#ef4444", "neutral": "#94a3b8"}
    colors = [colors_map.get(k, "#64748b") for k in labels]

    with col1:
        fig = go.Figure(data=[go.Pie(labels=labels, values=values, marker_colors=colors, hole=0.45)])
        fig.update_layout(template="plotly_dark", title="Sentiment Mix", height=320, paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig, use_container_width=True, key=f"{key_prefix}sentiment_pie")

    with col2:
        fig = go.Figure(data=[go.Bar(x=labels, y=values, marker_color=colors)])
        fig.update_layout(
            template="plotly_dark", title="Headline Counts", height=320, paper_bgcolor="rgba(0,0,0,0)"
        )
        st.plotly_chart(fig, use_container_width=True, key=f"{key_prefix}sentiment_bar")