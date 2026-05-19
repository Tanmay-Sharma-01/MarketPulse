"""Streamlit dashboard layout and visualizations."""

from __future__ import annotations

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
) -> None:
    """Overview tab: header, chart, key stats."""
    header_block(info)
    st.markdown("---")
    render_price_chart(tech_df if "MA20" in tech_df.columns else history, title="Price & Moving Averages")
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


def render_technicals_tab(technicals: dict[str, Any], tech_df: pd.DataFrame) -> None:
    """Technicals tab with indicators and charts."""
    latest = technicals.get("latest", {})
    cols = st.columns(5)
    labels = [
        ("RSI (14)", f"{latest.get('rsi', 0):.1f}" if latest.get("rsi") else "N/A"),
        ("MA 20", format_currency(latest.get("ma20"))),
        ("MA 50", format_currency(latest.get("ma50"))),
        ("MACD", f"{latest.get('macd', 0):.4f}" if latest.get("macd") else "N/A"),
        ("Volatility", format_percent(latest.get("volatility_annualized")) if latest.get("volatility_annualized") else "N/A"),
    ]
    for col, (label, val) in zip(cols, labels, strict=True):
        with col:
            st.metric(label, val)

    st.info(technicals.get("interpretation", ""))
    render_price_chart(tech_df, show_ma=True, title="Technical Chart")
    render_macd_chart(tech_df)
    render_volatility_chart(tech_df)


def render_sentiment_tab(sentiment: dict[str, Any], news: list[dict[str, Any]]) -> None:
    """Sentiment tab with gauge and charts."""
    sentiment_gauge(sentiment)
    st.write(sentiment.get("interpretation", ""))
    render_sentiment_charts(sentiment)

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
        dates = pd.to_datetime(df["Date"])
        if getattr(dates.dt, "tz", None) is not None:
            dates = dates.dt.tz_localize(None)
        return dates
    return pd.to_datetime(df.index)


def render_price_chart(
    df: pd.DataFrame,
    show_ma: bool = True,
    title: str = "Stock Price",
) -> None:
    """Plotly line/candlestick chart with optional MAs."""
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
    else:
        fig.add_trace(
            go.Scatter(
                x=dates,
                y=df["Close"],
                mode="lines",
                name="Close",
                line=dict(color="#38bdf8", width=2),
            )
        )

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
        title=title,
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis_rangeslider_visible=False,
        height=420,
        margin=dict(l=40, r=40, t=50, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )
    st.plotly_chart(fig, use_container_width=True)


def render_macd_chart(df: pd.DataFrame) -> None:
    """MACD subplot."""
    if "MACD" not in df.columns:
        return
    dates = _chart_dates(df)
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.6, 0.4], vertical_spacing=0.08)
    fig.add_trace(go.Scatter(x=dates, y=df["MACD"], name="MACD", line=dict(color="#38bdf8")), row=1, col=1)
    fig.add_trace(go.Scatter(x=dates, y=df["MACD_Signal"], name="Signal", line=dict(color="#f59e0b")), row=1, col=1)
    colors = ["#10b981" if v >= 0 else "#ef4444" for v in df["MACD_Hist"].fillna(0)]
    fig.add_trace(go.Bar(x=dates, y=df["MACD_Hist"], name="Histogram", marker_color=colors), row=2, col=1)
    fig.update_layout(template="plotly_dark", height=320, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", showlegend=False)
    st.plotly_chart(fig, use_container_width=True)


def render_volatility_chart(df: pd.DataFrame) -> None:
    """Rolling volatility chart."""
    if "Volatility_20d" not in df.columns:
        return
    dates = _chart_dates(df)
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=dates,
            y=df["Volatility_20d"],
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
    )
    st.plotly_chart(fig, use_container_width=True)


def render_sentiment_charts(sentiment: dict[str, Any]) -> None:
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
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        fig = go.Figure(data=[go.Bar(x=labels, y=values, marker_color=colors)])
        fig.update_layout(template="plotly_dark", title="Headline Counts", height=320, paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig, use_container_width=True)
