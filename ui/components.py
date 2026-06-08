"""Reusable Streamlit UI components."""

from __future__ import annotations

from html import escape
from typing import Any

import streamlit as st

from utils.helpers import format_currency, format_percent


def status_badge(label: str, level: str) -> None:
    """Render a colored status badge."""
    colors = {
        "low": ("#10b981", "Low"),
        "moderate": ("#f59e0b", "Moderate"),
        "high": ("#ef4444", "High"),
        "positive": ("#10b981", "Positive"),
        "negative": ("#ef4444", "Negative"),
        "neutral": ("#94a3b8", "Neutral"),
    }
    color, text = colors.get(level.lower(), ("#64748b", level))
    st.markdown(
        f'<span class="badge" style="background:{color}22;color:{color};border:1px solid {color}55;">'
        f"{label}: {text}</span>",
        unsafe_allow_html=True,
    )


def metric_card(label: str, value: str, delta: str | None = None, delta_color: str = "normal") -> None:
    """Styled metric display."""
    st.metric(label=label, value=value, delta=delta, delta_color=delta_color)  # type: ignore[arg-type]


def sector_metric_card(value: str) -> None:
    """Metric-style card that wraps long sector names without clipping."""
    safe_value = escape(value or "N/A")
    st.markdown(
        f"""
        <div class="metric-card sector-metric-card">
            <div class="metric-label">Sector</div>
            <div class="metric-value sector-metric-value">{safe_value}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def sentiment_gauge(sentiment: dict[str, Any]) -> None:
    """Display sentiment summary with badges."""
    cols = st.columns(3)
    counts = sentiment.get("counts", {})
    with cols[0]:
        metric_card("Positive", str(counts.get("positive", 0)))
    with cols[1]:
        metric_card("Neutral", str(counts.get("neutral", 0)))
    with cols[2]:
        metric_card("Negative", str(counts.get("negative", 0)))

    st.markdown("---")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.caption("Avg Score")
        st.write(f"**{sentiment.get('average_score', 0):+.2f}**")
    with c2:
        st.caption("Bullish Confidence")
        st.write(f"**{sentiment.get('bullish_confidence', 0):.0%}**")
    with c3:
        st.caption("Bearish Confidence")
        st.write(f"**{sentiment.get('bearish_confidence', 0):.0%}**")

    status_badge("Overall Sentiment", sentiment.get("overall_label", "neutral"))


def header_block(info: dict[str, Any]) -> None:
    """Large ticker header with price."""
    ticker = info.get("ticker", "")
    name = info.get("name", ticker)
    price = info.get("current_price")
    change_pct = info.get("daily_change_pct")

    st.markdown(f"# {ticker}")
    st.markdown(f"### {name}")

    delta = format_percent(change_pct) if change_pct is not None else None

    cols = st.columns(4)
    with cols[0]:
        metric_card("Price", format_currency(price), delta=delta, delta_color="normal")
    with cols[1]:
        metric_card("Market Cap", _fmt_cap(info.get("market_cap")))
    with cols[2]:
        metric_card("P/E Ratio", f"{info.get('pe_ratio', 'N/A')}")
    with cols[3]:
        sector_metric_card(str(info.get("sector", "N/A")))


def _fmt_cap(value: float | None) -> str:
    from utils.helpers import format_large_number

    return format_large_number(value)


def news_card(article: dict[str, Any]) -> None:
    """Expandable news item."""
    title = article.get("title", "Untitled")
    with st.expander(title):
        st.caption(f"{article.get('publisher', 'Unknown')} · {article.get('publish_date', '')}")
        st.write(article.get("summary", ""))
        url = article.get("url")
        if url:
            st.markdown(f"[Read article]({url})")
