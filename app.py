"""
AI Stock Research Assistant — main Streamlit entry point.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure project root is on path when running via streamlit
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import streamlit as st

from analysis.memo_generator import generate_investment_memo
from analysis.risk_analysis import evaluate_risk
from analysis.sentiment import analyze_news_sentiment
from analysis.technicals import compute_technicals
from data.news_fetcher import fetch_recent_news
from data.stock_data import get_stock_package
from ui.dashboard import (
    load_custom_css,
    render_memo_tab,
    render_overview_tab,
    render_sentiment_tab,
    render_technicals_tab,
)
from utils.config import APP_TITLE, DEFAULT_PERIOD, DEFAULT_TICKER, PAGE_ICON
from utils.logger import setup_logger

setup_logger()


@st.cache_data(show_spinner=False, ttl=300)
def cached_stock_package(ticker: str, period: str, interval: str) -> dict:
    return get_stock_package(ticker, period=period, interval=interval)


@st.cache_data(show_spinner=False, ttl=300)
def cached_news(ticker: str) -> list:
    return fetch_recent_news(ticker)


@st.cache_resource(show_spinner="Loading FinBERT model (first run may take a minute)...")
def load_finbert():
    from analysis.sentiment import _get_pipeline

    return _get_pipeline()


def build_report(ticker: str, period: str, interval: str) -> dict:
    """Orchestrate full research pipeline."""
    stock = cached_stock_package(ticker, period, interval)
    news = cached_news(ticker)
    load_finbert()
    sentiment = analyze_news_sentiment(news)
    technicals = compute_technicals(stock["history"])
    risk = evaluate_risk(stock["history"], technicals)
    memo = generate_investment_memo(stock["info"], news, sentiment, technicals, risk)

    return {
        "info": stock["info"],
        "history": stock["history"],
        "news": news,
        "sentiment": sentiment,
        "technicals": technicals,
        "tech_df": technicals["dataframe"],
        "risk": risk,
        "memo": memo,
    }


def main() -> None:
    st.set_page_config(
        page_title=APP_TITLE,
        page_icon=PAGE_ICON,
        layout="wide",
        initial_sidebar_state="expanded",
    )
    load_custom_css()

    st.sidebar.title("📊 Research Controls")
    ticker = st.sidebar.text_input("Stock Ticker", value=DEFAULT_TICKER, placeholder="e.g. AAPL").strip().upper()
    period = st.sidebar.selectbox(
        "Timeframe",
        options=["1mo", "3mo", "6mo", "1y", "2y"],
        index=2,
    )
    interval = st.sidebar.selectbox("Interval", options=["1d", "1wk"], index=0)
    generate = st.sidebar.button("Generate Research Report", type="primary", use_container_width=True)

    st.sidebar.markdown("---")
    st.sidebar.caption("Uses yfinance + local FinBERT. Free APIs only.")

    st.title(APP_TITLE)
    st.caption("Automated stock research with technicals, sentiment, risk, and investment memo.")

    if not generate:
        st.info("Enter a ticker in the sidebar and click **Generate Research Report** to begin.")
        return

    if not ticker:
        st.error("Please enter a valid ticker symbol.")
        return

    try:
        with st.spinner(f"Building research report for {ticker}..."):
            report = build_report(ticker, period or DEFAULT_PERIOD, interval)
    except Exception as exc:  # noqa: BLE001
        st.error(f"Failed to generate report: {exc}")
        st.exception(exc)
        return

    tab_overview, tab_technicals, tab_sentiment, tab_memo = st.tabs(
        ["Overview", "Technicals", "Sentiment", "Investment Memo"]
    )

    with tab_overview:
        render_overview_tab(
            report["info"],
            report["history"],
            report["tech_df"],
            report["news"],
            report["risk"],
        )

    with tab_technicals:
        render_technicals_tab(report["technicals"], report["tech_df"])

    with tab_sentiment:
        render_sentiment_tab(report["sentiment"], report["news"])

    with tab_memo:
        render_memo_tab(report["memo"])


if __name__ == "__main__":
    main()
