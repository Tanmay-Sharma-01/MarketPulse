"""AI Stock Research Assistant — conversational multi-agent edition.

Two ways to use the app:
1. **Chat** — ask natural-language questions (LangGraph multi-agent workflow)
2. **Quick Research** — enter a ticker and generate the classic dashboard report
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

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
from graph.workflow import build_workflow
from memory.session_memory import SessionMemory
from ui.dashboard import (
    load_custom_css,
    render_comparison_tab,
    render_memo_tab,
    render_overview_tab,
    render_sentiment_tab,
    render_technicals_tab,
)
from utils.config import APP_TITLE, DEFAULT_PERIOD, DEFAULT_TICKER, PAGE_ICON
from utils.logger import setup_logger

setup_logger()


@st.cache_resource(show_spinner="Loading FinBERT model (first run may take a minute)...")
def load_finbert():
    from analysis.sentiment import _get_pipeline

    return _get_pipeline()


@st.cache_resource(show_spinner=False)
def get_app_graph():
    return build_workflow()


@st.cache_data(show_spinner=False, ttl=300)
def cached_stock_package(ticker: str, period: str, interval: str) -> dict:
    return get_stock_package(ticker, period=period, interval=interval)


@st.cache_data(show_spinner=False, ttl=300)
def cached_news(ticker: str) -> list:
    return fetch_recent_news(ticker)


def build_report(ticker: str, period: str, interval: str) -> dict[str, Any]:
    """Classic sequential report pipeline (Quick Research tab)."""
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


def _init_session() -> None:
    if "chat_messages" not in st.session_state:
        st.session_state.chat_messages = []
    if "memory" not in st.session_state:
        st.session_state.memory = SessionMemory().to_dict()
    if "last_state" not in st.session_state:
        st.session_state.last_state = None
    if "quick_report" not in st.session_state:
        st.session_state.quick_report = None


def _render_chat() -> None:
    for msg in st.session_state.chat_messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])


def _append_message(role: str, content: str) -> None:
    st.session_state.chat_messages.append({"role": role, "content": content})


def _clear_chat_state() -> None:
    st.session_state.chat_messages = []
    st.session_state.memory = SessionMemory().to_dict()
    st.session_state.last_state = None


def _assistant_answer_from_state(state: dict[str, Any]) -> str:
    if state.get("errors") and not state.get("comparison"):
        hist = state.get("history")
        if hist is None or getattr(hist, "empty", True):
            return state.get("final_response") or "Could not load market data for that request."

    if state.get("comparison"):
        t = state["comparison"].get("tickers", [])
        if len(t) >= 2:
            return (
                f"Here's a quick comparison of **{t[0]}** vs **{t[1]}**.\n\n"
                "Open the **Dashboard** tab for side-by-side metrics."
            )
        return 'I can compare two tickers if you ask like: "AAPL vs MSFT".'

    rec = state.get("recommendation", "Hold/Neutral")
    conf = float(state.get("confidence", 50.0))
    ticker = (state.get("info") or {}).get("ticker") or state.get("primary_ticker") or ""
    final_response = state.get("final_response", "") or ""
    header = f"**{ticker}** — **{rec}** (confidence: **{int(conf)} / 100**)"
    return header + ("\n\n" + final_response if final_response else "")


def _render_stock_dashboard(report: dict[str, Any], key_prefix: str = "") -> None:
    """Render the classic Overview / Technicals / Sentiment / Memo tabs."""
    tab_overview, tab_technicals, tab_sentiment, tab_memo = st.tabs(
        ["Overview", "Technicals", "Sentiment", "Investment Memo"]
    )

    with tab_overview:
        render_overview_tab(
            report["info"],
            report["history"],
            report["tech_df"] if report.get("tech_df") is not None else report["history"],
            report.get("news", []),
            report.get("risk", {}),
            key_prefix=key_prefix,
        )

    with tab_technicals:
        render_technicals_tab(report["technicals"], report["tech_df"], key_prefix=key_prefix)

    with tab_sentiment:
        render_sentiment_tab(report["sentiment"], report["news"], key_prefix=key_prefix)

    with tab_memo:
        render_memo_tab(report["memo"])


def _render_dashboard_from_state(state: dict[str, Any]) -> None:
    if state.get("comparison"):
        render_comparison_tab(state["comparison"])
        return

    if not state.get("info") or state.get("history") is None or getattr(state.get("history"), "empty", True):
        st.info("No stock dashboard available for the last request.")
        return

    _render_stock_dashboard(
        {
            "info": state["info"],
            "history": state["history"],
            "news": state.get("news", []),
            "sentiment": state.get("sentiment", {}),
            "technicals": state.get("technicals", {}),
            "tech_df": (
                state["tech_df"]
                if state.get("tech_df") is not None
                else state["history"]
            ),
            "risk": state.get("risk", {}),
            "memo": state.get("memo", ""),
        },
        key_prefix="chat_",
    )


def main() -> None:
    st.set_page_config(
        page_title=APP_TITLE,
        page_icon=PAGE_ICON,
        layout="wide",
        initial_sidebar_state="collapsed",
    )
    load_custom_css()
    _init_session()

    st.title(APP_TITLE)
    st.caption("Chat with the multi-agent assistant, or run a quick ticker report — your choice.")

    tab_chat, tab_quick, tab_dashboard = st.tabs(["Chat", "Quick Research", "Dashboard"])

    # ── Chat tab ─────────────────────────────────────────────────────────────
    with tab_chat:
        st.markdown(
            'Ask questions like: **"Should I invest in NVDA?"**, **"Why did TSLA drop today?"**, '
            '**"Compare AAPL vs MSFT"**, **"What are the risks for AMD?"**'
        )

        if st.button("Clear Chat", key="clear_chat", use_container_width=False):
            _clear_chat_state()
            st.rerun()

        with st.expander("Chat options", expanded=False):
            chat_period = st.selectbox(
                "Analysis timeframe",
                options=["1mo", "3mo", "6mo", "1y", "2y"],
                index=2,
                key="chat_period",
            )
            chat_interval = st.selectbox("Interval", options=["1d", "1wk"], index=0, key="chat_interval")
            focus_ticker = st.text_input(
                "Focus ticker (optional)",
                value="",
                placeholder="Use when your question doesn't include a symbol, e.g. NVDA",
                key="chat_focus_ticker",
            ).strip().upper()
            show_reasoning = st.toggle("Show agent reasoning panels", value=True, key="show_reasoning")
            show_workflow = st.toggle("Show workflow status", value=True, key="show_workflow")

        _render_chat()

        user_text = st.chat_input("Ask about a stock, news move, technicals, risk, or comparisons...")
        if user_text:
            mem = SessionMemory.from_dict(st.session_state.memory)
            mem.add_user(user_text)
            _append_message("user", user_text)

            graph = get_app_graph()
            load_finbert()

            initial_state: dict[str, Any] = {
                "user_query": user_text,
                "period": chat_period or DEFAULT_PERIOD,
                "interval": chat_interval,
                "events": [],
                "errors": [],
                "memory": mem.to_dict(),
            }
            if focus_ticker:
                initial_state["primary_ticker"] = focus_ticker

            try:
                with st.spinner("Running research workflow..."):
                    final_state = graph.invoke(initial_state)
            except Exception as exc:  # noqa: BLE001
                st.error(f"Workflow failed: {exc}")
                st.exception(exc)
                return

            answer = _assistant_answer_from_state(final_state)
            _append_message("assistant", answer)
            mem.add_assistant(answer)
            if final_state.get("tickers"):
                mem.last_tickers = list(final_state["tickers"])
            st.session_state.memory = mem.to_dict()
            st.session_state.last_state = final_state
            st.rerun()

        latest = st.session_state.last_state
        if latest and (show_workflow or show_reasoning):
            st.markdown("---")
            cols = st.columns(2)
            if show_workflow:
                with cols[0]:
                    with st.expander("Workflow status", expanded=False):
                        for e in latest.get("events", []):
                            st.write(f"- {e}")
            if show_reasoning:
                with cols[1]:
                    with st.expander("Agent reasoning", expanded=False):
                        if latest.get("bull_findings"):
                            st.markdown("**Bull Analyst**")
                            for item in latest["bull_findings"]:
                                st.write(f"- {item}")
                        if latest.get("bear_findings"):
                            st.markdown("**Bear Analyst**")
                            for item in latest["bear_findings"]:
                                st.write(f"- {item}")
                        if latest.get("technical_findings"):
                            st.markdown("**Technical Analyst**")
                            for item in latest["technical_findings"]:
                                st.write(f"- {item}")
                        if latest.get("critique"):
                            st.markdown("**Critique**")
                            st.write(latest["critique"])

    # ── Quick Research tab ───────────────────────────────────────────────────
    with tab_quick:
        st.markdown("### Quick Research")
        st.caption("Enter a ticker and generate the full dashboard report — no chat required.")

        col1, col2, col3 = st.columns([2, 1, 1])
        with col1:
            ticker = st.text_input(
                "Stock Ticker", value=DEFAULT_TICKER, placeholder="e.g. AAPL", key="quick_ticker"
            ).strip().upper()
        with col2:
            period = st.selectbox(
                "Timeframe", options=["1mo", "3mo", "6mo", "1y", "2y"], index=2, key="quick_period"
            )
        with col3:
            interval = st.selectbox("Interval", options=["1d", "1wk"], index=0, key="quick_interval")

        generate = st.button("Generate Research Report", type="primary", use_container_width=True)

        if generate:
            if not ticker:
                st.error("Please enter a valid ticker symbol.")
            else:
                try:
                    with st.spinner(f"Building research report for {ticker}..."):
                        st.session_state.quick_report = build_report(ticker, period or DEFAULT_PERIOD, interval)
                except Exception as exc:  # noqa: BLE001
                    st.error(f"Failed to generate report: {exc}")
                    st.exception(exc)

        if st.session_state.quick_report:
            st.markdown("---")
            _render_stock_dashboard(st.session_state.quick_report, key_prefix="quick_")
        elif not generate:
            st.info("Enter a ticker above and click **Generate Research Report**.")

    # ── Dashboard tab ─────────────────────────────────────────────────────────
    with tab_dashboard:
        st.markdown("### Dashboard")
        source = st.radio(
            "Show results from",
            options=["Latest chat response", "Quick Research report"],
            horizontal=True,
            key="dashboard_source",
        )

        if source == "Latest chat response":
            state = st.session_state.last_state
            if not state:
                st.info("Ask a question in the **Chat** tab to populate this dashboard.")
            else:
                _render_dashboard_from_state(state)
        else:
            report = st.session_state.quick_report
            if not report:
                st.info("Generate a report in the **Quick Research** tab first.")
            else:
                _render_stock_dashboard(report, key_prefix="dash_")


if __name__ == "__main__":
    main()