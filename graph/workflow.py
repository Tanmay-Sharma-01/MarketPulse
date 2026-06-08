"""LangGraph workflow: intent -> context gathering -> DeepSeek synthesis."""

from __future__ import annotations

from typing import Any

import pandas as pd
from langgraph.graph import END, StateGraph

from agents.analysts import run_bear_analyst, run_bull_analyst, run_critique, run_portfolio_manager, run_technical_analyst
from analysis.final_response_generator import generate_final_response
from analysis.memo_generator import generate_investment_memo
from analysis.risk_analysis import evaluate_risk
from analysis.sentiment import analyze_news_sentiment
from analysis.technicals import compute_technicals
from data.news_fetcher import fetch_recent_news
from data.stock_data import get_stock_package
from graph.router import route_query
from graph.state import ResearchState
from utils.config import VOLATILITY_HIGH
from utils.logger import get_logger

logger = get_logger("graph.workflow")


def _event(state: ResearchState, msg: str) -> ResearchState:
    events = list(state.get("events", []))
    events.append(msg)
    return {**state, "events": events}


def node_route(state: ResearchState) -> ResearchState:
    forced = (state.get("primary_ticker") or "").strip().upper() or None
    r = route_query(state.get("user_query", ""), forced_ticker=forced)
    return _event(
        {
            **state,
            "intent": r.intent,  # type: ignore[typeddict-item]
            "tickers": r.tickers,
            "primary_ticker": r.primary_ticker or "",
            "comparison_tickers": r.comparison_tickers,
        },
        f"Routed intent={r.intent} tickers={r.tickers}",
    )


def node_fetch_stock(state: ResearchState) -> ResearchState:
    ticker = (state.get("primary_ticker") or (state.get("tickers") or [""])[0] or "").strip().upper()
    if not ticker:
        errs = list(state.get("errors", []))
        errs.append("No ticker symbol found. Mention one in chat or use focus ticker.")
        return _event({**state, "errors": errs}, "Missing ticker; skipped fetch")
    try:
        pkg = get_stock_package(ticker, period=state.get("period", "6mo"), interval=state.get("interval", "1d"))
    except Exception as exc:  # noqa: BLE001
        errs = list(state.get("errors", []))
        errs.append(f"Could not fetch data for {ticker}: {exc}")
        logger.exception("node_fetch_stock failed for %s", ticker)
        return _event(
            {**state, "errors": errs, "info": {"ticker": ticker, "name": ticker}, "history": pd.DataFrame()},
            f"Fetch failed for {ticker}",
        )
    return _event({**state, "info": pkg["info"], "history": pkg["history"]}, f"Fetched stock data for {ticker}")


def node_fetch_news(state: ResearchState) -> ResearchState:
    ticker = state.get("primary_ticker") or state.get("info", {}).get("ticker", "")
    news = fetch_recent_news(ticker)
    return _event({**state, "news": news}, f"Fetched {len(news)} news items")


def node_sentiment(state: ResearchState) -> ResearchState:
    sentiment = analyze_news_sentiment(state.get("news", []))
    return _event({**state, "sentiment": sentiment}, "Computed FinBERT sentiment")


def node_technicals(state: ResearchState) -> ResearchState:
    history = state.get("history")
    if history is None or getattr(history, "empty", True):
        return _event(state, "Skipped technicals: no history")
    tech = compute_technicals(history)
    return _event({**state, "technicals": tech, "tech_df": tech["dataframe"]}, "Computed technical indicators")


def node_risk(state: ResearchState) -> ResearchState:
    history = state.get("history")
    if history is None or getattr(history, "empty", True):
        return _event(state, "Skipped risk: no history")
    risk = evaluate_risk(history, state.get("technicals", {}))
    return _event({**state, "risk": risk}, f"Evaluated risk: {risk.get('risk_level')}")


def node_deeper_risk(state: ResearchState) -> ResearchState:
    risk = dict(state.get("risk", {}) or {})
    vol = float(risk.get("volatility_annualized") or 0.0)
    notes: list[str] = []
    if vol >= VOLATILITY_HIGH:
        notes.append("Elevated volatility implies wider expected outcome range.")
        notes.append("Catalyst-driven sessions may have amplified price reactions.")
    if float(risk.get("recent_drawdown") or 0.0) >= 0.12:
        notes.append("Recent drawdown is large enough to indicate fragile conviction.")
    risk["deep_dive"] = notes
    return _event({**state, "risk": risk}, "Performed deeper risk dive")


def node_memo(state: ResearchState) -> ResearchState:
    memo = generate_investment_memo(
        state.get("info", {}),
        state.get("news", []),
        state.get("sentiment", {}),
        state.get("technicals", {}),
        state.get("risk", {}),
    )
    return _event({**state, "memo": memo}, "Generated deterministic memo")


def node_compare(state: ResearchState) -> ResearchState:
    t1, t2 = (state.get("comparison_tickers") or ["", ""])[:2]
    period = state.get("period", "6mo")
    interval = state.get("interval", "1d")
    try:
        p1 = get_stock_package(t1, period=period, interval=interval)
        p2 = get_stock_package(t2, period=period, interval=interval)
    except Exception as exc:  # noqa: BLE001
        errs = list(state.get("errors", []))
        errs.append(f"Comparison fetch failed: {exc}")
        return _event({**state, "errors": errs}, "Comparison fetch failed")
    return _event({**state, "comparison": {"tickers": [t1, t2], "a": p1["info"], "b": p2["info"]}}, f"Built comparison package: {t1} vs {t2}")


def node_bull(state: ResearchState) -> ResearchState:
    return _event({**state, **run_bull_analyst(state)}, "Bull analyst findings ready")


def node_bear(state: ResearchState) -> ResearchState:
    return _event({**state, **run_bear_analyst(state)}, "Bear analyst findings ready")


def node_tech_agent(state: ResearchState) -> ResearchState:
    return _event({**state, **run_technical_analyst(state)}, "Technical analyst findings ready")


def node_pm_anchor(state: ResearchState) -> ResearchState:
    return _event({**state, **run_portfolio_manager(state)}, "PM recommendation anchor ready")


def node_final_response(state: ResearchState) -> ResearchState:
    out = generate_final_response(state)
    return _event({**state, **out}, "DeepSeek final synthesis complete")


def node_critique(state: ResearchState) -> ResearchState:
    return _event({**state, **run_critique(state)}, "Critique completed")


def node_refine(state: ResearchState) -> ResearchState:
    out = generate_final_response(state)
    return _event({**state, **out}, "Refined DeepSeek synthesis")


def node_abort_fetch(state: ResearchState) -> ResearchState:
    errs = state.get("errors") or ["Unknown fetch error."]
    msg = "**Could not load market data.**\n\n" + "\n".join(f"- {e}" for e in errs)
    return _event({**state, "final_response": msg, "recommendation": "Hold/Neutral", "confidence": 0.0}, "Aborted: no price history")


def _route_after_intent(state: ResearchState) -> str:
    return "compare" if state.get("intent") == "compare" else "fetch"


def _after_fetch_stock(state: ResearchState) -> str:
    hist = state.get("history")
    if hist is None or getattr(hist, "empty", True):
        return "abort"
    return "news"


def _need_deeper_risk(state: ResearchState) -> str:
    risk = state.get("risk", {}) or {}
    vol = float(risk.get("volatility_annualized") or 0.0)
    return "deep" if vol >= VOLATILITY_HIGH else "skip"


def _needs_refine(state: ResearchState) -> str:
    return "refine" if state.get("needs_refine") else "done"


def build_workflow() -> Any:
    g: StateGraph = StateGraph(ResearchState)

    g.add_node("route", node_route)
    g.add_node("compare", node_compare)
    g.add_node("fetch_stock", node_fetch_stock)
    g.add_node("abort_fetch", node_abort_fetch)
    g.add_node("fetch_news", node_fetch_news)
    g.add_node("sentiment", node_sentiment)
    g.add_node("technicals", node_technicals)
    g.add_node("risk", node_risk)
    g.add_node("deeper_risk", node_deeper_risk)
    g.add_node("memo", node_memo)
    g.add_node("bull", node_bull)
    g.add_node("bear", node_bear)
    g.add_node("tech_agent", node_tech_agent)
    g.add_node("pm_anchor", node_pm_anchor)
    g.add_node("final_response", node_final_response)
    g.add_node("critique", node_critique)
    g.add_node("refine", node_refine)

    g.set_entry_point("route")
    g.add_conditional_edges("route", _route_after_intent, {"compare": "compare", "fetch": "fetch_stock"})
    g.add_edge("compare", "final_response")
    g.add_conditional_edges("fetch_stock", _after_fetch_stock, {"abort": "abort_fetch", "news": "fetch_news"})
    g.add_edge("abort_fetch", END)
    g.add_edge("fetch_news", "sentiment")
    g.add_edge("sentiment", "technicals")
    g.add_edge("technicals", "risk")
    g.add_conditional_edges("risk", _need_deeper_risk, {"deep": "deeper_risk", "skip": "memo"})
    g.add_edge("deeper_risk", "memo")
    g.add_edge("memo", "bull")
    g.add_edge("bull", "bear")
    g.add_edge("bear", "tech_agent")
    g.add_edge("tech_agent", "pm_anchor")
    g.add_edge("pm_anchor", "final_response")
    g.add_edge("final_response", "critique")
    g.add_conditional_edges("critique", _needs_refine, {"refine": "refine", "done": END})
    g.add_edge("refine", END)

    return g.compile()

