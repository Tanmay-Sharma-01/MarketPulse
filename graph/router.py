"""Query routing and ticker extraction."""

from __future__ import annotations

import re
from dataclasses import dataclass

from utils.logger import get_logger

logger = get_logger("graph.router")


_TICKER_RE = re.compile(r"\$?[A-Za-z]{1,5}\b")


@dataclass(frozen=True)
class RouteResult:
    intent: str
    tickers: list[str]
    primary_ticker: str | None
    comparison_tickers: list[str]
    is_beginner: bool


def extract_tickers(text: str) -> list[str]:
    """Extract likely ticker symbols from text.

    This is intentionally conservative; it filters common false positives.
    """
    candidates: list[str] = []
    for m in _TICKER_RE.finditer(text):
        raw = m.group(0)
        # Only treat explicit uppercase tokens (or $prefixed) as ticker candidates.
        if raw.startswith("$"):
            token = raw[1:].upper()
        elif raw.isupper():
            token = raw.upper()
        else:
            continue
        candidates.append(token)
    stop = {
        "A", "I", "AN", "THE", "AND", "OR", "USD", "CEO", "CFO", "ETF", "AI",
        "IN", "ON", "AT", "TO", "FOR", "OF", "IS", "IT", "AS", "BE", "BY",
        "VS", "IF", "MY", "ME", "WE", "US", "SO", "DO", "AM", "PM",
        "BUY", "SELL", "HOLD", "GET", "ARE", "WAS", "HAS", "HAD", "CAN",
        "WHY", "HOW", "WHAT", "WHEN", "WHO", "ALL", "ANY", "NOT", "OUT",
        "UP", "DOWN", "DAY", "TODAY", "YTD", "QOQ", "YOY", "EPS", "PE",
    }
    tickers = [c for c in candidates if c not in stop]
    # de-dupe preserving order
    seen: set[str] = set()
    out: list[str] = []
    for t in tickers:
        if t not in seen:
            seen.add(t)
            out.append(t)
    return out


def classify_intent(text: str, tickers: list[str]) -> str:
    t = text.lower()
    if any(k in t for k in ("compare", "vs", "versus")) and len(tickers) >= 2:
        return "compare"
    if any(k in t for k in ("rsi", "macd", "moving average", "ma20", "ma50", "technicals", "chart")):
        return "technical_analysis"
    if any(k in t for k in ("risk", "drawdown", "volatility", "beta")):
        return "risk_analysis"
    if any(k in t for k in ("news", "headline", "why did", "drop today", "fell", "rally", "catalyst", "catalysts")):
        return "catalyst_analysis"
    if is_beginner_question(text):
        return "beginner_explanation"
    if tickers:
        return "investment_thesis"
    return "unknown"


def is_beginner_question(text: str) -> bool:
    t = text.lower()
    return any(
        k in t
        for k in (
            "explain like",
            "beginner",
            "simple",
            "what is",
            "how does",
            "eli5",
        )
    )


def route_query(text: str, forced_ticker: str | None = None) -> RouteResult:
    """Classify intent and extract tickers.

    If ``forced_ticker`` is set (e.g. Quick Research tab), it is prepended so the
    graph always has a primary symbol even when the user message has no uppercase token.
    """
    tickers = extract_tickers(text)
    ft = (forced_ticker or "").strip().upper()
    if ft:
        if ft not in tickers:
            tickers.insert(0, ft)
        else:
            tickers = [ft] + [t for t in tickers if t != ft]
    beginner = is_beginner_question(text)
    intent = classify_intent(text, tickers)

    primary = tickers[0] if tickers else None
    comparison = tickers[:2] if intent == "compare" else []
    logger.info("Routed query intent=%s tickers=%s", intent, tickers)
    return RouteResult(
        intent=intent,
        tickers=tickers,
        primary_ticker=primary,
        comparison_tickers=comparison,
        is_beginner=beginner,
    )

