"""LangChain tool wrappers around existing data/analysis modules."""

from __future__ import annotations

from typing import Any

from langchain_core.tools import tool

from analysis.memo_generator import generate_investment_memo
from analysis.risk_analysis import evaluate_risk
from analysis.sentiment import analyze_news_sentiment
from analysis.technicals import compute_technicals
from data.news_fetcher import fetch_recent_news
from data.stock_data import get_stock_package


@tool
def get_stock_data(ticker: str, period: str = "6mo", interval: str = "1d") -> dict[str, Any]:
    """Fetch price history and company info for a ticker via yfinance."""
    return get_stock_package(ticker, period=period, interval=interval)


@tool
def get_stock_news(ticker: str, limit: int = 15) -> list[dict[str, Any]]:
    """Fetch recent news articles for a ticker via yfinance."""
    return fetch_recent_news(ticker, limit=limit)


@tool
def compute_technical_indicators(history: Any) -> dict[str, Any]:
    """Compute technical indicators from a price history DataFrame."""
    return compute_technicals(history)


@tool
def compute_risk(history: Any, technicals: dict[str, Any]) -> dict[str, Any]:
    """Compute basic risk metrics from history + technicals."""
    return evaluate_risk(history, technicals)


@tool
def analyze_sentiment(news: list[dict[str, Any]]) -> dict[str, Any]:
    """Analyze FinBERT sentiment over news headlines."""
    return analyze_news_sentiment(news)


@tool
def build_investment_memo(
    info: dict[str, Any],
    news: list[dict[str, Any]],
    sentiment: dict[str, Any],
    technicals: dict[str, Any],
    risk: dict[str, Any],
) -> str:
    """Generate the deterministic analyst-style investment memo."""
    return generate_investment_memo(info, news, sentiment, technicals, risk)

