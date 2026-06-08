"""LangGraph shared state for the investment research workflow."""

from __future__ import annotations

from typing import Any, Literal, NotRequired, TypedDict

import pandas as pd


Intent = Literal[
    "investment_thesis",
    "catalyst_analysis",
    "compare",
    "beginner_explanation",
    "technical_analysis",
    "risk_analysis",
    "unknown",
]


class ResearchState(TypedDict, total=False):
    """Shared state passed between LangGraph nodes.

    Keep this dictionary lightweight and serializable where possible. Large objects
    like DataFrames are okay for local execution but should be treated as optional.
    """

    # Conversation input
    user_query: str
    intent: Intent
    tickers: list[str]
    primary_ticker: str
    comparison_tickers: list[str]

    # UI/status
    events: list[str]
    errors: list[str]

    # Memory (session-local)
    memory: dict[str, Any]

    # Time controls
    period: str
    interval: str

    # Core data artifacts (single stock)
    info: dict[str, Any]
    history: pd.DataFrame
    news: list[dict[str, Any]]
    sentiment: dict[str, Any]
    technicals: dict[str, Any]
    tech_df: pd.DataFrame
    risk: dict[str, Any]
    memo: str

    # Comparison artifacts
    comparison: dict[str, Any]

    # Multi-agent analysis
    bull_findings: list[str]
    bear_findings: list[str]
    technical_findings: list[str]
    recommendation: str
    confidence: float
    structured_context: dict[str, Any]
    final_response: str

    # Self-critique
    critique: str
    needs_refine: bool

