"""Deterministic investment memo generation (no external LLM)."""

from __future__ import annotations

from typing import Any

from utils.helpers import format_currency, format_large_number, format_percent
from utils.logger import get_logger

logger = get_logger("memo_generator")


def generate_investment_memo(
    info: dict[str, Any],
    news: list[dict[str, Any]],
    sentiment: dict[str, Any],
    technicals: dict[str, Any],
    risk: dict[str, Any],
) -> str:
    """Build a professional analyst-style memo from structured inputs."""
    logger.info("Generating investment memo for %s", info.get("ticker"))
    ticker = info.get("ticker", "N/A")
    name = info.get("name", ticker)
    sector = info.get("sector", "N/A")
    price = info.get("current_price")
    change_pct = info.get("daily_change_pct")

    outlook = _derive_outlook(sentiment, technicals, risk)

    news_bullets = _format_news(news[:5])
    tech_latest = technicals.get("latest", {})

    memo = f"""
# Investment Memo: {name} ({ticker})

**Date:** Auto-generated research snapshot  
**Sector:** {sector} | **Industry:** {info.get('industry', 'N/A')}  
**Recommendation tone:** {outlook['stance']} — Confidence: {outlook['confidence']}

---

## Executive Summary

{name} ({ticker}) is currently trading at {format_currency(price)} with a daily change of {format_percent(change_pct) if change_pct is not None else 'N/A'}. 
Market capitalization stands at {format_large_number(info.get('market_cap'))} with a trailing P/E of {info.get('pe_ratio') or 'N/A'}.
Revenue growth (YoY proxy) is reported at {format_percent(info.get('revenue_growth')) if info.get('revenue_growth') is not None else 'N/A'}.

Our quantitative review combines price action, news sentiment (FinBERT), and risk metrics. 
**Overall outlook:** {outlook['summary']}

---

## Company Overview

{info.get('description') or 'No company description available from data provider.'}

| Metric | Value |
|--------|-------|
| 52-Week High | {format_currency(info.get('fifty_two_week_high'))} |
| 52-Week Low | {format_currency(info.get('fifty_two_week_low'))} |
| Beta | {info.get('beta') or 'N/A'} |
| Exchange | {info.get('exchange', 'N/A')} |

---

## Recent News Summary

{news_bullets}

**Sentiment read-through:** {sentiment.get('interpretation', 'N/A')}  
Aggregate label: **{sentiment.get('overall_label', 'neutral').upper()}** | Bullish confidence: {sentiment.get('bullish_confidence', 0):.0%} | Bearish confidence: {sentiment.get('bearish_confidence', 0):.0%}

---

## Technical Analysis

{technicals.get('interpretation', 'N/A')}

| Indicator | Latest |
|-----------|--------|
| RSI (14) | {tech_latest.get('rsi', 'N/A')} |
| 20-Day MA | {format_currency(tech_latest.get('ma20'))} |
| 50-Day MA | {format_currency(tech_latest.get('ma50'))} |
| MACD | {tech_latest.get('macd', 'N/A')} |
| Ann. Volatility | {format_percent(tech_latest.get('volatility_annualized')) if tech_latest.get('volatility_annualized') else 'N/A'} |

---

## Risk Assessment

{risk.get('interpretation', 'N/A')}

**Risk rating:** {risk.get('risk_level', 'N/A').upper()}

---

## Investment View

{outlook['detail']}

---

*Disclaimer: This memo is generated automatically for educational purposes. It is not investment advice. 
Always conduct your own due diligence and consult a licensed financial advisor.*
""".strip()

    return memo


def _derive_outlook(
    sentiment: dict[str, Any],
    technicals: dict[str, Any],
    risk: dict[str, Any],
) -> dict[str, str]:
    """Score bullish/bearish factors deterministically."""
    score = 0.0

    label = sentiment.get("overall_label", "neutral")
    if label == "positive":
        score += 1.5
    elif label == "negative":
        score -= 1.5

    avg = sentiment.get("average_score", 0)
    score += avg * 2

    latest = technicals.get("latest", {})
    rsi = latest.get("rsi")
    if rsi and rsi < 35:
        score += 0.5
    elif rsi and rsi > 65:
        score -= 0.5

    price = latest.get("price")
    ma20, ma50 = latest.get("ma20"), latest.get("ma50")
    if price and ma20 and ma50:
        if price > ma20 > ma50:
            score += 1.0
        elif price < ma20 < ma50:
            score -= 1.0

    risk_level = risk.get("risk_level", "moderate")
    if risk_level == "high":
        score -= 1.0
    elif risk_level == "low":
        score += 0.5

    if score >= 1.5:
        stance = "Constructive / Overweight bias"
        summary = "Favorable alignment of sentiment and technical structure supports a constructive near-term view."
    elif score <= -1.5:
        stance = "Cautious / Underweight bias"
        summary = "Negative sentiment and weak technicals suggest caution; wait for clearer confirmation."
    else:
        stance = "Neutral / Market-perform"
        summary = "Signals are mixed; maintain a neutral stance until a clearer trend emerges."

    confidence = "High" if abs(score) >= 2.5 else "Medium" if abs(score) >= 1.0 else "Low"

    detail = (
        f"Composite score: {score:+.2f}. Sentiment is {label}; risk is {risk_level}. "
        f"{technicals.get('interpretation', '')}"
    )

    return {"stance": stance, "summary": summary, "confidence": confidence, "detail": detail}


def _format_news(articles: list[dict[str, Any]]) -> str:
    if not articles:
        return "_No recent headlines available._"
    lines = []
    for i, a in enumerate(articles, 1):
        lines.append(f"{i}. **{a.get('title', 'Untitled')}** — _{a.get('publisher', 'Unknown')}_ ({a.get('publish_date', '')})")
    return "\n".join(lines)
