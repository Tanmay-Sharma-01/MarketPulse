"""Central LLM synthesis layer for conversational responses.

This module is the only place where we call DeepSeek. It does not perform raw
calculations; it only reasons over structured evidence produced by other modules.
"""

from __future__ import annotations

import json
from typing import Any

import requests

from utils.config import DEEPSEEK_API_KEY, DEEPSEEK_BASE_URL, DEEPSEEK_MODEL
from utils.logger import get_logger

logger = get_logger("analysis.final_response_generator")


def _headlines(news: list[dict[str, Any]], limit: int = 5) -> list[str]:
    out: list[str] = []

    for item in news[:limit]:
        title = item.get("title", "Untitled")
        pub = item.get("publisher", "Unknown")
        out.append(f"{title} ({pub})")

    return out


def _semantic_sentiment(sentiment: dict[str, Any]) -> str:
    score = sentiment.get("average_score", 0)

    if score >= 0.4:
        return "investor sentiment appears strongly positive"

    if score >= 0.15:
        return "sentiment is cautiously optimistic"

    if score <= -0.4:
        return "negative sentiment appears dominant"

    if score <= -0.15:
        return "investors appear cautious"

    return "market sentiment appears mixed"


def _semantic_technical_state(tech: dict[str, Any]) -> str:
    latest = tech.get("latest", {}) or {}

    rsi = latest.get("rsi")
    macd = latest.get("macd")
    ma20 = latest.get("ma20")
    ma50 = latest.get("ma50")

    observations: list[str] = []

    if rsi is not None:
        if rsi >= 70:
            observations.append("momentum appears overheated")
        elif rsi <= 35:
            observations.append(
                "the stock may be approaching oversold conditions"
            )
        else:
            observations.append("momentum appears relatively neutral")

    if macd is not None:
        if macd > 0:
            observations.append("trend momentum remains constructive")
        else:
            observations.append("momentum has weakened recently")

    if ma20 is not None and ma50 is not None:
        if ma20 > ma50:
            observations.append(
                "short-term trend remains stronger than the longer-term trend"
            )
        else:
            observations.append("technical momentum has deteriorated")

    return ". ".join(observations)


def _semantic_risk_state(risk: dict[str, Any]) -> str:
    risk_level = (risk.get("risk_level") or "").upper()
    drawdown = risk.get("recent_drawdown")

    if risk_level == "HIGH":
        base = "the stock is trading in a high-risk environment"
    elif risk_level == "MEDIUM":
        base = "risk levels appear moderate"
    else:
        base = "risk levels appear relatively controlled"

    if drawdown is not None and drawdown > 0.1:
        base += (
            ", and recent price action suggests investor conviction "
            "has weakened"
        )

    return base


def _extract_market_narrative(
    state: dict[str, Any],
    context: dict[str, Any],
) -> str:
    question = (state.get("user_query") or "").lower()
    ticker = (context.get("ticker") or "").upper()

    headlines = " ".join(context.get("news_catalysts", [])).lower()

    if ticker == "TTWO":
        if "gta" in question or "gta" in headlines:
            return (
                "Investors appear focused on uncertainty surrounding "
                "GTA 6 updates and whether management is providing "
                "enough near-term catalysts to justify elevated expectations."
            )

        return (
            "The recent weakness in TTWO likely reflects a cooling of "
            "investor expectations after a period of elevated optimism "
            "around future game releases."
        )

    if ticker in ["NVDA", "AMD", "MSFT"]:
        return (
            "Investor positioning appears heavily influenced by AI-related "
            "growth expectations and concerns about whether current "
            "valuations can continue to expand."
        )

    if "drop" in question or "why" in question:
        return (
            "The recent move likely reflects a combination of shifting "
            "investor expectations, technical weakness, and broader "
            "market positioning."
        )

    return (
        "Investors appear to be balancing long-term opportunities "
        "against near-term uncertainty and market volatility."
    )


def build_structured_context(state: dict[str, Any]) -> dict[str, Any]:
    """Build concise, intent-aware evidence context for the LLM."""

    intent = state.get("intent", "investment_thesis")

    info = state.get("info", {}) or {}
    tech = state.get("technicals", {}) or {}
    risk = state.get("risk", {}) or {}
    sentiment = state.get("sentiment", {}) or {}
    news = state.get("news", []) or []

    context = {
        "intent": intent,
        "question": state.get("user_query", ""),
        "ticker": info.get("ticker"),
        "company": info.get("name"),
        "market_narrative": "",
        "fundamentals": {
            "market_cap": info.get("market_cap"),
            "pe_ratio": info.get("pe_ratio"),
            "revenue_growth": info.get("revenue_growth"),
        },
        "price_snapshot": {
            "current_price": info.get("current_price"),
            "daily_change_pct": info.get("daily_change_pct"),
        },
        "news_catalysts": _headlines(news),
        "sentiment_state": _semantic_sentiment(sentiment),
        "technical_state": _semantic_technical_state(tech),
        "risk_state": _semantic_risk_state(risk),
        "agent_findings": {
            "bull": state.get("bull_findings", []),
            "bear": state.get("bear_findings", []),
            "technical": state.get("technical_findings", []),
        },
    }

    context["market_narrative"] = _extract_market_narrative(
        state,
        context,
    )

    if intent == "comparison":
        context["comparison"] = state.get("comparison", {})

    return context


def _intent_prompt(intent: str) -> str:
    if intent == "catalyst_analysis":
        return (
            "Focus on the most likely market narrative and investor reaction. "
            "Be concise, conversational, and catalyst-focused."
        )

    if intent == "technical_analysis":
        return (
            "Focus more heavily on momentum, trend structure, and technical "
            "interpretation while still sounding natural and conversational."
        )

    if intent == "beginner_explanation":
        return (
            "Explain the situation simply and clearly without excessive "
            "financial jargon."
        )

    if intent == "comparison":
        return (
            "Compare the companies side-by-side and explain the key differences "
            "in sentiment, momentum, risk, and opportunity."
        )

    return (
        "Provide a balanced investment-style explanation with natural reasoning "
        "and concise supporting evidence."
    )


def _build_prompt(context: dict[str, Any]) -> str:
    intent = context.get("intent", "investment_thesis")

    style_instruction = _intent_prompt(intent)

    return f"""
You are a thoughtful equity research analyst.

Use the evidence below as grounding, but synthesize a coherent market narrative
and explain likely investor psychology.

IMPORTANT RULES:
- Lead with the most likely explanation FIRST
- Prioritize narrative over raw indicators
- Use technicals as supporting evidence only
- Avoid sounding robotic or template-driven
- Do not dump every metric
- Focus on the 2-4 most important factors
- Speak naturally like a real analyst
- Be nuanced and uncertainty-aware
- Avoid repeating volatility or RSI excessively
- Do NOT mention internal scoring systems

STYLE INSTRUCTIONS:
{style_instruction}

EVIDENCE CONTEXT:
{json.dumps(context, ensure_ascii=False, indent=2)}

Generate a natural conversational response.
"""


def generate_final_response(state: dict[str, Any]) -> dict[str, Any]:
    """Assemble context and call DeepSeek for final conversational synthesis."""

    context = build_structured_context(state)

    prompt = _build_prompt(context)

    if not DEEPSEEK_API_KEY:
        fallback = (
            "DeepSeek API key is not configured.\n\n"
            f"Market narrative: {context.get('market_narrative')}\n"
            f"Sentiment: {context.get('sentiment_state')}\n"
            f"Technical state: {context.get('technical_state')}\n"
            f"Risk state: {context.get('risk_state')}"
        )

        return {
            "final_response": fallback,
            "structured_context": context,
        }

    try:
        url = f"{DEEPSEEK_BASE_URL.rstrip('/')}/chat/completions"

        payload = {
            "model": DEEPSEEK_MODEL,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You are an experienced financial analyst who explains "
                        "market behavior naturally and insightfully."
                    ),
                },
                {
                    "role": "user",
                    "content": prompt,
                },
            ],
            "temperature": 0.45,
        }

        headers = {
            "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
            "Content-Type": "application/json",
        }

        resp = requests.post(
            url,
            headers=headers,
            json=payload,
            timeout=45,
        )

        resp.raise_for_status()

        data = resp.json()

        content = data["choices"][0]["message"]["content"].strip()

        return {
            "final_response": content,
            "structured_context": context,
        }

    except Exception as exc:  # noqa: BLE001
        logger.exception("DeepSeek response generation failed")

        fallback = (
            "I couldn't reach the DeepSeek reasoning service right now.\n\n"
            f"Market narrative: {context.get('market_narrative')}\n"
            f"Sentiment: {context.get('sentiment_state')}\n"
            f"Technical state: {context.get('technical_state')}\n"
            f"Risk state: {context.get('risk_state')}\n\n"
            f"Error: {exc}"
        )

        return {
            "final_response": fallback,
            "structured_context": context,
        }