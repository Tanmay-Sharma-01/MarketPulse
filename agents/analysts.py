"""Deterministic multi-agent findings for evidence construction.

These agents do not call an LLM. They convert computed outputs into structured
findings consumed by the final DeepSeek synthesis layer.
"""

from __future__ import annotations

from typing import Any

from utils.logger import get_logger

logger = get_logger("agents.analysts")


def run_bull_analyst(state: dict[str, Any]) -> dict[str, Any]:
    sentiment = (state.get("sentiment") or {}).get("overall_label", "neutral")
    avg_sent = float((state.get("sentiment") or {}).get("average_score", 0.0))
    tech = state.get("technicals") or {}
    latest = tech.get("latest", {}) or {}
    price = latest.get("price")
    ma20 = latest.get("ma20")
    ma50 = latest.get("ma50")

    findings: list[str] = []
    if sentiment == "positive":
        findings.append("News flow skews positive, which can support near-term risk appetite.")
    if avg_sent > 0.1:
        findings.append(f"Directional sentiment score is constructive ({avg_sent:+.2f}).")
    if price and ma20 and ma50 and price > ma20 > ma50:
        findings.append("Price is above MA20 and MA50, indicating bullish trend structure.")
    findings.append("Upside case depends on catalysts converting into sustained earnings expectations.")

    return {"bull_findings": findings}


def run_bear_analyst(state: dict[str, Any]) -> dict[str, Any]:
    risk = state.get("risk") or {}
    sentiment = (state.get("sentiment") or {}).get("overall_label", "neutral")
    recent_dd = float(risk.get("recent_drawdown") or 0.0)
    vol = float(risk.get("volatility_annualized") or 0.0)

    findings: list[str] = []
    if sentiment == "negative":
        findings.append("Headline tone is negative, increasing downside reflexivity risk.")
    if vol >= 0.30:
        findings.append(f"Annualized volatility is elevated ({vol:.1%}), implying wider downside tails.")
    if recent_dd >= 0.10:
        findings.append(f"Recent drawdown is meaningful ({recent_dd:.1%}), indicating fragile positioning.")
    findings.append("Bear case strengthens if momentum weakens while catalysts remain mixed.")

    return {"bear_findings": findings}


def run_technical_analyst(state: dict[str, Any]) -> dict[str, Any]:
    tech = state.get("technicals") or {}
    latest = tech.get("latest", {}) or {}
    rsi = latest.get("rsi")
    macd = latest.get("macd")
    macd_signal = latest.get("macd_signal")
    findings: list[str] = []
    if rsi is not None:
        if rsi >= 70:
            findings.append(f"RSI is elevated ({rsi:.1f}), suggesting crowded upside momentum.")
        elif rsi <= 30:
            findings.append(f"RSI is depressed ({rsi:.1f}), suggesting capitulation/mean-reversion potential.")
        else:
            findings.append(f"RSI is neutral ({rsi:.1f}), with no extreme positioning signal.")
    if macd is not None and macd_signal is not None:
        if macd > macd_signal:
            findings.append("MACD is above signal, indicating positive momentum bias.")
        else:
            findings.append("MACD is below signal, indicating fading momentum.")
    findings.append(tech.get("interpretation", "Technical structure is mixed."))
    return {"technical_findings": findings}


def run_portfolio_manager(state: dict[str, Any]) -> dict[str, Any]:
    """Generate a lightweight pre-LLM recommendation anchor."""
    sentiment = (state.get("sentiment") or {}).get("overall_label", "neutral")
    risk_level = (state.get("risk") or {}).get("risk_level", "moderate")
    score = 0
    if sentiment == "positive":
        score += 1
    elif sentiment == "negative":
        score -= 1
    if risk_level == "low":
        score += 1
    elif risk_level == "high":
        score -= 1
    if score >= 1:
        rec = "Buy-biased"
        conf = 65.0
    elif score <= -1:
        rec = "Avoid-biased"
        conf = 65.0
    else:
        rec = "Hold/Neutral"
        conf = 55.0
    return {"recommendation": rec, "confidence": conf}


def run_critique(state: dict[str, Any]) -> dict[str, Any]:
    if state.get("intent") == "compare":
        return {"critique": "PASS: comparison intent uses direct synthesis context", "needs_refine": False}
    has_bull = bool(state.get("bull_findings"))
    has_bear = bool(state.get("bear_findings"))
    has_tech = bool(state.get("technical_findings"))
    missing = []
    if not has_bull:
        missing.append("bull findings")
    if not has_bear:
        missing.append("bear findings")
    if not has_tech:
        missing.append("technical findings")
    if missing:
        critique = "FAIL: missing " + ", ".join(missing)
        return {"critique": critique, "needs_refine": True}
    return {"critique": "PASS: evidence blocks complete", "needs_refine": False}

