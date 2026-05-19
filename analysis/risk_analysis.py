"""Basic risk metrics and classification."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from utils.config import DRAWDOWN_HIGH, DRAWDOWN_MODERATE, VOLATILITY_HIGH, VOLATILITY_MODERATE
from utils.logger import get_logger

logger = get_logger("risk_analysis")


def evaluate_risk(history: pd.DataFrame, technicals: dict[str, Any]) -> dict[str, Any]:
    """Compute volatility, trend weakness, drawdown, and risk level."""
    logger.info("Evaluating risk metrics")
    close = history["Close"]
    returns = close.pct_change().dropna()

    volatility = float(returns.std() * np.sqrt(252)) if len(returns) > 1 else 0.0

    ma20 = technicals.get("latest", {}).get("ma20")
    ma50 = technicals.get("latest", {}).get("ma50")
    price = float(close.iloc[-1])
    trend_weakness = _trend_weakness(price, ma20, ma50)

    drawdown = _max_drawdown(close)
    recent_drawdown = _recent_drawdown(close, window=30)

    risk_level = _classify_risk(volatility, trend_weakness, recent_drawdown)
    interpretation = _interpret_risk(risk_level, volatility, recent_drawdown, trend_weakness)

    return {
        "volatility_annualized": round(volatility, 4),
        "trend_weakness": round(trend_weakness, 4),
        "max_drawdown": round(drawdown, 4),
        "recent_drawdown": round(recent_drawdown, 4),
        "risk_level": risk_level,
        "interpretation": interpretation,
    }


def _max_drawdown(close: pd.Series) -> float:
    rolling_max = close.cummax()
    drawdown = (close - rolling_max) / rolling_max
    return float(abs(drawdown.min())) if len(drawdown) else 0.0


def _recent_drawdown(close: pd.Series, window: int = 30) -> float:
    segment = close.tail(window)
    if len(segment) < 2:
        return 0.0
    peak = segment.max()
    trough = segment.min()
    return float((peak - trough) / peak) if peak else 0.0


def _trend_weakness(price: float, ma20: float | None, ma50: float | None) -> float:
    """Score 0-1 where higher means weaker trend structure."""
    if ma20 is None or ma50 is None:
        return 0.5
    if price < ma20 < ma50:
        return 0.85
    if price < ma20 or price < ma50:
        return 0.55
    if price > ma20 > ma50:
        return 0.15
    return 0.35


def _classify_risk(volatility: float, trend_weakness: float, recent_drawdown: float) -> str:
    score = 0
    if volatility >= VOLATILITY_HIGH:
        score += 2
    elif volatility >= VOLATILITY_MODERATE:
        score += 1

    if recent_drawdown >= DRAWDOWN_HIGH:
        score += 2
    elif recent_drawdown >= DRAWDOWN_MODERATE:
        score += 1

    if trend_weakness >= 0.7:
        score += 2
    elif trend_weakness >= 0.45:
        score += 1

    if score >= 4:
        return "high"
    if score >= 2:
        return "moderate"
    return "low"


def _interpret_risk(level: str, vol: float, drawdown: float, weakness: float) -> str:
    return (
        f"Risk is classified as {level.upper()}. Annualized volatility is {vol:.1%}, "
        f"recent drawdown is {drawdown:.1%}, and trend weakness score is {weakness:.2f}."
    )
