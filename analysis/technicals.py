"""Technical indicator calculations and interpretation."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from utils.logger import get_logger

logger = get_logger("technicals")


def _rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0).rolling(window=period).mean()
    loss = (-delta.clip(upper=0)).rolling(window=period).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def _macd(series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> tuple[pd.Series, pd.Series, pd.Series]:
    ema_fast = series.ewm(span=fast, adjust=False).mean()
    ema_slow = series.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram


def compute_technicals(history: pd.DataFrame) -> dict[str, Any]:
    """Compute RSI, moving averages, MACD, and volatility."""
    logger.info("Computing technical indicators")
    df = history.copy()
    if "Close" not in df.columns:
        raise ValueError("History DataFrame must include a Close column")

    close = df["Close"]
    df["MA20"] = close.rolling(window=20).mean()
    df["MA50"] = close.rolling(window=50).mean()
    df["RSI"] = _rsi(close)
    macd_line, signal_line, histogram = _macd(close)
    df["MACD"] = macd_line
    df["MACD_Signal"] = signal_line
    df["MACD_Hist"] = histogram
    df["Returns"] = close.pct_change()
    df["Volatility_20d"] = df["Returns"].rolling(window=20).std() * np.sqrt(252)

    latest = df.iloc[-1]
    rsi_val = float(latest["RSI"]) if pd.notna(latest["RSI"]) else None
    ma20 = float(latest["MA20"]) if pd.notna(latest["MA20"]) else None
    ma50 = float(latest["MA50"]) if pd.notna(latest["MA50"]) else None
    macd_val = float(latest["MACD"]) if pd.notna(latest["MACD"]) else None
    signal_val = float(latest["MACD_Signal"]) if pd.notna(latest["MACD_Signal"]) else None
    vol = float(latest["Volatility_20d"]) if pd.notna(latest["Volatility_20d"]) else None
    price = float(latest["Close"])

    interpretation = _interpret_technicals(price, rsi_val, ma20, ma50, macd_val, signal_val)

    return {
        "dataframe": df,
        "latest": {
            "rsi": rsi_val,
            "ma20": ma20,
            "ma50": ma50,
            "macd": macd_val,
            "macd_signal": signal_val,
            "volatility_annualized": vol,
            "price": price,
        },
        "interpretation": interpretation,
    }


def _interpret_technicals(
    price: float,
    rsi: float | None,
    ma20: float | None,
    ma50: float | None,
    macd: float | None,
    signal: float | None,
) -> str:
    """Generate a short technical narrative."""
    signals: list[str] = []

    if rsi is not None:
        if rsi >= 70:
            signals.append(f"RSI at {rsi:.1f} suggests overbought conditions.")
        elif rsi <= 30:
            signals.append(f"RSI at {rsi:.1f} suggests oversold conditions.")
        else:
            signals.append(f"RSI at {rsi:.1f} is in neutral territory.")

    if ma20 and ma50:
        if price > ma20 > ma50:
            signals.append("Price trades above rising short- and medium-term averages (bullish structure).")
        elif price < ma20 < ma50:
            signals.append("Price trades below declining averages (bearish structure).")
        else:
            signals.append("Moving averages are mixed, indicating consolidation or trend transition.")

    if macd is not None and signal is not None:
        if macd > signal:
            signals.append("MACD is above its signal line, supporting positive momentum.")
        else:
            signals.append("MACD is below its signal line, indicating weakening momentum.")

    return " ".join(signals) if signals else "Insufficient data for technical interpretation."
