"""Stock price and company metadata via yfinance."""

from __future__ import annotations

from typing import Any

import pandas as pd
import yfinance as yf

from utils.config import MAX_RETRIES
from utils.helpers import retry, safe_float
from utils.logger import get_logger

logger = get_logger("stock_data")


@retry(max_attempts=MAX_RETRIES)
def fetch_price_history(ticker: str, period: str = "6mo", interval: str = "1d") -> pd.DataFrame:
    """Download OHLCV history for a ticker."""
    logger.info("Fetching price history for %s (%s)", ticker, period)
    stock = yf.Ticker(ticker.upper())
    df = stock.history(period=period, interval=interval, auto_adjust=True)
    if df.empty:
        raise ValueError(f"No price data returned for {ticker}")
    df = df.reset_index()
    if "Date" not in df.columns and "Datetime" in df.columns:
        df = df.rename(columns={"Datetime": "Date"})
    return df


@retry(max_attempts=MAX_RETRIES)
def fetch_company_info(ticker: str) -> dict[str, Any]:
    """Fetch company metadata from yfinance."""
    logger.info("Fetching company info for %s", ticker)
    stock = yf.Ticker(ticker.upper())
    info = stock.info or {}

    fifty_two_high = safe_float(info.get("fiftyTwoWeekHigh"))
    fifty_two_low = safe_float(info.get("fiftyTwoWeekLow"))
    current_price = safe_float(
        info.get("currentPrice") or info.get("regularMarketPrice") or info.get("previousClose")
    )

    revenue_growth = safe_float(info.get("revenueGrowth"))
    if revenue_growth is None:
        revenue_growth = safe_float(info.get("earningsGrowth"))

    return {
        "ticker": ticker.upper(),
        "name": info.get("longName") or info.get("shortName") or ticker.upper(),
        "sector": info.get("sector", "N/A"),
        "industry": info.get("industry", "N/A"),
        "market_cap": safe_float(info.get("marketCap")),
        "pe_ratio": safe_float(info.get("trailingPE") or info.get("forwardPE")),
        "current_price": current_price,
        "previous_close": safe_float(info.get("previousClose")),
        "fifty_two_week_high": fifty_two_high,
        "fifty_two_week_low": fifty_two_low,
        "revenue_growth": revenue_growth,
        "beta": safe_float(info.get("beta")),
        "description": (info.get("longBusinessSummary") or "")[:500],
        "currency": info.get("currency", "USD"),
        "exchange": info.get("exchange", "N/A"),
    }


def compute_daily_change(info: dict[str, Any], history: pd.DataFrame) -> dict[str, float | None]:
    """Compute daily price change from history or metadata."""
    change_pct: float | None = None
    change_abs: float | None = None

    if len(history) >= 2 and "Close" in history.columns:
        last_close = float(history["Close"].iloc[-1])
        prev_close = float(history["Close"].iloc[-2])
        change_abs = last_close - prev_close
        change_pct = change_abs / prev_close if prev_close else None
        current = last_close
    else:
        current = info.get("current_price")
        prev = info.get("previous_close")
        if current and prev:
            change_abs = current - prev
            change_pct = change_abs / prev if prev else None

    return {
        "current_price": float(history["Close"].iloc[-1]) if len(history) and "Close" in history.columns else info.get("current_price"),
        "daily_change_abs": change_abs,
        "daily_change_pct": change_pct,
    }


def get_stock_package(ticker: str, period: str = "6mo", interval: str = "1d") -> dict[str, Any]:
    """Return combined stock data package."""
    history = fetch_price_history(ticker, period=period, interval=interval)
    info = fetch_company_info(ticker)
    price_metrics = compute_daily_change(info, history)

    return {
        "info": {**info, **price_metrics},
        "history": history,
    }
