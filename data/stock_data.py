"""Stock price and company metadata via yfinance."""

from __future__ import annotations

from typing import Any

import pandas as pd
import yfinance as yf

from utils.config import MAX_RETRIES
from utils.helpers import retry, safe_float
from utils.logger import get_logger

logger = get_logger("stock_data")


def _normalize_history(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize yfinance history output."""

    if df.empty:
        return df

    out = df.copy()

    # Flatten MultiIndex columns
    if isinstance(out.columns, pd.MultiIndex):
        out.columns = [
            c[0] if isinstance(c, tuple) else c
            for c in out.columns
        ]

    out = out.reset_index()

    # Normalize date column name
    for alias in ("Datetime", "index", "level_0"):
        if alias in out.columns and "Date" not in out.columns:
            out = out.rename(columns={alias: "Date"})

    if "Date" in out.columns:
        out["Date"] = pd.to_datetime(
            out["Date"],
            errors="coerce",
        )

        try:
            out["Date"] = out["Date"].dt.tz_localize(None)
        except Exception:
            pass

    return out


def _download_history(
    ticker: str,
    period: str,
    interval: str,
) -> pd.DataFrame:
    """Download stock history safely."""

    symbol = ticker.upper().strip()

    if not symbol:
        raise ValueError("Ticker symbol is empty")

    logger.info(
        "Downloading %s (%s, %s)",
        symbol,
        period,
        interval,
    )

    # PRIMARY METHOD
    try:
        df = yf.download(
            symbol,
            period=period,
            interval=interval,
            auto_adjust=True,
            progress=False,
            threads=False,
        )

        df = _normalize_history(df)

        if not df.empty:
            return df

        logger.warning(
            "yf.download returned empty dataframe for %s",
            symbol,
        )

    except Exception as exc:
        logger.warning(
            "yf.download failed for %s: %s",
            symbol,
            exc,
        )

    # FALLBACK METHOD
    try:
        stock = yf.Ticker(symbol)

        df = stock.history(
            period=period,
            interval=interval,
            auto_adjust=True,
        )

        df = _normalize_history(df)

        if not df.empty:
            return df

    except Exception as exc:
        logger.warning(
            "Ticker.history failed for %s: %s",
            symbol,
            exc,
        )

    raise ValueError(
        f"No price data returned for {symbol}. "
        "Yahoo Finance may be rate-limiting requests."
    )


@retry(max_attempts=MAX_RETRIES)
def fetch_price_history(
    ticker: str,
    period: str = "6mo",
    interval: str = "1d",
) -> pd.DataFrame:
    return _download_history(
        ticker,
        period,
        interval,
    )


@retry(max_attempts=MAX_RETRIES)
def fetch_company_info(
    ticker: str,
) -> dict[str, Any]:
    """Fetch company metadata."""

    symbol = ticker.upper().strip()

    stock = yf.Ticker(symbol)

    info = stock.info or {}

    fifty_two_high = safe_float(
        info.get("fiftyTwoWeekHigh")
    )

    fifty_two_low = safe_float(
        info.get("fiftyTwoWeekLow")
    )

    current_price = safe_float(
        info.get("currentPrice")
        or info.get("regularMarketPrice")
        or info.get("previousClose")
    )

    revenue_growth = safe_float(
        info.get("revenueGrowth")
    )

    if revenue_growth is None:
        revenue_growth = safe_float(
            info.get("earningsGrowth")
        )

    return {
        "ticker": symbol,
        "name": (
            info.get("longName")
            or info.get("shortName")
            or symbol
        ),
        "sector": info.get("sector", "N/A"),
        "industry": info.get("industry", "N/A"),
        "market_cap": safe_float(info.get("marketCap")),
        "pe_ratio": safe_float(
            info.get("trailingPE")
            or info.get("forwardPE")
        ),
        "current_price": current_price,
        "previous_close": safe_float(
            info.get("previousClose")
        ),
        "fifty_two_week_high": fifty_two_high,
        "fifty_two_week_low": fifty_two_low,
        "revenue_growth": revenue_growth,
        "beta": safe_float(info.get("beta")),
        "description": (
            info.get("longBusinessSummary") or ""
        )[:500],
        "currency": info.get("currency", "USD"),
        "exchange": info.get("exchange", "N/A"),
    }


def compute_daily_change(
    info: dict[str, Any],
    history: pd.DataFrame,
) -> dict[str, float | None]:
    """Compute daily price changes."""

    change_pct = None
    change_abs = None

    if len(history) >= 2 and "Close" in history.columns:
        last_close = float(history["Close"].iloc[-1])
        prev_close = float(history["Close"].iloc[-2])

        change_abs = last_close - prev_close

        if prev_close:
            change_pct = change_abs / prev_close

    else:
        current = info.get("current_price")
        prev = info.get("previous_close")

        if current and prev:
            change_abs = current - prev

            if prev:
                change_pct = change_abs / prev

    return {
        "current_price": (
            float(history["Close"].iloc[-1])
            if len(history) and "Close" in history.columns
            else info.get("current_price")
        ),
        "daily_change_abs": change_abs,
        "daily_change_pct": change_pct,
    }


def get_stock_package(
    ticker: str,
    period: str = "6mo",
    interval: str = "1d",
) -> dict[str, Any]:
    """Return combined stock package."""

    history = fetch_price_history(
        ticker,
        period=period,
        interval=interval,
    )

    info = fetch_company_info(ticker)

    price_metrics = compute_daily_change(
        info,
        history,
    )

    return {
        "info": {
            **info,
            **price_metrics,
        },
        "history": history,
    }