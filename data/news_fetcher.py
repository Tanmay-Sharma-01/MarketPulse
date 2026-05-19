"""Recent stock news via yfinance."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

import yfinance as yf

from utils.config import MAX_NEWS_ITEMS, MAX_RETRIES
from utils.helpers import retry
from utils.logger import get_logger

logger = get_logger("news_fetcher")


def _parse_timestamp(raw: Any) -> str:
    """Convert unix timestamp or string to readable date."""
    if raw is None:
        return "Unknown"
    try:
        if isinstance(raw, (int, float)):
            return datetime.fromtimestamp(raw, tz=timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
        return str(raw)[:19]
    except (OSError, ValueError, OverflowError):
        return "Unknown"


@retry(max_attempts=MAX_RETRIES)
def fetch_recent_news(ticker: str, limit: int = MAX_NEWS_ITEMS) -> list[dict[str, Any]]:
    """Fetch recent news articles for a ticker."""
    logger.info("Fetching news for %s (limit=%s)", ticker, limit)
    stock = yf.Ticker(ticker.upper())
    raw_news = stock.news or []

    articles: list[dict[str, Any]] = []
    for item in raw_news[:limit]:
        content = item.get("content") or item
        title = content.get("title") or item.get("title", "Untitled")
        summary = content.get("summary") or content.get("description") or item.get("summary", "")
        publisher = (
            (content.get("provider") or {}).get("displayName")
            or content.get("publisher")
            or item.get("publisher", "Unknown")
        )
        url = content.get("canonicalUrl") or content.get("clickThroughUrl") or item.get("link", "")
        if isinstance(url, dict):
            url = url.get("url", "")
        pub_date = _parse_timestamp(
            content.get("pubDate") or content.get("displayTime") or item.get("providerPublishTime")
        )

        articles.append(
            {
                "title": title,
                "publisher": publisher,
                "summary": summary or "No summary available.",
                "url": url if isinstance(url, str) else "",
                "publish_date": pub_date,
            }
        )

    if not articles:
        logger.warning("No news found for %s", ticker)

    return articles
