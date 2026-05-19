"""FinBERT-based news sentiment analysis."""

from __future__ import annotations

from typing import Any

from utils.config import FINBERT_MODEL, SENTIMENT_BATCH_SIZE
from utils.logger import get_logger

logger = get_logger("sentiment")

_pipeline = None


def _get_pipeline():
    """Lazy-load FinBERT sentiment pipeline."""
    global _pipeline  # noqa: PLW0603
    if _pipeline is not None:
        return _pipeline

    logger.info("Loading FinBERT model: %s", FINBERT_MODEL)
    from transformers import pipeline

    _pipeline = pipeline(
        "sentiment-analysis",
        model=FINBERT_MODEL,
        tokenizer=FINBERT_MODEL,
        truncation=True,
        max_length=512,
    )
    return _pipeline


def _normalize_label(label: str) -> str:
    """Map FinBERT labels to positive/negative/neutral."""
    label_lower = label.lower()
    if "positive" in label_lower:
        return "positive"
    if "negative" in label_lower:
        return "negative"
    return "neutral"


def analyze_news_sentiment(articles: list[dict[str, Any]]) -> dict[str, Any]:
    """Analyze sentiment for news headlines."""
    headlines = [a.get("title", "") for a in articles if a.get("title")]
    if not headlines:
        logger.warning("No headlines to analyze")
        return _empty_sentiment()

    pipe = _get_pipeline()
    results: list[dict[str, Any]] = []

    for i in range(0, len(headlines), SENTIMENT_BATCH_SIZE):
        batch = headlines[i : i + SENTIMENT_BATCH_SIZE]
        try:
            preds = pipe(batch)
        except Exception as exc:  # noqa: BLE001
            logger.error("Sentiment batch failed: %s", exc)
            preds = [{"label": "neutral", "score": 0.5} for _ in batch]

        for headline, pred in zip(batch, preds, strict=True):
            label = _normalize_label(pred.get("label", "neutral"))
            score = float(pred.get("score", 0.5))
            signed_score = score if label == "positive" else (-score if label == "negative" else 0.0)
            results.append(
                {
                    "headline": headline,
                    "label": label,
                    "confidence": score,
                    "signed_score": signed_score,
                }
            )

    return summarize_sentiment(results)


def _empty_sentiment() -> dict[str, Any]:
    return {
        "items": [],
        "counts": {"positive": 0, "negative": 0, "neutral": 0},
        "average_score": 0.0,
        "bullish_confidence": 0.0,
        "bearish_confidence": 0.0,
        "overall_label": "neutral",
        "interpretation": "No news headlines available for sentiment analysis.",
    }


def summarize_sentiment(items: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate per-headline sentiment into summary metrics."""
    counts = {"positive": 0, "negative": 0, "neutral": 0}
    for item in items:
        counts[item["label"]] = counts.get(item["label"], 0) + 1

    total = len(items) or 1
    avg_score = sum(i["signed_score"] for i in items) / total

    pos_scores = [i["confidence"] for i in items if i["label"] == "positive"]
    neg_scores = [i["confidence"] for i in items if i["label"] == "negative"]

    bullish = sum(pos_scores) / len(pos_scores) if pos_scores else 0.0
    bearish = sum(neg_scores) / len(neg_scores) if neg_scores else 0.0

    if counts["positive"] > counts["negative"]:
        overall = "positive"
    elif counts["negative"] > counts["positive"]:
        overall = "negative"
    else:
        overall = "neutral"

    interpretation = _interpret_sentiment(overall, avg_score, counts, total)

    return {
        "items": items,
        "counts": counts,
        "average_score": round(avg_score, 4),
        "bullish_confidence": round(bullish, 4),
        "bearish_confidence": round(bearish, 4),
        "overall_label": overall,
        "interpretation": interpretation,
    }


def _interpret_sentiment(
    overall: str,
    avg_score: float,
    counts: dict[str, int],
    total: int,
) -> str:
    pct_pos = counts.get("positive", 0) / total * 100
    pct_neg = counts.get("negative", 0) / total * 100

    if overall == "positive":
        tone = "predominantly bullish"
    elif overall == "negative":
        tone = "predominantly bearish"
    else:
        tone = "mixed to neutral"

    return (
        f"News flow appears {tone} with {pct_pos:.0f}% positive and {pct_neg:.0f}% negative headlines. "
        f"Average directional score: {avg_score:+.2f}."
    )
