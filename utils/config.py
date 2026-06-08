"""Application configuration and environment variables."""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

ROOT_DIR = Path(__file__).resolve().parent.parent
load_dotenv(ROOT_DIR / ".env")

# Paths
CACHE_DIR = ROOT_DIR / "data" / "cache"
ASSETS_DIR = ROOT_DIR / "assets"
STYLE_CSS = ASSETS_DIR / "style.css"

# Data defaults
DEFAULT_TICKER = "AAPL"
DEFAULT_PERIOD = "6mo"
DEFAULT_INTERVAL = "1d"
MAX_NEWS_ITEMS = 15

# FinBERT
FINBERT_MODEL = os.getenv("FINBERT_MODEL", "ProsusAI/finbert")
SENTIMENT_BATCH_SIZE = int(os.getenv("SENTIMENT_BATCH_SIZE", "8"))

# DeepSeek (final conversational synthesis layer)
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "")
DEEPSEEK_BASE_URL = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
DEEPSEEK_MODEL = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")

# Retry
MAX_RETRIES = int(os.getenv("MAX_RETRIES", "3"))
RETRY_DELAY_SECONDS = float(os.getenv("RETRY_DELAY_SECONDS", "1.5"))

# UI
APP_TITLE = "AI Stock Research Assistant"
PAGE_ICON = "📈"

# Risk thresholds
VOLATILITY_HIGH = 0.35
VOLATILITY_MODERATE = 0.20
DRAWDOWN_HIGH = 0.15
DRAWDOWN_MODERATE = 0.08

# Ensure cache directory exists
CACHE_DIR.mkdir(parents=True, exist_ok=True)
