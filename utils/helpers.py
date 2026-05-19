"""Shared utility helpers."""

from __future__ import annotations

import time
from functools import wraps
from typing import Any, Callable, TypeVar

from utils.config import RETRY_DELAY_SECONDS
from utils.logger import get_logger

logger = get_logger("helpers")

F = TypeVar("F", bound=Callable[..., Any])


def retry(max_attempts: int = 3, delay: float = RETRY_DELAY_SECONDS) -> Callable[[F], F]:
    """Retry decorator for flaky network calls."""

    def decorator(func: F) -> F:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            last_error: Exception | None = None
            for attempt in range(1, max_attempts + 1):
                try:
                    return func(*args, **kwargs)
                except Exception as exc:  # noqa: BLE001
                    last_error = exc
                    logger.warning(
                        "%s failed (attempt %s/%s): %s",
                        func.__name__,
                        attempt,
                        max_attempts,
                        exc,
                    )
                    if attempt < max_attempts:
                        time.sleep(delay * attempt)
            raise RuntimeError(f"{func.__name__} failed after {max_attempts} attempts") from last_error

        return wrapper  # type: ignore[return-value]

    return decorator


def format_currency(value: float | None, decimals: int = 2) -> str:
    """Format a number as USD currency."""
    if value is None:
        return "N/A"
    return f"${value:,.{decimals}f}"


def format_percent(value: float | None, decimals: int = 2, signed: bool = True) -> str:
    """Format a decimal ratio as a percentage string."""
    if value is None:
        return "N/A"
    pct = value * 100
    prefix = "+" if signed and pct > 0 else ""
    return f"{prefix}{pct:.{decimals}f}%"


def format_large_number(value: float | None) -> str:
    """Format large numbers (e.g. market cap) with B/M/K suffix."""
    if value is None:
        return "N/A"
    abs_val = abs(value)
    if abs_val >= 1e12:
        return f"${value / 1e12:.2f}T"
    if abs_val >= 1e9:
        return f"${value / 1e9:.2f}B"
    if abs_val >= 1e6:
        return f"${value / 1e6:.2f}M"
    if abs_val >= 1e3:
        return f"${value / 1e3:.2f}K"
    return f"${value:,.0f}"


def safe_float(value: Any, default: float | None = None) -> float | None:
    """Coerce a value to float safely."""
    try:
        if value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default
