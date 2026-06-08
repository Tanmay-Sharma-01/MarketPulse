"""Lightweight session-local conversation memory."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class SessionMemory:
    """Keep minimal state about the conversation in-memory (no external DB)."""

    messages: list[dict[str, str]] = field(default_factory=list)
    last_tickers: list[str] = field(default_factory=list)
    last_conclusion: str | None = None

    def add_user(self, content: str) -> None:
        self.messages.append({"role": "user", "content": content})

    def add_assistant(self, content: str) -> None:
        self.messages.append({"role": "assistant", "content": content})
        self.last_conclusion = content[:800]

    def to_dict(self) -> dict[str, Any]:
        return {
            "messages": self.messages[-20:],  # keep it small
            "last_tickers": self.last_tickers[-5:],
            "last_conclusion": self.last_conclusion,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "SessionMemory":
        if not data:
            return cls()
        return cls(
            messages=list(data.get("messages", [])),
            last_tickers=list(data.get("last_tickers", [])),
            last_conclusion=data.get("last_conclusion"),
        )

