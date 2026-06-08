"""Local-first LLM loader for LangChain agents.

This project prefers local inference. We use a small instruction model via
HuggingFace Transformers. If model load fails, callers should fall back to
deterministic templated output.
"""

from __future__ import annotations

import os
from functools import lru_cache
from typing import Optional

from utils.logger import get_logger

logger = get_logger("agents.llm")


DEFAULT_CHAT_MODEL = os.getenv("LOCAL_CHAT_MODEL", "google/flan-t5-base")


@lru_cache(maxsize=1)
def get_llm() -> Optional[object]:
    """Return a LangChain-compatible LLM instance or None if unavailable."""
    try:
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, pipeline
        from langchain_community.llms import HuggingFacePipeline

        model_id = DEFAULT_CHAT_MODEL
        logger.info("Loading local chat model: %s", model_id)
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_id)
        gen = pipeline(
            "text2text-generation",
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=512,
            do_sample=False,
            temperature=0.0,
        )
        return HuggingFacePipeline(pipeline=gen)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Local LLM unavailable; falling back to templates. Error: %s", exc)
        return None

