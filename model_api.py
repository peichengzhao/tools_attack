"""OpenAI-compatible model configuration loaded from environment variables."""

from __future__ import annotations

import os

try:
    from model_api_secret import (  # type: ignore
        OPENAI_API_KEY as _SECRET_KEY,
        OPENAI_BASE_URL as _SECRET_BASE_URL,
    )
except ImportError:
    _SECRET_KEY = ""
    _SECRET_BASE_URL = ""

OPENAI_BASE_URL = os.getenv(
    "OPENAI_BASE_URL",
    _SECRET_BASE_URL or "https://api.shredder.money/v1",
)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", _SECRET_KEY)

VICTIM_MODEL = os.getenv("VICTIM_MODEL", "deepseek-v4-flash")
OPTIMIZER_MODEL = os.getenv("OPTIMIZER_MODEL", VICTIM_MODEL)
PROTECT_MODEL = os.getenv("PROTECT_MODEL", VICTIM_MODEL)
WORK_AGENT = os.getenv("WORK_AGENT", "gpt-5.5")

# Backward-compatible alias used by older scripts.
ATTACK_MODEL = os.getenv("ATTACK_MODEL", VICTIM_MODEL)

try:
    from langchain_openai import ChatOpenAI

    llm = (
        ChatOpenAI(
            model=os.getenv("CHAT_MODEL", "nemotron-3-ultra"),
            temperature=0.5,
            base_url=OPENAI_BASE_URL,
            api_key=OPENAI_API_KEY,
            streaming=True,
        )
        if OPENAI_API_KEY
        else None
    )
except ImportError:
    llm = None


if __name__ == "__main__":
    if llm is None:
        raise RuntimeError(
            "Set OPENAI_API_KEY and install langchain_openai to use llm.invoke"
        )
    print(llm.invoke("who are you"))
