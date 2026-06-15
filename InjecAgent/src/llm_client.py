import sys
import time
from pathlib import Path
from typing import Any

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from model_api import ATTACK_MODEL, OPENAI_API_KEY, OPENAI_BASE_URL

try:
    from model_api import OPTIMIZER_MODEL
except ImportError:
    OPTIMIZER_MODEL = ATTACK_MODEL

LLM_TEMPERATURE = 0.5
MAX_RETRIES = 3
RETRY_DELAY_SEC = 2

_client = None


def get_openai_client():
    global _client
    if _client is None:
        from openai import OpenAI

        _client = OpenAI(
            api_key=OPENAI_API_KEY,
            base_url=OPENAI_BASE_URL,
        )
    return _client


def chat_completion(messages, model=None, tools=None, temperature=None):
    kwargs = {
        "model": model or ATTACK_MODEL,
        "messages": messages,
        "temperature": LLM_TEMPERATURE if temperature is None else temperature,
    }
    if tools is not None:
        kwargs["tools"] = tools
    response = get_openai_client().chat.completions.create(**kwargs)
    return response


def extract_chat_content(completion) -> tuple[str | None, str | None]:
    """Return (content, error_message). content may be empty string."""
    if completion is None:
        return None, "API returned None completion"

    choices = getattr(completion, "choices", None)
    if not choices:
        completion_id = getattr(completion, "id", None)
        return None, f"API returned empty/null choices (completion_id={completion_id})"

    message = getattr(choices[0], "message", None)
    if message is None:
        return None, "API choice has no message object"

    content = getattr(message, "content", None)
    if content is None:
        return "", None
    return content, None


def chat_completion_with_retry(
    messages,
    model=None,
    tools=None,
    temperature=None,
    max_retries=MAX_RETRIES,
    label="LLM",
):
    """
    Call chat completion with retries.
    Returns (content, error, raw_response).
    On failure, content is None and error describes the last failure.
    """
    last_error = "unknown API failure"
    for attempt in range(1, max_retries + 1):
        try:
            response = chat_completion(
                messages=messages,
                model=model,
                tools=tools,
                temperature=temperature,
            )
            content, parse_error = extract_chat_content(response)
            if parse_error is None:
                return content, None, response
            last_error = parse_error
            print(f"[{label}] attempt {attempt}/{max_retries} invalid response: {parse_error}")
        except Exception as exc:
            last_error = f"{type(exc).__name__}: {exc}"
            print(f"[{label}] attempt {attempt}/{max_retries} API error: {last_error}")

        if attempt < max_retries:
            time.sleep(RETRY_DELAY_SEC * attempt)

    return None, last_error, None
