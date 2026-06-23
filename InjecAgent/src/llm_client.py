import ast
import os
import sys
import time
import types
from pathlib import Path
from typing import Any

_TOOLS_ATTACK_ROOT = Path(__file__).resolve().parents[2]
if str(_TOOLS_ATTACK_ROOT) not in sys.path:
    sys.path.insert(0, str(_TOOLS_ATTACK_ROOT))

from paths import ensure_on_sys_path

ensure_on_sys_path()

def _load_model_api():
    """Load the legacy model_api module or an explicitly selected config file."""
    config_path = os.environ.get("TOOLS_ATTACK_MODEL_CONFIG", "").strip()
    if not config_path:
        import model_api

        return model_api

    path = Path(config_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"model config does not exist: {path}")
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    values: dict[str, Any] = {}
    for node in tree.body:
        if (
            isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
        ):
            try:
                values[node.targets[0].id] = ast.literal_eval(node.value)
            except (ValueError, TypeError):
                continue
    module = types.SimpleNamespace(**values)
    for required in ("OPENAI_BASE_URL", "OPENAI_API_KEY"):
        if not getattr(module, required, None):
            raise ValueError(f"model config is missing {required}: {path}")
    # Support both the legacy InjecAgent role names and the four-agent
    # contract_heldout_ipi naming used by model_api_ablation.py.
    if not hasattr(module, "VICTIM_MODEL"):
        module.VICTIM_MODEL = getattr(
            module,
            "WORK_AGENT",
            getattr(module, "DEFENSE_MODEL", getattr(module, "ATTACK_MODEL")),
        )
    if not hasattr(module, "ATTACK_MODEL"):
        module.ATTACK_MODEL = module.VICTIM_MODEL
    if not hasattr(module, "OPTIMIZER_MODEL"):
        module.OPTIMIZER_MODEL = getattr(
            module, "ATTACK_MODEL", module.VICTIM_MODEL
        )
    if not hasattr(module, "PROTECT_MODEL"):
        module.PROTECT_MODEL = getattr(
            module, "DEFENSE_MODEL", module.VICTIM_MODEL
        )
    return module


_model_api = _load_model_api()
OPENAI_API_KEY = _model_api.OPENAI_API_KEY
OPENAI_BASE_URL = _model_api.OPENAI_BASE_URL
OPTIMIZER_MODEL = _model_api.OPTIMIZER_MODEL
PROTECT_MODEL = _model_api.PROTECT_MODEL
VICTIM_MODEL = _model_api.VICTIM_MODEL

# Victim default for evaluate_* scripts (override with --model_name)
ATTACK_MODEL = VICTIM_MODEL

LLM_TEMPERATURE = 0.5
MAX_RETRIES = 3
RETRY_DELAY_SEC = 2
USE_STREAM = True


class _StreamFunction:
    def __init__(self, name: str = "", arguments: str = ""):
        self.name = name
        self.arguments = arguments

    def model_dump(self) -> dict[str, str]:
        return {"name": self.name, "arguments": self.arguments}


class _StreamToolCall:
    def __init__(self, id: str = "", name: str = "", arguments: str = ""):
        self.id = id
        self.type = "function"
        self.function = _StreamFunction(name, arguments)

    def model_dump(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "type": self.type,
            "function": self.function.model_dump(),
        }


class _StreamMessage:
    def __init__(
        self,
        content: str | None = None,
        tool_calls: list[_StreamToolCall] | None = None,
    ):
        self.content = content
        self.tool_calls = tool_calls


class _StreamChoice:
    def __init__(self, message: _StreamMessage):
        self.message = message


class _StreamCompletion:
    """Minimal completion object compatible with extract_chat_content / finetuned agent."""

    def __init__(
        self,
        content: str | None,
        tool_calls: list[_StreamToolCall] | None = None,
        completion_id: str | None = None,
    ):
        self.id = completion_id
        self.choices = [_StreamChoice(_StreamMessage(content, tool_calls))]


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


def _collect_stream_response(stream) -> tuple[str | None, list[_StreamToolCall] | None, str | None, str | None]:
    """Aggregate streaming chunks into content and optional tool_calls."""
    content_parts: list[str] = []
    tool_calls_by_index: dict[int, dict[str, Any]] = {}
    completion_id = None

    for chunk in stream:
        if completion_id is None:
            completion_id = getattr(chunk, "id", None)
        choices = getattr(chunk, "choices", None)
        if not choices:
            continue

        delta = getattr(choices[0], "delta", None)
        if delta is None:
            continue

        piece = getattr(delta, "content", None)
        if piece:
            content_parts.append(piece)

        for tc in getattr(delta, "tool_calls", None) or []:
            idx = tc.index
            entry = tool_calls_by_index.setdefault(
                idx,
                {"id": "", "type": "function", "function": {"name": "", "arguments": ""}},
            )
            if tc.id:
                entry["id"] = tc.id
            if tc.type:
                entry["type"] = tc.type
            fn = tc.function
            if fn:
                if fn.name:
                    entry["function"]["name"] += fn.name
                if fn.arguments:
                    entry["function"]["arguments"] += fn.arguments

    content = "".join(content_parts) if content_parts else None
    tool_calls = None
    if tool_calls_by_index:
        tool_calls = [
            _StreamToolCall(
                id=tool_calls_by_index[i]["id"],
                name=tool_calls_by_index[i]["function"]["name"],
                arguments=tool_calls_by_index[i]["function"]["arguments"],
            )
            for i in sorted(tool_calls_by_index)
        ]

    if content is None and not tool_calls:
        cid = completion_id or ""
        return None, None, completion_id, f"API returned empty stream (completion_id={cid})"

    return content if content is not None else "", tool_calls, completion_id, None


def chat_completion(messages, model=None, tools=None, temperature=None, stream: bool | None = None):
    kwargs = {
        "model": model or ATTACK_MODEL,
        "messages": messages,
        "temperature": LLM_TEMPERATURE if temperature is None else temperature,
    }
    if tools is not None:
        kwargs["tools"] = tools

    use_stream = USE_STREAM if stream is None else stream
    if use_stream:
        kwargs["stream"] = True
        stream_resp = get_openai_client().chat.completions.create(**kwargs)
        content, tool_calls, completion_id, error = _collect_stream_response(stream_resp)
        if error:
            raise ValueError(error)
        return _StreamCompletion(content, tool_calls, completion_id)

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

    tool_calls = getattr(message, "tool_calls", None)
    content = getattr(message, "content", None)
    if content is None and tool_calls:
        return "", None
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
    Call chat completion with retries (streaming by default for API-key calls).
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
