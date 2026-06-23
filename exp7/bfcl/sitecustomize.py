"""
BFCL runtime patch: register tools_attack OpenAI-compatible models with optional
system addendum (max_defense). Activated via PYTHONPATH=exp7/bfcl.
"""

from __future__ import annotations

import json
from pathlib import Path

_OVERRIDES = Path(__file__).resolve().parent / "model_overrides.json"


def _load_overrides() -> dict:
    if not _OVERRIDES.exists():
        return {}
    return json.loads(_OVERRIDES.read_text(encoding="utf-8"))


def _patch_bfcl() -> None:
    try:
        from bfcl_eval.constants import model_config
        from bfcl_eval.constants.model_config import ModelConfig
        from bfcl_eval.model_handler.api_inference.openai_completion import (
            OpenAICompletionsHandler,
        )
    except ImportError:
        return

    overrides = _load_overrides()
    if not overrides:
        return

    class ToolsAttackOpenAIHandler(OpenAICompletionsHandler):
        def _pre_query_processing_prompting(self, test_entry, functions, **kwargs):
            messages = super()._pre_query_processing_prompting(
                test_entry, functions, **kwargs
            )
            addendum = getattr(self, "_tools_attack_addendum", "")
            if not addendum:
                return messages
            if messages and messages[0].get("role") == "system":
                messages[0]["content"] = messages[0]["content"].rstrip() + "\n\n" + addendum
            else:
                messages.insert(0, {"role": "system", "content": addendum})
            return messages

        def _pre_query_processing_fc(self, test_entry, functions, **kwargs):
            messages = super()._pre_query_processing_fc(test_entry, functions, **kwargs)
            addendum = getattr(self, "_tools_attack_addendum", "")
            if not addendum:
                return messages
            if messages and messages[0].get("role") == "system":
                messages[0]["content"] = messages[0]["content"].rstrip() + "\n\n" + addendum
            else:
                messages.insert(0, {"role": "system", "content": addendum})
            return messages

    handler_map = {
        "OpenAICompletionsHandler": ToolsAttackOpenAIHandler,
    }

    for registry_name, cfg in overrides.items():
        base_cls = handler_map.get(cfg.get("model_handler", "OpenAICompletionsHandler"))
        if base_cls is None:
            continue

        addendum = cfg.get("system_addendum") or ""

        class _BoundHandler(base_cls):  # type: ignore[misc, valid-type]
            def __init__(self, model_name, temperature, registry_name, is_fc_model, **kwargs):
                super().__init__(model_name, temperature, registry_name, is_fc_model, **kwargs)
                self._tools_attack_addendum = addendum

        entry = ModelConfig(
            model_name=cfg["model_name"],
            display_name=cfg.get("display_name", registry_name),
            url="tools_attack",
            org="tools_attack",
            license="internal",
            model_handler=_BoundHandler,
            is_fc_model=bool(cfg.get("is_fc_model", True)),
        )
        model_config.api_inference_model_map[registry_name] = entry
        model_config.MODEL_CONFIG_MAPPING[registry_name] = entry
        # BFCL evaluate replaces "_" with "/" when resolving handlers from result dirs.
        escaped = registry_name.replace("_", "/")
        model_config.api_inference_model_map[escaped] = entry
        model_config.MODEL_CONFIG_MAPPING[escaped] = entry


_patch_bfcl()
