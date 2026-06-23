"""Cross-evaluation table constants."""

from __future__ import annotations

CROSS_EVAL_MODELS: tuple[str, ...] = (
    "deepseek-v3.2",
    "deepseek-v4-flash",
    "glm-5.1",
    "glm-5.1-fp8",
    "qwen3.5-27b",
)


def expand_model_selection(selected: str | None) -> list[str]:
    """Return all cross-eval models, or a single model (including custom API names)."""
    if not selected or selected == "all":
        return list(CROSS_EVAL_MODELS)
    return [selected]


CROSS_EVAL_DEFENSES: tuple[str, ...] = (
    "sandwich",
    "spotlight",
    "struq",
    "secalign",
    "ours",
)

CROSS_EVAL_ATTACKS: tuple[str, ...] = (
    "injecagent_default",
    "chatinject",
    "topicattack",
    "adaptive_attack",
    "injecagent_enhanced",
    "combined",
    "important_instructions",
    "tool_knowledge",
    "ours_attack",
)

# Not part of the default Table 2 cross-evaluation grid, but needed for the
# no-defense Table 1 rows.
EXTRA_CROSS_EVAL_ATTACKS: tuple[str, ...] = ("naive",)
EXTRA_CROSS_EVAL_DEFENSES: tuple[str, ...] = ("none",)

# Display labels for CSV / tables.
#
# StruQ and SecAlign are deliberately suffixed with "-Prompt": the current
# implementation modifies only prompts and does not run the official trained
# StruQ/SecAlign models. Internal IDs stay unchanged for result compatibility.
ATTACK_LABELS: dict[str, str] = {
    "injecagent_default": "InjecAgent-Default",
    "chatinject": "ChatInject",
    "topicattack": "TopicAttack",
    "adaptive_attack": "Adaptive Attack",
    "injecagent_enhanced": "InjecAgent-Enhanced",
    "combined": "Combined",
    "important_instructions": "Important Instructions",
    "tool_knowledge": "Tool Knowledge",
    "ours_attack": "Ours-Attack",
    "naive": "Naive",
}

DEFENSE_LABELS: dict[str, str] = {
    "sandwich": "Sandwich",
    "spotlight": "Spotlighting",
    "struq": "StruQ-Prompt",
    "secalign": "SecAlign-Prompt",
    "ours": "Ours",
    "none": "No Defense",
}

DEFENSE_IMPLEMENTATION_NOTES: dict[str, str] = {
    "sandwich": "Prompt-level task repetition; no trained model.",
    "spotlight": "Prompt-level delimiting; no trained model.",
    "struq": (
        "Prompt-modified StruQ-inspired baseline; not the official StruQ "
        "model or structured-query training."
    ),
    "secalign": (
        "Prompt-modified SecAlign-inspired baseline; not the official "
        "SecAlign/Meta-SecAlign model."
    ),
    "ours": "Validation-selected evolved defense harness.",
    "none": "No added defense; InjecAgent base prompt only.",
}

OFFLINE_POISON_ATTACKS: frozenset[str] = frozenset(
    {"adaptive_attack"}
)

EXTERNAL_ARTIFACT_ATTACKS: frozenset[str] = frozenset({"ours_attack"})

TEMPLATE_ATTACKS: frozenset[str] = frozenset(
    {"combined", "important_instructions", "tool_knowledge"}
)
