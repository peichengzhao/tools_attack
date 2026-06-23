from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path

from exp7.config import BFCL_TEST_CATEGORIES, METRIC_LABELS, EXP7_ROOT
from exp7.harness import build_unified_system_addendum
from exp7.runners._common import make_result, openai_env, require_command, run_cmd, write_result

BFCL_PATCH_ROOT = EXP7_ROOT / "bfcl"


def run(*, model: str, harness: str, force: bool = False, limit: int = 0) -> dict:
    metric = METRIC_LABELS["bfcl"]
    require_command("bfcl", "Install: pip install 'bfcl-eval==2025.12.17'")

    registry_name = _registry_name(model, harness)
    _ensure_bfcl_model_registered(registry_name, model, harness)

    env = openai_env(model)
    env["TOOLS_ATTACK_HARNESS"] = harness
    env["TOOLS_ATTACK_BFCL_MODEL"] = model
    env["PYTHONPATH"] = os.pathsep.join(
        [str(BFCL_PATCH_ROOT), env.get("PYTHONPATH", "")]
    )

    categories = list(BFCL_TEST_CATEGORIES)
    if limit > 0:
        categories = categories[:limit]

    for category in categories:
        gen_cmd = [
            "bfcl",
            "generate",
            "--model",
            registry_name,
            "--test-category",
            category,
            "--num-threads",
            "1",
            "--skip-server-setup",
        ]
        if force:
            gen_cmd.append("--allow-overwrite")
        proc = run_cmd(gen_cmd, env=env)
        if proc.returncode != 0:
            payload = make_result(
                benchmark="bfcl",
                model=model,
                harness=harness,
                utility=None,
                metric_name=metric,
                status="error",
                error=f"generate failed for {category}: {(proc.stderr or proc.stdout)[-1500:]}",
            )
            write_result(payload)
            return payload

    eval_scores: dict[str, float] = {}
    for category in categories:
        ev_cmd = ["bfcl", "evaluate", "--model", registry_name, "--test-category", category]
        proc = run_cmd(ev_cmd, env=env)
        if proc.returncode != 0:
            payload = make_result(
                benchmark="bfcl",
                model=model,
                harness=harness,
                utility=None,
                metric_name=metric,
                status="error",
                error=f"evaluate failed for {category}: {(proc.stderr or proc.stdout)[-1500:]}",
            )
            write_result(payload)
            return payload
        score = _parse_bfcl_accuracy(proc.stdout or "")
        if score is not None:
            eval_scores[category] = score

    utility = (
        sum(eval_scores.values()) / len(eval_scores)
        if eval_scores
        else None
    )
    payload = make_result(
        benchmark="bfcl",
        model=model,
        harness=harness,
        utility=utility,
        metric_name=metric,
        details={"categories": eval_scores, "registry_name": registry_name},
    )
    write_result(payload)
    return payload


def _registry_name(model: str, harness: str) -> str:
    safe = model.replace("/", "_").replace(".", "_")
    return f"tools_attack_{safe}_{harness}"


def _ensure_bfcl_model_registered(registry_name: str, model: str, harness: str) -> None:
    """Write BFCL user override map consumed by exp7/bfcl/sitecustomize.py."""
    overrides = BFCL_PATCH_ROOT / "model_overrides.json"
    overrides.parent.mkdir(parents=True, exist_ok=True)
    data = {}
    if overrides.exists():
        data = json.loads(overrides.read_text(encoding="utf-8"))

    data[registry_name] = {
        "model_name": model,
        "display_name": f"tools_attack/{model}/{harness}",
        "harness": harness,
        "system_addendum": build_unified_system_addendum() if harness == "max_defense" else "",
        "model_handler": "OpenAICompletionsHandler",
        "is_fc_model": True,
    }
    overrides.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _parse_bfcl_accuracy(text: str) -> float | None:
    patterns = (
        r"Overall accuracy[:\s]+([0-9]+(?:\.[0-9]+)?)\s*%",
        r"Accuracy[:\s]+([0-9]+(?:\.[0-9]+)?)\s*%",
        r"accuracy[:\s]+([0-9]+(?:\.[0-9]+)?)",
    )
    for pat in patterns:
        m = re.search(pat, text, re.IGNORECASE)
        if m:
            val = float(m.group(1))
            return val * 100 if val <= 1.0 else val
    return None
