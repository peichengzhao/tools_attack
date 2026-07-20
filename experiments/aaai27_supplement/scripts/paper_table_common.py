"""Shared paths and command launch for the paper table runners."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
EXPERIMENT_ROOT = SCRIPT_DIR.parent
REPO_ROOT = EXPERIMENT_ROOT.parents[1]
INJECAGENT_ROOT = REPO_ROOT / "InjecAgent"

TEST_CASES = INJECAGENT_ROOT / "data" / "splits" / "test_cases_211.json"
TRAIN_CASES = INJECAGENT_ROOT / "data" / "splits" / "train_cases_632.json"
OURS_ATTACKS = EXPERIMENT_ROOT / "artifacts" / "frozen_test_attack_211.jsonl"
SELECTED_HARNESS = EXPERIMENT_ROOT / "artifacts" / "selected_harness.json"
CHATINJECT_JSON = (
    REPO_ROOT
    / "related_work"
    / "ChatInject"
    / "src"
    / "agentdojo"
    / "attacks"
    / "multi_turn_data"
    / "with_utility_multi_turn_7_generation_output.json"
)
ADAPTIVE_CACHE = INJECAGENT_ROOT / "data" / "cross_eval" / "offline_poison"

TABLE1_MODELS = (
    "deepseek-v4-flash",
    "deepseek-v4-pro",
    "glm-5.1",
    "nemotron-3-ultra",
    "qwen3.5-27b",
    "deepseek-v3.2",
)
TABLE1_ATTACKS = (
    "naive",
    "combined",
    "important_instructions",
    "tool_knowledge",
    "injecagent_default",
    "injecagent_enhanced",
    "topicattack",
    "chatinject",
    "adaptive_attack",
    "ours_attack",
)
TABLE2_MODELS = (
    "deepseek-v4-flash",
    "glm-5.1-fp8",
    "qwen3.5-27b",
)
TABLE2_DEFENSES = (
    "sandwich",
    "spotlight",
    "struq",
    "secalign",
    "ours",
)
TABLE2_ATTACKS = (
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
TABLE3A_MODELS = ("gpt-5.5", "deepseek-v4-flash", "glm-5.1")
TABLE3A_BENCHMARKS = (
    "agentdojo",
    "injecagent",
    "bfcl",
    "agentbench",
    "appworld",
)
TABLE3B_ROUNDS = (4, 5, 6, 7, 8, 9)


def project_python() -> str:
    override = os.environ.get("AAAI27_PROJECT_PYTHON")
    if override:
        return override
    return sys.executable


def adaptive_cache_count(model: str) -> int:
    path = ADAPTIVE_CACHE / model / "adaptive_attack_base.jsonl"
    if not path.is_file():
        return 0
    return sum(1 for line in path.read_text(encoding="utf-8").splitlines() if line.strip())


def attack_blockers(model: str) -> dict[str, str]:
    blocked: dict[str, str] = {}
    if not CHATINJECT_JSON.is_file():
        blocked["chatinject"] = f"missing ChatInject source: {CHATINJECT_JSON}"
    cached = adaptive_cache_count(model)
    if cached < 211:
        blocked["adaptive_attack"] = (
            f"adaptive cache for {model} has {cached}/211 cases under {ADAPTIVE_CACHE}"
        )
    return blocked


def cross_eval_command(
    *,
    model: str,
    defense: str,
    attack: str,
    limit: int,
    use_cache: bool,
) -> list[str]:
    command = [
        project_python(),
        str(INJECAGENT_ROOT / "src" / "evaluate_cross_eval.py"),
        "--model",
        model,
        "--defense",
        defense,
        "--attack",
        attack,
        "--test_data",
        str(TEST_CASES),
        "--ours_attacks",
        str(OURS_ATTACKS),
        "--setting",
        "base",
        "--prompt_type",
        "InjecAgent",
    ]
    if defense == "ours":
        command.extend(["--harness_file", str(SELECTED_HARNESS)])
    if limit > 0:
        command.extend(["--limit", str(limit)])
    if use_cache:
        command.append("--use_cache")
    return command


def run_command(command: list[str], *, cwd: Path, dry_run: bool) -> int:
    print(" ".join(command))
    if dry_run:
        return 0
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join(
        [
            str(INJECAGENT_ROOT),
            str(REPO_ROOT),
            str(EXPERIMENT_ROOT),
            env.get("PYTHONPATH", ""),
        ]
    )
    completed = subprocess.run(command, cwd=str(cwd), env=env)
    return completed.returncode
