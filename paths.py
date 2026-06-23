"""
Canonical workspace root for all tools_attack experiments.

Every script, config, result, and patch in this project must live under
``TOOLS_ATTACK_ROOT`` (this directory). Do not write artifacts to
``/home/zpc/model_attack/InjecAgent``, ``/home/zpc/model_attack/ChatInject``, etc.
"""

from __future__ import annotations

import sys
from pathlib import Path

TOOLS_ATTACK_ROOT = Path(__file__).resolve().parent

MODEL_API_PATH = TOOLS_ATTACK_ROOT / "model_api.py"
INJECAGENT_ROOT = TOOLS_ATTACK_ROOT / "InjecAgent"
INJECAGENT_SRC = INJECAGENT_ROOT / "src"
INJECAGENT_RESULTS = INJECAGENT_ROOT / "results"

CHATINJECT_ROOT = TOOLS_ATTACK_ROOT / "related_work" / "ChatInject"
CHATINJECT_SRC = CHATINJECT_ROOT / "src"
CHATINJECT_RUNS = CHATINJECT_SRC / "runs"

TOPICATTACK_ROOT = TOOLS_ATTACK_ROOT / "related_work" / "topicattack"
TOPICATTACK_INJECAGENT_DATA = TOPICATTACK_ROOT / "injectagent" / "data"
TOPICATTACK_GRANT_CASES = TOPICATTACK_INJECAGENT_DATA / "crafted_instruction_data_agent_grant.json"

INJECAGENT_DATA = INJECAGENT_ROOT / "data"
INJECAGENT_USER_CASES = INJECAGENT_DATA / "user_cases.jsonl"
INJECAGENT_USER_CASES_CLEAN = INJECAGENT_DATA / "user_cases_clean.json"

EXPERIMENT_SCRIPTS = TOOLS_ATTACK_ROOT / "实验脚本.jsonl"
EXPERIMENT_LOG = TOOLS_ATTACK_ROOT / "实验记录.jsonl"

EXP7_ROOT = TOOLS_ATTACK_ROOT / "exp7"
EXP7_RESULTS = EXP7_ROOT / "results"

CROSS_EVAL_DATA = INJECAGENT_DATA / "cross_eval"
CHATINJECT_CASE_GOAL_MAP = CROSS_EVAL_DATA / "chatinject_case_goal_map.json"
CROSS_EVAL_OFFLINE_POISON = CROSS_EVAL_DATA / "offline_poison"
CROSS_EVAL_MANIFEST_CSV = CROSS_EVAL_DATA / "cell_manifest.csv"
CROSS_EVAL_MANIFEST_JSON = CROSS_EVAL_DATA / "cell_manifest.json"
CROSS_EVAL_RESULTS = INJECAGENT_RESULTS / "cross_eval"


def ensure_on_sys_path() -> Path:
    """Insert workspace root on ``sys.path`` so ``import model_api`` works."""
    root = str(TOOLS_ATTACK_ROOT)
    if root not in sys.path:
        sys.path.insert(0, root)
    return TOOLS_ATTACK_ROOT
