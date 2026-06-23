from __future__ import annotations

from pathlib import Path

TOOLS_ATTACK_ROOT = Path(__file__).resolve().parents[1]
EXP7_ROOT = TOOLS_ATTACK_ROOT / "exp7"
EXP7_RESULTS = EXP7_ROOT / "results"

# Table rows (5 benchmarks × 3 models × 2 harness modes)
EXP7_MODELS = ("gpt-5.5", "deepseek-v4-flash", "glm-5.1")
EXP7_BENCHMARKS = (
    "agentdojo",
    "injecagent",
    "bfcl",
    "agentbench",
    "appworld",
)
HARNESS_BASELINE = "baseline"
HARNESS_MAX_DEFENSE = "max_defense"
HARNESS_MODES = (HARNESS_BASELINE, HARNESS_MAX_DEFENSE)

# AgentDojo
AGENTDOJO_SUITES = ("banking", "slack", "travel", "workspace")
AGENTDOJO_VERSION = "v1.2.1"

# BFCL v3 representative categories (function calling + multi-turn)
BFCL_TEST_CATEGORIES = (
    "simple_python",
    "simple_java",
    "simple_javascript",
    "multi_turn_base",
)

# AgentBench lightweight default (Docker workers required)
AGENTBENCH_TASKS = ("dbbench-std", "os-std")
AGENTBENCH_ROOT = TOOLS_ATTACK_ROOT / "related_work" / "AgentBench"

# AppWorld
APPWORLD_SPLIT = "test_normal"
APPWORLD_ROOT = TOOLS_ATTACK_ROOT / "related_work" / "appworld"

BENCHMARK_LABELS = {
    "agentdojo": "AgentDojo — Benign Utility",
    "injecagent": "InjecAgent — User Cases",
    "bfcl": "BFCL v3/v4",
    "agentbench": "AgentBench",
    "appworld": "AppWorld",
}

METRIC_LABELS = {
    "agentdojo": "Benign Utility (%)",
    "injecagent": "Utility Rate (%)",
    "bfcl": "Overall Accuracy (%)",
    "agentbench": "Success Rate (%)",
    "appworld": "Task Success Rate (%)",
}
