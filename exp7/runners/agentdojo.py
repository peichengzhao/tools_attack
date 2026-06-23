from __future__ import annotations

import json
import re
import sys
from pathlib import Path

from exp7.config import (
    AGENTDOJO_SUITES,
    AGENTDOJO_VERSION,
    METRIC_LABELS,
    TOOLS_ATTACK_ROOT,
)
from exp7.runners._common import make_result, openai_env, run_cmd, write_result

CHATINJECT_SRC = TOOLS_ATTACK_ROOT / "related_work" / "ChatInject" / "src"


def _agentdojo_logdir(model: str, harness: str) -> Path:
    """Separate cache per victim model + harness (default runs/tools_attack ignores model)."""
    safe_model = model.replace("/", "_").replace(".", "_")
    return CHATINJECT_SRC / "runs" / "exp7" / safe_model / harness


def run(*, model: str, harness: str, force: bool = False, limit: int = 0) -> dict:
    metric = METRIC_LABELS["agentdojo"]
    if not CHATINJECT_SRC.exists():
        raise RuntimeError(f"AgentDojo path missing: {CHATINJECT_SRC}")

    logdir = _agentdojo_logdir(model, harness)
    suites = list(AGENTDOJO_SUITES)
    cmd = [
        sys.executable,
        "-m",
        "agentdojo.scripts.benchmark",
        *[f"-s{s}" for s in suites],
        "--model",
        "tools_attack",
        "--benchmark-version",
        AGENTDOJO_VERSION,
        "--logdir",
        str(logdir),
    ]
    if harness == "max_defense":
        cmd.extend(["--defense", "max_defense"])
    if force:
        cmd.append("-f")

    env = openai_env(model)
    env["VICTIM_MODEL"] = model
    env["TOOLS_ATTACK_VICTIM_MODEL"] = model

    proc = run_cmd(cmd, cwd=CHATINJECT_SRC, env=env)
    stdout = proc.stdout or ""
    stderr = proc.stderr or ""
    if proc.returncode != 0:
        payload = make_result(
            benchmark="agentdojo",
            model=model,
            harness=harness,
            utility=None,
            metric_name=metric,
            status="error",
            error=(stderr or stdout)[-2000:],
            details={"returncode": proc.returncode},
        )
        write_result(payload)
        return payload

    utility = _parse_benign_utility(stdout)
    cache_note = _detect_cache_reuse(stdout)
    payload = make_result(
        benchmark="agentdojo",
        model=model,
        harness=harness,
        utility=utility,
        metric_name=metric,
        details={
            "suites": suites,
            "logdir": str(logdir),
            "cache_reuse": cache_note,
            "stdout_tail": stdout[-4000:],
        },
    )
    write_result(payload)
    return payload


def _parse_benign_utility(text: str) -> float | None:
    combined = re.search(
        r"Results for suite combined\s*\n\s*Benign Utility:\s*([0-9]+(?:\.[0-9]+)?)\s*%",
        text,
        re.IGNORECASE,
    )
    if combined:
        return float(combined.group(1))

    per_suite: dict[str, float] = {}
    for m in re.finditer(
        r"Results for suite (\w+)\s*\n\s*Benign Utility:\s*([0-9]+(?:\.[0-9]+)?)\s*%",
        text,
        re.IGNORECASE,
    ):
        name, val = m.group(1), float(m.group(2))
        if name.lower() != "combined":
            per_suite[name] = val
    if per_suite:
        return round(sum(per_suite.values()) / len(per_suite), 2)

    m = re.search(r"Benign Utility:\s*([0-9]+(?:\.[0-9]+)?)\s*%", text)
    if m:
        return float(m.group(1))
    m = re.search(r'"benign_utility"\s*:\s*([0-9]+(?:\.[0-9]+)?)', text)
    if m:
        return float(m.group(1))
    return None


def _detect_cache_reuse(text: str) -> dict:
    skipped = len(re.findall(r"Skipping task .+ because it was already run", text))
    finished = len(re.findall(r"Finished benchmark for suite:", text))
    return {
        "skipped_tasks_logged": skipped,
        "suites_finished": finished,
        "likely_all_cached": skipped > 0 and "HTTP Request" not in text,
    }
