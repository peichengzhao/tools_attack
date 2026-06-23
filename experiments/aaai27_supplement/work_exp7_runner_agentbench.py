from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

from exp7.config import AGENTBENCH_ROOT, AGENTBENCH_TASKS, METRIC_LABELS, EXP7_ROOT
from exp7.harness import apply_unified_system_addendum
from exp7.runners._common import make_result, openai_env, run_cmd, write_result

AGENT_CONFIG_DIR = EXP7_ROOT / "agentbench" / "generated_configs"


def _agentbench_python() -> Path:
    candidates = [
        Path(os.environ["AGENTBENCH_PYTHON"])
        if os.environ.get("AGENTBENCH_PYTHON")
        else None,
        AGENTBENCH_ROOT.parents[1] / ".venvs" / "agentbench" / "bin" / "python",
    ]
    for candidate in candidates:
        if candidate is not None and candidate.is_file():
            return candidate
    raise RuntimeError(
        "AgentBench Python not found. Set AGENTBENCH_PYTHON or run: "
        "bash exp7/setup_benchmarks.sh agentbench"
    )


def _docker_ready() -> bool:
    if os.environ.get("DOCKER_HOST"):
        return True
    return Path("/var/run/docker.sock").exists()


def run(*, model: str, harness: str, force: bool = False, limit: int = 0) -> dict:
    from model_api import OPENAI_API_KEY, OPENAI_BASE_URL

    metric = METRIC_LABELS["agentbench"]
    if not AGENTBENCH_ROOT.exists():
        raise RuntimeError(
            "AgentBench not found. Run: bash exp7/setup_benchmarks.sh agentbench"
        )
    if not _docker_ready():
        raise RuntimeError(
            "AgentBench workers require a Docker API endpoint. "
            "Expected /var/run/docker.sock or DOCKER_HOST."
        )

    python_path = _agentbench_python()
    agent_name = _write_agent_config(model, harness, OPENAI_BASE_URL, OPENAI_API_KEY)
    tasks = list(AGENTBENCH_TASKS)
    if limit > 0:
        tasks = tasks[:limit]

    env = openai_env(model)
    env["PYTHONPATH"] = str(AGENTBENCH_ROOT) + os.pathsep + env.get("PYTHONPATH", "")
    cmd = [
        str(python_path),
        "-m",
        "src.client.agent_test",
        "--config",
        str(AGENT_CONFIG_DIR / f"{agent_name}.yaml"),
        "--agent",
        agent_name,
    ]
    for task in tasks:
        cmd.extend(["--task", task])

    proc = run_cmd(cmd, cwd=AGENTBENCH_ROOT, env=env)
    stdout = proc.stdout or ""
    stderr = proc.stderr or ""
    if proc.returncode != 0:
        payload = make_result(
            benchmark="agentbench",
            model=model,
            harness=harness,
            utility=None,
            metric_name=metric,
            status="error",
            error=(stderr or stdout)[-2500:],
            details={
                "tasks": tasks,
                "agent_config": str(AGENT_CONFIG_DIR / f"{agent_name}.yaml"),
                "python": str(python_path),
            },
        )
        write_result(payload)
        return payload

    utility = _parse_success_rate(stdout)
    payload = make_result(
        benchmark="agentbench",
        model=model,
        harness=harness,
        utility=utility,
        metric_name=metric,
        details={
            "tasks": tasks,
            "stdout_tail": stdout[-4000:],
            "python": str(python_path),
        },
    )
    write_result(payload)
    return payload


def _write_agent_config(
    model: str,
    harness: str,
    base_url: str,
    api_key: str,
) -> str:
    AGENT_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    safe = model.replace("/", "_").replace(".", "_")
    agent_name = f"tools_attack_{safe}_{harness}"

    system_prompt = apply_unified_system_addendum(
        "You are a helpful agent that follows instructions and uses tools carefully.",
        enabled=(harness == "max_defense"),
    )

    yaml_text = f"""module: src.client.agents.HTTPAgent
name: {agent_name}
parameters:
  url: "{base_url.rstrip('/')}/chat/completions"
  headers:
    Content-Type: application/json
    Authorization: "Bearer {api_key}"
  body:
    model: "{model}"
    temperature: 0
    messages:
      - role: system
        content: {json.dumps(system_prompt, ensure_ascii=False)}
  prompter:
    name: role_content_dict
    args:
      agent_role: assistant
  return_format: "{{{{response[choices][0][message][content]}}}}"
"""
    (AGENT_CONFIG_DIR / f"{agent_name}.yaml").write_text(yaml_text, encoding="utf-8")
    return agent_name


def _parse_success_rate(text: str) -> float | None:
    import re

    for pat in (
        r"success(?:ful)? rate[:\s]+([0-9]+(?:\.[0-9]+)?)\s*%",
        r"overall[:\s]+([0-9]+(?:\.[0-9]+)?)\s*%",
        r"score[:\s]+([0-9]+(?:\.[0-9]+)?)",
    ):
        m = re.search(pat, text, re.IGNORECASE)
        if m:
            val = float(m.group(1))
            return val * 100 if val <= 1.0 else val
    return None
