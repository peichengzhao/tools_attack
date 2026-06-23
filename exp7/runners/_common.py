from __future__ import annotations

import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from exp7.config import EXP7_RESULTS, TOOLS_ATTACK_ROOT

if str(TOOLS_ATTACK_ROOT) not in sys.path:
    sys.path.insert(0, str(TOOLS_ATTACK_ROOT))

from paths import ensure_on_sys_path  # noqa: E402

ensure_on_sys_path()
from model_api import OPENAI_API_KEY, OPENAI_BASE_URL  # noqa: E402


def result_path(benchmark: str, model: str, harness: str) -> Path:
    safe_model = model.replace("/", "_")
    return EXP7_RESULTS / benchmark / safe_model / f"{harness}.json"


def write_result(payload: dict[str, Any]) -> Path:
    path = result_path(payload["benchmark"], payload["model"], payload["harness"])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return path


def load_result(benchmark: str, model: str, harness: str) -> dict[str, Any] | None:
    path = result_path(benchmark, model, harness)
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def make_result(
    *,
    benchmark: str,
    model: str,
    harness: str,
    utility: float | None,
    metric_name: str,
    details: dict[str, Any] | None = None,
    status: str = "ok",
    error: str | None = None,
) -> dict[str, Any]:
    return {
        "benchmark": benchmark,
        "model": model,
        "harness": harness,
        "utility": utility,
        "metric_name": metric_name,
        "status": status,
        "error": error,
        "details": details or {},
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "api_base_url": OPENAI_BASE_URL,
    }


def openai_env(model: str) -> dict[str, str]:
    env = os.environ.copy()
    env["OPENAI_API_KEY"] = OPENAI_API_KEY
    env["OPENAI_BASE_URL"] = OPENAI_BASE_URL
    env["REMOTE_OPENAI_BASE_URL"] = OPENAI_BASE_URL
    env["REMOTE_OPENAI_API_KEY"] = OPENAI_API_KEY
    env["TOOLS_ATTACK_VICTIM_MODEL"] = model
    return env


def run_cmd(
    cmd: list[str],
    *,
    cwd: Path | None = None,
    env: dict[str, str] | None = None,
    check: bool = False,
) -> subprocess.CompletedProcess[str]:
    print("$", " ".join(cmd))
    return subprocess.run(
        cmd,
        cwd=str(cwd) if cwd else None,
        env=env,
        text=True,
        capture_output=True,
        check=check,
    )


def require_command(name: str, install_hint: str) -> None:
    from shutil import which

    if which(name) is None:
        raise RuntimeError(f"Missing `{name}`. {install_hint}")
