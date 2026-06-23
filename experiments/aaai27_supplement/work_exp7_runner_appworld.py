from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from pathlib import Path

from exp7.config import APPWORLD_ROOT, APPWORLD_SPLIT, METRIC_LABELS, EXP7_ROOT
from exp7.runners._common import make_result, openai_env, run_cmd, write_result

RUNNER = EXP7_ROOT / "appworld" / "run_benign.py"


def _appworld_python() -> Path:
    candidates = [
        Path(os.environ["APPWORLD_PYTHON"]) if os.environ.get("APPWORLD_PYTHON") else None,
        APPWORLD_ROOT.parents[1] / ".venvs" / "appworld" / "bin" / "python",
        Path.home() / "anaconda3" / "envs" / "appworld" / "bin" / "python",
        Path(sys.executable),
    ]
    for candidate in candidates:
        if candidate is not None and candidate.is_file():
            return candidate
    raise RuntimeError(
        "AppWorld Python not found. Set APPWORLD_PYTHON or run: "
        "bash exp7/setup_benchmarks.sh appworld"
    )


def _appworld_installed(python_path: Path) -> bool:
    proc = subprocess.run(
        [str(python_path), "-c", "import appworld"],
        capture_output=True,
        text=True,
    )
    return proc.returncode == 0


def run(*, model: str, harness: str, force: bool = False, limit: int = 0) -> dict:
    metric = METRIC_LABELS["appworld"]
    if not RUNNER.exists():
        raise RuntimeError(f"Missing AppWorld runner: {RUNNER}")
    python_path = _appworld_python()
    if not _appworld_installed(python_path):
        raise RuntimeError(
            f"AppWorld is not installed in {python_path}. Run: "
            "bash exp7/setup_benchmarks.sh appworld"
        )
    if not (APPWORLD_ROOT / "data" / "datasets" / "test_normal.txt").is_file():
        raise RuntimeError(
            f"AppWorld data missing under {APPWORLD_ROOT}. Run: "
            "bash exp7/setup_benchmarks.sh appworld"
        )

    env = openai_env(model)
    env["TOOLS_ATTACK_HARNESS"] = harness
    env["APPWORLD_ROOT"] = str(APPWORLD_ROOT)
    env["APPWORLD_SPLIT"] = APPWORLD_SPLIT
    env["APPWORLD_TASK_LIMIT"] = str(limit or 0)
    env["PYTHONPATH"] = str(APPWORLD_ROOT.parents[1]) + os.pathsep + env.get(
        "PYTHONPATH", ""
    )

    cmd = [
        str(python_path),
        str(RUNNER),
        "--model",
        model,
        "--harness",
        harness,
    ]
    if force:
        cmd.append("--force")

    proc = run_cmd(
        cmd,
        env=env,
        cwd=APPWORLD_ROOT if APPWORLD_ROOT.exists() else EXP7_ROOT,
    )
    stdout = proc.stdout or ""
    stderr = proc.stderr or ""
    if proc.returncode != 0:
        payload = make_result(
            benchmark="appworld",
            model=model,
            harness=harness,
            utility=None,
            metric_name=metric,
            status="error",
            error=(stderr or stdout)[-2500:],
            details={"python": str(python_path)},
        )
        write_result(payload)
        return payload

    result_json = _parse_result_json(stdout)
    utility = result_json.get("utility")
    payload = make_result(
        benchmark="appworld",
        model=model,
        harness=harness,
        utility=float(utility) if utility is not None else None,
        metric_name=metric,
        details={**result_json, "python": str(python_path)},
    )
    write_result(payload)
    return payload


def _parse_result_json(text: str) -> dict:
    for line in reversed(text.splitlines()):
        line = line.strip()
        if line.startswith("{") and line.endswith("}"):
            try:
                return json.loads(line)
            except json.JSONDecodeError:
                continue
    m = re.search(r"Task Success Rate:\s*([0-9]+(?:\.[0-9]+)?)\s*%", text)
    if m:
        return {"utility": float(m.group(1))}
    return {}
