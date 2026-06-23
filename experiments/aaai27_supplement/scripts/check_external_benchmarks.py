#!/usr/bin/env python3
"""Check AgentBench/AppWorld data, isolated Python environments, and container runtime."""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import shutil
import subprocess
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
EXPERIMENT_ROOT = SCRIPT_DIR.parent
REPO_ROOT = EXPERIMENT_ROOT.parents[1]


def default_python(env_name: str) -> Path:
    override = os.environ.get(f"{env_name.upper()}_PYTHON")
    if override:
        return Path(override)
    return REPO_ROOT / ".venvs" / env_name / "bin" / "python"


def command_ok(command: list[str], *, cwd: Path | None = None) -> tuple[bool, str]:
    try:
        proc = subprocess.run(
            command,
            cwd=str(cwd) if cwd else None,
            capture_output=True,
            text=True,
            timeout=30,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return False, str(exc)
    text = (proc.stdout or proc.stderr).strip()
    return proc.returncode == 0, text[-1000:]


def check_python(
    path: Path,
    imports: list[str],
    *,
    cwd: Path | None = None,
) -> dict[str, Any]:
    if not path.is_file():
        return {"ok": False, "path": str(path), "error": "missing interpreter"}
    code = (
        "import importlib.util, json, sys; "
        "mods=json.loads(sys.argv[1]); "
        "print(json.dumps({m: bool(importlib.util.find_spec(m)) for m in mods}))"
    )
    ok, output = command_ok(
        [str(path), "-c", code, json.dumps(imports)],
        cwd=cwd,
    )
    return {
        "ok": ok,
        "path": str(path),
        "imports": output if not ok else json.loads(output.splitlines()[-1]),
    }


def check_agentbench() -> dict[str, Any]:
    root = REPO_ROOT / "related_work" / "AgentBench"
    python_path = default_python("agentbench")
    data_files = [
        root / "data/dbbench/standard.jsonl",
        root / "data/os_interaction/res",
    ]
    python_report = check_python(
        python_path,
        ["docker", "numpy", "pydantic", "src.client.agent_test"],
        cwd=root if root.is_dir() else None,
    )
    docker_socket = Path("/var/run/docker.sock")
    docker_cli = shutil.which("docker")
    docker_env = os.environ.get("DOCKER_HOST", "")
    return {
        "root": str(root),
        "repo_exists": (root / ".git").exists(),
        "data": {str(path): path.exists() for path in data_files},
        "python": python_report,
        "docker_cli": docker_cli,
        "docker_socket": docker_socket.exists(),
        "docker_host": docker_env or None,
        "worker_ready": docker_socket.exists() or docker_env.startswith("unix://"),
    }


def check_appworld() -> dict[str, Any]:
    root = REPO_ROOT / "related_work" / "appworld"
    python_candidates = [
        default_python("appworld"),
        Path.home() / "anaconda3" / "envs" / "appworld" / "bin" / "python",
    ]
    python_path = next((path for path in python_candidates if path.is_file()), python_candidates[0])
    python_report = check_python(python_path, ["appworld", "openai"])
    data_files = [
        root / "data/datasets/train.txt",
        root / "data/datasets/dev.txt",
        root / "data/datasets/test_normal.txt",
        root / "data/datasets/test_challenge.txt",
        root / "data/base_dbs/admin.db",
    ]
    counts: dict[str, int | None] = {}
    if python_report["ok"]:
        code = (
            "import json, os, sys; "
            "os.environ['APPWORLD_ROOT']=sys.argv[1]; "
            "from appworld import load_task_ids; "
            "print(json.dumps({s: len(load_task_ids(s)) for s in "
            "['train','dev','test_normal','test_challenge']}))"
        )
        ok, output = command_ok([str(python_path), "-c", code, str(root)])
        if ok:
            counts = json.loads(output.splitlines()[-1])
    return {
        "root": str(root),
        "repo_exists": (root / ".git").exists(),
        "data": {str(path): path.exists() for path in data_files},
        "data_size": subprocess.run(
            ["du", "-sh", str(root / "data")],
            capture_output=True,
            text=True,
        ).stdout.split()[0]
        if (root / "data").exists()
        else None,
        "python": python_report,
        "task_counts": counts,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    report = {
        "repo_root": str(REPO_ROOT),
        "agentbench": check_agentbench(),
        "appworld": check_appworld(),
    }
    appworld = report["appworld"]
    agentbench = report["agentbench"]
    report["ok"] = (
        agentbench["repo_exists"]
        and all(agentbench["data"].values())
        and agentbench["python"]["ok"]
        and all(agentbench["python"]["imports"].values())
        and appworld["repo_exists"]
        and all(appworld["data"].values())
        and appworld["python"]["ok"]
        and all(appworld["python"]["imports"].values())
        and bool(appworld["task_counts"])
        and agentbench["worker_ready"]
    )

    if args.json:
        print(json.dumps(report, indent=2, ensure_ascii=False))
        return 0 if report["ok"] else 1

    print("| Component | Repository | Data | Python | Runtime | Status |")
    print("|---|---|---|---|---|---|")
    ab_py = agentbench["python"]["ok"] and all(agentbench["python"]["imports"].values())
    aw_py = appworld["python"]["ok"] and all(appworld["python"]["imports"].values())
    print(
        f"| AgentBench | {'OK' if agentbench['repo_exists'] else 'missing'} | "
        f"{'OK' if all(agentbench['data'].values()) else 'missing'} | "
        f"{'OK' if ab_py else 'missing'} | "
        f"{'Docker API ready' if agentbench['worker_ready'] else 'Docker daemon missing'} | "
        f"{'ready' if report['ok'] else 'blocked'} |"
    )
    print(
        f"| AppWorld | {'OK' if appworld['repo_exists'] else 'missing'} | "
        f"{'OK' if all(appworld['data'].values()) else 'missing'} | "
        f"{'OK' if aw_py else 'missing'} | "
        f"{appworld['data_size'] or '—'} | "
        f"{'ready' if aw_py and all(appworld['data'].values()) else 'blocked'} |"
    )
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
