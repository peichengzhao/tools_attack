#!/usr/bin/env python3
"""Config-driven runner for supplement experiments."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
EXPERIMENT_ROOT = SCRIPT_DIR.parent
REPO_ROOT = EXPERIMENT_ROOT.parents[1]
INJECAGENT_ROOT = REPO_ROOT / "InjecAgent"
DEFAULT_CONFIG = EXPERIMENT_ROOT / "configs" / "experiments.json"
OUTPUT_ROOT = REPO_ROOT / "results" / "aaai27_supplement"
PROJECT_PYTHON = os.environ.get("AAAI27_PROJECT_PYTHON") or (
    str(Path.home() / "anaconda3" / "envs" / "tools_attack" / "bin" / "python")
    if (Path.home() / "anaconda3" / "envs" / "tools_attack" / "bin" / "python").is_file()
    else sys.executable
)
USER_MODEL_CONFIG = os.environ.get("AAAI27_MODEL_CONFIG") or (
    str(Path.home() / "contract_heldout_ipi" / "model_api_ablation.py")
    if (Path.home() / "contract_heldout_ipi" / "model_api_ablation.py").is_file()
    else str(REPO_ROOT / "model_api_ablation.py")
)


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def load_config(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def expand_value(value: str, *, limit: int | None, max_rounds: int | None) -> str:
    values = {
        "{python}": sys.executable,
        "{project_python}": PROJECT_PYTHON,
        "{experiment_root}": str(EXPERIMENT_ROOT),
        "{repo_root}": str(REPO_ROOT),
        "{injecagent_root}": str(INJECAGENT_ROOT),
        "{limit}": str(limit if limit is not None else 0),
        "{max_rounds}": str(max_rounds if max_rounds is not None else 8),
    }
    for key, replacement in values.items():
        value = value.replace(key, replacement)
    return value


def select_runs(
    config: dict,
    *,
    run_id: str | None,
    table: str | None,
) -> list[dict]:
    runs = config.get("runs", [])
    if run_id:
        runs = [item for item in runs if item.get("id") == run_id]
    if table:
        runs = [item for item in runs if item.get("table") == table]
    return runs


def command_for(
    run: dict,
    *,
    limit: int | None,
    max_rounds: int | None,
) -> tuple[str, list[str]]:
    cwd = expand_value(
        run.get("cwd", "{repo_root}"), limit=limit, max_rounds=max_rounds
    )
    command = [
        expand_value(str(token), limit=limit, max_rounds=max_rounds)
        for token in run["command"]
    ]
    return cwd, command


def run_one(
    run: dict,
    *,
    limit: int | None,
    max_rounds: int | None,
    force: bool,
) -> int:
    if not run.get("enabled", False) and not force:
        print(f"SKIP disabled run {run['id']} (use --force to run)")
        return 0
    cwd, command = command_for(run, limit=limit, max_rounds=max_rounds)
    log_dir = OUTPUT_ROOT / "logs"
    run_dir = OUTPUT_ROOT / "runs"
    log_dir.mkdir(parents=True, exist_ok=True)
    run_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    log_path = log_dir / f"{run['id']}-{stamp}.log"
    manifest_path = run_dir / f"{run['id']}-{stamp}.json"
    env = os.environ.copy()
    # Do not resolve symlinks: conda's bin/python points to the base interpreter,
    # but CONDA_PREFIX must remain the tools_attack environment.
    project_bin = Path(PROJECT_PYTHON).parent
    env["CONDA_PREFIX"] = str(project_bin.parent)
    env["CONDA_DEFAULT_ENV"] = project_bin.parent.name
    env["PATH"] = str(project_bin) + os.pathsep + env.get("PATH", "")
    project_lib = project_bin.parent / "lib"
    env["LD_LIBRARY_PATH"] = str(project_lib) + os.pathsep + env.get(
        "LD_LIBRARY_PATH", ""
    )
    env["PYTHONPATH"] = cwd + os.pathsep + env.get("PYTHONPATH", "")
    env.setdefault(
        "TOOLS_ATTACK_MODEL_CONFIG",
        USER_MODEL_CONFIG
        if Path(USER_MODEL_CONFIG).is_file()
        else str(REPO_ROOT / "model_api_ablation.py"),
    )
    env.update({str(k): str(v) for k, v in run.get("env", {}).items()})

    print(json.dumps({"id": run["id"], "cwd": cwd, "command": command}, indent=2))
    started_at = utc_now()
    with log_path.open("w", encoding="utf-8") as log:
        log.write(f"# started {started_at}\n")
        log.write(f"# cwd {cwd}\n")
        log.write(f"# command {json.dumps(command, ensure_ascii=False)}\n\n")
        log.flush()
        proc = subprocess.run(
            command,
            cwd=cwd,
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
        )
        log.write(f"\n# exit_code {proc.returncode} finished {utc_now()}\n")

    manifest = {
        "id": run["id"],
        "table": run.get("table"),
        "description": run.get("description"),
        "budget": run.get("budget"),
        "cwd": cwd,
        "command": command,
        "limit": limit,
        "max_rounds": max_rounds,
        "started": started_at,
        "finished": utc_now(),
        "exit_code": proc.returncode,
        "log": str(log_path),
    }
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"exit={proc.returncode} log={log_path}")
    return proc.returncode


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "action",
        choices=["list", "dry-run", "run"],
        help="Action to perform",
    )
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--id", dest="run_id")
    parser.add_argument("--table", choices=["T1", "T2", "T3"])
    parser.add_argument("--limit", type=int)
    parser.add_argument("--max-rounds", type=int)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = load_config(Path(args.config))
    runs = select_runs(config, run_id=args.run_id, table=args.table)
    if not runs:
        raise SystemExit("No matching runs")

    if args.action == "list":
        for run in runs:
            print(
                f"{run['id']}\t{run.get('table')}\t"
                f"enabled={run.get('enabled', False)}\t{run.get('description', '')}"
            )
        return 0

    if args.action == "dry-run":
        for run in runs:
            cwd, command = command_for(
                run, limit=args.limit, max_rounds=args.max_rounds
            )
            print(json.dumps({"id": run["id"], "cwd": cwd, "command": command}, indent=2))
        return 0

    exit_code = 0
    for run in runs:
        exit_code = max(
            exit_code,
            run_one(
                run,
                limit=args.limit,
                max_rounds=args.max_rounds,
                force=args.force,
            ),
        )
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
