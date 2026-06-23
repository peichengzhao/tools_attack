#!/usr/bin/env python3
"""Plan or execute the separate interactive-defense baseline table."""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
EXPERIMENT_ROOT = SCRIPT_DIR.parent
REPO_ROOT = EXPERIMENT_ROOT.parents[1]
DEFAULT_CONFIG = EXPERIMENT_ROOT / "configs" / "interactive_baselines.json"
DEFAULT_OUTPUT = (
    REPO_ROOT / "results" / "aaai27_supplement" / "interactive_baselines"
)


def safe_name(value: str) -> str:
    return value.replace("/", "_").replace(".", "_")


def load_config(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def cells_for(config: dict[str, Any], models: list[str]) -> list[dict[str, Any]]:
    cells: list[dict[str, Any]] = []
    for model in models:
        for suite in config["suites"]:
            for defense in config["defenses"]:
                cells.append(
                    {
                        "cell_id": f"{model}|{suite}|{defense['id']}",
                        "model": model,
                        "suite": suite,
                        "defense": defense["id"],
                        "display_name": defense["display_name"],
                        "provenance": defense["provenance"],
                        "attack": config["attack"],
                        "benchmark": config["benchmark"],
                        "benchmark_version": config["benchmark_version"],
                        "output_dir": str(
                            DEFAULT_OUTPUT
                            / safe_name(model)
                            / suite
                            / defense["id"]
                        ),
                    }
                )
    return cells


def command_for(
    cell: dict[str, Any],
    *,
    python: str,
    model_config: str,
    limit: int,
) -> list[str]:
    command = [
        python,
        str(SCRIPT_DIR / "run_baseline.py"),
        "--baseline",
        cell["defense"],
        "--model",
        cell["model"],
        "--suite",
        cell["suite"],
        "--attack",
        cell["attack"],
        "--benchmark-version",
        cell["benchmark_version"],
        "--output-dir",
        str(Path(cell["output_dir"]).parent),
        "--model-config",
        model_config,
    ]
    if limit > 0:
        command.extend(["--limit", str(limit)])
    return command


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def parse_camel(text: str) -> dict[str, Any]:
    utility_match = re.search(
        r"overall\s*-\s*utility:\s*([0-9]+(?:\.[0-9]+)?)",
        text,
        re.IGNORECASE,
    )
    security_match = re.search(
        r"overall\s*-\s*security:\s*([0-9]+(?:\.[0-9]+)?)",
        text,
        re.IGNORECASE,
    )
    utility = float(utility_match.group(1)) if utility_match else None
    security = float(security_match.group(1)) if security_match else None
    return {
        "utility": utility,
        "asr": 1.0 - security if security is not None else None,
    }


def summary_for(defense: str, cell_dir: Path, stdout: str) -> dict[str, Any]:
    if defense == "task_shield":
        path = cell_dir / "task_shield_summary.json"
        if path.is_file():
            payload = json.loads(path.read_text(encoding="utf-8"))
            return {
                "utility": payload.get("utility"),
                "asr": payload.get("asr"),
                "n": payload.get("n_user_tasks"),
            }
    if defense == "melon":
        path = cell_dir / "melon_summary.json"
        if path.is_file():
            payload = json.loads(path.read_text(encoding="utf-8"))
            security = payload.get("security")
            return {
                "utility": payload.get("utility"),
                "asr": 1.0 - security if security is not None else None,
                "n": payload.get("n_user_tasks"),
            }
    if defense == "camel":
        return {**parse_camel(stdout), "n": None}
    return {"utility": None, "asr": None, "n": None}


def percent(value: Any) -> float | None:
    if value is None:
        return None
    number = float(value)
    return round(number * 100.0, 2) if abs(number) <= 1.0 else round(number, 2)


def execute_cell(
    cell: dict[str, Any],
    command: list[str],
    *,
    limit: int,
) -> dict[str, Any]:
    cell_dir = Path(cell["output_dir"])
    cell_dir.mkdir(parents=True, exist_ok=True)
    log_path = cell_dir / "run.log"
    started = utc_now()
    with log_path.open("w", encoding="utf-8") as log:
        log.write(f"# command: {json.dumps(command, ensure_ascii=False)}\n\n")
        log.flush()
        proc = subprocess.run(
            command,
            cwd=str(REPO_ROOT),
            env=os.environ.copy(),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        log.write(proc.stdout or "")
    metrics = (
        summary_for(cell["defense"], cell_dir, proc.stdout or "")
        if proc.returncode == 0
        else {"utility": None, "asr": None, "n": None}
    )
    payload = {
        **cell,
        "command": command,
        "limit": limit,
        "started": started,
        "finished": utc_now(),
        "exit_code": proc.returncode,
        "status": "ok" if proc.returncode == 0 else "failed",
        "utility_pct": percent(metrics.get("utility")),
        "asr_pct": percent(metrics.get("asr")),
        "n": metrics.get("n"),
        "log": str(log_path),
        "stdout_tail": (proc.stdout or "")[-4000:],
    }
    (cell_dir / "cell.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return payload


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument(
        "--models",
        default=None,
        help="Comma-separated model override.",
    )
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument(
        "--python",
        default=os.environ.get(
            "AAAI27_PROJECT_PYTHON",
            str(Path.home() / "anaconda3" / "envs" / "tools_attack" / "bin" / "python"),
        ),
    )
    parser.add_argument(
        "--model-config",
        default=os.environ.get(
            "AAAI27_MODEL_CONFIG",
            str(Path.home() / "contract_heldout_ipi" / "model_api_ablation.py"),
        ),
    )
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()

    config = load_config(Path(args.config))
    models = (
        [item.strip() for item in args.models.split(",") if item.strip()]
        if args.models
        else list(config["models"])
    )
    cells = cells_for(config, models)
    plan = [
        {
            **cell,
            "command": command_for(
                cell,
                python=args.python,
                model_config=args.model_config,
                limit=args.limit,
            ),
        }
        for cell in cells
    ]
    DEFAULT_OUTPUT.mkdir(parents=True, exist_ok=True)
    (DEFAULT_OUTPUT / "interactive_plan.json").write_text(
        json.dumps(
            {
                "table": config["table"],
                "benchmark": config["benchmark"],
                "benchmark_version": config["benchmark_version"],
                "attack": config["attack"],
                "limit": args.limit,
                "cells": plan,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    if not args.execute:
        print(
            json.dumps(
                {
                    "status": "planned",
                    "cells": len(plan),
                    "plan": str(DEFAULT_OUTPUT / "interactive_plan.json"),
                },
                indent=2,
            )
        )
        return 0

    failures = 0
    for cell in plan:
        result = execute_cell(cell, cell["command"], limit=args.limit)
        print(
            f"{result['status']}: {result['defense']} "
            f"{result['model']} {result['suite']}"
        )
        if result["status"] != "ok":
            failures += 1
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
