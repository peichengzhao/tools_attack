#!/usr/bin/env python3
"""AgentDojo runner for the vendored Task Shield reproduction."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
EXPERIMENT_ROOT = SCRIPT_DIR.parent
REPO_ROOT = EXPERIMENT_ROOT.parents[1]
TASK_SHIELD_ROOT = REPO_ROOT / "related_work" / "baselines" / "task_shield"
if str(TASK_SHIELD_ROOT) not in sys.path:
    sys.path.insert(0, str(TASK_SHIELD_ROOT))


def ensure_conda_runtime() -> None:
    lib_dir = str(Path(sys.prefix) / "lib")
    current = os.environ.get("LD_LIBRARY_PATH", "")
    if lib_dir in current.split(os.pathsep):
        return
    os.environ["LD_LIBRARY_PATH"] = os.pathsep.join(
        part for part in (lib_dir, current) if part
    )
    os.environ["_AAAI27_CONDA_REEXEC"] = "1"
    os.execv(sys.executable, [sys.executable, *sys.argv])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--suite", default="workspace")
    parser.add_argument("--attack", default="important_instructions")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--benchmark-version", default="v1.2")
    parser.add_argument(
        "--no-attack",
        action="store_true",
        help="Run clean utility only.",
    )
    parser.add_argument("--check", action="store_true")
    return parser.parse_args()


def main() -> int:
    if not os.environ.get("_AAAI27_CONDA_REEXEC"):
        ensure_conda_runtime()
    args = parse_args()

    from agentdojo import attacks, benchmark, logging
    from agentdojo.task_suite.load_suites import get_suite
    from task_shield.task_shield_pipeline import build_task_shield_pipeline

    pipeline = build_task_shield_pipeline(model=args.model)
    if args.check:
        print(
            json.dumps(
                {
                    "ok": True,
                    "baseline": "task_shield",
                    "pipeline": getattr(pipeline, "name", type(pipeline).__name__),
                    "class": type(pipeline).__name__,
                },
                indent=2,
            )
        )
        return 0
    suite = get_suite(args.benchmark_version, args.suite)
    user_tasks = list(suite.user_tasks)
    if args.limit > 0:
        user_tasks = user_tasks[: args.limit]

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with logging.OutputLogger(str(output_dir)):
        if args.no_attack:
            results = benchmark.benchmark_suite_without_injections(
                pipeline,
                suite,
                output_dir,
                force_rerun=False,
                user_tasks=user_tasks,
                benchmark_version=args.benchmark_version,
            )
        else:
            attack = attacks.load_attack(args.attack, suite, pipeline)
            results = benchmark.benchmark_suite_with_injections(
                pipeline,
                suite,
                attack,
                output_dir,
                force_rerun=False,
                user_tasks=user_tasks,
                benchmark_version=args.benchmark_version,
            )

    utility = results.get("utility_results", {})
    security = results.get("security_results", {})
    utility_rate = sum(utility.values()) / len(utility) if utility else None
    safe_rate = sum(security.values()) / len(security) if security else None
    payload = {
        "baseline": "task_shield",
        "model": args.model,
        "suite": args.suite,
        "attack": None if args.no_attack else args.attack,
        "n_user_tasks": len(utility),
        "utility": round(utility_rate, 6) if utility_rate is not None else None,
        "asr": round(1.0 - safe_rate, 6) if safe_rate is not None else None,
        "utility_results": utility,
        "security_results": security,
    }
    path = output_dir / "task_shield_summary.json"
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
