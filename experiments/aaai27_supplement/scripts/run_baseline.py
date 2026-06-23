#!/usr/bin/env python3
"""Launch a vendored defense baseline with a recorded, inspectable command."""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
EXPERIMENT_ROOT = SCRIPT_DIR.parent
REPO_ROOT = EXPERIMENT_ROOT.parents[1]
if str(EXPERIMENT_ROOT) not in sys.path:
    sys.path.insert(0, str(EXPERIMENT_ROOT))

from aaai27_supplement.baselines import (  # noqa: E402
    load_registry,
    read_model_runtime_config,
    repo_path,
)


def default_project_python() -> str:
    override = os.environ.get("AAAI27_PROJECT_PYTHON")
    if override:
        return override
    candidate = Path.home() / "anaconda3" / "envs" / "tools_attack" / "bin" / "python"
    return str(candidate) if candidate.is_file() else sys.executable


def default_model_config() -> Path:
    override = os.environ.get("AAAI27_MODEL_CONFIG")
    if override:
        return Path(override)
    candidate = Path.home() / "contract_heldout_ipi" / "model_api_ablation.py"
    return candidate if candidate.is_file() else REPO_ROOT / "model_api_ablation.py"


def api_environment(
    *,
    model: str,
    api_key: str | None,
    base_url: str | None,
    model_config: Path,
    allow_missing_key: bool,
    python_path: str,
) -> dict[str, str]:
    config = read_model_runtime_config(model_config)
    resolved_key = (
        api_key
        or os.environ.get("OPENAI_API_KEY")
        or config.get("OPENAI_API_KEY")
        or config.get("API_KEY")
    )
    resolved_base = (
        base_url
        or os.environ.get("OPENAI_BASE_URL")
        or config.get("OPENAI_BASE_URL")
        or config.get("BASE_URL")
    )
    if not resolved_key and not allow_missing_key:
        raise SystemExit(
            "Missing API key. Set OPENAI_API_KEY or pass --api-key; "
            f"checked model config {model_config}."
        )
    env = os.environ.copy()
    env["OPENAI_API_KEY"] = resolved_key or "<dry-run-placeholder>"
    env["DEEPSEEK_API_KEY"] = resolved_key or "<dry-run-placeholder>"
    if resolved_base:
        env["OPENAI_BASE_URL"] = resolved_base
        env["DEEPSEEK_BASE_URL"] = resolved_base
    env["DEEPSEEK_MODEL"] = model
    env["TOOLS_ATTACK_MODEL_CONFIG"] = str(model_config)
    runtime_prefix = Path(python_path).parent.parent
    env["CONDA_PREFIX"] = str(runtime_prefix)
    env["PATH"] = str(Path(python_path).parent) + os.pathsep + env.get("PATH", "")
    env["LD_LIBRARY_PATH"] = str(runtime_prefix / "lib") + os.pathsep + env.get(
        "LD_LIBRARY_PATH", ""
    )
    env["PYTHONPATH"] = os.pathsep.join(
        [
            str(REPO_ROOT),
            str(EXPERIMENT_ROOT),
            env.get("PYTHONPATH", ""),
        ]
    )
    return env


def build_command(args: argparse.Namespace, baseline_path: Path) -> tuple[list[str], Path]:
    runner = args.spec.runner
    if runner == "agentdojo_task_shield":
        command = [
            args.python,
            str(SCRIPT_DIR / "run_task_shield_agentdojo.py"),
            "--model",
            args.model,
            "--suite",
            args.suite,
            "--output-dir",
            str(args.output_dir),
            "--benchmark-version",
            args.benchmark_version,
        ]
        if args.attack:
            command.extend(["--attack", args.attack])
        else:
            command.append("--no-attack")
        if args.limit > 0:
            command.extend(["--limit", str(args.limit)])
        if args.check:
            command.append("--check")
        return command, REPO_ROOT

    if runner == "agentdojo_melon":
        command = [
            args.python,
            str(SCRIPT_DIR / "run_melon_agentdojo.py"),
            "--model",
            args.model,
            "--suite",
            args.suite,
            "--output-dir",
            str(args.output_dir),
            "--threshold",
            str(args.melon_threshold),
            "--benchmark-version",
            args.benchmark_version,
        ]
        if args.attack:
            command.extend(["--attack", args.attack])
        else:
            command.append("--no-attack")
        if args.limit > 0:
            command.extend(["--limit", str(args.limit)])
        if args.check:
            command.append("--check")
        return command, REPO_ROOT

    if runner == "agentdojo_camel":
        uv = os.environ.get("UV") or str(Path.home() / ".local" / "bin" / "uv")
        if args.check:
            return [uv, "run", "main.py", "--help"], baseline_path
        command = [
            uv,
            "run",
            "main.py",
            f"openai:{args.model}",
            "--suites",
            args.suite,
        ]
        if args.attack:
            command.append("--run-attack")
        if args.limit > 0:
            for task in agentdojo_user_tasks(
                args.suite,
                args.limit,
                args.benchmark_version,
            ):
                command.extend(["--user-tasks", task])
        return command, baseline_path

    if runner == "injecagent_piarena":
        command = [
            args.python,
            str(SCRIPT_DIR / "run_piarena_injecagent.py"),
            "--model",
            args.model,
            "--defense",
            args.piarena_defense,
            "--output-root",
            str(args.output_dir),
            "--test-data",
            str(
                REPO_ROOT
                / "InjecAgent"
                / "data"
                / "splits"
                / "test_cases_211.json"
            ),
        ]
        if args.limit > 0:
            command.extend(["--limit", str(args.limit)])
        if args.check:
            command.append("--check")
        return command, REPO_ROOT

    raise SystemExit(f"Unknown runner: {runner}")


def agentdojo_user_tasks(
    suite_name: str,
    limit: int,
    benchmark_version: str,
) -> list[str]:
    try:
        from agentdojo.task_suite.load_suites import get_suite

        suite = get_suite(benchmark_version, suite_name)
        return list(suite.user_tasks)[:limit]
    except Exception as exc:
        raise SystemExit(f"Cannot enumerate AgentDojo tasks for --limit: {exc}") from exc


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--model", default="deepseek-v4.1-flash-1017")
    parser.add_argument("--suite", default="workspace")
    parser.add_argument(
        "--attack",
        default="important_instructions",
        help="AgentDojo attack name; PIArena uses its native InjecAgent attack.",
    )
    parser.add_argument(
        "--no-attack",
        action="store_true",
        help="Run clean utility only for AgentDojo baselines.",
    )
    parser.add_argument(
        "--piarena-defense",
        default="pisanitizer",
        help="PIArena representative defense.",
    )
    parser.add_argument("--benchmark-version", default="v1.2")
    parser.add_argument("--melon-threshold", type=float, default=0.1)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--python", default=default_project_python())
    parser.add_argument("--model-config", default=str(default_model_config()))
    parser.add_argument("--api-key")
    parser.add_argument("--base-url")
    parser.add_argument(
        "--output-dir",
        default=str(
            REPO_ROOT
            / "results"
            / "aaai27_supplement"
            / "defense_baselines"
        ),
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--check",
        action="store_true",
        help="Only verify imports/pipeline construction; never call the model API.",
    )
    args = parser.parse_args()
    if args.no_attack:
        args.attack = None

    registry = load_registry(EXPERIMENT_ROOT / "configs" / "baselines.json")
    if args.baseline not in registry:
        raise SystemExit(f"Unknown baseline: {args.baseline}")
    args.spec = registry[args.baseline]
    baseline_path = repo_path(REPO_ROOT, args.spec)
    command, cwd = build_command(args, baseline_path)
    output_dir = Path(args.output_dir) / args.baseline
    output_dir.mkdir(parents=True, exist_ok=True)

    env = api_environment(
        model=args.model,
        api_key=args.api_key,
        base_url=args.base_url,
        model_config=Path(args.model_config),
        allow_missing_key=args.dry_run or args.check,
        python_path=args.python,
    )
    manifest = {
        "baseline": args.baseline,
        "provenance": args.spec.provenance,
        "commit": args.spec.commit,
        "cwd": str(cwd),
        "command": command,
        "model": args.model,
        "suite": args.suite,
        "attack": args.attack,
        "limit": args.limit,
        "api_base_url": env.get("OPENAI_BASE_URL"),
    }
    manifest_path = output_dir / f"command-{args.baseline}.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                **manifest,
                "shell": " ".join(shlex.quote(part) for part in command),
                "manifest": str(manifest_path),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    if args.dry_run:
        return 0

    proc = subprocess.run(command, cwd=str(cwd), env=env)
    return proc.returncode


if __name__ == "__main__":
    raise SystemExit(main())
