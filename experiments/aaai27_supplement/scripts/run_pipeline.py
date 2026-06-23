#!/usr/bin/env python3
"""End-to-end runner with stage logs, call estimates, and quota circuit breaking."""

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
RESULT_ROOT = REPO_ROOT / "results" / "aaai27_supplement"
ARTIFACT_ROOT = RESULT_ROOT / "artifacts"
PIPELINE_ROOT = RESULT_ROOT / "pipeline"

sys.path.insert(0, str(EXPERIMENT_ROOT))
from aaai27_supplement.estimate import estimate_pipeline_calls  # noqa: E402


QUOTA_MARKERS = (
    "quota exceeded",
    "insufficient_quota",
    "User has been banned",
    "too many requests",
    "rate limit",
    "error code: 429",
    "error code: 529",
)


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def default_project_python() -> str:
    override = os.environ.get("AAAI27_PROJECT_PYTHON")
    if override:
        return override
    candidate = Path.home() / "anaconda3" / "envs" / "tools_attack" / "bin" / "python"
    return str(candidate) if candidate.is_file() else sys.executable


def default_model_config() -> str:
    override = os.environ.get("AAAI27_MODEL_CONFIG")
    if override:
        return override
    candidate = Path.home() / "contract_heldout_ipi" / "model_api_ablation.py"
    if candidate.is_file():
        return str(candidate)
    return str(REPO_ROOT / "model_api_ablation.py")


def command_env(project_python: str, model_config: str) -> dict[str, str]:
    env = os.environ.copy()
    project_bin = Path(project_python).parent
    env["CONDA_PREFIX"] = str(project_bin.parent)
    env["CONDA_DEFAULT_ENV"] = project_bin.parent.name
    env["PATH"] = str(project_bin) + os.pathsep + env.get("PATH", "")
    env["LD_LIBRARY_PATH"] = str(project_bin.parent / "lib") + os.pathsep + env.get(
        "LD_LIBRARY_PATH", ""
    )
    env["PYTHONPATH"] = str(INJECAGENT_ROOT) + os.pathsep + env.get("PYTHONPATH", "")
    env["TOOLS_ATTACK_MODEL_CONFIG"] = model_config
    return env


def run_stage(
    name: str,
    command: list[str],
    *,
    cwd: Path,
    env: dict[str, str],
    dry_run: bool,
) -> dict:
    PIPELINE_ROOT.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    log_path = PIPELINE_ROOT / f"{name}-{stamp}.log"
    print(json.dumps({"stage": name, "cwd": str(cwd), "command": command}, indent=2))
    if dry_run:
        return {"stage": name, "status": "dry_run", "command": command}
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log:
        log.write(f"# started {utc_now()}\n")
        log.write(f"# command {json.dumps(command, ensure_ascii=False)}\n\n")
        log.flush()
        proc = subprocess.run(
            command,
            cwd=str(cwd),
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
        )
        log.write(f"\n# exit_code {proc.returncode} finished {utc_now()}\n")
    text = log_path.read_text(encoding="utf-8", errors="ignore")
    lowered = text.lower()
    quota_hit = any(marker.lower() in lowered for marker in QUOTA_MARKERS)
    result = {
        "stage": name,
        "status": "blocked" if quota_hit else ("ok" if proc.returncode == 0 else "failed"),
        "exit_code": proc.returncode,
        "quota_marker": quota_hit,
        "log": str(log_path),
    }
    print(json.dumps(result, indent=2))
    return result


def require(path: Path, label: str) -> None:
    if not path.exists():
        raise SystemExit(f"missing {label}: {path}")


def build_stages(args: argparse.Namespace, project_python: str) -> list[tuple[str, str, Path, list[str]]]:
    py = project_python
    exp = EXPERIMENT_ROOT
    result = RESULT_ROOT
    artifact = ARTIFACT_ROOT
    split_manifest = exp / "data" / "split_manifest.csv"
    train_file = INJECAGENT_ROOT / "data" / "splits" / "train_cases_632.json"
    validation_file = INJECAGENT_ROOT / "data" / "splits" / "validation_cases_211.json"
    test_file = INJECAGENT_ROOT / "data" / "splits" / "test_cases_211.json"
    train_output = artifact / "full_train632_deepseek-v4.1-flash-1017.jsonl"
    rule = artifact / "frozen_attack_rule.json"
    validation_attack = artifact / "validation_gstar_deepseek-v4.1-flash-1017.jsonl"
    harness_candidates = artifact / "harness_candidates.json"
    selected_harness = artifact / "selected_harness.json"
    test_attack = artifact / "test_gstar_deepseek-v4.1-flash-1017.jsonl"
    selected_only = artifact / "selected_harness_candidate.json"

    stages = [
        (
            "prepare-split",
            "materialize frozen split from the bundled manifest",
            INJECAGENT_ROOT,
            [
                py,
                str(exp / "scripts" / "materialize_split.py"),
                "--manifest",
                str(split_manifest),
                "--output-dir",
                "data/splits",
            ],
        ),
        (
            "train",
            "full co-evolution on 632 training cases",
            INJECAGENT_ROOT,
            [
                py,
                str(exp / "scripts" / "run_ablation.py"),
                "--mode",
                "full",
                "--model_name",
                args.model,
                "--optimizer_model",
                args.model,
                "--protect_model",
                args.model,
                "--data",
                str(train_file),
                "--workers",
                str(args.workers),
                "--max_rounds",
                str(args.max_rounds),
                "--limit",
                str(args.limit),
                "--use_cache",
                "--output",
                str(train_output),
            ],
        ),
        (
            "build-gstar",
            "synthesize frozen attack rule from training trajectories",
            INJECAGENT_ROOT,
            [
                py,
                str(exp / "scripts" / "build_frozen_attack_rule.py"),
                "--train-results",
                str(train_output),
                "--output",
                str(rule),
                "--model",
                args.model,
            ],
        ),
        (
            "validate-attack",
            "instantiate G* on 211 validation cases",
            INJECAGENT_ROOT,
            [
                py,
                str(exp / "scripts" / "evaluate_frozen_attack.py"),
                "--rule",
                str(rule),
                "--data",
                str(validation_file),
                "--output",
                str(validation_attack),
                "--model_name",
                args.model,
                "--rule_model",
                args.model,
                "--workers",
                str(args.workers),
                "--limit",
                str(args.limit),
                "--use_cache",
            ],
        ),
        (
            "select-harness-candidates",
            "extract diverse harness candidates from training",
            INJECAGENT_ROOT,
            [
                py,
                str(exp / "scripts" / "select_harness_candidates.py"),
                "--train-results",
                str(train_output),
                "--output",
                str(harness_candidates),
                "--top",
                str(args.harness_candidates),
            ],
        ),
        (
            "validate-harness",
            "evaluate harness candidates on validation and select one",
            INJECAGENT_ROOT,
            [
                py,
                str(exp / "scripts" / "evaluate_harness.py"),
                "--attacks",
                str(validation_attack),
                "--candidates",
                str(harness_candidates),
                "--output-dir",
                str(artifact / "validation_harness"),
                "--output-selection",
                str(selected_harness),
                "--model_name",
                args.model,
                "--workers",
                str(args.workers),
                "--limit",
                str(args.limit),
            ],
        ),
        (
            "test-attack",
            "instantiate frozen G* on 211 test cases",
            INJECAGENT_ROOT,
            [
                py,
                str(exp / "scripts" / "evaluate_frozen_attack.py"),
                "--rule",
                str(rule),
                "--data",
                str(test_file),
                "--output",
                str(test_attack),
                "--model_name",
                args.model,
                "--rule_model",
                args.model,
                "--workers",
                str(args.workers),
                "--limit",
                str(args.limit),
                "--use_cache",
            ],
        ),
        (
            "prepare-selected-harness",
            "materialize a single-candidate file for final test evaluation",
            INJECAGENT_ROOT,
            [
                py,
                "-c",
                (
                    "import json,pathlib;"
                    f"s=json.load(open({str(selected_harness)!r}));"
                    f"json.dump({{'candidates':[s['selected']['candidate']]}},open({str(selected_only)!r},'w'),ensure_ascii=False,indent=2)"
                ),
            ],
        ),
        (
            "test-harness",
            "evaluate the selected harness on test G* attacks",
            INJECAGENT_ROOT,
            [
                py,
                str(exp / "scripts" / "evaluate_harness.py"),
                "--attacks",
                str(test_attack),
                "--candidates",
                str(selected_only),
                "--output-dir",
                str(artifact / "test_harness"),
                "--output-selection",
                str(artifact / "test_harness_selection.json"),
                "--model_name",
                args.model,
                "--workers",
                str(args.workers),
                "--limit",
                str(args.limit),
            ],
        ),
        (
            "table3-clean",
            "run five poison-free clean suites with the selected harness",
            INJECAGENT_ROOT,
            [
                py,
                str(exp / "scripts" / "run_clean_utility.py"),
                "--selected-harness",
                str(selected_harness),
                "--models",
                args.clean_models,
                "--benchmarks",
                args.clean_benchmarks,
                "--limit",
                str(args.limit),
                "--output-dir",
                str(result / "table3_clean"),
            ],
        ),
        (
            "table3-cross",
            "evaluate OAtk/EAtk × OHar/EHar combinations",
            INJECAGENT_ROOT,
            [
                py,
                str(exp / "scripts" / "evaluate_cross_combinations.py"),
                "--test-data",
                str(test_file),
                "--evolved-attacks",
                str(test_attack),
                "--selected-harness",
                str(selected_harness),
                "--output-dir",
                str(result / "table3_cross"),
                "--model",
                args.model,
                "--workers",
                str(args.workers),
                "--limit",
                str(args.limit),
            ],
        ),
    ]
    return stages


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--stage",
        default="all",
        choices=[
            "all",
            "prepare-split",
            "train",
            "build-gstar",
            "validate-attack",
            "select-harness-candidates",
            "validate-harness",
            "test-attack",
            "prepare-selected-harness",
            "test-harness",
            "table3-clean",
            "table3-cross",
        ],
    )
    parser.add_argument("--model", default="deepseek-v4.1-flash-1017")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--max-rounds", type=int, default=8)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--harness-candidates", type=int, default=8)
    parser.add_argument(
        "--clean-models",
        default="gpt-5.5,deepseek-v4-flash,glm-5.1",
    )
    parser.add_argument(
        "--clean-benchmarks",
        default="agentdojo,injecagent,bfcl,agentbench,appworld",
    )
    parser.add_argument("--budget-calls", type=int, default=20000)
    parser.add_argument("--allow-over-budget", action="store_true")
    parser.add_argument("--project-python", default=default_project_python())
    parser.add_argument("--model-config", default=default_model_config())
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not Path(args.project_python).is_file():
        raise SystemExit(f"project python does not exist: {args.project_python}")
    if not Path(args.model_config).is_file():
        raise SystemExit(f"model config does not exist: {args.model_config}")
    require(EXPERIMENT_ROOT / "data" / "split_manifest.csv", "split manifest")

    train_cases = args.limit if args.limit > 0 else 632
    heldout_cases = args.limit if args.limit > 0 else 211
    estimate = estimate_pipeline_calls(
        train_cases=train_cases,
        heldout_cases=heldout_cases,
        rounds=args.max_rounds,
        harness_candidates=args.harness_candidates,
    )
    print(json.dumps({"estimated_logical_calls": estimate, "budget": args.budget_calls}, indent=2))
    if estimate["total"] > args.budget_calls and not args.allow_over_budget:
        raise SystemExit(
            "estimated call volume exceeds --budget-calls; reduce cases/rounds/candidates "
            "or pass --allow-over-budget"
        )

    env = command_env(args.project_python, args.model_config)
    stages = build_stages(args, args.project_python)
    selected = (
        stages
        if args.stage == "all"
        else [stage for stage in stages if stage[0] == args.stage]
    )
    for name, description, cwd, command in selected:
        print(f"\n=== {name}: {description} ===")
        result = run_stage(
            name,
            command,
            cwd=cwd,
            env=env,
            dry_run=args.dry_run,
        )
        if result["status"] in {"failed", "blocked"}:
            return 1 if result["status"] == "failed" else 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
