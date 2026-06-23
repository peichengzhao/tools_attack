#!/usr/bin/env python3
"""Evaluate the four artifact combinations used by Table 3(b)."""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
EXPERIMENT_ROOT = SCRIPT_DIR.parent
REPO_ROOT = EXPERIMENT_ROOT.parents[1]
INJECAGENT_ROOT = REPO_ROOT / "InjecAgent"
sys.path.insert(0, str(INJECAGENT_ROOT))

from src.prompts.protect_prompt import empty_defense_strategy  # noqa: E402

QUOTA_MARKERS = (
    "quota exceeded",
    "insufficient_quota",
    "user has been banned",
    "too many requests",
    "error code: 429",
    "error code: 529",
)


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
    return str(candidate) if candidate.is_file() else str(REPO_ROOT / "model_api_ablation.py")


def env_for(project_python: str, model_config: str) -> dict[str, str]:
    project_bin = Path(project_python).parent
    env = os.environ.copy()
    env["CONDA_PREFIX"] = str(project_bin.parent)
    env["PATH"] = str(project_bin) + os.pathsep + env.get("PATH", "")
    env["LD_LIBRARY_PATH"] = str(project_bin.parent / "lib") + os.pathsep + env.get(
        "LD_LIBRARY_PATH", ""
    )
    env["PYTHONPATH"] = str(INJECAGENT_ROOT) + os.pathsep + env.get("PYTHONPATH", "")
    env["TOOLS_ATTACK_MODEL_CONFIG"] = model_config
    return env


def write_candidate(path: Path, candidate_id: str, strategy: dict) -> None:
    path.write_text(
        json.dumps(
            {"candidates": [{"candidate_id": candidate_id, "strategy": strategy}]},
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--test-data", required=True)
    parser.add_argument("--evolved-attacks", required=True)
    parser.add_argument("--selected-harness", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--model", default="deepseek-v4.1-flash-1017")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--project-python", default=default_project_python())
    parser.add_argument("--model-config", default=default_model_config())
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    selected = json.loads(Path(args.selected_harness).read_text(encoding="utf-8"))
    selected_strategy = selected["selected"]["candidate"]["strategy"]
    empty_candidate = output_dir / "candidate_OHar.json"
    selected_candidate = output_dir / "candidate_EHar.json"
    write_candidate(empty_candidate, "OHar", empty_defense_strategy())
    write_candidate(selected_candidate, "EHar", selected_strategy)

    combos = [
        ("OAtk+OHar", args.test_data, empty_candidate),
        ("EAtk+OHar", args.evolved_attacks, empty_candidate),
        ("OAtk+EHar", args.test_data, selected_candidate),
        ("EAtk+EHar", args.evolved_attacks, selected_candidate),
    ]
    env = env_for(args.project_python, args.model_config)
    rows = []
    for name, attacks, candidates in combos:
        combo_dir = output_dir / name.replace("+", "_")
        selection_path = combo_dir / "selection.json"
        command = [
            args.project_python,
            str(SCRIPT_DIR / "evaluate_harness.py"),
            "--attacks",
            str(attacks),
            "--candidates",
            str(candidates),
            "--output-dir",
            str(combo_dir / "runs"),
            "--output-selection",
            str(selection_path),
            "--model_name",
            args.model,
            "--workers",
            str(args.workers),
        ]
        if args.limit > 0:
            command.extend(["--limit", str(args.limit)])
        print(json.dumps({"combo": name, "command": command}, indent=2))
        if args.dry_run:
            continue
        combo_dir.mkdir(parents=True, exist_ok=True)
        log_path = combo_dir / "run.log"
        with log_path.open("w", encoding="utf-8") as log:
            proc = subprocess.run(
                command,
                cwd=str(INJECAGENT_ROOT),
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
                text=True,
            )
        text = log_path.read_text(encoding="utf-8", errors="ignore").lower()
        if any(marker in text for marker in QUOTA_MARKERS):
            print(json.dumps({"status": "blocked", "combo": name, "log": str(log_path)}))
            return 2
        if proc.returncode != 0 or not selection_path.is_file():
            print(json.dumps({"status": "failed", "combo": name, "log": str(log_path)}))
            return 1
        result = json.loads(selection_path.read_text(encoding="utf-8"))
        summary = result["selected"]["summary"]
        rows.append(
            {
                "combination": name,
                "attack": name.split("+")[0],
                "defense": name.split("+")[1],
                "total_cases": summary["total_cases"],
                "valid": summary["valid"],
                "invalid": summary["invalid"],
                "asr_all_pct": summary["asr_all_pct"],
                "invalid_rate_pct": summary["invalid_rate_pct"],
                "utility_pct": summary["utility_pct_all"],
                "svu_pct": summary["svu_pct"],
            }
        )

    if rows:
        with (output_dir / "cross_combinations.csv").open(
            "w", newline="", encoding="utf-8"
        ) as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    lines = [
        "| Combination | Total | Valid | Invalid | ASR-all | Invalid rate | Utility | SVU |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['combination']} | {row['total_cases']} | {row['valid']} | "
            f"{row['invalid']} | {row['asr_all_pct']} | {row['invalid_rate_pct']} | "
            f"{row['utility_pct']} | {row['svu_pct']} |"
        )
    (output_dir / "cross_combinations.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    if rows:
        print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
