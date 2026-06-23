#!/usr/bin/env python3
"""Run Table 3(a) clean utility with an arbitrary validation-selected harness."""

from __future__ import annotations

import argparse
import csv
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
EXP7_RESULTS = REPO_ROOT / "exp7" / "results"


QUOTA_MARKERS = (
    "quota exceeded",
    "insufficient_quota",
    "user has been banned",
    "too many requests",
    "error code: 429",
    "error code: 529",
)


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")


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


def command_env(project_python: str, model_config: str, harness_file: Path) -> dict[str, str]:
    project_bin = Path(project_python).parent
    env = os.environ.copy()
    env["CONDA_PREFIX"] = str(project_bin.parent)
    env["PATH"] = str(project_bin) + os.pathsep + env.get("PATH", "")
    env["LD_LIBRARY_PATH"] = str(project_bin.parent / "lib") + os.pathsep + env.get(
        "LD_LIBRARY_PATH", ""
    )
    env["PYTHONPATH"] = str(REPO_ROOT) + os.pathsep + env.get("PYTHONPATH", "")
    env["TOOLS_ATTACK_MODEL_CONFIG"] = model_config
    env["TOOLS_ATTACK_HARNESS_STRATEGY_FILE"] = str(harness_file)
    return env


def result_path(benchmark: str, model: str, harness: str) -> Path:
    safe_model = model.replace("/", "_")
    return EXP7_RESULTS / benchmark / safe_model / f"{harness}.json"


def load_result(benchmark: str, model: str, harness: str) -> dict:
    path = result_path(benchmark, model, harness)
    if not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def fmt(value: float | None) -> str:
    return "—" if value is None else f"{value:.2f}"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--selected-harness", required=True)
    parser.add_argument(
        "--models",
        default="gpt-5.5,deepseek-v4-flash,glm-5.1",
        help="Comma-separated model names",
    )
    parser.add_argument(
        "--benchmarks",
        default="agentdojo,injecagent,bfcl,agentbench,appworld",
    )
    parser.add_argument("--selected-harness-label", default="Ours")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--output-dir",
        default=str(EXPERIMENT_ROOT.parents[1] / "results" / "aaai27_supplement" / "table3_clean"),
    )
    parser.add_argument("--project-python", default=default_project_python())
    parser.add_argument("--model-config", default=default_model_config())
    args = parser.parse_args()

    selection = json.loads(Path(args.selected_harness).read_text(encoding="utf-8"))
    candidate = selection["selected"]["candidate"]
    strategy = candidate["strategy"]
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    harness_file = output_dir / "selected_harness_strategy.json"
    harness_file.write_text(
        json.dumps(strategy, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    models = [item.strip() for item in args.models.split(",") if item.strip()]
    benchmarks = [item.strip() for item in args.benchmarks.split(",") if item.strip()]
    env = command_env(args.project_python, args.model_config, harness_file)
    log_dir = output_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    for model in models:
        for benchmark in benchmarks:
            for harness in ("baseline", "max_defense"):
                cmd = [
                    args.project_python,
                    "-m",
                    "exp7.run_exp7",
                    "--benchmark",
                    benchmark,
                    "--model",
                    model,
                    "--harness",
                    harness,
                ]
                if harness == "max_defense":
                    cmd.extend(["--harness-file", str(harness_file)])
                if args.limit > 0:
                    cmd.extend(["--limit", str(args.limit)])
                if args.force:
                    cmd.append("--force")
                print(json.dumps({"cwd": str(REPO_ROOT), "command": cmd}, indent=2))
                if args.dry_run:
                    continue
                log_path = log_dir / f"{benchmark}__{model.replace('/', '_')}__{harness}.log"
                with log_path.open("w", encoding="utf-8") as log:
                    proc = subprocess.run(
                        cmd,
                        cwd=str(REPO_ROOT),
                        env=env,
                        stdout=log,
                        stderr=subprocess.STDOUT,
                        text=True,
                    )
                text = log_path.read_text(encoding="utf-8", errors="ignore").lower()
                if any(marker in text for marker in QUOTA_MARKERS):
                    print(json.dumps({"status": "blocked", "log": str(log_path)}, indent=2))
                    return 2
                if proc.returncode != 0:
                    print(
                        json.dumps(
                            {
                                "status": "failed",
                                "exit_code": proc.returncode,
                                "log": str(log_path),
                                "command": cmd,
                            },
                            indent=2,
                        )
                    )
                    return 1

    rows = []
    for benchmark in benchmarks:
        for model in models:
            baseline = load_result(benchmark, model, "baseline")
            selected = load_result(benchmark, model, "max_defense")
            base_u = baseline.get("utility")
            selected_u = selected.get("utility")
            delta = (
                round(float(selected_u) - float(base_u), 2)
                if base_u is not None and selected_u is not None
                else None
            )
            rows.append(
                {
                    "benchmark": benchmark,
                    "model": model,
                    "baseline_utility": base_u,
                    "selected_utility": selected_u,
                    "delta": delta,
                    "baseline_status": baseline.get("status"),
                    "selected_status": selected.get("status"),
                }
            )

    (output_dir / "clean_utility_summary.json").write_text(
        json.dumps({"harness": candidate, "rows": rows}, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    with (output_dir / "clean_utility_summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]) if rows else [])
        if rows:
            writer.writeheader()
            writer.writerows(rows)
    lines = [
        "| Benchmark | Model | Orig. | Ours | ΔUtility |",
        "|---|---|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['benchmark']} | {row['model']} | {fmt(row['baseline_utility'])} | "
            f"{fmt(row['selected_utility'])} | {fmt(row['delta'])} |"
        )
    (output_dir / "clean_utility_summary.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
