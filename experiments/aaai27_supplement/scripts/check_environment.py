#!/usr/bin/env python3
"""Check paths, split integrity, model configuration, and Python dependencies."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
EXPERIMENT_ROOT = SCRIPT_DIR.parent
REPO_ROOT = EXPERIMENT_ROOT.parents[1]
INJECAGENT_ROOT = REPO_ROOT / "InjecAgent"
SPLIT_DIR = INJECAGENT_ROOT / "data" / "splits"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_model_config(path: Path) -> dict:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    values = {}
    for node in tree.body:
        if (
            isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
        ):
            try:
                values[node.targets[0].id] = ast.literal_eval(node.value)
            except (ValueError, TypeError):
                continue
    return values


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--project-python",
        default=str(Path.home() / "anaconda3" / "envs" / "tools_attack" / "bin" / "python"),
    )
    parser.add_argument(
        "--model-config",
        default=str(Path.home() / "contract_heldout_ipi" / "model_api_ablation.py"),
    )
    args = parser.parse_args()

    report = {
        "cwd": str(Path.cwd()),
        "repo_root": str(REPO_ROOT),
        "injecagent_root": str(INJECAGENT_ROOT),
        "project_python": args.project_python,
        "model_config": args.model_config,
        "checks": {},
    }
    checks = report["checks"]

    required_sources = [
        INJECAGENT_ROOT / "src/evaluate_prompted_agent.py",
        INJECAGENT_ROOT / "src/evaluate_protect_agent.py",
        INJECAGENT_ROOT / "src/cross_eval/scoring.py",
        INJECAGENT_ROOT / "src/prompts/optimizer_prompt.py",
        INJECAGENT_ROOT / "src/prompts/protect_prompt.py",
    ]
    checks["source_files"] = {
        str(path): path.is_file() for path in required_sources
    }

    manifest = EXPERIMENT_ROOT / "data" / "split_manifest.csv"
    checks["manifest_exists"] = manifest.is_file()
    if manifest.is_file():
        checks["manifest_sha256"] = sha256_file(manifest)
        checks["manifest_expected_sha256"] = (
            "695ccb17ad3ea55fc6d9e51016605912e91511a9e4187f3497cfa9e3477f051d"
        )
        checks["manifest_hash_ok"] = (
            checks["manifest_sha256"] == checks["manifest_expected_sha256"]
        )

    for name, expected in (
        ("train_case_ids.txt", 632),
        ("validation_case_ids.txt", 211),
        ("test_case_ids.txt", 211),
    ):
        path = EXPERIMENT_ROOT / "data" / name
        checks[name] = {
            "exists": path.is_file(),
            "lines": sum(1 for _ in path.open(encoding="utf-8")) if path.is_file() else 0,
            "expected": expected,
        }

    for name, expected in (
        ("train_cases_632.json", 632),
        ("validation_cases_211.json", 211),
        ("test_cases_211.json", 211),
    ):
        path = SPLIT_DIR / name
        count = 0
        if path.is_file():
            count = len(json.loads(path.read_text(encoding="utf-8")))
        checks[name] = {
            "exists": path.is_file(),
            "cases": count,
            "expected": expected,
        }

    model_path = Path(args.model_config)
    checks["model_config_exists"] = model_path.is_file()
    if model_path.is_file():
        try:
            config = read_model_config(model_path)
            checks["model_roles"] = {
                key: config.get(key)
                for key in (
                    "ATTACK_MODEL",
                    "DEFENSE_MODEL",
                    "JUDGE_MODEL",
                    "WORK_AGENT",
                    "VICTIM_MODEL",
                    "OPTIMIZER_MODEL",
                    "PROTECT_MODEL",
                )
            }
            checks["effective_roles"] = {
                "VICTIM_MODEL": config.get("VICTIM_MODEL")
                or config.get("WORK_AGENT")
                or config.get("DEFENSE_MODEL")
                or config.get("ATTACK_MODEL"),
                "OPTIMIZER_MODEL": config.get("OPTIMIZER_MODEL")
                or config.get("ATTACK_MODEL")
                or config.get("WORK_AGENT"),
                "PROTECT_MODEL": config.get("PROTECT_MODEL")
                or config.get("DEFENSE_MODEL")
                or config.get("WORK_AGENT"),
            }
            checks["base_url"] = config.get("OPENAI_BASE_URL")
            checks["api_key_present"] = bool(config.get("OPENAI_API_KEY"))
        except Exception as exc:
            checks["model_config_error"] = str(exc)

    python_path = Path(args.project_python)
    checks["project_python_exists"] = python_path.is_file()
    if python_path.is_file():
        proc = subprocess.run(
            [
                str(python_path),
                "-c",
                "import nltk, openai, tqdm; print('dependencies-ok')",
            ],
            capture_output=True,
            text=True,
            env={
                **dict(os.environ),
                "CONDA_PREFIX": str(python_path.parent.parent),
                "PATH": str(python_path.parent) + ":" + os.environ.get("PATH", ""),
                "LD_LIBRARY_PATH": str(python_path.parent.parent / "lib")
                + ":"
                + os.environ.get("LD_LIBRARY_PATH", ""),
            },
        )
        checks["dependencies"] = {
            "ok": proc.returncode == 0,
            "output": (proc.stdout or proc.stderr).strip()[-500:],
        }

    print(json.dumps(report, ensure_ascii=False, indent=2))
    checks_ok = (
        all(checks["source_files"].values())
        and checks.get("manifest_hash_ok", False)
        and checks.get("model_config_exists", False)
        and checks.get("api_key_present", False)
        and checks.get("project_python_exists", False)
        and checks.get("dependencies", {}).get("ok", False)
    )
    return 0 if checks_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
