#!/usr/bin/env python3
"""Audit paper-level code, data, artifacts, and external dependencies."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
EXPERIMENT_ROOT = SCRIPT_DIR.parent
REPO_ROOT = EXPERIMENT_ROOT.parents[1]
INJECAGENT_ROOT = REPO_ROOT / "InjecAgent"
ARTIFACT_ROOT = EXPERIMENT_ROOT / "artifacts"
RESULT_ROOT = REPO_ROOT / "results" / "aaai27_supplement"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def make_case_id(item: dict[str, Any]) -> str:
    return "|".join(
        [
            item.get("User Tool", ""),
            item.get("User Instruction", "")[:120],
            item.get("Attacker Instruction", "")[:120],
            (item.get("Attacker Tools") or [""])[0],
        ]
    )


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def check_core() -> dict[str, Any]:
    manifest = EXPERIMENT_ROOT / "data" / "split_manifest.csv"
    expected_hash = "695ccb17ad3ea55fc6d9e51016605912e91511a9e4187f3497cfa9e3477f051d"
    actual_hash = sha256(manifest) if manifest.is_file() else None
    split_paths = {
        "train": INJECAGENT_ROOT / "data" / "splits" / "train_cases_632.json",
        "validation": INJECAGENT_ROOT / "data" / "splits" / "validation_cases_211.json",
        "test": INJECAGENT_ROOT / "data" / "splits" / "test_cases_211.json",
    }
    expected_counts = {"train": 632, "validation": 211, "test": 211}
    counts = {
        name: len(load_json(path)) if path.is_file() else None
        for name, path in split_paths.items()
    }
    return {
        "manifest_hash": actual_hash,
        "manifest_hash_ok": actual_hash == expected_hash,
        "split_counts": counts,
        "split_counts_ok": counts == expected_counts,
    }


def check_artifacts() -> dict[str, Any]:
    manifest_path = ARTIFACT_ROOT / "MANIFEST.json"
    if not manifest_path.is_file():
        return {"ok": False, "error": "artifact manifest missing"}
    manifest = load_json(manifest_path)
    files: dict[str, Any] = {}
    ok = True
    for name, item in manifest.get("files", {}).items():
        path = ARTIFACT_ROOT / name
        actual = sha256(path) if path.is_file() else None
        file_ok = actual == item.get("sha256")
        ok = ok and file_ok
        files[name] = {
            "exists": path.is_file(),
            "sha256": actual,
            "expected_sha256": item.get("sha256"),
            "hash_ok": file_ok,
        }
    attack_path = ARTIFACT_ROOT / "frozen_test_attack_211.jsonl"
    attack_records = load_jsonl(attack_path)
    poison_count = sum(bool(record.get("frozen_poison")) for record in attack_records)
    return {
        "ok": ok and len(attack_records) == 211 and poison_count == 211,
        "files": files,
        "attack_records": len(attack_records),
        "attack_poison_coverage": poison_count,
    }


def cross_eval_contract() -> dict[str, Any]:
    sys.path.insert(0, str(INJECAGENT_ROOT))
    try:
        from src.cross_eval.constants import (
            CROSS_EVAL_ATTACKS,
            CROSS_EVAL_DEFENSES,
            EXTRA_CROSS_EVAL_ATTACKS,
            EXTRA_CROSS_EVAL_DEFENSES,
        )
    except Exception as exc:
        return {"ok": False, "error": str(exc)}
    manifest_path = INJECAGENT_ROOT / "data" / "cross_eval" / "cell_manifest.json"
    manifest = load_json(manifest_path) if manifest_path.is_file() else {}
    return {
        "ok": (
            len(CROSS_EVAL_ATTACKS) == 9
            and len(CROSS_EVAL_DEFENSES) == 5
            and manifest.get("total_cells") == 225
        ),
        "table2_attacks": list(CROSS_EVAL_ATTACKS),
        "table2_defenses": list(CROSS_EVAL_DEFENSES),
        "extra_table1_attacks": list(EXTRA_CROSS_EVAL_ATTACKS),
        "extra_table1_defenses": list(EXTRA_CROSS_EVAL_DEFENSES),
        "manifest_cells": manifest.get("total_cells"),
    }


def adaptive_coverage() -> dict[str, Any]:
    test_cases = load_json(
        INJECAGENT_ROOT / "data" / "splits" / "test_cases_211.json"
    )
    test_ids = {make_case_id(item) for item in test_cases}
    models = (
        "deepseek-v3.2",
        "deepseek-v4-flash",
        "glm-5.1",
        "glm-5.1-fp8",
        "qwen3.5-27b",
    )
    coverage: dict[str, Any] = {}
    for model in models:
        path = (
            INJECAGENT_ROOT
            / "data"
            / "cross_eval"
            / "offline_poison"
            / model
            / "adaptive_attack_base.jsonl"
        )
        records = load_jsonl(path)
        ids = {
            record.get("case_id") or make_case_id(record)
            for record in records
            if record.get("final_poison")
        }
        count = len(test_ids & ids)
        coverage[model] = {
            "cached": count,
            "expected": len(test_ids),
            "complete": count == len(test_ids),
        }
    return coverage


def external_status() -> dict[str, Any]:
    script = EXPERIMENT_ROOT / "scripts" / "check_external_benchmarks.py"
    if not script.is_file():
        return {"ok": False, "error": "external benchmark checker missing"}
    proc = subprocess.run(
        [sys.executable, str(script), "--json"],
        capture_output=True,
        text=True,
    )
    try:
        report = json.loads(proc.stdout)
    except json.JSONDecodeError:
        return {
            "ok": False,
            "error": (proc.stderr or proc.stdout).strip()[-1000:],
        }
    return report


def baseline_status() -> dict[str, Any]:
    registry_path = EXPERIMENT_ROOT / "configs" / "baselines.json"
    registry = load_json(registry_path)
    result: dict[str, Any] = {}
    for name, spec in registry["baselines"].items():
        path = (REPO_ROOT / spec["repo_dir"]).resolve()
        commit_file = path / "VENDORED_COMMIT.txt"
        commit = (
            commit_file.read_text(encoding="utf-8").strip()
            if commit_file.is_file()
            else None
        )
        result[name] = {
            "exists": path.is_dir(),
            "commit": commit,
            "commit_ok": commit == spec["commit"],
            "provenance": spec["provenance"],
        }
    return {"ok": all(item["exists"] and item["commit_ok"] for item in result.values()), "items": result}


def interactive_status() -> dict[str, Any]:
    config = load_json(EXPERIMENT_ROOT / "configs" / "interactive_baselines.json")
    root = RESULT_ROOT / "interactive_baselines"
    completed = len(list(root.rglob("cell.json"))) if root.is_dir() else 0
    expected = (
        len(config["models"])
        * len(config["suites"])
        * len(config["defenses"])
    )
    return {
        "config_ok": [item["id"] for item in config["defenses"]]
        == ["task_shield", "melon", "camel"],
        "expected_cells": expected,
        "completed_cells": completed,
    }


def build_report() -> dict[str, Any]:
    core = check_core()
    artifacts = check_artifacts()
    contract = cross_eval_contract()
    adaptive = adaptive_coverage()
    external = external_status()
    baselines = baseline_status()
    interactive = interactive_status()
    core_ok = (
        core["manifest_hash_ok"]
        and core["split_counts_ok"]
        and artifacts["ok"]
        and contract["ok"]
        and baselines["ok"]
    )
    adaptive_complete = all(item["complete"] for item in adaptive.values())
    docker_ready = bool(external.get("agentbench", {}).get("worker_ready"))
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "core_ok": core_ok,
        "core": core,
        "artifacts": artifacts,
        "cross_eval_contract": contract,
        "adaptive_attack": {
            "complete": adaptive_complete,
            "coverage": adaptive,
        },
        "external": external,
        "baselines": baselines,
        "interactive_baselines": interactive,
        "paper_tables": {
            "table1": {
                "code_ready": contract["ok"],
                "artifacts_ready": artifacts["ok"],
                "adaptive_complete": adaptive_complete,
                "status": "ready"
                if artifacts["ok"] and adaptive_complete
                else "partial",
            },
            "table2": {
                "code_ready": contract["ok"],
                "artifacts_ready": artifacts["ok"],
                "adaptive_complete": adaptive_complete,
                "status": "partial" if not adaptive_complete else "ready",
            },
            "table3": {
                "code_ready": True,
                "appworld_ready": external.get("appworld", {}).get("python", {}).get("ok", False),
                "agentbench_docker_ready": docker_ready,
                "status": "ready" if docker_ready else "partial",
            },
            "interactive": {
                "code_ready": interactive["config_ok"],
                "completed_cells": interactive["completed_cells"],
                "expected_cells": interactive["expected_cells"],
                "status": "ready" if interactive["completed_cells"] == interactive["expected_cells"] else "pending",
            },
        },
    }


def markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Reproducibility Status",
        "",
        f"Generated: `{report['generated_at']}`",
        "",
        "## Core checks",
        "",
        "| Check | Status |",
        "|---|---|",
        f"| Split manifest hash | {'PASS' if report['core']['manifest_hash_ok'] else 'FAIL'} |",
        f"| Split counts | {'PASS' if report['core']['split_counts_ok'] else 'FAIL'} |",
        f"| Frozen artifacts | {'PASS' if report['artifacts']['ok'] else 'FAIL'} |",
        f"| Table 2 cross-eval contract | {'PASS' if report['cross_eval_contract']['ok'] else 'FAIL'} |",
        f"| Baseline provenance | {'PASS' if report['baselines']['ok'] else 'FAIL'} |",
        "",
        "## Paper tables",
        "",
        "| Table | Code | Artifact/data state | Current status |",
        "|---|---|---|---|",
        f"| Table 1 | {'READY' if report['paper_tables']['table1']['code_ready'] else 'FAIL'} | "
        f"frozen attack {report['artifacts']['attack_poison_coverage']}/211; Adaptive caches "
        f"{'complete' if report['adaptive_attack']['complete'] else 'incomplete'} | "
        f"{report['paper_tables']['table1']['status']} |",
        f"| Table 2 | {'READY' if report['paper_tables']['table2']['code_ready'] else 'FAIL'} | "
        f"{report['cross_eval_contract']['manifest_cells']} cells; Adaptive caches "
        f"{'complete' if report['adaptive_attack']['complete'] else 'incomplete'} | "
        f"{report['paper_tables']['table2']['status']} |",
        f"| Table 3 | READY | AppWorld "
        f"{'ready' if report['paper_tables']['table3']['appworld_ready'] else 'blocked'}; "
        f"AgentBench Docker {'ready' if report['paper_tables']['table3']['agentbench_docker_ready'] else 'blocked'} | "
        f"{report['paper_tables']['table3']['status']} |",
        f"| Interactive baselines | {'READY' if report['paper_tables']['interactive']['code_ready'] else 'FAIL'} | "
        f"{report['interactive_baselines']['completed_cells']}/{report['interactive_baselines']['expected_cells']} cells | "
        f"{report['paper_tables']['interactive']['status']} |",
        "",
        "## Adaptive Attack coverage",
        "",
        "| Model | Cached | Expected |",
        "|---|---:|---:|",
    ]
    for model, item in report["adaptive_attack"]["coverage"].items():
        lines.append(f"| {model} | {item['cached']} | {item['expected']} |")
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- `PASS` means the code and tracked data satisfy the local reproducibility contract.",
            "- `partial` means some paper rows require regenerating per-victim Adaptive artifacts.",
            "- `blocked` means an external environment (usually Docker for AgentBench) is unavailable.",
            "- No model API is called by this audit.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--json-out")
    parser.add_argument("--markdown-out")
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Exit non-zero when a paper table is not fully ready.",
    )
    args = parser.parse_args()
    report = build_report()
    text = markdown(report)
    print(text)
    if args.json_out:
        Path(args.json_out).write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    if args.markdown_out:
        Path(args.markdown_out).write_text(text + "\n", encoding="utf-8")
    if not report["core_ok"]:
        return 1
    if args.strict and any(
        item["status"] != "ready"
        for item in report["paper_tables"].values()
    ):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
