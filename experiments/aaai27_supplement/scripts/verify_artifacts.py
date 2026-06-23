#!/usr/bin/env python3
"""Verify training and held-out artifacts produced by the pipeline."""

from __future__ import annotations

import json
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
EXPERIMENT_ROOT = SCRIPT_DIR.parent
REPO_ROOT = EXPERIMENT_ROOT.parents[1]
RESULT_ROOT = REPO_ROOT / "results" / "aaai27_supplement"
ARTIFACT_ROOT = RESULT_ROOT / "artifacts"


def line_count(path: Path) -> int:
    if not path.is_file():
        return 0
    return sum(1 for line in path.open(encoding="utf-8") if line.strip())


def eval_counts(path: Path) -> dict[str, int]:
    counts: dict[str, int] = {}
    if not path.is_file():
        return counts
    for line in path.open(encoding="utf-8"):
        if not line.strip():
            continue
        value = json.loads(line).get("eval", "unknown")
        counts[value] = counts.get(value, 0) + 1
    return counts


def main() -> int:
    base = ARTIFACT_ROOT
    validation_path = base / "validation_gstar_deepseek-v4.1-flash-1017.jsonl"
    test_path = base / "test_gstar_deepseek-v4.1-flash-1017.jsonl"
    validation_evals = eval_counts(validation_path)
    test_evals = eval_counts(test_path)
    checks = {
        "training": line_count(
            base / "full_train632_deepseek-v4.1-flash-1017.jsonl"
        ),
        "frozen_rule": (base / "frozen_attack_rule.json").is_file(),
        "harness_candidates": (base / "harness_candidates.json").is_file(),
        "selected_harness": (base / "selected_harness.json").is_file(),
        "validation_gstar": line_count(validation_path),
        "validation_eval_counts": validation_evals,
        "validation_harness": len(list((base / "validation_harness").glob("*.jsonl"))),
        "test_gstar": line_count(test_path),
        "test_eval_counts": test_evals,
        "test_harness": len(list((base / "test_harness").glob("*.jsonl"))),
    }
    status = "ok"
    if checks["training"] != 632:
        status = "incomplete"
    if not checks["frozen_rule"] or not checks["selected_harness"]:
        status = "incomplete"
    if checks["validation_gstar"] not in (0, 211):
        status = "incomplete"
    if checks["test_gstar"] not in (0, 211):
        status = "incomplete"
    if checks["test_gstar"] == 211 and not (
        test_evals.get("succ", 0) or test_evals.get("unsucc", 0)
    ):
        status = "test_invalid"

    report = {"status": status, "checks": checks, "artifact_root": str(base)}
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if status == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())
