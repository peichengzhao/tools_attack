#!/usr/bin/env python3
"""Verify the tracked reproducibility contract without calling a model API."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
EXPERIMENT_ROOT = SCRIPT_DIR.parent
REPO_ROOT = EXPERIMENT_ROOT.parents[1]
INJECAGENT_ROOT = REPO_ROOT / "InjecAgent"
ARTIFACT_ROOT = EXPERIMENT_ROOT / "artifacts"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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


def check_splits() -> dict[str, Any]:
    manifest = EXPERIMENT_ROOT / "data" / "split_manifest.csv"
    expected_hash = "695ccb17ad3ea55fc6d9e51016605912e91511a9e4187f3497cfa9e3477f051d"
    actual_hash = sha256(manifest) if manifest.is_file() else None
    paths = {
        "train": INJECAGENT_ROOT / "data" / "splits" / "train_cases_632.json",
        "validation": INJECAGENT_ROOT / "data" / "splits" / "validation_cases_211.json",
        "test": INJECAGENT_ROOT / "data" / "splits" / "test_cases_211.json",
    }
    counts = {
        name: len(load_json(path)) if path.is_file() else None
        for name, path in paths.items()
    }
    return {
        "manifest_hash": actual_hash,
        "manifest_hash_ok": actual_hash == expected_hash,
        "counts": counts,
        "counts_ok": counts == {"train": 632, "validation": 211, "test": 211},
    }


def check_artifacts() -> dict[str, Any]:
    manifest_path = ARTIFACT_ROOT / "MANIFEST.json"
    manifest = load_json(manifest_path) if manifest_path.is_file() else {}
    files: dict[str, Any] = {}
    ok = manifest_path.is_file()
    for name, item in manifest.get("files", {}).items():
        path = ARTIFACT_ROOT / name
        actual = sha256(path) if path.is_file() else None
        file_ok = actual == item.get("sha256")
        ok = ok and file_ok
        files[name] = {
            "exists": path.is_file(),
            "sha256": actual,
            "hash_ok": file_ok,
        }
    attack_records = load_jsonl(ARTIFACT_ROOT / "frozen_test_attack_211.jsonl")
    payload_count = sum(bool(record.get("frozen_poison")) for record in attack_records)
    return {
        "ok": ok and len(attack_records) == 211 and payload_count == 211,
        "files": files,
        "attack_records": len(attack_records),
        "attack_payloads": payload_count,
    }


def check_cross_eval_contract() -> dict[str, Any]:
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
        "attacks": list(CROSS_EVAL_ATTACKS),
        "defenses": list(CROSS_EVAL_DEFENSES),
        "table1_extra_attacks": list(EXTRA_CROSS_EVAL_ATTACKS),
        "table1_extra_defenses": list(EXTRA_CROSS_EVAL_DEFENSES),
        "manifest_cells": manifest.get("total_cells"),
    }


def check_baselines() -> dict[str, Any]:
    registry = load_json(EXPERIMENT_ROOT / "configs" / "baselines.json")
    items: dict[str, Any] = {}
    for name, spec in registry["baselines"].items():
        path = (REPO_ROOT / spec["repo_dir"]).resolve()
        commit_file = path / "VENDORED_COMMIT.txt"
        commit = (
            commit_file.read_text(encoding="utf-8").strip()
            if commit_file.is_file()
            else None
        )
        items[name] = {
            "exists": path.is_dir(),
            "commit": commit,
            "commit_ok": commit == spec["commit"],
            "provenance": spec["provenance"],
        }
    return {
        "ok": all(item["exists"] and item["commit_ok"] for item in items.values()),
        "items": items,
    }


def build_report() -> dict[str, Any]:
    splits = check_splits()
    artifacts = check_artifacts()
    contract = check_cross_eval_contract()
    baselines = check_baselines()
    checks = {
        "split_manifest_hash": splits["manifest_hash_ok"],
        "split_counts": splits["counts_ok"],
        "frozen_artifacts": artifacts["ok"],
        "cross_eval_contract": contract["ok"],
        "baseline_provenance": baselines["ok"],
    }
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "ok": all(checks.values()),
        "checks": checks,
        "splits": splits,
        "artifacts": artifacts,
        "cross_eval_contract": contract,
        "baselines": baselines,
    }


def markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Reproduction Contract Check",
        "",
        f"Generated: `{report['generated_at']}`",
        "",
        "| Contract | Status |",
        "|---|---|",
    ]
    labels = {
        "split_manifest_hash": "Split manifest SHA-256",
        "split_counts": "Split counts (632 / 211 / 211)",
        "frozen_artifacts": "Frozen attack and harness artifacts",
        "cross_eval_contract": "Table 2 cross-evaluation manifest",
        "baseline_provenance": "Pinned baseline provenance",
    }
    for key, label in labels.items():
        lines.append(f"| {label} | {'PASS' if report['checks'][key] else 'FAIL'} |")
    lines.extend(
        [
            "",
            f"- Frozen attack payloads: `{report['artifacts']['attack_payloads']}/211`",
            f"- Table 2 manifest: `{report['cross_eval_contract']['manifest_cells']} cells`",
            "- No model API was called.",
            "",
            "Runtime prerequisites for full execution are documented in",
            "`docs/REPRODUCIBILITY.md`.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--json-out")
    parser.add_argument("--markdown-out")
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
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
