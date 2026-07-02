#!/usr/bin/env python3
"""Check tracked files for repository-hygiene and reproducibility issues."""

from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REQUIRED = (
    "README.md",
    ".gitignore",
    ".gitattributes",
    ".editorconfig",
    "CONTRIBUTING.md",
    "SECURITY.md",
    "THIRD_PARTY_NOTICES.md",
    "docs/ARCHITECTURE.md",
    "docs/REPRODUCIBILITY.md",
    "experiments/aaai27_supplement/README.md",
    "experiments/aaai27_supplement/PAPER_METHOD_COVERAGE.md",
)
FORBIDDEN_PREFIXES = (
    ".venvs/",
    "results/",
    "outputs/",
    "exp7/results/",
    "exp7/agentbench/generated_configs/",
    "InjecAgent/results/",
    "InjecAgent/data/cross_eval/offline_poison/",
    "__pycache__/",
)
FORBIDDEN_SUFFIXES = (".pyc", ".pyo", ".zip")
TEXT_SUFFIXES = {
    ".md",
    ".py",
    ".json",
    ".yaml",
    ".yml",
    ".toml",
    ".sh",
    ".txt",
    ".mjs",
    ".js",
    ".jsx",
    ".css",
    ".html",
}


def tracked_files() -> list[str]:
    try:
        output = subprocess.check_output(
            ["git", "ls-files", "-z"],
            cwd=ROOT,
        )
        return [item for item in output.decode().split("\0") if item]
    except (OSError, subprocess.CalledProcessError):
        return sorted(
            str(path.relative_to(ROOT))
            for path in ROOT.rglob("*")
            if path.is_file() and ".git" not in path.parts
        )


def main() -> int:
    files = tracked_files()
    errors: list[str] = []
    warnings: list[str] = []

    for required in REQUIRED:
        if not (ROOT / required).is_file():
            errors.append(f"missing required file: {required}")

    for name in files:
        if name in {"model_api_secret.py", "model_api_ablation.py"}:
            errors.append(f"local secret/config file tracked: {name}")
        if name.startswith(FORBIDDEN_PREFIXES):
            errors.append(f"generated/runtime path tracked: {name}")
        if name.endswith(FORBIDDEN_SUFFIXES):
            errors.append(f"generated file tracked: {name}")
        if "/._" in name or name.startswith("._"):
            errors.append(f"AppleDouble metadata tracked: {name}")
        if "/work_" in name or name.startswith("work_"):
            errors.append(f"temporary work file tracked: {name}")
        if "/.DS_Store" in name or name.endswith(".DS_Store"):
            errors.append(f"platform metadata tracked: {name}")

    secret_prefix = "s" + "k-"
    secret_pattern = re.compile(
        (re.escape(secret_prefix) + r"[A-Za-z0-9_-]{20,}").encode()
    )
    for name in files:
        path = ROOT / name
        if not path.is_file():
            continue
        try:
            size = path.stat().st_size
        except OSError:
            warnings.append(f"cannot stat: {name}")
            continue
        if size > 20_000_000:
            errors.append(f"large tracked file ({size} bytes): {name}")
        if path.suffix.lower() not in TEXT_SUFFIXES:
            continue
        try:
            data = path.read_bytes()
        except OSError:
            warnings.append(f"cannot read: {name}")
            continue
        if secret_pattern.search(data):
            errors.append(f"secret-like token found in tracked file: {name}")

    for name in (
        "experiments/aaai27_supplement/configs/baselines.json",
        "experiments/aaai27_supplement/configs/experiments.json",
        "experiments/aaai27_supplement/configs/interactive_baselines.json",
        "InjecAgent/data/cross_eval/cell_manifest.json",
    ):
        path = ROOT / name
        try:
            json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:
            errors.append(f"invalid JSON {name}: {exc}")

    print(f"tracked files: {len(files)}")
    for warning in warnings:
        print(f"WARNING: {warning}")
    for error in errors:
        print(f"ERROR: {error}")
    if errors:
        return 1
    print("repository checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
