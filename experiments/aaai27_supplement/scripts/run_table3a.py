#!/usr/bin/env python3
"""Run the paper Table 3(a) clean-utility comparison."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from paper_table_common import (  # noqa: E402
    EXPERIMENT_ROOT,
    REPO_ROOT,
    SELECTED_HARNESS,
    TABLE3A_BENCHMARKS,
    TABLE3A_MODELS,
    project_python,
    run_command,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", default=",".join(TABLE3A_MODELS))
    parser.add_argument("--benchmarks", default=",".join(TABLE3A_BENCHMARKS))
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    if not SELECTED_HARNESS.is_file():
        raise SystemExit(f"Missing selected harness: {SELECTED_HARNESS}")

    command = [
        project_python(),
        str(EXPERIMENT_ROOT / "scripts" / "run_clean_utility.py"),
        "--selected-harness",
        str(SELECTED_HARNESS),
        "--models",
        args.models,
        "--benchmarks",
        args.benchmarks,
    ]
    if args.limit > 0:
        command.extend(["--limit", str(args.limit)])
    if args.force:
        command.append("--force")
    if args.dry_run:
        command.append("--dry-run")
    return run_command(command, cwd=EXPERIMENT_ROOT, dry_run=False)


if __name__ == "__main__":
    raise SystemExit(main())
