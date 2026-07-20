#!/usr/bin/env python3
"""Run the paper Table 3(b) sweep over co-evolution rounds N=4..9."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from paper_table_common import (  # noqa: E402
    EXPERIMENT_ROOT,
    INJECAGENT_ROOT,
    REPO_ROOT,
    SELECTED_HARNESS,
    TABLE3B_ROUNDS,
    TEST_CASES,
    TRAIN_CASES,
    project_python,
    run_command,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="deepseek-v4-flash")
    parser.add_argument("--rounds", default=",".join(str(n) for n in TABLE3B_ROUNDS))
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--evolved-dir",
        default="",
        help="Directory of existing round_{N}.jsonl files. Skips co-evolution when a file exists.",
    )
    args = parser.parse_args()

    if not SELECTED_HARNESS.is_file():
        raise SystemExit(f"Missing selected harness: {SELECTED_HARNESS}")
    if not TRAIN_CASES.is_file() or not TEST_CASES.is_file():
        raise SystemExit("Missing the 632/211 split files.")

    rounds = [int(item) for item in args.rounds.split(",") if item.strip()]
    output_root = REPO_ROOT / "results" / "aaai27_supplement" / "table3b" / args.model
    evolved_dir = Path(args.evolved_dir) if args.evolved_dir else output_root
    for n_rounds in rounds:
        evolved = evolved_dir / f"round_{n_rounds}.jsonl"
        if not evolved.is_file():
            train_command = [
                project_python(),
                str(EXPERIMENT_ROOT / "scripts" / "run_ablation.py"),
                "--mode",
                "full",
                "--model_name",
                args.model,
                "--optimizer_model",
                args.model,
                "--protect_model",
                args.model,
                "--data",
                str(TRAIN_CASES),
                "--workers",
                str(args.workers),
                "--max_rounds",
                str(n_rounds),
                "--output",
                str(evolved),
            ]
            if args.limit > 0:
                train_command.extend(["--limit", str(args.limit)])
            if args.dry_run:
                train_command.append("--dry_run")
            code = run_command(train_command, cwd=INJECAGENT_ROOT, dry_run=False)
            if code != 0:
                return code
        combo_command = [
            project_python(),
            str(EXPERIMENT_ROOT / "scripts" / "evaluate_cross_combinations.py"),
            "--test-data",
            str(TEST_CASES),
            "--evolved-attacks",
            str(evolved),
            "--selected-harness",
            str(SELECTED_HARNESS),
            "--output-dir",
            str(output_root / f"N{n_rounds}"),
            "--model",
            args.model,
            "--workers",
            str(args.workers),
        ]
        if args.limit > 0:
            combo_command.extend(["--limit", str(args.limit)])
        if args.dry_run:
            combo_command.append("--dry-run")
        code = run_command(combo_command, cwd=EXPERIMENT_ROOT, dry_run=False)
        if code != 0:
            return code
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
