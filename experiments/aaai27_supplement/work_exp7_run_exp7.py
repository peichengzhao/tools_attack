#!/usr/bin/env python3
"""
Experiment 7 — benign utility control study.

Compare original harness vs max_defense (unified system addendum) on clean agent
benchmarks. No injection / no attack.

Examples:
  python -m exp7.run_exp7 --benchmark agentdojo --model deepseek-v4-flash --harness baseline
  python -m exp7.run_exp7 --benchmark bfcl --model glm-5.1 --harness max_defense
  python -m exp7.run_exp7 --matrix --limit 1
  python -m exp7.run_exp7 --aggregate
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

TOOLS_ATTACK_ROOT = Path(__file__).resolve().parents[1]
if str(TOOLS_ATTACK_ROOT) not in sys.path:
    sys.path.insert(0, str(TOOLS_ATTACK_ROOT))

from exp7.aggregate import aggregate_results  # noqa: E402
from exp7.config import (  # noqa: E402
    EXP7_BENCHMARKS,
    EXP7_MODELS,
    HARNESS_MODES,
)
from exp7.harness import export_addendum_json  # noqa: E402
from exp7.runners.registry import get_runner  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Experiment 7 utility control study")
    parser.add_argument("--benchmark", choices=EXP7_BENCHMARKS)
    parser.add_argument("--model", choices=EXP7_MODELS)
    parser.add_argument("--harness", choices=HARNESS_MODES)
    parser.add_argument("--matrix", action="store_true", help="Run full 5×3×2 grid")
    parser.add_argument("--aggregate", action="store_true", help="Build summary table only")
    parser.add_argument("--export-addendum", action="store_true", help="Write unified system addendum JSON")
    parser.add_argument(
        "--harness-file",
        default="",
        help="Optional JSON harness strategy used in place of static max_defense",
    )
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--limit", type=int, default=0, help="Per-benchmark case/task limit (smoke test)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.harness_file:
        os.environ["TOOLS_ATTACK_HARNESS_STRATEGY_FILE"] = str(
            Path(args.harness_file).expanduser().resolve()
        )

    if args.export_addendum:
        path = export_addendum_json()
        print(f"Wrote {path}")
        return

    if args.aggregate:
        summary = aggregate_results()
        print(summary["markdown_table"])
        print(f"Saved: exp7/results/exp7_summary.json")
        return

    if args.matrix:
        results = []
        for benchmark in EXP7_BENCHMARKS:
            for model in EXP7_MODELS:
                for harness in HARNESS_MODES:
                    print(f"\n=== exp7 {benchmark} | {model} | {harness} ===")
                    try:
                        payload = get_runner(benchmark)(
                            model=model,
                            harness=harness,
                            force=args.force,
                            limit=args.limit,
                        )
                    except Exception as exc:
                        payload = {
                            "benchmark": benchmark,
                            "model": model,
                            "harness": harness,
                            "utility": None,
                            "status": "error",
                            "error": str(exc),
                        }
                    results.append(payload)
                    print(json.dumps(payload, ensure_ascii=False, indent=2))
        aggregate_results()
        return

    if not (args.benchmark and args.model and args.harness):
        raise SystemExit("Provide --benchmark, --model, --harness or use --matrix / --aggregate")

    payload = get_runner(args.benchmark)(
        model=args.model,
        harness=args.harness,
        force=args.force,
        limit=args.limit,
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
