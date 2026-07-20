#!/usr/bin/env python3
"""Run the paper Table 1 attack comparison (no added defense, 211 test cases)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from paper_table_common import (  # noqa: E402
    INJECAGENT_ROOT,
    TABLE1_ATTACKS,
    TABLE1_MODELS,
    attack_blockers,
    cross_eval_command,
    run_command,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", default=",".join(TABLE1_MODELS))
    parser.add_argument("--attacks", default=",".join(TABLE1_ATTACKS))
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--allow-missing",
        action="store_true",
        help="Still launch ChatInject or Adaptive cells when their inputs are absent.",
    )
    parser.add_argument("--no-cache", action="store_true")
    args = parser.parse_args()

    models = [item.strip() for item in args.models.split(",") if item.strip()]
    attacks = [item.strip() for item in args.attacks.split(",") if item.strip()]
    skipped: list[str] = []
    exit_code = 0
    for model in models:
        blocked = attack_blockers(model)
        for attack in attacks:
            reason = blocked.get(attack)
            if reason and not args.allow_missing:
                message = f"skip {model} none {attack}: {reason}"
                print(message)
                skipped.append(message)
                continue
            code = run_command(
                cross_eval_command(
                    model=model,
                    defense="none",
                    attack=attack,
                    limit=args.limit,
                    use_cache=not args.no_cache,
                ),
                cwd=INJECAGENT_ROOT,
                dry_run=args.dry_run,
            )
            if code != 0:
                exit_code = code
                break
        if exit_code != 0:
            break
    if skipped and exit_code == 0 and not args.dry_run:
        return 2
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
