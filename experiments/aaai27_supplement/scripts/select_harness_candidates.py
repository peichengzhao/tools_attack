#!/usr/bin/env python3
"""Extract a small validation-evaluation candidate pool from training harnesses."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path.cwd()))

from src.prompts.protect_prompt import empty_defense_strategy  # noqa: E402
from src.prompts.max_defense_harness import MAX_DEFENSE_STRATEGY  # noqa: E402


def strategy_key(strategy: dict) -> str:
    return json.dumps(strategy, ensure_ascii=False, sort_keys=True)


def round_state(record: dict, round_limit: int) -> tuple[dict, str, str, bool]:
    history = record.get("coevolution_history") or []
    if round_limit > 0:
        eligible = [
            item for item in history if int(item.get("round", 0)) <= round_limit
        ]
        if not eligible:
            return {}, "", "", False
        last = eligible[-1]
        strategy = (
            last.get("defense_strategy")
            or record.get("final_defense_strategy")
            or empty_defense_strategy()
        )
        eval_value = last.get("eval") or record.get("eval") or ""
        utility = last.get("utility_eval") or record.get("utility_final") or ""
        attacked = any(
            item.get("eval") in ("succ", "invalid") for item in eligible
        )
        return strategy, eval_value, utility, attacked
    strategy = record.get("final_defense_strategy") or empty_defense_strategy()
    return (
        strategy,
        record.get("eval") or "",
        record.get("utility_final") or "",
        bool(record.get("attacked_any_round")),
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--train-results",
        default="results/aaai27_supplement/artifacts/full_train632_deepseek-v4.1-flash-1017.jsonl",
    )
    parser.add_argument(
        "--output",
        default="results/aaai27_supplement/artifacts/harness_candidates.json",
    )
    parser.add_argument("--top", type=int, default=8)
    parser.add_argument("--round-limit", type=int, default=0)
    args = parser.parse_args()

    records = [
        json.loads(line)
        for line in Path(args.train_results).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    buckets: dict[str, list[dict]] = {}
    strategies: dict[str, dict] = {}
    states: dict[int, tuple[str, str, bool]] = {}
    for record in records:
        strategy, eval_value, utility, attacked = round_state(
            record, args.round_limit
        )
        if not strategy:
            continue
        key = strategy_key(strategy)
        strategies[key] = strategy
        buckets.setdefault(key, []).append(record)
        states[id(record)] = (
            eval_value,
            utility,
            attacked,
        )

    stats = []
    for key, items in buckets.items():
        total = len(items)
        item_states = [states[id(item)] for item in items if id(item) in states]
        attack_hits = sum(state[2] for state in item_states)
        invalid = sum(state[0] == "invalid" for state in item_states)
        utility = sum(state[1] == "succ" for state in item_states)
        clean_success = sum(
            state[0] == "unsucc" and state[1] == "succ" for state in item_states
        )
        stats.append(
            {
                "key": key,
                "strategy": strategies[key],
                "cases": total,
                "attack_hits": attack_hits,
                "attack_hit_rate": round(attack_hits / total, 4),
                "invalid": invalid,
                "invalid_rate": round(invalid / total, 4),
                "utility": utility,
                "utility_rate": round(utility / total, 4),
                "clean_success": clean_success,
                "clean_success_rate": round(clean_success / total, 4),
            }
        )

    selected: dict[str, dict] = {}

    def add_strategy(strategy: dict, source: str, stats_payload: dict | None = None):
        key = strategy_key(strategy)
        if key in selected:
            return
        selected[key] = {
            "key": key,
            "strategy": strategy,
            "source": source,
            "cases": (stats_payload or {}).get("cases", 0),
            "attack_hits": (stats_payload or {}).get("attack_hits", 0),
            "attack_hit_rate": (stats_payload or {}).get("attack_hit_rate", 0.0),
            "invalid": (stats_payload or {}).get("invalid", 0),
            "invalid_rate": (stats_payload or {}).get("invalid_rate", 0.0),
            "utility": (stats_payload or {}).get("utility", 0),
            "utility_rate": (stats_payload or {}).get("utility_rate", 0.0),
            "clean_success": (stats_payload or {}).get("clean_success", 0),
            "clean_success_rate": (stats_payload or {}).get(
                "clean_success_rate", 0.0
            ),
        }

    empty = empty_defense_strategy()
    empty_stats = next((item for item in stats if item["key"] == strategy_key(empty)), None)
    add_strategy(empty, "empty-baseline", empty_stats)
    add_strategy(MAX_DEFENSE_STRATEGY, "static-max-defense")

    diverse_pool = [
        item
        for item in stats
        if item["key"] != strategy_key(empty)
        and item["clean_success"] > 0
    ]
    diverse_pool.sort(
        key=lambda item: (
            len(json.dumps(item["strategy"], ensure_ascii=False)),
            item["key"],
        )
    )
    needed = max(0, args.top - len(selected))
    if diverse_pool and needed:
        if len(diverse_pool) <= needed:
            picks = diverse_pool
        else:
            picks = []
            for index in range(needed):
                position = round(index * (len(diverse_pool) - 1) / max(1, needed - 1))
                picks.append(diverse_pool[position])
        for item in picks:
            add_strategy(
                item["strategy"],
                f"train-clean:{item.get('key', '')[:12]}",
                item,
            )

    candidates = list(selected.values())[: args.top]
    for index, item in enumerate(candidates, 1):
        item["candidate_id"] = f"H{index:02d}"

    output = {
        "version": "AAAI27-harness-candidates-v1",
        "source_training_results": args.train_results,
        "unique_training_harnesses": len(stats),
        "round_limit": args.round_limit,
        "candidates": candidates,
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(output, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "output": str(output_path),
                "unique": len(stats),
                "selected": len(candidates),
                "candidates": [
                    {
                        "id": item["candidate_id"],
                        "cases": item["cases"],
                        "attack_hit_rate": item["attack_hit_rate"],
                        "utility_rate": item["utility_rate"],
                    }
                    for item in candidates
                ],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
