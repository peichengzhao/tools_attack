#!/usr/bin/env python3
"""
Consolidate Enhanced / Adaptive offline final_poison into data/cross_eval/offline_poison/.

Does not call any LLM API — only reads existing results/ optimizer outputs.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.cross_eval.attack_payloads import (  # noqa: E402
    OFFLINE_POISON_DIR,
    _load_json_records,
    _raw_offline_result_paths,
    extract_final_poison,
    offline_poison_coverage,
)
from src.cross_eval.constants import CROSS_EVAL_MODELS, OFFLINE_POISON_ATTACKS
from src.evaluate_attack_agent import make_case_id


def export_offline_cache(
    model: str,
    attack: str,
    setting: str = "base",
) -> dict:
    poison_map: dict[str, str] = {}
    sources: list[str] = []
    for path in _raw_offline_result_paths(attack, model, setting):
        if not path.exists():
            continue
        sources.append(str(path))
        for record in _load_json_records(path):
            cid = make_case_id(record)
            poison = extract_final_poison(record)
            if poison:
                poison_map[cid] = poison

    out_dir = OFFLINE_POISON_DIR / model
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{attack}_{setting}.jsonl"
    if poison_map:
        with open(out_path, "w", encoding="utf-8") as f:
            for cid, poison in sorted(poison_map.items()):
                f.write(
                    json.dumps({"case_id": cid, "final_poison": poison}, ensure_ascii=False)
                    + "\n"
                )
    elif out_path.exists():
        out_path.unlink()

    cov = offline_poison_coverage(attack, model, setting)
    return {
        "model": model,
        "attack": attack,
        "exported": len(poison_map),
        "output": str(out_path) if poison_map else None,
        "sources": sources,
        **cov,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Build offline poison cache for cross-eval")
    parser.add_argument("--model", default="all", help="Victim model or 'all'")
    parser.add_argument("--attack", default="all", help="injecagent_enhanced | adaptive_attack | all")
    parser.add_argument("--setting", default="base")
    args = parser.parse_args()

    models = CROSS_EVAL_MODELS if args.model == "all" else (args.model,)
    attacks = (
        tuple(OFFLINE_POISON_ATTACKS)
        if args.attack == "all"
        else (args.attack,)
    )

    summaries = []
    for model in models:
        for attack in attacks:
            if attack not in OFFLINE_POISON_ATTACKS:
                raise SystemExit(f"Unknown offline attack: {attack}")
            summaries.append(export_offline_cache(model, attack, args.setting))

    print(json.dumps(summaries, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
